import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from vit_models import VisionTransformer
from einops import rearrange


class VisionTransformerForMIM(VisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert self.num_classes == 0

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.dim))
        self._trunc_normal_(self.mask_token, std=.02)

    def _trunc_normal_(self, tensor, mean=0., std=1.):
        trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

    def forward(self, x, mask):

        x = self.patch_embed(x)
        assert mask is not None
        B, L, _ = x.shape
        mask_token = self.mask_token.expand(B, L, -1)
        w = mask.unsqueeze(-1).type_as(mask_token)
        x = x * (1 - w) + mask_token * w

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        x = self.transformer(x)
        x = self.norm(x)
        x = x[:, 1:]
        x = x.permute(0, 2, 1)

        return x


class HyperMIM(nn.Module):
    def __init__(self, encoder, band, input_size, pretrain_mode, encoder_stride=1):
        super().__init__()
        self.encoder = encoder
        self.encoder_stride = encoder_stride
        self.band = band
        self.input_size = input_size
        self.pretrain_mode = pretrain_mode

        if self.pretrain_mode == 'spatial':
            temp = self.band
        elif self.pretrain_mode == 'spectral':
            temp = self.input_size * self.input_size

        self.decoder = nn.Sequential(
            nn.Conv1d(
                in_channels=self.encoder.num_features,
                out_channels=self.encoder_stride * temp, kernel_size=1)
        )

        self.in_chans = self.encoder.in_chans
        self.patch_size = self.encoder.patch_size

    def forward(self, x, mask):
        z = self.encoder(x, mask)
        x_rec = self.decoder(z)

        if self.pretrain_mode == 'spatial':
            x_rec = x_rec.reshape(x.shape)
            mask = rearrange(mask, 'b (h w) -> b h w', h=self.input_size, w=self.input_size)
            mask = mask.unsqueeze(1).repeat_interleave(self.band, 1)
        elif self.pretrain_mode == 'spectral':
            x_rec = rearrange(x_rec, 'b c n -> b n c')
            mask = mask.unsqueeze(2).repeat_interleave(self.input_size * self.input_size, 2)
            x = rearrange(x, 'b c h w -> b c (h w)')

        criterion = nn.MSELoss().cuda()
        loss_recon = criterion(x, x_rec)

        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans
        return loss
