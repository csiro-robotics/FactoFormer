import torch
import torch.nn as nn
import numpy as np


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):

    def __init__(self, dim, n_heads=4, qkv_bias=True, attn_p=0., proj_p=0.):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):

        n_samples, n_tokens, dim = x.shape

        if dim != self.dim:
            raise ValueError

        qkv = self.qkv(x)
        qkv = qkv.reshape(n_samples, n_tokens, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]
        k_t = k.transpose(-2, -1)
        dp = (q @ k_t) * self.scale
        attn = dp.softmax(dim=-1)
        attn = self.attn_drop(attn)

        weighted_avg = attn @ v
        weighted_avg = weighted_avg.transpose(1, 2)
        weighted_avg = weighted_avg.flatten(2)

        x = self.proj(weighted_avg)
        x = self.proj_drop(x)

        return x


class PatchEmbed3D(nn.Module):

    def __init__(self, img_size, patch_size, embed_dim, stride):
        super().__init__()
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) *(img_size[2] // patch_size[2])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv3d(1, embed_dim, kernel_size=patch_size, stride=stride)

    def forward(self, x):
        B, T, H, W = x.shape
        assert H == self.img_size[1] and W == self.img_size[2], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[1]}*{self.img_size[2]})."

        x = self.proj(x.reshape((B, 1, T, H, W))).flatten(2).transpose(1, 2)

        return x


class Transformer(nn.Module):
    def __init__(self, dim, depth, n_heads, mlp_ratio, qkv_bias, attn_p, proj_p):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, n_heads=n_heads, qkv_bias=qkv_bias, attn_p=attn_p, proj_p=proj_p)),
                PreNorm(dim, FeedForward(dim, mlp_ratio, dropout=proj_p))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class FactoFormer(nn.Module):

    def __init__(self, img_size, spatial_patch, spectral_patch, n_classes, spatial_embed_dim, spectral_embed_dim, bands, depth, n_heads,
                 qkv_bias, attn_p, proj_p):

        super().__init__()

        self.spatial_patch_embed = PatchEmbed3D(img_size=img_size, patch_size=spatial_patch, embed_dim=spatial_embed_dim, stride=spatial_patch)
        self.spectral_patch_embed = PatchEmbed3D(img_size=img_size, patch_size=spectral_patch, embed_dim=spectral_embed_dim, stride=spectral_patch)

        num_patches_spatial = self.spatial_patch_embed.num_patches
        num_patches_spectral = bands

        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches_spatial, spatial_embed_dim))
        self.spectral_pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches_spectral, spectral_embed_dim))

        self.spatial_cls_token = nn.Parameter(torch.zeros(1, 1, spatial_embed_dim))
        self.spectral_cls_token = nn.Parameter(torch.zeros(1, 1, spectral_embed_dim))

        self.spatial_transformer = Transformer(dim=spatial_embed_dim, depth=depth, n_heads=n_heads, mlp_ratio=8,
                                               qkv_bias=qkv_bias, attn_p=attn_p, proj_p=proj_p)

        self.spectral_transformer = Transformer(dim=spectral_embed_dim, depth=depth, n_heads=n_heads, mlp_ratio=4,
                                                qkv_bias=qkv_bias, attn_p=attn_p, proj_p=proj_p)
        dim = spectral_embed_dim + spatial_embed_dim
        self.lin = nn.Linear(dim, dim)
        self.pos_drop = nn.Dropout(p=proj_p)
        self.norm_spatial = nn.LayerNorm(spatial_embed_dim, eps=1e-6)
        self.norm_spectral = nn.LayerNorm(spectral_embed_dim, eps=1e-6)
        self.head = nn.Linear(dim, n_classes)

    def forward(self, x):

        n_samples = x.shape[0]
        x1 = self.spatial_patch_embed(x)
        x2 = self.spectral_patch_embed(x)

        cls_token_spatial = self.spatial_cls_token.expand(n_samples, -1, -1)
        cls_token_spectral = self.spectral_cls_token.expand(n_samples, -1, -1)

        x1 = torch.cat((cls_token_spatial, x1), dim=1)
        x2 = torch.cat((cls_token_spectral, x2), dim=1)

        x1 = x1 + self.spatial_pos_embed
        x2 = x2 + self.spectral_pos_embed

        x1 = self.pos_drop(x1)
        x2 = self.pos_drop(x2)

        x1 = self.spatial_transformer(x1)
        x1 = self.norm_spatial(x1)

        x2 = self.spectral_transformer(x2)
        x2 = self.norm_spectral(x2)

        cls_token_spatial = x1[:, 0]
        cls_token_spectral = x2[:, 0]

        out_x = torch.cat((cls_token_spatial, cls_token_spectral), 1)

        out_x = self.lin(out_x)
        out_x = self.head(out_x)

        return out_x