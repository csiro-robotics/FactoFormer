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
    """ Image to Patch Embedding
    """
    def __init__(self, img_size, patch_size, embed_dim):
        super().__init__()
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) *(img_size[2] // patch_size[2])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv3d(1, embed_dim, kernel_size=patch_size, stride=patch_size)

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


def gen_mask(patch_size, num_bands, mask_ratio):
    number_patches = patch_size**2
    num_mask = int(mask_ratio * number_patches)
    mask = np.hstack([
            np.zeros(num_mask),
            np.ones(number_patches - num_mask),
        ])
    np.random.shuffle(mask)
    mask = np.reshape(mask, (patch_size,patch_size))
    mask = np.repeat(mask[:,:, np.newaxis], num_bands, axis=2)

    return mask


# Encoder
class VisionTransformer(nn.Module):

    def __init__(self,
                 image_size,
                 bands,
                 num_classes,
                 dim,
                 depth,
                 heads,
                 mlp_dim,
                 pretran_mode,
                 ):
        super().__init__()

        if pretran_mode == 'spatial':
            self.patch_embed = PatchEmbed3D(img_size=[bands, image_size, image_size], patch_size=[bands, 1, 1], embed_dim=dim)
            self.in_chans = 1
            self.patch_size = 7
            num_patches = self.patch_embed.num_patches
            self.num_features = self.dim = dim
            self.num_classes = num_classes
            self.num_patches = num_patches
        elif pretran_mode == 'spectral':
            self.patch_embed = PatchEmbed3D(img_size=[bands, image_size, image_size], patch_size=[1, image_size, image_size], embed_dim=dim)
            num_patches = self.patch_embed.num_patches
            self.num_features = self.dim = dim
            self.in_chans = 1
            self.patch_size = 49
            self.num_classes = num_classes
            self.num_patches = bands

        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))

        self.transformer = Transformer(dim=dim, depth=depth, n_heads=heads, mlp_ratio=mlp_dim,
                                       qkv_bias=True, attn_p=0.1, proj_p=0.1)

        self.pos_drop = nn.Dropout(p=0.1)
        self.norm = nn.LayerNorm(dim,  eps=1e-6)
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(nn.LayerNorm(dim),
                                      nn.Linear(dim, num_classes)) if num_classes > 0 else nn.Identity()

    def forward(self, x):

        n_samples = x.shape[0]
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(n_samples, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = self.transformer(x)
        x = self.norm(x)
        cls_token_final = x[:, 0]
        x = self.mlp_head(cls_token_final)

        return x

