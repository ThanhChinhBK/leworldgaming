"""ViT encoder — pixels -> latent embedding.

Compact ViT mirroring the architecture used in ``external/le-wm/train.py``
(``vit_hf("tiny")`` from ``stable_pretraining``): patch embedding, learnable
cls + positional tokens, N transformer blocks with pre-LayerNorm, and a
final ``LayerNorm`` applied to the cls token. The trailing LN is what
gives the encoder a stable output scale and prevents the representation
collapse the placeholder CNN suffered from.

Defaults are scaled-down from ViT-tiny (depth=12) to depth=6 for fast
iteration on MPS. Use ``encoder_depth: 12`` in ``configs/lewm.yaml`` for
the full thing.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class _PatchEmbed(nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_channels: int, embed_dim: int) -> None:
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(f"img_size {img_size} not divisible by patch_size {patch_size}")
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class _Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        drop = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop)
        out = out.transpose(1, 2).reshape(b, n, d)
        return self.proj(out)


class _Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _Attention(dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Encoder(nn.Module):
    """ViT encoder. Output is the LayerNorm'd cls token, optionally projected
    to ``latent_dim`` and re-normalized when ``latent_dim != embed_dim``.
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 192,
        image_size: int = 224,
        patch_size: int = 14,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.patch_embed = _PatchEmbed(image_size, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.blocks = nn.ModuleList(
            [_Block(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        if embed_dim != latent_dim:
            self.head: nn.Module = nn.Linear(embed_dim, latent_dim)
            self.out_norm: nn.Module = nn.LayerNorm(latent_dim)
        else:
            self.head = nn.Identity()
            self.out_norm = nn.Identity()

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.out_norm(self.head(x[:, 0]))
