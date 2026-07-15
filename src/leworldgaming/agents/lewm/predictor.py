"""Autoregressive transformer predictor — ported from ``external/le-wm``.

Predicts next-frame embeddings from a history of past embeddings, conditioned
on per-step action embeddings via AdaLN-zero modulation. Causal self-attention
over the time dimension makes each output depend only on past tokens, which
turns one forward pass into ``T`` parallel next-step predictions during
training.

Architecture matches ``external/le-wm/module.py::ARPredictor`` defaults
(``configs/train/lewm.yaml``):
    depth=6, heads=16, dim_head=64 (inner_dim=1024), mlp_dim=2048

API:
    forward(x, c) where
        x: (B, T, latent_dim)   — history of context embeddings
        c: (B, T, latent_dim)   — per-step action embeddings (from
                                   ``ActionEncoder``)
    returns: (B, T, latent_dim) — predicted next-step embeddings
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


class _CausalAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 16,
        dim_head: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        inner_dim = num_heads * dim_head
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        x = self.norm(x)
        qkv = self.to_qkv(x).reshape(b, t, 3, self.num_heads, self.dim_head)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        drop = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=True)
        out = out.transpose(1, 2).reshape(b, t, self.num_heads * self.dim_head)
        return self.to_out(out)


class _FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning on per-step actions."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attn = _CausalAttention(dim, num_heads, dim_head, dropout)
        self.mlp = _FeedForward(dim, mlp_dim, dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        # Zero-init so the block starts as identity (DiT-style stable training).
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN(c).chunk(
            6, dim=-1
        )
        x = x + gate_msa * self.attn(_modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(_modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Predictor(nn.Module):
    """Autoregressive transformer predictor with AdaLN-zero action conditioning.

    Takes a history of context embeddings ``x`` and per-step action embeddings
    ``c``, returns next-step predictions of the same shape. With causal
    self-attention, position ``t`` in the output is the prediction of position
    ``t+1`` in the original observation sequence.
    """

    def __init__(
        self,
        latent_dim: int = 192,
        action_dim: int = 192,  # action *embedding* dim — matches latent_dim
        history_size: int = 3,
        depth: int = 6,
        num_heads: int = 16,
        dim_head: int = 64,
        mlp_dim: int = 2048,
        dropout: float = 0.1,
        emb_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if action_dim != latent_dim:
            raise ValueError(
                "ARPredictor expects action embeddings to match latent_dim — wrap them via "
                "ActionEncoder(emb_dim=latent_dim)."
            )
        self.latent_dim = latent_dim
        self.history_size = history_size
        self.pos_embedding = nn.Parameter(torch.randn(1, history_size, latent_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.blocks = nn.ModuleList(
            [_ConditionalBlock(latent_dim, num_heads, dim_head, mlp_dim, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        if x.shape != c.shape:
            raise ValueError(f"x {tuple(x.shape)} and c {tuple(c.shape)} must match")
        t = x.size(1)
        if t > self.pos_embedding.size(1):
            raise ValueError(f"sequence length {t} exceeds history_size {self.pos_embedding.size(1)}")
        x = x + self.pos_embedding[:, :t]
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, c)
        return self.norm(x)
