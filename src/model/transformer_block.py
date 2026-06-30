"""
Transformer Block for MewtwoLLM

Architecture (pre-normalization, like LLaMA):
    x -> RMSNorm -> GQA Attention -> + x (residual)
    x -> RMSNorm -> SwiGLU FFN    -> + x (residual)

This is the standard block used in GPT-3, LLaMA, Mistral, etc.
"""

import torch
import torch.nn as nn

from .rmsnorm import RMSNorm
from .attention import GQAAttention
from .swiglu import SwiGLU


class TransformerBlock(nn.Module):
    """
    A single transformer block with:
    1. Pre-normalization (RMSNorm before attention/FFN)
    2. Grouped Query Attention with RoPE
    3. SwiGLU Feed-Forward Network
    4. Residual connections around both
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
        bias: bool = False,
    ):
        super().__init__()

        # Attention sub-block
        self.attn_norm = RMSNorm(dim)
        self.attention = GQAAttention(dim, n_heads, n_kv_heads, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)

        # FFN sub-block
        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLU(dim, ff_dim, bias=bias)
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # Attention with residual
        residual = x
        x = self.attn_norm(x)
        attn_out, new_kv_cache = self.attention(x, mask=mask, kv_cache=kv_cache)
        x = residual + self.attn_dropout(attn_out)

        # FFN with residual
        residual = x
        x = self.ffn_norm(x)
        ffn_out = self.ffn(x)
        x = residual + self.ffn_dropout(ffn_out)

        return x, new_kv_cache
