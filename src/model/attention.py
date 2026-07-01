"""
Grouped Query Attention (GQA)
Paper: "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"
(Ainslie et al., 2023)

Key insight:
MHA (all heads have unique KV) is slow at inference.
MQA (1 KV head shared across all Q heads) is fast but loses quality.
GQA (G KV groups shared across Q heads) is the sweet spot.

Used in: LLaMA 2/3, Mistral, Qwen, Gemma — de facto standard.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .rope import RotaryEmbedding, apply_rotary_pos_emb


class GQAAttention(nn.Module):
    """
    Grouped Query Attention with RoPE.

    - Q has n_heads heads
    - K, V have n_kv_heads heads
    - Each KV head is shared across (n_heads / n_kv_heads) Q heads
    - RoPE is applied to Q and K before attention computation
    """

    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, bias: bool = False):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.n_rep = n_heads // n_kv_heads  # Repetition factor

        # Projection matrices (no bias in modern LLMs)
        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=bias)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=bias)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=bias)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=bias)

        # RoPE
        self.rope = RotaryEmbedding(self.head_dim)

        self.scale = 1.0 / math.sqrt(self.head_dim)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """
        Repeat KV heads to match Q heads.
        If n_rep=2 and input shape is (B, n_kv, T, head_dim),
        output shape is (B, n_kv*n_rep, T, head_dim).
        """
        if self.n_rep == 1:
            return x
        bs, n_kv, slen, head_dim = x.shape
        return (
            x[:, :, None, :, :]
            .expand(bs, n_kv, self.n_rep, slen, head_dim)
            .reshape(bs, n_kv * self.n_rep, slen, head_dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: (B, T, dim) input tensor
            mask: (T, T) causal mask (optional)
            kv_cache: cached K, V from previous positions (optional)

        Returns:
            output: (B, T, dim)
            new_kv_cache: (K, V) for caching
        """
        B, T, _ = x.shape

        # Project to Q, K, V
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K
        cos, sin = self.rope(T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Handle KV cache
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
        new_kv_cache = (k, v)

        # Repeat KV heads to match Q heads
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        # Use PyTorch SDPA (automatically uses FlashAttention on GPU)
        if hasattr(F, 'scaled_dot_product_attention'):
            # SDPA handles scaling internally; pass causal mask as attn_mask
            # mask dimensions: (1, 1, T, total_len) or (1, 1, T, T)
            # After KV concatenation, k/v have shape (B, n_kv, total_len, head_dim)
            # So we need mask with last dim matching total_len
            if mask is not None:
                # Use mask as-is if dimensions match, otherwise slice
                if mask.dim() == 4:
                    attn_mask = mask[:, :, :T, :k.size(2)]
                else:
                    attn_mask = mask[:T, :T]
            else:
                attn_mask = None
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=False,
            )
        else:
            # Fallback to manual attention
            attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            if mask is not None:
                if mask.dim() == 4:
                    m = mask[:, :, :T, :k.size(2)]
                else:
                    m = mask[:T, :T]
                attn = attn.masked_fill(m == 0, float("-inf"))
            attn = F.softmax(attn, dim=-1)
            out = torch.matmul(attn, v)

        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        out = self.wo(out)

        return out, new_kv_cache
