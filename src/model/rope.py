"""
Rotary Position Embedding (RoPE)
Paper: "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021)

Key insight:
Instead of ADDING position info to embeddings, RoPE MULTIPLIES (rotates) them.
The inner product of two rotated vectors depends only on their RELATIVE position.

Used in: LLaMA, Mistral, Qwen, Gemma, Phi — virtually all modern LLMs.
"""

import torch
import torch.nn as nn
import math


class RotaryEmbedding(nn.Module):
    """
    Precomputes rotation frequencies for RoPE.

    For each pair of dimensions (2i, 2i+1), we compute:
        theta_i = 1 / (base^(2i / dim))

    The rotation matrix for position m is:
        R(m) = [[cos(m*theta), -sin(m*theta)],
                [sin(m*theta),  cos(m*theta)]]
    """

    def __init__(self, dim: int, base: float = 10000.0, max_seq_len: int = 2048):
        super().__init__()
        self.dim = dim
        self.base = base

        # Compute inverse frequencies: theta_i = 1 / (base^(2i/dim))
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos/sin for max_seq_len positions
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        """Precompute cos and sin for all positions up to seq_len."""
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int):
        """Return cos and sin caches for the given sequence length."""
        if seq_len > self.cos_cached.shape[0]:
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate half the hidden dims of the input.
    x = [x0, x1, x2, x3] -> [-x1, x0, -x3, x2]
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to queries and keys.

    The rotation is:
        q_rotated = q * cos + rotate_half(q) * sin
        k_rotated = k * cos + rotate_half(k) * sin

    After rotation, the attention score q^T k depends only on (m - n),
    where m and n are the positions of q and k respectively.
    """
    # Reshape cos/sin for broadcasting: (1, 1, seq_len, dim)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
