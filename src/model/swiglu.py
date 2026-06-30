"""
SwiGLU Feed-Forward Network
Paper: "GLU Variants Improve Transformer" (Shazeer, 2020)

Key insight:
Replace ReLU/GELU with gated linear units.
SwiGLU learns which information to pass or block via a multiplicative gate.

Formula: SwiGLU(x) = W2(SiLU(W1(x)) * W3(x))

Used in: LLaMA, PaLM, Mistral, Qwen, Gemma — standard FFN for modern LLMs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """
    SwiGLU Feed-Forward Network.

    Three weight matrices (W1, W2, W3) instead of two.
    Hidden dimension is reduced to keep total params equal to standard FFN.

    Standard FFN: W1 (dim -> 4*dim), W2 (4*dim -> dim) = 2 * dim * 4*dim params
    SwiGLU: W1 (dim -> hidden), W2 (hidden -> dim), W3 (dim -> hidden)
            where hidden = (2/3) * 4 * dim, so params ≈ 3 * dim * hidden ≈ 2 * dim * 4*dim
    """

    def __init__(self, dim: int, hidden_dim: int, bias: bool = False):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: gate = SiLU(W1(x)), value = W3(x)
        # Output = W2(gate * value)
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
