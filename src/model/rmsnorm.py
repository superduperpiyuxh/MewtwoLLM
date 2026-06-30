"""
RMSNorm — Root Mean Square Layer Normalization
Paper: "Root Mean Square Layer Normalization" (Zhang & Sennrich, 2019)

Why RMSNorm over LayerNorm:
- 15% faster (no mean subtraction)
- Same or better quality
- Used in: LLaMA, Mistral, Gemma, PaLM, Qwen
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    RMSNorm(x) = x / RMS(x) * gamma

    where RMS(x) = sqrt(mean(x^2) + eps)

    Key insight from the paper:
    Re-centering (subtracting mean) is dispensable.
    Only re-scaling (dividing by RMS) matters for LayerNorm's success.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        return (x / rms) * self.weight
