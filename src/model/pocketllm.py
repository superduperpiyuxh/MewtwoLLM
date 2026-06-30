"""
MewtwoLLM — The Full Model

Assembled from:
- Token embeddings (with weight tying to output head)
- Position embeddings via RoPE (rotary, not additive)
- N transformer blocks: RMSNorm -> GQA -> +residual -> RMSNorm -> SwiGLU -> +residual
- Final RMSNorm
- Linear projection to vocabulary

Every component comes from a research paper.
No shortcuts. No high-level abstractions. Pure PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .rmsnorm import RMSNorm
from .rope import RotaryEmbedding, apply_rotary_pos_emb
from .attention import GQAAttention
from .swiglu import SwiGLU
from .transformer_block import TransformerBlock


class MewtwoLLM(nn.Module):
    """
    MewtwoLLM: An LLM built from scratch.

    Architecture:
        tokens -> token_embedding -> [TransformerBlock x N] -> RMSNorm -> lm_head -> logits

    Key design choices:
        - Pre-normalization (RMSNorm before attention/FFN)
        - RoPE (rotation-based position encoding)
        - GQA (grouped query attention)
        - SwiGLU (gated feed-forward)
        - Weight tying (embedding = output projection)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Token embedding (no positional embedding — RoPE handles position)
        self.token_embed = nn.Embedding(config.vocab_size, config.dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=config.dim,
                n_heads=config.n_heads,
                n_kv_heads=config.n_kv_heads,
                ff_dim=config.ff_dim,
                dropout=config.dropout,
                bias=config.bias,
            )
            for _ in range(config.n_layers)
        ])

        # Final normalization
        self.norm = RMSNorm(config.dim)

        # Output projection (tied to token embedding weights)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight  # Weight tying

        # Initialize weights
        self.apply(self._init_weights)

        # Report parameter count
        n_params = sum(p.numel() for p in self.parameters())
        # Subtract token embedding (tied, counted once)
        n_params -= self.token_embed.weight.numel()
        print(f"MewtwoLLM initialized: {n_params:,} parameters (excluding tied embedding)")

    def _init_weights(self, module):
        """Initialize weights using scaled normal initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass.

        Args:
            idx: (B, T) token indices
            targets: (B, T) target indices for loss computation (optional)
            kv_caches: list of cached K,V pairs for each layer (optional, for generation)

        Returns:
            logits: (B, T, vocab_size)
            loss: scalar loss (if targets provided)
            new_kv_caches: updated KV caches
        """
        B, T = idx.shape
        device = idx.device

        # Token embedding (no positional embedding — RoPE is inside attention)
        x = self.token_embed(idx)

        # Create causal mask (lower triangular)
        # When using KV cache, we need to handle the mask differently
        if kv_caches is not None:
            # During generation with KV cache, we attend to all cached + current tokens
            # No causal masking needed since we only generate one token at a time
            total_len = T + kv_caches[0][0].size(2) if kv_caches[0] is not None else T
            mask = torch.ones(1, 1, T, total_len, device=device)
        else:
            # During training, use causal mask
            mask = torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(0)

        # Forward through transformer blocks
        new_kv_caches = []
        for i, block in enumerate(self.blocks):
            cache = kv_caches[i] if kv_caches is not None else None
            x, new_cache = block(x, mask=mask, kv_cache=cache)
            new_kv_caches.append(new_cache)

        # Final normalization
        x = self.norm(x)

        # Project to vocabulary
        logits = self.lm_head(x)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,  # Ignore padding
            )

        return logits, loss, new_kv_caches

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """
        Autoregressive text generation with KV cache.

        Args:
            idx: (B, T) initial token indices
            max_new_tokens: maximum number of tokens to generate
            temperature: sampling temperature
            top_k: top-k sampling
            top_p: nucleus sampling

        Returns:
            (B, T + max_new_tokens) generated token indices
        """
        kv_caches = None

        for _ in range(max_new_tokens):
            # If we exceed context length, trim the input
            idx_cond = idx if idx.size(1) <= self.config.context_length else idx[:, -self.config.context_length:]

            # Forward pass
            logits, _, kv_caches = self.forward(idx_cond, kv_caches=kv_caches)

            # Get logits for the last position
            logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k > 0:
                top_k_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < top_k_values[:, -1:]] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float("-inf")

            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            idx = torch.cat([idx, idx_next], dim=1)

        return idx
