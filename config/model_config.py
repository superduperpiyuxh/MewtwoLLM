"""
MewtwoLLM Configuration
Every hyperparameter justified by research papers.
"""

from dataclasses import dataclass


@dataclass
class MewtwoConfig:
    """Model configuration based on modern LLM best practices."""

    # Model architecture
    vocab_size: int = 32000          # SentencePiece BPE (Paper: SentencePiece)
    context_length: int = 1024       # GPT-2 standard
    n_layers: int = 12               # Depth
    n_heads: int = 8                 # Attention heads
    n_kv_heads: int = 4              # GQA: 4 KV groups (Paper: GQA)
    dim: int = 512                   # Hidden dimension
    ff_dim: int = 1408               # SwiGLU FFN dim: (8/3) * 4 * dim, rounded to 64 (Paper: SwiGLU)
    dropout: float = 0.1             # Standard dropout (Paper: Transformer)
    bias: bool = False               # No bias in linear layers (modern best practice)

    # RoPE (Paper: RoFormer)
    rope_base: float = 10000.0       # Base frequency for rotation
    rope_dim: int = 64               # Rotary dimension (head_dim)

    # Training (Paper: Chinchilla)
    batch_size: int = 32             # Per-device batch size
    gradient_accumulation_steps: int = 8  # Effective batch = 256
    max_lr: float = 3e-4             # Peak learning rate
    min_lr: float = 3e-5             # Final learning rate (10% of peak)
    warmup_steps: int = 100          # 1% warmup
    stable_steps: int = 8500         # 85% stable phase
    decay_steps: int = 1500          # 15% decay phase
    total_steps: int = 10000         # Total training steps
    weight_decay: float = 0.1        # AdamW weight decay
    max_grad_norm: float = 1.0       # Gradient clipping
    betas: tuple = (0.9, 0.95)       # AdamW betas (modern LLM standard)

    # Alignment
    sft_epochs: int = 3              # SFT epochs
    sft_lr: float = 5e-5             # SFT learning rate
    dpo_beta: float = 0.1            # DPO temperature
    dpo_epochs: int = 1              # DPO epochs
    ppo_clip_ratio: float = 0.2      # PPO clip ratio
    ppo_kl_coeff: float = 0.02       # KL penalty coefficient

    # Inference
    temperature: float = 0.8         # Sampling temperature
    top_k: int = 40                  # Top-k sampling
    top_p: float = 0.9               # Nucleus sampling
    max_new_tokens: int = 256        # Max generation length

    # System
    device: str = "cpu"              # Training device
    dtype: str = "float32"           # Training dtype
    num_workers: int = 4             # Data loading workers
    log_interval: int = 100          # Logging frequency
    eval_interval: int = 1000        # Evaluation frequency
    save_interval: int = 2000        # Checkpoint frequency

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    @property
    def n_rep(self) -> int:
        """Number of times each KV head is repeated to match Q heads."""
        return self.n_heads // self.n_kv_heads

    def __post_init__(self):
        assert self.dim % self.n_heads == 0, "dim must be divisible by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        assert self.ff_dim % 64 == 0, "ff_dim should be divisible by 64 for GPU efficiency"
