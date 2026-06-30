"""
Training Loop for MewtwoLLM Pretraining

Implements:
- Next-token prediction (cross-entropy loss)
- WSD (Warmup-Stable-Decay) learning rate schedule
- AdamW optimizer with gradient clipping
- Mixed precision (when GPU available)
- Checkpointing
- Logging to TensorBoard
"""

import os
import math
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from config.model_config import MewtwoConfig
from src.model.pocketllm import MewtwoLLM


class TokenDataset(Dataset):
    """Simple dataset that loads tokenized text and creates sliding window chunks."""

    def __init__(self, token_file: str, block_size: int = 1024):
        self.block_size = block_size

        # Load tokens
        with open(token_file, "r") as f:
            tokens = list(map(int, f.read().split()))

        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.n_tokens = len(self.tokens)
        self.n_samples = max(0, self.n_tokens - block_size - 1)

        print(f"Dataset: {self.n_tokens:,} tokens, {self.n_samples:,} samples (block_size={block_size})")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = self.tokens[idx : idx + self.block_size]
        y = self.tokens[idx + 1 : idx + self.block_size + 1]
        return x, y


class WSDScheduler:
    """
    Warmup-Stable-Decay learning rate schedule.

    Phase 1 (Warmup): Linear increase from min_lr to max_lr over warmup_steps
    Phase 2 (Stable): Constant at max_lr for stable_steps
    Phase 3 (Decay): Cosine decay from max_lr to min_lr over decay_steps

    Based on MiniCPM finding that WSD outperforms cosine for LLM training.
    """

    def __init__(self, optimizer, warmup_steps, stable_steps, decay_steps, max_lr, min_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.stable_steps = stable_steps
        self.decay_steps = decay_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.current_step = 0

    def step(self):
        self.current_step += 1
        lr = self._get_lr(self.current_step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _get_lr(self, step):
        if step <= self.warmup_steps:
            # Linear warmup
            return self.min_lr + (self.max_lr - self.min_lr) * (step / self.warmup_steps)
        elif step <= self.warmup_steps + self.stable_steps:
            # Stable phase
            return self.max_lr
        else:
            # Cosine decay
            decay_progress = (step - self.warmup_steps - self.stable_steps) / self.decay_steps
            decay_progress = min(decay_progress, 1.0)
            return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * decay_progress))


def _log_step(step, config, total_loss, start_time, optimizer, checkpoint_dir):
    """Log training metrics and write to file."""
    avg_loss = total_loss / config.log_interval
    perplexity = math.exp(min(avg_loss, 20))
    elapsed = time.time() - start_time
    lr = optimizer.param_groups[0]["lr"]
    tokens_per_sec = (config.batch_size * config.gradient_accumulation_steps * config.context_length * config.log_interval) / elapsed

    print(
        f"Step {step:>6d}/{config.total_steps} | "
        f"Loss: {avg_loss:.4f} | PPL: {perplexity:.2f} | "
        f"LR: {lr:.2e} | Tok/s: {tokens_per_sec:.0f} | Time: {elapsed:.0f}s"
    )

    log_entry = {
        "step": step, "loss": avg_loss, "perplexity": perplexity,
        "lr": lr, "tokens_per_sec": tokens_per_sec, "elapsed": elapsed,
    }
    with open(os.path.join(checkpoint_dir, "training_log.jsonl"), "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return 0.0, time.time()


def _save_checkpoint(step, model, optimizer, config, checkpoint_dir):
    """Save model checkpoint."""
    ckpt_path = os.path.join(checkpoint_dir, f"mewtwo_step_{step}.pt")
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
    }, ckpt_path)
    print(f"  Checkpoint saved: {ckpt_path}")


def train(config: MewtwoConfig, data_path: str, checkpoint_dir: str = "checkpoints"):
    """Main training loop with optional mixed precision."""

    os.makedirs(checkpoint_dir, exist_ok=True)

    print("=" * 60)
    print("MewtwoLLM Pretraining")
    print("=" * 60)

    model = MewtwoLLM(config)
    model = model.to(config.device)

    # Mixed precision setup
    use_amp = config.device == "cuda" and hasattr(torch, 'amp')
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16 if use_amp else torch.float32
    print(f"Mixed precision: {'enabled (' + str(dtype) + ')' if use_amp else 'disabled'}")

    optimizer = AdamW(
        model.parameters(),
        lr=config.max_lr,
        betas=config.betas,
        weight_decay=config.weight_decay,
    )

    scheduler = WSDScheduler(
        optimizer=optimizer,
        warmup_steps=config.warmup_steps,
        stable_steps=config.stable_steps,
        decay_steps=config.decay_steps,
        max_lr=config.max_lr,
        min_lr=config.min_lr,
    )

    dataset = TokenDataset(data_path, block_size=config.context_length)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Training
    model.train()
    total_loss = 0.0
    start_time = time.time()
    step = 0
    epoch = 0

    print(f"\nTraining for {config.total_steps} steps...")
    print(f"Batch size: {config.batch_size} x {config.gradient_accumulation_steps} = {config.batch_size * config.gradient_accumulation_steps} effective")
    print(f"LR: {config.min_lr} -> {config.max_lr} -> {config.min_lr}")
    print(f"Device: {config.device}\n")

    while step < config.total_steps:
        epoch += 1
        for batch_idx, (x, y) in enumerate(dataloader):
            if step >= config.total_steps:
                break

            x, y = x.to(config.device), y.to(config.device)

            # Forward pass with optional mixed precision
            with torch.amp.autocast('cuda', enabled=use_amp, dtype=dtype):
                logits, loss, _ = model(x, targets=y)
                loss = loss / config.gradient_accumulation_steps

            # Backward pass with optional gradient scaling
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            total_loss += loss.item()

            if (batch_idx + 1) % config.gradient_accumulation_steps != 0:
                continue

            # Gradient clipping
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad()
            step += 1

            if step % config.log_interval == 0:
                total_loss, start_time = _log_step(
                    step, config, total_loss, start_time, optimizer, checkpoint_dir
                )

            if step % config.save_interval == 0:
                _save_checkpoint(step, model, optimizer, config, checkpoint_dir)

    # Save final model
    final_path = os.path.join(checkpoint_dir, "mewtwo_final.pt")
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
    }, final_path)
    print(f"\nTraining complete! Final model saved: {final_path}")

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train MewtwoLLM")
    parser.add_argument("--data", required=True, help="Path to tokenized data file")
    parser.add_argument("--checkpoint_dir", default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    config = MewtwoConfig()
    model = train(config, args.data, args.checkpoint_dir)
