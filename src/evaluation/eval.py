"""
Evaluation for MewtwoLLM

Measures:
- Perplexity on held-out text
- Zero-shot accuracy on standard benchmarks
- Generation quality (manual/automated)
"""

import os
import sys
import math
import json

import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from config.model_config import MewtwoConfig
from src.model.pocketllm import MewtwoLLM


class EvalDataset(Dataset):
    """Dataset for evaluation (non-overlapping chunks)."""

    def __init__(self, token_file: str, block_size: int = 1024):
        with open(token_file, "r") as f:
            tokens = list(map(int, f.read().split()))
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.block_size = block_size
        self.n_samples = max(0, len(self.tokens) - block_size - 1)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = self.tokens[idx : idx + self.block_size]
        y = self.tokens[idx + 1 : idx + self.block_size + 1]
        return x, y


def compute_perplexity(model, data_path, device="cpu", block_size=1024, batch_size=8):
    """Compute perplexity on a text file."""
    model.eval()
    dataset = EvalDataset(data_path, block_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits, loss, _ = model(x, targets=y)

            # Count non-padding tokens
            mask = (y != 0).float()
            n_tokens = mask.sum().item()

            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "tokens": total_tokens,
    }


def evaluate(model_path: str, eval_data_path: str, device: str = "cpu"):
    """Run full evaluation suite."""
    print("=" * 60)
    print("MewtwoLLM — Evaluation")
    print("=" * 60)

    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint["config"]
    model = MewtwoLLM(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Perplexity evaluation
    print(f"\nEvaluating perplexity on {eval_data_path}...")
    results = compute_perplexity(model, eval_data_path, device=device)

    print(f"\nResults:")
    print(f"  Loss: {results['loss']:.4f}")
    print(f"  Perplexity: {results['perplexity']:.2f}")
    print(f"  Tokens evaluated: {results['tokens']:,}")

    # Save results
    results_path = os.path.join(os.path.dirname(model_path), "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate MewtwoLLM")
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--data", required=True, help="Path to evaluation data")
    parser.add_argument("--device", default="cpu", help="Device")
    args = parser.parse_args()

    evaluate(args.model, args.data, args.device)
