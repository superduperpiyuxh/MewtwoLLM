"""
Supervised Fine-Tuning (SFT) for MewtwoLLM

Turns the pretrained base model into an instruction-following assistant.
Based on the InstructGPT paper (Ouyang et al., 2022).

Uses Alpaca-style instruction format:
    ### Instruction: {instruction}
    ### Input: {input}
    ### Response: {response}
"""

import os
import json
import time

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from config.model_config import MewtwoConfig
from src.model.pocketllm import MewtwoLLM
from src.tokenizer.tokenizer import load_tokenizer


# Instruction prompt template
INSTRUCTION_TEMPLATE = """Below is an instruction that describes a task. Write a response.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{response}"""

INSTRUCTION_NO_INPUT = """Below is an instruction that describes a task. Write a response.

### Instruction:
{instruction}

### Response:
{response}"""


class InstructionDataset(Dataset):
    """Dataset for instruction fine-tuning."""

    def __init__(self, data_path: str, tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(data_path, "r") as f:
            content = f.read().strip()

        # Support both JSON array and JSONL formats
        if content.startswith("["):
            self.data = json.loads(content)
        else:
            self.data = [json.loads(line) for line in content.split("\n") if line.strip()]

        print(f"Loaded {len(self.data)} instruction examples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Format prompt
        if item.get("input", ""):
            prompt = INSTRUCTION_TEMPLATE.format(
                instruction=item["instruction"],
                input=item["input"],
                response=item["response"],
            )
        else:
            prompt = INSTRUCTION_NO_INPUT.format(
                instruction=item["instruction"],
                response=item["response"],
            )

        # Tokenize
        tokens = self.tokenizer.encode(prompt, out_type=int)

        # Truncate
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]

        # Create input/target pairs
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)

        return x, y


def _log_sft(epoch, config, global_step, total_loss, start_time):
    """Log SFT training progress."""
    avg_loss = total_loss / 10
    elapsed = time.time() - start_time
    print(
        f"Epoch {epoch+1}/{config.sft_epochs} | "
        f"Step {global_step} | Loss: {avg_loss:.4f} | Time: {elapsed:.0f}s"
    )
    return 0.0, time.time()


def train_sft(
    config: MewtwoConfig,
    model_path: str,
    data_path: str,
    tokenizer_path: str = "mewtwo_tokenizer.model",
    output_dir: str = "checkpoints/sft",
):
    """Train SFT on instruction data."""

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("MewtwoLLM — Supervised Fine-Tuning")
    print("=" * 60)

    # Load tokenizer
    tokenizer = load_tokenizer(tokenizer_path)

    # Load pretrained model
    # SECURITY: weights_only=False needed for config object; only load trusted checkpoints
    checkpoint = torch.load(model_path, map_location=config.device, weights_only=False)
    sft_config = checkpoint["config"]
    model = MewtwoLLM(sft_config)

    # Strip torch.compile() prefix if present
    state_dict = checkpoint["model_state_dict"]
    cleaned = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned)
    model = model.to(config.device)
    print(f"Loaded pretrained model from {model_path}")

    # Optimizer (lower LR for fine-tuning)
    optimizer = AdamW(
        model.parameters(),
        lr=config.sft_lr,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )

    # Dataset
    dataset = InstructionDataset(data_path, tokenizer, max_length=config.context_length)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
    )

    # Training
    model.train()
    total_loss = 0.0
    start_time = time.time()

    print(f"\nFine-tuning for {config.sft_epochs} epochs...")
    print(f"LR: {config.sft_lr}")
    print()

    global_step = 0
    for epoch in range(config.sft_epochs):
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(config.device), y.to(config.device)
            logits, loss, _ = model(x, targets=y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            global_step += 1

            if global_step % 10 == 0:
                total_loss, start_time = _log_sft(
                    epoch, config, global_step, total_loss, start_time
                )

    # Save SFT model
    sft_path = os.path.join(output_dir, "mewtwo_sft.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": sft_config,
    }, sft_path)
    print(f"\nSFT complete! Model saved: {sft_path}")

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SFT for MewtwoLLM")
    parser.add_argument("--model", required=True, help="Path to pretrained model")
    parser.add_argument("--data", required=True, help="Path to instruction data (JSON)")
    parser.add_argument("--tokenizer", default="mewtwo_tokenizer.model", help="Tokenizer path")
    parser.add_argument("--output_dir", default="checkpoints/sft", help="Output directory")
    args = parser.parse_args()

    config = MewtwoConfig()
    train_sft(config, args.model, args.data, args.tokenizer, args.output_dir)
