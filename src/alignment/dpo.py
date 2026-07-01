"""
Direct Preference Optimization (DPO) for MewtwoLLM
Paper: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
(Rafailov et al., 2023)

Key insight:
Instead of training a reward model + RL (PPO), directly optimize the policy
using a simple binary cross-entropy loss on preference pairs.

The policy implicitly defines a reward:
    r(x, y) = beta * log(pi(y|x) / pi_ref(y|x))

DPO loss:
    L = -log(sigma(beta * [log(pi(y_w|x)/pi_ref(y_w|x)) - log(pi(y_l|x)/pi_ref(y_l|x))]))

where y_w = preferred response, y_l = rejected response.
"""

import os
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from config.model_config import MewtwoConfig
from src.model.pocketllm import MewtwoLLM
from src.tokenizer.tokenizer import load_tokenizer


class PreferenceDataset(Dataset):
    """
    Dataset of preference pairs (chosen, rejected) responses.

    Supports both JSON array and JSONL formats.
    """

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

        print(f"Loaded {len(self.data)} preference pairs")

    def __len__(self):
        return len(self.data)

    def _tokenize(self, prompt: str, response: str) -> list[int]:
        """Tokenize prompt + response."""
        full_text = f"{prompt}\n\n{response}"
        tokens = self.tokenizer.encode(full_text, out_type=int)
        if len(tokens) > self.max_length:
            tokens = tokens[: self.max_length]
        return tokens

    def __getitem__(self, idx):
        item = self.data[idx]

        chosen_tokens = self._tokenize(item["prompt"], item["chosen"])
        rejected_tokens = self._tokenize(item["prompt"], item["rejected"])

        return {
            "chosen": torch.tensor(chosen_tokens, dtype=torch.long),
            "rejected": torch.tensor(rejected_tokens, dtype=torch.long),
            "chosen_len": len(chosen_tokens),
            "rejected_len": len(rejected_tokens),
        }


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """
    Compute DPO loss.

    L = -E[sigma(beta * (log(pi(y_w)/pi_ref(y_w)) - log(pi(y_l)/pi_ref(y_l))))]

    The loss increases the likelihood of chosen responses and decreases
    rejected responses, weighted by how incorrectly the implicit reward
    ranks the pair.
    """
    chosen_logratios = policy_chosen_logps - reference_chosen_logps
    rejected_logratios = policy_rejected_logps - reference_rejected_logps
    logits = beta * (chosen_logratios - rejected_logratios)
    return -F.logsigmoid(logits).mean()


def compute_log_probs(model, tokens, device):
    """Compute log probabilities of tokens under the model."""
    tokens = tokens.to(device)
    input_ids = tokens[:, :-1]
    targets = tokens[:, 1:]

    with torch.no_grad():
        logits, _, _ = model(input_ids)

    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = torch.gather(log_probs, 2, targets.unsqueeze(-1)).squeeze(-1)

    # Sum log probabilities (excluding padding)
    mask = (targets != 0).float()
    return (token_log_probs * mask).sum(dim=1)


def _log_dpo(epoch, config, global_step, total_loss, start_time,
             policy_chosen_logps, policy_rejected_logps,
             reference_chosen_logps, reference_rejected_logps):
    """Log DPO training progress with accuracy."""
    avg_loss = total_loss / 10
    with torch.no_grad():
        chosen_rewards = config.dpo_beta * (policy_chosen_logps - reference_chosen_logps)
        rejected_rewards = config.dpo_beta * (policy_rejected_logps - reference_rejected_logps)
        accuracy = (chosen_rewards > rejected_rewards).float().mean().item()

    elapsed = time.time() - start_time
    print(
        f"Epoch {epoch+1}/{config.dpo_epochs} | "
        f"Step {global_step} | Loss: {avg_loss:.4f} | "
        f"Accuracy: {accuracy:.2%} | Time: {elapsed:.0f}s"
    )
    return 0.0, time.time()


def train_dpo(
    config: MewtwoConfig,
    sft_model_path: str,
    data_path: str,
    tokenizer_path: str = "mewtwo_tokenizer.model",
    output_dir: str = "checkpoints/dpo",
):
    """Train DPO alignment."""

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("MewtwoLLM — Direct Preference Optimization")
    print("=" * 60)

    # Load tokenizer
    tokenizer = load_tokenizer(tokenizer_path)

    # Load SFT model (policy)
    # SECURITY: weights_only=False needed for config object; only load trusted checkpoints
    checkpoint = torch.load(sft_model_path, map_location=config.device, weights_only=False)
    sft_config = checkpoint["config"]
    policy = MewtwoLLM(sft_config)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy = policy.to(config.device)

    # Reference model (frozen copy of SFT model)
    reference = MewtwoLLM(sft_config)
    reference.load_state_dict(checkpoint["model_state_dict"])
    reference = reference.to(config.device)
    reference.eval()
    for param in reference.parameters():
        param.requires_grad = False

    print(f"Loaded SFT model from {sft_model_path}")

    # Optimizer
    optimizer = AdamW(
        policy.parameters(),
        lr=config.sft_lr,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )

    # Dataset
    dataset = PreferenceDataset(data_path, tokenizer, max_length=config.context_length)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
    )

    # Training
    policy.train()
    total_loss = 0.0
    start_time = time.time()

    print(f"\nDPO training for {config.dpo_epochs} epochs...")
    print(f"Beta: {config.dpo_beta}")
    print(f"LR: {config.sft_lr}")
    print()

    global_step = 0
    for epoch in range(config.dpo_epochs):
        for batch_idx, batch in enumerate(dataloader):
            policy_chosen_logps = compute_log_probs(policy, batch["chosen"], config.device)
            policy_rejected_logps = compute_log_probs(policy, batch["rejected"], config.device)
            reference_chosen_logps = compute_log_probs(reference, batch["chosen"], config.device)
            reference_rejected_logps = compute_log_probs(reference, batch["rejected"], config.device)

            loss = dpo_loss(
                policy_chosen_logps, policy_rejected_logps,
                reference_chosen_logps, reference_rejected_logps,
                beta=config.dpo_beta,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            global_step += 1

            if global_step % 10 == 0:
                total_loss, start_time = _log_dpo(
                    epoch, config, global_step, total_loss, start_time,
                    policy_chosen_logps, policy_rejected_logps,
                    reference_chosen_logps, reference_rejected_logps,
                )

    # Save DPO model
    dpo_path = os.path.join(output_dir, "mewtwo_dpo.pt")
    torch.save({
        "model_state_dict": policy.state_dict(),
        "config": sft_config,
    }, dpo_path)
    print(f"\nDPO complete! Model saved: {dpo_path}")

    return policy


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DPO for MewtwoLLM")
    parser.add_argument("--model", required=True, help="Path to SFT model")
    parser.add_argument("--data", required=True, help="Path to preference data (JSON)")
    parser.add_argument("--tokenizer", default="mewtwo_tokenizer.model", help="Tokenizer path")
    parser.add_argument("--output_dir", default="checkpoints/dpo", help="Output directory")
    args = parser.parse_args()

    config = MewtwoConfig()
    train_dpo(config, args.model, args.data, args.tokenizer, args.output_dir)
