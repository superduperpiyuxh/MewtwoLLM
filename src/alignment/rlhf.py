"""
Reinforcement Learning from Human Feedback (RLHF) for MewtwoLLM
Paper: "Training language models to follow instructions with human feedback"
(Ouyang et al., 2022)

Three-step pipeline:
1. Supervised Fine-Tuning (SFT) — already done
2. Reward Model Training — train a model to predict human preferences
3. PPO Optimization — optimize policy against reward model with KL penalty

PPO objective:
    max E[reward] - beta * KL(policy || reference)
"""

import os
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from config.model_config import MewtwoConfig
from src.model.pocketllm import MewtwoLLM
from src.model.rmsnorm import RMSNorm
from src.tokenizer.tokenizer import load_tokenizer


class RewardModel(nn.Module):
    """
    Reward model that scores responses.

    Based on the SFT model but with a scalar output head
    instead of a language modeling head.
    """

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        # Replace LM head with scalar reward head
        self.reward_head = nn.Linear(base_model.config.dim, 1, bias=False)

    def forward(self, x):
        """Return scalar reward for each sequence."""
        # Get last hidden state from transformer
        B, T = x.shape
        device = x.device

        # Token embedding
        h = self.base_model.token_embed(x)

        # Create causal mask
        mask = torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(0)

        # Forward through blocks
        for block in self.base_model.blocks:
            h, _ = block(h, mask=mask)

        h = self.base_model.norm(h)

        # Get reward from last token's hidden state
        reward = self.reward_head(h[:, -1, :]).squeeze(-1)
        return reward


class RewardDataset(Dataset):
    """
    Dataset for reward model training.

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

        print(f"Loaded {len(self.data)} preference pairs for reward model")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        chosen_tokens = self._tokenize(item["prompt"], item["chosen"])
        rejected_tokens = self._tokenize(item["prompt"], item["rejected"])
        return {
            "chosen": chosen_tokens,
            "rejected": rejected_tokens,
        }

    def _tokenize(self, prompt: str, response: str) -> list[int]:
        full_text = f"{prompt}\n\n{response}"
        tokens = self.tokenizer.encode(full_text)
        if len(tokens) > self.max_length:
            tokens = tokens[: self.max_length]
        return tokens


def _log_rm(epoch, epochs, batch_idx, total_loss, total_correct, total_samples, start_time):
    """Log reward model training progress."""
    avg_loss = total_loss / 10
    accuracy = total_correct / total_samples
    elapsed = time.time() - start_time
    print(
        f"Epoch {epoch+1}/{epochs} | Step {batch_idx+1} | "
        f"Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2%} | Time: {elapsed:.0f}s"
    )
    return 0.0, 0, 0, time.time()


def train_reward_model(
    config: MewtwoConfig,
    sft_model_path: str,
    data_path: str,
    tokenizer_path: str = "mewtwo_tokenizer.model",
    output_dir: str = "checkpoints/rm",
    epochs: int = 1,
):
    """Train the reward model on preference data."""

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("MewtwoLLM — Reward Model Training")
    print("=" * 60)

    tokenizer = load_tokenizer(tokenizer_path)

    # Load SFT model
    # SECURITY: weights_only=False needed for config object; only load trusted checkpoints
    checkpoint = torch.load(sft_model_path, map_location=config.device, weights_only=False)
    sft_config = checkpoint["config"]
    base_model = MewtwoLLM(sft_config)
    base_model.load_state_dict(checkpoint["model_state_dict"])

    # Create reward model
    reward_model = RewardModel(base_model).to(config.device)
    print(f"Reward model: {sum(p.numel() for p in reward_model.parameters()):,} parameters")

    optimizer = AdamW(reward_model.parameters(), lr=1e-5, weight_decay=0.01)

    dataset = RewardDataset(data_path, tokenizer, max_length=config.context_length)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)

    # Training
    reward_model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    start_time = time.time()

    print(f"\nTraining reward model for {epochs} epochs...")
    print()

    for epoch in range(epochs):
        for batch_idx, batch in enumerate(dataloader):
            chosen = batch["chosen"].to(config.device)
            rejected = batch["rejected"].to(config.device)

            chosen_rewards = reward_model(chosen)
            rejected_rewards = reward_model(rejected)
            loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(reward_model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            total_correct += (chosen_rewards > rejected_rewards).sum().item()
            total_samples += chosen.size(0)

            if (batch_idx + 1) % 10 == 0:
                total_loss, total_correct, total_samples, start_time = _log_rm(
                    epoch, epochs, batch_idx, total_loss, total_correct,
                    total_samples, start_time
                )

    # Save reward model
    rm_path = os.path.join(output_dir, "mewtwo_rm.pt")
    torch.save({
        "model_state_dict": reward_model.state_dict(),
        "config": sft_config,
    }, rm_path)
    print(f"\nReward model saved: {rm_path}")

    return reward_model


class PPOTrainer:
    """
    PPO trainer for RLHF.

    Optimizes the policy (SFT model) against the reward model
    with a KL divergence penalty to prevent reward hacking.
    """

    def __init__(
        self,
        config: MewtwoConfig,
        policy: MewtwoLLM,
        reward_model: RewardModel,
        reference: MewtwoLLM,
        tokenizer,
    ):
        self.config = config
        self.policy = policy
        self.reward_model = reward_model
        self.reference = reference
        self.tokenizer = tokenizer

        self.optimizer = AdamW(policy.parameters(), lr=1e-5)

        # Freeze reward model
        for param in self.reward_model.parameters():
            param.requires_grad = False

    def compute_rewards(self, prompts: list[str], responses: list[str]):
        """Compute rewards with KL penalty."""
        rewards = []
        for prompt, response in zip(prompts, responses):
            tokens = self.tokenizer.encode(f"{prompt}\n\n{response}", out_type=int)
            tokens = torch.tensor([tokens[:self.config.context_length]], device=self.config.device)

            with torch.no_grad():
                reward = self.reward_model(tokens).item()
                # KL penalty (simplified)
                policy_logprob = self._compute_logprob(self.policy, tokens)
                ref_logprob = self._compute_logprob(self.reference, tokens)
                kl = policy_logprob - ref_logprob

            rewards.append(reward - self.config.ppo_kl_coeff * kl)
        return rewards

    def _compute_logprob(self, model, tokens):
        """Compute sequence log probability."""
        input_ids = tokens[:, :-1]
        targets = tokens[:, 1:]
        logits, _, _ = model(input_ids)
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = torch.gather(log_probs, 2, targets.unsqueeze(-1)).squeeze(-1)
        return token_log_probs.sum().item()

    def train_step(self, prompts: list[str], responses: list[str]):
        """Single PPO training step."""
        # Compute rewards
        rewards = self.compute_rewards(prompts, responses)

        # Simple policy gradient (simplified PPO)
        total_loss = 0
        for prompt, response, reward in zip(prompts, responses, rewards):
            tokens = self.tokenizer.encode(f"{prompt}\n\n{response}", out_type=int)
            tokens = torch.tensor([tokens[:self.config.context_length]], device=self.config.device)

            input_ids = tokens[:, :-1]
            targets = tokens[:, 1:]

            logits, _, _ = self.policy(input_ids)
            log_probs = F.log_softmax(logits, dim=-1)
            token_log_probs = torch.gather(log_probs, 2, targets.unsqueeze(-1)).squeeze(-1)

            # Policy gradient loss (maximize reward)
            loss = -reward * token_log_probs.sum()
            total_loss += loss

        total_loss /= len(prompts)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()

        return total_loss.item(), rewards


def train_rlhf(
    config: MewtwoConfig,
    sft_model_path: str,
    rm_path: str,
    data_path: str,
    tokenizer_path: str = "mewtwo_tokenizer.model",
    output_dir: str = "checkpoints/rlhf",
    epochs: int = 1,
):
    """Full RLHF training pipeline."""

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("MewtwoLLM — RLHF (PPO)")
    print("=" * 60)

    tokenizer = load_tokenizer(tokenizer_path)

    # Load models
    # SECURITY: weights_only=False needed for config object; only load trusted checkpoints
    checkpoint = torch.load(sft_model_path, map_location=config.device, weights_only=False)
    sft_config = checkpoint["config"]

    policy = MewtwoLLM(sft_config)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy = policy.to(config.device)

    reference = MewtwoLLM(sft_config)
    reference.load_state_dict(checkpoint["model_state_dict"])
    reference = reference.to(config.device)
    reference.eval()
    for param in reference.parameters():
        param.requires_grad = False

    # SECURITY: weights_only=False needed for config object; only load trusted checkpoints
    rm_checkpoint = torch.load(rm_path, map_location=config.device, weights_only=False)
    base_model = MewtwoLLM(sft_config)
    reward_model = RewardModel(base_model)
    reward_model.load_state_dict(rm_checkpoint["model_state_dict"])
    reward_model = reward_model.to(config.device)
    reward_model.eval()

    print("Loaded policy, reference, and reward models")

    # Load preference data for training
    with open(data_path, "r") as f:
        preference_data = json.load(f)

    # PPO training
    ppo = PPOTrainer(config, policy, reward_model, reference, tokenizer)

    print(f"\nPPO training for {epochs} epochs...")
    print()

    for epoch in range(epochs):
        for i in range(0, len(preference_data), config.batch_size):
            batch = preference_data[i : i + config.batch_size]
            prompts = [item["prompt"] for item in batch]
            responses = [item["chosen"] for item in batch]  # Use chosen responses

            loss, rewards = ppo.train_step(prompts, responses)

            avg_reward = sum(rewards) / len(rewards)
            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Loss: {loss:.4f} | "
                f"Avg Reward: {avg_reward:.4f}"
            )

    # Save RLHF model
    rlhf_path = os.path.join(output_dir, "mewtwo_rlhf.pt")
    torch.save({
        "model_state_dict": policy.state_dict(),
        "config": sft_config,
    }, rlhf_path)
    print(f"\nRLHF complete! Model saved: {rlhf_path}")

    return policy


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RLHF for MewtwoLLM")
    parser.add_argument("--model", required=True, help="Path to SFT model")
    parser.add_argument("--reward_model", required=True, help="Path to reward model")
    parser.add_argument("--data", required=True, help="Path to preference data (JSON)")
    parser.add_argument("--tokenizer", default="mewtwo_tokenizer.model", help="Tokenizer path")
    parser.add_argument("--output_dir", default="checkpoints/rlhf", help="Output directory")
    parser.add_argument("--mode", choices=["train_rm", "train_ppo"], default="train_rm",
                       help="Train reward model or PPO")
    args = parser.parse_args()

    config = MewtwoConfig()

    if args.mode == "train_rm":
        train_reward_model(config, args.model, args.data, args.tokenizer, args.output_dir)
    else:
        train_rlhf(config, args.model, args.reward_model, args.data, args.tokenizer, args.output_dir)
