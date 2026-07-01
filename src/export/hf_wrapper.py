"""
HuggingFace Model Wrapper for MewtwoLLM

Wraps MewtwoLLM as a PreTrainedModel for use with:
- transformers (inference, fine-tuning)
- trl (DPO, PPO alignment)
- axolotl (unified fine-tuning)
"""

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from config.model_config import MewtwoConfig
from src.model.pocketllm import MewtwoLLM


class MewtwoForCausalLM(nn.Module):
    """
    HuggingFace-compatible wrapper for MewtwoLLM.

    Usage:
        model = MewtwoForCausalLM.from_pretrained("path/to/checkpoint")
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        logits = outputs.logits
    """

    def __init__(self, config: MewtwoConfig):
        super().__init__()
        self.config = config
        self.mewtwo = MewtwoLLM(config)

    @classmethod
    def from_pretrained(cls, path: str, device: str = "cpu"):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint["config"]
        model = cls(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        return model

    def save_pretrained(self, path: str):
        """Save model to checkpoint."""
        os.makedirs(path, exist_ok=True)
        torch.save({
            "config": self.config,
            "model_state_dict": self.state_dict(),
        }, os.path.join(path, "model.pt"))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ):
        """
        Forward pass compatible with HuggingFace API.

        Args:
            input_ids: (B, T) token indices
            attention_mask: (B, T) attention mask (optional)
            labels: (B, T) target indices for loss (optional)

        Returns:
            outputs with loss and logits
        """
        logits, loss, _ = self.mewtwo(input_ids, targets=labels)

        return {"logits": logits, "loss": loss}

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.9,
    ):
        """Generate text (compatible with HuggingFace generate)."""
        return self.mewtwo.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

    def get_input_embeddings(self):
        return self.mewtwo.token_embed

    def get_output_embeddings(self):
        return self.mewtwo.lm_head


class MewtwoForSequenceClassification(nn.Module):
    """
    Reward model wrapper for RLHF/DPO training.

    Usage:
        reward_model = MewtwoForSequenceClassification.from_pretrained("path/to/checkpoint")
        rewards = reward_model(input_ids)
    """

    def __init__(self, config: MewtwoConfig):
        super().__init__()
        self.config = config
        self.mewtwo = MewtwoLLM(config)
        self.reward_head = nn.Linear(config.dim, 1, bias=False)

    @classmethod
    def from_pretrained(cls, path: str, device: str = "cpu"):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint["config"]
        model = cls(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        return model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        """
        Forward pass for reward model.

        Args:
            input_ids: (B, T) token indices
            attention_mask: (B, T) attention mask (optional)

        Returns:
            rewards: (B,) scalar rewards
        """
        # Get hidden states from last token
        x = self.mewtwo.token_embed(input_ids)

        # Forward through blocks
        mask = torch.tril(torch.ones(input_ids.shape[1], input_ids.shape[1], device=input_ids.device))
        mask = mask.unsqueeze(0).unsqueeze(0)

        for block in self.mewtwo.blocks:
            x, _ = block(x, mask=mask)

        x = self.mewtwo.norm(x)

        # Take last token's hidden state
        last_hidden = x[:, -1, :]

        # Project to reward
        rewards = self.reward_head(last_hidden).squeeze(-1)

        return rewards


def push_to_hub(
    model,
    repo_id: str,
    token: str | None = None,
):
    """Push model to HuggingFace Hub."""
    try:
        from huggingface_hub import HfApi

        api = HfApi()

        # Create repo
        api.create_repo(repo_id, exist_ok=True, token=token)

        # Save model
        model.save_pretrained("tmp_model")

        # Upload
        api.upload_folder(
            folder_path="tmp_model",
            repo_id=repo_id,
            token=token,
        )

        # Cleanup
        import shutil
        shutil.rmtree("tmp_model")

        print(f"Model pushed to: https://huggingface.co/{repo_id}")
        return True

    except Exception as e:
        print(f"Failed to push to hub: {e}")
        return False
