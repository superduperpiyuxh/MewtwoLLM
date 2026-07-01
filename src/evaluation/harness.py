"""
LM Eval Harness Integration for MewtwoLLM

Integrates with EleutherAI's lm-evaluation-harness for standardized benchmarks.
"""

import os
import sys
import json
from typing import Any

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from config.model_config import MewtwoConfig
from src.model.pocketllm import MewtwoLLM


class MewtwoLMEval:
    """
    Wrapper for lm-evaluation-harness.

    Usage:
        model = MewtwoLMEval.from_pretrained("path/to/checkpoint", tokenizer_path="path/to/tokenizer")
        results = model.evaluate(tasks=["mmlu", "hellaswag", "winogrande"])
    """

    def __init__(self, model: MewtwoLLM, config: MewtwoConfig, tokenizer=None):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.model.eval()

    @classmethod
    def from_pretrained(cls, path: str, device: str = "cpu", tokenizer_path: str = None):
        """Load model from checkpoint."""
        # SECURITY: weights_only=False needed for config object; only load trusted checkpoints
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config = checkpoint["config"]
        model = MewtwoLLM(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)

        # Load tokenizer
        tokenizer = None
        if tokenizer_path:
            from src.tokenizer.tokenizer import load_tokenizer
            tokenizer = load_tokenizer(tokenizer_path)

        return cls(model, config, tokenizer)

    def loglikelihood(self, context: str, continuation: str) -> float:
        """
        Compute log-likelihood of continuation given context.

        Args:
            context: The context string
            continuation: The continuation string

        Returns:
            Log-likelihood score
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Pass tokenizer_path to from_pretrained().")

        # Tokenize
        context_tokens = self.tokenizer.encode(context)
        continuation_tokens = self.tokenizer.encode(continuation)

        # Combine
        full_tokens = context_tokens + continuation_tokens
        input_ids = torch.tensor([full_tokens], dtype=torch.long, device=self.model.token_embed.weight.device)

        # Forward pass
        with torch.no_grad():
            logits, _, _ = self.model(input_ids)

        # Compute log-likelihood of continuation
        log_probs = torch.log_softmax(logits[:, len(context_tokens)-1:-1], dim=-1)

        # Get log probs for continuation tokens
        cont_logits = log_probs[0, :len(continuation_tokens)]
        cont_targets = torch.tensor(continuation_tokens, dtype=torch.long, device=log_probs.device)

        # Sum log probs
        log_likelihood = cont_logits.gather(1, cont_targets.unsqueeze(1)).squeeze(1).sum().item()

        return log_likelihood

    def loglikelihood_oneshot(self, context: str, continuation: str) -> float:
        """Log-likelihood with 1-shot context."""
        return self.loglikelihood(context, continuation)

    def generate_until(self, context: str, max_tokens: int = 100, **kwargs) -> str:
        """
        Generate text until stopping condition.

        Args:
            context: The context string
            max_tokens: Maximum tokens to generate

        Returns:
            Generated continuation string
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Pass tokenizer_path to from_pretrained().")

        tokens = self.tokenizer.encode(context)
        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.model.token_embed.weight.device)

        output = self.model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=kwargs.get("temperature", 0.0),
            top_k=kwargs.get("top_k", 1),
        )

        generated = self.tokenizer.decode(output[0].tolist())
        # Remove the context from the generated text
        continuation = generated[len(context):]

        return continuation

    def evaluate(self, tasks: list[str], num_fewshot: int = 0) -> dict[str, Any]:
        """
        Evaluate model on specified tasks.

        Args:
            tasks: List of task names (e.g., ["mmlu", "hellaswag"])
            num_fewshot: Number of few-shot examples

        Returns:
            Dictionary of results
        """
        results = {}
        evaluators = {
            "mmlu": self._eval_mmlu,
            "hellaswag": self._eval_hellaswag,
            "winogrande": self._eval_winogrande,
            "lambada": self._eval_lambada,
        }

        for task in tasks:
            results[task] = self._run_task(task, evaluators.get(task), num_fewshot)

        return results

    def _run_task(self, task: str, evaluator, num_fewshot: int) -> dict:
        """Run a single evaluation task."""
        print(f"Evaluating {task}...")

        if evaluator is None:
            print(f"Unknown task: {task}")
            return {"error": f"Unknown task: {task}"}

        try:
            return evaluator(num_fewshot)
        except Exception as e:
            print(f"Error on {task}: {e}")
            return {"error": str(e)}

    def _eval_mmlu(self, num_fewshot: int) -> dict:
        """Evaluate MMLU."""
        from src.evaluation.eval import evaluate_mmlu

        if self.tokenizer is None:
            return {"error": "Tokenizer not loaded. Pass tokenizer_path to from_pretrained()."}

        # Use default MMLU data directory
        data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "mmlu")

        if not os.path.exists(data_dir):
            return {"error": "MMLU data not found. Run download script first."}

        results = evaluate_mmlu(self.model, self.tokenizer, data_dir, num_fewshot=num_fewshot)
        return {"accuracy": results["overall_accuracy"], "details": results}

    def _eval_hellaswag(self, num_fewshot: int) -> dict:
        """Evaluate HellaSwag."""
        # Placeholder for HellaSwag evaluation
        return {"accuracy": 0.0, "note": "HellaSwag evaluation not implemented yet"}

    def _eval_winogrande(self, num_fewshot: int) -> dict:
        """Evaluate WinoGrande."""
        return {"accuracy": 0.0, "note": "WinoGrande evaluation not implemented yet"}

    def _eval_lambada(self, num_fewshot: int) -> dict:
        """Evaluate LAMBADA."""
        return {"accuracy": 0.0, "note": "LAMBADA evaluation not implemented yet"}


def run_harness_eval(
    checkpoint_path: str,
    tasks: list[str] | None = None,
    num_fewshot: int = 0,
    device: str = "cpu",
):
    """
    Run LM Eval Harness evaluation.

    Args:
        checkpoint_path: Path to model checkpoint
        tasks: List of tasks to evaluate
        num_fewshot: Number of few-shot examples
        device: Device to use

    Returns:
        Evaluation results
    """
    if tasks is None:
        tasks = ["mmlu", "hellaswag", "winogrande", "lambada"]

    print("=" * 60)
    print("LM Eval Harness — MewtwoLLM")
    print("=" * 60)

    model = MewtwoLMEval.from_pretrained(checkpoint_path, device)
    results = model.evaluate(tasks, num_fewshot)

    # Print summary
    print("\nResults:")
    for task, result in results.items():
        if "accuracy" in result:
            print(f"  {task}: {result['accuracy']:.1%}")
        else:
            print(f"  {task}: {result}")

    # Save results
    output_path = os.path.join(os.path.dirname(checkpoint_path), "harness_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run LM Eval Harness")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--tasks", nargs="+", default=["mmlu"], help="Tasks to evaluate")
    parser.add_argument("--num-fewshot", type=int, default=0, help="Number of few-shot examples")
    parser.add_argument("--device", default="cpu", help="Device")
    args = parser.parse_args()

    run_harness_eval(args.checkpoint, args.tasks, args.num_fewshot, args.device)
