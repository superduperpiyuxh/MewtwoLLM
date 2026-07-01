"""
Evaluation for MewtwoLLM

Measures:
- Perplexity on held-out text
- MMLU (Massive Multitask Language Understanding) zero-shot accuracy
- Generation quality (manual/automated)
"""

import os
import sys
import math
import json
import re

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

            # Use -1 as ignore index (matches model's cross_entropy ignore_index=-1)
            mask = (y != -1).float()
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


# MMLU subject names (57 subjects)
MMLU_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge",
    "college_biology", "college_chemistry", "college_computer_science", "college_mathematics",
    "college_medicine", "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics", "formal_logic",
    "global_facts", "high_school_biology", "high_school_chemistry",
    "high_school_computer_science", "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics", "high_school_physics",
    "high_school_psychology", "high_school_statistics", "high_school_us_history",
    "high_school_world_history", "human_aging", "human_sexuality", "international_law",
    "jurisprudence", "logical_fallacies", "machine_learning", "management", "marketing",
    "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios",
    "nutrition", "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology", "us_foreign_policy",
    "virology", "world_religions",
]

CHOICES = ["A", "B", "C", "D"]


def format_mmlu_prompt(question: str, choices: list[str]) -> str:
    """Format MMLU prompt as multiple choice."""
    prompt = f"{question}\n\n"
    for i, choice in enumerate(choices):
        prompt += f"{CHOICES[i]}. {choice}\n"
    prompt += "\nAnswer:"
    return prompt


def extract_answer(response: str) -> str:
    """Extract letter answer (A/B/C/D) from model response."""
    response = response.strip().upper()
    # Try to find A/B/C/D at the start
    if response and response[0] in CHOICES:
        return response[0]
    # Try to find "answer is X" pattern
    match = re.search(r'(?:answer\s+is|ANSWER\s+IS)\s*[:\s]*([A-D])', response)
    if match:
        return match.group(1)
    # Try to find any A/B/C/D
    match = re.search(r'\b([A-D])\b', response)
    if match:
        return match.group(1)
    return "A"  # Default fallback


def evaluate_mmlu_subject(
    model, tokenizer, subject: str, data_dir: str, device: str = "cpu",
    n_shots: int = 0, max_length: int = 256
) -> dict:
    """Evaluate MMLU on a single subject."""
    import csv

    # Load dev examples (for few-shot)
    dev_path = os.path.join(data_dir, f"{subject}_dev.csv")
    dev_examples = []
    if n_shots > 0 and os.path.exists(dev_path):
        with open(dev_path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 6:
                    dev_examples.append({
                        "question": row[0],
                        "choices": [row[1], row[2], row[3], row[4]],
                        "answer": row[5],
                    })
        dev_examples = dev_examples[:n_shots]

    # Load test examples
    test_path = os.path.join(data_dir, f"{subject}_test.csv")
    test_examples = []
    with open(test_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 6:
                test_examples.append({
                    "question": row[0],
                    "choices": [row[1], row[2], row[3], row[4]],
                    "answer": row[5],
                })

    correct = 0
    total = len(test_examples)

    for example in test_examples:
        # Build prompt with few-shot examples
        prompt = ""
        for dev_ex in dev_examples:
            prompt += format_mmlu_prompt(dev_ex["question"], dev_ex["choices"])
            prompt += f" {dev_ex['answer']}\n\n"
        prompt += format_mmlu_prompt(example["question"], example["choices"])

        # Tokenize
        tokens = tokenizer.encode(prompt)
        tokens = tokens[-max_length:]  # Truncate to max_length
        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

        # Generate response
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=1, temperature=0.0)

        # Extract answer
        response = tokenizer.decode(output[0].tolist())
        predicted = extract_answer(response)

        if predicted == example["answer"]:
            correct += 1

    accuracy = correct / total if total > 0 else 0.0

    return {
        "subject": subject,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "n_shots": n_shots,
    }


def evaluate_mmlu(
    model, tokenizer, data_dir: str, device: str = "cpu",
    n_shots: int = 0, subjects: list[str] | None = None,
) -> dict:
    """Evaluate MMLU across all subjects."""
    if subjects is None:
        subjects = MMLU_SUBJECTS

    print(f"\nEvaluating MMLU ({n_shots}-shot) on {len(subjects)} subjects...")
    results = []
    total_correct = 0
    total_count = 0

    for i, subject in enumerate(subjects):
        try:
            result = evaluate_mmlu_subject(
                model, tokenizer, subject, data_dir, device, n_shots
            )
            results.append(result)
            total_correct += result["correct"]
            total_count += result["total"]

            print(f"  [{i+1}/{len(subjects)}] {subject}: {result['accuracy']:.1%} "
                  f"({result['correct']}/{result['total']})")
        except Exception as e:
            print(f"  [{i+1}/{len(subjects)}] {subject}: ERROR - {e}")

    overall_accuracy = total_correct / total_count if total_count > 0 else 0.0

    return {
        "overall_accuracy": overall_accuracy,
        "total_correct": total_correct,
        "total_count": total_count,
        "n_shots": n_shots,
        "subjects": results,
    }


def evaluate(
    model_path: str,
    eval_data_path: str | None = None,
    mmlu_data_dir: str | None = None,
    device: str = "cpu",
    n_shots: int = 0,
):
    """Run full evaluation suite."""
    print("=" * 60)
    print("MewtwoLLM — Evaluation")
    print("=" * 60)

    # Load model
    # SECURITY: weights_only=False needed for config object; only load trusted checkpoints
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = MewtwoLLM(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {param_count:,} parameters")

    results = {"param_count": param_count}

    # Perplexity evaluation
    if eval_data_path:
        print(f"\nEvaluating perplexity on {eval_data_path}...")
        ppl_results = compute_perplexity(model, eval_data_path, device=device)
        results["perplexity"] = ppl_results
        print(f"  Loss: {ppl_results['loss']:.4f}")
        print(f"  Perplexity: {ppl_results['perplexity']:.2f}")
        print(f"  Tokens: {ppl_results['tokens']:,}")

    # MMLU evaluation
    if mmlu_data_dir:
        from src.tokenizer.tokenizer import load_tokenizer
        tokenizer = load_tokenizer()
        mmlu_results = evaluate_mmlu(
            model, tokenizer, mmlu_data_dir, device, n_shots
        )
        results["mmlu"] = mmlu_results
        print(f"\nMMLU ({n_shots}-shot): {mmlu_results['overall_accuracy']:.1%}")
        print(f"  Correct: {mmlu_results['total_correct']}/{mmlu_results['total_count']}")

        # Top 5 best/worst subjects
        subjects = sorted(mmlu_results["subjects"], key=lambda x: x["accuracy"], reverse=True)
        print(f"\n  Top 5: {', '.join(f'{s[\"subject\"]} ({s[\"accuracy\"]:.0%})' for s in subjects[:5])}")
        print(f"  Bottom 5: {', '.join(f'{s[\"subject\"]} ({s[\"accuracy\"]:.0%})' for s in subjects[-5:])}")

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
    parser.add_argument("--data", help="Path to evaluation data (perplexity)")
    parser.add_argument("--mmlu-data-dir", help="Path to MMLU data directory")
    parser.add_argument("--n-shots", type=int, default=0, help="Number of shots for MMLU")
    parser.add_argument("--device", default="cpu", help="Device")
    args = parser.parse_args()

    evaluate(args.model, args.data, args.mmlu_data_dir, args.device, args.n_shots)
