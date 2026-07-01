"""
Text Generation for MewtwoLLM

Supports:
- Greedy decoding
- Top-k sampling
- Top-p (nucleus) sampling
- Temperature scaling
- KV cache for efficient autoregressive generation
"""

import os
import sys
import torch
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from config.model_config import MewtwoConfig
from src.model.pocketllm import MewtwoLLM
from src.tokenizer.tokenizer import load_tokenizer


def generate(
    model: MewtwoLLM,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9,
    device: str = "cpu",
) -> str:
    """
    Generate text given a prompt.

    Args:
        model: Trained MewtwoLLM model
        tokenizer: SentencePiece tokenizer
        prompt: Input text
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Top-k filtering
        top_p: Nucleus sampling threshold
        device: Device to run on

    Returns:
        Generated text (prompt + new text)
    """
    model.eval()

    # Tokenize prompt
    tokens = tokenizer.encode(prompt, out_type=int)
    x = torch.tensor([tokens], dtype=torch.long, device=device)

    # Generate
    with torch.no_grad():
        output = model.generate(
            x,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

    # Decode
    generated_tokens = output[0].tolist()
    generated_text = tokenizer.decode(generated_tokens)

    return generated_text


def interactive_generation(
    model: MewtwoLLM,
    tokenizer,
    device: str = "cpu",
    **generation_kwargs,
):
    """Interactive text generation loop."""
    print("=" * 60)
    print("MewtwoLLM — Interactive Generation")
    print("Type 'quit' to exit, 'config' to change settings")
    print("=" * 60)
    print()

    current_kwargs = {
        "max_new_tokens": 256,
        "temperature": 0.8,
        "top_k": 40,
        "top_p": 0.9,
    }
    current_kwargs.update(generation_kwargs)

    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if prompt.lower() == "quit":
            print("Goodbye!")
            break

        if prompt.lower() == "config":
            current_kwargs = _update_config(current_kwargs)
            continue

        if not prompt:
            continue

        response = generate(model, tokenizer, prompt, device=device, **current_kwargs)
        print(f"\nMewtwo: {response}\n")


def _update_config(current_kwargs):
    """Update generation settings interactively."""
    print(f"\nCurrent settings: {current_kwargs}")
    try:
        key = input("Setting to change: ").strip()
        if key in current_kwargs:
            value = input(f"New value for {key} ({type(current_kwargs[key]).__name__}): ").strip()
            current_kwargs[key] = type(current_kwargs[key])(value)
            print(f"Updated: {key} = {current_kwargs[key]}")
        else:
            print(f"Unknown setting: {key}")
    except (ValueError, EOFError):
        print("Invalid value")
    return current_kwargs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text with MewtwoLLM")
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--tokenizer", default="mewtwo_tokenizer.model", help="Tokenizer path")
    parser.add_argument("--prompt", default=None, help="Prompt for single generation")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    # Load model
    # SECURITY: weights_only=False needed for config object; only load trusted checkpoints
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = MewtwoLLM(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(config.device)

    # Load tokenizer
    tokenizer = load_tokenizer(args.tokenizer)

    if args.interactive:
        interactive_generation(
            model, tokenizer,
            device=config.device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
    elif args.prompt:
        response = generate(
            model, tokenizer, args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=config.device,
        )
        print(response)
    else:
        print("Provide --prompt or --interactive")
