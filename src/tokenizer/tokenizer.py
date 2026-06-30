"""
SentencePiece BPE Tokenizer for MewtwoLLM

Trains a BPE tokenizer on collected text data.
32K vocabulary, byte fallback for any Unicode, digit splitting.
"""

import os
import sentencepiece as spm


def train_tokenizer(
    input_files: str,
    model_prefix: str = "mewtwo_tokenizer",
    vocab_size: int = 32000,
    character_coverage: float = 0.9995,
):
    """
    Train a SentencePiece BPE tokenizer.

    Args:
        input_files: comma-separated list of input text files
        model_prefix: output model prefix
        vocab_size: vocabulary size
        character_coverage: character coverage ratio
    """
    print(f"Training tokenizer with vocab_size={vocab_size}...")
    print(f"Input files: {input_files}")

    spm.SentencePieceTrainer.train(
        input=input_files,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=character_coverage,
        byte_fallback=True,       # Handle any Unicode via byte fallback
        split_digits=True,        # Split numbers into individual digits
        num_threads=os.cpu_count() or 4,
        # Special tokens
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        # Normalization
        normalization_rule_name="identity",  # No NFKC — preserve raw text
    )

    print(f"Tokenizer trained: {model_prefix}.model")
    return f"{model_prefix}.model"


def load_tokenizer(model_path: str = "mewtwo_tokenizer.model"):
    """Load a trained SentencePiece tokenizer."""
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    return sp


def tokenize_file(
    input_path: str,
    output_path: str,
    tokenizer_path: str = "mewtwo_tokenizer.model",
):
    """Tokenize a text file and save as token IDs."""
    sp = load_tokenizer(tokenizer_path)

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    tokens = sp.encode(text, out_type=int)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(" ".join(map(str, tokens)))

    print(f"Tokenized {input_path} -> {output_path} ({len(tokens)} tokens)")
    return tokens


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train MewtwoLLM tokenizer")
    parser.add_argument("--input", required=True, help="Input text file(s)")
    parser.add_argument("--model_prefix", default="mewtwo_tokenizer", help="Output model prefix")
    parser.add_argument("--vocab_size", type=int, default=32000, help="Vocabulary size")
    args = parser.parse_args()

    train_tokenizer(args.input, args.model_prefix, args.vocab_size)
