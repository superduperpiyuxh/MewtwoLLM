"""
Memory-Mapped Token Dataset for MewtwoLLM

Implements efficient data loading using memory-mapped files:
- No full dataset load into RAM
- Fast random access
- Supports very large datasets (OpenWebText: ~17GB)
"""

import os
import sys
import mmap
import struct
from pathlib import Path

import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


class MMapTokenDataset(Dataset):
    """
    Memory-mapped dataset for pre-tokenized text.

    File format:
    - First 8 bytes: uint64 number of tokens
    - Remaining bytes: uint32 tokens (4 bytes each)

    This allows O(1) random access without loading entire file into RAM.
    """

    def __init__(self, token_file: str, block_size: int = 1024):
        self.block_size = block_size
        self.token_file = token_file

        # Open memory-mapped file
        self.file_size = os.path.getsize(token_file)
        self.f = open(token_file, "rb")
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)

        # Read header (number of tokens)
        self.n_tokens = struct.unpack("Q", self.mm[:8])[0]

        # Validate file size
        expected_size = 8 + self.n_tokens * 4
        if self.file_size != expected_size:
            raise ValueError(
                f"File size mismatch: expected {expected_size}, got {self.file_size}"
            )

        self.n_samples = max(0, self.n_tokens - block_size - 1)
        print(f"MMapDataset: {self.n_tokens:,} tokens, {self.n_samples:,} samples "
              f"(block_size={block_size}, file={token_file})")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.n_samples:
            raise IndexError(f"Index {idx} out of range [0, {self.n_samples})")

        # Calculate byte offset
        offset = 8 + idx * 4  # 8 bytes header + idx * 4 bytes per token

        # Read tokens directly from memory-mapped file
        x_bytes = self.mm[offset : offset + self.block_size * 4]
        y_bytes = self.mm[offset + 4 : offset + (self.block_size + 1) * 4]

        x = torch.frombuffer(x_bytes, dtype=torch.int32).long()
        y = torch.frombuffer(y_bytes, dtype=torch.int32).long()

        return x, y

    def close(self):
        """Clean up resources."""
        self.mm.close()
        self.f.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def create_mmap_dataset(
    text_file: str,
    output_file: str,
    tokenizer,
    block_size: int = 1024,
):
    """
    Create a memory-mapped dataset from a text file.

    Args:
        text_file: Path to raw text file
        output_file: Path to output .mmap file
        tokenizer: Tokenizer instance (encode method)
        block_size: Context window size
    """
    print(f"Creating memory-mapped dataset...")
    print(f"  Input: {text_file}")
    print(f"  Output: {output_file}")

    # Read and tokenize text
    with open(text_file, "r", errors="ignore") as f:
        text = f.read()

    tokens = tokenizer.encode(text)
    n_tokens = len(tokens)
    print(f"  Tokens: {n_tokens:,}")

    # Calculate output size
    output_size = 8 + n_tokens * 4  # 8 bytes header + 4 bytes per token
    n_samples = max(0, n_tokens - block_size - 1)
    print(f"  Samples: {n_samples:,}")
    print(f"  File size: {output_size / 1e9:.2f} GB")

    # Write to file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "wb") as f:
        # Write header
        f.write(struct.pack("Q", n_tokens))

        # Write tokens
        f.write(struct.pack(f"{n_tokens}I", *tokens))

    print(f"  Saved to: {output_file}")
    return output_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create memory-mapped dataset")
    parser.add_argument("--input", required=True, help="Input text file")
    parser.add_argument("--output", required=True, help="Output .mmap file")
    parser.add_argument("--tokenizer", default="data/tokenizer", help="Tokenizer path")
    parser.add_argument("--block-size", type=int, default=1024, help="Block size")
    args = parser.parse_args()

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.tokenizer.tokenizer import MewtwoTokenizer

    tokenizer = MewtwoTokenizer.from_pretrained(args.tokenizer)
    create_mmap_dataset(args.input, args.output, tokenizer, args.block_size)
