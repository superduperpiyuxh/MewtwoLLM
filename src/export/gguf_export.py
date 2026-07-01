"""
GGUF Export for MewtwoLLM

Exports model weights to GGUF format for use with llama.cpp.
"""

import os
import sys
import struct
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


class GGUFWriter:
    """Simple GGUF file writer."""

    def __init__(self, path: str):
        self.path = path
        self.f = open(path, "wb")
        self.write_header()

    def write_header(self):
        """Write GGUF magic number and version."""
        self.f.write(b"GGUF")
        self.f.write(struct.pack("<I", 3))  # GGUF version 3

    def write_metadata(self, key: str, value: str):
        """Write string metadata."""
        key_bytes = key.encode("utf-8")
        val_bytes = value.encode("utf-8")
        self.f.write(struct.pack("<Q", len(key_bytes)))
        self.f.write(key_bytes)
        self.f.write(struct.pack("<I", 8))  # STRING type
        self.f.write(struct.pack("<Q", len(val_bytes)))
        self.f.write(val_bytes)

    def write_metadata_int(self, key: str, value: int):
        """Write integer metadata."""
        key_bytes = key.encode("utf-8")
        self.f.write(struct.pack("<Q", len(key_bytes)))
        self.f.write(key_bytes)
        self.f.write(struct.pack("<I", 4))  # UINT32 type
        self.f.write(struct.pack("<I", value))

    def write_tensor(self, name: str, tensor: torch.Tensor):
        """Write tensor data."""
        name_bytes = name.encode("utf-8")
        self.f.write(struct.pack("<Q", len(name_bytes)))
        self.f.write(name_bytes)

        # Write shape
        ndim = len(tensor.shape)
        self.f.write(struct.pack("<I", ndim))
        for dim in reversed(tensor.shape):
            self.f.write(struct.pack("<Q", dim))

        # Write type (FP32 = 0)
        self.f.write(struct.pack("<I", 0))

        # Write data
        data = tensor.detach().cpu().numpy().astype(np.float32).tobytes()
        # Pad to 32-byte alignment
        padding = (32 - (self.f.tell() % 32)) % 32
        self.f.write(b"\x00" * padding)
        self.f.write(data)

    def close(self):
        self.f.close()


def export_to_gguf(
    model,
    config,
    output_path: str,
    model_name: str = "MewtwoLLM",
):
    """
    Export MewtwoLLM weights to GGUF format.

    Args:
        model: MewtwoLLM model instance
        config: MewtwoConfig
        output_path: Path to output .gguf file
        model_name: Name for the model
    """
    print(f"Exporting to GGUF: {output_path}")

    writer = GGUFWriter(output_path)

    # Write metadata
    writer.write_metadata("general.name", model_name)
    writer.write_metadata("general.architecture", "llama")
    writer.write_metadata_int("llama.context_length", config.context_length)
    writer.write_metadata_int("llama.embedding_length", config.dim)
    writer.write_metadata_int("llama.block_count", config.n_layers)
    writer.write_metadata_int("llama.feed_forward_length", config.ff_dim)
    writer.write_metadata_int("llama.attention.head_count", config.n_heads)
    writer.write_metadata_int("llama.attention.head_count_kv", config.n_kv_heads)
    writer.write_metadata_int("llama.rope.dimension_count", config.rope_dim)

    # Write tensors
    state_dict = model.state_dict()

    # Map our layer names to GGUF/llama.cpp format
    tensor_map = {
        "token_embed.weight": "token_embd.weight",
        "norm.weight": "output_norm.weight",
        "lm_head.weight": "output.weight",
    }

    for i, block in enumerate(model.blocks):
        tensor_map.update({
            f"blocks.{i}.attn_norm.weight": f"blk.{i}.attn_norm.weight",
            f"blocks.{i}.attention.wq.weight": f"blk.{i}.attn_q.weight",
            f"blocks.{i}.attention.wk.weight": f"blk.{i}.attn_k.weight",
            f"blocks.{i}.attention.wv.weight": f"blk.{i}.attn_v.weight",
            f"blocks.{i}.attention.wo.weight": f"blk.{i}.attn_output.weight",
            f"blocks.{i}.ffn_norm.weight": f"blk.{i}.ffn_norm.weight",
            f"blocks.{i}.ffn.w1.weight": f"blk.{i}.ffn_gate.weight",
            f"blocks.{i}.ffn.w2.weight": f"blk.{i}.ffn_down.weight",
            f"blocks.{i}.ffn.w3.weight": f"blk.{i}.ffn_up.weight",
        })

    for our_name, gguf_name in tensor_map.items():
        if our_name in state_dict:
            writer.write_tensor(gguf_name, state_dict[our_name])
            print(f"  {our_name} -> {gguf_name}")

    writer.close()
    print(f"GGUF exported: {output_path}")

    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export MewtwoLLM to GGUF")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--output", required=True, help="Output .gguf file")
    parser.add_argument("--model-name", default="MewtwoLLM", help="Model name")
    args = parser.parse_args()

    from config.model_config import MewtwoConfig
    from src.model.pocketllm import MewtwoLLM

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint["config"]
    model = MewtwoLLM(config)
    model.load_state_dict(checkpoint["model_state_dict"])

    export_to_gguf(model, config, args.output, args.model_name)
