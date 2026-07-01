#!/usr/bin/env python3
"""
Upload MewtwoLLM checkpoints and tokenizer to HuggingFace Hub.

Usage:
    python upload_to_hub.py --token hf_xxxxx --repo superduperpiyuxh/MewtwoLLM-40M
"""

import argparse
import os
import sys
from huggingface_hub import HfApi, login


def main():
    parser = argparse.ArgumentParser(description="Upload MewtwoLLM to HF Hub")
    parser.add_argument("--token", required=True, help="HF token (from https://huggingface.co/settings/tokens)")
    parser.add_argument("--repo", default="superduperpiyuxh/MewtwoLLM-40M", help="HF repo name")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Directory with checkpoints")
    parser.add_argument("--tokenizer-dir", default="data/tokenizer", help="Tokenizer directory")
    args = parser.parse_args()

    login(token=args.token)
    api = HfApi()

    # Create repo
    api.create_repo(repo_id=args.repo, exist_ok=True)
    print(f"Repo: https://huggingface.co/{args.repo}")

    # Find and upload best checkpoint
    uploaded = False
    for stage in ["rlhf", "dpo", "sft", "pretrain"]:
        for name in ["mewtwo_final.pt", "mewtwo_best.pt", "model.pt"]:
            ckpt = os.path.join(args.checkpoint_dir, stage, name)
            if os.path.exists(ckpt):
                print(f"Uploading {stage}/{name}...")
                api.upload_file(
                    path_or_fileobj=ckpt,
                    path_in_repo=f"{stage}/{name}",
                    repo_id=args.repo,
                )
                uploaded = True
                break
        if uploaded:
            break

    if not uploaded:
        print("WARNING: No checkpoint found! Run training first.")

    # Upload tokenizer
    if os.path.isdir(args.tokenizer_dir):
        print("Uploading tokenizer...")
        for fname in os.listdir(args.tokenizer_dir):
            fpath = os.path.join(args.tokenizer_dir, fname)
            if os.path.isfile(fpath):
                api.upload_file(
                    path_or_fileobj=fpath,
                    path_in_repo=f"tokenizer/{fname}",
                    repo_id=args.repo,
                )

    # Upload config
    if os.path.exists("config/model_config.py"):
        api.upload_file(
            path_or_fileobj="config/model_config.py",
            path_in_repo="config/model_config.py",
            repo_id=args.repo,
        )

    print(f"\nDone! Model at: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
