"""
MewtwoLLM — Modal Training Script
Runs pretraining on a cloud T4 GPU.
"""

import modal

app = modal.App("mewtwo-train")

# Define container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "sentencepiece", "huggingface_hub", "datasets", "numpy")
    .apt_install("git")
)

@app.function(
    image=image,
    gpu="T4",
    timeout=7200,  # 2 hours max
    volumes={"/data": modal.Volume.from_name("mewtwo-data", create_if_missing=True)},
)
def train():
    import os
    import time
    import torch
    import sys

    print("=" * 60)
    print("MewtwoLLM — Modal Training")
    print("=" * 60)

    # Clone repo
    if not os.path.exists("/data/MewtwoLLM"):
        print("Cloning MewtwoLLM repo...")
        os.system("git clone https://github.com/superduperpiyuxh/MewtwoLLM.git /data/MewtwoLLM")

    sys.path.insert(0, "/data/MewtwoLLM")

    from config.model_config import MewtwoConfig
    from src.model.pocketllm import MewtwoLLM
    from src.training.pretrain import train as pretrain

    # Config
    config = MewtwoConfig()
    config.total_steps = 500  # Remaining steps (already did 500)
    config.batch_size = 4
    config.gradient_accumulation_steps = 4
    config.device = "cuda"
    config.use_compile = False  # Save VRAM

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Download data if needed
    data_path = "/data/openwebtext_sample_tokens.txt"
    if not os.path.exists(data_path):
        print("Downloading OpenWebText (streaming 500 samples)...")
        from datasets import load_dataset

        ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True, trust_remote_code=True)

        # Save raw text
        raw_path = "/data/openwebtext_sample.txt"
        count = 0
        with open(raw_path, "w") as f:
            for item in ds:
                text = item["text"].strip()
                if len(text) > 100:
                    f.write(text + "\n\n")
                    count += 1
                if count >= 500:
                    break
                if count % 100 == 0:
                    print(f"  {count}/500...")

        print(f"Downloaded {count} documents")

        # Tokenize
        print("Tokenizing...")
        import sentencepiece as spm

        # Train tokenizer
        spm.SentencePieceTrainer.Train(
            input=raw_path,
            model_prefix="/data/mewtwo_sp",
            vocab_size=32000,
            model_type="bpe",
            character_coverage=1.0,
            pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        )

        sp = spm.SentencePieceProcessor()
        sp.Load("/data/mewtwo_sp.model")

        with open(raw_path, "r") as f:
            text = f.read()

        tokens = sp.EncodeAsIds(text)
        with open(data_path, "w") as f:
            f.write(" ".join(map(str, tokens)))

        print(f"Tokenized: {len(tokens):,} tokens")
    else:
        print(f"Using existing data: {data_path}")

    # Train
    print("\nStarting training...")
    pretrain(config, data_path, checkpoint_dir="/data/checkpoints/pretrain")

    # Save final model to volume
    print("\nTraining complete! Checkpoints saved to /data/checkpoints/")
    return "Training complete!"


@app.local_entrypoint()
def main():
    train.remote()
