"""Download a small OpenWebText sample for local training."""
import os
from datasets import load_dataset

os.makedirs("data/raw", exist_ok=True)
output_path = "data/raw/openwebtext_sample.txt"

print("Downloading OpenWebText (streaming 200 samples)...")
ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True, trust_remote_code=True)

count = 0
with open(output_path, "w") as f:
    for item in ds:
        text = item["text"].strip()
        if len(text) > 100:
            f.write(text + "\n\n")
            count += 1
        if count >= 200:
            break
        if count % 50 == 0:
            print(f"  {count}/200...")

size_mb = os.path.getsize(output_path) / 1e6
print(f"Done: {count} documents, {size_mb:.1f} MB")
