#!/usr/bin/env bash
set -euo pipefail

echo "=== Step 9: Evaluation ==="

PROJECT_DIR="$(dirname "$0")/.."
DATA_DIR="$PROJECT_DIR/data"
CKPT_DIR="$PROJECT_DIR/checkpoints"

python3 -c "
import sys
import os
import torch

sys.path.insert(0, '$PROJECT_DIR')
from config.model_config import MewtwoConfig
from src.model.pocketllm import MewtwoLLM
from src.evaluation.eval import compute_perplexity

config = MewtwoConfig()

model = MewtwoLLM(config)
model.eval()

# Load latest checkpoint
for ckpt_name in ['best_model.pt', 'model.pt']:
    for stage in ['rlhf', 'dpo', 'sft', 'pretrain']:
        ckpt_path = f'$CKPT_DIR/{stage}/{ckpt_name}'
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f'Loaded checkpoint: {ckpt_path}')
            break
    else:
        continue
    break
else:
    print('WARNING: No checkpoint found. Using random weights.')

# Evaluate on scraped data
data_dir = '$DATA_DIR/raw'
token_file = None
for fname in os.listdir(data_dir):
    fpath = os.path.join(data_dir, fname)
    if fname.endswith('.tokens'):
        token_file = fpath
        break

if token_file:
    results = compute_perplexity(model, token_file, device='cpu')
    print(f'Perplexity: {results[\"perplexity\"]:.2f}')
    print(f'Loss: {results[\"loss\"]:.4f}')
else:
    print('No tokenized data found. Run 03_tokenize.sh first.')
"
