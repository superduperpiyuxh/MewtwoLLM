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
from src.tokenizer.tokenizer import MewtwoTokenizer
from src.evaluation.eval import evaluate_perplexity

config = MewtwoConfig()
tokenizer = MewtwoTokenizer.from_pretrained('$DATA_DIR/tokenizer')

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
texts = []
for fname in os.listdir(data_dir):
    fpath = os.path.join(data_dir, fname)
    if os.path.isfile(fpath) and not fname.startswith('.'):
        try:
            with open(fpath, 'r', errors='ignore') as f:
                text = f.read()
                if len(text.strip()) > 200:
                    texts.append(text[:5000])
        except:
            pass

if texts:
    results = evaluate_perplexity(model, tokenizer, texts[:5])
    print(f'Perplexity: {results[\"perplexity\"]:.2f}')
    print(f'Loss: {results[\"loss\"]:.4f}')
else:
    print('No evaluation data found. Run 01_scrape.sh first.')
"
