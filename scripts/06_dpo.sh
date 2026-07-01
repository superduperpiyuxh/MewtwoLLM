#!/usr/bin/env bash
set -euo pipefail

echo "=== Step 6: DPO Alignment ==="

PROJECT_DIR="$(dirname "$0")/.."
DATA_DIR="$PROJECT_DIR/data"
CKPT_DIR="$PROJECT_DIR/checkpoints"

mkdir -p "$CKPT_DIR/dpo" "$DATA_DIR/preference"

# Create preference dataset if it doesn't exist
if [ ! -f "$DATA_DIR/preference/preferences.jsonl" ]; then
    echo "Creating preference dataset..."
    python3 -c "
import json
import os

data_dir = '$DATA_DIR/preference'
os.makedirs(data_dir, exist_ok=True)

preferences = [
    {
        'prompt': 'What is machine learning?',
        'chosen': 'Machine learning is a subset of artificial intelligence where systems learn patterns from data.',
        'rejected': 'idk lol its when computers learn stuff i guess.'
    },
    {
        'prompt': 'Explain gradient descent.',
        'chosen': 'Gradient descent is an optimization algorithm that iteratively adjusts model parameters to minimize a loss function.',
        'rejected': 'gradient descent is when you go downhill on a graph.'
    },
    {
        'prompt': 'What is attention in transformers?',
        'chosen': 'Attention computes a weighted sum of all input tokens for each output position.',
        'rejected': 'attention is when the model pays attention to important words.'
    },
]

with open(os.path.join(data_dir, 'preferences.jsonl'), 'w') as f:
    for item in preferences:
        f.write(json.dumps(item) + '\n')

print(f'Created {len(preferences)} preference examples')
"
fi

# Run DPO
echo "Starting DPO alignment..."
python3 -c "
import sys
import os

sys.path.insert(0, '$PROJECT_DIR')
from config.model_config import MewtwoConfig
from src.alignment.dpo import train_dpo

config = MewtwoConfig()

# Load SFT checkpoint
sft_ckpt = '$CKPT_DIR/sft/best_model.pt'
if not os.path.exists(sft_ckpt):
    sft_ckpt = '$CKPT_DIR/sft/model.pt'
if not os.path.exists(sft_ckpt):
    print('ERROR: No SFT checkpoint found. Run 05_sft.sh first.')
    sys.exit(1)

train_dpo(
    config=config,
    model_path=sft_ckpt,
    data_path='$DATA_DIR/preference/preferences.jsonl',
    tokenizer_path='$DATA_DIR/tokenizer/mewtwo.model',
    output_dir='$CKPT_DIR/dpo',
)
print('DPO alignment complete!')
"
