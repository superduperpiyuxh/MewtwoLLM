#!/usr/bin/env bash
set -euo pipefail

echo "=== Step 7: RLHF Alignment ==="

PROJECT_DIR="$(dirname "$0")/.."
DATA_DIR="$PROJECT_DIR/data"
CKPT_DIR="$PROJECT_DIR/checkpoints"

mkdir -p "$CKPT_DIR/rlhf"

# Create prompts dataset
if [ ! -f "$DATA_DIR/instruction/prompts.jsonl" ]; then
    echo "Creating prompts dataset..."
    python3 -c "
import json
import os

data_dir = '$DATA_DIR/instruction'
os.makedirs(data_dir, exist_ok=True)

prompts = [
    'Explain artificial intelligence in simple terms.',
    'What are the benefits of deep learning?',
    'How do neural networks learn?',
    'What is the difference between AI and ML?',
    'Describe the future of technology.',
]

with open(os.path.join(data_dir, 'prompts.jsonl'), 'w') as f:
    for p in prompts:
        f.write(json.dumps({'prompt': p}) + '\n')

print(f'Created {len(prompts)} prompts')
"
fi

# Run RLHF
echo "Starting RLHF alignment..."
python3 -c "
import sys
import os
import torch

sys.path.insert(0, '$PROJECT_DIR')
from config.model_config import MewtwoConfig
from src.tokenizer.tokenizer import MewtwoTokenizer
from src.alignment.rlhf import RLHFTrainer

config = MewtwoConfig()
tokenizer = MewtwoTokenizer.from_pretrained('$DATA_DIR/tokenizer')

# Load DPO checkpoint
dpo_ckpt = '$CKPT_DIR/dpo/best_model.pt'
if os.path.exists(dpo_ckpt):
    print(f'Loading DPO checkpoint: {dpo_ckpt}')
else:
    print('WARNING: No DPO checkpoint found. Using SFT/pretrain checkpoint.')

rlhf = RLHFTrainer(
    config=config,
    tokenizer=tokenizer,
    output_dir='$CKPT_DIR/rlhf',
    lr=1e-5,
    max_steps=200,
    batch_size=2,
    gradient_accumulation_steps=4,
    kl_coef=0.1,
)

rlhf.train('$DATA_DIR/instruction/prompts.jsonl')
print('RLHF alignment complete!')
"
