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

sys.path.insert(0, '$PROJECT_DIR')
from config.model_config import MewtwoConfig
from src.alignment.rlhf import train_rlhf

config = MewtwoConfig()

# Load DPO checkpoint
dpo_ckpt = '$CKPT_DIR/dpo/best_model.pt'
if not os.path.exists(dpo_ckpt):
    dpo_ckpt = '$CKPT_DIR/dpo/model.pt'
if not os.path.exists(dpo_ckpt):
    print('ERROR: No DPO checkpoint found. Run 06_dpo.sh first.')
    sys.exit(1)

train_rlhf(
    config=config,
    model_path=dpo_ckpt,
    data_path='$DATA_DIR/instruction/prompts.jsonl',
    tokenizer_path='$DATA_DIR/tokenizer/mewtwo.model',
    output_dir='$CKPT_DIR/rlhf',
)
print('RLHF alignment complete!')
"
