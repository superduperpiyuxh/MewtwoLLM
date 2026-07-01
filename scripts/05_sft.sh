#!/usr/bin/env bash
set -euo pipefail

echo "=== Step 5: Supervised Fine-Tuning (SFT) ==="

PROJECT_DIR="$(dirname "$0")/.."
DATA_DIR="$PROJECT_DIR/data"
CKPT_DIR="$PROJECT_DIR/checkpoints"

mkdir -p "$CKPT_DIR/sft" "$DATA_DIR/instruction"

# Create instruction dataset if it doesn't exist
if [ ! -f "$DATA_DIR/instruction/instructions.jsonl" ]; then
    echo "Creating instruction dataset..."
    python3 -c "
import json
import os

data_dir = '$DATA_DIR/instruction'
os.makedirs(data_dir, exist_ok=True)

instructions = [
    {'instruction': 'Explain what a transformer is in machine learning.', 'input': '', 'output': 'A transformer is a neural network architecture that uses self-attention mechanisms to process sequential data.'},
    {'instruction': 'What is the difference between DPO and RLHF?', 'input': '', 'output': 'RLHF trains a reward model then uses PPO. DPO directly optimizes using preference data.'},
    {'instruction': 'What does RoPE do?', 'input': '', 'output': 'RoPE encodes positional information by rotating query and key vectors in attention.'},
    {'instruction': 'Why use RMSNorm instead of LayerNorm?', 'input': '', 'output': 'RMSNorm is simpler and faster while maintaining equivalent performance.'},
    {'instruction': 'What is SwiGLU?', 'input': '', 'output': 'SwiGLU is an activation function that combines Swish with a Gated Linear Unit.'},
    {'instruction': 'How does GQA work?', 'input': '', 'output': 'GQA groups query heads to share KV pairs, reducing memory while maintaining quality.'},
]

with open(os.path.join(data_dir, 'instructions.jsonl'), 'w') as f:
    for item in instructions:
        f.write(json.dumps(item) + '\n')

print(f'Created {len(instructions)} instruction examples')
"
fi

# Run SFT
echo "Starting supervised fine-tuning..."
python3 -c "
import sys
import os

sys.path.insert(0, '$PROJECT_DIR')
from config.model_config import MewtwoConfig
from src.alignment.sft import train_sft

config = MewtwoConfig()

# Load latest pretrain checkpoint
ckpt_dir = '$CKPT_DIR/pretrain'
best_ckpt = os.path.join(ckpt_dir, 'best_model.pt')
if not os.path.exists(best_ckpt):
    best_ckpt = os.path.join(ckpt_dir, 'model.pt')
if not os.path.exists(best_ckpt):
    print('ERROR: No pretrain checkpoint found. Run 04_pretrain.sh first.')
    sys.exit(1)

train_sft(
    config=config,
    model_path=best_ckpt,
    data_path='$DATA_DIR/instruction/instructions.jsonl',
    tokenizer_path='$DATA_DIR/tokenizer/mewtwo.model',
    output_dir='$CKPT_DIR/sft',
)
print('SFT complete!')
"
