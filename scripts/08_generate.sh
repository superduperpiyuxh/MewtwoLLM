#!/usr/bin/env bash
set -euo pipefail

echo "=== Step 8: Text Generation ==="

PROJECT_DIR="$(dirname "$0")/.."
DATA_DIR="$PROJECT_DIR/data"
CKPT_DIR="$PROJECT_DIR/checkpoints"

# Default prompt
PROMPT="${1:-What is the meaning of life?}"
MAX_TOKENS="${2:-200}"
TEMPERATURE="${3:-0.8}"

echo "Prompt: $PROMPT"
echo "Max tokens: $MAX_TOKENS"
echo "Temperature: $TEMPERATURE"
echo ""

python3 -c "
import sys
import os
import torch

sys.path.insert(0, '$PROJECT_DIR')
from config.model_config import MewtwoConfig
from src.model.pocketllm import MewtwoLLM
from src.tokenizer.tokenizer import MewtwoTokenizer
from src.inference.generate import generate

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

print()
print('=' * 60)
print(f'Prompt: $PROMPT')
print('=' * 60)

output = generate(
    model=model,
    tokenizer=tokenizer,
    prompt='$PROMPT',
    max_new_tokens=$MAX_TOKENS,
    temperature=$TEMPERATURE,
    top_k=50,
    top_p=0.95,
)

print(output)
print('=' * 60)
"
