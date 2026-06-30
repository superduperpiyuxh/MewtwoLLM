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
        'chosen': 'Machine learning is a subset of artificial intelligence where systems learn patterns from data to make predictions or decisions without being explicitly programmed. The key types are supervised learning (learning from labeled examples), unsupervised learning (finding patterns in unlabeled data), and reinforcement learning (learning through trial and error with rewards).',
        'rejected': 'idk lol its when computers learn stuff i guess. like they look at data and figure things out maybe.'
    },
    {
        'prompt': 'Explain gradient descent.',
        'chosen': 'Gradient descent is an optimization algorithm that iteratively adjusts model parameters to minimize a loss function. At each step, it computes the gradient (direction of steepest increase) of the loss with respect to the parameters, then moves in the opposite direction. The step size is controlled by the learning rate. Variants include SGD, Adam, and AdaFactor.',
        'rejected': 'gradient descent is when you go downhill on a graph. you follow the slope down until you reach the bottom.'
    },
    {
        'prompt': 'What is attention in transformers?',
        'chosen': 'Attention in transformers computes a weighted sum of all input tokens for each output position. For each token, it creates a query (what am I looking for?), a key (what do I contain?), and a value (what information do I provide?). The attention score between two tokens is the dot product of their query and key, scaled and softmaxed. This allows the model to focus on relevant parts of the input.',
        'rejected': 'attention is when the model pays attention to important words. it uses some math to figure out which words matter more than others.'
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
import torch

sys.path.insert(0, '$PROJECT_DIR')
from config.model_config import MewtwoConfig
from src.tokenizer.tokenizer import MewtwoTokenizer
from src.alignment.dpo import DPOTrainer

config = MewtwoConfig()
tokenizer = MewtwoTokenizer.from_pretrained('$DATA_DIR/tokenizer')

# Load SFT checkpoint
sft_ckpt = '$CKPT_DIR/sft/best_model.pt'
if os.path.exists(sft_ckpt):
    print(f'Loading SFT checkpoint: {sft_ckpt}')
else:
    print('WARNING: No SFT checkpoint found. Using pretrain checkpoint.')

dpo = DPOTrainer(
    config=config,
    tokenizer=tokenizer,
    output_dir='$CKPT_DIR/dpo',
    lr=5e-6,
    max_steps=200,
    batch_size=2,
    gradient_accumulation_steps=4,
    beta=0.1,
)

dpo.train('$DATA_DIR/preference/preferences.jsonl')
print('DPO alignment complete!')
"
