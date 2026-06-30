#!/usr/bin/env bash
set -euo pipefail

echo "=== Step 4: Pretrain MewtwoLLM ==="

PROJECT_DIR="$(dirname "$0")/.."
DATA_DIR="$PROJECT_DIR/data"
CKPT_DIR="$PROJECT_DIR/checkpoints"
CONFIG_FILE="$PROJECT_DIR/config/model_config.py"

mkdir -p "$CKPT_DIR/pretrain"

echo "Configuration:"
echo "  Project dir: $PROJECT_DIR"
echo "  Data dir:    $DATA_DIR"
echo "  Checkpoints: $CKPT_DIR/pretrain"
echo ""

# Run pretraining
python3 -c "
import sys
import os
import torch

sys.path.insert(0, '$PROJECT_DIR')
from config.model_config import MewtwoConfig
from src.tokenizer.tokenizer import MewtwoTokenizer
from src.training.pretrain import Pretrainer

# Load config
config = MewtwoConfig()
print(f'Model: {config.d_model}d, {config.n_layers}L, {config.n_heads}H, {config.n_kv_heads}KV')
print(f'Params: {sum(p.numel() for p in torch.zeros(1).new_empty(0)):,} (will print real count after model init)')

# Load tokenizer
tokenizer_path = '$DATA_DIR/tokenizer'
if os.path.exists(os.path.join(tokenizer_path, 'mewtwo.model')):
    tokenizer = MewtwoTokenizer.from_pretrained(tokenizer_path)
    print(f'Tokenizer: vocab={tokenizer.vocab_size}')
else:
    print('WARNING: No tokenizer found. Run 03_tokenize.sh first.')
    tokenizer = MewtwoTokenizer(vocab_size=32000)

# Prepare data
data_dir = '$DATA_DIR/raw'
train_data = []
for fname in os.listdir(data_dir):
    fpath = os.path.join(data_dir, fname)
    if os.path.isfile(fpath) and not fname.startswith('.'):
        try:
            with open(fpath, 'r', errors='ignore') as f:
                text = f.read()
                if len(text.strip()) > 100:
                    train_data.append(text)
        except:
            pass

if not train_data:
    print('ERROR: No training data found. Run 01_scrape.sh and 02_download.sh first.')
    sys.exit(1)

print(f'Training data: {len(train_data)} chunks')

# Train
trainer = Pretrainer(
    config=config,
    tokenizer=tokenizer,
    output_dir='$CKPT_DIR/pretrain',
    lr=3e-4,
    warmup_steps=100,
    max_steps=1000,
    batch_size=4,
    gradient_accumulation_steps=4,
    log_every=10,
    eval_every=100,
    save_every=500,
    grad_clip=1.0,
)

trainer.train(train_data)
print('Pretraining complete!')
"
