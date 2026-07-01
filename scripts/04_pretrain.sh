#!/usr/bin/env bash
set -euo pipefail

echo "=== Step 4: Pretrain MewtwoLLM ==="

PROJECT_DIR="$(dirname "$0")/.."
DATA_DIR="$PROJECT_DIR/data"
CKPT_DIR="$PROJECT_DIR/checkpoints"

mkdir -p "$CKPT_DIR/pretrain"

echo "Configuration:"
echo "  Project dir: $PROJECT_DIR"
echo "  Data dir:    $DATA_DIR"
echo "  Checkpoints: $CKPT_DIR/pretrain"
echo ""

# Check for training data
if [ ! -d "$DATA_DIR/raw" ] && [ ! -d "$DATA_DIR/openwebtext" ]; then
    echo "ERROR: No training data found. Run 02_download.sh first."
    exit 1
fi

# Run pretraining
python3 -c "
import sys
import os
import torch

sys.path.insert(0, '$PROJECT_DIR')
from config.model_config import MewtwoConfig
from src.model.pocketllm import MewtwoLLM

# Load config
config = MewtwoConfig()
print(f'Model: {config.dim}d, {config.n_layers}L, {config.n_heads}H, {config.n_kv_heads}KV')

# Check for OpenWebText (HuggingFace datasets format)
data_path = '$DATA_DIR/raw'
openwebtext_path = '$DATA_DIR/openwebtext'

if os.path.exists(openwebtext_path):
    print(f'Using OpenWebText dataset from: {openwebtext_path}')
    print('Note: Use 04_pretrain_hf.py for HuggingFace datasets integration')
elif os.path.exists(data_path):
    print(f'Using raw text data from: {data_path}')
    # List available files
    files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
    print(f'Available files: {len(files)}')
else:
    print('ERROR: No training data found')
    sys.exit(1)

print('\\nPretraining script ready.')
print('For full training, use: python3 -m src.training.pretrain')
print('Or run on Colab with: python3 colab/MewtwoLLM_Train.ipynb')
"

echo ""
echo "Training infrastructure ready!"
echo "Next steps:"
echo "  1. Tokenize data:     ./03_tokenize.sh"
echo "  2. Start training:    python3 -m src.training.pretrain"
echo "  3. Or use Colab:      Open colab/MewtwoLLM_Train.ipynb"
