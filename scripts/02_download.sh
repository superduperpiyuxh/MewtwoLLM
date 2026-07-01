#!/usr/bin/env bash
set -euo pipefail

echo "=== Step 2: Download OpenWebText dataset ==="

DATA_DIR="$(dirname "$0")/../data"

mkdir -p "$DATA_DIR/raw" "$DATA_DIR/processed"

# Download OpenWebText using HuggingFace datasets
echo "[1/1] Downloading OpenWebText..."
python3 -c "
import os
import sys

data_dir = '$DATA_DIR/raw'

# Try huggingface datasets first
try:
    from datasets import load_dataset
    
    print('Downloading OpenWebText from HuggingFace...')
    print('(This may take a while — ~8M documents, ~17GB compressed)')
    
    # Load dataset
    dataset = load_dataset('Skylion007/openwebtext', trust_remote_code=True)
    
    # Save train split
    print(f'Train samples: {len(dataset[\"train\"]):,}')
    
    # Save as Arrow format for memory-mapped access
    dataset['train'].save_to_disk(os.path.join(data_dir, 'openwebtext'))
    print(f'Saved to: {os.path.join(data_dir, \"openwebtext\")}')
    
except Exception as e:
    print(f'datasets library failed: {e}')
    print('Trying huggingface_hub fallback...')
    
    try:
        from huggingface_hub import hf_hub_download
        
        # Download a small sample for testing
        print('Downloading OpenWebText sample (10K docs)...')
        path = hf_hub_download(
            repo_id='Skylion007/openwebtext',
            filename='openwebtext_train.parquet',
            repo_type='dataset',
            local_dir=data_dir,
        )
        print(f'Downloaded to: {path}')
        
    except Exception as e2:
        print(f'huggingface_hub also failed: {e2}')
        print('ERROR: Could not download OpenWebText')
        print('Please install: pip install datasets')
        sys.exit(1)
"

echo ""
echo "Download complete. Data files:"
ls -la "$DATA_DIR/raw/"
