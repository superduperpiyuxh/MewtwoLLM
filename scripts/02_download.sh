#!/usr/bin/env bash
set -euo pipefail

echo "=== Step 2: Download open datasets ==="

DATA_DIR="$(dirname "$0")/../data"

mkdir -p "$DATA_DIR/raw" "$DATA_DIR/processed"

# Download FineWeb-Edu sample from HuggingFace
echo "[1/2] Downloading FineWeb-Edu sample..."
python3 -c "
import os
import json

data_dir = '$DATA_DIR/raw'

# Try huggingface_hub first, fall back to manual download
try:
    from huggingface_hub import hf_hub_download
    
    # Download a small sample (10K rows) of FineWeb-Edu
    print('Downloading FineWeb-Edu-100k sample from HuggingFace...')
    path = hf_hub_download(
        repo_id='HuggingFaceFW/fineweb-edu',
        filename='sample/100k.txt',
        repo_type='dataset',
        local_dir=data_dir,
    )
    print(f'Downloaded to: {path}')
except Exception as e:
    print(f'huggingface_hub download failed: {e}')
    print('Downloading via requests...')
    
    import requests
    
    url = 'https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/100k.txt'
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    
    out_path = os.path.join(data_dir, 'fineweb_edu_100k.txt')
    with open(out_path, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f'Downloaded to: {out_path}')
"

# Download OpenWebText sample
echo "[2/2] Downloading OpenWebText sample..."
python3 -c "
import os
import json

data_dir = '$DATA_DIR/raw'

try:
    from huggingface_hub import hf_hub_download
    
    print('Downloading OpenWebText sample from HuggingFace...')
    path = hf_hub_download(
        repo_id='Skylion007/openwebtext',
        filename='openwebtext-10k.txt',
        repo_type='dataset',
        local_dir=data_dir,
    )
    print(f'Downloaded to: {path}')
except Exception as e:
    print(f'huggingface_hub download failed: {e}')
    print('Falling back to creating synthetic text corpus...')
    
    # Create a synthetic corpus from our scraped data
    scraped_dir = os.path.join(data_dir, 'raw')
    combined = []
    for fname in os.listdir(scraped_dir):
        fpath = os.path.join(scraped_dir, fname)
        if os.path.isfile(fpath) and not fname.startswith('.'):
            try:
                with open(fpath, 'r', errors='ignore') as f:
                    combined.append(f.read())
            except:
                pass
    
    if combined:
        out_path = os.path.join(data_dir, 'scraped_combined.txt')
        with open(out_path, 'w') as f:
            f.write('\n\n'.join(combined))
        print(f'Combined scraped data to: {out_path}')
    else:
        print('No scraped data found yet. Run 01_scrape.sh first.')
"

echo "Download complete. Raw data:"
ls -la "$DATA_DIR/raw/"
