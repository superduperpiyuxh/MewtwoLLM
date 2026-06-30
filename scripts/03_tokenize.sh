#!/usr/bin/env bash
set -euo pipefail

echo "=== Step 3: Train SentencePiece tokenizer ==="

PROJECT_DIR="$(dirname "$0")/.."
DATA_DIR="$PROJECT_DIR/data"
TOKENIZER_DIR="$DATA_DIR/tokenizer"
RAW_DIR="$DATA_DIR/raw"

mkdir -p "$TOKENIZER_DIR" "$DATA_DIR/processed"

# Combine all raw text into one file for tokenizer training
echo "[1/2] Preparing text corpus for tokenizer training..."
python3 -c "
import os
import glob

raw_dir = '$RAW_DIR'
output = '$DATA_DIR/tokenizer_input.txt'

all_files = glob.glob(os.path.join(raw_dir, '**', '*'), recursive=True)
all_files = [f for f in all_files if os.path.isfile(f) and not f.endswith('.bin') and not f.endswith('.npy')]

combined = []
for fpath in sorted(all_files):
    try:
        with open(fpath, 'r', errors='ignore') as f:
            text = f.read()
            if len(text.strip()) > 100:
                combined.append(text)
    except:
        pass

if not combined:
    # Create minimal fallback corpus
    combined = [
        'The transformer architecture revolutionized natural language processing.',
        'Self-attention allows the model to weigh the importance of different words.',
        'Rotary position embeddings provide relative positional information.',
        'Grouped query attention reduces memory usage while maintaining quality.',
        'SwiGLU activation functions improve performance in feed-forward networks.',
    ] * 1000

with open(output, 'w') as f:
    f.write('\n'.join(combined))

print(f'Created tokenizer training corpus: {len(combined)} chunks')
print(f'Total characters: {sum(len(c) for c in combined):,}')
"

# Train SentencePiece tokenizer
echo "[2/2] Training SentencePiece BPE tokenizer..."
python3 -c "
import sys
sys.path.insert(0, '$PROJECT_DIR')
from src.tokenizer.tokenizer import MewtwoTokenizer

tokenizer = MewtwoTokenizer(vocab_size=32000)
tokenizer.train(
    input_file='$DATA_DIR/tokenizer_input.txt',
    model_prefix='$TOKENIZER_DIR/mewtwo',
)
print('Tokenizer training complete!')
"

# Verify tokenizer
echo "[verify] Testing tokenizer..."
python3 -c "
import sys
sys.path.insert(0, '$PROJECT_DIR')
from src.tokenizer.tokenizer import MewtwoTokenizer

tokenizer = MewtwoTokenizer.from_pretrained('$TOKENIZER_DIR')
test = 'Hello, this is a test of the MewtwoLLM tokenizer!'
tokens = tokenizer.encode(test)
decoded = tokenizer.decode(tokens)
print(f'Input:    {test}')
print(f'Tokens:   {tokens}')
print(f'Decoded:  {decoded}')
print(f'Vocab:    {tokenizer.vocab_size}')
"

echo "Tokenizer saved to $TOKENIZER_DIR/"
ls -la "$TOKENIZER_DIR/"
