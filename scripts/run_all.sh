#!/usr/bin/env bash
set -euo pipefail

echo "============================================"
echo "  MewtwoLLM — Full Training Pipeline"
echo "============================================"
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Step 1/9: Scrape training data..."
bash "$SCRIPT_DIR/01_scrape.sh"
echo ""

echo "Step 2/9: Download open datasets..."
bash "$SCRIPT_DIR/02_download.sh"
echo ""

echo "Step 3/9: Train SentencePiece tokenizer..."
bash "$SCRIPT_DIR/03_tokenize.sh"
echo ""

echo "Step 4/9: Pretrain MewtwoLLM..."
bash "$SCRIPT_DIR/04_pretrain.sh"
echo ""

echo "Step 5/9: Supervised Fine-Tuning..."
bash "$SCRIPT_DIR/05_sft.sh"
echo ""

echo "Step 6/9: DPO Alignment..."
bash "$SCRIPT_DIR/06_dpo.sh"
echo ""

echo "Step 7/9: RLHF Alignment..."
bash "$SCRIPT_DIR/07_rlhf.sh"
echo ""

echo "Step 8/9: Generate text..."
bash "$SCRIPT_DIR/08_generate.sh" "What is the meaning of life?" 200 0.8
echo ""

echo "Step 9/9: Evaluate model..."
bash "$SCRIPT_DIR/09_eval.sh"
echo ""

echo "============================================"
echo "  Pipeline complete!"
echo "============================================"
