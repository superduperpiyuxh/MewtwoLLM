#!/usr/bin/env bash
set -euo pipefail

echo "=== Step 1: Scrape training data with Scrapling ==="

SCRAPELING="/home/piyuxhh/.local/bin/scrapling"
DATA_DIR="$(dirname "$0")/../data"

mkdir -p "$DATA_DIR/raw"

# Scrape Wikipedia articles (plain text)
echo "[1/3] Scraping Wikipedia..."
python3 -c "
from src.data.scraper import WebScraper
scraper = WebScraper(output_dir='$DATA_DIR/raw')
urls = [
    'https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)',
    'https://en.wikipedia.org/wiki/Large_language_model',
    'https://en.wikipedia.org/wiki/Attention_(machine_learning)',
    'https://en.wikipedia.org/wiki/GPT-3',
    'https://en.wikipedia.org/wiki/BERT_(language_model)',
    'https://en.wikipedia.org/wiki/Fine-tuning_(deep_learning)',
    'https://en.wikipedia.org/wiki/Reinforcement_learning_from_human_feedback',
]
scraper.scrape_urls(urls)
print(f'Scraped {len(urls)} Wikipedia articles')
"

# Scrape arXiv abstracts
echo "[2/3] Scraping arXiv..."
python3 -c "
from src.data.scraper import WebScraper
scraper = WebScraper(output_dir='$DATA_DIR/raw')
urls = [
    'https://arxiv.org/abs/1706.03762',  # Attention Is All You Need
    'https://arxiv.org/abs/2005.14165',  # GPT-3
    'https://arxiv.org/abs/2107.03374',  # SwiGLU
    'https://arxiv.org/abs/2305.18290',  # LLaMA
    'https://arxiv.org/abs/2307.09288',  # LLaMA 2
]
scraper.scrape_urls(urls)
print(f'Scraped {len(urls)} arXiv pages')
"

# Scrape GitHub READMEs (for code data)
echo "[3/3] Scraping GitHub..."
python3 -c "
from src.data.scraper import WebScraper
scraper = WebScraper(output_dir='$DATA_DIR/raw')
urls = [
    'https://github.com/huggingface/transformers',
    'https://github.com/meta-llama/llama',
    'https://github.com/google-research/google-research/tree/master/t5',
]
scraper.scrape_urls(urls)
print(f'Scraped {len(urls)} GitHub pages')
"

echo "Scraping complete. Data saved to $DATA_DIR/raw/"
ls -la "$DATA_DIR/raw/"
