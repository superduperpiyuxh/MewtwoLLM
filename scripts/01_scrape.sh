#!/usr/bin/env bash
set -euo pipefail

echo "=== Step 1: Scrape training data ==="

PROJECT_DIR="$(dirname "$0")/.."
DATA_DIR="$PROJECT_DIR/data"

mkdir -p "$DATA_DIR/raw"

# Scrape using Python
python3 -c "
import sys
sys.path.insert(0, '$PROJECT_DIR')
from src.data.scraper import scrape_all

# Scrape all targets
results = scrape_all('$DATA_DIR/raw', delay=2.0)
print(f'Scraped {len(results)} pages total')
"

echo "Scraping complete. Data saved to $DATA_DIR/raw/"
ls -la "$DATA_DIR/raw/"
