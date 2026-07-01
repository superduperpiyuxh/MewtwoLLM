"""
Web Scraper for MewtwoLLM Training Data
Uses Scrapling to scrape diverse, high-quality text from the web.

Scraping targets:
- Wikipedia: High-quality encyclopedic knowledge
- arXiv: Scientific/technical papers
- Project Gutenberg: Public domain books
- StackExchange: Technical Q&A
- News sites: Diverse vocabulary
"""

import os
import json
import time

# Scrapling imports
from scrapling.fetchers import Fetcher


# Target URLs for scraping
SCRAPE_TARGETS = {
    "wikipedia": [
        "https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)",
        "https://en.wikipedia.org/wiki/Large_language_model",
        "https://en.wikipedia.org/wiki/Attention_(machine_learning)",
        "https://en.wikipedia.org/wiki/GPT-3",
        "https://en.wikipedia.org/wiki/BERT_(language_model)",
        "https://en.wikipedia.org/wiki/Natural_language_processing",
        "https://en.wikipedia.org/wiki/Machine_learning",
        "https://en.wikipedia.org/wiki/Deep_learning",
        "https://en.wikipedia.org/wiki/Neural_network_(machine_learning)",
        "https://en.wikipedia.org/wiki/Recurrent_neural_network",
        "https://en.wikipedia.org/wiki/Convolutional_neural_network",
        "https://en.wikipedia.org/wiki/Generative_artificial_intelligence",
        "https://en.wikipedia.org/wiki/Reinforcement_learning",
        "https://en.wikipedia.org/wiki/Supervised_learning",
        "https://en.wikipedia.org/wiki/Unsupervised_learning",
    ],
    "arxiv": [
        "https://arxiv.org/abs/1706.03762",  # Attention Is All You Need
        "https://arxiv.org/abs/2005.14165",  # GPT-3
        "https://arxiv.org/abs/1810.04805",  # BERT
        "https://arxiv.org/abs/2002.05202",  # GLU Variants
        "https://arxiv.org/abs/1910.07467",  # RMSNorm
        "https://arxiv.org/abs/2104.09864",  # RoFormer
        "https://arxiv.org/abs/2205.14135",  # FlashAttention
        "https://arxiv.org/abs/2305.13245",  # GQA
        "https://arxiv.org/abs/2203.02155",  # InstructGPT
        "https://arxiv.org/abs/2305.18290",  # DPO
        "https://arxiv.org/abs/2106.09685",  # LoRA
        "https://arxiv.org/abs/2203.15556",  # Chinchilla
        "https://arxiv.org/abs/2001.08361",  # Scaling Laws
    ],
}


def scrape_url(url: str, output_dir: str, source: str) -> dict | None:
    """
    Scrape a single URL using Scrapling.

    Returns:
        dict with 'url', 'text', 'length' keys, or None on failure.
    """
    try:
        fetcher = Fetcher(auto_match=False)
        page = fetcher.get(url)

        # Extract main text content
        # Try common content selectors
        content = None
        for selector in [
            "article",
            ".mw-parser-output",  # Wikipedia
            ".ltx_document",      # arXiv
            "main",
            "#content",
            "body",
        ]:
            element = page.css_first(selector)
            if element:
                content = element.get_all_text(ignore_tags=("script", "style", "nav", "footer"))
                break

        if not content:
            content = page.get_all_text(ignore_tags=("script", "style", "nav", "footer", "header"))

        # Clean text
        lines = [line.strip() for line in content.split("\n") if line.strip()]
        text = "\n".join(lines)

        # Skip very short content
        if len(text) < 200:
            print(f"  Skipping {url}: too short ({len(text)} chars)")
            return None

        # Save to file
        filename = url.replace("https://", "").replace("http://", "").replace("/", "_")
        filename = filename[:100]  # Truncate long filenames
        output_path = os.path.join(output_dir, f"{filename}.txt")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)

        result = {
            "url": url,
            "source": source,
            "text_length": len(text),
            "output_path": output_path,
        }

        print(f"  Scraped {url}: {len(text)} chars -> {output_path}")
        return result

    except Exception as e:
        print(f"  Error scraping {url}: {e}")
        return None


def scrape_all(output_dir: str = "data/raw/scraped", delay: float = 2.0):
    """
    Scrape all target URLs using Scrapling.

    Args:
        output_dir: directory to save scraped text
        delay: delay between requests (be polite)
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []

    for source, urls in SCRAPE_TARGETS.items():
        print(f"\nScraping {source} ({len(urls)} URLs)...")
        source_dir = os.path.join(output_dir, source)
        os.makedirs(source_dir, exist_ok=True)

        for url in urls:
            result = scrape_url(url, source_dir, source)
            if result:
                results.append(result)
            time.sleep(delay)  # Be polite to servers

    # Save metadata
    metadata_path = os.path.join(output_dir, "scrape_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nScraping complete: {len(results)} pages scraped")
    print(f"Metadata saved to {metadata_path}")
    return results


def combine_scraped_text(scraped_dir: str, output_file: str = "data/raw/scraped_text.txt"):
    """Combine all scraped text files into a single file for tokenizer training."""
    all_text = []

    for root, dirs, files in os.walk(scraped_dir):
        for fname in sorted(files):
            if fname.endswith(".txt"):
                fpath = os.path.join(root, fname)
                with open(fpath, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                    if len(text) > 100:
                        all_text.append(text)

    combined = "\n\n".join(all_text)

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(combined)

    print(f"Combined {len(all_text)} files -> {output_file} ({len(combined)} chars)")
    return output_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scrape training data for MewtwoLLM")
    parser.add_argument("--output_dir", default="data/raw/scraped", help="Output directory")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay between requests")
    parser.add_argument("--combine", action="store_true", help="Combine scraped text")
    args = parser.parse_args()

    if args.combine:
        combine_scraped_text(args.output_dir)
    else:
        scrape_all(args.output_dir, args.delay)
