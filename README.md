# MewtwoLLM

> *"I see now that the circumstances of one's birth are irrelevant; it is what you do with the gift of life that determines who you are."* — Mewtwo

**MewtwoLLM** is a language model built entirely from scratch — no `transformers`, no `peft`, no `trl`. Every component is hand-written in pure PyTorch.

Like Mewtwo itself, this model is **engineered intelligence**. Every architectural choice comes from a research paper, every training trick is deliberate, and every line of code exists for a reason.

## Architecture

| Component | Technology | Paper |
|-----------|------------|-------|
| Position Encoding | **RoPE** (Rotary Position Embedding) | Su et al. 2021 |
| Normalization | **RMSNorm** (pre-normalization) | Zhang & Sennrich 2019 |
| FFN Activation | **SwiGLU** | Shazeer 2020 |
| Attention | **GQA** (Grouped Query Attention) | Ainslie et al. 2023 |
| Tokenizer | **SentencePiece BPE** (32K vocab) | Kudo & Richardson 2018 |
| Scaling | **Chinchilla-optimal** training | Hoffmann et al. 2022 |

## Model Specs

| Parameter | Value |
|-----------|-------|
| Parameters | ~130M |
| Layers | 12 |
| Hidden Dim | 768 |
| Attention Heads | 12 |
| KV Groups | 6 (GQA) |
| FFN Hidden Dim | 3072 |
| Context Length | 1024 |
| Vocab Size | 32,000 |

## Project Structure

```
MewtwoLLM/
├── config/              # Hyperparameters
├── src/
│   ├── tokenizer/       # SentencePiece BPE
│   ├── model/           # RoPE, RMSNorm, SwiGLU, GQA, Transformer
│   ├── data/            # Scraping (Scrapling) + preprocessing
│   ├── training/        # Pretraining loop, optimizer, scheduler
│   ├── alignment/       # SFT, DPO, RLHF
│   ├── inference/       # Text generation with KV cache
│   └── evaluation/      # Perplexity + benchmarks
├── data/                # Raw, processed, alignment data
├── scripts/             # Shell scripts for each pipeline stage
└── tests/               # Unit tests
```

## Pipeline

```
1. Data Collection    → Scrapling web scraping + FineWeb/RedPajama downloads
2. Tokenization       → SentencePiece BPE (32K vocab)
3. Pretraining        → Next-token prediction on 10B+ tokens
4. Supervised FT      → Instruction following (Alpaca-style)
5. Alignment          → DPO + RLHF (PPO)
6. Evaluation         → lm-evaluation-harness benchmarks
7. Inference          → Text generation with KV cache
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Scrape training data
python src/data/scraper.py

# Tokenize data
python src/tokenizer/tokenizer.py

# Train the model
python src/training/pretrain.py

# Fine-tune (SFT)
python src/alignment/sft.py

# Align (DPO)
python src/alignment/dpo.py

# Align (RLHF)
python src/alignment/rlhf.py

# Generate text
python src/inference/generate.py --prompt "The meaning of life is"

# Evaluate
python src/evaluation/eval.py
```

## Papers Referenced

This project implements techniques from:

1. Vaswani et al. 2017 — Attention Is All You Need
2. Sennrich et al. 2016 — BPE for NMT
3. Kudo & Richardson 2018 — SentencePiece
4. Hoffmann et al. 2022 — Chinchilla Scaling Laws
5. Su et al. 2021 — RoFormer (RoPE)
6. Shazeer 2020 — GLU Variants (SwiGLU)
7. Zhang & Sennrich 2019 — RMSNorm
8. Dao et al. 2022 — FlashAttention
9. Ainslie et al. 2023 — GQA
10. Ouyang et al. 2022 — InstructGPT (RLHF)
11. Rafailov et al. 2023 — DPO
12. Hu et al. 2021 — LoRA
13. Hendrycks et al. 2020 — MMLU

## License

MIT

## Acknowledgments

Built with guidance from:
- Sebastian Raschka's *Build a Large Language Model From Scratch*
- Andrej Karpathy's *nanoGPT*
- Every researcher who open-sourced their work
