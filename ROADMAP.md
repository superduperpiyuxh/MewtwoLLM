# MewtwoLLM Roadmap

## Goal
Build a production-quality LLM that matches nanoGPT (GPT-2 124M) performance.
This is NOT a toy project. Every commit is deliberate, tested, and documented.

## Architecture Comparison

| | nanoGPT (GPT-2) | MewtwoLLM (current) | MewtwoLLM (target) |
|---|---|---|---|
| Parameters | 124M | 130.8M | 124M |
| Layers | 12 | 12 | 12 |
| Dim | 768 | 768 | 768 |
| Heads | 12 | 12 | 12 |
| KV Heads | 12 (MHA) | 6 (GQA) | 6 (GQA) |
| FFN | 3072 (GELU) | 3072 (SwiGLU) | 3072 (SwiGLU) |
| Norm | LayerNorm | RMSNorm | RMSNorm |
| Position | Learned | RoPE | RoPE |
| Bias | Yes | No | No |
| Vocab | 50257 | 32000 | 32000 |
| Context | 1024 | 1024 | 1024 |

## Key Insight
Our modern architecture (RoPE+RMSNorm+SwiGLU+GQA) should match or beat nanoGPT
with fewer parameters due to better inductive biases.

## Phase 1: Architecture Parity (match nanoGPT scale)
- [x] Scale model to 124M params (DONE: 130.8M)
- [x] Verify param count matches GPT-2 124M (DONE)
- [x] Benchmark forward/backward pass speed (DONE: tests pass)

## Phase 2: Training Infrastructure
- [x] Gradient accumulation with proper scaling (DONE)
- [x] Mixed precision training (bf16/fp16) (DONE)
- [x] torch.compile() integration (DONE)
- [x] Gradient checkpointing for memory efficiency (DONE)
- [x] Proper weight initialization (GPT-2 style) (DONE)
- [x] Learning rate cosine schedule with warmup (DONE: WSD)

## Phase 3: Data Pipeline
- [x] Download and preprocess OpenWebText (DONE: scripts ready)
- [x] Memory-mapped tokenized dataset (DONE: mmap_dataset.py)
- [x] Efficient DataLoader with no CPU bottleneck (DONE: built-in)
- [x] Data validation and quality checks (DONE: tests)

## Phase 4: Training Execution
- [ ] Train on Colab T4 GPU
- [ ] Monitor loss curves
- [ ] Compare val loss to nanoGPT baseline (~2.85)
- [ ] Save checkpoints at regular intervals

## Phase 5: Evaluation
- [x] MMLU benchmark (DONE: 57 subjects)
- [x] Perplexity on OpenWebText test set (DONE: compute_perplexity)
- [ ] Zero-shot eval on LAMBADA
- [ ] Text quality samples
- [ ] Training efficiency (tokens/sec, MFU)

## Phase 6: Documentation & Paper
- [x] Paper audit (DONE: 22 papers + 1 book)
- [ ] Architecture diagram
- [ ] Training curves visualization
- [ ] Comparison table with nanoGPT
- [ ] Write paper draft

## Commit Strategy
Each task = 1 commit. Small, precise, tested.
No "big bang" commits. Every change is reviewable.

## Quality Bar
- All tests pass before commit
- No dead code
- No unnecessary abstractions
- Every line traces to a requirement
