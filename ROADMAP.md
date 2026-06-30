# MewtwoLLM Roadmap

## Goal
Build a production-quality LLM that matches nanoGPT (GPT-2 124M) performance.
This is NOT a toy project. Every commit is deliberate, tested, and documented.

## Architecture Comparison

| | nanoGPT (GPT-2) | MewtwoLLM (current) | MewtwoLLM (target) |
|---|---|---|---|
| Parameters | 124M | 51.8M | 124M |
| Layers | 12 | 12 | 12 |
| Dim | 768 | 512 | 768 |
| Heads | 12 | 8 | 12 |
| KV Heads | 12 (MHA) | 4 (GQA) | 6 (GQA) |
| FFN | 3072 (GELU) | 1408 (SwiGLU) | 3072 (SwiGLU) |
| Norm | LayerNorm | RMSNorm | RMSNorm |
| Position | Learned | RoPE | RoPE |
| Bias | Yes | No | No |
| Vocab | 50257 | 32000 | 32000 |
| Context | 1024 | 1024 | 1024 |

## Key Insight
Our modern architecture (RoPE+RMSNorm+SwiGLU+GQA) should match or beat nanoGPT
with fewer parameters due to better inductive biases.

## Phase 1: Architecture Parity (match nanoGPT scale)
- [ ] Scale model to 124M params
- [ ] Verify param count matches GPT-2 124M
- [ ] Benchmark forward/backward pass speed

## Phase 2: Training Infrastructure
- [ ] Gradient accumulation with proper scaling
- [ ] Mixed precision training (bf16/fp16)
- [ ] torch.compile() integration
- [ ] Gradient checkpointing for memory efficiency
- [ ] Proper weight initialization (GPT-2 style)
- [ ] Learning rate cosine schedule with warmup

## Phase 3: Data Pipeline
- [ ] Download and preprocess OpenWebText (same as nanoGPT)
- [ ] Memory-mapped tokenized dataset
- [ ] Efficient DataLoader with no CPU bottleneck
- [ ] Data validation and quality checks

## Phase 4: Training Execution
- [ ] Train on Colab T4 GPU
- [ ] Monitor loss curves
- [ ] Compare val loss to nanoGPT baseline (~2.85)
- [ ] Save checkpoints at regular intervals

## Phase 5: Evaluation
- [ ] Perplexity on OpenWebText test set
- [ ] Zero-shot eval on LAMBADA
- [ ] Text quality samples
- [ ] Training efficiency (tokens/sec, MFU)

## Phase 6: Documentation & Paper
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
