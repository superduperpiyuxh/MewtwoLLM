# MewtwoLLM Daily Work Log

## How This Works
Each day, I work on ONE small thing:
1. **Build** — implement the feature/fix
2. **Test** — add test cases + edge cases
3. **Research** — if stuck, use web/scrapling to find solutions
4. **Fix** — apply what I learned
5. **Verify** — all tests pass
6. **Commit** — small, precise commit with clear message

## Day 1 — June 30, 2026

### Goal: Fix tokenizer compatibility across all modules

### Problem
The tokenizer has two interfaces:
- `MewtwoTokenizer` class (used by training scripts, Colab, tests)
- `load_tokenizer()` returning raw SentencePiece (used by alignment, inference)

This causes import errors and inconsistent usage.

### Work Done
- Created `MewtwoTokenizer` class wrapper
- Fixed `ff_dim` divisibility (1376 → 1408)
- Fixed `generate()` KV cache mask
- Extracted `_sample()` helper
- Flattened nesting in training loops
- Extracted logging helpers (_log_step, _log_sft, _log_dpo, _log_rm, _update_config)

### Tests Added
- `test_model_init` — param count validation
- `test_forward_pass` — shape verification
- `test_forward_with_targets` — loss computation
- `test_generate` — autoregressive generation
- `test_rope` — rotary embedding correctness
- `test_rmsnorm` — normalization verification
- `test_swiglu` — FFN output shape
- `test_weight_tying` — embedding/head sharing
- `test_param_count` — 51.8M param verification

### Results
- All 9 tests passing
- All files pass complexity checker (100/100)
- 9 commits pushed to GitHub

---

## Day 2 — [NEXT]

### Planned: Scale model to 124M params (match nanoGPT)
- Change dim from 512 → 768
- Change n_heads from 8 → 12
- Change n_kv_heads from 4 → 6
- Change ff_dim to match SwiGLU formula
- Verify param count matches GPT-2 124M
- Add test for new config
- Research GPT-2 initialization
