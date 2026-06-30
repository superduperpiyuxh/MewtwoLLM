import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.model_config import MewtwoConfig
from src.model.pocketllm import MewtwoLLM
from src.tokenizer.tokenizer import MewtwoTokenizer


def test_model_init():
    config = MewtwoConfig()
    model = MewtwoLLM(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model initialized: {n_params:,} parameters')
    assert n_params > 1_000_000, f'Too few parameters: {n_params}'
    assert n_params < 200_000_000, f'Too many parameters: {n_params}'
    print('  PASS: test_model_init')


def test_forward_pass():
    config = MewtwoConfig()
    model = MewtwoLLM(config)
    model.eval()
    x = torch.randint(0, config.vocab_size, (2, 128))
    with torch.no_grad():
        logits, loss = model(x)
    assert logits.shape == (2, 128, config.vocab_size), f'Wrong logits shape: {logits.shape}'
    assert loss is None or loss == 0.0, f'Unexpected loss on unlabelled input: {loss}'
    print('  PASS: test_forward_pass')


def test_forward_with_targets():
    config = MewtwoConfig()
    model = MewtwoLLM(config)
    model.train()
    x = torch.randint(0, config.vocab_size, (2, 128))
    targets = torch.randint(0, config.vocab_size, (2, 128))
    logits, loss = model(x, targets)
    assert logits.shape == (2, 128, config.vocab_size)
    assert loss is not None and loss.item() > 0, f'Loss should be positive: {loss}'
    print('  PASS: test_forward_with_targets')


def test_generate():
    config = MewtwoConfig()
    model = MewtwoLLM(config)
    model.eval()
    x = torch.randint(0, config.vocab_size, (1, 10))
    with torch.no_grad():
        generated = model.generate(x, max_new_tokens=20, temperature=1.0, top_k=50)
    assert generated.shape[0] == 1
    assert generated.shape[1] == 30, f'Expected 30 tokens, got {generated.shape[1]}'
    print('  PASS: test_generate')


def test_rope():
    config = MewtwoConfig()
    model = MewtwoLLM(config)
    rope = model.blocks[0].attention.rope
    q = torch.randn(1, 8, 64, 64)
    k = torch.randn(1, 8, 64, 64)
    q_rot, k_rot = rope(q, k)
    assert q_rot.shape == q.shape, f'RoPE changed shape: {q_rot.shape}'
    assert k_rot.shape == k.shape
    print('  PASS: test_rope')


def test_rmsnorm():
    config = MewtwoConfig()
    model = MewtwoLLM(config)
    norm = model.norm
    x = torch.randn(2, 128, config.d_model)
    out = norm(x)
    assert out.shape == x.shape
    rms = out.pow(2).mean(dim=-1, keepdim=True).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5), f'Not normalized: {rms}'
    print('  PASS: test_rmsnorm')


def test_swiglu():
    config = MewtwoConfig()
    model = MewtwoLLM(config)
    swiglu = model.blocks[0].ffn
    x = torch.randn(2, 128, config.d_model)
    out = swiglu(x)
    assert out.shape == x.shape, f'SwiGLU wrong shape: {out.shape}'
    print('  PASS: test_swiglu')


def test_weight_tying():
    config = MewtwoConfig()
    model = MewtwoLLM(config)
    tok_emb = model.tok_emb.weight
    lm_head = model.lm_head.weight
    assert torch.equal(tok_emb, lm_head), 'Token embedding and LM head should share weights'
    print('  PASS: test_weight_tying')


def test_param_count():
    config = MewtwoConfig()
    model = MewtwoLLM(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Model has {n_params:,} parameters ({n_params/1e6:.1f}M)')
    print('  PASS: test_param_count')


if __name__ == '__main__':
    print('=' * 60)
    print('MewtwoLLM Test Suite')
    print('=' * 60)

    test_model_init()
    test_forward_pass()
    test_forward_with_targets()
    test_generate()
    test_rope()
    test_rmsnorm()
    test_swiglu()
    test_weight_tying()
    test_param_count()

    print()
    print('=' * 60)
    print('ALL TESTS PASSED')
    print('=' * 60)
