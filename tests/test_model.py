"""
MewtwoLLM Comprehensive Test Suite

Dynamic, parametrized tests covering:
- Core model components
- Edge cases (empty input, single token, max context)
- Numerical stability
- Memory efficiency
- Integration tests
"""

import torch
import sys
import os
import math
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.model_config import MewtwoConfig
from src.model.pocketllm import MewtwoLLM
from src.model.rope import apply_rotary_pos_emb, RotaryEmbedding
from src.model.rmsnorm import RMSNorm
from src.model.swiglu import SwiGLU
from src.model.attention import GQAAttention
from src.model.transformer_block import TransformerBlock
from src.tokenizer.tokenizer import MewtwoTokenizer


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def small_config():
    """Small config for fast tests."""
    return MewtwoConfig(
        dim=64,
        n_heads=4,
        n_kv_heads=2,
        ff_dim=128,
        n_layers=2,
        vocab_size=1000,
        context_length=128,
    )


@pytest.fixture
def full_config():
    """Full-sized config."""
    return MewtwoConfig()


@pytest.fixture
def small_model(small_config):
    """Small model for fast tests."""
    return MewtwoLLM(small_config)


@pytest.fixture
def full_model(full_config):
    """Full-sized model."""
    return MewtwoLLM(full_config)


# ============================================================================
# Parametrized Model Tests
# ============================================================================

class TestModelArchitecture:
    """Test model architecture parameters."""

    @pytest.mark.parametrize("dim,n_heads,n_kv_heads", [
        (64, 4, 2),
        (128, 8, 4),
        (256, 8, 8),
        (512, 16, 8),
    ])
    def test_model_creation(self, dim, n_heads, n_kv_heads):
        """Test model creation with various architectures."""
        config = MewtwoConfig(
            dim=dim, n_heads=n_heads, n_kv_heads=n_kv_heads,
            ff_dim=dim * 4, n_layers=2, vocab_size=1000,
            context_length=64,
        )
        model = MewtwoLLM(config)
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0, "Model has no parameters"

    def test_model_init_range(self, full_model):
        """Test parameter initialization is in reasonable range."""
        for name, param in full_model.named_parameters():
            if "weight" in name:
                # RMSNorm weights are initialized to 1.0 (correct behavior)
                if "norm" in name:
                    assert torch.allclose(param, torch.ones_like(param), atol=0.1), \
                        f"{name} not initialized to 1.0: {param.mean()}"
                else:
                    # Just check no extreme values
                    assert param.mean().abs() < 1.0, f"{name} mean too large: {param.mean()}"
                    assert param.std() > 0.0001, f"{name} std too small: {param.std()}"
                    assert param.std() < 2.0, f"{name} std too large: {param.std()}"

    def test_weight_tying(self, full_model):
        """Test weight tying between token embedding and LM head."""
        assert torch.equal(
            full_model.token_embed.weight,
            full_model.lm_head.weight
        ), "Weights not tied"

    def test_no_bias(self, full_config):
        """Test no bias in linear layers."""
        config = MewtwoConfig(bias=False)
        model = MewtwoLLM(config)
        for name, param in model.named_parameters():
            if "bias" in name:
                pytest.fail(f"Found bias parameter: {name}")


# ============================================================================
# Forward Pass Tests
# ============================================================================

class TestForwardPass:
    """Test forward pass with various inputs."""

    def test_basic_forward(self, small_model, small_config):
        """Test basic forward pass."""
        x = torch.randint(0, small_config.vocab_size, (2, 32))
        logits, loss, _ = small_model(x)
        assert logits.shape == (2, 32, small_config.vocab_size)
        assert loss is None

    def test_forward_with_targets(self, small_model, small_config):
        """Test forward pass with loss computation."""
        x = torch.randint(0, small_config.vocab_size, (2, 32))
        targets = torch.randint(0, small_config.vocab_size, (2, 32))
        logits, loss, _ = small_model(x, targets=targets)
        assert loss is not None
        assert loss.item() > 0, "Loss should be positive"

    @pytest.mark.parametrize("batch_size,seq_len", [
        (1, 1),
        (1, 10),
        (4, 32),
        (8, 64),
        (16, 128),
    ])
    def test_various_shapes(self, small_model, small_config, batch_size, seq_len):
        """Test forward with various input shapes."""
        x = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))
        logits, _, _ = small_model(x)
        assert logits.shape == (batch_size, seq_len, small_config.vocab_size)


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_token(self, small_model, small_config):
        """Test with single token input."""
        x = torch.randint(0, small_config.vocab_size, (1, 1))
        logits, _, _ = small_model(x)
        assert logits.shape == (1, 1, small_config.vocab_size)

    def test_empty_sequence(self, small_model):
        """Test with empty sequence (should handle gracefully)."""
        x = torch.randint(0, 1000, (1, 0))
        try:
            logits, _, _ = small_model(x)
            # If it doesn't raise, check output shape
            assert logits.shape[1] == 0
        except (RuntimeError, IndexError):
            # Expected behavior - empty sequence not supported
            pass

    def test_max_context_length(self, small_model, small_config):
        """Test with maximum context length."""
        x = torch.randint(0, small_config.vocab_size, (1, small_config.context_length))
        logits, _, _ = small_model(x)
        assert logits.shape == (1, small_config.context_length, small_config.vocab_size)

    def test_exceeds_context_length(self, small_model, small_config):
        """Test with sequence exceeding context length."""
        x = torch.randint(0, small_config.vocab_size, (1, small_config.context_length + 10))
        # Should truncate or raise
        try:
            logits, _, _ = small_model(x)
            # If it works, output should be truncated
        except Exception:
            # Expected behavior
            pass

    def test_all_same_tokens(self, small_model, small_config):
        """Test with all same tokens."""
        x = torch.ones(1, 32, dtype=torch.long) * 42
        logits, _, _ = small_model(x)
        assert not torch.isnan(logits).any(), "NaN in output"
        assert not torch.isinf(logits).any(), "Inf in output"


# ============================================================================
# Numerical Stability Tests
# ============================================================================

class TestNumericalStability:
    """Test numerical stability."""

    def test_no_nan_output(self, small_model, small_config):
        """Test no NaN in output."""
        x = torch.randint(0, small_config.vocab_size, (2, 32))
        logits, loss, _ = small_model(x, targets=x)
        assert not torch.isnan(logits).any(), "NaN in logits"
        assert not torch.isnan(loss), "NaN in loss"

    def test_no_inf_output(self, small_model, small_config):
        """Test no Inf in output."""
        x = torch.randint(0, small_config.vocab_size, (2, 32))
        logits, loss, _ = small_model(x, targets=x)
        assert not torch.isinf(logits).any(), "Inf in logits"
        assert not torch.isinf(loss), "Inf in loss"

    def test_loss_finite(self, small_model, small_config):
        """Test loss is finite."""
        x = torch.randint(0, small_config.vocab_size, (2, 32))
        targets = torch.randint(0, small_config.vocab_size, (2, 32))
        _, loss, _ = small_model(x, targets=targets)
        assert math.isfinite(loss.item()), f"Loss not finite: {loss.item()}"

    def test_gradient_norm(self, small_model, small_config):
        """Test gradient norms are reasonable."""
        small_model.train()
        x = torch.randint(0, small_config.vocab_size, (2, 32))
        targets = torch.randint(0, small_config.vocab_size, (2, 32))
        _, loss, _ = small_model(x, targets=targets)
        loss.backward()

        for name, param in small_model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                assert grad_norm < 100, f"{name} gradient too large: {grad_norm}"
                assert not math.isnan(grad_norm), f"{name} gradient is NaN"
                assert not math.isinf(grad_norm), f"{name} gradient is Inf"


# ============================================================================
# Component Tests
# ============================================================================

class TestComponents:
    """Test individual model components."""

    def test_rope_shapes(self, small_config):
        """Test RoPE output shapes."""
        rope = RotaryEmbedding(small_config.head_dim)
        x = torch.randn(2, small_config.n_heads, 32, small_config.head_dim)
        cos, sin = rope(32)
        q_rot, k_rot = apply_rotary_pos_emb(x, x.clone(), cos, sin)
        assert q_rot.shape == x.shape
        assert k_rot.shape == x.shape

    def test_rmsnorm_normalization(self, small_config):
        """Test RMSNorm normalizes correctly."""
        norm = RMSNorm(small_config.dim)
        x = torch.randn(2, 32, small_config.dim)
        out = norm(x)
        rms = out.pow(2).mean(dim=-1, keepdim=True).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5)

    def test_swiglu_shape(self, small_config):
        """Test SwiGLU output shape."""
        ffn = SwiGLU(small_config.dim, small_config.ff_dim)
        x = torch.randn(2, 32, small_config.dim)
        out = ffn(x)
        assert out.shape == x.shape

    def test_attention_output_shape(self, small_config):
        """Test attention output shape."""
        attn = GQAAttention(
            small_config.dim,
            small_config.n_heads,
            small_config.n_kv_heads,
        )
        x = torch.randn(2, 32, small_config.dim)
        out, _ = attn(x)
        assert out.shape == x.shape

    def test_transformer_block_shape(self, small_config):
        """Test transformer block output shape."""
        block = TransformerBlock(
            small_config.dim,
            small_config.n_heads,
            small_config.n_kv_heads,
            small_config.ff_dim,
        )
        x = torch.randn(2, 32, small_config.dim)
        mask = torch.tril(torch.ones(32, 32)).unsqueeze(0).unsqueeze(0)
        out, _ = block(x, mask=mask)
        assert out.shape == x.shape


# ============================================================================
# Generation Tests
# ============================================================================

class TestGeneration:
    """Test text generation."""

    def test_basic_generation(self, small_model, small_config):
        """Test basic generation."""
        x = torch.randint(0, small_config.vocab_size, (1, 10))
        with torch.no_grad():
            generated = small_model.generate(x, max_new_tokens=20)
        assert generated.shape[0] == 1
        assert generated.shape[1] == 30  # 10 + 20

    @pytest.mark.parametrize("temperature,top_k", [
        (0.0, 1),
        (0.5, 10),
        (1.0, 50),
        (2.0, 100),
    ])
    def test_generation_params(self, small_model, small_config, temperature, top_k):
        """Test generation with various parameters."""
        x = torch.randint(0, small_config.vocab_size, (1, 10))
        with torch.no_grad():
            generated = small_model.generate(
                x, max_new_tokens=10,
                temperature=temperature, top_k=top_k,
            )
        assert generated.shape[0] == 1

    def test_generation_deterministic(self, small_model, small_config):
        """Test generation is deterministic with temperature=0."""
        x = torch.randint(0, small_config.vocab_size, (1, 10))
        with torch.no_grad():
            gen1 = small_model.generate(x, max_new_tokens=10, temperature=0.0)
            gen2 = small_model.generate(x, max_new_tokens=10, temperature=0.0)
        assert torch.equal(gen1, gen2), "Generation not deterministic"


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Test full pipeline integration."""

    def test_train_eval_mode(self, small_model, small_config):
        """Test switching between train and eval modes."""
        small_model.train()
        assert small_model.training

        x = torch.randint(0, small_config.vocab_size, (2, 32))
        _, loss1, _ = small_model(x, targets=x)
        loss1.backward()

        small_model.eval()
        assert not small_model.training

        with torch.no_grad():
            _, loss2, _ = small_model(x, targets=x)
        # Loss should be same in eval mode (no dropout)
        assert abs(loss1.item() - loss2.item()) < 0.1

    def test_checkpoint_save_load(self, small_model, small_config, tmp_path):
        """Test model checkpoint save/load."""
        # Save
        checkpoint = {
            "config": small_config,
            "model_state_dict": small_model.state_dict(),
        }
        path = tmp_path / "model.pt"
        torch.save(checkpoint, path)

        # Load
        loaded = torch.load(path, weights_only=False)
        new_model = MewtwoLLM(loaded["config"])
        new_model.load_state_dict(loaded["model_state_dict"])

        # Verify
        for (n1, p1), (n2, p2) in zip(
            small_model.named_parameters(),
            new_model.named_parameters(),
        ):
            assert torch.equal(p1, p2), f"Parameter mismatch: {n1}"

    def test_gradient_checkpointing(self, small_config):
        """Test gradient checkpointing works."""
        config = MewtwoConfig(
            dim=64, n_heads=4, n_kv_heads=2, ff_dim=128,
            n_layers=2, vocab_size=1000, context_length=64,
        )
        model = MewtwoLLM(config, use_gradient_checkpointing=True)
        model.train()

        x = torch.randint(0, config.vocab_size, (2, 32))
        targets = torch.randint(0, config.vocab_size, (2, 32))
        _, loss, _ = model(x, targets=targets)
        loss.backward()

        # Check gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


# ============================================================================
# Tokenizer Tests
# ============================================================================

class TestTokenizer:
    """Test tokenizer functionality."""

    @pytest.mark.skip(reason="Tokenizer not trained in test environment")
    def test_encode_decode(self):
        """Test encode/decode roundtrip."""
        tokenizer = MewtwoTokenizer(vocab_size=1000)
        text = "Hello, world!"
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        assert len(tokens) > 0
        assert isinstance(decoded, str)

    @pytest.mark.skip(reason="Tokenizer not trained in test environment")
    def test_encode_empty(self):
        """Test encoding empty string."""
        tokenizer = MewtwoTokenizer(vocab_size=1000)
        tokens = tokenizer.encode("")
        assert tokens == []

    @pytest.mark.skip(reason="Tokenizer not trained in test environment")
    def test_encode_special_chars(self):
        """Test encoding special characters."""
        tokenizer = MewtwoTokenizer(vocab_size=1000)
        text = "Hello! @#$%^&*()"
        tokens = tokenizer.encode(text)
        assert len(tokens) > 0

    def test_vocab_size(self):
        """Test vocab size is correct."""
        tokenizer = MewtwoTokenizer(vocab_size=32000)
        assert tokenizer.vocab_size == 32000


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Test performance characteristics."""

    def test_forward_speed(self, small_model, small_config):
        """Test forward pass speed."""
        import time
        x = torch.randint(0, small_config.vocab_size, (1, 32))

        # Warmup
        for _ in range(3):
            with torch.no_grad():
                small_model(x)

        # Benchmark
        start = time.time()
        for _ in range(10):
            with torch.no_grad():
                small_model(x)
        elapsed = time.time() - start

        # Should be fast (< 1 second for 10 forward passes)
        assert elapsed < 1.0, f"Forward pass too slow: {elapsed:.2f}s"

    def test_memory_usage(self, small_model, small_config):
        """Test memory usage is reasonable."""
        import sys
        x = torch.randint(0, small_config.vocab_size, (1, 32))

        # Estimate model size
        param_size = sum(p.nelement() * p.element_size() for p in small_model.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in small_model.buffers())
        total_size = param_size + buffer_size

        # Should be reasonable (< 100MB for small model)
        assert total_size < 100 * 1024 * 1024, f"Model too large: {total_size / 1e6:.1f}MB"


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
