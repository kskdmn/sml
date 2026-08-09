from __future__ import annotations

import mlx.core as mx
import pytest
from sml.model.config import ModelConfig
from sml.model.rope import RotaryEmbedding, apply_rotary, rotate_half


@pytest.fixture
def tiny_model_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=64,
        hidden_size=16,
        num_layers=2,
        num_q_heads=4,
        num_kv_heads=2,
        intermediate_size=32,
        original_context_length=8,
        rope_scaling_factor=2.0,
        hidden_dropout=0.0,
    )


def assert_close(actual: mx.array, expected: mx.array, *, atol: float, rtol: float):
    mx.eval(actual, expected)
    assert bool(mx.allclose(actual, expected, atol=atol, rtol=rtol).item())


def test_rope_matches_captured_reference(legacy_arrays, tiny_model_config):
    """Incorrect YaRN bands or positions would diverge from frozen legacy Q/K."""
    rope = RotaryEmbedding(tiny_model_config)

    actual_q, actual_k = rope(
        legacy_arrays["rope.q"],
        legacy_arrays["rope.k"],
        legacy_arrays["rope.positions"],
    )

    mx.eval(actual_q, actual_k)
    assert actual_q.dtype == mx.bfloat16
    assert actual_k.dtype == mx.bfloat16
    assert_close(actual_q, legacy_arrays["rope.output_q"], atol=2e-2, rtol=2e-2)
    assert_close(actual_k, legacy_arrays["rope.output_k"], atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("position", [-1, 16])
def test_rope_fails_closed_for_out_of_bounds_positions(tiny_model_config, position):
    """Wrapped cache indexing must never yield a plausible but wrong rotation."""
    rope = RotaryEmbedding(tiny_model_config)
    q = mx.ones((1, 4, 1, 4), dtype=mx.bfloat16)
    k = mx.ones((1, 2, 1, 4), dtype=mx.bfloat16)

    actual_q, actual_k = rope(q, k, mx.array([position], dtype=mx.int32))

    mx.eval(actual_q, actual_k)
    assert bool(mx.all(mx.isnan(actual_q)).item())
    assert bool(mx.all(mx.isnan(actual_k)).item())


def test_apply_rotary_returns_bfloat16_after_fp32_cache_multiplication():
    """A widened result would change downstream attention precision and memory use."""
    q = mx.array([[[[1.0, 2.0, 3.0, 4.0]]]], dtype=mx.bfloat16)
    k = mx.array([[[[4.0, 3.0, 2.0, 1.0]]]], dtype=mx.bfloat16)
    cos = mx.ones((1, 4), dtype=mx.float32)
    sin = mx.zeros((1, 4), dtype=mx.float32)

    actual_q, actual_k = apply_rotary(q, k, cos, sin)

    mx.eval(actual_q, actual_k)
    assert actual_q.dtype == mx.bfloat16
    assert actual_k.dtype == mx.bfloat16
    assert_close(actual_q, q, atol=0.0, rtol=0.0)
    assert_close(actual_k, k, atol=0.0, rtol=0.0)


def test_rotate_half_swaps_and_negates_the_second_half():
    """A wrong half ordering rotates every RoPE feature in the wrong direction."""
    x = mx.array([[1.0, 2.0, 3.0, 4.0]], dtype=mx.float32)

    actual = rotate_half(x)

    assert_close(
        actual,
        mx.array([[-3.0, -4.0, 1.0, 2.0]], dtype=mx.float32),
        atol=0.0,
        rtol=0.0,
    )
