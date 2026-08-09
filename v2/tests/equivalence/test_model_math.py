from __future__ import annotations

import math

import mlx.core as mx
import pytest
from sml.model.config import ModelConfig
from sml.model.rope import (
    RotaryEmbedding,
    find_correction_dimension,
    find_correction_range,
    resolve_attention_factor,
)


def assert_close(actual: mx.array, expected: mx.array, *, atol: float, rtol: float):
    mx.eval(actual, expected)
    assert bool(mx.allclose(actual, expected, atol=atol, rtol=rtol).item())


@pytest.mark.parametrize(
    ("factor", "suffix"),
    [(2.0, ""), (4.0, ".factor_4")],
)
def test_rope_caches_match_captured_reference(legacy_arrays, factor, suffix):
    """A cache formula drift would affect all rotated attention keys and queries."""
    config = ModelConfig(
        vocab_size=64,
        hidden_size=16,
        num_layers=2,
        num_q_heads=4,
        num_kv_heads=2,
        intermediate_size=32,
        original_context_length=8,
        rope_scaling_factor=factor,
        hidden_dropout=0.0,
    )

    rope = RotaryEmbedding(config)

    mx.eval(rope.cos_cached, rope.sin_cached)
    assert rope.cos_cached.dtype == mx.float32
    assert rope.sin_cached.dtype == mx.float32
    assert_close(
        rope.cos_cached,
        legacy_arrays[f"rope.cos_cache{suffix}"],
        atol=1e-6,
        rtol=1e-6,
    )
    assert_close(
        rope.sin_cached,
        legacy_arrays[f"rope.sin_cache{suffix}"],
        atol=1e-6,
        rtol=1e-6,
    )


def test_yarn_correction_math_matches_captured_factor_two_metadata(legacy_control):
    """Cutoff or attention-scale changes would select incorrect YaRN frequency bands."""
    factor_metadata = legacy_control["cases"]["rope"]["factors"]["2.0"]

    low, high = find_correction_range(32.0, 1.0, 4, 10_000.0, 8, truncate=True)

    assert (low, high) == tuple(factor_metadata["correction_range"])
    assert find_correction_dimension(32.0, 4, 10_000.0, 8) < 0.0
    assert resolve_attention_factor(2.0) == pytest.approx(1.0 + 0.1 * math.log(2.0))
