from __future__ import annotations

import inspect

import mlx.core as mx
from sml.model.layers import GroupedQueryAttention, RMSNorm, _linear, keyed_dropout


def _assert_close(
    actual: mx.array,
    expected: mx.array,
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> None:
    mx.eval(actual, expected)
    assert bool(mx.allclose(actual, expected, atol=atol, rtol=rtol).item())


def test_rms_norm_reduces_in_float32_and_returns_bfloat16():
    layer = RMSNorm(hidden_size=4, epsilon=1e-6)
    layer.weight = mx.array([0.5, 1.0, 1.5, 2.0], dtype=mx.bfloat16)
    inputs = mx.array(
        [[[300.0, 0.125, -96.0, 7.5]]],
        dtype=mx.bfloat16,
    )

    actual = layer(inputs)
    inputs_fp32 = inputs.astype(mx.float32)
    expected = (
        inputs_fp32
        * mx.rsqrt(mx.mean(mx.square(inputs_fp32), axis=-1, keepdims=True) + 1e-6)
        * layer.weight.astype(mx.float32)
    ).astype(mx.bfloat16)

    assert actual.dtype == mx.bfloat16
    _assert_close(actual, expected)


def test_keyed_dropout_zero_probability_consumes_no_key():
    inputs = mx.arange(8, dtype=mx.float32).reshape((2, 4))
    key = mx.random.key(123)

    actual, next_key = keyed_dropout(inputs, 0.0, key)

    assert actual is inputs
    _assert_close(next_key, key)


def test_keyed_dropout_replays_from_the_same_explicit_key():
    inputs = mx.ones((32, 32), dtype=mx.bfloat16)
    key = mx.random.key(456)

    first, first_next_key = keyed_dropout(inputs, 0.5, key)
    replay, replay_next_key = keyed_dropout(inputs, 0.5, key)

    _assert_close(first, replay)
    _assert_close(first_next_key, replay_next_key)
    assert first.dtype == mx.bfloat16
    mx.eval(first_next_key, key)
    assert not bool(mx.array_equal(first_next_key, key).item())


def test_grouped_query_attention_uses_fused_gqa_without_tiling_kv_heads():
    source = inspect.getsource(GroupedQueryAttention.forward_arrays)

    assert "mx.fast.scaled_dot_product_attention" in source
    assert "mx.tile" not in source
    assert "mx.repeat" not in source


def test_linear_applies_live_lora_formula_when_projection_is_wrapped():
    x = mx.ones((1, 2, 4), dtype=mx.bfloat16)
    base_weight = mx.arange(12, dtype=mx.bfloat16).reshape((3, 4))
    lora_a = mx.arange(8, dtype=mx.float32).reshape((2, 4))
    lora_b = mx.arange(6, dtype=mx.float32).reshape((3, 2))
    scale = mx.array(0.5, dtype=mx.float32)

    actual = _linear(
        x,
        {
            "base": {"weight": base_weight},
            "lora_a": lora_a,
            "lora_b": lora_b,
            "scale": scale,
        },
    )
    adapter = scale.astype(mx.float32) * ((x.astype(mx.float32) @ lora_a.T) @ lora_b.T)
    expected = ((x @ base_weight.T) + adapter.astype(mx.bfloat16)).astype(mx.bfloat16)

    assert actual.dtype == mx.bfloat16
    _assert_close(actual, expected)
