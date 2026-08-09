from __future__ import annotations

from dataclasses import replace

import mlx.core as mx
from sml.model.cache import KVCache, allocate_kv_state, append_kv_state
from sml.model.config import ModelConfig


def _cache_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=16,
        hidden_size=8,
        num_layers=2,
        num_q_heads=2,
        num_kv_heads=1,
        intermediate_size=16,
        original_context_length=8,
        hidden_dropout=0.0,
    )


def _assert_array_equal(actual: mx.array, expected: mx.array) -> None:
    mx.eval(actual, expected)
    assert bool(mx.array_equal(actual, expected).item())


def test_allocate_kv_state_has_fixed_bfloat16_capacity_and_int32_lengths():
    config = _cache_config()

    keys, values, lengths = allocate_kv_state(
        config,
        batch_size=2,
        capacity=5,
        dtype=mx.bfloat16,
    )

    assert isinstance(keys, tuple)
    assert isinstance(values, tuple)
    assert len(keys) == len(values) == config.num_layers
    assert all(array.shape == (2, 1, 5, 4) for array in (*keys, *values))
    assert all(array.dtype == mx.bfloat16 for array in (*keys, *values))
    assert lengths.shape == (2,)
    assert lengths.dtype == mx.int32
    _assert_array_equal(lengths, mx.zeros((2,), dtype=mx.int32))


def test_append_kv_state_is_functional_and_writes_absolute_valid_positions():
    config = _cache_config()
    original = allocate_kv_state(config, 2, 4, mx.bfloat16)
    keys = mx.array(
        [
            [[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]]],
            [[[[9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]]],
        ],
        dtype=mx.bfloat16,
    ).reshape((2, 1, 2, 4))
    values = keys + mx.array(20.0, dtype=mx.bfloat16)
    positions = mx.array([[0, 1], [0, 3]], dtype=mx.int32)
    valid_mask = mx.array([[True, True], [True, False]])

    updated, view = append_kv_state(
        original,
        layer_index=0,
        keys=keys,
        values=values,
        positions=positions,
        valid_mask=valid_mask,
    )

    _assert_array_equal(original[2], mx.zeros((2,), dtype=mx.int32))
    _assert_array_equal(original[0][0], mx.zeros_like(original[0][0]))
    _assert_array_equal(updated[2], mx.array([2, 1], dtype=mx.int32))
    _assert_array_equal(
        view.valid_mask,
        mx.array([[True, True, False, False], [True, False, False, False]]),
    )
    _assert_array_equal(view.keys[0, 0, :2], keys[0, 0])
    _assert_array_equal(view.values[0, 0, :2], values[0, 0])
    _assert_array_equal(view.keys[1, 0, 0], keys[1, 0, 0])
    _assert_array_equal(view.keys[1, 0, 1:], mx.zeros((3, 4), dtype=mx.bfloat16))


def test_invalid_duplicate_position_cannot_overwrite_a_valid_cache_write():
    config = replace(_cache_config(), num_layers=1)
    original = allocate_kv_state(config, 1, 2, mx.bfloat16)
    keys = mx.array(
        [[[[1.0, 2.0, 3.0, 4.0], [9.0, 9.0, 9.0, 9.0]]]],
        dtype=mx.bfloat16,
    )
    values = keys + mx.array(10.0, dtype=mx.bfloat16)

    updated, _view = append_kv_state(
        original,
        0,
        keys,
        values,
        mx.array([[0, 0]], dtype=mx.int32),
        mx.array([[True, False]]),
    )

    _assert_array_equal(updated[0][0][0, 0, 0], keys[0, 0, 0])
    _assert_array_equal(updated[1][0][0, 0, 0], values[0, 0, 0])
    _assert_array_equal(updated[2], mx.array([1], dtype=mx.int32))


def test_compiled_cache_core_carries_returned_lengths_and_payload_forward():
    config = replace(_cache_config(), num_layers=1)
    cache = KVCache.allocate(config, 1, 4, mx.bfloat16)

    @mx.compile
    def append_core(state, keys, values, positions, valid_mask):
        updated, _view = append_kv_state(
            state,
            0,
            keys,
            values,
            positions,
            valid_mask,
        )
        return updated

    first_keys = mx.full((1, 1, 2, 4), 1.0, dtype=mx.bfloat16)
    first_values = mx.full((1, 1, 2, 4), 2.0, dtype=mx.bfloat16)
    first_state = append_core(
        cache.state,
        first_keys,
        first_values,
        mx.array([[0, 1]], dtype=mx.int32),
        mx.array([[True, True]]),
    )
    mx.eval(first_state)
    cache.replace_state(first_state)

    second_keys = mx.full((1, 1, 1, 4), 3.0, dtype=mx.bfloat16)
    second_values = mx.full((1, 1, 1, 4), 4.0, dtype=mx.bfloat16)
    second_state = append_core(
        cache.state,
        second_keys,
        second_values,
        mx.array([[2]], dtype=mx.int32),
        mx.array([[True]]),
    )
    mx.eval(second_state)
    cache.replace_state(second_state)

    _assert_array_equal(cache.state[2], mx.array([3], dtype=mx.int32))
    _assert_array_equal(
        cache.state[0][0][0, 0, :3],
        mx.array(
            [[1.0] * 4, [1.0] * 4, [3.0] * 4],
            dtype=mx.bfloat16,
        ),
    )
    _assert_array_equal(
        cache.state[1][0][0, 0, :3],
        mx.array(
            [[2.0] * 4, [2.0] * 4, [4.0] * 4],
            dtype=mx.bfloat16,
        ),
    )


def test_cache_reset_clears_logical_lengths_without_changing_capacity():
    config = _cache_config()
    cache = KVCache.allocate(config, 1, 3, mx.bfloat16)
    updated, _view = append_kv_state(
        cache.state,
        0,
        mx.ones((1, 1, 1, 4), dtype=mx.bfloat16),
        mx.ones((1, 1, 1, 4), dtype=mx.bfloat16),
        mx.array([[0]], dtype=mx.int32),
        mx.array([[True]]),
    )
    mx.eval(updated)
    cache.replace_state(updated)

    cache.reset()

    assert cache.state[0][0].shape == (1, 1, 3, 4)
    _assert_array_equal(cache.state[2], mx.zeros((1,), dtype=mx.int32))
