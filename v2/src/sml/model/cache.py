from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from sml.model.config import ModelConfig

KVArrayState = tuple[
    tuple[mx.array, ...],
    tuple[mx.array, ...],
    mx.array,
]


@dataclass(frozen=True, slots=True)
class KVView:
    keys: mx.array
    values: mx.array
    valid_mask: mx.array


def allocate_kv_state(
    config: ModelConfig,
    batch_size: int,
    capacity: int,
    dtype: mx.Dtype,
) -> KVArrayState:
    shape = (
        batch_size,
        config.num_kv_heads,
        capacity,
        config.head_dim,
    )
    keys = tuple(mx.zeros(shape, dtype=dtype) for _ in range(config.num_layers))
    values = tuple(mx.zeros(shape, dtype=dtype) for _ in range(config.num_layers))
    lengths = mx.zeros((batch_size,), dtype=mx.int32)
    return keys, values, lengths


def append_kv_state(
    state: KVArrayState,
    layer_index: int,
    keys: mx.array,
    values: mx.array,
    positions: mx.array,
    valid_mask: mx.array,
) -> tuple[KVArrayState, KVView]:
    key_layers, value_layers, lengths = state
    cached_keys = key_layers[layer_index]
    cached_values = value_layers[layer_index]
    capacity = cached_keys.shape[2]

    in_bounds = (positions >= 0) & (positions < capacity)
    write_mask = valid_mask & in_bounds
    has_valid_write = mx.any(write_mask, axis=1)
    fallback_token = mx.argmax(write_mask.astype(mx.int32), axis=1)
    fallback_position = mx.take_along_axis(
        positions,
        fallback_token[:, None],
        axis=1,
    )
    fallback_position = mx.where(
        has_valid_write[:, None],
        fallback_position,
        mx.zeros_like(fallback_position),
    )
    scatter_positions = mx.where(write_mask, positions, fallback_position)
    scatter_indices = mx.broadcast_to(
        scatter_positions[:, None, :, None],
        keys.shape,
    )
    fallback_indices = mx.broadcast_to(
        fallback_token[:, None, None, None],
        (*keys.shape[:2], 1, keys.shape[3]),
    )
    fallback_keys = mx.take_along_axis(keys, fallback_indices, axis=2)
    fallback_values = mx.take_along_axis(values, fallback_indices, axis=2)
    key_updates = mx.where(write_mask[:, None, :, None], keys, fallback_keys)
    value_updates = mx.where(write_mask[:, None, :, None], values, fallback_values)
    prior_keys = mx.take_along_axis(cached_keys, scatter_indices, axis=2)
    prior_values = mx.take_along_axis(cached_values, scatter_indices, axis=2)
    key_updates = mx.where(
        has_valid_write[:, None, None, None],
        key_updates,
        prior_keys,
    )
    value_updates = mx.where(
        has_valid_write[:, None, None, None],
        value_updates,
        prior_values,
    )
    updated_keys = mx.put_along_axis(
        cached_keys,
        scatter_indices,
        key_updates,
        axis=2,
    )
    updated_values = mx.put_along_axis(
        cached_values,
        scatter_indices,
        value_updates,
        axis=2,
    )

    written_lengths = mx.max(
        mx.where(write_mask, positions + 1, mx.zeros_like(positions)),
        axis=1,
    ).astype(mx.int32)
    updated_lengths = mx.maximum(lengths, written_lengths).astype(mx.int32)
    updated_key_layers = (
        key_layers[:layer_index] + (updated_keys,) + key_layers[layer_index + 1 :]
    )
    updated_value_layers = (
        value_layers[:layer_index] + (updated_values,) + value_layers[layer_index + 1 :]
    )
    returned_state = updated_key_layers, updated_value_layers, updated_lengths
    slot_indices = mx.arange(capacity, dtype=mx.int32)[None, :]
    view = KVView(
        keys=updated_keys,
        values=updated_values,
        valid_mask=slot_indices < updated_lengths[:, None],
    )
    return returned_state, view


class KVCache:
    def __init__(self, state: KVArrayState) -> None:
        self._state = state

    @classmethod
    def allocate(
        cls,
        config: ModelConfig,
        batch_size: int,
        capacity: int,
        dtype: mx.Dtype,
    ) -> KVCache:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        return cls(allocate_kv_state(config, batch_size, capacity, dtype))

    @property
    def state(self) -> KVArrayState:
        return self._state

    def replace_state(self, state: KVArrayState) -> None:
        self._state = state

    def reset(self) -> None:
        keys, values, lengths = self._state
        self._state = keys, values, mx.zeros_like(lengths)
