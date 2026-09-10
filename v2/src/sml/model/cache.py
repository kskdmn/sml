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


def reset_kv_state(state: KVArrayState) -> KVArrayState:
    keys, values, lengths = state
    for layer in (*keys, *values, lengths):
        layer[:] = 0
    mx.eval(*keys, *values, lengths)
    return keys, values, lengths


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
    """Update prevalidated state without synchronizing the compiled array path.

    Every valid position must fit the allocated capacity, and every slot below
    the resulting logical length must have been populated. Public model calls
    enforce contiguous appends through ``KVCache.validate_append``; compiled
    inference enforces these bounds while preparing its request buckets.
    """
    key_layers, value_layers, lengths = state
    cached_keys = key_layers[layer_index]
    cached_values = value_layers[layer_index]
    capacity = cached_keys.shape[2]

    in_bounds = (positions >= 0) & (positions < capacity)
    write_mask = valid_mask & in_bounds
    batch_size, query_length = positions.shape
    slot_indices = mx.arange(capacity, dtype=mx.int32)[None, :]
    if query_length == 1:
        # Decode has one candidate per row, so no duplicate-write arbitration
        # or capacity-sized payload gathers are needed.
        written_slots = ((slot_indices == positions) & write_mask)[:, None, :, None]
        selected_keys = keys
        selected_values = values
    else:
        safe_positions = mx.where(in_bounds, positions, mx.zeros_like(positions))
        batch_indices = mx.broadcast_to(
            mx.arange(batch_size, dtype=mx.int32)[:, None],
            positions.shape,
        )
        token_indices = mx.broadcast_to(
            mx.arange(query_length, dtype=mx.int32)[None, :],
            positions.shape,
        )
        priorities = mx.where(
            write_mask,
            token_indices,
            mx.full(positions.shape, -1, dtype=mx.int32),
        )
        last_valid_token = (
            mx.full(
                (batch_size, capacity),
                -1,
                dtype=mx.int32,
            )
            .at[batch_indices, safe_positions]
            .maximum(priorities)
        )
        selected_token = mx.maximum(last_valid_token, 0)
        selected_indices = mx.broadcast_to(
            selected_token[:, None, :, None],
            cached_keys.shape,
        )
        selected_keys = mx.take_along_axis(keys, selected_indices, axis=2)
        selected_values = mx.take_along_axis(values, selected_indices, axis=2)
        written_slots = last_valid_token[:, None, :, None] >= 0
    updated_keys = mx.where(
        written_slots,
        selected_keys,
        cached_keys,
    )
    updated_values = mx.where(
        written_slots,
        selected_values,
        cached_values,
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

    def validate_append(
        self,
        attention_mask: mx.array,
        positions: mx.array | None,
    ) -> mx.array:
        """Validate a public append and return its absolute token positions."""
        key_layers, _value_layers, lengths = self._state
        if attention_mask.ndim != 2 or attention_mask.shape[0] != lengths.shape[0]:
            raise ValueError("attention_mask batch size must match the cache")
        if positions is not None and positions.shape != attention_mask.shape:
            raise ValueError("positions must match the input shape")
        capacity = key_layers[0].shape[2]
        attention_mask = attention_mask.astype(mx.bool_)
        relative_positions = mx.cumsum(attention_mask.astype(mx.int32), axis=1) - 1
        expected_positions = lengths[:, None] + relative_positions
        if positions is None:
            positions = mx.where(attention_mask, expected_positions, 0)
        next_lengths = lengths + mx.sum(attention_mask, axis=1)
        overflow = mx.any(next_lengths > capacity) | mx.any(
            attention_mask & (positions >= capacity)
        )
        noncontiguous = mx.any(attention_mask & (positions != expected_positions))
        overflow, noncontiguous = mx.stack((overflow, noncontiguous)).tolist()
        if overflow:
            raise ValueError(f"cache capacity {capacity} would be exceeded")
        if noncontiguous:
            raise ValueError(
                "cache positions must append contiguously from each logical length"
            )
        return positions.astype(mx.int32)

    def reset(self) -> None:
        keys, values, lengths = self._state
        self._state = (
            tuple(mx.zeros_like(layer) for layer in keys),
            tuple(mx.zeros_like(layer) for layer in values),
            mx.zeros_like(lengths),
        )
