from __future__ import annotations

import math

import mlx.core as mx
from mlx import nn

from sml.model.cache import KVArrayState, append_kv_state
from sml.model.config import ModelConfig
from sml.model.rope import RotaryEmbedding, rotate_half


class _Linear(nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.weight = mx.zeros((output_size, input_size), dtype=mx.bfloat16)

    def __call__(self, x: mx.array) -> mx.array:
        return x @ self.weight.T


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, epsilon: float) -> None:
        super().__init__()
        self.weight = mx.ones((hidden_size,), dtype=mx.bfloat16)
        self.epsilon = epsilon

    @staticmethod
    def forward_arrays(weight: mx.array, x: mx.array, epsilon: float) -> mx.array:
        x_fp32 = x.astype(mx.float32)
        mean_square = mx.mean(mx.square(x_fp32), axis=-1, keepdims=True)
        normalized = x_fp32 * mx.rsqrt(mean_square + epsilon)
        return (normalized * weight.astype(mx.float32)).astype(mx.bfloat16)

    def __call__(self, x: mx.array) -> mx.array:
        return self.forward_arrays(self.weight, x, self.epsilon)


def keyed_dropout(
    x: mx.array,
    probability: float,
    key: mx.array,
) -> tuple[mx.array, mx.array]:
    if probability == 0.0:
        return x, key
    next_key, dropout_key = mx.random.split(key)
    keep_probability = 1.0 - probability
    mask = mx.random.bernoulli(keep_probability, shape=x.shape, key=dropout_key)
    return mx.where(mask, x / keep_probability, 0.0).astype(x.dtype), next_key


def _linear(x: mx.array, parameters: dict[str, mx.array]) -> mx.array:
    return x @ parameters["weight"].T


def _apply_rotary_batched(
    rope: RotaryEmbedding,
    q: mx.array,
    k: mx.array,
    positions: mx.array,
) -> tuple[mx.array, mx.array]:
    in_bounds = (positions >= 0) & (positions < rope.config.effective_context_length)
    safe_positions = mx.where(in_bounds, positions, mx.zeros_like(positions))
    cos = rope.cos_cached[safe_positions][:, None, :, :]
    sin = rope.sin_cached[safe_positions][:, None, :, :]
    q_fp32 = q.astype(mx.float32)
    k_fp32 = k.astype(mx.float32)
    rotated_q = q_fp32 * cos + rotate_half(q_fp32) * sin
    rotated_k = k_fp32 * cos + rotate_half(k_fp32) * sin
    valid = in_bounds[:, None, :, None]
    q_result = mx.where(
        valid,
        rotated_q,
        mx.full(rotated_q.shape, float("nan"), dtype=mx.float32),
    ).astype(mx.bfloat16)
    k_result = mx.where(
        valid,
        rotated_k,
        mx.full(rotated_k.shape, float("nan"), dtype=mx.float32),
    ).astype(mx.bfloat16)
    return q_result, k_result


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: ModelConfig, layer_index: int) -> None:
        super().__init__()
        self.config = config
        self.layer_index = layer_index
        self.q_proj = _Linear(config.hidden_size, config.hidden_size)
        self.k_proj = _Linear(
            config.hidden_size,
            config.num_kv_heads * config.head_dim,
        )
        self.v_proj = _Linear(
            config.hidden_size,
            config.num_kv_heads * config.head_dim,
        )
        self.o_proj = _Linear(config.hidden_size, config.hidden_size)
        self.rope = RotaryEmbedding(config)

    def forward_arrays(
        self,
        parameters: dict[str, dict[str, mx.array]],
        x: mx.array,
        *,
        attention_mask: mx.array,
        positions: mx.array,
        cache_state: KVArrayState | None,
    ) -> tuple[mx.array, KVArrayState | None]:
        batch_size, query_length, _hidden_size = x.shape
        q = _linear(x, parameters["q_proj"])
        k = _linear(x, parameters["k_proj"])
        v = _linear(x, parameters["v_proj"])
        q = q.reshape(
            (batch_size, query_length, self.config.num_q_heads, self.config.head_dim)
        ).swapaxes(1, 2)
        k = k.reshape(
            (batch_size, query_length, self.config.num_kv_heads, self.config.head_dim)
        ).swapaxes(1, 2)
        v = v.reshape(
            (batch_size, query_length, self.config.num_kv_heads, self.config.head_dim)
        ).swapaxes(1, 2)
        q, k = _apply_rotary_batched(self.rope, q, k, positions)

        if cache_state is None:
            attended_keys = k
            attended_values = v
            key_mask = attention_mask
            key_positions = positions
            updated_cache_state = None
        else:
            updated_cache_state, view = append_kv_state(
                cache_state,
                self.layer_index,
                k,
                v,
                positions,
                attention_mask,
            )
            attended_keys = view.keys
            attended_values = view.values
            key_mask = view.valid_mask
            capacity = attended_keys.shape[2]
            key_positions = mx.broadcast_to(
                mx.arange(capacity, dtype=mx.int32)[None, :],
                (batch_size, capacity),
            )

        causal_mask = key_positions[:, None, None, :] <= positions[:, None, :, None]
        boolean_mask = causal_mask & key_mask[:, None, None, :]
        output = mx.fast.scaled_dot_product_attention(
            q,
            attended_keys,
            attended_values,
            scale=1.0 / math.sqrt(self.config.head_dim),
            mask=boolean_mask,
        ).astype(mx.bfloat16)
        output = output.swapaxes(1, 2).reshape(
            (batch_size, query_length, self.config.hidden_size)
        )
        output = _linear(output, parameters["o_proj"]).astype(mx.bfloat16)
        output = mx.where(attention_mask[:, :, None], output, 0.0).astype(mx.bfloat16)
        return output, updated_cache_state

    def __call__(
        self,
        x: mx.array,
        *,
        attention_mask: mx.array | None = None,
        positions: mx.array | None = None,
    ) -> mx.array:
        batch_size, query_length, _hidden_size = x.shape
        if attention_mask is None:
            attention_mask = mx.ones((batch_size, query_length), dtype=mx.bool_)
        if positions is None:
            positions = mx.broadcast_to(
                mx.arange(query_length, dtype=mx.int32)[None, :],
                (batch_size, query_length),
            )
        output, _cache_state = self.forward_arrays(
            self.parameters(),
            x,
            attention_mask=attention_mask,
            positions=positions,
            cache_state=None,
        )
        return output


class SwiGLUFeedForward(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.dropout_probability = config.hidden_dropout
        self.gate_proj = _Linear(config.hidden_size, config.intermediate_size)
        self.up_proj = _Linear(config.hidden_size, config.intermediate_size)
        self.down_proj = _Linear(config.intermediate_size, config.hidden_size)

    def forward_arrays(
        self,
        parameters: dict[str, dict[str, mx.array]],
        x: mx.array,
        *,
        training: bool,
        key: mx.array | None,
    ) -> tuple[mx.array, mx.array | None]:
        hidden = nn.silu(_linear(x, parameters["gate_proj"])) * _linear(
            x, parameters["up_proj"]
        )
        output = _linear(hidden, parameters["down_proj"]).astype(mx.bfloat16)
        if training and self.dropout_probability > 0.0:
            if key is None:
                raise ValueError("training with dropout requires an explicit key")
            output, key = keyed_dropout(output, self.dropout_probability, key)
        return output, key

    def __call__(
        self,
        x: mx.array,
        *,
        training: bool = False,
        key: mx.array | None = None,
    ) -> tuple[mx.array, mx.array | None]:
        return self.forward_arrays(
            self.parameters(),
            x,
            training=training,
            key=key,
        )


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, layer_index: int) -> None:
        super().__init__()
        self.input_norm = RMSNorm(config.hidden_size, config.rms_norm_epsilon)
        self.self_attn = GroupedQueryAttention(config, layer_index)
        self.post_attn_norm = RMSNorm(config.hidden_size, config.rms_norm_epsilon)
        self.mlp = SwiGLUFeedForward(config)

    def forward_arrays(
        self,
        parameters: dict[str, object],
        x: mx.array,
        *,
        attention_mask: mx.array,
        positions: mx.array,
        cache_state: KVArrayState | None,
        training: bool,
        key: mx.array | None,
    ) -> tuple[mx.array, KVArrayState | None, mx.array | None]:
        attention_input = RMSNorm.forward_arrays(
            parameters["input_norm"]["weight"],
            x,
            self.input_norm.epsilon,
        )
        attention_output, cache_state = self.self_attn.forward_arrays(
            parameters["self_attn"],
            attention_input,
            attention_mask=attention_mask,
            positions=positions,
            cache_state=cache_state,
        )
        hidden = (x + attention_output).astype(mx.bfloat16)
        mlp_input = RMSNorm.forward_arrays(
            parameters["post_attn_norm"]["weight"],
            hidden,
            self.post_attn_norm.epsilon,
        )
        mlp_output, key = self.mlp.forward_arrays(
            parameters["mlp"],
            mlp_input,
            training=training,
            key=key,
        )
        hidden = (hidden + mlp_output).astype(mx.bfloat16)
        return hidden, cache_state, key

    def __call__(
        self,
        x: mx.array,
        *,
        attention_mask: mx.array,
        positions: mx.array,
        cache_state: KVArrayState | None = None,
        training: bool = False,
        key: mx.array | None = None,
    ) -> tuple[mx.array, KVArrayState | None, mx.array | None]:
        return self.forward_arrays(
            self.parameters(),
            x,
            attention_mask=attention_mask,
            positions=positions,
            cache_state=cache_state,
            training=training,
            key=key,
        )
