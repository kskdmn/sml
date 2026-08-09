from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import mlx.core as mx
from mlx import nn

from sml.model.cache import KVArrayState, KVCache
from sml.model.config import InitializerConfig, ModelConfig
from sml.model.layers import RMSNorm, TransformerBlock


class _Embedding(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.weight = mx.zeros((vocab_size, hidden_size), dtype=mx.bfloat16)

    def __call__(self, input_ids: mx.array) -> mx.array:
        return self.weight[input_ids]


class _VocabularyProjection(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.weight = mx.zeros((vocab_size, hidden_size), dtype=mx.bfloat16)

    def __call__(self, hidden: mx.array) -> mx.array:
        return hidden @ self.weight.T


@dataclass(slots=True)
class ForwardOutput:
    logits: mx.array
    cache: KVCache | None
    next_key: mx.array | None


def _random_weight(
    key: mx.array,
    shape: tuple[int, ...],
    scale: float,
) -> tuple[mx.array, mx.array]:
    next_key, parameter_key = mx.random.split(key)
    weight = mx.random.normal(shape=shape, scale=scale, key=parameter_key).astype(
        mx.bfloat16
    )
    return weight, next_key


class SMLLanguageModel(nn.Module):
    def __init__(self, config: ModelConfig, *, key: mx.array) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = _Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            TransformerBlock(config, layer_index)
            for layer_index in range(config.num_layers)
        ]
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_epsilon)
        if not config.tie_word_embeddings:
            self.lm_head = _VocabularyProjection(config.vocab_size, config.hidden_size)
        self.initialize_state(key)

    def initialize_state(self, key: mx.array) -> None:
        config = self.config
        initializers = cast(InitializerConfig, config.initializers)
        embedding_weight, key = _random_weight(
            key,
            (config.vocab_size, config.hidden_size),
            initializers.embed_tokens,
        )
        vocabulary_rows = mx.arange(config.vocab_size, dtype=mx.int32)[:, None]
        self.embed_tokens.weight = mx.where(
            vocabulary_rows == config.pad_token_id,
            mx.zeros_like(embedding_weight),
            embedding_weight,
        ).astype(mx.bfloat16)

        for layer in self.layers:
            layer.input_norm.weight = mx.ones((config.hidden_size,), dtype=mx.bfloat16)
            for projection_name, shape, scale in (
                (
                    "q_proj",
                    (config.hidden_size, config.hidden_size),
                    initializers.q_proj,
                ),
                (
                    "k_proj",
                    (
                        config.num_kv_heads * config.head_dim,
                        config.hidden_size,
                    ),
                    initializers.k_proj,
                ),
                (
                    "v_proj",
                    (
                        config.num_kv_heads * config.head_dim,
                        config.hidden_size,
                    ),
                    initializers.v_proj,
                ),
                (
                    "o_proj",
                    (config.hidden_size, config.hidden_size),
                    initializers.o_proj,
                ),
            ):
                weight, key = _random_weight(key, shape, scale)
                getattr(layer.self_attn, projection_name).weight = weight
            layer.post_attn_norm.weight = mx.ones(
                (config.hidden_size,), dtype=mx.bfloat16
            )
            for projection_name, shape, scale in (
                (
                    "gate_proj",
                    (config.intermediate_size, config.hidden_size),
                    initializers.gate_proj,
                ),
                (
                    "up_proj",
                    (config.intermediate_size, config.hidden_size),
                    initializers.up_proj,
                ),
                (
                    "down_proj",
                    (config.hidden_size, config.intermediate_size),
                    initializers.down_proj,
                ),
            ):
                weight, key = _random_weight(key, shape, scale)
                getattr(layer.mlp, projection_name).weight = weight

        self.norm.weight = mx.ones((config.hidden_size,), dtype=mx.bfloat16)
        if not config.tie_word_embeddings:
            head_weight, _next_key = _random_weight(
                key,
                (config.vocab_size, config.hidden_size),
                initializers.lm_head,
            )
            self.lm_head.weight = head_weight

    def project_vocabulary(self, hidden: mx.array) -> mx.array:
        if self.config.tie_word_embeddings:
            return hidden @ self.embed_tokens.weight.T
        return self.lm_head(hidden)

    def forward_arrays(
        self,
        parameters: dict[str, object],
        input_ids: mx.array,
        *,
        attention_mask: mx.array | None,
        positions: mx.array | None,
        cache_state: KVArrayState | None,
        training: bool,
        key: mx.array | None,
    ) -> tuple[mx.array, KVArrayState | None, mx.array | None]:
        batch_size, query_length = input_ids.shape
        if attention_mask is None:
            attention_mask = mx.ones((batch_size, query_length), dtype=mx.bool_)
        else:
            attention_mask = attention_mask.astype(mx.bool_)

        if positions is None:
            if cache_state is None:
                base_lengths = mx.zeros((batch_size, 1), dtype=mx.int32)
            else:
                base_lengths = cache_state[2][:, None]
            relative_positions = mx.cumsum(attention_mask.astype(mx.int32), axis=1) - 1
            positions = mx.where(
                attention_mask,
                base_lengths + relative_positions,
                mx.zeros_like(relative_positions),
            ).astype(mx.int32)
        else:
            positions = positions.astype(mx.int32)

        hidden = parameters["embed_tokens"]["weight"][input_ids]
        hidden = hidden.astype(mx.bfloat16)
        for layer_index, layer in enumerate(self.layers):
            hidden, cache_state, key = layer.forward_arrays(
                parameters["layers"][layer_index],
                hidden,
                attention_mask=attention_mask,
                positions=positions,
                cache_state=cache_state,
                training=training,
                key=key,
            )
        hidden = RMSNorm.forward_arrays(
            parameters["norm"]["weight"],
            hidden,
            self.norm.epsilon,
        )
        if self.config.tie_word_embeddings:
            logits = hidden @ parameters["embed_tokens"]["weight"].T
        else:
            logits = hidden @ parameters["lm_head"]["weight"].T
        logits = logits.astype(mx.bfloat16)
        return logits, cache_state, key

    def __call__(
        self,
        input_ids: mx.array,
        *,
        attention_mask: mx.array | None = None,
        positions: mx.array | None = None,
        cache: KVCache | None = None,
        training: bool = False,
        key: mx.array | None = None,
    ) -> ForwardOutput:
        cache_state = None if cache is None else cache.state
        logits, returned_cache_state, next_key = self.forward_arrays(
            self.parameters(),
            input_ids,
            attention_mask=attention_mask,
            positions=positions,
            cache_state=cache_state,
            training=training,
            key=key,
        )
        if cache is not None:
            if returned_cache_state is None:
                raise RuntimeError("cached forward did not return cache state")
            mx.eval(logits, returned_cache_state)
            cache.replace_state(returned_cache_state)
        return ForwardOutput(logits=logits, cache=cache, next_key=next_key)


def causal_lm_loss(
    logits: mx.array,
    labels: mx.array,
    valid_mask: mx.array | None = None,
) -> mx.array:
    logits_fp32 = logits.astype(mx.float32)
    flat_logits = logits_fp32.reshape((-1, logits.shape[-1]))
    flat_labels = labels.reshape((-1,))
    losses = nn.losses.cross_entropy(
        flat_logits,
        flat_labels,
        reduction="none",
    )
    if valid_mask is None:
        flat_mask = mx.ones(flat_labels.shape, dtype=mx.bool_)
    else:
        flat_mask = valid_mask.reshape((-1,)).astype(mx.bool_)
    masked_losses = mx.where(flat_mask, losses, 0.0).astype(mx.float32)
    valid_count = mx.maximum(
        mx.sum(flat_mask).astype(mx.float32),
        mx.array(1.0, dtype=mx.float32),
    )
    return (mx.sum(masked_losses).astype(mx.float32) / valid_count).astype(mx.float32)
