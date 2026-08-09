from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from sml.model.config import GenerationConfig


@dataclass(frozen=True, slots=True)
class TokenSelection:
    token_ids: mx.array
    next_key: mx.array


def _seen_token_mask(
    tokens: mx.array,
    logical_lengths: mx.array,
    vocab_size: int,
) -> mx.array:
    batch_size, capacity = tokens.shape
    valid = mx.arange(capacity, dtype=mx.int32)[None, :] < logical_lengths[:, None]
    sentinel = mx.array(vocab_size, dtype=tokens.dtype)
    indices = mx.where(valid, tokens, sentinel)
    seen = mx.put_along_axis(
        mx.zeros((batch_size, vocab_size + 1), dtype=mx.bool_),
        indices,
        mx.ones(tokens.shape, dtype=mx.bool_),
        axis=-1,
    )
    return seen[:, :vocab_size]


def apply_repetition_penalty(
    logits: mx.array,
    tokens: mx.array,
    logical_lengths: mx.array,
    penalty: float,
) -> mx.array:
    if penalty == 1.0:
        return logits

    seen = _seen_token_mask(tokens, logical_lengths, logits.shape[-1])
    penalized = mx.where(logits > 0, logits / penalty, logits * penalty)
    return mx.where(seen, penalized, logits)


def apply_no_repeat_ngram(
    logits: mx.array,
    tokens: mx.array,
    logical_lengths: mx.array,
    ngram_size: int,
) -> mx.array:
    if ngram_size <= 0:
        return logits
    if ngram_size == 1:
        banned = _seen_token_mask(tokens, logical_lengths, logits.shape[-1])
        return mx.where(banned, float("-inf"), logits)

    capacity = tokens.shape[1]
    if ngram_size > capacity:
        return logits

    prefix_size = ngram_size - 1
    start_count = capacity - prefix_size
    suffix_offsets = mx.arange(prefix_size, dtype=mx.int32)[None, :]
    suffix_positions = logical_lengths[:, None] - prefix_size + suffix_offsets
    safe_suffix_positions = mx.clip(suffix_positions, 0, capacity - 1)
    suffix = mx.take_along_axis(tokens, safe_suffix_positions, axis=1)

    candidate_prefixes = mx.stack(
        [tokens[:, offset : offset + start_count] for offset in range(prefix_size)],
        axis=-1,
    )
    prefix_matches = mx.all(candidate_prefixes == suffix[:, None, :], axis=-1)
    starts = mx.arange(start_count, dtype=mx.int32)[None, :]
    complete = starts + ngram_size <= logical_lengths[:, None]
    eligible = logical_lengths[:, None] >= prefix_size
    matching_starts = prefix_matches & complete & eligible

    successor_ids = tokens[:, prefix_size : prefix_size + start_count]
    vocab_size = logits.shape[-1]
    sentinel = mx.array(vocab_size, dtype=tokens.dtype)
    banned_ids = mx.where(matching_starts, successor_ids, sentinel)
    banned = mx.put_along_axis(
        mx.zeros((tokens.shape[0], vocab_size + 1), dtype=mx.bool_),
        banned_ids,
        mx.ones(banned_ids.shape, dtype=mx.bool_),
        axis=-1,
    )
    return mx.where(banned[:, :vocab_size], float("-inf"), logits)


def select_next_token_arrays(
    logits_row: mx.array,
    key: mx.array,
    *,
    temperature: float,
    top_p: float,
) -> tuple[mx.array, mx.array]:
    if temperature <= 0.0:
        return mx.argmax(logits_row, axis=-1), key

    scaled_logits = logits_row / temperature
    if top_p < 1.0:
        sorted_indices = mx.argsort(-scaled_logits, axis=-1)
        sorted_logits = mx.take_along_axis(
            scaled_logits,
            sorted_indices,
            axis=-1,
        )
        sorted_probabilities = mx.softmax(sorted_logits, axis=-1)
        cumulative_probabilities = mx.cumsum(sorted_probabilities, axis=-1)
        sorted_mask = cumulative_probabilities - sorted_probabilities >= top_p
        sorted_logits = mx.where(sorted_mask, float("-inf"), sorted_logits)
        scaled_logits = mx.put_along_axis(
            mx.zeros_like(scaled_logits),
            sorted_indices,
            sorted_logits,
            axis=-1,
        )

    next_key, sample_key = mx.random.split(key)
    token_id = mx.random.categorical(scaled_logits, key=sample_key)
    return token_id, next_key


def select_next_token(
    logits: mx.array,
    config: GenerationConfig,
    key: mx.array,
) -> TokenSelection:
    if config.temperature <= 0.0:
        return TokenSelection(mx.argmax(logits, axis=-1), key)

    token_ids = []
    next_key = key
    for logits_row in logits:
        token_id, next_key = select_next_token_arrays(
            logits_row,
            next_key,
            temperature=config.temperature,
            top_p=config.top_p,
        )
        token_ids.append(token_id)
    return TokenSelection(mx.stack(token_ids), next_key)
