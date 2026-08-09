from __future__ import annotations

import math

import mlx.core as mx

from sml.model.config import ModelConfig


def find_correction_dimension(
    num_rotations: float,
    rotary_dim: int,
    base: float,
    original_context_length: int,
) -> float:
    return (
        rotary_dim
        * math.log(original_context_length / (num_rotations * 2.0 * math.pi))
        / (2.0 * math.log(base))
    )


def find_correction_range(
    low_rot: float,
    high_rot: float,
    rotary_dim: int,
    base: float,
    original_context_length: int,
    *,
    truncate: bool = True,
) -> tuple[float, float]:
    num_rotary_bands = rotary_dim // 2
    max_band = max(num_rotary_bands - 1, 0)
    low = find_correction_dimension(low_rot, rotary_dim, base, original_context_length)
    high = find_correction_dimension(
        high_rot, rotary_dim, base, original_context_length
    )
    if truncate:
        low = math.floor(low)
        high = math.ceil(high)
    return max(0.0, min(low, max_band)), max(0.0, min(high, max_band))


def _get_mscale(rope_scaling_factor: float, multiplier: float = 1.0) -> float:
    if rope_scaling_factor <= 1.0:
        return 1.0
    return 0.1 * multiplier * math.log(rope_scaling_factor) + 1.0


def resolve_attention_factor(
    rope_scaling_factor: float,
    *,
    attention_factor: float | None = None,
    mscale: float | None = None,
    mscale_all_dim: float | None = None,
) -> float:
    if attention_factor is not None:
        return attention_factor
    if rope_scaling_factor <= 1.0:
        return 1.0
    if mscale is not None and mscale_all_dim is not None:
        return _get_mscale(rope_scaling_factor, mscale) / _get_mscale(
            rope_scaling_factor, mscale_all_dim
        )
    return _get_mscale(rope_scaling_factor)


def _linear_ramp_mask(low: float, high: float, num_rotary_bands: int) -> mx.array:
    if num_rotary_bands == 0:
        return mx.array([], dtype=mx.float32)
    if low == high:
        high = min(high + 1, num_rotary_bands - 1)
        if low == high:
            high += 1
    positions = mx.arange(num_rotary_bands, dtype=mx.float32)
    return mx.clip((positions - low) / (high - low), a_min=0.0, a_max=1.0)


class RotaryEmbedding:
    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self._inv_freq = self._compute_inv_freq()
        positions = mx.arange(config.effective_context_length, dtype=mx.float32)
        frequencies = mx.outer(positions, self._inv_freq)
        embeddings = mx.concatenate([frequencies, frequencies], axis=-1)
        attention_factor = resolve_attention_factor(
            config.rope_scaling_factor,
            attention_factor=config.yarn_attention_factor,
            mscale=config.yarn_mscale,
            mscale_all_dim=config.yarn_mscale_all_dim,
        )
        self._cos_cached = mx.cos(embeddings) * attention_factor
        self._sin_cached = mx.sin(embeddings) * attention_factor
        mx.eval(self._inv_freq, self._cos_cached, self._sin_cached)

    @property
    def cos_cached(self) -> mx.array:
        return self._cos_cached

    @property
    def sin_cached(self) -> mx.array:
        return self._sin_cached

    def _compute_inv_freq(self) -> mx.array:
        config = self.config
        rotary_dims = mx.arange(0, config.head_dim, 2, dtype=mx.float32)
        extrapolated = 1.0 / (config.rope_theta ** (rotary_dims / config.head_dim))
        if config.rope_scaling_factor == 1.0:
            return extrapolated

        interpolated = extrapolated / config.rope_scaling_factor
        low, high = find_correction_range(
            config.yarn_beta_fast,
            config.yarn_beta_slow,
            config.head_dim,
            config.rope_theta,
            config.original_context_length,
            truncate=config.yarn_truncate,
        )
        interpolation_mask = _linear_ramp_mask(low, high, config.head_dim // 2)
        return interpolated * interpolation_mask + extrapolated * (
            1.0 - interpolation_mask
        )

    def __call__(
        self,
        q: mx.array,
        k: mx.array,
        positions: mx.array,
    ) -> tuple[mx.array, mx.array]:
        return apply_rotary(
            q,
            k,
            self.cos_cached[positions],
            self.sin_cached[positions],
        )


def rotate_half(x: mx.array) -> mx.array:
    half = x.shape[-1] // 2
    return mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def apply_rotary(
    q: mx.array,
    k: mx.array,
    cos: mx.array,
    sin: mx.array,
) -> tuple[mx.array, mx.array]:
    cos = mx.expand_dims(mx.expand_dims(cos, axis=0), axis=0)
    sin = mx.expand_dims(mx.expand_dims(sin, axis=0), axis=0)
    q_fp32 = q.astype(mx.float32)
    k_fp32 = k.astype(mx.float32)
    rotated_q = q_fp32 * cos + rotate_half(q_fp32) * sin
    rotated_k = k_fp32 * cos + rotate_half(k_fp32) * sin
    return rotated_q.astype(q.dtype), rotated_k.astype(k.dtype)
