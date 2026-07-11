from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten


__all__ = [
    "GenerationConfig",
    "GroupedQueryAttention",
    "KVCache",
    "RMSNorm",
    "RotaryEmbedding",
    "SMLConfig",
    "SMLForwardOutput",
    "SMLLanguageModel",
    "SwiGLUFeedForward",
    "TransformerBlock",
    "apply_no_repeat_ngram",
    "apply_repetition_penalty",
    "apply_rotary_pos_emb",
    "compute_causal_lm_loss",
    "count_parameters",
    "create_model",
    "estimate_model_size",
    "lr_lambda",
    "resolve_yarn_attention_factor",
    "rotate_half",
    "select_next_token",
    "yarn_find_correction_dim",
    "yarn_find_correction_range",
    "yarn_get_mscale",
    "yarn_linear_ramp_mask",
]


@dataclass(slots=True)
class SMLConfig:
    """
    Model hyperparameters.

    ``rope_scaling_factor`` is the inference context multiplier saved in checkpoints.
    Training disables YaRN and uses standard RoPE; see
    ``train_sml.model_config_for_training``.
    """

    vocab_size: int = 28_672
    hidden_size: int = 768
    num_layers: int = 12
    num_q_heads: int = 12
    num_kv_heads: int = 3
    intermediate_size: int = 2_176
    original_max_position_embeddings: int = 1_024  # RoPE design window; YaRN stretches beyond this.
    rope_theta: float = 10_000.0  # RoPE base (theta in inv_freq = 1 / theta^(2k/d)).
    rope_scaling_factor: float = 4.0  # Inference context multiplier; 1 disables YaRN.
    yarn_beta_fast: float = 32.0  # Rotation-count cutoff for fast bands (extrapolate).
    yarn_beta_slow: float = 1.0  # Rotation-count cutoff for slow bands (interpolate).
    yarn_attention_factor: float | None = None  # Override cos/sin scaling; None infers from factor.
    yarn_mscale: float | None = None  # Optional numerator for inferred attention scaling. Valid if yarn_attention_factor is not set and yarn_mscale_all_dim is set.
    yarn_mscale_all_dim: float | None = None  # Optional denominator for inferred attention scaling. Valid if yarn_attention_factor is not set and yarn_mscale is set.
    yarn_truncate: bool = True  # Floor/ceil band cutoffs in the YaRN correction range.
    rms_norm_eps: float = 1e-6
    attention_dropout: float = 0.005  # If overfitting, try 0.05 (usually more disruptive than hidden_dropout)
    hidden_dropout: float = 0.01  # If overfitting, try 0.1
    initializer_range: float = 0.02
    gradient_checkpointing: bool = False  # Trade extra compute for lower activation memory during training.
    pad_token_id: int = 3
    bos_token_id: int = 1
    eos_token_id: int = 2
    unk_token_id: int = 0
    tie_word_embeddings: bool = True
    use_cache: bool = True  # For inference/generation.

    def __post_init__(self) -> None:
        """
        Validate shape and context-scaling invariants before model code relies on
        derived head dimensions and effective context length.
        """
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.hidden_size % self.num_q_heads != 0:
            raise ValueError("hidden_size must be divisible by num_q_heads")
        if self.num_q_heads % self.num_kv_heads != 0:
            raise ValueError("num_q_heads must be divisible by num_kv_heads")
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary embeddings")
        if self.intermediate_size <= 0:
            raise ValueError("intermediate_size must be positive")
        if self.original_max_position_embeddings <= 0:
            raise ValueError("original_max_position_embeddings must be positive")
        if (
            not math.isfinite(self.rope_scaling_factor)
            or self.rope_scaling_factor < 1.0
        ):
            raise ValueError("rope_scaling_factor must be at least 1.0")
        if (
            not math.isfinite(self.yarn_beta_fast)
            or not math.isfinite(self.yarn_beta_slow)
            or self.yarn_beta_fast <= 0.0
            or self.yarn_beta_slow <= 0.0
        ):
            raise ValueError("yarn_beta_fast and yarn_beta_slow must be positive")
        if self.yarn_beta_fast <= self.yarn_beta_slow:
            raise ValueError("yarn_beta_fast must be greater than yarn_beta_slow")
        if self.yarn_attention_factor is not None and (
            not math.isfinite(self.yarn_attention_factor)
            or self.yarn_attention_factor <= 0.0
        ):
            raise ValueError("yarn_attention_factor must be positive when set")
        for field_name, value in (
            ("yarn_mscale", self.yarn_mscale),
            ("yarn_mscale_all_dim", self.yarn_mscale_all_dim),
        ):
            if value is not None and (not math.isfinite(value) or value <= 0.0):
                raise ValueError(f"{field_name} must be positive when set")
        if (self.yarn_mscale is None) ^ (self.yarn_mscale_all_dim is None):
            raise ValueError(
                "yarn_mscale and yarn_mscale_all_dim must both be set or both be None"
            )

    @property
    def head_dim(self) -> int:
        """
        Per-head width for Q/K/V projections; RoPE uses head_dim // 2 frequency bands.
        """
        return self.hidden_size // self.num_q_heads

    @property
    def effective_max_position_embeddings(self) -> int:
        """
        Usable context length after YaRN scaling.

        ceil(original_max_position_embeddings * rope_scaling_factor); e.g. 1024 * 4
        yields 4096 positions in the RoPE cache.
        """
        return math.ceil(
            self.original_max_position_embeddings * self.rope_scaling_factor
        )


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """
    Inference-time decoding controls for ``SMLLanguageModel.generate``.

    These settings are applied when converting logits into the next token. They do
    not change model weights and can be tuned per request without retraining.

    Decoding order inside ``generate``:

    1. ``repetition_penalty`` down-weights logits for tokens already in the prefix.
    2. ``no_repeat_ngram_size`` hard-blocks tokens that would repeat an n-gram.
    3. ``temperature`` and ``top_p`` choose the next token (greedy or sampled).

    Defaults preserve legacy greedy decoding: ``temperature=0`` selects argmax and
    leaves repetition controls disabled.
    """

    temperature: float = 0.0
    # Sampling temperature; <= 0 keeps greedy argmax (default).
    # With sampling enabled, try 0.7-1.0 for natural variation; 0.8 is a common start.
    # Values above ~1.5 often look incoherent on small models.

    top_p: float = 1.0
    # Nucleus cutoff in (0, 1]; 1.0 disables top-p. Ignored when temperature <= 0.
    # With sampling, try 0.9-0.95 to trim low-probability tails without much quality loss.

    repetition_penalty: float = 1.0
    # Down-weight tokens already in the prefix; 1.0 disables. Must stay > 0.
    # For phrase loops on small models, try 1.05-1.25; start at 1.15. Above ~1.3 can sound odd.

    no_repeat_ngram_size: int = 0
    # Hard-block tokens that would repeat an n-gram of this length; 0 disables.
    # Use 3 or 4 when the same phrase repeats verbatim; pair with repetition_penalty.
    # 3 is a common starting point for small models and stricter than 4.

    seed: int | None = None
    # RNG seed for MLX sampling; ignored when temperature <= 0.
    # Set for reproducible sampling; omit to get different continuations each run.

    def __post_init__(self) -> None:
        """
        Reject non-finite or out-of-range values before decoding starts.
        """
        if not math.isfinite(self.temperature):
            raise ValueError("temperature must be finite")
        if self.top_p <= 0.0 or self.top_p > 1.0 or not math.isfinite(self.top_p):
            raise ValueError("top_p must be in (0, 1]")
        if self.repetition_penalty <= 0.0 or not math.isfinite(self.repetition_penalty):
            raise ValueError("repetition_penalty must be positive")
        if self.no_repeat_ngram_size < 0:
            raise ValueError("no_repeat_ngram_size must be non-negative")


def yarn_find_correction_dim(
    num_rotations: float,
    dim: int,
    base: float,
    original_max_position_embeddings: int,
) -> float:
    """
    Map a target rotation count to a rotary band index.

    Over the original context length L, band k completes
    R_k = L / (2 * pi * base^(2k/d)) full turns, where d is head_dim.
    Given a target rotation count N, solve R_k = N for k:

        k = d * ln(L / (2 * pi * N)) / (2 * ln(base))

    ``yarn_beta_fast`` and ``yarn_beta_slow`` are typical values of N.
    """
    return (
        dim
        * math.log(
            original_max_position_embeddings / (num_rotations * 2.0 * math.pi)
        )
        / (2.0 * math.log(base))
    )


def yarn_find_correction_range(
    low_rot: float,
    high_rot: float,
    rotary_dim: int,
    base: float,
    original_max_position_embeddings: int,
    *,
    truncate: bool = True,
) -> tuple[int, int]:
    """
    Convert YaRN rotation thresholds into band-index cutoffs for blending.

    ``low_rot`` (``yarn_beta_fast``) marks fast-rotating bands that keep extrapolated
    frequencies for local positional detail. ``high_rot`` (``yarn_beta_slow``) marks
    slow-rotating bands that use interpolated frequencies for long-range reach.

    Returns (low, high) band indices over ``rotary_dim // 2`` bands; bands below low
    extrapolate, above high interpolate, and bands in between are partially stretched.
    """
    num_rotary_bands = rotary_dim // 2
    max_band = max(num_rotary_bands - 1, 0)
    low = yarn_find_correction_dim(
        low_rot, rotary_dim, base, original_max_position_embeddings
    )
    high = yarn_find_correction_dim(
        high_rot, rotary_dim, base, original_max_position_embeddings
    )
    if truncate:
        low = math.floor(low)
        high = math.ceil(high)
    return max(0, min(int(low), max_band)), max(0, min(int(high), max_band))


def yarn_get_mscale(rope_scaling_factor: float, multiplier: float = 1.0) -> float:
    """
    Log-scaled magnitude correction used by YaRN attention scaling helpers.
    """
    if rope_scaling_factor <= 1.0:
        return 1.0
    return 0.1 * multiplier * math.log(rope_scaling_factor) + 1.0


def resolve_yarn_attention_factor(
    rope_scaling_factor: float,
    *,
    attention_factor: float | None = None,
    mscale: float | None = None,
    mscale_all_dim: float | None = None,
) -> float:
    """
    Rescale cached cos/sin values when context is extended.

    Uses an explicit ``attention_factor`` when provided. Otherwise infers the YaRN
    paper default from ``rope_scaling_factor``, or the ratio of two m-scales when
    both ``mscale`` and ``mscale_all_dim`` are set (useful for factors well above 2).
    """
    if attention_factor is not None:
        return attention_factor
    if rope_scaling_factor <= 1.0:
        return 1.0
    if mscale is not None and mscale_all_dim is not None:
        return yarn_get_mscale(rope_scaling_factor, mscale) / yarn_get_mscale(
            rope_scaling_factor, mscale_all_dim
        )
    return yarn_get_mscale(rope_scaling_factor)


@dataclass(slots=True)
class SMLForwardOutput:
    logits: mx.array
    loss: mx.array | None = None


def _init_linear(linear: nn.Linear, initializer_range: float) -> None:
    linear.weight = mx.random.normal(
        shape=linear.weight.shape,
        scale=initializer_range,
    )
    if "bias" in linear:
        linear.bias = mx.zeros(linear.bias.shape)


def _init_embedding(
    embedding: nn.Embedding,
    initializer_range: float,
    pad_token_id: int | None,
) -> None:
    weight = mx.random.normal(
        shape=embedding.weight.shape,
        scale=initializer_range,
    )
    if pad_token_id is not None and 0 <= pad_token_id < weight.shape[0]:
        rows = []
        if pad_token_id > 0:
            rows.append(weight[:pad_token_id])
        rows.append(mx.zeros((1, weight.shape[1]), dtype=weight.dtype))
        if pad_token_id + 1 < weight.shape[0]:
            rows.append(weight[pad_token_id + 1 :])
        weight = mx.concatenate(rows, axis=0)
    embedding.weight = weight


def _replace_vector_value(vector: mx.array, index: int, value: mx.array) -> mx.array:
    value = value.reshape((1,))
    pieces = []
    if index > 0:
        pieces.append(vector[:index])
    pieces.append(value)
    if index + 1 < vector.shape[0]:
        pieces.append(vector[index + 1 :])
    return mx.concatenate(pieces, axis=0)


def apply_repetition_penalty(
    logits: mx.array,
    input_ids: mx.array,
    penalty: float,
) -> mx.array:
    """
    Down-weight logits for tokens already present in ``input_ids``.
    """
    if penalty == 1.0:
        return logits

    adjusted_rows = []
    for batch_idx in range(input_ids.shape[0]):
        row = logits[batch_idx]
        token_ids = {int(token_id) for token_id in input_ids[batch_idx].tolist()}
        for token_id in token_ids:
            score = row[token_id]
            replacement = mx.where(score > 0, score / penalty, score * penalty)
            row = _replace_vector_value(row, token_id, replacement)
        adjusted_rows.append(row)
    return mx.stack(adjusted_rows, axis=0)


def apply_no_repeat_ngram(
    logits: mx.array,
    generated: mx.array,
    ngram_size: int,
) -> mx.array:
    """
    Hard-block tokens that would complete a repeated n-gram.
    """
    if ngram_size <= 0:
        return logits

    adjusted_rows = []
    for batch_idx in range(generated.shape[0]):
        row = logits[batch_idx]
        token_ids = [int(token_id) for token_id in generated[batch_idx].tolist()]
        if len(token_ids) + 1 >= ngram_size:
            prefix = tuple(token_ids[-(ngram_size - 1) :])
            banned: set[int] = set()
            for start in range(len(token_ids) - ngram_size + 1):
                if tuple(token_ids[start : start + ngram_size - 1]) == prefix:
                    banned.add(token_ids[start + ngram_size - 1])
            for token_id in banned:
                row = _replace_vector_value(
                    row,
                    token_id,
                    mx.array(-math.inf, dtype=row.dtype),
                )
        adjusted_rows.append(row)
    return mx.stack(adjusted_rows, axis=0)


def select_next_token(
    logits: mx.array,
    generation_config: GenerationConfig,
    *,
    key: mx.array | None = None,
) -> mx.array:
    """
    Choose the next token from adjusted logits.
    """
    if generation_config.temperature <= 0.0:
        return mx.expand_dims(mx.argmax(logits, axis=-1), axis=-1)

    scaled_logits = logits / generation_config.temperature
    if generation_config.top_p < 1.0:
        sorted_indices = mx.argsort(-scaled_logits, axis=-1)
        sorted_logits = mx.take_along_axis(scaled_logits, sorted_indices, axis=-1)
        sorted_probs = mx.softmax(sorted_logits, axis=-1)
        cumulative_probs = mx.cumsum(sorted_probs, axis=-1)
        sorted_mask = cumulative_probs - sorted_probs >= generation_config.top_p
        sorted_logits = mx.where(sorted_mask, -math.inf, sorted_logits)
        scaled_logits = mx.put_along_axis(
            mx.zeros_like(scaled_logits),
            sorted_indices,
            sorted_logits,
            axis=-1,
        )

    return mx.random.categorical(scaled_logits, num_samples=1, key=key)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = mx.ones((hidden_size,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps)


def yarn_linear_ramp_mask(
    low: int,
    high: int,
    num_rotary_bands: int,
) -> mx.array:
    if num_rotary_bands == 0:
        return mx.array([], dtype=mx.float32)
    if low == high:
        high = min(high + 1, num_rotary_bands - 1)
        if low == high:
            high += 1
    positions = mx.arange(num_rotary_bands, dtype=mx.float32)
    return mx.clip((positions - low) / (high - low), a_min=0.0, a_max=1.0)


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        original_max_position_embeddings: int,
        effective_max_position_embeddings: int,
        base: float,
        rope_scaling_factor: float,
        yarn_beta_fast: float,
        yarn_beta_slow: float,
        yarn_attention_factor: float | None = None,
        yarn_mscale: float | None = None,
        yarn_mscale_all_dim: float | None = None,
        yarn_truncate: bool = True,
    ) -> None:
        super().__init__()
        inv_freq = self._compute_inv_freq(
            dim=dim,
            original_max_position_embeddings=original_max_position_embeddings,
            base=base,
            rope_scaling_factor=rope_scaling_factor,
            yarn_beta_fast=yarn_beta_fast,
            yarn_beta_slow=yarn_beta_slow,
            yarn_truncate=yarn_truncate,
        )
        positions = mx.arange(effective_max_position_embeddings, dtype=mx.float32)
        freqs = mx.outer(positions, inv_freq)
        emb = mx.concatenate([freqs, freqs], axis=-1)
        attention_factor = resolve_yarn_attention_factor(
            rope_scaling_factor,
            attention_factor=yarn_attention_factor,
            mscale=yarn_mscale,
            mscale_all_dim=yarn_mscale_all_dim,
        )
        self._cos_cached = mx.cos(emb) * attention_factor
        self._sin_cached = mx.sin(emb) * attention_factor

    @property
    def cos_cached(self) -> mx.array:
        return self._cos_cached

    @property
    def sin_cached(self) -> mx.array:
        return self._sin_cached

    def _compute_inv_freq(
        self,
        dim: int,
        original_max_position_embeddings: int,
        base: float,
        rope_scaling_factor: float,
        yarn_beta_fast: float,
        yarn_beta_slow: float,
        yarn_truncate: bool,
    ) -> mx.array:
        rotary_dims = mx.arange(0, dim, 2, dtype=mx.float32)
        inv_freq_extrapolation = 1.0 / (base ** (rotary_dims / dim))
        if rope_scaling_factor == 1.0:
            return inv_freq_extrapolation

        inv_freq_interpolation = inv_freq_extrapolation / rope_scaling_factor
        low, high = yarn_find_correction_range(
            low_rot=yarn_beta_fast,
            high_rot=yarn_beta_slow,
            rotary_dim=dim,
            base=base,
            original_max_position_embeddings=original_max_position_embeddings,
            truncate=yarn_truncate,
        )
        interpolation_mask = yarn_linear_ramp_mask(low, high, dim // 2)
        extrapolation_mask = 1.0 - interpolation_mask
        return (
            inv_freq_interpolation * interpolation_mask
            + inv_freq_extrapolation * extrapolation_mask
        )

    def __call__(
        self,
        seq_len: int,
        position_offset: int = 0,
    ) -> tuple[mx.array, mx.array]:
        end = position_offset + seq_len
        if end > self.cos_cached.shape[0]:
            raise ValueError(
                "rotary positions exceed effective_max_position_embeddings: "
                f"{end} > {self.cos_cached.shape[0]}"
            )
        return (
            self.cos_cached[position_offset:end],
            self.sin_cached[position_offset:end],
        )


def rotate_half(x: mx.array) -> mx.array:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(
    q: mx.array,
    k: mx.array,
    cos: mx.array,
    sin: mx.array,
) -> tuple[mx.array, mx.array]:
    cos = mx.expand_dims(mx.expand_dims(cos, axis=0), axis=0)
    sin = mx.expand_dims(mx.expand_dims(sin, axis=0), axis=0)
    cos = cos.astype(q.dtype)
    sin = sin.astype(q.dtype)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class KVCache:
    def __init__(self) -> None:
        self.key_cache: list[mx.array] = []
        self.value_cache: list[mx.array] = []

    def get_seq_len(self, layer_idx: int) -> int:
        if layer_idx >= len(self.key_cache):
            return 0
        return self.key_cache[layer_idx].shape[2]

    def update(
        self,
        layer_idx: int,
        key: mx.array,
        value: mx.array,
    ) -> tuple[mx.array, mx.array]:
        if layer_idx >= len(self.key_cache):
            self.key_cache.append(key)
            self.value_cache.append(value)
        else:
            self.key_cache[layer_idx] = mx.concatenate(
                [self.key_cache[layer_idx], key],
                axis=2,
            )
            self.value_cache[layer_idx] = mx.concatenate(
                [self.value_cache[layer_idx], value],
                axis=2,
            )

        return self.key_cache[layer_idx], self.value_cache[layer_idx]


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: SMLConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_q_heads = config.num_q_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_q_heads * self.head_dim,
            bias=True,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_kv_heads * self.head_dim,
            bias=True,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_kv_heads * self.head_dim,
            bias=True,
        )
        self.o_proj = nn.Linear(
            config.num_q_heads * self.head_dim,
            config.hidden_size,
            bias=False,
        )
        for linear in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
            _init_linear(linear, config.initializer_range)
        self.rope = RotaryEmbedding(
            dim=self.head_dim,
            original_max_position_embeddings=config.original_max_position_embeddings,
            effective_max_position_embeddings=config.effective_max_position_embeddings,
            base=config.rope_theta,
            rope_scaling_factor=config.rope_scaling_factor,
            yarn_beta_fast=config.yarn_beta_fast,
            yarn_beta_slow=config.yarn_beta_slow,
            yarn_attention_factor=config.yarn_attention_factor,
            yarn_mscale=config.yarn_mscale,
            yarn_mscale_all_dim=config.yarn_mscale_all_dim,
            yarn_truncate=config.yarn_truncate,
        )

    def __call__(
        self,
        x: mx.array,
        kv_cache: KVCache | None = None,
    ) -> mx.array:
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.reshape((batch_size, seq_len, self.num_q_heads, self.head_dim))
        k = k.reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))
        v = v.reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))
        q = mx.swapaxes(q, 1, 2)
        k = mx.swapaxes(k, 1, 2)
        v = mx.swapaxes(v, 1, 2)

        position_offset = 0 if kv_cache is None else kv_cache.get_seq_len(self.layer_idx)
        cos, sin = self.rope(seq_len, position_offset)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        past_seq_len = position_offset
        if kv_cache is not None:
            k, v = kv_cache.update(self.layer_idx, k, v)

        # MLX fast attention does not expose attention-dropout, so the MLX
        # model ignores config.attention_dropout to keep the fused GQA path.
        output = mx.fast.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=1.0 / math.sqrt(self.head_dim),
            mask="causal" if past_seq_len == 0 else None,
        )
        output = mx.swapaxes(output, 1, 2)
        output = output.reshape((batch_size, seq_len, self.hidden_size))
        return self.o_proj(output)


class SwiGLUFeedForward(nn.Module):
    def __init__(self, config: SMLConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
        )
        self.up_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
        )
        self.down_proj = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
        )
        for linear in (self.gate_proj, self.up_proj, self.down_proj):
            _init_linear(linear, config.initializer_range)
        self.dropout = nn.Dropout(config.hidden_dropout)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.dropout(self.down_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: SMLConfig, layer_idx: int) -> None:
        super().__init__()
        self.input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = GroupedQueryAttention(config, layer_idx)
        self.post_attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = SwiGLUFeedForward(config)

    def __call__(
        self,
        x: mx.array,
        kv_cache: KVCache | None = None,
    ) -> mx.array:
        x = x + self.self_attn(self.input_norm(x), kv_cache)
        x = x + self.mlp(self.post_attn_norm(x))
        return x


class SMLLanguageModel(nn.Module):
    def __init__(self, config: SMLConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        _init_embedding(
            self.embed_tokens,
            config.initializer_range,
            config.pad_token_id,
        )
        self.layers = [
            TransformerBlock(config, layer_idx=i) for i in range(config.num_layers)
        ]
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        _init_linear(self.lm_head, config.initializer_range)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def __call__(
        self,
        input_ids: mx.array,
        labels: mx.array | None = None,
        kv_cache: KVCache | None = None,
    ) -> SMLForwardOutput:
        total_seq_len = input_ids.shape[-1]
        if kv_cache is not None:
            total_seq_len += kv_cache.get_seq_len(0)
        if total_seq_len > self.config.effective_max_position_embeddings:
            raise ValueError(
                "input sequence length exceeds effective_max_position_embeddings: "
                f"{total_seq_len} > {self.config.effective_max_position_embeddings}"
            )

        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, kv_cache)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = compute_causal_lm_loss(
                logits,
                labels,
                pad_token_id=self.config.pad_token_id,
            )
        return SMLForwardOutput(logits=logits, loss=loss)

    def generate(
        self,
        input_ids: mx.array,
        max_new_tokens: int = 50,
        eos_token_id: int | None = None,
        generation_config: GenerationConfig | None = None,
    ) -> mx.array:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        requested_seq_len = input_ids.shape[-1] + max_new_tokens
        if requested_seq_len > self.config.effective_max_position_embeddings:
            raise ValueError(
                "requested generation length exceeds effective_max_position_embeddings: "
                f"{requested_seq_len} > {self.config.effective_max_position_embeddings}"
            )
        self.eval()
        config = generation_config or GenerationConfig()
        eos_token_id = self.config.eos_token_id if eos_token_id is None else eos_token_id
        generated = input_ids
        kv_cache = KVCache() if self.config.use_cache else None
        key = mx.random.key(config.seed) if config.seed is not None else None

        for _ in range(max_new_tokens):
            if kv_cache is not None and len(kv_cache.key_cache) > 0:
                model_input = generated[:, -1:]
            else:
                model_input = generated

            output = self(model_input, kv_cache=kv_cache)
            next_token_logits = output.logits[:, -1, :]
            next_token_logits = apply_repetition_penalty(
                next_token_logits,
                generated,
                config.repetition_penalty,
            )
            next_token_logits = apply_no_repeat_ngram(
                next_token_logits,
                generated,
                config.no_repeat_ngram_size,
            )
            sample_key = None
            if key is not None:
                keys = mx.random.split(key)
                key = keys[0]
                sample_key = keys[1]
            next_token = select_next_token(
                next_token_logits,
                config,
                key=sample_key,
            ).astype(input_ids.dtype)
            generated = mx.concatenate([generated, next_token], axis=1)
            if eos_token_id is not None and bool(mx.all(next_token == eos_token_id).item()):
                break

        return generated


def compute_causal_lm_loss(
    logits: mx.array,
    labels: mx.array,
    pad_token_id: int,
) -> mx.array:
    if logits.shape[:-1] != labels.shape:
        raise ValueError("labels must have the same batch and sequence shape as logits")

    flat_logits = logits.reshape((-1, logits.shape[-1]))
    flat_labels = labels.reshape((-1,))
    losses = nn.losses.cross_entropy(flat_logits, flat_labels, reduction="none")
    mask = flat_labels != pad_token_id
    masked_losses = mx.where(mask, losses, 0.0)
    valid_count = mx.maximum(mx.sum(mask), 1)
    return mx.sum(masked_losses) / valid_count


def count_parameters(model: nn.Module) -> tuple[int, int]:
    def count_unique(parameters: dict) -> int:
        seen: set[int] = set()
        total = 0
        for _, parameter in tree_flatten(parameters):
            parameter_id = id(parameter)
            if parameter_id in seen:
                continue
            seen.add(parameter_id)
            total += math.prod(parameter.shape)
        return total

    return count_unique(model.parameters()), count_unique(model.trainable_parameters())


def create_model(config: SMLConfig | None = None) -> SMLLanguageModel:
    return SMLLanguageModel(config or SMLConfig())


def estimate_model_size(config: SMLConfig) -> int:
    """
    Count parameters analytically so model size can be inspected without allocating
    tensors.
    """
    attention_params = config.num_layers * (
        config.hidden_size * config.num_q_heads * config.head_dim
        + 2 * config.hidden_size * config.num_kv_heads * config.head_dim
        + config.hidden_size * config.hidden_size
    )
    mlp_params = config.num_layers * (
        2 * config.hidden_size * config.intermediate_size
        + config.intermediate_size * config.hidden_size
    )
    embedding_params = config.vocab_size * config.hidden_size
    norm_params = (config.num_layers * 2 + 1) * config.hidden_size
    untied_head_params = 0 if config.tie_word_embeddings else embedding_params
    return int(
        attention_params
        + mlp_params
        + embedding_params
        + norm_params
        + untied_head_params
    )


def lr_lambda(
    step: int,
    total_steps: int | None,
    warmup_steps: int,
    min_lr_ratio: float,
) -> float:
    """
    Warm up linearly, then either hold constant when no horizon is known or cosine-decay
    to a configured floor.
    """
    if step < warmup_steps:
        return float(step + 1) / float(max(1, warmup_steps))
    if total_steps is None:
        return 1.0
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return max(min_lr_ratio, cosine)
