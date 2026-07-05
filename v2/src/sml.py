from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint


@dataclass(slots=True)
class SMLConfig:
    """
    Model hyperparameters.

    ``rope_scaling_factor`` is the inference context multiplier saved in checkpoints.
    Training disables YaRN and uses standard RoPE; see
    ``train_sml.model_config_for_training``.
    """

    vocab_size: int = 24_576
    hidden_size: int = 512
    num_layers: int = 12
    num_q_heads: int = 8
    num_kv_heads: int = 2
    intermediate_size: int = 1_536
    original_max_position_embeddings: int = 1_024  # RoPE design window; YaRN stretches beyond this.
    rope_theta: float = 10_000.0  # RoPE base (theta in inv_freq = 1 / theta^(2k/d)).
    rope_scaling_factor: float = 2.0  # Inference context multiplier; 1 disables YaRN.
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
    # RNG seed for torch.multinomial; ignored when temperature <= 0.
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


@dataclass(slots=True)
class SMLForwardOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None


def apply_repetition_penalty(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    """
    Down-weight logits for tokens already present in ``input_ids``.

    Follows the Hugging Face repetition-penalty rule: positive logits are divided by
    ``penalty`` and negative logits are multiplied by it. A penalty of ``1.0`` is a
    no-op and returns ``logits`` unchanged.
    """
    if penalty == 1.0:
        return logits

    adjusted = logits.clone()
    for batch_idx in range(input_ids.shape[0]):
        token_ids = input_ids[batch_idx].unique()
        scores = adjusted[batch_idx, token_ids]
        adjusted[batch_idx, token_ids] = torch.where(
            scores > 0,
            scores / penalty,
            scores * penalty,
        )
    return adjusted


def apply_no_repeat_ngram(
    logits: torch.Tensor,
    generated: torch.Tensor,
    ngram_size: int,
) -> torch.Tensor:
    """
    Hard-block tokens that would complete a repeated n-gram.

    When the last ``ngram_size - 1`` tokens match an earlier prefix in
    ``generated``, any token that previously completed that prefix is set to
    ``-inf``. ``ngram_size <= 0`` disables blocking.
    """
    if ngram_size <= 0:
        return logits

    adjusted = logits.clone()
    for batch_idx in range(generated.shape[0]):
        token_ids = generated[batch_idx].tolist()
        if len(token_ids) + 1 < ngram_size:
            continue

        prefix = tuple(token_ids[-(ngram_size - 1) :])
        banned: set[int] = set()
        for start in range(len(token_ids) - ngram_size + 1):
            if tuple(token_ids[start : start + ngram_size - 1]) == prefix:
                banned.add(token_ids[start + ngram_size - 1])
        for token_id in banned:
            adjusted[batch_idx, token_id] = float("-inf")
    return adjusted


def select_next_token(
    logits: torch.Tensor,
    generation_config: GenerationConfig,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Choose the next token from adjusted logits.

    ``generation_config.temperature <= 0`` selects greedy argmax. Otherwise logits
    are scaled by temperature, optionally filtered with nucleus (top-p) sampling,
    and drawn from the resulting categorical distribution.
    """
    if generation_config.temperature <= 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    scaled_logits = logits / generation_config.temperature
    if generation_config.top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_mask = cumulative_probs - sorted_probs >= generation_config.top_p
        sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))
        scaled_logits = torch.zeros_like(scaled_logits).scatter(
            -1,
            sorted_indices,
            sorted_logits,
        )

    probs = F.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        """
        RMSNorm uses no bias or mean subtraction; the learned vector only rescales
        RMS-normalized activations.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))  # [hidden_size]
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        RMSNorm normalizes each hidden vector by its root mean square.
        Assume `x.shape = (batch_size, seq_len, hidden_size)`.

        mean_square = average(x^2)
        RMS = sqrt(mean_square)
        x_norm = x / RMS
        output = weight * x_norm = x * (weight / RMS)
        The output has the same shape as the input, but the RMS of the output is approximately 1.
        RMSNorm scales the vector so its RMS is approximately 1, but it does not force mean 0 or variance 1.
        """
        dtype = x.dtype
        x = x.float()  # Convert to float32 for numerical stability. [batch_size, seq_len, hidden_size]
        mean_square = x.pow(2).mean(dim=-1, keepdim=True)  # [batch_size, seq_len, 1]; `x` will be `average(x^2)` along the last dimension.
        x = x * torch.rsqrt(mean_square + self.eps)  # [batch_size, seq_len, hidden_size]; `x` will be `x / sqrt(mean_square + eps)` along the last dimension.
        return (self.weight * x).to(dtype)  # [batch_size, seq_len, hidden_size]


# YaRN (Yet another RoPE extension) stretches context beyond
# original_max_position_embeddings without retraining RoPE from scratch.
# Training uses standard RoPE (rope_scaling_factor=1.0); checkpoints store the
# inference rope_scaling_factor and apply YaRN when the model is loaded for generation.
# Each attention head has head_dim values; RoPE uses head_dim // 2 frequency bands
# (band k covers dimensions 2k and 2k+1). Band 0 rotates fastest; higher k rotates
# slower. When rope_scaling_factor > 1, YaRN blends two inverse-frequency sets per band:
# extrapolated (trained) and interpolated (divided by rope_scaling_factor).


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


def yarn_linear_ramp_mask(
    low: int,
    high: int,
    num_rotary_bands: int,
) -> torch.Tensor:
    """
    Linear interpolation weight alpha_k in [0, 1] for each rotary band.

    alpha_k = 0 for k < low (extrapolated only), alpha_k = 1 for k > high
    (interpolated only), and alpha_k = (k - low) / (high - low) in between.
    Partial stretch per band is:

        inv_freq_final = alpha_k * inv_freq_int + (1 - alpha_k) * inv_freq_ext

    where inv_freq_int = inv_freq_ext / rope_scaling_factor.
    """
    if num_rotary_bands == 0:
        return torch.empty(0, dtype=torch.float32)
    if low == high:
        high = min(high + 1, num_rotary_bands - 1)
        if low == high:
            high += 1
    positions = torch.arange(num_rotary_bands, dtype=torch.float32)
    return torch.clamp((positions - low) / (high - low), min=0.0, max=1.0)


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
        """
        RoPE cos/sin cache with optional YaRN long-context correction.

        Caches cover effective_max_position_embeddings positions. When
        rope_scaling_factor is 1, this is standard RoPE. Otherwise YaRN blends
        extrapolated frequencies (trained, good for nearby tokens) with
        interpolated frequencies (divided by rope_scaling_factor, good for
        distant tokens) across rotary bands; see ``_compute_inv_freq``.
        """
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
        positions = torch.arange(effective_max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        attention_factor = resolve_yarn_attention_factor(
            rope_scaling_factor,
            attention_factor=yarn_attention_factor,
            mscale=yarn_mscale,
            mscale_all_dim=yarn_mscale_all_dim,
        )
        self.register_buffer("cos_cached", emb.cos() * attention_factor, persistent=False)
        self.register_buffer("sin_cached", emb.sin() * attention_factor, persistent=False)

    def _compute_inv_freq(
        self,
        dim: int,
        original_max_position_embeddings: int,
        base: float,
        rope_scaling_factor: float,
        yarn_beta_fast: float,
        yarn_beta_slow: float,
        yarn_truncate: bool,
    ) -> torch.Tensor:
        """
        Per-band inverse RoPE frequencies with YaRN blending.

        Extrapolated (trained) frequency for band k:

            inv_freq_ext = 1 / base^(2k/d)

        Interpolated (stretched) frequency:

            inv_freq_int = inv_freq_ext / rope_scaling_factor

        Rotation count over original length L:

            R_k = L * inv_freq_ext / (2 * pi)

        Band cutoffs come from ``yarn_find_correction_range`` using
        ``yarn_beta_fast`` and ``yarn_beta_slow``. The final frequency is a
        linear blend controlled by ``yarn_linear_ramp_mask``; fast bands (low k,
        high R_k) keep extrapolated frequencies, slow bands (high k, low R_k)
        use interpolated ones, and middle bands are partially stretched.
        """
        rotary_dims = torch.arange(0, dim, 2, dtype=torch.float32)
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

    def forward(
        self,
        seq_len: int,
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        The offset supports KV-cache decoding, where new tokens need positions after the
        already-cached prefix.
        """
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


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    RoPE stores paired channels as two halves, so rotation swaps halves and negates the
    second component.
    """
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Broadcast cached sin and cos over batch and heads so query and key use the same
    positional frame.
    """
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class KVCache:
    def __init__(self) -> None:
        """
        Layer-indexed lists let autoregressive decoding append keys and values one step
        at a time.
        """
        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []

    def get_seq_len(self, layer_idx: int) -> int:
        """
        Missing layers report zero length, which marks the first decode pass as the
        uncached prompt pass.
        """
        if layer_idx >= len(self.key_cache):
            return 0
        return self.key_cache[layer_idx].shape[2]

    def update(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Concatenate along the sequence axis so future attention sees the full prefix for
        this layer.
        """
        if layer_idx >= len(self.key_cache):
            self.key_cache.append(key)
            self.value_cache.append(value)
        else:
            self.key_cache[layer_idx] = torch.cat((self.key_cache[layer_idx], key), dim=2)
            self.value_cache[layer_idx] = torch.cat(
                (self.value_cache[layer_idx], value),
                dim=2,
            )

        return self.key_cache[layer_idx], self.value_cache[layer_idx]


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: SMLConfig, layer_idx: int) -> None:
        """
        Query heads may outnumber KV heads; `num_groups` records how many query heads
        share each key/value head.
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_q_heads = config.num_q_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.num_groups = self.num_q_heads // self.num_kv_heads
        self.attention_dropout = config.attention_dropout
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

    def repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expand KV heads to the query-head count expected by PyTorch attention without
        changing their grouping semantics.
        """
        batch_size, kv_heads, seq_len, head_dim = x.shape
        x = x.unsqueeze(2)
        x = x.expand(batch_size, kv_heads, self.num_groups, seq_len, head_dim)
        return x.reshape(batch_size, self.num_q_heads, seq_len, head_dim)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: KVCache | None = None,
    ) -> torch.Tensor:
        """
        Cached decoding projects only new tokens, appends past keys and values, and
        disables causal masking after the prompt pass.
        """
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.view(batch_size, seq_len, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        position_offset = 0 if kv_cache is None else kv_cache.get_seq_len(self.layer_idx)
        cos, sin = self.rope(seq_len, position_offset)
        q, k = apply_rotary_pos_emb(
            q,
            k,
            cos.to(device=x.device, dtype=q.dtype),
            sin.to(device=x.device, dtype=q.dtype),
        )

        past_seq_len = position_offset
        if kv_cache is not None:
            k, v = kv_cache.update(self.layer_idx, k, v)

        k = self.repeat_kv(k)
        v = self.repeat_kv(v)
        dropout_p = self.attention_dropout if self.training else 0.0
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=past_seq_len == 0,
        )
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.hidden_size)
        return self.o_proj(output)


class SwiGLUFeedForward(nn.Module):
    def __init__(self, config: SMLConfig) -> None:
        """
        The gate, up, and down projections implement LLaMA-style SwiGLU with dropout
        after projecting back to hidden size.
        """
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
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Gate the up projection with SiLU before returning to the model hidden size.
        """
        x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.dropout(self.down_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: SMLConfig, layer_idx: int) -> None:
        """
        The block uses pre-norm residual order: normalize before each sublayer, then add
        the sublayer output back to the stream.
        """
        super().__init__()
        self.input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = GroupedQueryAttention(config, layer_idx)
        self.post_attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = SwiGLUFeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: KVCache | None = None,
    ) -> torch.Tensor:
        """
        Pass the optional KV cache only through attention; the MLP always sees the
        current residual stream.
        """
        x = x + self.self_attn(self.input_norm(x), kv_cache)
        x = x + self.mlp(self.post_attn_norm(x))
        return x


class SMLLanguageModel(nn.Module):
    def __init__(self, config: SMLConfig) -> None:
        """
        When embeddings are tied, the input table and LM head share the same parameter
        tensor rather than copied weights.
        """
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
        )
        self.layers = nn.ModuleList(
            TransformerBlock(config, layer_idx=i) for i in range(config.num_layers)
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """
        Use the configured normal initializer and zero the padding row so pad tokens
        start inert.
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.initializer_range,
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.initializer_range,
            )
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
    ) -> SMLForwardOutput:
        """
        Cached prefix length counts against the effective context window; labels are
        optional so the same path serves training and inference.
        """
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
            if self._should_checkpoint_layers(kv_cache):
                x = torch.utils.checkpoint.checkpoint(
                    layer,
                    x,
                    use_reentrant=False,
                )
            else:
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

    def _should_checkpoint_layers(self, kv_cache: KVCache | None) -> bool:
        """
        Checkpointing is training-only and disabled with KV cache because cached
        decoding depends on layer side effects.
        """
        return (
            self.config.gradient_checkpointing
            and self.training
            and torch.is_grad_enabled()
            and kv_cache is None
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        eos_token_id: int | None = None,
        generation_config: GenerationConfig | None = None,
    ) -> torch.Tensor:
        """
        Autoregressively extend ``input_ids`` by up to ``max_new_tokens``.

        KV cache is reused after the prompt pass. Decoding knobs live in
        ``GenerationConfig`` (see this module); when omitted, greedy argmax is
        used. Generation stops early only when every batch item emits ``eos_token_id``.
        """
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
        generator = None
        if config.seed is not None:
            generator = torch.Generator(device=input_ids.device)
            generator.manual_seed(config.seed)

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
            next_token = select_next_token(
                next_token_logits,
                config,
                generator=generator,
            )
            generated = torch.cat((generated, next_token), dim=1)
            if eos_token_id is not None and bool((next_token == eos_token_id).all()):
                break

        return generated


def compute_causal_lm_loss(
    logits: torch.Tensor,  # [batch_size, seq_len, vocab_size]
    labels: torch.Tensor,  # [batch_size, seq_len]
    pad_token_id: int,
) -> torch.Tensor:
    """
    Labels are already shifted by the dataset; this function masks padding and flattens
    tensors for cross-entropy.
    """
    if logits.shape[:-1] != labels.shape:
        raise ValueError(
            "labels must have the same batch and sequence shape as logits"
        )
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),  # [batch_size * seq_len, vocab_size]
        labels.reshape(-1),  # [batch_size * seq_len]
        ignore_index=pad_token_id,
    )


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """
    Report both all parameters and optimizer-participating parameters for training logs.
    """
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return total, trainable


def create_model(config: SMLConfig | None = None) -> SMLLanguageModel:
    """
    Small factory for callers that want a default-config model without constructing
    SMLConfig themselves.
    """
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
