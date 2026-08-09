from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass


def _require_positive_int(value: object, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be positive")


def _require_finite(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite")
    return value


@dataclass(frozen=True, slots=True)
class InitializerConfig:
    embed_tokens: float = 0.02
    lm_head: float = 0.02
    q_proj: float = 0.02
    k_proj: float = 0.02
    v_proj: float = 0.02
    o_proj: float = 0.02 / math.sqrt(24)
    gate_proj: float = 0.02
    up_proj: float = 0.02
    down_proj: float = 0.02 / math.sqrt(24)
    other: float = 0.02

    def __post_init__(self) -> None:
        for field_name in (
            "embed_tokens",
            "lm_head",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "other",
        ):
            value = _require_finite(getattr(self, field_name), field_name)
            if value < 0.0:
                raise ValueError(f"{field_name} must be non-negative and finite")

    @classmethod
    def depth_scaled(
        cls,
        initializer_range: float,
        num_layers: int,
    ) -> InitializerConfig:
        initializer_range = _require_finite(initializer_range, "initializer_range")
        if initializer_range < 0.0:
            raise ValueError("initializer_range must be non-negative and finite")
        _require_positive_int(num_layers, "num_layers")

        residual_initializer_range = initializer_range / math.sqrt(2 * num_layers)
        return cls(
            embed_tokens=initializer_range,
            lm_head=initializer_range,
            q_proj=initializer_range,
            k_proj=initializer_range,
            v_proj=initializer_range,
            o_proj=residual_initializer_range,
            gate_proj=initializer_range,
            up_proj=initializer_range,
            down_proj=residual_initializer_range,
            other=initializer_range,
        )


@dataclass(frozen=True, slots=True)
class ModelConfig:
    vocab_size: int = 28_672
    hidden_size: int = 768
    num_layers: int = 12
    num_q_heads: int = 12
    num_kv_heads: int = 3
    intermediate_size: int = 2_176
    original_context_length: int = 1_024
    rope_theta: float = 10_000.0
    rope_scaling_factor: float = 1.0
    yarn_beta_fast: float = 32.0
    yarn_beta_slow: float = 1.0
    yarn_attention_factor: float | None = None
    yarn_mscale: float | None = None
    yarn_mscale_all_dim: float | None = None
    yarn_truncate: bool = True
    rms_norm_epsilon: float = 1e-6
    hidden_dropout: float = 0.01
    initializer_range: float = 0.02
    initializers: InitializerConfig | Mapping[str, float] | None = None
    pad_token_id: int = 3
    bos_token_id: int = 1
    eos_token_id: int = 2
    unk_token_id: int = 0
    tie_word_embeddings: bool = True
    use_cache: bool = True

    def __post_init__(self) -> None:
        for field_name in (
            "vocab_size",
            "hidden_size",
            "num_layers",
            "num_q_heads",
            "num_kv_heads",
            "intermediate_size",
            "original_context_length",
        ):
            _require_positive_int(getattr(self, field_name), field_name)
        if self.hidden_size % self.num_q_heads != 0:
            raise ValueError("hidden_size must be divisible by num_q_heads")
        if self.num_q_heads % self.num_kv_heads != 0:
            raise ValueError("num_q_heads must be divisible by num_kv_heads")
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary embeddings")

        if _require_finite(self.rope_theta, "rope_theta") <= 0.0:
            raise ValueError("rope_theta must be positive")
        if _require_finite(self.rope_scaling_factor, "rope_scaling_factor") < 1.0:
            raise ValueError("rope_scaling_factor must be at least 1.0")
        for field_name in ("yarn_beta_fast", "yarn_beta_slow"):
            value = _require_finite(getattr(self, field_name), field_name)
            if value <= 0.0:
                raise ValueError("yarn_beta_fast and yarn_beta_slow must be positive")
        if self.yarn_beta_fast <= self.yarn_beta_slow:
            raise ValueError("yarn_beta_fast must be greater than yarn_beta_slow")
        if self.yarn_attention_factor is not None:
            attention_factor = _require_finite(
                self.yarn_attention_factor, "yarn_attention_factor"
            )
            if attention_factor <= 0.0:
                raise ValueError("yarn_attention_factor must be positive when set")
        for field_name in ("yarn_mscale", "yarn_mscale_all_dim"):
            value = getattr(self, field_name)
            if value is not None:
                value = _require_finite(value, field_name)
                if value <= 0.0:
                    raise ValueError(f"{field_name} must be positive when set")
        if (self.yarn_mscale is None) != (self.yarn_mscale_all_dim is None):
            raise ValueError(
                "yarn_mscale and yarn_mscale_all_dim must both be set or both be None"
            )

        if _require_finite(self.rms_norm_epsilon, "rms_norm_epsilon") <= 0.0:
            raise ValueError("rms_norm_epsilon must be positive")
        hidden_dropout = _require_finite(self.hidden_dropout, "hidden_dropout")
        if not 0.0 <= hidden_dropout < 1.0:
            raise ValueError("hidden_dropout must be in [0, 1)")
        initializer_range = _require_finite(self.initializer_range, "initializer_range")
        if initializer_range < 0.0:
            raise ValueError("initializer_range must be non-negative and finite")

        if self.initializers is None:
            object.__setattr__(
                self,
                "initializers",
                InitializerConfig.depth_scaled(initializer_range, self.num_layers),
            )
        elif isinstance(self.initializers, Mapping):
            try:
                normalized_initializers = InitializerConfig(**dict(self.initializers))
            except TypeError as error:
                raise ValueError(
                    "initializers must contain initializer fields"
                ) from error
            object.__setattr__(self, "initializers", normalized_initializers)
        elif not isinstance(self.initializers, InitializerConfig):
            raise ValueError("initializers must be an InitializerConfig or mapping")

        token_ids = {
            "pad_token_id": self.pad_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "unk_token_id": self.unk_token_id,
        }
        for field_name, token_id in token_ids.items():
            if (
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or not 0 <= token_id < self.vocab_size
            ):
                raise ValueError(f"{field_name} must be within vocab_size")
        if len(set(token_ids.values())) != len(token_ids):
            raise ValueError("special token IDs must be unique")
        for field_name in ("yarn_truncate", "tie_word_embeddings", "use_cache"):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be a bool")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_q_heads

    @property
    def effective_context_length(self) -> int:
        return math.ceil(self.original_context_length * self.rope_scaling_factor)


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    temperature: float = 0.0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    seed: int | None = None

    def __post_init__(self) -> None:
        _require_finite(self.temperature, "temperature")
        top_p = _require_finite(self.top_p, "top_p")
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        repetition_penalty = _require_finite(
            self.repetition_penalty, "repetition_penalty"
        )
        if repetition_penalty <= 0.0:
            raise ValueError("repetition_penalty must be positive")
        if (
            isinstance(self.no_repeat_ngram_size, bool)
            or not isinstance(self.no_repeat_ngram_size, int)
            or self.no_repeat_ngram_size < 0
        ):
            raise ValueError("no_repeat_ngram_size must be non-negative")
        if self.seed is not None and (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed <= 2**32 - 1
        ):
            raise ValueError("seed must be an integer in [0, 2**32 - 1]")
