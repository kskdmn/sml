from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from config import OUTPUT_DIR, PROJECT_DIR, TOKENIZER_MODEL_PATH

INPUT_DIR = Path("~/Documents/data-common_pile/")
INPUT_FILE_NAME_REGEX = r".*-0000\.jsonl\.zst\Z"


@dataclass(slots=True)
class SMLConfig:
    vocab_size: int = 49_152
    hidden_size: int = 768
    num_layers: int = 24
    num_q_heads: int = 12
    num_kv_heads: int = 2
    intermediate_size: int = 2_304
    original_max_position_embeddings: int = 1_024
    rope_theta: float = 10_000.0
    rope_scaling_factor: float = 2.0
    # YaRN blends interpolated (long-context) and extrapolated (local) RoPE frequencies.
    # Rotation-count thresholds mark where that blend starts and ends across head dims.
    yarn_beta_fast: float = 32.0  # High-frequency dims at/above this keep extrapolated frequencies.
    yarn_beta_slow: float = 1.0  # Low-frequency dims at/below this use interpolated frequencies.
    rms_norm_eps: float = 1e-6
    attention_dropout: float = 0.005  # If overfitting, try 0.05 (usually more disruptive than hidden_dropout)
    hidden_dropout: float = 0.01  # If overfitting, try 0.1
    initializer_range: float = 0.02
    gradient_checkpointing: bool = True  # Trade extra compute for lower activation memory during training.
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

    @property
    def head_dim(self) -> int:
        """
        Grouped-query attention uses this derived width for both query and key/value
        projections.
        """
        return self.hidden_size // self.num_q_heads

    @property
    def effective_max_position_embeddings(self) -> int:
        """
        Round scaled context upward so fractional RoPE scaling never shortens the usable
        window.
        """
        return math.ceil(
            self.original_max_position_embeddings * self.rope_scaling_factor
        )


@dataclass(slots=True)
class TrainingConfig:
    input_dir: Path = INPUT_DIR
    input_file_name_regex: str = INPUT_FILE_NAME_REGEX  # Regex matched against each file name.
    output_dir: Path = OUTPUT_DIR
    tokenizer_model_path: Path = TOKENIZER_MODEL_PATH
    checkpoint_name: str = "sml.pt"
    sequence_length: int = 1_024
    batch_size: int = 1
    max_steps: int | None = 1_000  # Maximum optimizer steps. Set to None to train until epochs/data end.
    lr_total_steps: int | None = None  # LR schedule horizon. Falls back to max_steps when None.
    epochs: int = 1
    max_rows_per_file: int | None = 10_000  # Maximum rows read from each input file per epoch. Set to None for all rows.
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    min_lr_ratio: float = 0.1
    log_every: int = 100
    save_every: int = 0  # 0 for never, positive for every N steps
    seed: int = 42
    device: str = "auto"
    autocast_dtype: str = "bfloat16"
