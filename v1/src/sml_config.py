from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "output"
TOKENIZER_MODEL_PATH = OUTPUT_DIR / "bpe_tokenizer.model"
INPUT_DIR = Path("~/Documents/data-common_pile/")
INPUT_FILE_NAME_REGEX = r".*-000[0-9]\.jsonl\.zst\Z"


@dataclass(slots=True)
class SMLConfig:
    vocab_size: int = 49_152
    hidden_size: int = 512
    num_layers: int = 8
    num_q_heads: int = 8
    num_kv_heads: int = 2
    intermediate_size: int = 1_536
    max_position_embeddings: int = 1_024
    rope_theta: float = 10_000.0
    rms_norm_eps: float = 1e-6
    attention_dropout: float = 0.0  # If overfitting, try 0.05 (usually more disruptive than hidden_dropout)
    hidden_dropout: float = 0.0  # If overfitting, try 0.1
    initializer_range: float = 0.02
    gradient_checkpointing: bool = True  # Trade extra compute for lower activation memory during training.
    pad_token_id: int = 3
    bos_token_id: int = 1
    eos_token_id: int = 2
    unk_token_id: int = 0
    tie_word_embeddings: bool = True
    use_cache: bool = True  # For inference/generation.

    def __post_init__(self) -> None:
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
        if self.max_position_embeddings <= 0:
            raise ValueError("max_position_embeddings must be positive")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_q_heads


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
    epochs: int = 1
    max_rows_per_file: int | None = 10_000  # Maximum rows read from each input file per epoch. Set to None for all rows.
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    min_lr_ratio: float = 0.1
    log_every: int = 100
    save_every: int = 100
    seed: int = 42
    device: str = "auto"
    autocast_dtype: str = "bfloat16"
