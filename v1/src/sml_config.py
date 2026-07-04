from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path

from config import OUTPUT_DIR, PROJECT_DIR, TOKENIZER_MODEL_PATH

INPUT_DIR = Path("~/Documents/data-common_pile/")
INPUT_FILE_NAME_REGEX = r".*-00[0-9][0-9]\.jsonl\.zst\Z"


@dataclass(slots=True)
class SMLConfig:
    """
    Model hyperparameters.

    ``rope_scaling_factor`` is the inference context multiplier saved in checkpoints.
    Training disables YaRN and uses standard RoPE; see ``model_config_for_training``.
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


def model_config_for_training(config: SMLConfig) -> SMLConfig:
    """
    Return a copy that trains with standard RoPE inside ``sequence_length``.

    YaRN is applied only when checkpoints are loaded for inference.
    """
    return replace(config, rope_scaling_factor=1.0)


@dataclass(slots=True)
class TrainingConfig:
    """
    Training hyperparameters and I/O paths.

    ``train_sml.py`` reads these defaults from code; the CLI only exposes
    ``--resume``. Edit fields here (or pass a custom instance to
    ``train_sml.train_model``) instead of adding CLI flags.
    """

    input_dir: Path = INPUT_DIR
    input_file_name_regex: str = INPUT_FILE_NAME_REGEX  # Regex matched against each file name.
    output_dir: Path = OUTPUT_DIR
    tokenizer_model_path: Path = TOKENIZER_MODEL_PATH
    checkpoint_name: str = "sml.pt"
    sequence_length: int = 1_024
    batch_size: int = 1
    max_steps: int | None = None # Maximum optimizer steps. Set to None to train until epochs/data end.
    lr_total_steps: int | None = 100_000  # LR schedule horizon. Falls back to max_steps when None.
    epochs: int = 1
    max_rows_per_file: int | None = 32_768  # Maximum rows read from each input file per epoch. Set to None for all rows.
    shuffle_input_files: bool = True  # Shuffle model-training shards deterministically with seed.
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0
    warmup_steps: int = 100  # TODO for v2: set to int(lr_total_steps * 0.01)
    min_lr_ratio: float = 0.1
    log_every: int = 10
    save_every: int = 1_000  # 0 for never, positive for every N steps
    seed: int = 42
    device: str = "auto"
    autocast_dtype: str = "bfloat16"


@dataclass(slots=True)
class LoRAConfig:
    """
    Low-rank adapter settings for SWAG fine-tuning.

    Adapters attach to attention and MLP projections by module name.
    """

    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj",
        #"k_proj",
        "v_proj",
        #"o_proj",
        #"gate_proj",
        #"up_proj",
        #"down_proj",
    )


@dataclass(slots=True)
class SwagFineTuneConfig:
    """
    LoRA fine-tuning hyperparameters for SWAG continuation training.

    ``ft_swag.py`` reads these defaults from code; the CLI only exposes
    ``--resume``. Edit fields here (or pass a custom instance to
    ``ft_swag.fine_tune_swag``) instead of adding CLI flags.
    """

    dataset_name: str = "allenai/swag"
    dataset_config: str = "regular"
    dataset_split: str = "train"
    hf_cache_dir: Path | None = PROJECT_DIR.parent / ".hf-cache"
    pretrained_checkpoint_path: Path = OUTPUT_DIR / "sml.pt"
    output_dir: Path = OUTPUT_DIR
    tokenizer_model_path: Path = TOKENIZER_MODEL_PATH
    checkpoint_name: str = "sml-swag.pt"
    sequence_length: int = 256
    batch_size: int = 1
    max_steps: int | None = 5_000
    lr_total_steps: int | None = 5_000
    epochs: int = 1
    max_examples: int | None = None
    shuffle_examples: bool = True
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    min_lr_ratio: float = 0.1
    log_every: int = 10
    save_every: int = 500
    seed: int = 42
    device: str = "auto"
    autocast_dtype: str = "bfloat16"


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
