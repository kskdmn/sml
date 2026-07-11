"""
Fine-tune an SML MLX checkpoint on the SWAG train split with LoRA adapters.

Loads ``allenai/swag`` (``regular`` config) and trains low-rank adapters on the
gold continuation for each ``startphrase``. Defaults read from
``SwagFineTuneConfig`` in this module; the CLI exposes ``--resume`` only.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Iterator, Sequence

from config import (
    DEFAULT_MODEL_PATH,
    DEFAULT_TOKENIZER_MODEL_PATH,
    OUTPUT_DIR,
    PROJECT_DIR,
    SUCCESS_RETURN_CODE,
    resolve_path,
)
from infer_sml import load_checkpoint_metadata, normalize_model_config
from lora import (
    LoRAConfig,
    apply_lora,
    load_lora_state_dict,
    lora_parameters,
    lora_state_dict,
    merge_lora,
    require_lora_modules,
)
from train_sml import (
    ROW_INCREMENT,
    ReadingProgress,
    ResumeProgress,
    TrainingDataState,
    TrainingResumeState,
    apply_model_dtype,
    build_lr_schedule,
    clip_gradients_by_global_norm,
    count_resume_batches,
    format_training_log,
    get_special_token_id,
    is_step_limit_reached,
    iter_mlx_batches,
    iter_unseen_batches,
    load_tokenizer,
    parse_checkpoint_data_state,
    reset_training_data_state,
    resolve_lr_total_steps,
    set_seed,
    tree_add,
    tree_scale,
)

ENDING_KEY_PREFIX = "ending"
SWAG_PROGRESS_NAME = "swag-train"
MODEL_WEIGHTS_NAME = "model.safetensors"
LORA_STATE_NAME = "lora.npz"
OPTIMIZER_STATE_NAME = "optimizer.npz"
METADATA_NAME = "metadata.json"
STOCHASTIC_RESUME_NOTE = (
    "Resume restores model adapters, optimizer state, and data position; "
    "stochastic continuity is not guaranteed."
)
DEFAULT_PAD_TOKEN_ID = 3
DEFAULT_BOS_TOKEN_ID = 1
DEFAULT_EOS_TOKEN_ID = 2


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
    pretrained_checkpoint_path: Path = DEFAULT_MODEL_PATH
    output_dir: Path = OUTPUT_DIR
    tokenizer_model_path: Path = DEFAULT_TOKENIZER_MODEL_PATH
    checkpoint_name: str = "sml-swag"
    sequence_length: int = 256
    batch_size: int = 1
    max_steps: int | None = 5_000
    lr_total_steps: int | None = 5_000
    epochs: int = 5
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
    autocast_dtype: str = "bfloat16"


def resolve_swag_label(value: object) -> int:
    """
    SWAG labels are stored as ints, strings, or Hugging Face ClassLabel values.
    """
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value)
    if hasattr(value, "item"):
        return int(value.item())
    raise ValueError(f"Unsupported SWAG label value: {value!r}")


def format_swag_parts(row: Mapping[str, object]) -> tuple[str, str]:
    """
    Return the conditioning start phrase and gold continuation separately.
    """
    label = resolve_swag_label(row["label"])
    startphrase = row["startphrase"]
    ending = row[f"{ENDING_KEY_PREFIX}{label}"]
    if not isinstance(startphrase, str):
        raise ValueError("SWAG startphrase must be a string")
    if not isinstance(ending, str):
        raise ValueError(f"SWAG ending{label} must be a string")
    return startphrase.rstrip(), ending.lstrip()


def join_swag_parts(startphrase: str, ending: str) -> str:
    return f"{startphrase} {ending}"


def format_swag_example(row: Mapping[str, object]) -> str:
    """
    Build a causal-LM target by appending the gold ending to the start phrase.
    """
    startphrase, ending = format_swag_parts(row)
    return join_swag_parts(startphrase, ending)


def load_swag_dataset(fine_tune_config: SwagFineTuneConfig):
    """
    Download or load the SWAG split from the Hugging Face datasets hub.
    """
    from datasets import load_dataset

    cache_dir = (
        str(resolve_path(fine_tune_config.hf_cache_dir))
        if fine_tune_config.hf_cache_dir is not None
        else None
    )
    return load_dataset(
        fine_tune_config.dataset_name,
        fine_tune_config.dataset_config,
        split=fine_tune_config.dataset_split,
        cache_dir=cache_dir,
    )


def iter_swag_parts(
    fine_tune_config: SwagFineTuneConfig,
    epoch: int = 0,
    progress: ReadingProgress | None = None,
    data_state: TrainingDataState | None = None,
) -> Iterator[tuple[str, str]]:
    """
    Yield SWAG context/ending pairs, optionally shuffled and resumable by position.
    """
    dataset = load_swag_dataset(fine_tune_config)
    example_count = len(dataset)
    if fine_tune_config.max_examples is not None:
        example_count = min(example_count, fine_tune_config.max_examples)

    indices = list(range(example_count))
    if fine_tune_config.shuffle_examples:
        random.Random(fine_tune_config.seed + epoch).shuffle(indices)

    start_position = 0
    if data_state is not None and data_state.line_number is not None:
        start_position = data_state.line_number + ROW_INCREMENT

    for position, example_index in enumerate(indices):
        if position < start_position:
            continue

        row = dataset[example_index]
        if progress is not None:
            progress.input_file = SWAG_PROGRESS_NAME
            progress.line_number = position
            progress.example_index = example_index
        if data_state is not None:
            data_state.input_file_index = 0
            data_state.line_number = position

        yield format_swag_parts(row)


def iter_swag_texts(
    fine_tune_config: SwagFineTuneConfig,
    epoch: int = 0,
    progress: ReadingProgress | None = None,
    data_state: TrainingDataState | None = None,
) -> Iterator[str]:
    """
    Yield formatted SWAG examples, optionally shuffled and resumable by position.
    """
    for startphrase, ending in iter_swag_parts(
        fine_tune_config,
        epoch=epoch,
        progress=progress,
        data_state=data_state,
    ):
        yield join_swag_parts(startphrase, ending)


class SwagExampleDataset:
    def __init__(
        self,
        examples: Iterable[tuple[str, str]],
        tokenizer: object,
        sequence_length: int,
        pad_token_id: int,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        data_state: TrainingDataState | None = None,
    ) -> None:
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        self.examples = examples
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.data_state = data_state

    def __iter__(self) -> Iterator[dict[str, list[int]]]:
        """
        Yield one fixed-length causal-LM pair per SWAG example.

        Labels outside the gold ending and EOS are set to ``pad_token_id`` so
        the model loss ignores the conditioning context and padding positions.
        """
        bos_token_id = get_special_token_id(
            self.tokenizer,
            "bos_id",
            self.bos_token_id,
        )
        eos_token_id = get_special_token_id(
            self.tokenizer,
            "eos_id",
            self.eos_token_id,
        )
        if self.data_state is not None:
            self.data_state.token_buffer = []

        tokens_per_example = self.sequence_length + ROW_INCREMENT
        for startphrase, ending in self.examples:
            context_ids = list(self.tokenizer.encode(startphrase, out_type=int))
            ending_ids = list(
                self.tokenizer.encode(join_swag_parts("", ending), out_type=int)
            )
            if eos_token_id is not None:
                ending_ids.append(eos_token_id)
            example_tokens: list[int] = []
            if bos_token_id is not None:
                example_tokens.append(bos_token_id)
            example_tokens.extend(int(token_id) for token_id in context_ids)
            ending_start = len(example_tokens)
            example_tokens.extend(int(token_id) for token_id in ending_ids)
            if len(example_tokens) > self.sequence_length:
                if self.data_state is not None:
                    self.data_state.token_buffer = []
                continue

            padded_tokens = example_tokens + [self.pad_token_id] * (
                tokens_per_example - len(example_tokens)
            )
            labels = list(padded_tokens[1:])
            ending_label_start = max(ending_start - ROW_INCREMENT, 0)
            ending_label_end = ending_label_start + len(ending_ids)
            labels[:ending_label_start] = [self.pad_token_id] * ending_label_start
            labels[ending_label_end:] = [self.pad_token_id] * (
                len(labels) - ending_label_end
            )
            if self.data_state is not None:
                self.data_state.token_buffer = []

            yield {
                "input_ids": [int(token_id) for token_id in padded_tokens[:-1]],
                "labels": [int(token_id) for token_id in labels],
            }


def build_swag_batches(
    fine_tune_config: SwagFineTuneConfig,
    tokenizer: object,
    epoch: int,
    pad_token_id: int = DEFAULT_PAD_TOKEN_ID,
    bos_token_id: int | None = DEFAULT_BOS_TOKEN_ID,
    eos_token_id: int | None = DEFAULT_EOS_TOKEN_ID,
    progress: ReadingProgress | None = None,
    data_state: TrainingDataState | None = None,
) -> Iterator[dict[str, object]]:
    """
    Stream fixed-length SWAG examples in MLX batches with context labels masked.
    """
    examples = SwagExampleDataset(
        examples=iter_swag_parts(
            fine_tune_config,
            epoch=epoch,
            progress=progress,
            data_state=data_state,
        ),
        tokenizer=tokenizer,
        sequence_length=fine_tune_config.sequence_length,
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        data_state=data_state,
    )
    return iter_mlx_batches(examples, batch_size=fine_tune_config.batch_size)


def build_swag_dataloader(*args, **kwargs):
    """
    Backward-compatible alias for callers that still use the old helper name.
    """
    return build_swag_batches(*args, **kwargs)


def load_pretrained_model_config(checkpoint_path: Path):
    """
    Read ``model_config`` from MLX checkpoint metadata without loading weights.
    """
    from sml import SMLConfig

    metadata = load_checkpoint_metadata(checkpoint_path)
    model_config = metadata.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("Checkpoint is missing model_config")
    return SMLConfig(**normalize_model_config(model_config))


def load_pretrained_weights(model, checkpoint_path: Path) -> None:
    """
    Initialize model parameters from a pretrained MLX checkpoint directory.
    """
    import mlx.core as mx

    checkpoint_dir = resolve_path(checkpoint_path)
    model.load_weights(str(checkpoint_dir / MODEL_WEIGHTS_NAME))
    mx.eval(model.parameters())


def prepare_lora_model(
    model,
    fine_tune_config: SwagFineTuneConfig,
) -> None:
    """
    Attach LoRA adapters after base weights are loaded and freeze base parameters.
    """
    apply_lora(model, fine_tune_config.lora)
    require_lora_modules(model, fine_tune_config.lora.target_modules)


def _json_ready(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def build_merged_model(model):
    """
    Copy the fine-tuning model and merge adapters for inference-compatible weights.
    """
    merged_model = copy.deepcopy(model)
    merge_lora(merged_model)
    return merged_model


def save_lora_checkpoint(
    path: Path,
    model,
    optimizer,
    model_config,
    fine_tune_config: SwagFineTuneConfig,
    step: int,
    data_state: TrainingDataState | None = None,
) -> None:
    """
    Persist merged weights, LoRA adapters, optimizer state, and training metadata.
    """
    import mlx.core as mx
    from mlx.utils import tree_flatten

    path.mkdir(parents=True, exist_ok=True)
    merged_model = build_merged_model(model)
    merged_model.save_weights(str(path / MODEL_WEIGHTS_NAME))
    mx.savez(str(path / LORA_STATE_NAME), **lora_state_dict(model))
    optimizer_state = tree_flatten(optimizer.state, destination={})
    mx.savez(str(path / OPTIMIZER_STATE_NAME), **optimizer_state)
    metadata = {
        "step": step,
        "model_config": _json_ready(asdict(model_config)),
        "training_config": _json_ready(asdict(fine_tune_config)),
        "lora_config": _json_ready(asdict(fine_tune_config.lora)),
        "pretrained_checkpoint_path": str(
            resolve_path(fine_tune_config.pretrained_checkpoint_path)
        ),
        "data_state": None if data_state is None else _json_ready(asdict(data_state)),
        "stochastic_resume": "not_guaranteed",
        "resume_note": STOCHASTIC_RESUME_NOTE,
    }
    with (path / METADATA_NAME).open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2, sort_keys=True)


def load_lora_checkpoint(
    path: Path,
    model,
    optimizer,
) -> TrainingResumeState:
    """
    Restore LoRA adapters, optimizer state, and data position from a checkpoint.
    """
    import mlx.core as mx
    from mlx.utils import tree_unflatten

    checkpoint_path = resolve_path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    metadata_path = checkpoint_path / METADATA_NAME
    lora_path = checkpoint_path / LORA_STATE_NAME
    optimizer_path = checkpoint_path / OPTIMIZER_STATE_NAME
    for required_path in (metadata_path, lora_path, optimizer_path):
        if not required_path.exists():
            raise ValueError(f"Checkpoint is missing {required_path.name}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"Checkpoint metadata must be a dictionary: {metadata_path}")
    step = metadata.get("step")
    if not isinstance(step, int):
        raise ValueError("Checkpoint metadata is missing step")

    lora_arrays = mx.load(str(lora_path))
    optimizer_arrays = mx.load(str(optimizer_path))
    if not isinstance(lora_arrays, dict):
        raise ValueError("LoRA checkpoint must contain a state dictionary")
    if not isinstance(optimizer_arrays, dict):
        raise ValueError("Optimizer checkpoint must contain a state dictionary")

    load_lora_state_dict(model, lora_arrays)
    optimizer.state = tree_unflatten(optimizer_arrays)
    mx.eval(model.parameters(), optimizer.state)
    data_state = parse_checkpoint_data_state(metadata.get("data_state"))
    return TrainingResumeState(step=step, data_state=data_state)


def fine_tune_swag(
    fine_tune_config: SwagFineTuneConfig | None = None,
    resume_from_checkpoint: bool = False,
) -> Path:
    """
    LoRA fine-tune ``pretrained_checkpoint_path`` on SWAG train examples.

    Fresh runs load frozen base weights, attach adapters, and write
    ``output_dir / checkpoint_name``. ``--resume`` continues from the LoRA
    checkpoint, restoring adapters, optimizer state, and data position.
    """
    fine_tune_config = (
        SwagFineTuneConfig() if fine_tune_config is None else fine_tune_config
    )

    output_dir = resolve_path(fine_tune_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / fine_tune_config.checkpoint_name

    if resume_from_checkpoint:
        pretrained_path = resolve_path(fine_tune_config.pretrained_checkpoint_path)
        config_checkpoint_path = checkpoint_path
    else:
        pretrained_path = resolve_path(fine_tune_config.pretrained_checkpoint_path)
        if not pretrained_path.exists():
            raise FileNotFoundError(
                f"Pretrained checkpoint does not exist: {pretrained_path}"
            )
        config_checkpoint_path = pretrained_path

    set_seed(fine_tune_config.seed)
    tokenizer = load_tokenizer(fine_tune_config.tokenizer_model_path)
    base_model_config = load_pretrained_model_config(config_checkpoint_path)
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from sml import SMLConfig, SMLLanguageModel, count_parameters

    checkpoint_model_config = replace(
        base_model_config,
        vocab_size=tokenizer.get_piece_size(),
        rope_scaling_factor=SMLConfig().rope_scaling_factor,
        original_max_position_embeddings=max(
            base_model_config.original_max_position_embeddings,
            fine_tune_config.sequence_length,
        ),
    )
    model = SMLLanguageModel(checkpoint_model_config)
    load_pretrained_weights(model, pretrained_path)
    prepare_lora_model(model, fine_tune_config)
    apply_model_dtype(model, fine_tune_config.autocast_dtype)
    total_params, trainable_params = count_parameters(model)

    lr_schedule = build_lr_schedule(
        learning_rate=fine_tune_config.learning_rate,
        total_steps=resolve_lr_total_steps(fine_tune_config),
        warmup_steps=fine_tune_config.warmup_steps,
        min_lr_ratio=fine_tune_config.min_lr_ratio,
    )
    optimizer = optim.AdamW(
        learning_rate=lr_schedule,
        weight_decay=fine_tune_config.weight_decay,
    )
    optimizer.init(model.trainable_parameters())
    if not lora_parameters(model):
        require_lora_modules(model, fine_tune_config.lora.target_modules)

    resume_state = TrainingResumeState()
    if resume_from_checkpoint:
        resume_state = load_lora_checkpoint(
            checkpoint_path,
            model,
            optimizer,
        )
        apply_model_dtype(model, fine_tune_config.autocast_dtype)

    global_step = resume_state.step
    data_state = resume_state.data_state or TrainingDataState()
    legacy_batches_to_skip = (
        count_resume_batches(global_step, fine_tune_config)
        if resume_from_checkpoint and resume_state.data_state is None
        else 0
    )
    resume_progress = ResumeProgress(batches_to_skip=legacy_batches_to_skip)
    model.train()
    micro_step = 0
    loss_sum = 0.0
    accumulated_grads = None
    reading_progress = ReadingProgress()
    loss_and_grad = nn.value_and_grad(model, _loss_fn(model))

    print(f"Dataset: {fine_tune_config.dataset_name}/{fine_tune_config.dataset_config}")
    print(f"Split: {fine_tune_config.dataset_split}")
    print(f"LoRA rank: {fine_tune_config.lora.rank}")
    print(f"LoRA alpha: {fine_tune_config.lora.alpha}")
    print(f"Tokenizer vocab: {checkpoint_model_config.vocab_size:,}")
    print("Device: mlx")
    print(f"Compute dtype: {fine_tune_config.autocast_dtype}")
    print(f"Parameters: total={total_params:,} trainable={trainable_params:,}")

    for epoch in range(data_state.epoch, fine_tune_config.epochs):
        if epoch != data_state.epoch:
            reset_training_data_state(data_state, epoch)
        batches = build_swag_batches(
            fine_tune_config=fine_tune_config,
            tokenizer=tokenizer,
            epoch=epoch,
            pad_token_id=checkpoint_model_config.pad_token_id,
            bos_token_id=checkpoint_model_config.bos_token_id,
            eos_token_id=checkpoint_model_config.eos_token_id,
            progress=reading_progress,
            data_state=data_state,
        )
        for batch in iter_unseen_batches(batches, resume_progress):
            loss, grads = loss_and_grad(batch["input_ids"], batch["labels"])
            mx.eval(loss, grads)
            loss_sum += float(loss.item())
            micro_step += ROW_INCREMENT

            scaled_grads = tree_scale(
                grads,
                1.0 / fine_tune_config.gradient_accumulation_steps,
            )
            accumulated_grads = (
                scaled_grads
                if accumulated_grads is None
                else tree_add(accumulated_grads, scaled_grads)
            )

            if micro_step % fine_tune_config.gradient_accumulation_steps != 0:
                continue

            clipped_grads, grad_norm = clip_gradients_by_global_norm(
                accumulated_grads,
                fine_tune_config.max_grad_norm,
            )
            optimizer.update(model, clipped_grads)
            mx.eval(model.parameters(), optimizer.state)
            global_step += ROW_INCREMENT
            accumulated_grads = None

            if global_step % fine_tune_config.log_every == 0 or global_step == 1:
                avg_loss = loss_sum / fine_tune_config.gradient_accumulation_steps
                lr = float(optimizer.learning_rate.item())
                print(
                    format_training_log(
                        epoch=epoch + 1,
                        global_step=global_step,
                        lr=lr,
                        avg_loss=avg_loss,
                        grad_norm=float(grad_norm.item()),
                        timestamp=datetime.now(),
                        progress=reading_progress,
                    )
                )
            loss_sum = 0.0

            if (
                fine_tune_config.save_every > 0
                and global_step % fine_tune_config.save_every == 0
            ):
                save_lora_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    checkpoint_model_config,
                    fine_tune_config,
                    global_step,
                    data_state=data_state,
                )

            if is_step_limit_reached(global_step, fine_tune_config.max_steps):
                save_lora_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    checkpoint_model_config,
                    fine_tune_config,
                    global_step,
                    data_state=data_state,
                )
                return checkpoint_path

    save_lora_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        checkpoint_model_config,
        fine_tune_config,
        global_step,
        data_state=data_state,
    )
    return checkpoint_path


def _loss_fn(model):
    def loss_fn(input_ids, labels):
        output = model(input_ids, labels=labels)
        if output.loss is None:
            raise RuntimeError("Model did not return a training loss")
        return output.loss

    return loss_fn


def build_parser() -> argparse.ArgumentParser:
    """
    Keep CLI surface narrow; fine-tuning configuration defaults live in SwagFineTuneConfig.
    """
    parser = argparse.ArgumentParser(
        description="LoRA fine-tune the SML language model on SWAG train examples.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"pretrained model checkpoint directory (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--tokenizer-model",
        type=Path,
        default=DEFAULT_TOKENIZER_MODEL_PATH,
        help=f"SentencePiece model path (default: {DEFAULT_TOKENIZER_MODEL_PATH})",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from output_dir/checkpoint_name. Restores model adapters, "
            "optimizer state, and data position, then continues from the saved step. "
            "Stochastic continuity is not guaranteed."
        ),
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """
    Parse the small CLI surface, then delegate fine-tuning to the config-based API.
    """
    args = parse_args(argv)
    checkpoint_path = fine_tune_swag(
        fine_tune_config=SwagFineTuneConfig(
            pretrained_checkpoint_path=args.model,
            tokenizer_model_path=args.tokenizer_model,
        ),
        resume_from_checkpoint=args.resume,
    )
    print(f"Checkpoint: {checkpoint_path}")
    return SUCCESS_RETURN_CODE


if __name__ == "__main__":
    raise SystemExit(main())
