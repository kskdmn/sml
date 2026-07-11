from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Protocol, Sequence, TypeVar

import sentencepiece as spm

from config import PROJECT_DIR, resolve_path
from tokenizer import (
    HIDDEN_FILE_PREFIX,
    TEXT_COLUMN,
    filter_text,
    iter_jsonl_records,
)


ROW_INCREMENT = 1
SUCCESS_RETURN_CODE = 0
INPUT_DIR = Path("~/Documents/data-common_pile/")
INPUT_FILE_NAME_REGEX = r".*-00[0-9][0-9]\.jsonl\.zst\Z"
OUTPUT_DIR = PROJECT_DIR / "output"
DEFAULT_MODEL_PATH = OUTPUT_DIR / "sml"
DEFAULT_TOKENIZER_MODEL_PATH = OUTPUT_DIR / "bpe_tokenizer.model"
MODEL_WEIGHTS_NAME = "model.safetensors"
OPTIMIZER_STATE_NAME = "optimizer.npz"
METADATA_NAME = "metadata.json"
STOCHASTIC_RESUME_NOTE = (
    "Resume restores model weights, optimizer state, and data position; "
    "stochastic continuity is not guaranteed."
)
BatchT = TypeVar("BatchT")
__all__ = [
    "METADATA_NAME",
    "MODEL_WEIGHTS_NAME",
    "OPTIMIZER_STATE_NAME",
    "SUCCESS_RETURN_CODE",
    "ROW_INCREMENT",
    "ReadingProgress",
    "ResumeProgress",
    "TextTokenizer",
    "TrainingConfig",
    "TrainingDataState",
    "TrainingResumeState",
    "apply_model_dtype",
    "build_parser",
    "build_lr_schedule",
    "clip_gradients_by_global_norm",
    "count_resume_batches",
    "discover_input_files",
    "format_training_log",
    "get_special_token_id",
    "global_grad_norm",
    "is_step_limit_reached",
    "iter_mlx_batches",
    "iter_mlx_token_blocks",
    "iter_texts",
    "iter_unseen_batches",
    "load_tokenizer",
    "load_training_checkpoint",
    "lr_lambda",
    "main",
    "model_config_for_training",
    "parse_args",
    "parse_checkpoint_data_state",
    "parse_checkpoint_input_files",
    "reset_training_data_state",
    "resolve_compute_dtype",
    "resolve_lr_total_steps",
    "resolve_mlx_checkpoint_path",
    "save_checkpoint",
    "set_seed",
    "shuffle_input_files",
    "train_model",
    "tree_add",
    "tree_scale",
]


@dataclass(slots=True)
class TrainingConfig:
    """
    Training hyperparameters and I/O paths.

    ``train_sml.py`` reads these defaults from code; the CLI only exposes
    ``--resume``. Edit fields here, or pass a custom instance to
    ``train_sml.train_model``, instead of adding CLI flags.
    """

    input_dir: Path = INPUT_DIR
    input_file_name_regex: str = INPUT_FILE_NAME_REGEX
    output_dir: Path = OUTPUT_DIR
    model_path: Path | None = None
    tokenizer_model_path: Path = DEFAULT_TOKENIZER_MODEL_PATH
    checkpoint_name: str = "sml"
    sequence_length: int = 1_024
    batch_size: int = 1
    max_steps: int | None = None
    lr_total_steps: int | None = 100_000
    epochs: int = 1
    max_rows_per_file: int | None = 32_768
    shuffle_input_files: bool = True
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0
    warmup_steps: int = int(
        lr_total_steps if lr_total_steps is not None else 100 * 0.01
    )
    min_lr_ratio: float = 0.1
    log_every: int = 10
    save_every: int = 1_000
    seed: int = 42
    autocast_dtype: str = "bfloat16"


def model_config_for_training(config):
    """
    Return a copy that trains with standard RoPE inside ``sequence_length``.

    YaRN is applied only when checkpoints are loaded for inference.
    """
    return replace(config, rope_scaling_factor=1.0)


@dataclass(slots=True)
class ReadingProgress:
    input_file: str | None = None
    line_number: int | None = None
    example_index: int | None = None


@dataclass(slots=True)
class ResumeProgress:
    batches_to_skip: int = 0


@dataclass(slots=True)
class TrainingDataState:
    epoch: int = 0
    input_file_index: int = 0
    line_number: int | None = None
    token_buffer: list[int] = field(default_factory=list)


@dataclass(slots=True)
class TrainingResumeState:
    step: int = 0
    input_files: tuple[Path, ...] = ()
    data_state: TrainingDataState | None = None


class TextTokenizer(Protocol):
    def encode(self, text: str, out_type: type = int) -> list[int]:
        """
        Training only needs SentencePiece-style integer encoding, which keeps the
        protocol narrow enough for lightweight tests.
        """
        ...


def discover_input_files(
    input_dir: Path,
    file_name_regex: str,
) -> tuple[Path, ...]:
    """
    Match the regex against file names rather than paths, and skip hidden files such as
    local filesystem metadata.
    """
    root = resolve_path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {root}")

    pattern = re.compile(file_name_regex)
    files = [
        path
        for path in root.iterdir()
        if path.is_file()
        and not path.name.startswith(HIDDEN_FILE_PREFIX)
        and pattern.fullmatch(path.name) is not None
    ]
    return tuple(sorted(files, key=lambda path: path.name))


def shuffle_input_files(input_files: Iterable[Path], seed: int) -> tuple[Path, ...]:
    shuffled_files = list(input_files)
    random.Random(seed).shuffle(shuffled_files)
    return tuple(shuffled_files)


def iter_texts(
    input_files: Iterable[Path],
    max_rows_per_file: int | None,
    progress: ReadingProgress | None = None,
    data_state: TrainingDataState | None = None,
) -> Iterator[str]:
    """
    Reuse tokenizer-training text filters so tokenizer and model training agree on which
    rows are usable.
    """
    start_file_index = 0 if data_state is None else data_state.input_file_index
    for input_file_index, input_file in enumerate(input_files):
        if input_file_index < start_file_index:
            continue
        start_after_line = (
            data_state.line_number
            if data_state is not None and input_file_index == start_file_index
            else None
        )
        for row, line_number in iter_jsonl_records(
            input_file,
            max_rows_per_file,
            start_after_line=start_after_line,
        ):
            if progress is not None:
                progress.input_file = input_file.name
                progress.line_number = line_number
            if data_state is not None:
                data_state.input_file_index = input_file_index
                data_state.line_number = line_number
            text = filter_text(row.get(TEXT_COLUMN))
            if text is not None:
                yield text


def get_special_token_id(
    tokenizer: object,
    name: str,
    fallback: int | None,
) -> int | None:
    """
    SentencePiece reports negative IDs for disabled special tokens, so config fallbacks
    are used in that case.
    """
    value = getattr(tokenizer, name, None)
    if callable(value):
        value = value()
    if value is None or value < 0:
        return fallback
    return int(value)


def load_tokenizer(path: Path) -> spm.SentencePieceProcessor:
    """
    Fail before training starts if the configured SentencePiece model path is missing.
    """
    model_path = resolve_path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"Tokenizer model does not exist: {model_path}")
    return spm.SentencePieceProcessor(model_file=str(model_path))


def parse_checkpoint_input_files(input_files: object) -> tuple[Path, ...]:
    if input_files is None:
        return ()
    if not isinstance(input_files, (list, tuple)):
        raise ValueError("Checkpoint input_files must be a list")
    return tuple(resolve_path(Path(str(input_file))) for input_file in input_files)


def parse_checkpoint_data_state(data_state: object) -> TrainingDataState | None:
    if data_state is None:
        return None
    if not isinstance(data_state, dict):
        raise ValueError("Checkpoint data_state must be a dictionary")

    token_buffer = data_state.get("token_buffer", [])
    if not isinstance(token_buffer, list):
        raise ValueError("Checkpoint data_state token_buffer must be a list")
    line_number = data_state.get("line_number")
    if line_number is not None and not isinstance(line_number, int):
        raise ValueError("Checkpoint data_state line_number must be an integer")
    return TrainingDataState(
        epoch=int(data_state.get("epoch", 0)),
        input_file_index=int(data_state.get("input_file_index", 0)),
        line_number=line_number,
        token_buffer=[int(token_id) for token_id in token_buffer],
    )


def reset_training_data_state(data_state: TrainingDataState, epoch: int) -> None:
    data_state.epoch = epoch
    data_state.input_file_index = 0
    data_state.line_number = None
    data_state.token_buffer = []


def count_resume_batches(global_step: int, training_config: TrainingConfig) -> int:
    """
    A checkpoint step represents completed optimizer steps; each completed step consumed
    one gradient-accumulation window of streamed batches.
    """
    return global_step * training_config.gradient_accumulation_steps


def iter_unseen_batches(
    dataloader: Iterable[BatchT],
    progress: ResumeProgress,
) -> Iterator[BatchT]:
    """
    Rebuild the deterministic input stream and discard batches already reflected in the
    loaded optimizer step.
    """
    for batch in dataloader:
        if progress.batches_to_skip > 0:
            progress.batches_to_skip -= ROW_INCREMENT
            continue
        yield batch


def is_step_limit_reached(global_step: int, max_steps: int | None) -> bool:
    """
    A None limit means data and epoch boundaries, not optimizer steps, decide when
    training stops.
    """
    return max_steps is not None and global_step >= max_steps


def resolve_lr_total_steps(training_config: TrainingConfig) -> int | None:
    """
    Prefer an explicit schedule horizon, but fall back to max_steps so finite runs decay
    as expected.
    """
    if training_config.lr_total_steps is not None:
        return training_config.lr_total_steps
    if training_config.max_steps is not None:
        return training_config.max_steps
    return None


def format_training_log(
    epoch: int,
    global_step: int,
    lr: float,
    avg_loss: float,
    grad_norm: float,
    timestamp: datetime,
    progress: ReadingProgress | None = None,
) -> str:
    """
    Log gradient norm before clipping so exploding gradients remain visible even when
    clipping succeeds.
    """
    parts = [
        f"time={timestamp:%Y-%m-%d %H:%M:%S}",
        f"epoch={epoch}",
        f"step={global_step}",
    ]
    if progress is not None and progress.input_file is not None:
        parts.append(f"input={progress.input_file}")
    if progress is not None and progress.line_number is not None:
        parts.append(f"line={progress.line_number}")
    parts.extend(
        [
            f"lr={lr:.3e}",
            f"loss={avg_loss:.4f}",
            f"grad_norm={grad_norm:.3f} (before clipping)",
        ]
    )
    return " ".join(parts)


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


def _mlx_core():
    import mlx.core as mx

    return mx


def _mlx_nn():
    import mlx.nn as nn

    return nn


def _mlx_optimizers():
    import mlx.optimizers as optim

    return optim


def _mlx_tree_utils():
    from mlx.utils import tree_flatten, tree_map, tree_unflatten

    return tree_flatten, tree_map, tree_unflatten


def _model_modules():
    from sml import SMLConfig, SMLLanguageModel, count_parameters

    return SMLConfig, SMLLanguageModel, count_parameters


def set_seed(seed: int) -> None:
    random.seed(seed)
    mx = _mlx_core()
    mx.random.seed(seed)


def resolve_compute_dtype(name: str):
    """
    Map training dtype names to MLX dtypes.

    ``none`` keeps float32 weights; otherwise parameters are cast before training.
    """
    mx = _mlx_core()
    if name == "none":
        return None
    if name == "bfloat16":
        return mx.bfloat16
    if name == "float16":
        return mx.float16
    raise ValueError(f"Unsupported compute dtype: {name}")


def _retie_embeddings_if_needed(model) -> None:
    if model.config.tie_word_embeddings:
        model.lm_head.weight = model.embed_tokens.weight


def apply_model_dtype(model, autocast_dtype: str) -> None:
    """
    Cast model parameters to the configured MLX dtype and restore tied embeddings.
    """
    dtype = resolve_compute_dtype(autocast_dtype)
    if dtype is None:
        return

    mx = _mlx_core()
    _, tree_map, _ = _mlx_tree_utils()

    def cast_array(value: object) -> object:
        if isinstance(value, mx.array):
            return value.astype(dtype)
        return value

    model.update(tree_map(cast_array, model.parameters()))
    _retie_embeddings_if_needed(model)
    mx.eval(model.parameters())


def iter_mlx_token_blocks(
    texts: Iterable[str],
    tokenizer,
    sequence_length: int,
    stride: int | None = None,
    bos_token_id: int | None = None,
    eos_token_id: int | None = None,
    data_state: TrainingDataState | None = None,
) -> Iterator[dict[str, list[int]]]:
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")

    buffer = [] if data_state is None else list(data_state.token_buffer)
    tokens_per_block = sequence_length + 1
    stride = sequence_length if stride is None else stride
    bos_token_id = get_special_token_id(tokenizer, "bos_id", bos_token_id)
    eos_token_id = get_special_token_id(tokenizer, "eos_id", eos_token_id)

    def iter_ready_blocks() -> Iterator[dict[str, list[int]]]:
        while len(buffer) >= tokens_per_block:
            block = buffer[:tokens_per_block]
            del buffer[:stride]
            if data_state is not None:
                data_state.token_buffer = list(buffer)
            yield {
                "input_ids": [int(token_id) for token_id in block[:-1]],
                "labels": [int(token_id) for token_id in block[1:]],
            }

    yield from iter_ready_blocks()
    for text in texts:
        token_ids = tokenizer.encode(text, out_type=int)
        if bos_token_id is not None:
            buffer.append(bos_token_id)
        buffer.extend(int(token_id) for token_id in token_ids)
        if eos_token_id is not None:
            buffer.append(eos_token_id)
        if data_state is not None:
            data_state.token_buffer = list(buffer)
        yield from iter_ready_blocks()


def iter_mlx_batches(
    examples: Iterable[dict[str, object]],
    batch_size: int,
) -> Iterator[dict[str, object]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    batch: list[dict[str, object]] = []
    for example in examples:
        batch.append(example)
        if len(batch) == batch_size:
            yield _collate_mlx_batch(batch)
            batch = []
    if batch:
        yield _collate_mlx_batch(batch)


def _as_token_list(value: object) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        raise TypeError("token values must be list-like")
    return [int(token_id) for token_id in value]


def _collate_mlx_batch(batch: list[dict[str, object]]) -> dict[str, object]:
    mx = _mlx_core()
    input_ids = [_as_token_list(example["input_ids"]) for example in batch]
    labels = [_as_token_list(example["labels"]) for example in batch]
    return {
        "input_ids": mx.array(input_ids, dtype=mx.int32),
        "labels": mx.array(labels, dtype=mx.int32),
    }


def build_lr_schedule(
    learning_rate: float,
    total_steps: int | None,
    warmup_steps: int,
    min_lr_ratio: float,
):
    mx = _mlx_core()

    def schedule(step):
        warmup_denominator = float(max(1, warmup_steps))
        warmup_multiplier = (step + 1) / warmup_denominator
        if total_steps is None:
            decay_multiplier = mx.array(1.0)
        else:
            progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            progress = mx.minimum(mx.array(1.0), progress)
            cosine = 0.5 * (1.0 + mx.cos(math.pi * progress))
            decay_multiplier = mx.maximum(mx.array(min_lr_ratio), cosine)
        multiplier = mx.where(step < warmup_steps, warmup_multiplier, decay_multiplier)
        return learning_rate * multiplier

    return schedule


def tree_add(left: dict, right: dict) -> dict:
    _, tree_map, _ = _mlx_tree_utils()
    return tree_map(lambda a, b: a + b, left, right)


def tree_scale(tree: dict, scale: float | object) -> dict:
    _, tree_map, _ = _mlx_tree_utils()
    return tree_map(lambda value: value * scale, tree)


def global_grad_norm(grads: dict):
    mx = _mlx_core()
    tree_flatten, _, _ = _mlx_tree_utils()
    total = mx.array(0.0)
    for _, grad in tree_flatten(grads):
        total = total + mx.sum(grad.astype(mx.float32) * grad.astype(mx.float32))
    return mx.sqrt(total)


def clip_gradients_by_global_norm(
    grads: dict,
    max_norm: float,
) -> tuple[dict, object]:
    mx = _mlx_core()
    grad_norm = global_grad_norm(grads)
    scale = mx.where(
        grad_norm > max_norm,
        mx.array(max_norm) / mx.maximum(grad_norm, mx.array(1e-12)),
        mx.array(1.0),
    )
    return tree_scale(grads, scale), grad_norm


def resolve_mlx_checkpoint_path(training_config: TrainingConfig) -> Path:
    checkpoint_path = (
        Path(training_config.model_path)
        if training_config.model_path is not None
        else resolve_path(training_config.output_dir) / training_config.checkpoint_name
    )
    return resolve_path(checkpoint_path)


def _json_ready(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def save_checkpoint(
    checkpoint_path: Path,
    model,
    optimizer,
    model_config,
    training_config: TrainingConfig,
    step: int,
    input_files: Iterable[Path] = (),
    data_state: TrainingDataState | None = None,
) -> None:
    mx = _mlx_core()
    tree_flatten, _, _ = _mlx_tree_utils()
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(checkpoint_path / MODEL_WEIGHTS_NAME))
    optimizer_state = tree_flatten(optimizer.state, destination={})
    mx.savez(str(checkpoint_path / OPTIMIZER_STATE_NAME), **optimizer_state)
    metadata = {
        "step": step,
        "model_config": _json_ready(asdict(model_config)),
        "training_config": _json_ready(asdict(training_config)),
        "input_files": [str(input_file) for input_file in input_files],
        "data_state": None if data_state is None else _json_ready(asdict(data_state)),
        "stochastic_resume": "not_guaranteed",
        "resume_note": STOCHASTIC_RESUME_NOTE,
    }
    with (checkpoint_path / METADATA_NAME).open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2, sort_keys=True)


def load_training_checkpoint(
    checkpoint_path: Path,
    model,
    optimizer,
) -> TrainingResumeState:
    checkpoint_path = resolve_path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    mx = _mlx_core()
    _, _, tree_unflatten = _mlx_tree_utils()
    weights_path = checkpoint_path / MODEL_WEIGHTS_NAME
    optimizer_path = checkpoint_path / OPTIMIZER_STATE_NAME
    metadata_path = checkpoint_path / METADATA_NAME
    for required_path in (weights_path, optimizer_path, metadata_path):
        if not required_path.exists():
            raise ValueError(f"Checkpoint is missing {required_path.name}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"Checkpoint metadata must be a dictionary: {metadata_path}")
    step = metadata.get("step")
    if not isinstance(step, int):
        raise ValueError("Checkpoint metadata is missing step")

    model.load_weights(str(weights_path))
    optimizer_arrays = mx.load(str(optimizer_path))
    if not isinstance(optimizer_arrays, dict):
        raise ValueError("Optimizer checkpoint must contain a state dictionary")
    optimizer.state = tree_unflatten(optimizer_arrays)
    mx.eval(model.parameters(), optimizer.state)

    input_files = metadata.get("input_files", ())
    if not isinstance(input_files, list):
        raise ValueError("Checkpoint metadata input_files must be a list")
    data_state = parse_checkpoint_data_state(metadata.get("data_state"))
    return TrainingResumeState(
        step=step,
        input_files=tuple(
            resolve_path(Path(str(input_file))) for input_file in input_files
        ),
        data_state=data_state,
    )


def train_model(
    training_config: TrainingConfig | None = None,
    model_config=None,
    resume_from_checkpoint: bool = False,
) -> Path:
    training_config = TrainingConfig() if training_config is None else training_config
    SMLConfig, SMLLanguageModel, count_parameters = _model_modules()
    base_model_config = SMLConfig() if model_config is None else model_config

    mx = _mlx_core()
    nn = _mlx_nn()
    optim = _mlx_optimizers()

    set_seed(training_config.seed)
    output_dir = resolve_path(training_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = discover_input_files(
        training_config.input_dir,
        training_config.input_file_name_regex,
    )
    if not input_files:
        raise FileNotFoundError(
            f"No supported input files found in {resolve_path(training_config.input_dir)}"
        )
    if training_config.shuffle_input_files:
        input_files = shuffle_input_files(input_files, seed=training_config.seed)

    tokenizer = load_tokenizer(training_config.tokenizer_model_path)
    checkpoint_model_config = replace(
        base_model_config,
        vocab_size=tokenizer.get_piece_size(),
        original_max_position_embeddings=max(
            base_model_config.original_max_position_embeddings,
            training_config.sequence_length,
        ),
    )
    training_model_config = model_config_for_training(checkpoint_model_config)
    model = SMLLanguageModel(training_model_config)
    apply_model_dtype(model, training_config.autocast_dtype)
    model.train()
    mx.eval(model.parameters())
    total_params, trainable_params = count_parameters(model)

    lr_schedule = build_lr_schedule(
        learning_rate=training_config.learning_rate,
        total_steps=resolve_lr_total_steps(training_config),
        warmup_steps=training_config.warmup_steps,
        min_lr_ratio=training_config.min_lr_ratio,
    )
    optimizer = optim.AdamW(
        learning_rate=lr_schedule,
        weight_decay=training_config.weight_decay,
    )
    optimizer.init(model.trainable_parameters())

    checkpoint_path = resolve_mlx_checkpoint_path(training_config)
    resume_state = TrainingResumeState()
    if resume_from_checkpoint:
        resume_state = load_training_checkpoint(checkpoint_path, model, optimizer)
        apply_model_dtype(model, training_config.autocast_dtype)
        if resume_state.input_files:
            input_files = resume_state.input_files

    global_step = resume_state.step
    data_state = resume_state.data_state or TrainingDataState()
    legacy_batches_to_skip = (
        count_resume_batches(global_step, training_config)
        if resume_from_checkpoint and resume_state.data_state is None
        else 0
    )
    resume_progress = ResumeProgress(batches_to_skip=legacy_batches_to_skip)
    micro_step = 0
    loss_sum = 0.0
    accumulated_grads = None
    reading_progress = ReadingProgress()
    loss_and_grad = nn.value_and_grad(model, _loss_fn(model))

    print(f"Input files: {len(input_files)}")
    print(f"Tokenizer vocab: {checkpoint_model_config.vocab_size:,}")
    print("Backend: mlx")
    print(f"Compute dtype: {training_config.autocast_dtype}")
    print(f"Parameters: total={total_params:,} trainable={trainable_params:,}")

    for epoch in range(data_state.epoch, training_config.epochs):
        if epoch != data_state.epoch:
            reset_training_data_state(data_state, epoch)
        blocks = iter_mlx_token_blocks(
            texts=iter_texts(
                input_files,
                max_rows_per_file=training_config.max_rows_per_file,
                progress=reading_progress,
                data_state=data_state,
            ),
            tokenizer=tokenizer,
            sequence_length=training_config.sequence_length,
            data_state=data_state,
        )
        batches = iter_mlx_batches(blocks, batch_size=training_config.batch_size)
        for batch in iter_unseen_batches(batches, resume_progress):
            loss, grads = loss_and_grad(batch["input_ids"], batch["labels"])
            mx.eval(loss, grads)
            loss_sum += float(loss.item())
            micro_step += ROW_INCREMENT

            scaled_grads = tree_scale(
                grads,
                1.0 / training_config.gradient_accumulation_steps,
            )
            accumulated_grads = (
                scaled_grads
                if accumulated_grads is None
                else tree_add(accumulated_grads, scaled_grads)
            )

            if micro_step % training_config.gradient_accumulation_steps != 0:
                continue

            clipped_grads, grad_norm = clip_gradients_by_global_norm(
                accumulated_grads,
                training_config.max_grad_norm,
            )
            optimizer.update(model, clipped_grads)
            _retie_embeddings_if_needed(model)
            mx.eval(model.parameters(), optimizer.state)
            global_step += ROW_INCREMENT
            accumulated_grads = None

            if global_step % training_config.log_every == 0 or global_step == 1:
                avg_loss = loss_sum / training_config.gradient_accumulation_steps
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
                training_config.save_every > 0
                and global_step % training_config.save_every == 0
            ):
                save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    checkpoint_model_config,
                    training_config,
                    global_step,
                    input_files=input_files,
                    data_state=data_state,
                )

            if is_step_limit_reached(global_step, training_config.max_steps):
                save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    checkpoint_model_config,
                    training_config,
                    global_step,
                    input_files=input_files,
                    data_state=data_state,
                )
                return checkpoint_path

    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        checkpoint_model_config,
        training_config,
        global_step,
        input_files=input_files,
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
    parser = argparse.ArgumentParser(description="Train the MLX SML language model.")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"model checkpoint directory (default: {DEFAULT_MODEL_PATH})",
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
            "Resume from the MLX checkpoint directory derived from "
            "output_dir/checkpoint_name. The checkpoint must already exist. "
            "Stochastic continuity is not guaranteed."
        ),
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    checkpoint_path = train_model(
        training_config=TrainingConfig(
            model_path=args.model,
            tokenizer_model_path=args.tokenizer_model,
        ),
        resume_from_checkpoint=args.resume,
    )
    print(f"Checkpoint: {checkpoint_path}")
    return SUCCESS_RETURN_CODE


if __name__ == "__main__":
    raise SystemExit(main())
