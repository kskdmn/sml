from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Protocol, Sequence, TypeVar

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map, tree_unflatten

from config import (
    DEFAULT_MODEL_PATH,
    DEFAULT_TOKENIZER_MODEL_PATH,
    INPUT_DIR,
    INPUT_FILE_NAME_REGEX,
    METADATA_NAME,
    MODEL_WEIGHTS_NAME,
    OPTIMIZER_STATE_NAME,
    OUTPUT_DIR,
    SUCCESS_RETURN_CODE,
    resolve_path,
)
from sml import SMLConfig, SMLLanguageModel, count_parameters
from utils import (
    TEXT_COLUMN,
    build_loss_fn,
    build_lr_schedule,
    discover_input_files,
    filter_text,
    get_special_token_id,
    iter_jsonl_records,
    json_ready,
    load_tokenizer,
    set_seed,
    shuffle_input_files,
)


STOCHASTIC_RESUME_NOTE = (
    "Resume restores model weights, optimizer state, and data position; "
    "stochastic continuity is not guaranteed."
)
BatchT = TypeVar("BatchT")


@dataclass(slots=True)
class ParameterWeightDecayConfig:
    """
    Per-parameter-type weight decay values.

    ``lm_head`` only matters when embeddings are untied; tied checkpoints retie
    ``lm_head.weight`` to ``embed_tokens.weight`` after each optimizer step.
    """

    embed_tokens: float = 0.0
    lm_head: float = 0.0
    rms_norm: float = 0.0
    q_proj: float = 0.1
    k_proj: float = 0.1
    v_proj: float = 0.1
    o_proj: float = 0.1
    gate_proj: float = 0.1
    up_proj: float = 0.1
    down_proj: float = 0.1
    other: float = 0.1

    def validate_weight_decay(self, value: float, field_name: str) -> None:
        if value is None or not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{field_name} must be non-negative and finite")

    def __post_init__(self) -> None:
        for field_name in (
            "embed_tokens",
            "lm_head",
            "rms_norm",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "other",
        ):
            self.validate_weight_decay(getattr(self, field_name), field_name)


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
    lr_total_steps: int | None = 268_000
    epochs: int = 1
    max_rows_per_file: int | None = 40_960
    shuffle_input_files: bool = True
    learning_rate: float = 3e-4
    parameter_weight_decay: ParameterWeightDecayConfig = field(
        default_factory=ParameterWeightDecayConfig
    )
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0
    warmup_steps: int | None = None  # None derives 1% of lr_total_steps.
    min_lr_ratio: float = 0.1
    log_every: int = 10
    save_every: int = 1_000
    seed: int = 42
    autocast_dtype: str = "bfloat16"

    def __post_init__(self) -> None:
        """
        Derive warmup from the schedule horizon unless the caller sets it explicitly.
        """
        if self.warmup_steps is None:
            horizon = 10_000 if self.lr_total_steps is None else self.lr_total_steps
            self.warmup_steps = int(horizon * 0.01)


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
            progress.batches_to_skip -= 1
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


def resolve_compute_dtype(name: str):
    """
    Map training dtype names to MLX dtypes.

    ``none`` keeps float32 weights; otherwise parameters are cast before training.
    """
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
    input_ids = [_as_token_list(example["input_ids"]) for example in batch]
    labels = [_as_token_list(example["labels"]) for example in batch]
    return {
        "input_ids": mx.array(input_ids, dtype=mx.int32),
        "labels": mx.array(labels, dtype=mx.int32),
    }


def tree_add(left: dict, right: dict) -> dict:
    return tree_map(lambda a, b: a + b, left, right)


def tree_scale(tree: dict, scale: float | object) -> dict:
    return tree_map(lambda value: value * scale, tree)


def map_named_tree(tree: object, fn, prefix: str = ""):
    if isinstance(tree, dict):
        return {
            key: map_named_tree(
                value,
                fn,
                f"{prefix}.{key}" if prefix else str(key),
            )
            for key, value in tree.items()
        }
    if isinstance(tree, list):
        return [
            map_named_tree(
                value,
                fn,
                f"{prefix}.{index}" if prefix else str(index),
            )
            for index, value in enumerate(tree)
        ]
    if isinstance(tree, tuple):
        return type(tree)(
            map_named_tree(
                value,
                fn,
                f"{prefix}.{index}" if prefix else str(index),
            )
            for index, value in enumerate(tree)
        )
    return fn(prefix, tree)


def resolve_parameter_weight_decay(
    parameter_name: str,
    parameter_weight_decay: ParameterWeightDecayConfig,
) -> float:
    if parameter_name.endswith(".lora_A") and hasattr(parameter_weight_decay, "lora_a"):
        return parameter_weight_decay.lora_a
    if parameter_name.endswith(".lora_B") and hasattr(parameter_weight_decay, "lora_b"):
        return parameter_weight_decay.lora_b
    if parameter_name == "embed_tokens.weight":
        return parameter_weight_decay.embed_tokens
    if parameter_name == "lm_head.weight":
        return parameter_weight_decay.lm_head
    if (
        parameter_name == "norm.weight"
        or parameter_name.endswith(".input_norm.weight")
        or parameter_name.endswith(".post_attn_norm.weight")
    ):
        return parameter_weight_decay.rms_norm
    parameter_suffixes = (
        (".self_attn.q_proj.weight", parameter_weight_decay.q_proj),
        (".self_attn.k_proj.weight", parameter_weight_decay.k_proj),
        (".self_attn.v_proj.weight", parameter_weight_decay.v_proj),
        (".self_attn.o_proj.weight", parameter_weight_decay.o_proj),
        (".mlp.gate_proj.weight", parameter_weight_decay.gate_proj),
        (".mlp.up_proj.weight", parameter_weight_decay.up_proj),
        (".mlp.down_proj.weight", parameter_weight_decay.down_proj),
    )
    for suffix, weight_decay in parameter_suffixes:
        if parameter_name.endswith(suffix):
            return weight_decay
    return parameter_weight_decay.other


def build_parameter_weight_decay_tree(
    parameters: dict,
    parameter_weight_decay: ParameterWeightDecayConfig,
) -> dict:
    return map_named_tree(
        parameters,
        lambda name, _value: resolve_parameter_weight_decay(
            name,
            parameter_weight_decay,
        ),
    )


def apply_decoupled_weight_decay(
    model,
    weight_decay_tree: dict,
    learning_rate,
) -> None:
    lr = (
        learning_rate
        if isinstance(learning_rate, mx.array)
        else mx.array(learning_rate)
    )

    def decay_parameter(parameter, weight_decay: float):
        if weight_decay == 0.0:
            return parameter
        return parameter * (1.0 - lr.astype(parameter.dtype) * weight_decay)

    model.update(
        tree_map(
            decay_parameter,
            model.trainable_parameters(),
            weight_decay_tree,
        )
    )


def global_grad_norm(grads: dict):
    total = mx.array(0.0)
    for _, grad in tree_flatten(grads):
        total = total + mx.sum(grad.astype(mx.float32) * grad.astype(mx.float32))
    return mx.sqrt(total)


def clip_gradients_by_global_norm(
    grads: dict,
    max_norm: float,
) -> tuple[dict, object]:
    grad_norm = global_grad_norm(grads)
    scale = mx.where(
        grad_norm > max_norm,
        mx.array(max_norm) / mx.maximum(grad_norm, mx.array(1e-12)),
        mx.array(1.0),
    )
    return tree_scale(grads, scale), grad_norm


@dataclass(slots=True)
class GradientAccumulationWindow:
    micro_step: int = 0
    loss_sum: float = 0.0
    accumulated_grads: dict | None = None


def reset_gradient_accumulation_window(window: GradientAccumulationWindow) -> None:
    window.micro_step = 0
    window.loss_sum = 0.0
    window.accumulated_grads = None


def accumulate_gradients(
    window: GradientAccumulationWindow,
    grads: dict,
    loss: float,
    gradient_accumulation_steps: int,
) -> None:
    window.micro_step += 1
    window.loss_sum += loss
    scaled_grads = tree_scale(grads, 1.0 / gradient_accumulation_steps)
    window.accumulated_grads = (
        scaled_grads
        if window.accumulated_grads is None
        else tree_add(window.accumulated_grads, scaled_grads)
    )


def is_accumulation_window_ready(
    window: GradientAccumulationWindow,
    gradient_accumulation_steps: int,
) -> bool:
    return (
        window.micro_step > 0 and window.micro_step % gradient_accumulation_steps == 0
    )


def consume_accumulated_grads(
    window: GradientAccumulationWindow,
    gradient_accumulation_steps: int,
) -> tuple[dict, float, int]:
    if window.accumulated_grads is None:
        raise ValueError("No accumulated gradients to consume")

    remainder = window.micro_step % gradient_accumulation_steps
    micro_batches = gradient_accumulation_steps if remainder == 0 else remainder
    grads = window.accumulated_grads
    if micro_batches != gradient_accumulation_steps:
        grads = tree_scale(
            grads,
            gradient_accumulation_steps / micro_batches,
        )
    avg_loss = window.loss_sum / micro_batches
    reset_gradient_accumulation_window(window)
    return grads, avg_loss, micro_batches


def resolve_mlx_checkpoint_path(training_config: TrainingConfig) -> Path:
    checkpoint_path = (
        Path(training_config.model_path)
        if training_config.model_path is not None
        else resolve_path(training_config.output_dir) / training_config.checkpoint_name
    )
    return resolve_path(checkpoint_path)


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
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(checkpoint_path / MODEL_WEIGHTS_NAME))
    optimizer_state = tree_flatten(optimizer.state, destination={})
    mx.savez(str(checkpoint_path / OPTIMIZER_STATE_NAME), **optimizer_state)
    metadata = {
        "step": step,
        "model_config": json_ready(asdict(model_config)),
        "training_config": json_ready(asdict(training_config)),
        "input_files": [str(input_file) for input_file in input_files],
        "data_state": None if data_state is None else json_ready(asdict(data_state)),
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
    base_model_config = SMLConfig() if model_config is None else model_config

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
        weight_decay=0.0,
    )
    optimizer.init(model.trainable_parameters())
    weight_decay_tree = build_parameter_weight_decay_tree(
        model.trainable_parameters(),
        parameter_weight_decay=training_config.parameter_weight_decay,
    )

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
    accumulation = GradientAccumulationWindow()
    reading_progress = ReadingProgress()
    loss_and_grad = nn.value_and_grad(model, build_loss_fn(model))

    def complete_optimizer_step(
        grads_to_step: dict,
        avg_loss: float,
        epoch_index: int,
    ) -> bool:
        nonlocal global_step
        clipped_grads, grad_norm = clip_gradients_by_global_norm(
            grads_to_step,
            training_config.max_grad_norm,
        )
        apply_decoupled_weight_decay(
            model,
            weight_decay_tree=weight_decay_tree,
            learning_rate=lr_schedule(optimizer.step),
        )
        _retie_embeddings_if_needed(model)
        optimizer.update(model, clipped_grads)
        _retie_embeddings_if_needed(model)
        mx.eval(model.parameters(), optimizer.state)
        global_step += 1

        if global_step % training_config.log_every == 0 or global_step == 1:
            lr = float(optimizer.learning_rate.item())
            print(
                format_training_log(
                    epoch=epoch_index + 1,
                    global_step=global_step,
                    lr=lr,
                    avg_loss=avg_loss,
                    grad_norm=float(grad_norm.item()),
                    timestamp=datetime.now(),
                    progress=reading_progress,
                )
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
            return True

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
        return False

    print(f"Input files: {len(input_files)}")
    print(f"Tokenizer vocab: {checkpoint_model_config.vocab_size:,}")
    print("Backend: mlx")
    print(f"Compute dtype: {training_config.autocast_dtype}")
    print(f"Parameters: total={total_params:,} trainable={trainable_params:,}")

    for epoch in range(data_state.epoch, training_config.epochs):
        if epoch != data_state.epoch:
            reset_training_data_state(data_state, epoch)
        reset_gradient_accumulation_window(accumulation)
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
            accumulate_gradients(
                accumulation,
                grads,
                float(loss.item()),
                training_config.gradient_accumulation_steps,
            )
            if not is_accumulation_window_ready(
                accumulation,
                training_config.gradient_accumulation_steps,
            ):
                continue

            grads_to_step, avg_loss, _ = consume_accumulated_grads(
                accumulation,
                training_config.gradient_accumulation_steps,
            )
            if complete_optimizer_step(grads_to_step, avg_loss, epoch):
                return checkpoint_path

        if accumulation.accumulated_grads is not None:
            grads_to_step, avg_loss, _ = consume_accumulated_grads(
                accumulation,
                training_config.gradient_accumulation_steps,
            )
            if complete_optimizer_step(grads_to_step, avg_loss, epoch):
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
