from __future__ import annotations

import argparse
import json
import math
import random
from collections.abc import Iterable, Iterator
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Sequence

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map, tree_unflatten

from config import resolve_path
from sml import SMLConfig
from sml_mlx import SMLLanguageModel, count_parameters
from train_sml import ROW_INCREMENT
from train_sml import ReadingProgress
from train_sml import ResumeProgress
from train_sml import TrainingConfig
from train_sml import TrainingDataState
from train_sml import TrainingResumeState
from train_sml import count_resume_batches
from train_sml import discover_input_files
from train_sml import format_training_log
from train_sml import get_special_token_id
from train_sml import is_step_limit_reached
from train_sml import iter_texts
from train_sml import iter_unseen_batches
from train_sml import load_tokenizer
from train_sml import model_config_for_training
from train_sml import parse_checkpoint_data_state
from train_sml import reset_training_data_state
from train_sml import resolve_lr_total_steps
from train_sml import shuffle_input_files


SUCCESS_RETURN_CODE = 0
MODEL_WEIGHTS_NAME = "model.safetensors"
OPTIMIZER_STATE_NAME = "optimizer.npz"
METADATA_NAME = "metadata.json"


def set_seed(seed: int) -> None:
    random.seed(seed)
    mx.random.seed(seed)


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
) -> Iterator[dict[str, mx.array]]:
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


def _collate_mlx_batch(batch: list[dict[str, object]]) -> dict[str, mx.array]:
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
    def schedule(step: mx.array) -> mx.array:
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
    return tree_map(lambda a, b: a + b, left, right)


def tree_scale(tree: dict, scale: float | mx.array) -> dict:
    return tree_map(lambda value: value * scale, tree)


def global_grad_norm(grads: dict) -> mx.array:
    total = mx.array(0.0)
    for _, grad in tree_flatten(grads):
        total = total + mx.sum(grad.astype(mx.float32) * grad.astype(mx.float32))
    return mx.sqrt(total)


def clip_gradients_by_global_norm(
    grads: dict,
    max_norm: float,
) -> tuple[dict, mx.array]:
    grad_norm = global_grad_norm(grads)
    scale = mx.where(
        grad_norm > max_norm,
        mx.array(max_norm) / mx.maximum(grad_norm, mx.array(1e-12)),
        mx.array(1.0),
    )
    return tree_scale(grads, scale), grad_norm


def resolve_mlx_checkpoint_path(training_config) -> Path:
    output_dir = resolve_path(training_config.output_dir)
    checkpoint_name = Path(training_config.checkpoint_name)
    if checkpoint_name.suffix == ".pt":
        checkpoint_name = checkpoint_name.with_suffix("")
        checkpoint_name = Path(f"{checkpoint_name.name}_mlx")
    return output_dir / checkpoint_name


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
    model: SMLLanguageModel,
    optimizer: optim.Optimizer,
    model_config: SMLConfig,
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
        "model_config": _json_ready(asdict(model_config)),
        "training_config": _json_ready(asdict(training_config)),
        "input_files": [str(input_file) for input_file in input_files],
        "data_state": None if data_state is None else _json_ready(asdict(data_state)),
    }
    (checkpoint_path / METADATA_NAME).write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_training_checkpoint(
    checkpoint_path: Path,
    model: SMLLanguageModel,
    optimizer: optim.Optimizer,
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
        input_files=tuple(resolve_path(Path(str(input_file))) for input_file in input_files),
        data_state=data_state,
    )


def train_model(
    training_config: TrainingConfig | None = None,
    model_config: SMLConfig | None = None,
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
    print("Device: mlx")
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


def _loss_fn(model: SMLLanguageModel):
    def loss_fn(input_ids: mx.array, labels: mx.array) -> mx.array:
        output = model(input_ids, labels=labels)
        if output.loss is None:
            raise RuntimeError("Model did not return a training loss")
        return output.loss

    return loss_fn


def _retie_embeddings_if_needed(model: SMLLanguageModel) -> None:
    if model.config.tie_word_embeddings:
        model.lm_head.weight = model.embed_tokens.weight


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the MLX SML language model.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from the MLX checkpoint directory derived from "
            "output_dir/checkpoint_name. The checkpoint must already exist."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    checkpoint_path = train_model(
        training_config=TrainingConfig(),
        resume_from_checkpoint=args.resume,
    )
    print(f"Checkpoint: {checkpoint_path}")
    return SUCCESS_RETURN_CODE


if __name__ == "__main__":
    raise SystemExit(main())
