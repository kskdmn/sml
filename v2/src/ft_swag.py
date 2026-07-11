"""
Fine-tune an SML checkpoint on the SWAG train split with LoRA adapters.

Loads ``allenai/swag`` (``regular`` config) and trains low-rank adapters on the
gold continuation for each ``startphrase``. Defaults read from
``SwagFineTuneConfig`` in this module; the CLI exposes ``--resume`` only.
"""

from __future__ import annotations

import argparse
import copy
import pathlib
import random
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Iterator, Sequence

import torch
from torch.serialization import safe_globals
from torch.utils.data import DataLoader, IterableDataset

from config import (
    OUTPUT_DIR,
    PROJECT_DIR,
    SUCCESS_RETURN_CODE,
    TOKENIZER_MODEL_PATH,
    resolve_path,
)
from infer_sml import load_checkpoint, normalize_model_config
from lora import (
    LoRAConfig,
    apply_lora,
    load_lora_state_dict,
    lora_parameters,
    lora_state_dict,
    merge_lora,
    require_lora_modules,
)
from sml import SMLConfig, SMLLanguageModel, count_parameters, lr_lambda
from train_sml import (
    ROW_INCREMENT,
    ReadingProgress,
    ResumeProgress,
    TrainingDataState,
    TrainingResumeState,
    capture_rng_state,
    count_resume_batches,
    format_training_log,
    get_special_token_id,
    is_step_limit_reached,
    iter_unseen_batches,
    load_tokenizer,
    reset_training_data_state,
    resolve_autocast_dtype,
    resolve_device,
    resolve_lr_total_steps,
    restore_rng_state,
    set_seed,
)

ENDING_KEY_PREFIX = "ending"
SWAG_PROGRESS_NAME = "swag-train"


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
    device: str = "auto"
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


class SwagExampleDataset(IterableDataset[dict[str, torch.Tensor]]):
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
        super().__init__()
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        self.examples = examples
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.data_state = data_state

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
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

        tokens_per_example = self.sequence_length + 1
        for startphrase, ending in self.examples:
            context_ids = list(self.tokenizer.encode(startphrase, out_type=int))
            ending_ids = list(
                self.tokenizer.encode(join_swag_parts("", ending), out_type=int)
            )
            if eos_token_id is not None:
                ending_ids.append(eos_token_id)
            example_tokens = []
            if bos_token_id is not None:
                example_tokens.append(bos_token_id)
            example_tokens.extend(context_ids)
            ending_start = len(example_tokens)
            example_tokens.extend(ending_ids)
            if len(example_tokens) > self.sequence_length:
                if self.data_state is not None:
                    self.data_state.token_buffer = []
                continue

            padded_tokens = example_tokens + [self.pad_token_id] * (
                tokens_per_example - len(example_tokens)
            )
            labels = list(padded_tokens[1:])
            ending_label_start = max(ending_start - 1, 0)
            ending_label_end = ending_label_start + len(ending_ids)
            labels[:ending_label_start] = [self.pad_token_id] * ending_label_start
            labels[ending_label_end:] = [self.pad_token_id] * (
                len(labels) - ending_label_end
            )
            if self.data_state is not None:
                self.data_state.token_buffer = []

            yield {
                "input_ids": torch.tensor(padded_tokens[:-1], dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            }


def build_swag_dataloader(
    fine_tune_config: SwagFineTuneConfig,
    tokenizer: object,
    epoch: int,
    pad_token_id: int = SMLConfig().pad_token_id,
    bos_token_id: int | None = SMLConfig().bos_token_id,
    eos_token_id: int | None = SMLConfig().eos_token_id,
    progress: ReadingProgress | None = None,
    data_state: TrainingDataState | None = None,
) -> DataLoader[dict[str, torch.Tensor]]:
    """
    Stream one fixed-length SWAG example per batch with context labels masked.
    """
    dataset = SwagExampleDataset(
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
    return DataLoader(dataset, batch_size=1)


def load_pretrained_model_config(
    checkpoint_path: Path,
    device: torch.device,
) -> SMLConfig:
    """
    Read ``model_config`` from a checkpoint without loading weights.
    """
    checkpoint = load_checkpoint(checkpoint_path, device)
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("Checkpoint is missing model_config")
    return SMLConfig(**normalize_model_config(model_config))


def load_pretrained_weights(
    model: SMLLanguageModel,
    checkpoint_path: Path,
    device: torch.device,
) -> None:
    """
    Initialize model parameters from a pretrained checkpoint.
    """
    checkpoint = load_checkpoint(checkpoint_path, device)
    model_state_dict = checkpoint.get("model_state_dict")
    if not isinstance(model_state_dict, dict):
        raise ValueError("Checkpoint is missing model_state_dict")
    model.load_state_dict(model_state_dict)


def prepare_lora_model(
    model: SMLLanguageModel,
    fine_tune_config: SwagFineTuneConfig,
) -> None:
    """
    Attach LoRA adapters after base weights are loaded and freeze base parameters.
    """
    apply_lora(model, fine_tune_config.lora)
    require_lora_modules(model, fine_tune_config.lora.target_modules)


def build_merged_state_dict(model: SMLLanguageModel) -> dict[str, torch.Tensor]:
    """
    Merge adapters into base weights and return an inference-compatible state dict.
    """
    merged_model = copy.deepcopy(model)
    merge_lora(merged_model)
    return merged_model.state_dict()


def save_lora_checkpoint(
    path: Path,
    model: SMLLanguageModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    model_config: SMLConfig,
    fine_tune_config: SwagFineTuneConfig,
    step: int,
    data_state: TrainingDataState | None = None,
) -> None:
    """
    Persist LoRA adapters for resume and a merged state dict for inference.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model_config": asdict(model_config),
            "training_config": asdict(fine_tune_config),
            "lora_config": asdict(fine_tune_config.lora),
            "pretrained_checkpoint_path": str(
                resolve_path(fine_tune_config.pretrained_checkpoint_path)
            ),
            "lora_state_dict": lora_state_dict(model),
            "model_state_dict": build_merged_state_dict(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "data_state": None if data_state is None else asdict(data_state),
            **capture_rng_state(),
        },
        path,
    )


def load_lora_checkpoint(
    path: Path,
    model: SMLLanguageModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: torch.device,
) -> TrainingResumeState:
    """
    Restore LoRA adapters, optimizer, scheduler, and RNG state from a checkpoint.
    """
    checkpoint_path = resolve_path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    with safe_globals([pathlib.PosixPath]):
        checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint must contain a dictionary: {checkpoint_path}")

    step = checkpoint.get("step")
    lora_weights = checkpoint.get("lora_state_dict")
    optimizer_state_dict = checkpoint.get("optimizer_state_dict")
    scheduler_state_dict = checkpoint.get("scheduler_state_dict")
    if not isinstance(step, int):
        raise ValueError("Checkpoint is missing step")
    if not isinstance(lora_weights, dict):
        raise ValueError("Checkpoint is missing lora_state_dict")
    if not isinstance(optimizer_state_dict, dict):
        raise ValueError("Checkpoint is missing optimizer_state_dict")
    if not isinstance(scheduler_state_dict, dict):
        raise ValueError("Checkpoint is missing scheduler_state_dict")

    load_lora_state_dict(model, lora_weights)
    optimizer.load_state_dict(optimizer_state_dict)
    scheduler.load_state_dict(scheduler_state_dict)
    restore_rng_state(checkpoint)

    data_state = None
    raw_data_state = checkpoint.get("data_state")
    if isinstance(raw_data_state, dict):
        data_state = TrainingDataState(**raw_data_state)

    return TrainingResumeState(step=step, data_state=data_state)


def fine_tune_swag(
    fine_tune_config: SwagFineTuneConfig | None = None,
    resume_from_checkpoint: bool = False,
) -> Path:
    """
    LoRA fine-tune ``pretrained_checkpoint_path`` on SWAG train examples.

    Fresh runs load frozen base weights, attach adapters, and write
    ``output_dir / checkpoint_name``. ``--resume`` continues from the LoRA
    checkpoint, restoring adapters, optimizer, scheduler, and data position.
    """
    fine_tune_config = SwagFineTuneConfig() if fine_tune_config is None else fine_tune_config

    set_seed(fine_tune_config.seed)
    output_dir = resolve_path(fine_tune_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / fine_tune_config.checkpoint_name

    tokenizer = load_tokenizer(fine_tune_config.tokenizer_model_path)
    device = resolve_device(fine_tune_config.device)
    autocast_dtype = resolve_autocast_dtype(fine_tune_config.autocast_dtype)

    if resume_from_checkpoint:
        base_model_config = load_pretrained_model_config(checkpoint_path, device)
        pretrained_path = resolve_path(fine_tune_config.pretrained_checkpoint_path)
    else:
        pretrained_path = resolve_path(fine_tune_config.pretrained_checkpoint_path)
        if not pretrained_path.exists():
            raise FileNotFoundError(
                f"Pretrained checkpoint does not exist: {pretrained_path}"
            )
        base_model_config = load_pretrained_model_config(pretrained_path, device)

    checkpoint_model_config = replace(
        base_model_config,
        vocab_size=tokenizer.get_piece_size(),
        rope_scaling_factor=SMLConfig().rope_scaling_factor,
        original_max_position_embeddings=max(
            base_model_config.original_max_position_embeddings,
            fine_tune_config.sequence_length,
        ),
    )
    model = SMLLanguageModel(checkpoint_model_config).to(device)
    load_pretrained_weights(model, pretrained_path, device)
    prepare_lora_model(model, fine_tune_config)
    total_params, trainable_params = count_parameters(model)

    optimizer = torch.optim.AdamW(
        lora_parameters(model),
        lr=fine_tune_config.learning_rate,
        weight_decay=fine_tune_config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: lr_lambda(
            step=step,
            total_steps=resolve_lr_total_steps(fine_tune_config),
            warmup_steps=fine_tune_config.warmup_steps,
            min_lr_ratio=fine_tune_config.min_lr_ratio,
        ),
    )

    resume_state = TrainingResumeState()
    if resume_from_checkpoint:
        resume_state = load_lora_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            scheduler,
            device,
        )

    global_step = resume_state.step
    data_state = resume_state.data_state or TrainingDataState()
    legacy_batches_to_skip = (
        count_resume_batches(global_step, fine_tune_config)
        if resume_from_checkpoint and resume_state.data_state is None
        else 0
    )
    resume_progress = ResumeProgress(batches_to_skip=legacy_batches_to_skip)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    micro_step = 0
    loss_sum = 0.0
    reading_progress = ReadingProgress()

    print(f"Dataset: {fine_tune_config.dataset_name}/{fine_tune_config.dataset_config}")
    print(f"Split: {fine_tune_config.dataset_split}")
    print(f"LoRA rank: {fine_tune_config.lora.rank}")
    print(f"LoRA alpha: {fine_tune_config.lora.alpha}")
    print(f"Tokenizer vocab: {checkpoint_model_config.vocab_size:,}")
    print(f"Device: {device}")
    print(f"Parameters: total={total_params:,} trainable={trainable_params:,}")

    for epoch in range(data_state.epoch, fine_tune_config.epochs):
        if epoch != data_state.epoch:
            reset_training_data_state(data_state, epoch)
        dataloader = build_swag_dataloader(
            fine_tune_config=fine_tune_config,
            tokenizer=tokenizer,
            epoch=epoch,
            pad_token_id=checkpoint_model_config.pad_token_id,
            progress=reading_progress,
            data_state=data_state,
        )
        for batch in iter_unseen_batches(dataloader, resume_progress):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            use_autocast = autocast_dtype is not None and device.type != "cpu"
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype or torch.float32,
                enabled=use_autocast,
            ):
                output = model(input_ids, labels=labels)
                if output.loss is None:
                    raise RuntimeError("Model did not return a training loss")
                loss = output.loss

            loss_sum += loss.item()
            micro_step += ROW_INCREMENT
            loss = loss / fine_tune_config.gradient_accumulation_steps
            loss.backward()

            if micro_step % fine_tune_config.gradient_accumulation_steps != 0:
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(
                lora_parameters(model),
                fine_tune_config.max_grad_norm,
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += ROW_INCREMENT

            if global_step % fine_tune_config.log_every == 0 or global_step == 1:
                avg_loss = loss_sum / fine_tune_config.gradient_accumulation_steps
                lr = scheduler.get_last_lr()[0]
                print(
                    format_training_log(
                        epoch=epoch + 1,
                        global_step=global_step,
                        lr=lr,
                        avg_loss=avg_loss,
                        grad_norm=float(grad_norm),
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
                    scheduler,
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
                    scheduler,
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
        scheduler,
        checkpoint_model_config,
        fine_tune_config,
        global_step,
        data_state=data_state,
    )
    return checkpoint_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """
    Keep CLI surface narrow; fine-tuning configuration defaults live in SwagFineTuneConfig.
    """
    parser = argparse.ArgumentParser(
        description="LoRA fine-tune the SML language model on SWAG train examples.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from output_dir/checkpoint_name (default v2/output/sml-swag.pt). "
            "Restores model, optimizer, scheduler, and RNG state, then continues "
            "from the saved training step and data position."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """
    Parse the small CLI surface, then delegate fine-tuning to the config-based API.
    """
    args = parse_args(argv)
    checkpoint_path = fine_tune_swag(
        fine_tune_config=SwagFineTuneConfig(),
        resume_from_checkpoint=args.resume,
    )
    print(f"Checkpoint: {checkpoint_path}")
    return SUCCESS_RETURN_CODE


if __name__ == "__main__":
    raise SystemExit(main())
