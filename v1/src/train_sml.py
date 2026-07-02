from __future__ import annotations

import argparse
import io
import itertools
import json
import pathlib
import random
import re
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator, Protocol, Sequence, TypeVar

import sentencepiece as spm
import torch
import zstandard as zstd
from torch.serialization import safe_globals
from torch.utils.data import DataLoader, IterableDataset

from config import SUCCESS_RETURN_CODE
from sml import SMLLanguageModel, count_parameters, lr_lambda
from sml_config import (
    SMLConfig,
    TrainingConfig,
    model_config_for_training,
)
from train_tokenizer import (
    FIRST_LINE_NUMBER,
    HIDDEN_FILE_PREFIX,
    TEXT_COLUMN,
    TEXT_DECODE_ERRORS,
    TEXT_ENCODING,
    filter_text,
    resolve_path,
)


ROW_INCREMENT = 1
BatchT = TypeVar("BatchT")


@dataclass(slots=True)
class ReadingProgress:
    input_file: str | None = None
    line_number: int | None = None


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


def iter_jsonl_records(
    path: Path,
    max_rows_per_file: int | None,
    start_after_line: int | None = None,
) -> Iterator[tuple[dict[str, object], int]]:
    """
    Ignore blank and non-object rows, but include file and line number when malformed
    JSON is encountered.
    """
    for line_number, line in iter_zstd_jsonl_lines(
        path,
        max_rows_per_file,
        start_after_line=start_after_line,
    ):
        line = line.strip()
        if not line:
            continue

        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path} at line {line_number}") from exc

        if isinstance(row, dict):
            yield row, line_number


def iter_zstd_jsonl_lines(
    path: Path,
    max_rows_per_file: int | None,
    start_after_line: int | None = None,
) -> Iterator[tuple[int, str]]:
    """
    Stream zstd shards instead of materializing them, applying the optional row cap as
    lines are decoded.
    """
    try:
        with path.open("rb") as compressed_stream:
            decompressor = zstd.ZstdDecompressor()
            with decompressor.stream_reader(compressed_stream) as zstd_stream:
                with io.TextIOWrapper(
                    zstd_stream,
                    encoding=TEXT_ENCODING,
                    errors=TEXT_DECODE_ERRORS,
                ) as text_stream:
                    limited_stream = (
                        text_stream
                        if max_rows_per_file is None
                        else itertools.islice(text_stream, max_rows_per_file)
                    )
                    for line_number, line in enumerate(
                        limited_stream,
                        start=FIRST_LINE_NUMBER,
                    ):
                        if (
                            start_after_line is not None
                            and line_number <= start_after_line
                        ):
                            continue
                        yield line_number, line
    except zstd.ZstdError as exc:
        raise RuntimeError(f"zstd failed for {path}: {exc}") from exc


class TokenBlockDataset(IterableDataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        texts: Iterable[str],
        tokenizer: TextTokenizer,
        sequence_length: int,
        stride: int | None = None,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        data_state: TrainingDataState | None = None,
    ) -> None:
        """
        Stride defaults to non-overlapping blocks; special-token fallbacks are resolved
        later because tokenizers may expose methods instead of IDs.
        """
        super().__init__()
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        self.texts = texts
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.stride = sequence_length if stride is None else stride
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.data_state = data_state

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        """
        Maintain a rolling token buffer so streamed documents become fixed-size
        next-token training pairs.
        """
        buffer = [] if self.data_state is None else list(self.data_state.token_buffer)
        tokens_per_block = self.sequence_length + 1
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

        def iter_ready_blocks() -> Iterator[dict[str, torch.Tensor]]:
            while len(buffer) >= tokens_per_block:
                block = buffer[:tokens_per_block]
                del buffer[: self.stride]
                if self.data_state is not None:
                    self.data_state.token_buffer = list(buffer)
                yield {
                    "input_ids": torch.tensor(block[:-1], dtype=torch.long),
                    "labels": torch.tensor(block[1:], dtype=torch.long),
                }

        yield from iter_ready_blocks()
        for text in self.texts:
            """
            Current: Add BOS and EOS tokens to the buffer.

            Other options:
            - Add EOS to the buffer only.
            - Add document-boundary attention masks.
            - Add padding tokens.
            """
            token_ids = self.tokenizer.encode(text, out_type=int)
            if bos_token_id is not None:
                buffer.append(bos_token_id)
            buffer.extend(token_ids)
            if eos_token_id is not None:
                buffer.append(eos_token_id)
            if self.data_state is not None:
                self.data_state.token_buffer = list(buffer)

            yield from iter_ready_blocks()


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


def resolve_device(device: str) -> torch.device:
    """
    Auto mode prefers MPS, then CUDA, then CPU so local accelerator support is used when
    available.
    """
    if device != "auto":
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_autocast_dtype(name: str) -> torch.dtype | None:
    """
    Keep mixed precision opt-in by name because CPU runs should avoid autocast while
    accelerators may benefit.
    """
    if name == "none":
        return None
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError(f"Unsupported autocast dtype: {name}")


def set_seed(seed: int) -> None:
    """
    Seed CUDA separately when present so runs are closer to reproducible across
    supported devices.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloader(
    input_files: Iterable[Path],
    tokenizer: TextTokenizer,
    sequence_length: int,
    batch_size: int,
    max_rows_per_file: int | None,
    progress: ReadingProgress | None = None,
    data_state: TrainingDataState | None = None,
) -> DataLoader[dict[str, torch.Tensor]]:
    """
    Wrap the streaming dataset directly; no extra workers are introduced because the
    iterator owns shard state.
    """
    dataset = TokenBlockDataset(
        texts=iter_texts(
            input_files,
            max_rows_per_file=max_rows_per_file,
            progress=progress,
            data_state=data_state,
        ),
        tokenizer=tokenizer,
        sequence_length=sequence_length,
        data_state=data_state,
    )
    return DataLoader(dataset, batch_size=batch_size)


def is_mps_rng_available() -> bool:
    return (
        hasattr(torch, "mps")
        and hasattr(torch.mps, "get_rng_state")
        and hasattr(torch.mps, "set_rng_state")
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )


def capture_rng_state() -> dict[str, object]:
    return {
        "python_rng_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "mps_rng_state": torch.mps.get_rng_state() if is_mps_rng_available() else None,
    }


def restore_rng_state(checkpoint: dict[object, object]) -> None:
    random.setstate(checkpoint["python_rng_state"])
    torch.set_rng_state(checkpoint["torch_rng_state"].cpu())

    cuda_rng_state_all = checkpoint["cuda_rng_state_all"]
    if cuda_rng_state_all is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [cuda_rng_state.cpu() for cuda_rng_state in cuda_rng_state_all]
        )

    mps_rng_state = checkpoint["mps_rng_state"]
    if mps_rng_state is not None and is_mps_rng_available():
        torch.mps.set_rng_state(mps_rng_state.cpu())


def save_checkpoint(
    path: Path,
    model: SMLLanguageModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    model_config: SMLConfig,
    training_config: TrainingConfig,
    step: int,
    input_files: Iterable[Path] = (),
    data_state: TrainingDataState | None = None,
) -> None:
    """
    Persist configs with state dicts so inference can reconstruct the exact model shape
    later.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_input_files = [str(input_file) for input_file in input_files]
    torch.save(
        {
            "step": step,
            "model_config": asdict(model_config),
            "training_config": asdict(training_config),
            "input_files": checkpoint_input_files,
            "data_state": None if data_state is None else asdict(data_state),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            **capture_rng_state(),
        },
        path,
    )


def load_training_checkpoint(
    path: Path,
    model: SMLLanguageModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: torch.device,
) -> TrainingResumeState:
    """
    Restore model, optimizer, scheduler, and RNG state from a training checkpoint.

    Missing checkpoints are an invalid explicit resume request. Checkpoints are
    full training dictionaries written to ``output_dir / checkpoint_name``
    (default ``v1/output/sml.pt``) with keys such as ``step``, ``model_config``,
    ``training_config``, ``input_files``, ``data_state``, ``model_state_dict``,
    ``optimizer_state_dict``, and ``scheduler_state_dict``.

    PyTorch 2.6+ defaults ``torch.load`` to ``weights_only=True``. Use
    ``weights_only=False`` for trusted local checkpoints, or load through this
    helper or ``infer_sml.load_checkpoint``, which allowlist the needed types.
    """
    checkpoint_path = resolve_path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    with safe_globals([pathlib.PosixPath]):
        checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint must contain a dictionary: {checkpoint_path}")

    step = checkpoint.get("step")
    model_state_dict = checkpoint.get("model_state_dict")
    optimizer_state_dict = checkpoint.get("optimizer_state_dict")
    scheduler_state_dict = checkpoint.get("scheduler_state_dict")
    if not isinstance(step, int):
        raise ValueError("Checkpoint is missing step")
    if not isinstance(model_state_dict, dict):
        raise ValueError("Checkpoint is missing model_state_dict")
    if not isinstance(optimizer_state_dict, dict):
        raise ValueError("Checkpoint is missing optimizer_state_dict")
    if not isinstance(scheduler_state_dict, dict):
        raise ValueError("Checkpoint is missing scheduler_state_dict")

    model.load_state_dict(model_state_dict)
    optimizer.load_state_dict(optimizer_state_dict)
    scheduler.load_state_dict(scheduler_state_dict)
    restore_rng_state(checkpoint)

    input_files = parse_checkpoint_input_files(checkpoint.get("input_files", ()))
    data_state = parse_checkpoint_data_state(checkpoint.get("data_state"))
    return TrainingResumeState(
        step=step,
        input_files=input_files,
        data_state=data_state,
    )


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
    one gradient-accumulation window of dataloader batches.
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


def train_model(
    training_config: TrainingConfig | None = None,
    model_config: SMLConfig | None = None,
    resume_from_checkpoint: bool = False,
) -> Path:
    """
    The tokenizer vocab and requested sequence length are folded into the checkpoint
    config before weights are allocated. Training runs with YaRN disabled; the saved
    config keeps the inference rope_scaling_factor for load-time context extension.
    """
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
    device = resolve_device(training_config.device)
    autocast_dtype = resolve_autocast_dtype(training_config.autocast_dtype)
    model = SMLLanguageModel(training_model_config).to(device)
    total_params, trainable_params = count_parameters(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: lr_lambda(
            step=step,
            total_steps=resolve_lr_total_steps(training_config),
            warmup_steps=training_config.warmup_steps,
            min_lr_ratio=training_config.min_lr_ratio,
        ),
    )

    checkpoint_path = output_dir / training_config.checkpoint_name
    resume_state = TrainingResumeState()
    if resume_from_checkpoint:
        resume_state = load_training_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            scheduler,
            device,
        )
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
    model.train()
    optimizer.zero_grad(set_to_none=True)
    micro_step = 0
    loss_sum = 0.0
    reading_progress = ReadingProgress()

    print(f"Input files: {len(input_files)}")
    print(f"Tokenizer vocab: {checkpoint_model_config.vocab_size:,}")
    print(f"Device: {device}")
    print(f"Parameters: total={total_params:,} trainable={trainable_params:,}")

    for epoch in range(data_state.epoch, training_config.epochs):
        if epoch != data_state.epoch:
            reset_training_data_state(data_state, epoch)
        dataloader = build_dataloader(
            input_files=input_files,
            tokenizer=tokenizer,
            sequence_length=training_config.sequence_length,
            batch_size=training_config.batch_size,
            max_rows_per_file=training_config.max_rows_per_file,
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
            loss = loss / training_config.gradient_accumulation_steps
            loss.backward()

            if micro_step % training_config.gradient_accumulation_steps != 0:
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                training_config.max_grad_norm,
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += ROW_INCREMENT

            if global_step % training_config.log_every == 0 or global_step == 1:
                avg_loss = loss_sum / training_config.gradient_accumulation_steps
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
                training_config.save_every > 0
                and global_step % training_config.save_every == 0
            ):
                save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    scheduler,
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
                    scheduler,
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
        scheduler,
        checkpoint_model_config,
        training_config,
        global_step,
        input_files=input_files,
        data_state=data_state,
    )
    return checkpoint_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """
    Keep CLI surface narrow; training configuration defaults live in TrainingConfig.
    """
    parser = argparse.ArgumentParser(description="Train the SML language model.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from output_dir/checkpoint_name (default v1/output/sml.pt). "
            "Restores model, optimizer, scheduler, and RNG state, then continues "
            "from the saved training step and data position. The checkpoint must "
            "already exist; without --resume, training starts from scratch and "
            "overwrites the checkpoint when it is saved."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """
    Parse the small CLI surface, then delegate training to the config-based API.
    """
    args = parse_args(argv)
    checkpoint_path = train_model(
        training_config=TrainingConfig(),
        resume_from_checkpoint=args.resume,
    )
    print(f"Checkpoint: {checkpoint_path}")
    return SUCCESS_RETURN_CODE


if __name__ == "__main__":
    raise SystemExit(main())
