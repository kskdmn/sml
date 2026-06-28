from __future__ import annotations

import io
import itertools
import json
import random
import re
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator, Protocol

import sentencepiece as spm
import torch
import zstandard as zstd
from torch.utils.data import DataLoader, IterableDataset

from config import SUCCESS_RETURN_CODE
from sml import SMLLanguageModel, count_parameters, lr_lambda
from sml_config import (
    SMLConfig,
    TrainingConfig,
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


def iter_texts(
    input_files: Iterable[Path],
    max_rows_per_file: int | None,
) -> Iterator[str]:
    """
    Reuse tokenizer-training text filters so tokenizer and model training agree on which
    rows are usable.
    """
    for input_file in input_files:
        for row in iter_jsonl_records(input_file, max_rows_per_file):
            text = filter_text(row.get(TEXT_COLUMN))
            if text is not None:
                yield text


def iter_jsonl_records(
    path: Path,
    max_rows_per_file: int | None,
) -> Iterator[dict[str, object]]:
    """
    Ignore blank and non-object rows, but include file and line number when malformed
    JSON is encountered.
    """
    for line_number, line in iter_zstd_jsonl_lines(path, max_rows_per_file):
        line = line.strip()
        if not line:
            continue

        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path} at line {line_number}") from exc

        if isinstance(row, dict):
            yield row


def iter_zstd_jsonl_lines(
    path: Path,
    max_rows_per_file: int | None,
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
                    yield from enumerate(limited_stream, start=FIRST_LINE_NUMBER)
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

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        """
        Maintain a rolling token buffer so streamed documents become fixed-size
        next-token training pairs.
        """
        buffer: list[int] = []
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

            while len(buffer) >= tokens_per_block:
                block = buffer[:tokens_per_block]
                yield {
                    "input_ids": torch.tensor(block[:-1], dtype=torch.long),
                    "labels": torch.tensor(block[1:], dtype=torch.long),
                }
                del buffer[: self.stride]


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
) -> DataLoader[dict[str, torch.Tensor]]:
    """
    Wrap the streaming dataset directly; no extra workers are introduced because the
    iterator owns shard state.
    """
    dataset = TokenBlockDataset(
        texts=iter_texts(input_files, max_rows_per_file=max_rows_per_file),
        tokenizer=tokenizer,
        sequence_length=sequence_length,
    )
    return DataLoader(dataset, batch_size=batch_size)


def save_checkpoint(
    path: Path,
    model: SMLLanguageModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    model_config: SMLConfig,
    training_config: TrainingConfig,
    step: int,
) -> None:
    """
    Persist configs with state dicts so inference can reconstruct the exact model shape
    later.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model_config": asdict(model_config),
            "training_config": asdict(training_config),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        path,
    )


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
) -> str:
    """
    Log gradient norm before clipping so exploding gradients remain visible even when
    clipping succeeds.
    """
    return (
        f"time={timestamp:%Y-%m-%d %H:%M:%S} "
        f"epoch={epoch} step={global_step} "
        f"lr={lr:.3e} loss={avg_loss:.4f} "
        f"grad_norm={grad_norm:.3f} (before clipping)"
    )


def train_model(
    training_config: TrainingConfig | None = None,
    model_config: SMLConfig | None = None,
) -> Path:
    """
    The tokenizer vocab and requested sequence length are folded into model_config
    before weights are allocated, keeping checkpoints aligned with the data pipeline.
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

    tokenizer = load_tokenizer(training_config.tokenizer_model_path)
    model_config = replace(
        base_model_config,
        vocab_size=tokenizer.get_piece_size(),
        original_max_position_embeddings=max(
            base_model_config.original_max_position_embeddings,
            training_config.sequence_length,
        ),
    )
    device = resolve_device(training_config.device)
    autocast_dtype = resolve_autocast_dtype(training_config.autocast_dtype)
    model = SMLLanguageModel(model_config).to(device)
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
    model.train()
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    micro_step = 0
    loss_sum = 0.0

    print(f"Input files: {len(input_files)}")
    print(f"Tokenizer vocab: {model_config.vocab_size:,}")
    print(f"Device: {device}")
    print(f"Parameters: total={total_params:,} trainable={trainable_params:,}")

    for epoch in range(training_config.epochs):
        dataloader = build_dataloader(
            input_files=input_files,
            tokenizer=tokenizer,
            sequence_length=training_config.sequence_length,
            batch_size=training_config.batch_size,
            max_rows_per_file=training_config.max_rows_per_file,
        )
        for batch in dataloader:
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
                    model_config,
                    training_config,
                    global_step,
                )

            if is_step_limit_reached(global_step, training_config.max_steps):
                save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    scheduler,
                    model_config,
                    training_config,
                    global_step,
                )
                return checkpoint_path

    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        scheduler,
        model_config,
        training_config,
        global_step,
    )
    return checkpoint_path


def main() -> int:
    """
    Keep the entry point thin so tests and callers can exercise train_model without CLI
    parsing.
    """
    checkpoint_path = train_model()
    print(f"Checkpoint: {checkpoint_path}")
    return SUCCESS_RETURN_CODE


if __name__ == "__main__":
    raise SystemExit(main())
