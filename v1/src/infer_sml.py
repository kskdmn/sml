from __future__ import annotations

import argparse
import pathlib
from pathlib import Path
from typing import Any, Protocol, Sequence

import torch
from torch.serialization import safe_globals

from config import OUTPUT_DIR, SUCCESS_RETURN_CODE, TOKENIZER_MODEL_PATH
from sml import SMLLanguageModel
from sml_config import SMLConfig
from train_sml import load_tokenizer, resolve_device
from train_tokenizer import resolve_path


DEFAULT_CHECKPOINT_PATH = OUTPUT_DIR / "sml.pt"
DEFAULT_MAX_NEW_TOKENS = 100


class InferenceTokenizer(Protocol):
    def encode(self, text: str, out_type: type = int) -> list[int]:
        ...

    def decode(self, ids: list[int]) -> str:
        ...


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> dict[str, Any]:
    path = resolve_path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")

    with safe_globals([pathlib.PosixPath]):
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint must contain a dictionary: {path}")
    return checkpoint


def load_model(checkpoint_path: Path, device: torch.device) -> SMLLanguageModel:
    checkpoint = load_checkpoint(checkpoint_path, device)
    model_config = checkpoint.get("model_config")
    model_state_dict = checkpoint.get("model_state_dict")
    if not isinstance(model_config, dict):
        raise ValueError("Checkpoint is missing model_config")
    if not isinstance(model_state_dict, dict):
        raise ValueError("Checkpoint is missing model_state_dict")

    model = SMLLanguageModel(SMLConfig(**model_config))
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()
    return model


def encode_prompt(
    tokenizer: InferenceTokenizer,
    prompt: str,
    bos_token_id: int | None,
    device: torch.device,
) -> torch.Tensor:
    token_ids = tokenizer.encode(prompt, out_type=int)
    if bos_token_id is not None:
        token_ids = [bos_token_id, *token_ids]
    return torch.tensor([token_ids], dtype=torch.long, device=device)


def decode_token_ids(
    tokenizer: InferenceTokenizer,
    token_ids: Sequence[int],
    bos_token_id: int | None,
    eos_token_id: int | None,
    pad_token_id: int | None,
) -> str:
    decoded_ids: list[int] = []
    skipped_ids = {
        token_id for token_id in (bos_token_id, pad_token_id) if token_id is not None
    }
    for token_id in token_ids:
        if eos_token_id is not None and token_id == eos_token_id:
            break
        if token_id in skipped_ids:
            continue
        decoded_ids.append(int(token_id))
    return tokenizer.decode(decoded_ids)


def generate_text(
    prompt: str,
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    tokenizer_model_path: Path = TOKENIZER_MODEL_PATH,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    device_name: str = "auto",
    include_prompt: bool = False,
) -> str:
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")

    device = resolve_device(device_name)
    tokenizer = load_tokenizer(tokenizer_model_path)
    model = load_model(checkpoint_path, device)
    input_ids = encode_prompt(
        tokenizer,
        prompt,
        bos_token_id=model.config.bos_token_id,
        device=device,
    )

    with torch.no_grad():
        generated = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=model.config.eos_token_id,
        )

    start_index = 0 if include_prompt else input_ids.shape[1]
    generated_ids = generated[0, start_index:].detach().cpu().tolist()
    return decode_token_ids(
        tokenizer,
        generated_ids,
        bos_token_id=model.config.bos_token_id,
        eos_token_id=model.config.eos_token_id,
        pad_token_id=model.config.pad_token_id,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate text from an SML checkpoint."
    )
    parser.add_argument("prompt", help="Prompt text to continue.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=f"Maximum generated tokens. Defaults to {DEFAULT_MAX_NEW_TOKENS}.",
    )
    parser.add_argument(
        "--include-prompt",
        action="store_true",
        help="Print the prompt with the generated completion.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    text = generate_text(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        include_prompt=args.include_prompt,
    )
    print(text)
    return SUCCESS_RETURN_CODE


if __name__ == "__main__":
    raise SystemExit(main())
