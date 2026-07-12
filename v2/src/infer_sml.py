"""
Inference entrypoint for SML checkpoints.

Default files (under ``v2/output/``): checkpoint directory ``sml`` and tokenizer
``bpe_tokenizer.model``. Both must exist before inference.

CLI generation accepts decoding flags documented on ``sml.GenerationConfig`` and
exposed as ``--temperature``, ``--top-p``, ``--repetition-penalty``,
``--no-repeat-ngram-size``, and ``--seed``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Protocol, Sequence

import mlx.core as mx

from config import (
    DEFAULT_MODEL_PATH,
    DEFAULT_TOKENIZER_MODEL_PATH,
    METADATA_NAME,
    MODEL_WEIGHTS_NAME,
    SUCCESS_RETURN_CODE,
    resolve_path,
)
from sml import GenerationConfig, SMLConfig, SMLLanguageModel
from utils import load_tokenizer


class InferenceTokenizer(Protocol):
    def encode(self, text: str, out_type: type = int) -> list[int]:
        """
        Inference relies on the SentencePiece-style `out_type=int` contract and does not
        need the concrete tokenizer type.
        """
        ...

    def decode(self, ids: list[int]) -> str:
        """Decode receives integer IDs after caller-side special-token filtering."""
        ...


def load_checkpoint_metadata(checkpoint_path: Path) -> dict[str, Any]:
    """
    Load the JSON metadata saved next to MLX checkpoint weights.
    """
    metadata_path = resolve_path(checkpoint_path) / METADATA_NAME
    if not metadata_path.exists():
        raise FileNotFoundError(f"Checkpoint metadata does not exist: {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(
            f"Checkpoint metadata must contain a dictionary: {metadata_path}"
        )
    return metadata


def load_model(checkpoint_path: Path) -> SMLLanguageModel:
    """
    Checkpoint directories must contain shape config metadata and MLX weights.
    """
    checkpoint_dir = resolve_path(checkpoint_path)
    metadata = load_checkpoint_metadata(checkpoint_dir)
    model_config = metadata.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("Checkpoint is missing model_config")

    model = SMLLanguageModel(SMLConfig(**model_config))
    model.load_weights(str(checkpoint_dir / MODEL_WEIGHTS_NAME))
    model.eval()
    mx.eval(model.parameters())
    return model


def encode_prompt(
    tokenizer: InferenceTokenizer,
    prompt: str,
    bos_token_id: int | None,
) -> mx.array:
    """
    Insert BOS before batching because training may have taught the model to expect an
    explicit document-start token.
    """
    token_ids = tokenizer.encode(prompt, out_type=int)
    if bos_token_id is not None:
        token_ids = [bos_token_id, *token_ids]
    return mx.array([token_ids], dtype=mx.int32)


def decode_token_ids(
    tokenizer: InferenceTokenizer,
    token_ids: Sequence[int],
    bos_token_id: int | None,
    eos_token_id: int | None,
    pad_token_id: int | None,
) -> str:
    """
    Generation can include echoed BOS, padding, or EOS; remove those control tokens
    before handing IDs back to SentencePiece.
    """
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


def resolve_max_new_tokens(
    max_new_tokens: int | None,
    max_length: int,
    input_length: int,
) -> int:
    """
    Fill the remaining context window when callers omit an explicit token budget.

    When ``max_new_tokens`` is ``None``, return ``max_length - input_length``.
    """
    if max_new_tokens is None:
        return max(0, max_length - input_length)
    return max_new_tokens


def generate_text(
    prompt: str,
    checkpoint_path: Path = DEFAULT_MODEL_PATH,
    tokenizer_model_path: Path = DEFAULT_TOKENIZER_MODEL_PATH,
    max_new_tokens: int | None = None,
    include_prompt: bool = False,
    generation_config: GenerationConfig | None = None,
) -> str:
    """
    This one-shot path loads model and tokenizer per call, then decodes only the
    continuation unless the caller asks to include the prompt.
    """
    if max_new_tokens is not None and max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")

    tokenizer = load_tokenizer(tokenizer_model_path)
    model = load_model(checkpoint_path)
    input_ids = encode_prompt(
        tokenizer,
        prompt,
        bos_token_id=model.config.bos_token_id,
    )
    max_length = model.config.effective_max_position_embeddings
    input_length = input_ids.shape[1]
    if input_length > max_length:
        raise ValueError(
            "prompt exceeds the checkpoint context window: "
            f"{input_length} > {max_length}"
        )
    resolved_max_new_tokens = resolve_max_new_tokens(
        max_new_tokens,
        max_length,
        input_length,
    )

    generated = model.generate(
        input_ids,
        max_new_tokens=resolved_max_new_tokens,
        eos_token_id=model.config.eos_token_id,
        generation_config=generation_config,
    )
    mx.eval(generated)

    start_index = 0 if include_prompt else input_ids.shape[1]
    generated_ids = generated[0, start_index:].tolist()
    return decode_token_ids(
        tokenizer,
        generated_ids,
        bos_token_id=model.config.bos_token_id,
        eos_token_id=model.config.eos_token_id,
        pad_token_id=model.config.pad_token_id,
    )


def generation_config_from_args(args: argparse.Namespace) -> GenerationConfig:
    """
    Build decoding settings from CLI flags while keeping greedy decoding as default.

    ``infer_sml.py`` exposes these knobs as ``--temperature``, ``--top-p``,
    ``--repetition-penalty``, ``--no-repeat-ngram-size``, and ``--seed``.
    """
    return GenerationConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        seed=args.seed,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Accept an explicit argv for tests while keeping CLI defaults in one place."""
    parser = argparse.ArgumentParser(
        description="Generate text from an SML checkpoint."
    )
    parser.add_argument("prompt", help="Prompt text to continue.")
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
        "--max-new-tokens",
        type=int,
        default=None,
        help=(
            "Maximum generated tokens. Defaults to the remaining context window "
            "(effective_max_position_embeddings minus prompt length)."
        ),
    )
    parser.add_argument(
        "--include-prompt",
        action="store_true",
        help="Print the prompt with the generated completion.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help=(
            "Sampling temperature; 0 keeps greedy decoding (default). "
            "With sampling, try 0.7-1.0; 0.8 is a common start. "
            "See GenerationConfig in sml.py."
        ),
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help=(
            "Nucleus sampling cutoff in (0, 1]; 1.0 disables (default). "
            "Ignored when --temperature is 0. With sampling, try 0.9-0.95."
        ),
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help=(
            "Down-weight tokens already in the prefix; 1.0 disables (default). "
            "For phrase loops, try 1.05-1.25; start at 1.15."
        ),
    )
    parser.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=0,
        help=(
            "Hard-block tokens that would repeat an n-gram of this length; "
            "0 disables (default). Try 3 or 4; 3 is stricter than 4."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for sampling. Ignored when --temperature is 0.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI args, run one-shot generation, and return the project success code."""
    args = parse_args(argv)
    text = generate_text(
        prompt=args.prompt,
        checkpoint_path=args.model,
        tokenizer_model_path=args.tokenizer_model,
        max_new_tokens=args.max_new_tokens,
        include_prompt=args.include_prompt,
        generation_config=generation_config_from_args(args),
    )
    print(text)
    return SUCCESS_RETURN_CODE


if __name__ == "__main__":
    raise SystemExit(main())
