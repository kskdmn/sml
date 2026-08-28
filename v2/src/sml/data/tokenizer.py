"""Training and verified loading for immutable SentencePiece bundles."""

from __future__ import annotations

import io
import itertools
import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from sml.artifacts.checkpoint import publish_immutable_bundle
from sml.artifacts.manifest import (
    OpenedArtifact,
    PayloadRef,
    TokenizerManifest,
    VerificationLevel,
    file_identity,
    open_artifact,
)
from sml.data.corpus import CorpusConfig, discover_corpus_files, iter_filtered_texts
from sml.errors import SMLArtifactError, SMLDataError

CONVERSATION_USER_SYMBOLS = ("<|system|>", "<|user|>", "<|assistant|>")
_PLACEHOLDER_IDENTITY = "sha256:" + "0" * 64
_MODEL_FILENAME = "tokenizer.model"
_VOCAB_FILENAME = "tokenizer.vocab"
_TRAINING_KEYS = frozenset(
    {
        "algorithm",
        "vocab_size",
        "character_coverage",
        "byte_fallback",
        "normalization_rule_name",
        "num_threads",
        "input_sentence_size",
        "shuffle_input_sentence",
        "self_test_sample_size",
        "hard_vocab_limit",
        "train_extremely_large_corpus",
        "maximum_sentence_length",
        "conversation_user_symbols",
        "unk_id",
        "bos_id",
        "eos_id",
        "pad_id",
        "corpus",
    }
)
_CORPUS_KEYS = frozenset(
    {
        "filename_pattern",
        "shuffle_files",
        "file_order_seed",
        "text_field",
        "min_text_bytes",
        "max_text_bytes",
        "max_rows_per_file",
    }
)


def _require_plain_int(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


@dataclass(frozen=True, slots=True)
class TokenizerTrainingConfig:
    """Complete, immutable configuration for the legacy SentencePiece BPE run."""

    corpus: CorpusConfig
    algorithm: str = "bpe"
    vocab_size: int = 28_672
    character_coverage: float = 0.9995
    byte_fallback: bool = True
    normalization_rule_name: str = "nmt_nfkc"
    num_threads: int = 8
    input_sentence_size: int = 0
    shuffle_input_sentence: bool = True
    self_test_sample_size: int = 0
    hard_vocab_limit: bool = True
    train_extremely_large_corpus: bool = False
    maximum_sentence_length: int | None = 16_384
    conversation_user_symbols: tuple[str, str, str] = CONVERSATION_USER_SYMBOLS
    unk_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    pad_id: int = 3

    def __post_init__(self) -> None:
        if not isinstance(self.corpus, CorpusConfig):
            raise TypeError("corpus must be a CorpusConfig")
        if self.algorithm != "bpe":
            raise ValueError("algorithm must be 'bpe'")
        _require_plain_int(self.vocab_size, "vocab_size", minimum=5)
        if isinstance(self.character_coverage, bool) or not isinstance(
            self.character_coverage, (int, float)
        ):
            raise TypeError("character_coverage must be a number")
        if not 0.0 < float(self.character_coverage) <= 1.0:
            raise ValueError("character_coverage must be greater than 0 and at most 1")
        object.__setattr__(self, "character_coverage", float(self.character_coverage))
        for name in (
            "byte_fallback",
            "shuffle_input_sentence",
            "hard_vocab_limit",
            "train_extremely_large_corpus",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")
        if self.normalization_rule_name != "nmt_nfkc":
            raise ValueError("normalization_rule_name must be 'nmt_nfkc'")
        _require_plain_int(self.num_threads, "num_threads", minimum=1)
        _require_plain_int(self.input_sentence_size, "input_sentence_size")
        _require_plain_int(self.self_test_sample_size, "self_test_sample_size")
        if self.maximum_sentence_length is not None:
            _require_plain_int(
                self.maximum_sentence_length,
                "maximum_sentence_length",
                minimum=1,
            )
        if not isinstance(self.conversation_user_symbols, tuple):
            raise TypeError("conversation_user_symbols must be a tuple")
        if len(self.conversation_user_symbols) != 3:
            raise ValueError("conversation_user_symbols must contain exactly 3 symbols")
        if not all(
            isinstance(symbol, str) and symbol
            for symbol in self.conversation_user_symbols
        ):
            raise ValueError("conversation user symbols must be nonempty strings")
        if len(set(self.conversation_user_symbols)) != 3:
            raise ValueError("conversation user symbols must be unique")
        if self.conversation_user_symbols != CONVERSATION_USER_SYMBOLS:
            raise ValueError("conversation user symbols must preserve legacy ordering")
        expected_ids = {"unk_id": 0, "bos_id": 1, "eos_id": 2, "pad_id": 3}
        for name, expected in expected_ids.items():
            value = _require_plain_int(getattr(self, name), name)
            if value != expected:
                raise ValueError(f"{name} must be {expected}")


@dataclass(frozen=True, slots=True)
class TokenizerBundle:
    path: Path
    manifest: TokenizerManifest
    verification: VerificationLevel


@dataclass(frozen=True, slots=True)
class LoadedTokenizer:
    path: Path
    manifest: TokenizerManifest
    verification: VerificationLevel
    processor: Any


def _payload_ref(path: Path, logical_path: str) -> PayloadRef:
    with path.open("rb") as payload:
        identity = file_identity(payload)
    return PayloadRef(
        logical_path=logical_path,
        identity=identity,
        byte_size=path.stat().st_size,
    )


def _corpus_training_projection(config: CorpusConfig) -> Mapping[str, object]:
    return {
        "filename_pattern": config.filename_pattern,
        "shuffle_files": config.shuffle_files,
        "file_order_seed": config.file_order_seed,
        "text_field": config.text_field,
        "min_text_bytes": config.min_text_bytes,
        "max_text_bytes": config.max_text_bytes,
        "max_rows_per_file": config.max_rows_per_file,
    }


def _training_projection(config: TokenizerTrainingConfig) -> Mapping[str, object]:
    return {
        "algorithm": config.algorithm,
        "vocab_size": config.vocab_size,
        "character_coverage": config.character_coverage,
        "byte_fallback": config.byte_fallback,
        "normalization_rule_name": config.normalization_rule_name,
        "num_threads": config.num_threads,
        "input_sentence_size": config.input_sentence_size,
        "shuffle_input_sentence": config.shuffle_input_sentence,
        "self_test_sample_size": config.self_test_sample_size,
        "hard_vocab_limit": config.hard_vocab_limit,
        "train_extremely_large_corpus": config.train_extremely_large_corpus,
        "maximum_sentence_length": config.maximum_sentence_length,
        "conversation_user_symbols": config.conversation_user_symbols,
        "unk_id": config.unk_id,
        "bos_id": config.bos_id,
        "eos_id": config.eos_id,
        "pad_id": config.pad_id,
        "corpus": _corpus_training_projection(config.corpus),
    }


def _require_nonempty(texts: Iterator[str]) -> Iterator[str]:
    try:
        first = next(texts)
    except StopIteration as error:
        raise SMLDataError(
            "No usable text rows found in the configured tokenizer corpus"
        ) from error
    return itertools.chain((first,), texts)


def train_tokenizer_bundle(
    config: TokenizerTrainingConfig,
    output: Path,
) -> TokenizerBundle:
    """Train SentencePiece inside a private sibling and atomically publish it."""
    if not isinstance(config, TokenizerTrainingConfig):
        raise TypeError("config must be a TokenizerTrainingConfig")
    if not isinstance(output, Path):
        raise TypeError("output must be a Path")
    files = discover_corpus_files(config.corpus)
    texts = iter_filtered_texts(config.corpus, files)
    sentence_iterator = _require_nonempty(iter(texts))

    # The dependency is intentionally absent from package import state until valid,
    # nonempty training input has been established.
    import sentencepiece

    def build(private_path: Path) -> TokenizerManifest:
        model_writer = io.BytesIO()
        trainer_arguments: dict[str, object] = {
            "sentence_iterator": sentence_iterator,
            "model_writer": model_writer,
            "vocab_size": config.vocab_size,
            "model_type": config.algorithm,
            "character_coverage": config.character_coverage,
            "byte_fallback": config.byte_fallback,
            "normalization_rule_name": config.normalization_rule_name,
            "num_threads": config.num_threads,
            "input_sentence_size": config.input_sentence_size,
            "shuffle_input_sentence": config.shuffle_input_sentence,
            "self_test_sample_size": config.self_test_sample_size,
            "hard_vocab_limit": config.hard_vocab_limit,
            "train_extremely_large_corpus": config.train_extremely_large_corpus,
            "unk_id": config.unk_id,
            "bos_id": config.bos_id,
            "eos_id": config.eos_id,
            "pad_id": config.pad_id,
            "user_defined_symbols": list(config.conversation_user_symbols),
        }
        if config.maximum_sentence_length is not None:
            trainer_arguments["max_sentence_length"] = config.maximum_sentence_length
        sentencepiece.SentencePieceTrainer.train(**trainer_arguments)

        model_path = private_path / _MODEL_FILENAME
        vocab_path = private_path / _VOCAB_FILENAME
        model_bytes = model_writer.getvalue()
        if not model_bytes:
            raise SMLArtifactError("SentencePiece produced an empty tokenizer model")
        try:
            processor = sentencepiece.SentencePieceProcessor(model_proto=model_bytes)
            piece_count = processor.get_piece_size()
            pieces_and_scores = tuple(
                (processor.id_to_piece(index), processor.get_score(index))
                for index in range(piece_count)
            )
            generated_special_ids = {
                "bos": processor.bos_id(),
                "eos": processor.eos_id(),
                "pad": processor.pad_id(),
                "unk": processor.unk_id(),
            }
        except Exception as error:
            raise SMLArtifactError(
                "SentencePiece produced an invalid tokenizer model"
            ) from error
        if piece_count < 1:
            raise SMLArtifactError(
                "SentencePiece produced an empty tokenizer vocabulary"
            )
        expected_special_ids = {
            "bos": config.bos_id,
            "eos": config.eos_id,
            "pad": config.pad_id,
            "unk": config.unk_id,
        }
        for name, expected in expected_special_ids.items():
            actual = generated_special_ids[name]
            if actual != expected:
                raise SMLArtifactError(
                    f"generated tokenizer {name} special ID mismatch: "
                    f"expected {expected}, got {actual}"
                )
        for piece, score in pieces_and_scores:
            if not isinstance(piece, str) or any(
                character in piece for character in "\t\r\n"
            ):
                raise SMLArtifactError(
                    "SentencePiece produced an invalid vocabulary piece"
                )
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(float(score))
            ):
                raise SMLArtifactError(
                    "SentencePiece produced a non-finite vocabulary score"
                )
        model_path.write_bytes(model_bytes)
        vocab_path.write_text(
            "".join(f"{piece}\t{score:g}\n" for piece, score in pieces_and_scores),
            encoding="utf-8",
        )
        manifest = TokenizerManifest(
            kind="tokenizer",
            version=1,
            identity=_PLACEHOLDER_IDENTITY,
            algorithm="bpe",
            training=_training_projection(config),
            vocab_size=piece_count,
            bos_token_id=config.bos_id,
            eos_token_id=config.eos_id,
            pad_token_id=config.pad_id,
            unk_token_id=config.unk_id,
            model=_payload_ref(model_path, _MODEL_FILENAME),
            vocab=_payload_ref(vocab_path, _VOCAB_FILENAME),
            diagnostic_source_locator=str(config.corpus.input_root),
        )
        return replace(manifest, identity=manifest.recompute_identity())

    published = publish_immutable_bundle(output, build)
    return TokenizerBundle(
        path=published.path,
        manifest=published.manifest,
        verification=published.verification,
    )


def _read_vocab(vocab_bytes: bytes) -> tuple[tuple[str, float], ...]:
    try:
        lines = vocab_bytes.decode("utf-8").splitlines()
    except UnicodeError as error:
        raise SMLArtifactError("invalid tokenizer vocabulary bytes") from error
    entries: list[tuple[str, float]] = []
    for line_number, line in enumerate(lines, start=1):
        if "\t" not in line:
            raise SMLArtifactError(
                f"invalid tokenizer vocabulary at line {line_number}"
            )
        piece, score_text = line.split("\t", 1)
        try:
            score = float(score_text)
        except ValueError as error:
            raise SMLArtifactError(
                f"invalid tokenizer vocabulary score at line {line_number}"
            ) from error
        if not math.isfinite(score):
            raise SMLArtifactError(
                f"non-finite tokenizer vocabulary score at line {line_number}"
            )
        entries.append((piece, score))
    return tuple(entries)


def _validate_training_metadata(manifest: TokenizerManifest) -> None:
    try:
        training = manifest.training
        if set(training) != _TRAINING_KEYS:
            missing = sorted(_TRAINING_KEYS - set(training))
            unknown = sorted(set(training) - _TRAINING_KEYS)
            raise ValueError(
                f"training keys mismatch: missing={missing}, unknown={unknown}"
            )
        corpus_raw = training["corpus"]
        if not isinstance(corpus_raw, Mapping):
            raise TypeError("corpus training metadata must be a mapping")
        if set(corpus_raw) != _CORPUS_KEYS:
            missing = sorted(_CORPUS_KEYS - set(corpus_raw))
            unknown = sorted(set(corpus_raw) - _CORPUS_KEYS)
            raise ValueError(
                f"corpus keys mismatch: missing={missing}, unknown={unknown}"
            )

        corpus = CorpusConfig(input_root=Path("."), **dict(corpus_raw))
        config_values = {key: training[key] for key in _TRAINING_KEYS - {"corpus"}}
        config = TokenizerTrainingConfig(corpus=corpus, **config_values)
        if _training_projection(config) != training:
            raise ValueError("training values are not in canonical configuration form")
        outer_ids = {
            "bos": manifest.bos_token_id,
            "eos": manifest.eos_token_id,
            "pad": manifest.pad_token_id,
            "unk": manifest.unk_token_id,
        }
        configured_ids = {
            "bos": config.bos_id,
            "eos": config.eos_id,
            "pad": config.pad_id,
            "unk": config.unk_id,
        }
        if config.algorithm != manifest.algorithm:
            raise ValueError("training algorithm does not match manifest algorithm")
        if configured_ids != outer_ids:
            raise ValueError("training special IDs do not match manifest special IDs")
        if manifest.vocab_size > config.vocab_size:
            raise ValueError("actual vocab_size exceeds requested training vocab_size")
        if config.hard_vocab_limit and manifest.vocab_size != config.vocab_size:
            raise ValueError(
                "hard_vocab_limit requires actual and requested vocab_size equality"
            )
    except (KeyError, TypeError, ValueError) as error:
        raise SMLArtifactError(
            f"invalid tokenizer training metadata: {error}"
        ) from error


def _load_opened_tokenizer_bundle(
    artifact: OpenedArtifact[TokenizerManifest],
) -> LoadedTokenizer:
    """Build a tokenizer from one retained, manifest-bound artifact owner."""
    if not isinstance(artifact, OpenedArtifact):
        raise TypeError("artifact must be an OpenedArtifact")
    manifest = artifact.manifest
    if manifest.algorithm != "bpe":
        raise SMLArtifactError("tokenizer manifest algorithm must be 'bpe'")
    if manifest.model.logical_path != _MODEL_FILENAME:
        raise SMLArtifactError("tokenizer model logical path must be tokenizer.model")
    if manifest.vocab.logical_path != _VOCAB_FILENAME:
        raise SMLArtifactError("tokenizer vocab logical path must be tokenizer.vocab")
    _validate_training_metadata(manifest)

    with (
        artifact.open_payload(manifest.model) as model_payload,
        artifact.open_payload(manifest.vocab) as vocab_payload,
    ):
        model_bytes = model_payload.stream.read()
        vocab_bytes = vocab_payload.stream.read()
        vocab_entries = _read_vocab(vocab_bytes)

        import sentencepiece

        try:
            processor = sentencepiece.SentencePieceProcessor(model_proto=model_bytes)
            piece_count = processor.get_piece_size()
        except Exception as error:
            raise SMLArtifactError("invalid tokenizer model payload") from error
        if piece_count != manifest.vocab_size:
            raise SMLArtifactError(
                "tokenizer processor piece count does not match manifest vocab_size"
            )
        if len(vocab_entries) != manifest.vocab_size:
            raise SMLArtifactError(
                "tokenizer vocabulary piece count does not match manifest vocab_size"
            )
        for index, (expected_piece, expected_score) in enumerate(vocab_entries):
            try:
                actual_piece = processor.id_to_piece(index)
                actual_score = processor.get_score(index)
            except Exception as error:
                raise SMLArtifactError(
                    "tokenizer vocabulary consistency check failed"
                ) from error
            if actual_piece != expected_piece:
                raise SMLArtifactError(
                    f"tokenizer vocabulary mismatch at piece ID {index}"
                )
            if actual_score != expected_score:
                raise SMLArtifactError(
                    f"tokenizer vocabulary score mismatch at piece ID {index}"
                )

        special_ids = {
            "bos": (processor.bos_id(), manifest.bos_token_id),
            "eos": (processor.eos_id(), manifest.eos_token_id),
            "pad": (processor.pad_id(), manifest.pad_token_id),
            "unk": (processor.unk_id(), manifest.unk_token_id),
        }
        for name, (actual, expected) in special_ids.items():
            if actual != expected:
                raise SMLArtifactError(
                    f"tokenizer {name} special ID mismatch: "
                    f"expected {expected}, got {actual}"
                )
        return LoadedTokenizer(
            path=artifact.path,
            manifest=manifest,
            verification=artifact.verification,
            processor=processor,
        )


def load_tokenizer_bundle(
    path: Path,
    verification: VerificationLevel,
) -> LoadedTokenizer:
    """Verify the bundle and materialize a tokenizer independent of its files."""
    if not isinstance(path, Path):
        raise TypeError("path must be a Path")
    if not isinstance(verification, VerificationLevel):
        raise TypeError("verification must be a VerificationLevel")
    if not path.is_dir():
        raise SMLArtifactError(f"tokenizer bundle directory is required: {path}")
    with open_artifact(path, (TokenizerManifest,), verification) as artifact:
        return _load_opened_tokenizer_bundle(artifact)


__all__ = [
    "CONVERSATION_USER_SYMBOLS",
    "LoadedTokenizer",
    "TokenizerBundle",
    "TokenizerTrainingConfig",
    "load_tokenizer_bundle",
    "train_tokenizer_bundle",
]
