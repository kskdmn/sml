from __future__ import annotations

import io
from dataclasses import replace
from pathlib import Path

from sml.artifacts.manifest import (
    PayloadRef,
    TokenizerManifest,
    canonical_json_bytes,
    file_identity,
)

IDENTITY_A = "sha256:" + "a" * 64


IDENTITY_B = "sha256:" + "b" * 64


def tokenizer_manifest_fixture(**overrides: object) -> TokenizerManifest:
    values: dict[str, object] = {
        "kind": "tokenizer",
        "version": 1,
        "identity": IDENTITY_A,
        "algorithm": "sentencepiece-bpe-v1",
        "training": {"normalization": "nmt_nfkc", "byte_fallback": True},
        "vocab_size": 256,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 3,
        "unk_token_id": 0,
        "model": PayloadRef("tokenizer.model", IDENTITY_A, 10),
        "vocab": PayloadRef("tokenizer.vocab", IDENTITY_B, 20),
        "diagnostic_source_locator": "/source/tokenizer",
    }
    values.update(overrides)
    return TokenizerManifest(**values)


def _write_manifest(root: Path, manifest: object, filename: str) -> object:
    identified = replace(manifest, identity=manifest.recompute_identity())
    (root / filename).write_bytes(canonical_json_bytes(identified))
    return identified


def _write_tokenizer_artifact(
    root: Path,
    *,
    model_data: bytes = b"model-bytes",
    vocab_data: bytes = b"vocab-bytes",
    diagnostic_source_locator: str = "/original",
) -> TokenizerManifest:
    root.mkdir(parents=True, exist_ok=True)
    (root / "tokenizer.model").write_bytes(model_data)
    (root / "tokenizer.vocab").write_bytes(vocab_data)
    manifest = tokenizer_manifest_fixture(
        model=PayloadRef(
            "tokenizer.model", file_identity(io.BytesIO(model_data)), len(model_data)
        ),
        vocab=PayloadRef(
            "tokenizer.vocab", file_identity(io.BytesIO(vocab_data)), len(vocab_data)
        ),
        diagnostic_source_locator=diagnostic_source_locator,
    )
    return _write_manifest(root, manifest, "manifest.json")
