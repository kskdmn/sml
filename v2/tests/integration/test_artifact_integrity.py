from __future__ import annotations

from dataclasses import replace

import pytest
from sml.artifacts.manifest import (
    PayloadRef,
    TokenizerManifest,
    VerificationLevel,
    canonical_json_bytes,
    file_identity,
    read_manifest,
)
from sml.errors import SMLArtifactError


def _published_tokenizer_manifest(root):
    model_path = root / "tokenizer.model"
    vocab_path = root / "tokenizer.vocab"
    model_path.write_bytes(b"model bytes")
    vocab_path.write_bytes(b"vocab bytes")
    with model_path.open("rb") as file:
        model_identity = file_identity(file)
    with vocab_path.open("rb") as file:
        vocab_identity = file_identity(file)
    manifest = TokenizerManifest(
        kind="tokenizer",
        version=1,
        identity="sha256:" + "0" * 64,
        algorithm="sentencepiece-bpe-v1",
        training={"byte_fallback": True},
        vocab_size=256,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=3,
        unk_token_id=0,
        model=PayloadRef("tokenizer.model", model_identity, model_path.stat().st_size),
        vocab=PayloadRef("tokenizer.vocab", vocab_identity, vocab_path.stat().st_size),
        diagnostic_source_locator="/source/tokenizer",
    )
    manifest = replace(manifest, identity=manifest.recompute_identity())
    (root / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    return manifest


def test_full_manifest_verification_rehashes_payloads(tmp_path):
    """Reporting full without hashing changed bytes would overstate integrity."""
    expected = _published_tokenizer_manifest(tmp_path)

    verified = read_manifest(tmp_path, TokenizerManifest, VerificationLevel.FULL)

    assert verified.manifest == expected
    assert verified.verification is VerificationLevel.FULL

    (tmp_path / "tokenizer.model").write_bytes(b"tampered!!!")
    with pytest.raises(SMLArtifactError, match="payload identity"):
        read_manifest(tmp_path, TokenizerManifest, VerificationLevel.FULL)


def test_manifest_trusted_verification_does_not_claim_or_perform_full_rehash(tmp_path):
    """A read-only trusted open must remain distinct from content verification."""
    expected = _published_tokenizer_manifest(tmp_path)
    (tmp_path / "tokenizer.model").write_bytes(b"tampered!!!")

    verified = read_manifest(
        tmp_path, TokenizerManifest, VerificationLevel.MANIFEST_TRUSTED
    )

    assert verified.manifest == expected
    assert verified.verification is VerificationLevel.MANIFEST_TRUSTED
