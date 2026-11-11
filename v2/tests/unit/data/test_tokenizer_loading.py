from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from sml.artifacts.manifest import (
    OpenedArtifact,
    PayloadRef,
    TokenizerManifest,
    VerificationLevel,
    canonical_json_bytes,
    open_artifact,
    read_manifest,
)
from sml.data import tokenizer as tokenizer_module
from sml.data.tokenizer import (
    load_tokenizer_bundle,
    train_tokenizer_bundle,
)
from sml.errors import SMLArtifactError

from .tokenizer_helpers import (
    _config,
    _install_fake_sentencepiece,
    _Processor,
)


def test_loader_rejects_loose_model_before_sentencepiece_interprets_it(
    tmp_path, monkeypatch
):
    loose = tmp_path / "tokenizer.model"
    loose.write_bytes(b"not-a-bundle")
    monkeypatch.delitem(sys.modules, "sentencepiece", raising=False)

    with pytest.raises(SMLArtifactError, match="tokenizer bundle directory"):
        load_tokenizer_bundle(loose, VerificationLevel.FULL)

    assert "sentencepiece" not in sys.modules


@pytest.mark.parametrize(
    ("pieces", "ids", "message"),
    [
        (("<unk>", "<s>", "</s>", "<pad>"), (1, 2, 3, 0), "piece count"),
        (("wrong", "<s>", "</s>", "<pad>", "piece"), (1, 2, 3, 0), "vocabulary"),
        (("<unk>", "<s>", "</s>", "<pad>", "piece"), (4, 2, 3, 0), "bos"),
        (("<unk>", "<s>", "</s>", "<pad>", "piece"), (1, 4, 3, 0), "eos"),
        (("<unk>", "<s>", "</s>", "<pad>", "piece"), (1, 2, 4, 0), "pad"),
        (("<unk>", "<s>", "</s>", "<pad>", "piece"), (1, 2, 3, 4), "unk"),
    ],
)
def test_loader_fails_closed_on_processor_or_vocab_mismatch(
    tmp_path, monkeypatch, pieces, ids, message
):
    _install_fake_sentencepiece(monkeypatch)
    shard = tmp_path / "a.jsonl.zst"
    shard.write_bytes(b"unused")
    monkeypatch.setattr(
        "sml.data.tokenizer.discover_corpus_files", lambda _config: (shard,)
    )
    monkeypatch.setattr(
        "sml.data.tokenizer.iter_filtered_texts", lambda *_args: iter(("text",))
    )
    output = tmp_path / "bundle"
    train_tokenizer_bundle(_config(tmp_path), output)
    _install_fake_sentencepiece(monkeypatch, pieces=pieces, ids=ids)

    with pytest.raises(SMLArtifactError, match=message):
        load_tokenizer_bundle(output, VerificationLevel.MANIFEST_TRUSTED)


@pytest.mark.parametrize(
    ("replacement", "message"),
    [(b"-3", "score mismatch"), (b"xx", "score")],
)
def test_trusted_loader_rejects_same_size_altered_or_malformed_vocab_score(
    tmp_path, monkeypatch, replacement, message
):
    _install_fake_sentencepiece(monkeypatch)
    shard = tmp_path / "a.jsonl.zst"
    shard.write_bytes(b"unused")
    monkeypatch.setattr(
        "sml.data.tokenizer.discover_corpus_files", lambda _config: (shard,)
    )
    monkeypatch.setattr(
        "sml.data.tokenizer.iter_filtered_texts", lambda *_args: iter(("text",))
    )
    output = tmp_path / "bundle"
    train_tokenizer_bundle(_config(tmp_path), output)
    vocab_path = output / "tokenizer.vocab"
    vocab_path.write_bytes(
        vocab_path.read_bytes().replace(b"piece\t-4", b"piece\t" + replacement)
    )

    with pytest.raises(SMLArtifactError, match=message):
        load_tokenizer_bundle(output, VerificationLevel.MANIFEST_TRUSTED)


def test_loader_consumes_descriptor_read_model_bytes_not_payload_paths(
    tmp_path, monkeypatch
):
    _install_fake_sentencepiece(monkeypatch)
    shard = tmp_path / "a.jsonl.zst"
    shard.write_bytes(b"unused")
    monkeypatch.setattr(
        "sml.data.tokenizer.discover_corpus_files", lambda _config: (shard,)
    )
    monkeypatch.setattr(
        "sml.data.tokenizer.iter_filtered_texts", lambda *_args: iter(("text",))
    )
    output = tmp_path / "bundle"
    train_tokenizer_bundle(_config(tmp_path), output)
    processor_calls = []

    def processor(**kwargs):
        processor_calls.append(kwargs)
        return _Processor(pieces=("<unk>", "<s>", "</s>", "<pad>", "piece"))

    monkeypatch.setitem(
        sys.modules,
        "sentencepiece",
        SimpleNamespace(SentencePieceProcessor=processor),
    )
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("payload path read is forbidden")
        ),
    )

    load_tokenizer_bundle(output, VerificationLevel.MANIFEST_TRUSTED)

    assert processor_calls == [{"model_proto": b"model"}]


def test_loader_retains_both_payload_descriptors_through_all_semantic_checks(
    tmp_path, monkeypatch
):
    """Model and vocab proofs must remain stable until every tokenizer check ends."""
    _install_fake_sentencepiece(monkeypatch)
    shard = tmp_path / "a.jsonl.zst"
    shard.write_bytes(b"unused")
    monkeypatch.setattr(
        "sml.data.tokenizer.discover_corpus_files", lambda _config: (shard,)
    )
    monkeypatch.setattr(
        "sml.data.tokenizer.iter_filtered_texts", lambda *_args: iter(("text",))
    )
    output = tmp_path / "bundle"
    train_tokenizer_bundle(_config(tmp_path), output)
    payloads = {}
    descriptors = {}
    real_open_payload = OpenedArtifact.open_payload
    real_read_vocab = tokenizer_module._read_vocab

    def record_payload(artifact, reference):
        payload = real_open_payload(artifact, reference)
        payloads[reference.logical_path] = payload
        descriptors[reference.logical_path] = payload.stream.fileno()
        return payload

    def assert_both_open():
        assert set(payloads) == {"tokenizer.model", "tokenizer.vocab"}
        for payload in payloads.values():
            os.fstat(payload.stream.fileno())

    def read_vocab_while_open(encoded):
        assert_both_open()
        return real_read_vocab(encoded)

    class DescriptorCheckingProcessor(_Processor):
        def __init__(self, **_kwargs):
            assert_both_open()
            super().__init__(pieces=("<unk>", "<s>", "</s>", "<pad>", "piece"))

        def id_to_piece(self, index):
            assert_both_open()
            return super().id_to_piece(index)

        def bos_id(self):
            assert_both_open()
            return super().bos_id()

    monkeypatch.setattr(OpenedArtifact, "open_payload", record_payload)
    monkeypatch.setattr(tokenizer_module, "_read_vocab", read_vocab_while_open)
    monkeypatch.setitem(
        sys.modules,
        "sentencepiece",
        SimpleNamespace(SentencePieceProcessor=DescriptorCheckingProcessor),
    )

    loaded = load_tokenizer_bundle(output, VerificationLevel.FULL)

    assert loaded.manifest.kind == "tokenizer"
    for logical_path, payload in payloads.items():
        assert payload.closed is True
        with pytest.raises(OSError):
            os.fstat(descriptors[logical_path])


def test_loader_closes_its_retained_owner_when_processor_construction_fails(
    tmp_path, monkeypatch
):
    """Catches tokenizer construction leaking the verified root on failure."""
    _install_fake_sentencepiece(monkeypatch)
    shard = tmp_path / "a.jsonl.zst"
    shard.write_bytes(b"unused")
    monkeypatch.setattr(
        "sml.data.tokenizer.discover_corpus_files", lambda _config: (shard,)
    )
    monkeypatch.setattr(
        "sml.data.tokenizer.iter_filtered_texts", lambda *_args: iter(("text",))
    )
    output = tmp_path / "bundle"
    train_tokenizer_bundle(_config(tmp_path), output)
    descriptors: list[int] = []
    real_open_artifact = open_artifact
    real_open_payload = OpenedArtifact.open_payload

    def record_artifact(*args):
        artifact = real_open_artifact(*args)
        descriptors.append(artifact.root.fileno())
        return artifact

    def record_payload(artifact, reference):
        payload = real_open_payload(artifact, reference)
        descriptors.append(payload.stream.fileno())
        return payload

    monkeypatch.setattr("sml.data.tokenizer.open_artifact", record_artifact)
    monkeypatch.setattr(OpenedArtifact, "open_payload", record_payload)
    monkeypatch.setitem(
        sys.modules,
        "sentencepiece",
        SimpleNamespace(
            SentencePieceProcessor=lambda **_kwargs: (_ for _ in ()).throw(
                RuntimeError("processor failed")
            )
        ),
    )

    with pytest.raises(SMLArtifactError, match="tokenizer model"):
        load_tokenizer_bundle(output, VerificationLevel.FULL)
    assert len(descriptors) == 3
    for descriptor in descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_loader_consumes_open_root_after_outer_name_replacement(tmp_path, monkeypatch):
    """A tokenizer opened before a parent swap must use its original inode."""
    _install_fake_sentencepiece(monkeypatch)
    shard = tmp_path / "a.jsonl.zst"
    shard.write_bytes(b"unused")
    monkeypatch.setattr(
        "sml.data.tokenizer.discover_corpus_files", lambda _config: (shard,)
    )
    monkeypatch.setattr(
        "sml.data.tokenizer.iter_filtered_texts", lambda *_args: iter(("text",))
    )
    output = tmp_path / "bundle"
    displaced = tmp_path / "displaced-bundle"
    train_tokenizer_bundle(_config(tmp_path), output)

    def swap_after_open(*args):
        artifact = open_artifact(*args)
        output.rename(displaced)
        output.mkdir()
        return artifact

    monkeypatch.setattr("sml.data.tokenizer.open_artifact", swap_after_open)
    try:
        loaded = load_tokenizer_bundle(output, VerificationLevel.FULL)
    finally:
        output.rmdir()
        displaced.rename(output)

    assert loaded.manifest.kind == "tokenizer"


def test_full_loader_rehashes_the_exact_bytes_passed_to_processor(
    tmp_path, monkeypatch
):
    _install_fake_sentencepiece(monkeypatch)
    shard = tmp_path / "a.jsonl.zst"
    shard.write_bytes(b"unused")
    monkeypatch.setattr(
        "sml.data.tokenizer.discover_corpus_files", lambda _config: (shard,)
    )
    monkeypatch.setattr(
        "sml.data.tokenizer.iter_filtered_texts", lambda *_args: iter(("text",))
    )
    output = tmp_path / "bundle"
    train_tokenizer_bundle(_config(tmp_path), output)
    closed: list[OpenedArtifact] = []
    original_close = OpenedArtifact.close

    def record_close(artifact):
        closed.append(artifact)
        original_close(artifact)

    def swap_after_open(*args):
        artifact = open_artifact(*args)
        (output / "tokenizer.model").write_bytes(b"other")
        return artifact

    monkeypatch.setattr("sml.data.tokenizer.open_artifact", swap_after_open)
    monkeypatch.setattr(OpenedArtifact, "close", record_close)

    with pytest.raises(SMLArtifactError, match="model.*identity|identity.*model"):
        load_tokenizer_bundle(output, VerificationLevel.FULL)
    assert len(closed) == 1
    assert closed[0].closed


@pytest.mark.parametrize("mutation", ["algorithm", "model_path", "vocab_path"])
def test_loader_requires_canonical_tokenizer_manifest_contract(
    tmp_path, monkeypatch, mutation
):
    _install_fake_sentencepiece(monkeypatch)
    shard = tmp_path / "a.jsonl.zst"
    shard.write_bytes(b"unused")
    monkeypatch.setattr(
        "sml.data.tokenizer.discover_corpus_files", lambda _config: (shard,)
    )
    monkeypatch.setattr(
        "sml.data.tokenizer.iter_filtered_texts", lambda *_args: iter(("text",))
    )
    output = tmp_path / "bundle"
    train_tokenizer_bundle(_config(tmp_path), output)
    verified = read_manifest(output, TokenizerManifest, VerificationLevel.FULL)

    manifest = verified.manifest
    if mutation == "algorithm":
        manifest = replace(manifest, algorithm="unigram")
    else:
        payload = manifest.model if mutation == "model_path" else manifest.vocab
        logical_path = f"nested/{payload.logical_path}"
        nested = output / "nested"
        nested.mkdir(exist_ok=True)
        source = output / payload.logical_path
        destination = output / logical_path
        source.rename(destination)
        changed = PayloadRef(logical_path, payload.identity, payload.byte_size)
        field = "model" if mutation == "model_path" else "vocab"
        manifest = replace(manifest, **{field: changed})
    manifest = replace(manifest, identity=manifest.recompute_identity())
    (output / "manifest.json").write_bytes(canonical_json_bytes(manifest))

    with pytest.raises(SMLArtifactError, match="algorithm|logical path"):
        load_tokenizer_bundle(output, VerificationLevel.MANIFEST_TRUSTED)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_training_key",
        "unknown_training_key",
        "missing_corpus_key",
        "unknown_corpus_key",
        "invalid_training_value",
        "training_outer_id_mismatch",
        "actual_vocab_exceeds_requested",
        "hard_limit_vocab_mismatch",
        "missing_sampling_key",
        "invalid_sampling_limit",
    ],
)
def test_loader_rejects_noncanonical_training_schema_before_processor_import(
    tmp_path, monkeypatch, mutation
):
    _install_fake_sentencepiece(monkeypatch)
    shard = tmp_path / "a.jsonl.zst"
    shard.write_bytes(b"unused")
    monkeypatch.setattr(
        "sml.data.tokenizer.discover_corpus_files", lambda _config: (shard,)
    )
    monkeypatch.setattr(
        "sml.data.tokenizer.iter_filtered_texts", lambda *_args: iter(("text",))
    )
    output = tmp_path / "bundle"
    train_tokenizer_bundle(_config(tmp_path), output)
    manifest = read_manifest(output, TokenizerManifest, VerificationLevel.FULL).manifest
    training = json.loads(canonical_json_bytes(manifest.training))

    if mutation == "missing_training_key":
        del training["byte_fallback"]
    elif mutation == "unknown_training_key":
        training["unknown"] = True
    elif mutation == "missing_corpus_key":
        del training["corpus"]["text_field"]
    elif mutation == "unknown_corpus_key":
        training["corpus"]["unknown"] = True
    elif mutation == "invalid_training_value":
        training["num_threads"] = 0
    elif mutation == "training_outer_id_mismatch":
        manifest = replace(manifest, bos_token_id=2, eos_token_id=1)
    elif mutation == "actual_vocab_exceeds_requested":
        manifest = replace(manifest, vocab_size=6)
    elif mutation == "hard_limit_vocab_mismatch":
        training["vocab_size"] = 6
        training["hard_vocab_limit"] = True
    elif mutation == "missing_sampling_key":
        del training["sampling"]["seed"]
    elif mutation == "invalid_sampling_limit":
        training["sampling"]["max_bytes"] = 0
    manifest = replace(manifest, training=training)
    manifest = replace(manifest, identity=manifest.recompute_identity())
    (output / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    monkeypatch.delitem(sys.modules, "sentencepiece", raising=False)

    with pytest.raises(SMLArtifactError, match="training metadata"):
        load_tokenizer_bundle(output, VerificationLevel.MANIFEST_TRUSTED)

    assert "sentencepiece" not in sys.modules


def test_loader_accepts_legacy_unsampled_tokenizer_metadata(tmp_path, monkeypatch):
    _install_fake_sentencepiece(monkeypatch)
    monkeypatch.setattr(
        tokenizer_module, "iter_filtered_texts", lambda *_args: iter(("text",))
    )
    output = tmp_path / "legacy"
    bundle = train_tokenizer_bundle(_config(tmp_path), output)
    training = json.loads(canonical_json_bytes(bundle.manifest.training))
    training.pop("sampling")
    training["corpus"].pop("max_files")
    manifest = replace(bundle.manifest, training=training)
    manifest = replace(manifest, identity=manifest.recompute_identity())
    original = canonical_json_bytes(manifest)
    (output / "manifest.json").write_bytes(original)

    loaded = load_tokenizer_bundle(output, VerificationLevel.FULL)

    assert loaded.manifest.identity == manifest.identity
    assert (output / "manifest.json").read_bytes() == original
