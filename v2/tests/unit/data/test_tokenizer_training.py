from __future__ import annotations

import json
import subprocess
import sys

import pytest
from sml.artifacts.manifest import (
    VerificationLevel,
)
from sml.data import tokenizer as tokenizer_module
from sml.data.corpus import CorpusSamplingConfig
from sml.data.tokenizer import (
    CONVERSATION_USER_SYMBOLS,
    train_tokenizer_bundle,
)
from sml.errors import SMLArtifactError, SMLDataError

from .tokenizer_helpers import (
    _config,
    _install_fake_sentencepiece,
)


def test_importing_tokenizer_module_does_not_import_sentencepiece():
    code = (
        "import sys; import sml.data.tokenizer; print('sentencepiece' in sys.modules)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )

    assert completed.returncode == 0
    assert completed.stdout.strip() == "False"
    assert completed.stderr == ""


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"algorithm": "unigram"}, "algorithm"),
        ({"vocab_size": 4}, "vocab_size"),
        ({"character_coverage": 0.0}, "character_coverage"),
        ({"num_threads": 0}, "num_threads"),
        ({"input_sentence_size": -1}, "input_sentence_size"),
        ({"self_test_sample_size": -1}, "self_test_sample_size"),
        ({"maximum_sentence_length": 0}, "maximum_sentence_length"),
        ({"conversation_user_symbols": ("x", "x", "y")}, "symbols"),
    ],
)
def test_training_config_rejects_invalid_values_before_workflow(
    tmp_path, overrides, message
):
    sentencepiece_was_imported = "sentencepiece" in sys.modules
    with pytest.raises((TypeError, ValueError), match=message):
        _config(tmp_path, **overrides)
    assert ("sentencepiece" in sys.modules) is sentencepiece_was_imported


def test_training_uses_exact_explicit_sentencepiece_arguments(tmp_path, monkeypatch):
    shard = tmp_path / "a.jsonl.zst"
    shard.write_bytes(b"unused")
    calls = _install_fake_sentencepiece(monkeypatch)
    monkeypatch.setattr(
        "sml.data.tokenizer.discover_corpus_files", lambda _config: (shard,)
    )
    monkeypatch.setattr(
        "sml.data.tokenizer.iter_filtered_texts",
        lambda *_args: iter(("usable text",)),
    )
    config = _config(tmp_path)

    bundle = train_tokenizer_bundle(config, tmp_path / "bundle")

    assert bundle.verification is VerificationLevel.FULL
    assert len(calls) == 1
    call = calls[0]
    model_writer = call.pop("model_writer")
    assert model_writer.getvalue() == b"model"
    assert "model_prefix" not in call
    sentence_iterator = call.pop("sentence_iterator")
    assert list(sentence_iterator) == []
    assert call == {
        "vocab_size": 5,
        "model_type": "bpe",
        "character_coverage": 0.9995,
        "byte_fallback": True,
        "normalization_rule_name": "nmt_nfkc",
        "num_threads": 8,
        "input_sentence_size": 0,
        "shuffle_input_sentence": True,
        "self_test_sample_size": 0,
        "hard_vocab_limit": False,
        "train_extremely_large_corpus": False,
        "max_sentence_length": 16_384,
        "unk_id": 0,
        "bos_id": 1,
        "eos_id": 2,
        "pad_id": 3,
        "user_defined_symbols": list(CONVERSATION_USER_SYMBOLS),
    }


def test_training_serializes_vocab_from_processor_pieces_and_scores(
    tmp_path, monkeypatch
):
    pieces = ("<unk>", "<s>", "</s>", "<pad>", "piece")
    scores = (0.0, 0.0, 0.0, 0.0, -17.25)
    _install_fake_sentencepiece(monkeypatch, pieces=pieces, scores=scores)
    shard = tmp_path / "a.jsonl.zst"
    shard.write_bytes(b"unused")
    monkeypatch.setattr(
        "sml.data.tokenizer.discover_corpus_files", lambda _config: (shard,)
    )
    monkeypatch.setattr(
        "sml.data.tokenizer.iter_filtered_texts", lambda *_args: iter(("text",))
    )

    bundle = train_tokenizer_bundle(_config(tmp_path), tmp_path / "bundle")

    assert (bundle.path / "tokenizer.model").read_bytes() == b"model"
    assert (bundle.path / "tokenizer.vocab").read_text() == (
        "<unk>\t0\n<s>\t0\n</s>\t0\n<pad>\t0\npiece\t-17.25\n"
    )


@pytest.mark.parametrize(
    ("ids", "message"),
    [
        ((4, 2, 3, 0), "bos"),
        ((1, 4, 3, 0), "eos"),
        ((1, 2, 4, 0), "pad"),
        ((1, 2, 3, 4), "unk"),
    ],
)
def test_training_rejects_generated_model_with_wrong_special_ids(
    tmp_path, monkeypatch, ids, message
):
    _install_fake_sentencepiece(monkeypatch, ids=ids)
    shard = tmp_path / "a.jsonl.zst"
    shard.write_bytes(b"unused")
    monkeypatch.setattr(
        "sml.data.tokenizer.discover_corpus_files", lambda _config: (shard,)
    )
    monkeypatch.setattr(
        "sml.data.tokenizer.iter_filtered_texts", lambda *_args: iter(("text",))
    )
    output = tmp_path / "bundle"

    with pytest.raises(SMLArtifactError, match=message):
        train_tokenizer_bundle(_config(tmp_path), output)

    assert not output.exists()


def test_empty_filtered_corpus_fails_before_trainer_invocation(tmp_path, monkeypatch):
    calls = _install_fake_sentencepiece(monkeypatch)
    monkeypatch.setattr("sml.data.tokenizer.discover_corpus_files", lambda _config: ())

    with pytest.raises(SMLDataError, match="No usable text"):
        train_tokenizer_bundle(_config(tmp_path), tmp_path / "bundle")

    assert calls == []


@pytest.mark.parametrize("error_type", [TypeError, ValueError, RuntimeError])
def test_training_preserves_unexpected_corpus_errors(tmp_path, monkeypatch, error_type):
    error = error_type("unexpected corpus failure")

    def texts(*_args):
        yield "usable text"
        raise error

    _install_fake_sentencepiece(monkeypatch)
    monkeypatch.setattr(tokenizer_module, "iter_filtered_texts", texts)
    output = tmp_path / "bundle"

    with pytest.raises(error_type) as caught:
        train_tokenizer_bundle(_config(tmp_path), output)

    assert caught.value is error
    assert not output.exists()
    assert not list(tmp_path.glob(".sml-tmp-*"))


def test_training_preserves_unexpected_trainer_errors(tmp_path, monkeypatch):
    _install_fake_sentencepiece(monkeypatch)
    closed = []

    def texts(*_args):
        try:
            yield "usable text"
        finally:
            closed.append(True)

    monkeypatch.setattr(tokenizer_module, "iter_filtered_texts", texts)
    error = RuntimeError("unexpected trainer failure")

    def train(**_kwargs):
        raise error

    monkeypatch.setattr(
        sys.modules["sentencepiece"].SentencePieceTrainer, "train", train
    )
    output = tmp_path / "bundle"

    with pytest.raises(RuntimeError) as caught:
        train_tokenizer_bundle(_config(tmp_path), output)

    assert caught.value is error
    assert closed == [True]
    assert not output.exists()
    assert not list(tmp_path.glob(".sml-tmp-*"))


def test_changed_config_collides_and_identical_target_is_fully_verified(
    tmp_path, monkeypatch
):
    calls = _install_fake_sentencepiece(monkeypatch)
    shard = tmp_path / "a.jsonl.zst"
    shard.write_bytes(b"unused")
    monkeypatch.setattr(
        "sml.data.tokenizer.discover_corpus_files", lambda _config: (shard,)
    )
    monkeypatch.setattr(
        "sml.data.tokenizer.iter_filtered_texts",
        lambda *_args: iter(("usable text",)),
    )
    output = tmp_path / "bundle"
    first = train_tokenizer_bundle(_config(tmp_path), output)
    (output / "tokenizer.model").write_bytes(b"bad!!")

    with pytest.raises(SMLArtifactError, match="verification|identity"):
        train_tokenizer_bundle(_config(tmp_path), output)

    (output / "tokenizer.model").write_bytes(b"model")
    with pytest.raises(SMLArtifactError, match="collision|different identity"):
        train_tokenizer_bundle(_config(tmp_path, character_coverage=1.0), output)
    assert first.manifest.training["character_coverage"] == 0.9995
    assert len(calls) == 3


def test_training_rejects_nonfinite_generated_vocab_score(tmp_path, monkeypatch):
    _install_fake_sentencepiece(
        monkeypatch,
        scores=(0.0, 0.0, 0.0, 0.0, float("nan")),
    )
    shard = tmp_path / "a.jsonl.zst"
    shard.write_bytes(b"unused")
    monkeypatch.setattr(
        "sml.data.tokenizer.discover_corpus_files", lambda _config: (shard,)
    )
    monkeypatch.setattr(
        "sml.data.tokenizer.iter_filtered_texts", lambda *_args: iter(("text",))
    )

    with pytest.raises(SMLArtifactError, match="finite.*score|score.*finite"):
        train_tokenizer_bundle(_config(tmp_path), tmp_path / "bundle")


def test_manifest_training_is_recursive_and_source_locator_is_diagnostic(
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

    bundle = train_tokenizer_bundle(_config(tmp_path), tmp_path / "bundle")
    manifest_json = json.loads((bundle.path / "manifest.json").read_text())

    assert manifest_json["algorithm"] == "bpe"
    assert manifest_json["training"]["corpus"] == {
        "filename_pattern": r".*\.jsonl\.zst\Z",
        "file_order_seed": 42,
        "max_files": None,
        "max_rows_per_file": 40960,
        "max_text_bytes": 16384,
        "min_text_bytes": 1,
        "shuffle_files": True,
        "text_field": "text",
    }
    assert manifest_json["training"]["sampling"] == {
        "max_documents": 100_000,
        "max_bytes": 128 * 1024 * 1024,
        "seed": 42,
    }
    assert manifest_json["diagnostic_source_locator"] == str(tmp_path)


def test_tokenizer_sampling_caps_actual_sentencepiece_input(tmp_path, monkeypatch):
    _install_fake_sentencepiece(monkeypatch)
    received = []

    def train(**kwargs):
        received.extend(kwargs["sentence_iterator"])
        kwargs["model_writer"].write(b"model")

    monkeypatch.setattr(
        sys.modules["sentencepiece"].SentencePieceTrainer, "train", train
    )
    texts = [f"document {i} é" for i in range(50)]
    monkeypatch.setattr(
        tokenizer_module, "iter_filtered_texts", lambda *_args: iter(texts * 3)
    )
    config = _config(
        tmp_path, sampling=CorpusSamplingConfig(max_documents=7, max_bytes=80, seed=13)
    )
    bundle = train_tokenizer_bundle(config, tmp_path / "sampled")

    assert 0 < len(received) <= 7
    assert len(received) == len(set(received))
    assert sum(len(text.encode("utf-8")) for text in received) <= 80
    assert set(received) <= set(texts)
    assert bundle.manifest.training["sampling"]["seed"] == 13
