from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from sml.artifacts.manifest import (
    PayloadRef,
    TokenizerManifest,
    VerificationLevel,
    canonical_json_bytes,
    read_manifest,
)
from sml.data.corpus import CorpusConfig
from sml.data.tokenizer import (
    CONVERSATION_USER_SYMBOLS,
    TokenizerTrainingConfig,
    load_tokenizer_bundle,
    train_tokenizer_bundle,
)
from sml.errors import SMLArtifactError, SMLDataError


class _Processor:
    def __init__(
        self,
        *,
        pieces: tuple[str, ...],
        scores: tuple[float, ...] | None = None,
        ids: tuple[int, int, int, int] = (1, 2, 3, 0),
    ) -> None:
        self._pieces = pieces
        self._scores = scores or tuple(float(-index) for index in range(len(pieces)))
        self._ids = ids

    def get_piece_size(self):
        return len(self._pieces)

    def id_to_piece(self, index):
        return self._pieces[index]

    def get_score(self, index):
        return self._scores[index]

    def bos_id(self):
        return self._ids[0]

    def eos_id(self):
        return self._ids[1]

    def pad_id(self):
        return self._ids[2]

    def unk_id(self):
        return self._ids[3]


def _install_fake_sentencepiece(
    monkeypatch, *, pieces=None, scores=None, ids=(1, 2, 3, 0)
):
    pieces = pieces or ("<unk>", "<s>", "</s>", "<pad>", "piece")
    calls = []

    def train(**kwargs):
        calls.append(kwargs)
        list(kwargs["sentence_iterator"])
        kwargs["model_writer"].write(b"model")

    module = SimpleNamespace(
        SentencePieceTrainer=SimpleNamespace(train=train),
        SentencePieceProcessor=lambda **_kwargs: _Processor(
            pieces=pieces,
            scores=scores,
            ids=ids,
        ),
    )
    monkeypatch.setitem(sys.modules, "sentencepiece", module)
    return calls


def _config(tmp_path: Path, **overrides) -> TokenizerTrainingConfig:
    values = {
        "corpus": CorpusConfig(input_root=tmp_path, min_text_bytes=1),
        "vocab_size": 5,
        "hard_vocab_limit": False,
    }
    values.update(overrides)
    return TokenizerTrainingConfig(**values)


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
        "filename_pattern": r".*-00[0-9][0-9]\.jsonl\.zst\Z",
        "file_order_seed": 42,
        "max_rows_per_file": 8192,
        "max_text_bytes": 16384,
        "min_text_bytes": 1,
        "shuffle_files": True,
        "text_field": "text",
    }
    assert manifest_json["diagnostic_source_locator"] == str(tmp_path)


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

    def swap_after_manifest(*args):
        verified = read_manifest(*args)
        (output / "tokenizer.model").write_bytes(b"other")
        return verified

    monkeypatch.setattr("sml.data.tokenizer.read_manifest", swap_after_manifest)

    with pytest.raises(SMLArtifactError, match="model.*identity|identity.*model"):
        load_tokenizer_bundle(output, VerificationLevel.FULL)


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
    manifest = replace(manifest, training=training)
    manifest = replace(manifest, identity=manifest.recompute_identity())
    (output / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    monkeypatch.delitem(sys.modules, "sentencepiece", raising=False)

    with pytest.raises(SMLArtifactError, match="training metadata"):
        load_tokenizer_bundle(output, VerificationLevel.MANIFEST_TRUSTED)

    assert "sentencepiece" not in sys.modules
