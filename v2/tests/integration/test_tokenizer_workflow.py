from __future__ import annotations

from pathlib import Path

import pytest
import zstandard as zstd
from sml.artifacts.manifest import VerificationLevel
from sml.data.corpus import CorpusConfig
from sml.data.tokenizer import (
    TokenizerTrainingConfig,
    load_tokenizer_bundle,
    train_tokenizer_bundle,
)
from sml.errors import SMLArtifactError


@pytest.fixture
def tiny_corpus(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    lines = [
        ('{"text":"' + ("alpha beta gamma delta " * 20) + f'{index}"}}').encode()
        for index in range(40)
    ]
    (corpus / "tiny-0000.jsonl.zst").write_bytes(
        zstd.ZstdCompressor().compress(b"\n".join(lines) + b"\n")
    )
    return corpus


def tiny_tokenizer_config(corpus: Path, **overrides):
    values = {
        "corpus": CorpusConfig(input_root=corpus),
        "vocab_size": 300,
        "hard_vocab_limit": False,
        "num_threads": 1,
    }
    values.update(overrides)
    return TokenizerTrainingConfig(**values)


def test_tokenizer_bundle_is_self_describing_and_idempotent(tiny_corpus, tmp_path):
    output = tmp_path / "tokenizer"
    first = train_tokenizer_bundle(tiny_tokenizer_config(tiny_corpus), output)
    assert b".sml-tmp-" not in (output / "tokenizer.model").read_bytes()
    second = train_tokenizer_bundle(tiny_tokenizer_config(tiny_corpus), output)

    assert first.manifest.identity == second.manifest.identity
    assert {path.name for path in output.iterdir()} == {
        "manifest.json",
        "tokenizer.model",
        "tokenizer.vocab",
    }
    loaded = load_tokenizer_bundle(output, VerificationLevel.FULL)
    assert loaded.processor.get_piece_size() == first.manifest.vocab_size
    assert loaded.manifest.algorithm == "bpe"
    assert loaded.verification is VerificationLevel.FULL


def test_changed_configuration_collides_with_immutable_bundle(tiny_corpus, tmp_path):
    output = tmp_path / "tokenizer"
    train_tokenizer_bundle(tiny_tokenizer_config(tiny_corpus), output)

    with pytest.raises(SMLArtifactError, match="collision|different identity"):
        train_tokenizer_bundle(
            tiny_tokenizer_config(tiny_corpus, character_coverage=1.0), output
        )


@pytest.mark.parametrize("payload", ["tokenizer.model", "tokenizer.vocab"])
def test_full_loading_rejects_corrupt_payload_bytes(tiny_corpus, tmp_path, payload):
    output = tmp_path / "tokenizer"
    train_tokenizer_bundle(tiny_tokenizer_config(tiny_corpus), output)
    path = output / payload
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)

    with pytest.raises(SMLArtifactError, match="payload identity mismatch"):
        load_tokenizer_bundle(output, VerificationLevel.FULL)


def test_downstream_loader_rejects_loose_model(tmp_path):
    loose = tmp_path / "tokenizer.model"
    loose.write_bytes(b"not-a-bundle")

    with pytest.raises(SMLArtifactError, match="tokenizer bundle directory"):
        load_tokenizer_bundle(loose, VerificationLevel.FULL)
