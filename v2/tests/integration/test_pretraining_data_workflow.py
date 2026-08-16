from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import zstandard as zstd
from sml.artifacts.manifest import (
    PretrainingDataManifest,
    TokenizerManifest,
    VerificationLevel,
    read_manifest,
)
from sml.data.corpus import CorpusConfig
from sml.data.pretraining import (
    PretrainingPreparationConfig,
    prepare_pretraining_bundle,
)
from sml.data.tokenizer import TokenizerTrainingConfig, train_tokenizer_bundle
from sml.errors import SMLArtifactError, SMLDataError


def _write_corpus(root: Path, texts: list[str]) -> Path:
    root.mkdir()
    midpoint = max(1, len(texts) // 2)
    partitions = (texts[:midpoint], texts[midpoint:])
    for index, partition in enumerate(partitions):
        if not partition:
            continue
        payload = b"".join(
            json.dumps({"text": text}).encode("utf-8") + b"\n" for text in partition
        )
        (root / f"tiny-00{index:02d}.jsonl.zst").write_bytes(
            zstd.ZstdCompressor().compress(payload)
        )
    return root


@pytest.fixture(scope="module")
def prepared_sources(tmp_path_factory):
    root = tmp_path_factory.mktemp("pretraining-sources")
    tokenizer_corpus = _write_corpus(
        root / "tokenizer-corpus",
        [f"alpha beta gamma delta epsilon {index} " * 12 for index in range(40)],
    )
    tokenizer = train_tokenizer_bundle(
        TokenizerTrainingConfig(
            corpus=CorpusConfig(
                input_root=tokenizer_corpus,
                min_text_bytes=1,
                max_rows_per_file=None,
            ),
            vocab_size=300,
            hard_vocab_limit=False,
            num_threads=1,
        ),
        root / "tokenizer",
    )
    data_corpus = _write_corpus(
        root / "data-corpus",
        ["alpha beta gamma delta epsilon " * (4 + index) for index in range(12)],
    )
    return tokenizer, data_corpus


def _config(prepared_sources, **overrides):
    tokenizer, corpus = prepared_sources
    values = {
        "corpus": CorpusConfig(
            input_root=corpus,
            shuffle_files=False,
            min_text_bytes=1,
            max_rows_per_file=None,
        ),
        "tokenizer_bundle": tokenizer.path,
        "sequence_length": 8,
        "shuffle_window_rows": 5,
        "output_shard_rows": 3,
        "seed": 17,
    }
    values.update(overrides)
    return PretrainingPreparationConfig(**values)


def _load_rows(bundle_path: Path) -> np.ndarray:
    manifest = read_manifest(
        bundle_path, PretrainingDataManifest, VerificationLevel.FULL
    ).manifest
    arrays = [
        np.load(bundle_path / shard.logical_path, allow_pickle=False)
        for shard in manifest.shards
    ]
    return np.concatenate(arrays, axis=0)


def test_prepared_bundle_is_closed_self_describing_and_mmap_ready(
    prepared_sources, tmp_path
):
    output = tmp_path / "prepared"
    bundle = prepare_pretraining_bundle(_config(prepared_sources), output)

    assert bundle.path == output
    assert bundle.verification is VerificationLevel.FULL
    assert bundle.manifest.identity == bundle.manifest.recompute_identity()
    assert {path.name for path in output.iterdir()} == {
        "manifest.json",
        "shards",
        "tokenizer",
    }
    assert {path.name for path in (output / "tokenizer").iterdir()} == {
        "manifest.json",
        "tokenizer.model",
        "tokenizer.vocab",
    }
    nested = read_manifest(
        output / "tokenizer", TokenizerManifest, VerificationLevel.FULL
    ).manifest
    assert nested.identity == bundle.manifest.tokenizer_identity
    source_tokenizer, _corpus = prepared_sources
    for name in ("manifest.json", "tokenizer.model", "tokenizer.vocab"):
        assert (output / "tokenizer" / name).read_bytes() == (
            source_tokenizer.path / name
        ).read_bytes()
    assert [shard.logical_path for shard in bundle.manifest.shards] == [
        f"shards/train-{index:06d}.npy" for index in range(len(bundle.manifest.shards))
    ]
    assert set(bundle.manifest.row_order_policy) == {
        "algorithm",
        "shuffle_window_rows",
        "output_shard_rows",
    }
    assert set(bundle.manifest.source_summary) == {
        "corpus",
        "ordered_files",
        "physical_lines_read",
        "object_rows_read",
        "texts_used",
    }
    assert "input_root" not in bundle.manifest.source_summary["corpus"]
    assert bundle.manifest.source_summary["ordered_files"] == (
        "tiny-0000.jsonl.zst",
        "tiny-0001.jsonl.zst",
    )
    assert bundle.manifest.source_summary["physical_lines_read"] == 12
    assert bundle.manifest.source_summary["object_rows_read"] == 12
    assert bundle.manifest.source_summary["texts_used"] == 12
    assert bundle.manifest.diagnostic_source_locator == str(
        _config(prepared_sources).corpus.input_root
    )

    for count, shard in zip(
        bundle.manifest.shard_row_counts, bundle.manifest.shards, strict=True
    ):
        array = np.load(output / shard.logical_path, mmap_mode="r", allow_pickle=False)
        assert array.shape == (count, bundle.manifest.sequence_length + 1)
        assert array.dtype == np.dtype("<i4")
        assert array.flags.c_contiguous
        assert 0 < count <= _config(prepared_sources).output_shard_rows


def test_resharding_preserves_rows_and_row_identity_but_changes_bundle_identity(
    prepared_sources, tmp_path
):
    first = prepare_pretraining_bundle(
        _config(prepared_sources, output_shard_rows=2), tmp_path / "two"
    )
    second = prepare_pretraining_bundle(
        _config(prepared_sources, output_shard_rows=7), tmp_path / "seven"
    )

    assert np.array_equal(_load_rows(first.path), _load_rows(second.path))
    assert first.manifest.row_content_identity == second.manifest.row_content_identity
    assert first.manifest.identity != second.manifest.identity


def test_seed_and_window_are_semantic_row_order_inputs(prepared_sources, tmp_path):
    original = prepare_pretraining_bundle(
        _config(prepared_sources), tmp_path / "original"
    )
    changed_seed = prepare_pretraining_bundle(
        _config(prepared_sources, seed=19), tmp_path / "seed"
    )
    changed_window = prepare_pretraining_bundle(
        _config(prepared_sources, shuffle_window_rows=7), tmp_path / "window"
    )

    assert (
        original.manifest.row_content_identity
        != changed_seed.manifest.row_content_identity
    )
    assert (
        original.manifest.row_content_identity
        != changed_window.manifest.row_content_identity
    )


def test_identical_retry_is_accepted_and_changed_config_collides(
    prepared_sources, tmp_path
):
    output = tmp_path / "prepared"
    first = prepare_pretraining_bundle(_config(prepared_sources), output)
    second = prepare_pretraining_bundle(_config(prepared_sources), output)
    assert first.manifest.identity == second.manifest.identity

    with pytest.raises(SMLArtifactError, match="collision|different identity"):
        prepare_pretraining_bundle(_config(prepared_sources, seed=18), output)


def test_retry_rejects_corrupt_existing_shard(prepared_sources, tmp_path):
    output = tmp_path / "prepared"
    bundle = prepare_pretraining_bundle(_config(prepared_sources), output)
    shard = output / bundle.manifest.shards[0].logical_path
    payload = bytearray(shard.read_bytes())
    payload[-1] ^= 1
    shard.write_bytes(payload)

    with pytest.raises(
        SMLArtifactError, match="existing target failed full verification"
    ):
        prepare_pretraining_bundle(_config(prepared_sources), output)


def test_preparation_refuses_zero_complete_rows_without_visible_output(
    prepared_sources, tmp_path
):
    tokenizer, _corpus = prepared_sources
    empty = _write_corpus(tmp_path / "empty", ["x"])
    config = _config(
        prepared_sources,
        corpus=CorpusConfig(
            input_root=empty,
            shuffle_files=False,
            min_text_bytes=100,
            max_rows_per_file=None,
        ),
        tokenizer_bundle=tokenizer.path,
    )
    output = tmp_path / "prepared"

    with pytest.raises(SMLDataError, match="no complete pretraining rows"):
        prepare_pretraining_bundle(config, output)

    assert not output.exists()


@pytest.mark.parametrize("invalid_id", [-1, 10_000])
def test_preparation_rejects_invalid_processor_token_ids_before_publication(
    prepared_sources, tmp_path, monkeypatch, invalid_id
):
    from sml.data import pretraining

    loaded = pretraining.load_tokenizer_bundle(
        _config(prepared_sources).tokenizer_bundle,
        VerificationLevel.FULL,
    )

    class InvalidProcessor:
        def encode(self, _text):
            return [4, invalid_id, 5]

    monkeypatch.setattr(
        pretraining,
        "load_tokenizer_bundle",
        lambda _path, _verification: replace(loaded, processor=InvalidProcessor()),
    )
    output = tmp_path / f"invalid-{invalid_id}"

    with pytest.raises((SMLArtifactError, SMLDataError), match="token ID"):
        prepare_pretraining_bundle(_config(prepared_sources), output)

    assert not output.exists()
