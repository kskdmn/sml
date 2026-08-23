from __future__ import annotations

import sys
from collections.abc import Iterator, Mapping
from pathlib import Path

import numpy as np
import pytest
from sml.artifacts.manifest import (
    PayloadRef,
    TokenizerManifest,
    VerificationLevel,
)
from sml.errors import SMLArtifactError
from sml.inference import ResolvedModel
from sml.model.config import ModelConfig

IDENTITY_A = "sha256:" + "a" * 64
IDENTITY_B = "sha256:" + "b" * 64
IDENTITY_C = "sha256:" + "c" * 64

VALID_ROW: dict[str, object] = {
    "context": "the cat sat",
    "endings": ("on the mat", "in the car", "by the door", "near a tree"),
    "label": 1,
}


class RecordingProcessor:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def encode(self, text: str) -> list[int]:
        self.calls.append(text)
        words = text.split()
        if not words:
            return []
        return [10 + (index % 20) for index, _ in enumerate(words)]


class FakeSwagProvider:
    def __init__(self, rows: tuple[Mapping[str, object], ...]) -> None:
        self.rows = rows
        self.resolve_calls = 0
        self.iter_calls = 0
        self.fail_resolve = False

    def resolve(self, source):
        self.resolve_calls += 1
        if self.fail_resolve:
            raise RuntimeError("provider unavailable")
        from sml.data.swag import ResolvedSwagSource

        return ResolvedSwagSource(
            backend=source.backend,
            namespace=source.namespace,
            name=source.name,
            dataset_config=source.dataset_config,
            revision=source.revision,
            split=source.split,
            commit="abc123def456",
            provider_fingerprint="fingerprint-v1",
            provider_package="datasets",
            provider_version="2.0.0",
        )

    def iter_rows(self, resolved) -> Iterator[Mapping[str, object]]:
        self.iter_calls += 1
        yield from self.rows


class RecordingTokenizer:
    def __init__(self, manifest: TokenizerManifest, processor: object) -> None:
        self.path = Path("recording-tokenizer")
        self.manifest = manifest
        self.verification = VerificationLevel.FULL
        self.processor = processor


def tokenizer_manifest() -> TokenizerManifest:
    return TokenizerManifest(
        kind="tokenizer",
        version=1,
        identity=IDENTITY_A,
        algorithm="sentencepiece-bpe-v1",
        training={"normalization": "nmt_nfkc"},
        vocab_size=64,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=3,
        unk_token_id=0,
        model=PayloadRef("tokenizer.model", IDENTITY_A, 10),
        vocab=PayloadRef("tokenizer.vocab", IDENTITY_B, 20),
        diagnostic_source_locator="/source/tokenizer",
    )


def tiny_base_model() -> ResolvedModel:
    return ResolvedModel(
        artifact_kind="pretraining-checkpoint",
        run_identity=IDENTITY_B,
        step=1,
        checkpoint_identity=IDENTITY_C,
        run_step_identity=IDENTITY_A,
        verification=VerificationLevel.FULL,
        model_config=ModelConfig(
            vocab_size=64,
            hidden_size=16,
            num_layers=2,
            num_q_heads=4,
            num_kv_heads=2,
            intermediate_size=32,
            original_context_length=32,
            hidden_dropout=0.0,
        ),
        tokenizer=RecordingTokenizer(tokenizer_manifest(), RecordingProcessor()),
        model_arrays={},
    )


def tiny_swag_config(provider, **overrides):
    from sml.data.swag import SwagPreparationConfig, SwagSourceConfig

    values = {
        "provider": provider,
        "source": SwagSourceConfig(revision="deadbeef" * 5),
        "maximum_length": 32,
        "bucket_boundaries": (16, 32),
    }
    values.update(overrides)
    return SwagPreparationConfig(**values)


def test_prepare_swag_bundle_writes_mmap_npy_layout(tmp_path):
    from sml.data.swag import load_swag_bundle, prepare_swag_bundle

    provider = FakeSwagProvider((VALID_ROW,))
    output = tmp_path / "swag"
    bundle = prepare_swag_bundle(tiny_swag_config(provider), tiny_base_model(), output)

    bucket_length = bundle.buckets[0].input_ids.shape[-1]
    bucket_dir = output / "buckets" / f"length-{bucket_length:04d}"
    for name in (
        "input_ids.npy",
        "valid_token_mask.npy",
        "score_mask.npy",
        "labels.npy",
    ):
        path = bucket_dir / name
        assert path.is_file()
        loaded = np.load(path, mmap_mode="r", allow_pickle=False)
        assert loaded.ndim >= 1

    reloaded = load_swag_bundle(output, VerificationLevel.FULL)
    assert reloaded.manifest.identity == bundle.manifest.identity
    assert reloaded.path == output
    np.testing.assert_array_equal(
        reloaded.buckets[0].input_ids, bundle.buckets[0].input_ids
    )
    np.testing.assert_array_equal(reloaded.buckets[0].labels, bundle.buckets[0].labels)


def test_reopening_valid_bundle_does_not_call_provider(tmp_path):
    from sml.data.swag import load_swag_bundle, prepare_swag_bundle

    provider = FakeSwagProvider((VALID_ROW,))
    output = tmp_path / "swag"
    first = prepare_swag_bundle(tiny_swag_config(provider), tiny_base_model(), output)
    provider.fail_resolve = True
    provider.resolve_calls = 0
    provider.iter_calls = 0

    second = prepare_swag_bundle(tiny_swag_config(provider), tiny_base_model(), output)
    loaded = load_swag_bundle(output, VerificationLevel.FULL)

    assert provider.resolve_calls == 0
    assert provider.iter_calls == 0
    assert second.manifest.identity == first.manifest.identity
    assert loaded.manifest.identity == first.manifest.identity


def test_identity_mismatch_at_existing_target_is_a_collision(tmp_path):
    from sml.data.swag import SwagSourceConfig, prepare_swag_bundle

    provider = FakeSwagProvider((VALID_ROW,))
    output = tmp_path / "swag"
    prepare_swag_bundle(tiny_swag_config(provider), tiny_base_model(), output)
    with pytest.raises(SMLArtifactError, match="collision"):
        prepare_swag_bundle(
            tiny_swag_config(
                provider,
                source=SwagSourceConfig(
                    revision="deadbeef" * 5,
                    namespace="other-org",
                ),
            ),
            tiny_base_model(),
            output,
        )


def test_swag_module_import_does_not_load_datasets(monkeypatch):
    monkeypatch.delitem(sys.modules, "datasets", raising=False)
    monkeypatch.delitem(sys.modules, "sml.data.swag", raising=False)
    from sml.data import swag

    assert "datasets" not in sys.modules
    assert swag.HuggingFaceDatasetsSwagProvider is not None
