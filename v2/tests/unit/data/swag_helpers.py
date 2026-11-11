from __future__ import annotations

import io
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
from sml.artifacts.manifest import (
    PayloadRef,
    SwagDataManifest,
    TokenizerManifest,
    VerificationLevel,
    canonical_json_bytes,
    file_identity,
)
from sml.errors import SMLDataError
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
        self.encode_calls: list[object] = []

    def encode(self, text: str | Sequence[str]) -> list[int] | list[list[int]]:
        self.encode_calls.append(text)
        if isinstance(text, str):
            self.calls.append(text)
            return self._encode_one(text)
        encoded = []
        for item in text:
            self.calls.append(item)
            encoded.append(self._encode_one(item))
        return encoded

    @staticmethod
    def _encode_one(text: str) -> list[int]:
        words = text.split()
        if not words:
            return []
        return [10 + (index % 20) for index, _ in enumerate(words)]


class FakeSwagProvider:
    def __init__(
        self,
        rows: tuple[Mapping[str, object], ...],
        *,
        commit: str = "abc123def456",
        fingerprint: str = "fingerprint-v1",
        package: str = "datasets",
        version: str = "2.0.0",
    ) -> None:
        self.rows = rows
        self.commit = commit
        self.fingerprint = fingerprint
        self.package = package
        self.version = version
        self.resolve_calls = 0
        self.iter_calls = 0
        self.fail_resolve = False
        self.fail_iter = False
        self.fail_iter_after_rows = False
        self.fail_iter_data_error = False
        self.fail_iter_after_rows_data_error = False
        self.resolved_namespace = None

    def resolve(self, source):
        self.resolve_calls += 1
        if self.fail_resolve:
            raise RuntimeError("provider unavailable")
        from sml.data.swag import ResolvedSwagSource

        namespace = (
            source.namespace
            if self.resolved_namespace is None
            else self.resolved_namespace
        )
        return ResolvedSwagSource(
            backend=source.backend,
            namespace=namespace,
            name=source.name,
            dataset_config=source.dataset_config,
            revision=source.revision,
            split=source.split,
            commit=self.commit,
            provider_fingerprint=self.fingerprint,
            provider_package=self.package,
            provider_version=self.version,
        )

    def iter_rows(self, resolved) -> Iterator[Mapping[str, object]]:
        self.iter_calls += 1
        if self.fail_iter:
            raise RuntimeError("provider unavailable")
        if self.fail_iter_data_error:
            raise SMLDataError("the datasets package is required")
        yield from self.rows
        if self.fail_iter_after_rows:
            raise RuntimeError("provider unavailable")
        if self.fail_iter_after_rows_data_error:
            raise SMLDataError("the datasets package is required")


class RecordingTokenizer:
    def __init__(self, manifest: TokenizerManifest, processor: object) -> None:
        self.path = Path("recording-tokenizer")
        self.manifest = manifest
        self.verification = VerificationLevel.FULL
        self.processor = processor

    @property
    def calls(self) -> list[str]:
        return self.processor.calls


def tokenizer_manifest(**overrides: object) -> TokenizerManifest:
    values: dict[str, object] = {
        "kind": "tokenizer",
        "version": 1,
        "identity": IDENTITY_A,
        "algorithm": "sentencepiece-bpe-v1",
        "training": {"normalization": "nmt_nfkc"},
        "vocab_size": 64,
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


def tiny_model_config(**overrides: object) -> ModelConfig:
    values: dict[str, object] = {
        "vocab_size": 64,
        "hidden_size": 16,
        "num_layers": 2,
        "num_q_heads": 4,
        "num_kv_heads": 2,
        "intermediate_size": 32,
        "original_context_length": 32,
        "hidden_dropout": 0.0,
    }
    values.update(overrides)
    return ModelConfig(**values)


def tiny_base_model(
    *,
    processor: object | None = None,
    tokenizer_manifest_overrides: Mapping[str, object] | None = None,
    model_overrides: Mapping[str, object] | None = None,
) -> ResolvedModel:
    processor = RecordingProcessor() if processor is None else processor
    tokenizer = RecordingTokenizer(
        tokenizer_manifest(**dict(tokenizer_manifest_overrides or {})),
        processor,
    )
    return ResolvedModel(
        artifact_kind="pretraining-checkpoint",
        artifact_identity=IDENTITY_B,
        run_identity=IDENTITY_B,
        step=1,
        checkpoint_identity=IDENTITY_C,
        run_step_identity=IDENTITY_A,
        verification=VerificationLevel.FULL,
        model_config=tiny_model_config(**dict(model_overrides or {})),
        tokenizer=tokenizer,
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


def _array_logical_path(manifest: SwagDataManifest, name: str) -> str:
    suffix = f"/{name}.npy"
    for reference in manifest.buckets:
        if reference.payload.logical_path.endswith(suffix):
            return reference.payload.logical_path
    raise AssertionError(f"missing SWAG array {name}")


def _npy_bytes(array: np.ndarray, *, version: tuple[int, int] = (1, 0)) -> bytes:
    payload = io.BytesIO()
    np.lib.format.write_array(payload, array, version=version, allow_pickle=False)
    return payload.getvalue()


def _replace_array_bytes(
    path: Path,
    manifest: SwagDataManifest,
    logical_path: str,
    payload_bytes: bytes,
) -> SwagDataManifest:
    array_path = path / logical_path
    array_path.write_bytes(payload_bytes)
    with array_path.open("rb") as payload:
        identity = file_identity(payload)
    references = tuple(
        replace(
            reference,
            payload=replace(
                reference.payload,
                identity=identity,
                byte_size=len(payload_bytes),
            ),
        )
        if reference.payload.logical_path == logical_path
        else reference
        for reference in manifest.buckets
    )
    updated = replace(manifest, buckets=references)
    updated = replace(updated, identity=updated.recompute_identity())
    (path / "manifest.json").write_bytes(canonical_json_bytes(updated))
    return updated


def _one_example_bundle(tmp_path: Path):
    from sml.data.swag import prepare_swag_bundle

    return prepare_swag_bundle(
        tiny_swag_config(FakeSwagProvider((VALID_ROW,))),
        tiny_base_model(),
        tmp_path / "swag",
    )


def replace_row(**overrides: object) -> dict[str, object]:
    row = dict(VALID_ROW)
    row.update(overrides)
    return row
