"""Immutable encoded SWAG buckets with offline cache identity."""

from __future__ import annotations

import importlib.metadata
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal, Protocol

import numpy as np

from sml.artifacts.checkpoint import publish_immutable_bundle
from sml.artifacts.manifest import (
    ArrayPayloadRef,
    ArraySpec,
    PayloadRef,
    SwagDataManifest,
    VerificationLevel,
    canonical_json_bytes,
    file_identity,
    read_manifest,
    structured_identity,
)
from sml.errors import SMLArtifactError, SMLDataError
from sml.inference import ResolvedModel

_PLACEHOLDER_IDENTITY = "sha256:" + "0" * 64
_INT32 = np.dtype("<i4")
_BOOL = np.dtype("bool")
_SOURCE_REQUEST_FIELDS = (
    "backend",
    "namespace",
    "name",
    "dataset_config",
    "revision",
    "split",
)
_ARRAY_NAMES = ("input_ids", "valid_token_mask", "score_mask", "labels")
JOIN_POLICY_V1 = "separate-context-ending-v1"
OVERLENGTH_POLICY_V1 = "drop-complete-row-v1"
BOS_POLICY_V1 = "context-bos-v1"
EOS_POLICY_V1 = "scored-ending-eos-v1"

SWAG_IDENTITY_FIELDS = (
    "namespace",
    "name",
    "dataset_config",
    "revision",
    "split",
    "preprocessing_schema_version",
    "join_policy",
    "overlength_policy",
    "bos_policy",
    "eos_policy",
    "maximum_length",
    "bucket_boundaries",
)


def _require_plain_int(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _require_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a nonempty string")
    return value


@dataclass(frozen=True, slots=True)
class SwagSourceConfig:
    revision: str
    backend: Literal["huggingface-datasets"] = "huggingface-datasets"
    namespace: str = "allenai"
    name: str = "swag"
    dataset_config: str = "regular"
    split: str = "train"

    def __post_init__(self) -> None:
        _require_string(self.revision, "revision")
        if self.backend != "huggingface-datasets":
            raise ValueError("backend must be 'huggingface-datasets'")
        _require_string(self.namespace, "namespace")
        _require_string(self.name, "name")
        _require_string(self.dataset_config, "dataset_config")
        _require_string(self.split, "split")


@dataclass(frozen=True, slots=True)
class ResolvedSwagSource:
    backend: str
    namespace: str
    name: str
    dataset_config: str
    revision: str
    split: str
    commit: str
    provider_fingerprint: str
    provider_package: str
    provider_version: str

    def __post_init__(self) -> None:
        for name in (
            "backend",
            "namespace",
            "name",
            "dataset_config",
            "revision",
            "split",
            "commit",
            "provider_fingerprint",
            "provider_package",
            "provider_version",
        ):
            _require_string(getattr(self, name), name)


class SwagProvider(Protocol):
    def resolve(self, source: SwagSourceConfig) -> ResolvedSwagSource: ...

    def iter_rows(
        self, resolved: ResolvedSwagSource
    ) -> Iterator[Mapping[str, object]]: ...


@dataclass(frozen=True, slots=True)
class SwagPreparationConfig:
    provider: SwagProvider = field(repr=False, compare=False)
    source: SwagSourceConfig
    preprocessing_schema_version: int = 1
    join_policy: Literal["separate-context-ending-v1"] = JOIN_POLICY_V1
    overlength_policy: Literal["drop-complete-row-v1"] = OVERLENGTH_POLICY_V1
    bos_policy: Literal["context-bos-v1"] = BOS_POLICY_V1
    eos_policy: Literal["scored-ending-eos-v1"] = EOS_POLICY_V1
    maximum_length: int = 256
    bucket_boundaries: tuple[int, ...] = (64, 128, 256)
    maximum_examples: int | None = None

    def __post_init__(self) -> None:
        if not callable(getattr(self.provider, "resolve", None)):
            raise TypeError("provider must implement resolve")
        if not callable(getattr(self.provider, "iter_rows", None)):
            raise TypeError("provider must implement iter_rows")
        if not isinstance(self.source, SwagSourceConfig):
            raise TypeError("source must be a SwagSourceConfig")
        _require_plain_int(
            self.preprocessing_schema_version,
            "preprocessing_schema_version",
            minimum=1,
        )
        for name in ("join_policy", "overlength_policy", "bos_policy", "eos_policy"):
            _require_string(getattr(self, name), name)
        _require_plain_int(self.maximum_length, "maximum_length", minimum=1)
        if not isinstance(self.bucket_boundaries, tuple) or not self.bucket_boundaries:
            raise TypeError("bucket_boundaries must be a nonempty tuple")
        previous = 0
        for boundary in self.bucket_boundaries:
            _require_plain_int(boundary, "bucket boundary", minimum=1)
            if boundary <= previous:
                raise ValueError("bucket_boundaries must be strictly increasing")
            previous = boundary
        if self.maximum_examples is not None:
            _require_plain_int(self.maximum_examples, "maximum_examples", minimum=1)


@dataclass(frozen=True, slots=True)
class SwagBucket:
    length: int
    input_ids: np.ndarray
    valid_token_mask: np.ndarray
    score_mask: np.ndarray
    labels: np.ndarray


@dataclass(frozen=True, slots=True)
class SwagDataBundle:
    path: Path
    manifest: SwagDataManifest
    verification: VerificationLevel
    buckets: tuple[SwagBucket, ...]


class HuggingFaceDatasetsSwagProvider:
    """SWAG rows from Hugging Face datasets, imported only when used."""

    def resolve(self, source: SwagSourceConfig) -> ResolvedSwagSource:
        commit = _huggingface_commit(source)
        dataset = _load_huggingface_split(source, revision=commit)
        fingerprint = getattr(dataset, "_fingerprint", commit)
        return ResolvedSwagSource(
            backend=source.backend,
            namespace=source.namespace,
            name=source.name,
            dataset_config=source.dataset_config,
            revision=source.revision,
            split=source.split,
            commit=commit,
            provider_fingerprint=str(fingerprint),
            provider_package="datasets",
            provider_version=importlib.metadata.version("datasets"),
        )

    def iter_rows(self, resolved: ResolvedSwagSource) -> Iterator[Mapping[str, object]]:
        dataset = _load_huggingface_split(
            SwagSourceConfig(
                revision=resolved.revision,
                backend=resolved.backend,
                namespace=resolved.namespace,
                name=resolved.name,
                dataset_config=resolved.dataset_config,
                split=resolved.split,
            ),
            revision=resolved.commit,
        )
        for row in dataset:
            yield _normalize_huggingface_row(row)


def _import_datasets():
    try:
        import datasets
    except ImportError as error:
        raise SMLDataError(
            "the datasets package is required for HuggingFaceDatasetsSwagProvider"
        ) from error
    return datasets


def _import_huggingface_hub():
    try:
        import huggingface_hub
    except ImportError as error:
        raise SMLDataError(
            "the huggingface_hub package is required for "
            "HuggingFaceDatasetsSwagProvider"
        ) from error
    return huggingface_hub


def _load_huggingface_split(source: SwagSourceConfig, *, revision: str | None = None):
    datasets = _import_datasets()
    return datasets.load_dataset(
        f"{source.namespace}/{source.name}",
        source.dataset_config,
        split=source.split,
        revision=source.revision if revision is None else revision,
    )


def _huggingface_commit(source: SwagSourceConfig) -> str:
    huggingface_hub = _import_huggingface_hub()
    info = huggingface_hub.HfApi().dataset_info(
        f"{source.namespace}/{source.name}",
        revision=source.revision,
        timeout=100.0,
    )
    commit = getattr(info, "sha", None)
    if not isinstance(commit, str) or not commit:
        raise SMLDataError("Hugging Face dataset_info did not return a commit SHA")
    return commit


def _normalize_huggingface_row(row: Mapping[str, object]) -> Mapping[str, object]:
    startphrase = row.get("startphrase")
    if not isinstance(startphrase, str) or not startphrase:
        sent1 = row.get("sent1")
        sent2 = row.get("sent2")
        if not isinstance(sent1, str) or not isinstance(sent2, str):
            raise SMLDataError("SWAG row is missing a context string")
        startphrase = f"{sent1} {sent2}"
    endings = []
    for index in range(4):
        ending = row.get(f"ending{index}")
        if not isinstance(ending, str):
            raise SMLDataError("SWAG row must contain four ending strings")
        endings.append(ending)
    return {
        "context": startphrase,
        "endings": tuple(endings),
        "label": row.get("label"),
    }


def _source_request_projection(source: SwagSourceConfig) -> dict[str, object]:
    return {
        "backend": source.backend,
        "namespace": source.namespace,
        "name": source.name,
        "dataset_config": source.dataset_config,
        "revision": source.revision,
        "split": source.split,
    }


def _resolved_source_projection(resolved: ResolvedSwagSource) -> dict[str, object]:
    return {
        "backend": resolved.backend,
        "namespace": resolved.namespace,
        "name": resolved.name,
        "dataset_config": resolved.dataset_config,
        "revision": resolved.revision,
        "split": resolved.split,
        "commit": resolved.commit,
        "provider_fingerprint": resolved.provider_fingerprint,
        "provider_package": resolved.provider_package,
        "provider_version": resolved.provider_version,
    }


def _require_resolved_matches_request(
    resolved: ResolvedSwagSource, source: SwagSourceConfig
) -> None:
    actual = {name: getattr(resolved, name) for name in _SOURCE_REQUEST_FIELDS}
    if actual != _source_request_projection(source):
        raise SMLDataError("resolved SWAG source does not correspond to the request")


def _source_manifest_projection(source: Mapping[str, object]) -> dict[str, object]:
    return {field_name: source.get(field_name) for field_name in _SOURCE_REQUEST_FIELDS}


def _preprocessing_projection(config: SwagPreparationConfig) -> dict[str, object]:
    return {
        "schema_version": config.preprocessing_schema_version,
        "join_policy": config.join_policy,
        "overlength_policy": config.overlength_policy,
        "bos_policy": config.bos_policy,
        "eos_policy": config.eos_policy,
        "maximum_length": config.maximum_length,
        "bucket_boundaries": list(config.bucket_boundaries),
    }


def _base_identity(base: ResolvedModel) -> str:
    identity = base.identity()
    return structured_identity(
        "sml-resolved-model-identity-v1",
        {
            "artifact_kind": identity.artifact_kind,
            "run_identity": identity.run_identity,
            "step": identity.step,
            "checkpoint_identity": identity.checkpoint_identity,
            "run_step_identity": identity.run_step_identity,
            "tokenizer_identity": identity.tokenizer_identity,
        },
    )


def _special_ids(base: ResolvedModel) -> dict[str, int]:
    manifest = base.tokenizer.manifest
    return {
        "vocab_size": manifest.vocab_size,
        "bos_token_id": manifest.bos_token_id,
        "eos_token_id": manifest.eos_token_id,
        "pad_token_id": manifest.pad_token_id,
        "unk_token_id": manifest.unk_token_id,
    }


def _require_full_base(base: ResolvedModel) -> None:
    if base.verification is not VerificationLevel.FULL:
        raise SMLArtifactError("SWAG preparation requires a FULL-verified base")


def _require_tokenizer_matches_model(base: ResolvedModel) -> None:
    manifest = base.tokenizer.manifest
    model_config = base.model_config
    if manifest.vocab_size != model_config.vocab_size:
        raise SMLDataError("SWAG tokenizer vocab_size does not match the base model")
    for name in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
        if getattr(manifest, name) != getattr(model_config, name):
            raise SMLDataError("SWAG tokenizer special IDs do not match the base model")


def _recorded_identity_is_valid(value: object) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        return False
    digest = value.removeprefix("sha256:")
    return len(digest) == 64 and set(digest) <= set("0123456789abcdef")


def _validate_recorded_projections(manifest: SwagDataManifest) -> None:
    preprocessing = manifest.preprocessing
    if preprocessing.get("schema_version") != 1:
        raise SMLArtifactError("unsupported SWAG preprocessing schema")
    expected_policies = {
        "join_policy": JOIN_POLICY_V1,
        "overlength_policy": OVERLENGTH_POLICY_V1,
        "bos_policy": BOS_POLICY_V1,
        "eos_policy": EOS_POLICY_V1,
    }
    for name, expected in expected_policies.items():
        if preprocessing.get(name) != expected:
            raise SMLArtifactError(f"unsupported SWAG {name}")
    maximum_length = preprocessing.get("maximum_length")
    if (
        isinstance(maximum_length, bool)
        or not isinstance(maximum_length, int)
        or maximum_length < 1
    ):
        raise SMLArtifactError("invalid SWAG maximum_length")
    boundaries = preprocessing.get("bucket_boundaries")
    if not isinstance(boundaries, (list, tuple)) or not boundaries:
        raise SMLArtifactError("invalid SWAG bucket_boundaries")
    previous = 0
    for boundary in boundaries:
        if (
            isinstance(boundary, bool)
            or not isinstance(boundary, int)
            or boundary <= previous
        ):
            raise SMLArtifactError("invalid SWAG bucket_boundaries")
        previous = boundary
    if boundaries[-1] < maximum_length:
        raise SMLArtifactError("SWAG bucket_boundaries do not cover maximum_length")
    if not _recorded_identity_is_valid(manifest.tokenizer_identity):
        raise SMLArtifactError("invalid SWAG tokenizer identity")
    vocab_size = manifest.vocab_size
    if (
        isinstance(vocab_size, bool)
        or not isinstance(vocab_size, int)
        or vocab_size < 1
    ):
        raise SMLArtifactError("invalid SWAG vocab_size")
    special_ids = {
        "bos_token_id": manifest.bos_token_id,
        "eos_token_id": manifest.eos_token_id,
        "pad_token_id": manifest.pad_token_id,
        "unk_token_id": manifest.unk_token_id,
    }
    for name, token_id in special_ids.items():
        if (
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or not 0 <= token_id < vocab_size
        ):
            raise SMLArtifactError(f"invalid SWAG {name}")
    if len(set(special_ids.values())) != len(special_ids):
        raise SMLArtifactError("SWAG special token IDs must be unique")


def _cache_key(config: SwagPreparationConfig, base: ResolvedModel) -> str:
    return structured_identity(
        "sml-swag-cache-key-v1",
        {
            "source": _source_request_projection(config.source),
            "preprocessing": _preprocessing_projection(config),
            "tokenizer_identity": base.tokenizer.manifest.identity,
            "base_identity": _base_identity(base),
            "special_ids": _special_ids(base),
        },
    )


def _requested_projections(
    config: SwagPreparationConfig, base: ResolvedModel
) -> dict[str, object]:
    return {
        "source": _source_request_projection(config.source),
        "preprocessing": _preprocessing_projection(config),
        "base_identity": _base_identity(base),
        "tokenizer_identity": base.tokenizer.manifest.identity,
        **_special_ids(base),
    }


def _manifest_projections(manifest: SwagDataManifest) -> dict[str, object]:
    return {
        "source": _source_manifest_projection(manifest.source),
        "preprocessing": dict(manifest.preprocessing),
        "base_identity": manifest.base_identity,
        "tokenizer_identity": manifest.tokenizer_identity,
        "vocab_size": manifest.vocab_size,
        "bos_token_id": manifest.bos_token_id,
        "eos_token_id": manifest.eos_token_id,
        "pad_token_id": manifest.pad_token_id,
        "unk_token_id": manifest.unk_token_id,
    }


def _encode_text(processor: object, text: str) -> tuple[int, ...]:
    try:
        encoded = processor.encode(text)
    except Exception as error:
        raise SMLDataError("tokenizer failed while encoding SWAG text") from error
    try:
        return tuple(int(token) for token in encoded)
    except TypeError as error:
        raise SMLDataError(
            "tokenizer produced a non-iterable token ID range"
        ) from error


def _parse_row(row: Mapping[str, object]) -> tuple[str, tuple[str, ...], int]:
    if not isinstance(row, Mapping):
        raise SMLDataError("SWAG row must be a mapping")
    context = row.get("context")
    if not isinstance(context, str):
        raise SMLDataError("SWAG context must be a string")
    endings = row.get("endings")
    if not isinstance(endings, Sequence) or isinstance(endings, (str, bytes)):
        raise SMLDataError("SWAG row must contain four candidate endings")
    if len(endings) != 4:
        raise SMLDataError("SWAG row must contain four candidate endings")
    normalized: list[str] = []
    for ending in endings:
        if not isinstance(ending, str):
            raise SMLDataError("SWAG endings must be strings")
        normalized.append(ending.lstrip())
    label = row.get("label")
    if isinstance(label, bool) or not isinstance(label, int) or not 0 <= label < 4:
        raise SMLDataError("SWAG label must be an integer in 0..3")
    return context.rstrip(), tuple(normalized), label


def _select_bucket(length: int, boundaries: tuple[int, ...]) -> int:
    for boundary in boundaries:
        if length <= boundary:
            return boundary
    raise SMLDataError("encoded SWAG candidate did not fit any bucket")


def _encode_candidates(
    *,
    context: str,
    endings: tuple[str, ...],
    processor: object,
    bos_token_id: int,
    eos_token_id: int,
    vocab_size: int,
    maximum_length: int,
) -> tuple[tuple[tuple[int, ...], tuple[int, int]], ...] | None:
    context_ids = _encode_text(processor, context)
    encoded_endings: list[tuple[int, ...]] = []
    for ending in endings:
        ending_ids = _encode_text(processor, ending)
        if not ending_ids:
            raise SMLDataError("SWAG candidate is missing scored continuation tokens")
        encoded_endings.append(ending_ids)

    candidates: list[tuple[tuple[int, ...], tuple[int, int]]] = []
    for ending_ids in encoded_endings:
        token_ids = (bos_token_id, *context_ids, *ending_ids, eos_token_id)
        for token_id in token_ids:
            if token_id < 0 or token_id >= vocab_size:
                raise SMLDataError("SWAG token id is outside the tokenizer vocabulary")
        if len(token_ids) > maximum_length:
            return None
        continuation_start = 1 + len(context_ids)
        candidates.append((token_ids, (continuation_start, len(token_ids))))
    return tuple(candidates)


def _pad_candidate(
    token_ids: tuple[int, ...],
    continuation: tuple[int, int],
    *,
    bucket_length: int,
    pad_token_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    input_ids = np.full(bucket_length, pad_token_id, dtype=_INT32)
    valid_token_mask = np.zeros(bucket_length, dtype=_BOOL)
    score_mask = np.zeros(bucket_length, dtype=_BOOL)
    length = len(token_ids)
    input_ids[:length] = np.asarray(token_ids, dtype=_INT32)
    valid_token_mask[:length] = True
    start, end = continuation
    if end > start:
        score_mask[start:end] = True
    score_mask[0] = False
    if not bool(score_mask.any()):
        raise SMLDataError("SWAG candidate is missing scored continuation tokens")
    return input_ids, valid_token_mask, score_mask


def _write_npy(path: Path, array: np.ndarray, logical_path: str) -> ArrayPayloadRef:
    contiguous = np.ascontiguousarray(array)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as destination:
        np.save(destination, contiguous, allow_pickle=False)
    with path.open("rb") as payload:
        identity = file_identity(payload)
    dtype_name = "int32" if contiguous.dtype == _INT32 else "bool"
    return ArrayPayloadRef(
        payload=PayloadRef(
            logical_path=logical_path,
            identity=identity,
            byte_size=path.stat().st_size,
        ),
        arrays=(
            ArraySpec(
                name=path.stem,
                shape=tuple(int(dimension) for dimension in contiguous.shape),
                dtype=dtype_name,
            ),
        ),
    )


def _load_npy(path: Path, *, dtype: np.dtype, shape: tuple[int, ...]) -> np.ndarray:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if tuple(array.shape) != shape:
        raise SMLArtifactError(f"SWAG array shape mismatch: {path.name}")
    if np.dtype(array.dtype) != dtype:
        raise SMLArtifactError(f"SWAG array dtype mismatch: {path.name}")
    return array


def _logical_array_path(length: int, name: str) -> str:
    return f"buckets/length-{length:04d}/{name}.npy"


def _group_manifest_buckets(
    manifest: SwagDataManifest,
) -> tuple[tuple[int, dict[str, ArrayPayloadRef]], ...]:
    grouped: dict[int, dict[str, ArrayPayloadRef]] = {}
    for reference in manifest.buckets:
        parts = reference.payload.logical_path.split("/")
        if (
            len(parts) != 3
            or parts[0] != "buckets"
            or not parts[1].startswith("length-")
            or not parts[2].endswith(".npy")
        ):
            raise SMLArtifactError(
                f"invalid SWAG bucket path: {reference.payload.logical_path}"
            )
        length = int(parts[1].removeprefix("length-"))
        name = Path(parts[2]).stem
        if name not in _ARRAY_NAMES:
            raise SMLArtifactError(f"unexpected SWAG array name: {name}")
        bucket = grouped.setdefault(length, {})
        if name in bucket:
            raise SMLArtifactError(f"duplicate SWAG array in bucket {length}: {name}")
        bucket[name] = reference
    ordered: list[tuple[int, dict[str, ArrayPayloadRef]]] = []
    for length in sorted(grouped):
        arrays = grouped[length]
        if set(arrays) != set(_ARRAY_NAMES):
            raise SMLArtifactError(f"SWAG bucket {length} is missing arrays")
        ordered.append((length, arrays))
    if not ordered:
        raise SMLArtifactError("SWAG bundle does not contain any buckets")
    return tuple(ordered)


def _open_buckets(path: Path, manifest: SwagDataManifest) -> tuple[SwagBucket, ...]:
    buckets: list[SwagBucket] = []
    total_examples = 0
    for length, arrays in _group_manifest_buckets(manifest):
        input_ids = _load_npy(
            path / arrays["input_ids"].payload.logical_path,
            dtype=_INT32,
            shape=tuple(arrays["input_ids"].arrays[0].shape),
        )
        valid_token_mask = _load_npy(
            path / arrays["valid_token_mask"].payload.logical_path,
            dtype=_BOOL,
            shape=tuple(arrays["valid_token_mask"].arrays[0].shape),
        )
        score_mask = _load_npy(
            path / arrays["score_mask"].payload.logical_path,
            dtype=_BOOL,
            shape=tuple(arrays["score_mask"].arrays[0].shape),
        )
        labels = _load_npy(
            path / arrays["labels"].payload.logical_path,
            dtype=_INT32,
            shape=tuple(arrays["labels"].arrays[0].shape),
        )
        example_count = int(input_ids.shape[0])
        expected = (example_count, 4, length)
        if (
            input_ids.shape != expected
            or valid_token_mask.shape != expected
            or score_mask.shape != expected
            or labels.shape != (example_count,)
        ):
            raise SMLArtifactError("SWAG bucket arrays have incompatible shapes")
        if bool(np.any(score_mask[:, :, 0])):
            raise SMLArtifactError("SWAG score mask must be false at position zero")
        if example_count and (int(labels.min()) < 0 or int(labels.max()) >= 4):
            raise SMLArtifactError("SWAG labels must be in 0..3")
        buckets.append(
            SwagBucket(
                length=length,
                input_ids=input_ids,
                valid_token_mask=valid_token_mask,
                score_mask=score_mask,
                labels=labels,
            )
        )
        total_examples += example_count
    if total_examples != manifest.example_count:
        raise SMLArtifactError("SWAG example_count does not match bucket rows")
    return tuple(buckets)


def load_swag_bundle(path: Path, verification: VerificationLevel) -> SwagDataBundle:
    if not isinstance(path, Path):
        raise TypeError("path must be a Path")
    if not isinstance(verification, VerificationLevel):
        raise TypeError("verification must be a VerificationLevel")
    verified = read_manifest(path, SwagDataManifest, verification)
    _validate_recorded_projections(verified.manifest)
    return SwagDataBundle(
        path=path,
        manifest=verified.manifest,
        verification=verified.verification,
        buckets=_open_buckets(path, verified.manifest),
    )


def _provider_failure_message(
    config: SwagPreparationConfig, base: ResolvedModel, error: BaseException
) -> str:
    source = config.source
    cache_key = _cache_key(config, base)
    detail = f": {error}" if str(error) else ""
    return (
        "SWAG provider unavailable for "
        f"namespace={source.namespace!r} dataset_config={source.dataset_config!r} "
        f"revision={source.revision!r}; cache key {cache_key}{detail}"
    )


def _iter_resolved_rows(
    config: SwagPreparationConfig,
    base: ResolvedModel,
    resolved: ResolvedSwagSource,
) -> Iterator[Mapping[str, object]]:
    try:
        rows = config.provider.iter_rows(resolved)
    except SMLDataError:
        raise
    except Exception as error:
        raise SMLDataError(_provider_failure_message(config, base, error)) from error
    try:
        yield from rows
    except SMLDataError:
        raise
    except Exception as error:
        raise SMLDataError(_provider_failure_message(config, base, error)) from error


def _build_encoded_rows(
    config: SwagPreparationConfig,
    base: ResolvedModel,
    resolved: ResolvedSwagSource,
) -> tuple[dict[int, list[tuple[np.ndarray, np.ndarray, np.ndarray, int]]], int]:
    special = _special_ids(base)
    processor = base.tokenizer.processor
    grouped: dict[int, list[tuple[np.ndarray, np.ndarray, np.ndarray, int]]] = {
        boundary: [] for boundary in config.bucket_boundaries
    }
    dropped = 0
    kept = 0
    for row in _iter_resolved_rows(config, base, resolved):
        context, endings, label = _parse_row(row)
        encoded = _encode_candidates(
            context=context,
            endings=endings,
            processor=processor,
            bos_token_id=special["bos_token_id"],
            eos_token_id=special["eos_token_id"],
            vocab_size=special["vocab_size"],
            maximum_length=config.maximum_length,
        )
        if encoded is None:
            dropped += 1
            continue
        longest = max(len(token_ids) for token_ids, _continuation in encoded)
        bucket_length = _select_bucket(longest, config.bucket_boundaries)
        padded_ids = []
        padded_valid = []
        padded_score = []
        for token_ids, continuation in encoded:
            input_ids, valid_token_mask, score_mask = _pad_candidate(
                token_ids,
                continuation,
                bucket_length=bucket_length,
                pad_token_id=special["pad_token_id"],
            )
            padded_ids.append(input_ids)
            padded_valid.append(valid_token_mask)
            padded_score.append(score_mask)
        grouped[bucket_length].append(
            (
                np.stack(padded_ids, axis=0),
                np.stack(padded_valid, axis=0),
                np.stack(padded_score, axis=0),
                label,
            )
        )
        kept += 1
        if config.maximum_examples is not None and kept >= config.maximum_examples:
            break
    if kept == 0:
        raise SMLDataError("no usable SWAG examples were produced")
    return grouped, dropped


def _write_buckets(
    grouped: Mapping[int, list[tuple[np.ndarray, np.ndarray, np.ndarray, int]]],
    private_path: Path,
) -> tuple[tuple[ArrayPayloadRef, ...], int]:
    references: list[ArrayPayloadRef] = []
    example_count = 0
    for length in sorted(grouped):
        rows = grouped[length]
        if not rows:
            continue
        input_ids = np.stack([row[0] for row in rows], axis=0).astype(
            _INT32, copy=False
        )
        valid_token_mask = np.stack([row[1] for row in rows], axis=0).astype(
            _BOOL, copy=False
        )
        score_mask = np.stack([row[2] for row in rows], axis=0).astype(
            _BOOL, copy=False
        )
        labels = np.asarray([row[3] for row in rows], dtype=_INT32)
        arrays = {
            "input_ids": input_ids,
            "valid_token_mask": valid_token_mask,
            "score_mask": score_mask,
            "labels": labels,
        }
        for name, array in arrays.items():
            logical_path = _logical_array_path(length, name)
            references.append(
                _write_npy(private_path / logical_path, array, logical_path)
            )
        example_count += int(labels.shape[0])
    if example_count == 0:
        raise SMLDataError("no usable SWAG examples were produced")
    return tuple(references), example_count


def prepare_swag_bundle(
    config: SwagPreparationConfig,
    base: ResolvedModel,
    output: Path,
) -> SwagDataBundle:
    if not isinstance(config, SwagPreparationConfig):
        raise TypeError("config must be a SwagPreparationConfig")
    if not isinstance(base, ResolvedModel):
        raise TypeError("base must be a ResolvedModel")
    if not isinstance(output, Path):
        raise TypeError("output must be a Path")

    _require_full_base(base)
    _require_tokenizer_matches_model(base)

    if config.maximum_length > base.model_config.effective_context_length:
        raise SMLDataError(
            "SWAG maximum_length exceeds the base model effective context length"
        )
    if config.bucket_boundaries[-1] < config.maximum_length:
        raise SMLDataError("bucket_boundaries must cover maximum_length")

    if output.exists():
        existing = load_swag_bundle(output, VerificationLevel.FULL)
        if canonical_json_bytes(
            _manifest_projections(existing.manifest)
        ) != canonical_json_bytes(_requested_projections(config, base)):
            raise SMLArtifactError(
                f"existing target has a different identity collision: {output}"
            )
        return existing

    try:
        resolved = config.provider.resolve(config.source)
    except Exception as error:
        raise SMLDataError(_provider_failure_message(config, base, error)) from error
    _require_resolved_matches_request(resolved, config.source)

    special = _special_ids(base)

    def build(private_path: Path) -> SwagDataManifest:
        grouped, dropped = _build_encoded_rows(config, base, resolved)
        bucket_refs, example_count = _write_buckets(grouped, private_path)
        source = _resolved_source_projection(resolved)
        manifest = SwagDataManifest(
            kind="swag-data",
            version=1,
            identity=_PLACEHOLDER_IDENTITY,
            source=source,
            preprocessing=_preprocessing_projection(config),
            base_identity=_base_identity(base),
            tokenizer_identity=base.tokenizer.manifest.identity,
            vocab_size=special["vocab_size"],
            bos_token_id=special["bos_token_id"],
            eos_token_id=special["eos_token_id"],
            pad_token_id=special["pad_token_id"],
            unk_token_id=special["unk_token_id"],
            example_count=example_count,
            dropped_overlength_rows=dropped,
            buckets=bucket_refs,
        )
        return replace(manifest, identity=manifest.recompute_identity())

    published = publish_immutable_bundle(output, build)
    return SwagDataBundle(
        path=published.path,
        manifest=published.manifest,
        verification=published.verification,
        buckets=_open_buckets(published.path, published.manifest),
    )


__all__ = [
    "BOS_POLICY_V1",
    "EOS_POLICY_V1",
    "JOIN_POLICY_V1",
    "OVERLENGTH_POLICY_V1",
    "SWAG_IDENTITY_FIELDS",
    "HuggingFaceDatasetsSwagProvider",
    "ResolvedSwagSource",
    "SwagBucket",
    "SwagDataBundle",
    "SwagPreparationConfig",
    "SwagProvider",
    "SwagSourceConfig",
    "load_swag_bundle",
    "prepare_swag_bundle",
]
