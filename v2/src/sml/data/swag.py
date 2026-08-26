"""Immutable encoded SWAG buckets with offline cache identity."""

from __future__ import annotations

import hashlib
import importlib.metadata
import mmap
import queue
import tempfile
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from math import prod
from pathlib import Path
from traceback import clear_frames
from typing import Literal, Protocol, Self

import numpy as np

from sml.artifacts.checkpoint import publish_immutable_bundle
from sml.artifacts.manifest import (
    ArrayPayloadRef,
    ArraySpec,
    ArtifactRoot,
    OpenedArtifact,
    PayloadRef,
    SwagDataManifest,
    VerificationLevel,
    VerifiedPayload,
    canonical_json_bytes,
    open_artifact,
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
_INGEST_CHUNK_SIZE = 1024
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


class _OwnedNpyMapping:
    """One read-only NPY view backed by its proven payload descriptor."""

    __slots__ = ("_closed", "array", "logical_path", "mapping", "payload")

    def __init__(
        self,
        logical_path: str,
        payload: VerifiedPayload,
        mapping: mmap.mmap,
        array: np.ndarray,
    ) -> None:
        self.logical_path = logical_path
        self.payload = payload
        self.mapping = mapping
        self.array = array
        self._closed = False

    @classmethod
    def open(
        cls,
        artifact: OpenedArtifact[SwagDataManifest],
        reference: ArrayPayloadRef,
    ) -> _OwnedNpyMapping:
        if len(reference.arrays) != 1:
            raise SMLArtifactError(
                f"SWAG NPY payload must declare one array: {reference.payload.logical_path}"
            )
        spec = reference.arrays[0]
        expected_dtypes = {"int32": _INT32, "bool": _BOOL}
        try:
            expected_dtype = expected_dtypes[spec.dtype]
        except KeyError as error:
            raise SMLArtifactError(
                f"unsupported SWAG array dtype: {reference.payload.logical_path}"
            ) from error
        expected_shape = tuple(spec.shape)

        payload = artifact.open_payload(reference.payload)
        mapping: mmap.mmap | None = None
        array: np.ndarray | None = None
        try:
            stream = payload.stream
            try:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(
                        stream
                    )
                elif version == (2, 0):
                    shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(
                        stream
                    )
                else:
                    raise SMLArtifactError(f"unsupported SWAG NPY version: {version}")
                data_offset = stream.tell()
            except SMLArtifactError:
                raise
            except (EOFError, OSError, TypeError, ValueError) as error:
                raise SMLArtifactError(
                    f"invalid SWAG array NPY header: {reference.payload.logical_path}"
                ) from error

            parsed_dtype = np.dtype(dtype)
            if fortran_order:
                raise SMLArtifactError(
                    f"SWAG array must use C order: {reference.payload.logical_path}"
                )
            if parsed_dtype.hasobject or parsed_dtype != expected_dtype:
                raise SMLArtifactError(
                    f"SWAG array dtype mismatch: {reference.payload.logical_path}"
                )
            if tuple(shape) != expected_shape:
                raise SMLArtifactError(
                    f"SWAG array shape mismatch: {reference.payload.logical_path}"
                )
            element_count = prod(expected_shape)
            expected_size = data_offset + element_count * parsed_dtype.itemsize
            if expected_size != payload.opened_stat.st_size:
                raise SMLArtifactError(
                    f"SWAG array payload size mismatch: {reference.payload.logical_path}"
                )

            mapping = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
            array = np.ndarray(
                expected_shape,
                dtype=parsed_dtype,
                buffer=mapping,
                offset=data_offset,
                order="C",
            )
            array.setflags(write=False)
            return cls(reference.payload.logical_path, payload, mapping, array)
        except BaseException as error:
            if error.__traceback__ is not None:
                clear_frames(error.__traceback__)
            array = None
            cleanup_errors: list[BaseException] = []
            try:
                if mapping is not None:
                    mapping.close()
            except BaseException as cleanup_error:  # noqa: BLE001 - cleanup continues
                cleanup_errors.append(cleanup_error)
            try:
                payload.close()
            except BaseException as cleanup_error:  # noqa: BLE001 - cleanup continues
                cleanup_errors.append(cleanup_error)
            if cleanup_errors:
                raise error from cleanup_errors[0]
            raise

    def _release_view(self) -> None:
        self.array = np.empty((0,), dtype=self.array.dtype)
        self.array.setflags(write=False)

    def _close_mapping(self) -> None:
        self.mapping.close()

    def _close_payload(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.payload.close()

    def close(self) -> None:
        if self._closed:
            return
        errors: list[BaseException] = []
        try:
            self._release_view()
        except BaseException as error:  # noqa: BLE001 - cleanup must continue
            errors.append(error)
        try:
            self._close_mapping()
        except BaseException as error:  # noqa: BLE001 - cleanup must continue
            errors.append(error)
        try:
            self._close_payload()
        except BaseException as error:  # noqa: BLE001 - cleanup must continue
            errors.append(error)
        if errors:
            raise errors[0]

    def __enter__(self) -> Self:
        if self._closed:
            raise SMLArtifactError("SWAG NPY mapping is closed")
        return self

    def __exit__(
        self,
        exception_type: object,
        exception: object,
        traceback: object,
    ) -> None:
        try:
            self.close()
        except BaseException as close_error:
            if isinstance(exception, BaseException):
                raise exception from close_error
            raise


def _close_swag_resources(
    mappings: Sequence[_OwnedNpyMapping],
    root: ArtifactRoot | None,
) -> None:
    errors: list[BaseException] = []
    for owner in reversed(mappings):
        try:
            owner._release_view()
        except BaseException as error:  # noqa: BLE001 - cleanup continues
            errors.append(error)
    for owner in reversed(mappings):
        try:
            owner._close_mapping()
        except BaseException as error:  # noqa: BLE001 - cleanup continues
            errors.append(error)
    for owner in reversed(mappings):
        try:
            owner._close_payload()
        except BaseException as error:  # noqa: BLE001 - cleanup continues
            errors.append(error)
    if root is not None:
        try:
            root.close()
        except BaseException as error:  # noqa: BLE001 - cleanup completes
            errors.append(error)
    if errors:
        raise errors[0]


@dataclass(frozen=True, slots=True)
class SwagDataBundle:
    path: Path
    manifest: SwagDataManifest
    verification: VerificationLevel
    _buckets: tuple[SwagBucket, ...]
    _root: ArtifactRoot = field(repr=False, compare=False)
    _mappings: tuple[_OwnedNpyMapping, ...] = field(repr=False, compare=False)
    _closed: bool = field(default=False, init=False, repr=False, compare=False)

    @property
    def buckets(self) -> tuple[SwagBucket, ...]:
        if self._closed:
            raise SMLDataError("SWAG data bundle is closed")
        return self._buckets

    def close(self) -> None:
        if self._closed:
            return
        object.__setattr__(self, "_closed", True)
        mappings = self._mappings
        root = self._root
        object.__setattr__(self, "_buckets", ())
        object.__setattr__(self, "_mappings", ())
        _close_swag_resources(mappings, root)

    def __enter__(self) -> Self:
        if self._closed:
            raise SMLDataError("SWAG data bundle is closed")
        return self

    def __exit__(
        self,
        exception_type: object,
        exception: object,
        traceback: object,
    ) -> None:
        try:
            self.close()
        except BaseException as close_error:
            if isinstance(exception, BaseException):
                raise exception from close_error
            raise


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
        "maximum_examples": config.maximum_examples,
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


def _encode_texts(
    processor: object, texts: Sequence[str]
) -> tuple[tuple[int, ...], ...]:
    if not texts:
        return ()
    try:
        encoded = processor.encode(list(texts))
    except Exception as error:
        raise SMLDataError("tokenizer failed while encoding SWAG text") from error
    try:
        sequences = tuple(
            tuple(int(token) for token in sequence) for sequence in encoded
        )
    except TypeError as error:
        raise SMLDataError(
            "tokenizer produced a non-iterable token ID range"
        ) from error
    if len(sequences) != len(texts):
        raise SMLDataError("tokenizer produced a mismatched encoding batch")
    return sequences


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


def _assemble_candidates(
    *,
    context_ids: tuple[int, ...],
    encoded_endings: Sequence[tuple[int, ...]],
    bos_token_id: int,
    eos_token_id: int,
    vocab_size: int,
    maximum_length: int,
) -> tuple[tuple[tuple[int, ...], tuple[int, int]], ...] | None:
    for ending_ids in encoded_endings:
        if not ending_ids:
            raise SMLDataError("SWAG candidate is missing scored continuation tokens")

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


class _HashingWriter:
    def __init__(self, raw) -> None:
        self._raw = raw
        self._digest = hashlib.sha256()

    def write(self, data: bytes | bytearray | memoryview) -> int:
        view = memoryview(data).cast("B")
        self._digest.update(view)
        written = self._raw.write(data)
        return int(len(view) if written is None else written)

    def flush(self) -> None:
        flush = getattr(self._raw, "flush", None)
        if flush is not None:
            flush()

    def identity(self) -> str:
        return f"sha256:{self._digest.hexdigest()}"


def _write_npy(path: Path, array: np.ndarray, logical_path: str) -> ArrayPayloadRef:
    contiguous = np.ascontiguousarray(array)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as destination:
        writer = _HashingWriter(destination)
        np.save(writer, contiguous, allow_pickle=False)
        writer.flush()
        identity = writer.identity()
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


def _validate_bucket_arrays(
    *,
    input_ids: np.ndarray,
    valid_token_mask: np.ndarray,
    score_mask: np.ndarray,
    vocab_size: int,
    pad_token_id: int,
    eos_token_id: int,
    bos_token_id: int,
    maximum_length: int,
    bucket_length: int,
    bucket_boundaries: Sequence[int],
) -> None:
    if bucket_length not in bucket_boundaries:
        raise SMLArtifactError("SWAG bucket length is not a declared boundary")
    if np.any(input_ids < 0) or np.any(input_ids >= vocab_size):
        raise SMLArtifactError("SWAG token id is outside the tokenizer vocabulary")
    invalid = ~valid_token_mask
    if np.any(score_mask & invalid):
        raise SMLArtifactError("SWAG score mask is true where the valid mask is false")
    lengths = valid_token_mask.sum(axis=-1, dtype=np.int64)
    expected_valid = np.arange(input_ids.shape[-1]) < lengths[..., None]
    if not np.array_equal(valid_token_mask, expected_valid) or np.any(
        input_ids[invalid] != pad_token_id
    ):
        raise SMLArtifactError("SWAG padding is inconsistent with the valid mask")
    if input_ids.shape[0] and np.any(~score_mask.any(axis=-1)):
        raise SMLArtifactError("SWAG candidate is missing scored continuation tokens")
    last_index = np.maximum(lengths - 1, 0)
    example_count, candidate_count, _ = input_ids.shape
    example_index = np.arange(example_count)[:, None]
    candidate_index = np.arange(candidate_count)[None, :]
    last_tokens = input_ids[example_index, candidate_index, last_index]
    last_scored = score_mask[example_index, candidate_index, last_index]
    if input_ids.shape[0] and np.any(lengths <= 0):
        raise SMLArtifactError("SWAG candidate is missing scored continuation tokens")
    if input_ids.shape[0] and np.any(input_ids[:, :, 0] != bos_token_id):
        raise SMLArtifactError("SWAG candidate does not start with BOS")
    if input_ids.shape[0] and np.any(lengths > maximum_length):
        raise SMLArtifactError("SWAG valid sequence exceeds maximum_length")
    if input_ids.shape[0] and np.any(last_tokens != eos_token_id):
        raise SMLArtifactError("SWAG candidate does not end with EOS")
    if input_ids.shape[0] and np.any(~last_scored):
        raise SMLArtifactError("SWAG EOS token is not scored")
    first_score = np.argmax(score_mask, axis=-1)
    positions = np.arange(input_ids.shape[-1])
    expected_score = (positions >= first_score[..., None]) & (
        positions < lengths[..., None]
    )
    if input_ids.shape[0] and not np.array_equal(score_mask, expected_score):
        raise SMLArtifactError(
            "SWAG score mask must be a contiguous suffix of the valid prefix"
        )


def _open_buckets(
    artifact: OpenedArtifact[SwagDataManifest],
    manifest: SwagDataManifest,
) -> tuple[tuple[SwagBucket, ...], tuple[_OwnedNpyMapping, ...]]:
    buckets: list[SwagBucket] = []
    mappings: list[_OwnedNpyMapping] = []
    owners: dict[str, _OwnedNpyMapping] = {}
    owner: _OwnedNpyMapping | None = None
    input_ids: np.ndarray | None = None
    valid_token_mask: np.ndarray | None = None
    score_mask: np.ndarray | None = None
    labels: np.ndarray | None = None
    total_examples = 0
    try:
        for length, arrays in _group_manifest_buckets(manifest):
            owners = {}
            for name in _ARRAY_NAMES:
                owner = _OwnedNpyMapping.open(artifact, arrays[name])
                owners[name] = owner
                mappings.append(owner)
            input_ids = owners["input_ids"].array
            valid_token_mask = owners["valid_token_mask"].array
            score_mask = owners["score_mask"].array
            labels = owners["labels"].array
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
            _validate_bucket_arrays(
                input_ids=input_ids,
                valid_token_mask=valid_token_mask,
                score_mask=score_mask,
                vocab_size=manifest.vocab_size,
                pad_token_id=manifest.pad_token_id,
                eos_token_id=manifest.eos_token_id,
                bos_token_id=manifest.bos_token_id,
                maximum_length=manifest.preprocessing["maximum_length"],
                bucket_length=length,
                bucket_boundaries=manifest.preprocessing["bucket_boundaries"],
            )
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
        return tuple(buckets), tuple(mappings)
    except BaseException as error:
        if error.__traceback__ is not None:
            clear_frames(error.__traceback__)
        buckets.clear()
        owners.clear()
        owner = None
        input_ids = None
        valid_token_mask = None
        score_mask = None
        labels = None
        try:
            _close_swag_resources(mappings, None)
        except BaseException as cleanup_error:
            raise error from cleanup_error
        raise


def _load_swag_bundle(
    path: Path,
    verification: VerificationLevel,
    *,
    validate_projections: bool,
) -> SwagDataBundle:
    if not isinstance(path, Path):
        raise TypeError("path must be a Path")
    if not isinstance(verification, VerificationLevel):
        raise TypeError("verification must be a VerificationLevel")
    artifact = open_artifact(path, (SwagDataManifest,), verification)
    buckets: tuple[SwagBucket, ...] = ()
    mappings: tuple[_OwnedNpyMapping, ...] = ()
    detached_root: ArtifactRoot | None = None
    try:
        if validate_projections:
            _validate_recorded_projections(artifact.manifest)
        buckets, mappings = _open_buckets(artifact, artifact.manifest)
        detached_root = artifact.detach_root()
        bundle = SwagDataBundle(
            path,
            artifact.manifest,
            artifact.verification,
            buckets,
            detached_root,
            mappings,
        )
        mappings = ()
        detached_root = None
        return bundle
    except BaseException as error:
        if error.__traceback__ is not None:
            clear_frames(error.__traceback__)
        buckets = ()
        cleanup_errors: list[BaseException] = []
        try:
            _close_swag_resources(mappings, detached_root)
        except BaseException as cleanup_error:  # noqa: BLE001 - cleanup continues
            cleanup_errors.append(cleanup_error)
        try:
            artifact.close()
        except BaseException as cleanup_error:  # noqa: BLE001 - cleanup completes
            cleanup_errors.append(cleanup_error)
        if cleanup_errors:
            raise error from cleanup_errors[0]
        raise


def load_swag_bundle(path: Path, verification: VerificationLevel) -> SwagDataBundle:
    return _load_swag_bundle(path, verification, validate_projections=True)


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
    except Exception as error:
        raise SMLDataError(_provider_failure_message(config, base, error)) from error
    try:
        yield from rows
    except Exception as error:
        raise SMLDataError(_provider_failure_message(config, base, error)) from error


def _close_memmap(array: np.memmap) -> None:
    array.flush()
    mmap_obj = getattr(array, "_mmap", None)
    if mmap_obj is not None:
        mmap_obj.close()


class _BucketWriter:
    def __init__(self, length: int, scratch: Path) -> None:
        self.length = length
        self.scratch = scratch
        self.count = 0
        self._pending_ids: list[np.ndarray] = []
        self._pending_valid: list[np.ndarray] = []
        self._pending_score: list[np.ndarray] = []
        self._pending_labels: list[int] = []
        self._capacity = 0
        self._maps: dict[str, np.memmap] | None = None

    def append(
        self,
        input_ids: np.ndarray,
        valid_token_mask: np.ndarray,
        score_mask: np.ndarray,
        label: int,
    ) -> None:
        self._pending_ids.append(input_ids)
        self._pending_valid.append(valid_token_mask)
        self._pending_score.append(score_mask)
        self._pending_labels.append(label)
        if len(self._pending_ids) >= _INGEST_CHUNK_SIZE:
            self.flush()

    def flush(self) -> None:
        if not self._pending_ids:
            return
        n = len(self._pending_ids)
        self._ensure_capacity(self.count + n)
        start = self.count
        stop = start + n
        assert self._maps is not None
        self._maps["input_ids"][start:stop] = np.stack(self._pending_ids, axis=0)
        self._maps["valid_token_mask"][start:stop] = np.stack(
            self._pending_valid, axis=0
        )
        self._maps["score_mask"][start:stop] = np.stack(self._pending_score, axis=0)
        self._maps["labels"][start:stop] = np.asarray(
            self._pending_labels, dtype=_INT32
        )
        self.count = stop
        self._pending_ids.clear()
        self._pending_valid.clear()
        self._pending_score.clear()
        self._pending_labels.clear()

    def _ensure_capacity(self, needed: int) -> None:
        if self._capacity >= needed:
            return
        capacity = 8 if self._capacity == 0 else self._capacity
        while capacity < needed:
            capacity *= 2
        specs = (
            ("input_ids", _INT32, (4, self.length)),
            ("valid_token_mask", _BOOL, (4, self.length)),
            ("score_mask", _BOOL, (4, self.length)),
            ("labels", _INT32, ()),
        )
        new_maps: dict[str, np.memmap] = {}
        for name, dtype, extra in specs:
            path = self.scratch / f"length-{self.length:04d}-{name}-{capacity}.dat"
            mm = np.memmap(path, mode="w+", dtype=dtype, shape=(capacity, *extra))
            if self._maps is not None:
                mm[: self.count] = self._maps[name][: self.count]
                old_path = Path(str(self._maps[name].filename))
                _close_memmap(self._maps[name])
                old_path.unlink(missing_ok=True)
            new_maps[name] = mm
        self._maps = new_maps
        self._capacity = capacity

    def arrays(self) -> dict[str, np.ndarray] | None:
        self.flush()
        if self.count == 0 or self._maps is None:
            return None
        return {name: mm[: self.count] for name, mm in self._maps.items()}

    def close(self) -> None:
        if self._maps is None:
            return
        for mm in self._maps.values():
            _close_memmap(mm)
        self._maps = None


def _ingest_and_write_buckets(
    config: SwagPreparationConfig,
    base: ResolvedModel,
    resolved: ResolvedSwagSource,
    private_path: Path,
) -> tuple[tuple[ArrayPayloadRef, ...], int, int]:
    special = _special_ids(base)
    processor = base.tokenizer.processor
    with tempfile.TemporaryDirectory(prefix="swag-ingest-") as scratch_name:
        scratch = Path(scratch_name)
        writers = {
            boundary: _BucketWriter(boundary, scratch)
            for boundary in config.bucket_boundaries
        }
        dropped = 0
        kept = 0
        parsed_rows: list[tuple[str, tuple[str, ...], int]] = []

        def flush_parsed() -> bool:
            nonlocal dropped, kept
            if not parsed_rows:
                return False
            texts: list[str] = []
            for context, endings, _label in parsed_rows:
                texts.append(context)
                texts.extend(endings)
            encoded_texts = _encode_texts(processor, texts)
            offset = 0
            stop = False
            for _context, _endings, label in parsed_rows:
                context_ids = encoded_texts[offset]
                encoded_endings = encoded_texts[offset + 1 : offset + 5]
                offset += 5
                encoded = _assemble_candidates(
                    context_ids=context_ids,
                    encoded_endings=encoded_endings,
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
                writers[bucket_length].append(
                    np.stack(padded_ids, axis=0),
                    np.stack(padded_valid, axis=0),
                    np.stack(padded_score, axis=0),
                    label,
                )
                kept += 1
                if (
                    config.maximum_examples is not None
                    and kept >= config.maximum_examples
                ):
                    stop = True
                    break
            parsed_rows.clear()
            return stop

        try:
            for row in _iter_resolved_rows(config, base, resolved):
                parsed_rows.append(_parse_row(row))
                if len(parsed_rows) >= _INGEST_CHUNK_SIZE and flush_parsed():
                    break
            else:
                flush_parsed()
            if kept == 0:
                raise SMLDataError("no usable SWAG examples were produced")
            references: list[ArrayPayloadRef] = []
            example_count = 0
            for length in sorted(writers):
                writer = writers[length]
                arrays = writer.arrays()
                if arrays is None:
                    continue
                for name in _ARRAY_NAMES:
                    logical_path = _logical_array_path(length, name)
                    references.append(
                        _write_npy(
                            private_path / logical_path,
                            arrays[name],
                            logical_path,
                        )
                    )
                example_count += writer.count
            if example_count == 0:
                raise SMLDataError("no usable SWAG examples were produced")
            return tuple(references), example_count, dropped
        finally:
            for writer in writers.values():
                writer.close()


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
            error = SMLArtifactError(
                f"existing target has a different identity collision: {output}"
            )
            try:
                existing.close()
            except BaseException as close_error:
                raise error from close_error
            raise error
        return existing

    try:
        resolved = config.provider.resolve(config.source)
    except Exception as error:
        raise SMLDataError(_provider_failure_message(config, base, error)) from error
    _require_resolved_matches_request(resolved, config.source)

    special = _special_ids(base)

    def build(private_path: Path) -> SwagDataManifest:
        bucket_refs, example_count, dropped = _ingest_and_write_buckets(
            config, base, resolved, private_path
        )
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
    return _load_swag_bundle(
        published.path,
        published.verification,
        validate_projections=True,
    )


_EMPTY_INT = np.empty((0, 0, 0), dtype=_INT32)
_EMPTY_INT.setflags(write=False)
_EMPTY_BOOL = np.empty((0, 0, 0), dtype=_BOOL)
_EMPTY_BOOL.setflags(write=False)
_EMPTY_LABELS = np.empty((0,), dtype=_INT32)
_EMPTY_LABELS.setflags(write=False)
_EMPTY_MASK = np.empty((0,), dtype=_BOOL)
_EMPTY_MASK.setflags(write=False)
_QUEUE_STOP = object()


def _owned_readonly(array: np.ndarray) -> np.ndarray:
    owned = np.array(array, copy=True)
    owned.setflags(write=False)
    return owned


def _readonly_numpy_array(
    array: object,
    name: str,
    *,
    ndim: int,
    dtype: np.dtype,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if array.ndim != ndim:
        raise ValueError(f"{name} must be a {ndim}-dimensional array")
    if array.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}")
    if shape is not None and tuple(array.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}")
    readonly = array.view()
    readonly.setflags(write=False)
    return readonly


def _envelope_array(
    array: object,
    name: str,
    *,
    ndim: int,
    dtype: np.dtype,
    shape: tuple[int, ...] | None = None,
    copy: bool,
) -> np.ndarray:
    if not copy:
        if not isinstance(array, np.ndarray):
            raise TypeError(f"{name} must be a NumPy array")
        if array.flags.writeable or not array.flags.owndata:
            raise ValueError(f"{name} must be non-writeable owned storage")
    validated = _readonly_numpy_array(array, name, ndim=ndim, dtype=dtype, shape=shape)
    if copy:
        return _owned_readonly(validated)
    return validated


@dataclass(frozen=True, slots=True)
class SwagCursor:
    """Canonical location of the next real SWAG example, never a padded slot."""

    epoch: int
    bucket_order_position: int
    row_offset: int

    def __post_init__(self) -> None:
        _require_plain_int(self.epoch, "epoch")
        _require_plain_int(self.bucket_order_position, "bucket_order_position")
        _require_plain_int(self.row_offset, "row_offset")

    @classmethod
    def initial(cls) -> SwagCursor:
        return cls(epoch=0, bucket_order_position=0, row_offset=0)


def _epoch_bucket_plan(
    buckets: tuple[SwagBucket, ...],
    *,
    epoch_seed: int,
    epoch: int,
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    generator = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence([epoch_seed, epoch]))
    )
    bucket_order = tuple(int(index) for index in generator.permutation(len(buckets)))
    plan: list[tuple[int, tuple[int, ...]]] = []
    for bucket_index in bucket_order:
        row_count = int(buckets[bucket_index].input_ids.shape[0])
        if row_count == 0:
            continue
        row_permutation = tuple(
            int(index) for index in generator.permutation(row_count)
        )
        plan.append((bucket_index, row_permutation))
    return tuple(plan)


def _synthetic_candidates(
    length: int, manifest: SwagDataManifest
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    input_ids = np.full((4, length), manifest.pad_token_id, dtype=_INT32)
    input_ids[:, 0] = manifest.bos_token_id
    input_ids[:, 1] = manifest.eos_token_id
    valid_token_mask = np.zeros((4, length), dtype=_BOOL)
    valid_token_mask[:, :2] = True
    score_mask = np.zeros((4, length), dtype=_BOOL)
    score_mask[:, 1] = True
    return input_ids, valid_token_mask, score_mask


def _assemble_batch_arrays(
    bucket: SwagBucket,
    row_indices: tuple[int, ...],
    *,
    batch_size: int,
    manifest: SwagDataManifest,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    length = bucket.length
    input_ids = np.empty((batch_size, 4, length), dtype=_INT32)
    valid_token_mask = np.empty((batch_size, 4, length), dtype=_BOOL)
    score_mask = np.empty((batch_size, 4, length), dtype=_BOOL)
    labels = np.empty((batch_size,), dtype=_INT32)
    example_mask = np.zeros((batch_size,), dtype=_BOOL)
    for slot, row_index in enumerate(row_indices):
        input_ids[slot] = bucket.input_ids[row_index]
        valid_token_mask[slot] = bucket.valid_token_mask[row_index]
        score_mask[slot] = bucket.score_mask[row_index]
        labels[slot] = int(bucket.labels[row_index])
        example_mask[slot] = True
    if len(row_indices) < batch_size:
        syn_ids, syn_valid, syn_score = _synthetic_candidates(length, manifest)
        for slot in range(len(row_indices), batch_size):
            input_ids[slot] = syn_ids
            valid_token_mask[slot] = syn_valid
            score_mask[slot] = syn_score
            labels[slot] = 0
            example_mask[slot] = False
    for array in (input_ids, valid_token_mask, score_mask, labels, example_mask):
        array.setflags(write=False)
    return (
        input_ids,
        valid_token_mask,
        score_mask,
        labels,
        example_mask,
    )


def _advance_cursor(
    cursor: SwagCursor,
    *,
    consumed: int,
    remaining_in_bucket: int,
    remaining_buckets: int,
) -> SwagCursor:
    if consumed < remaining_in_bucket:
        return SwagCursor(
            cursor.epoch, cursor.bucket_order_position, cursor.row_offset + consumed
        )
    if remaining_buckets > 0:
        return SwagCursor(cursor.epoch, cursor.bucket_order_position + 1, 0)
    return SwagCursor(cursor.epoch + 1, 0, 0)


class SwagBatchEnvelope:
    """A read-only NumPy SWAG batch whose owned storage has explicit lifetime."""

    __slots__ = (
        "_cursor_after",
        "_example_mask",
        "_input_ids",
        "_labels",
        "_release_callback",
        "_release_lock",
        "_released",
        "_score_mask",
        "_source_epoch",
        "_valid_token_mask",
    )

    def __init__(
        self,
        input_ids: np.ndarray,
        score_mask: np.ndarray,
        labels: np.ndarray,
        example_mask: np.ndarray,
        valid_token_mask: np.ndarray,
        cursor_after: SwagCursor,
        *,
        source_epoch: int,
    ) -> None:
        self._initialize(
            input_ids,
            score_mask,
            labels,
            example_mask,
            valid_token_mask,
            cursor_after,
            source_epoch=source_epoch,
            copy=True,
        )

    @classmethod
    def _owned(
        cls,
        input_ids: np.ndarray,
        score_mask: np.ndarray,
        labels: np.ndarray,
        example_mask: np.ndarray,
        valid_token_mask: np.ndarray,
        cursor_after: SwagCursor,
        *,
        source_epoch: int,
    ) -> SwagBatchEnvelope:
        envelope = cls.__new__(cls)
        envelope._initialize(
            input_ids,
            score_mask,
            labels,
            example_mask,
            valid_token_mask,
            cursor_after,
            source_epoch=source_epoch,
            copy=False,
        )
        return envelope

    def _initialize(
        self,
        input_ids: np.ndarray,
        score_mask: np.ndarray,
        labels: np.ndarray,
        example_mask: np.ndarray,
        valid_token_mask: np.ndarray,
        cursor_after: SwagCursor,
        *,
        source_epoch: int,
        copy: bool,
    ) -> None:
        if not isinstance(cursor_after, SwagCursor):
            raise TypeError("cursor_after must be a SwagCursor")
        _require_plain_int(source_epoch, "source_epoch")
        input_ids = _envelope_array(
            input_ids, "input_ids", ndim=3, dtype=_INT32, copy=copy
        )
        if input_ids.shape[1] != 4:
            raise ValueError("input_ids must have 4 candidates")
        batch_shape = tuple(int(dimension) for dimension in input_ids.shape)
        label_shape = (batch_shape[0],)
        self._input_ids = input_ids
        self._score_mask = _envelope_array(
            score_mask,
            "score_mask",
            ndim=3,
            dtype=_BOOL,
            shape=batch_shape,
            copy=copy,
        )
        self._labels = _envelope_array(
            labels, "labels", ndim=1, dtype=_INT32, shape=label_shape, copy=copy
        )
        self._example_mask = _envelope_array(
            example_mask,
            "example_mask",
            ndim=1,
            dtype=_BOOL,
            shape=label_shape,
            copy=copy,
        )
        self._valid_token_mask = _envelope_array(
            valid_token_mask,
            "valid_token_mask",
            ndim=3,
            dtype=_BOOL,
            shape=batch_shape,
            copy=copy,
        )
        self._cursor_after = cursor_after
        self._source_epoch = source_epoch
        self._release_callback: Callable[[], None] | None = None
        self._release_lock = threading.Lock()
        self._released = False

    @property
    def input_ids(self) -> np.ndarray:
        return self._input_ids

    @property
    def score_mask(self) -> np.ndarray:
        return self._score_mask

    @property
    def labels(self) -> np.ndarray:
        return self._labels

    @property
    def example_mask(self) -> np.ndarray:
        return self._example_mask

    @property
    def valid_token_mask(self) -> np.ndarray:
        return self._valid_token_mask

    @property
    def cursor_after(self) -> SwagCursor:
        return self._cursor_after

    def _set_release_callback(self, callback: Callable[[], None]) -> None:
        self._release_callback = callback

    def release(self) -> None:
        callback: Callable[[], None] | None
        with self._release_lock:
            if self._released:
                return
            self._released = True
            self._input_ids = _EMPTY_INT
            self._score_mask = _EMPTY_BOOL
            self._labels = _EMPTY_LABELS
            self._example_mask = _EMPTY_MASK
            self._valid_token_mask = _EMPTY_BOOL
            callback = self._release_callback
            self._release_callback = None
        if callback is not None:
            callback()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exception_type: object,
        exception: object,
        traceback: object,
    ) -> None:
        self.release()


class SwagBatch:
    """On-device SWAG batch transferred on the main thread."""

    __slots__ = (
        "cursor_after",
        "example_mask",
        "input_ids",
        "labels",
        "score_mask",
        "valid_token_mask",
    )

    def __init__(
        self,
        input_ids: object,
        score_mask: object,
        labels: object,
        example_mask: object,
        valid_token_mask: object,
        cursor_after: SwagCursor,
    ) -> None:
        self.input_ids = input_ids
        self.score_mask = score_mask
        self.labels = labels
        self.example_mask = example_mask
        self.valid_token_mask = valid_token_mask
        self.cursor_after = cursor_after

    @classmethod
    def from_envelope(cls, envelope: SwagBatchEnvelope) -> SwagBatch:
        import mlx.core as mx

        batch = cls(
            mx.array(envelope.input_ids),
            mx.array(envelope.score_mask),
            mx.array(envelope.labels),
            mx.array(envelope.example_mask),
            mx.array(envelope.valid_token_mask),
            envelope.cursor_after,
        )
        envelope.release()
        return batch


class _ProducerFailure:
    def __init__(self, error: SMLDataError) -> None:
        self.error = error


class SwagLoaderPolicy(Protocol):
    prefetch_depth: int
    microbatch_size: int
    epoch_seed: int


def _require_swag_loader_policy(loader: object) -> SwagLoaderPolicy:
    required = ("prefetch_depth", "microbatch_size", "epoch_seed")
    missing = [name for name in required if not hasattr(loader, name)]
    if missing:
        raise TypeError(
            "loader must provide prefetch_depth, microbatch_size, and epoch_seed"
        )
    _require_plain_int(loader.prefetch_depth, "prefetch_depth", minimum=1)
    _require_plain_int(loader.microbatch_size, "microbatch_size", minimum=1)
    epoch_seed = _require_plain_int(loader.epoch_seed, "epoch_seed")
    if epoch_seed > 2**32 - 1:
        raise ValueError("epoch_seed must be an unsigned 32-bit integer")
    return loader  # type: ignore[return-value]


class SwagBatchStream:
    """Bounded CPU prefetch over permuted SWAG buckets with fixed-shape tails."""

    def __init__(
        self,
        bundle: SwagDataBundle,
        loader: SwagLoaderPolicy,
        *,
        cursor: SwagCursor,
    ) -> None:
        self._initialize(bundle, loader, cursor=cursor, owns_bundle=True)

    def _initialize(
        self,
        bundle: SwagDataBundle,
        loader: SwagLoaderPolicy,
        *,
        cursor: SwagCursor,
        owns_bundle: bool,
    ) -> None:
        if not isinstance(bundle, SwagDataBundle):
            raise TypeError("bundle must be a SwagDataBundle")
        loader = _require_swag_loader_policy(loader)
        if not isinstance(cursor, SwagCursor):
            raise TypeError("cursor must be a SwagCursor")
        if bundle._closed:
            raise SMLDataError("SWAG data bundle is closed")
        self._bundle = bundle
        self._owns_bundle = owns_bundle
        self._loader = loader
        self._stop = threading.Event()
        self._queue: queue.Queue[SwagBatchEnvelope | _ProducerFailure | object] = (
            queue.Queue(maxsize=loader.prefetch_depth)
        )
        self._consumer_lock = threading.RLock()
        self._state_lock = threading.Lock()
        self._committed_cursor = cursor
        self._initial_epoch = cursor.epoch
        self._epoch_complete = False
        self._closed = False
        self._producer: threading.Thread | None = None
        try:
            self._producer = threading.Thread(
                target=self._produce,
                name="sml-swag-prefetch",
                daemon=True,
            )
            self._producer.start()
        except BaseException as error:
            self._producer = None
            self._stop.set()
            self._closed = True
            if self._owns_bundle:
                try:
                    self._bundle.close()
                except BaseException as close_error:
                    raise error from close_error
            raise

    @classmethod
    def _borrowing_bundle(
        cls,
        bundle: SwagDataBundle,
        loader: SwagLoaderPolicy,
        *,
        cursor: SwagCursor,
    ) -> SwagBatchStream:
        stream = cls.__new__(cls)
        stream._initialize(bundle, loader, cursor=cursor, owns_bundle=False)
        return stream

    @property
    def committed_cursor(self) -> SwagCursor:
        with self._state_lock:
            return self._committed_cursor

    def _next_from_cursor(
        self, cursor: SwagCursor
    ) -> tuple[SwagBatchEnvelope, SwagCursor]:
        plan = _epoch_bucket_plan(
            self._bundle.buckets,
            epoch_seed=self._loader.epoch_seed,
            epoch=cursor.epoch,
        )
        if not plan:
            raise SMLDataError("SWAG bundle does not contain any examples")
        if cursor.bucket_order_position >= len(plan):
            return self._next_from_cursor(SwagCursor(cursor.epoch + 1, 0, 0))
        _bucket_index, row_permutation = plan[cursor.bucket_order_position]
        if cursor.row_offset >= len(row_permutation):
            if cursor.bucket_order_position + 1 < len(plan):
                next_cursor = SwagCursor(
                    cursor.epoch, cursor.bucket_order_position + 1, 0
                )
            else:
                next_cursor = SwagCursor(cursor.epoch + 1, 0, 0)
            return self._next_from_cursor(next_cursor)
        remaining = row_permutation[cursor.row_offset :]
        take = min(self._loader.microbatch_size, len(remaining))
        selected = remaining[:take]
        bucket = self._bundle.buckets[plan[cursor.bucket_order_position][0]]
        arrays = _assemble_batch_arrays(
            bucket,
            selected,
            batch_size=self._loader.microbatch_size,
            manifest=self._bundle.manifest,
        )
        cursor_after = _advance_cursor(
            cursor,
            consumed=take,
            remaining_in_bucket=len(remaining),
            remaining_buckets=len(plan) - cursor.bucket_order_position - 1,
        )
        input_ids, valid_token_mask, score_mask, labels, example_mask = arrays
        envelope = SwagBatchEnvelope._owned(
            input_ids,
            score_mask,
            labels,
            example_mask,
            valid_token_mask,
            cursor_after,
            source_epoch=cursor.epoch,
        )
        return envelope, cursor_after

    def _put(self, item: SwagBatchEnvelope | _ProducerFailure | object) -> bool:
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.05)
                return True
            except queue.Full:
                continue
        return False

    def _produce(self) -> None:
        try:
            cursor = self._committed_cursor
            while not self._stop.is_set():
                envelope, cursor = self._next_from_cursor(cursor)
                if not self._put(envelope):
                    envelope.release()
                    return
        except BaseException as error:  # noqa: BLE001 - cross-thread error boundary
            failure = SMLDataError("SWAG batch producer failed")
            failure.__cause__ = error
            self._put(_ProducerFailure(failure))
        finally:
            self._put(_QUEUE_STOP)

    def _pull(self) -> SwagBatchEnvelope:
        while True:
            if self._closed:
                raise StopIteration
            try:
                item = self._queue.get(timeout=0.05)
            except queue.Empty:
                producer = self._producer
                if (
                    producer is not None
                    and not producer.is_alive()
                    and self._queue.empty()
                ):
                    raise StopIteration
                continue
            if isinstance(item, SwagBatchEnvelope):
                return item
            if isinstance(item, _ProducerFailure):
                try:
                    self.close()
                except BaseException as cleanup_error:
                    raise item.error from cleanup_error
                raise item.error
            if item is _QUEUE_STOP:
                raise StopIteration

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> SwagBatchEnvelope:
        with self._consumer_lock:
            if self._epoch_complete:
                self.close()
                raise StopIteration
            try:
                envelope = self._pull()
            except StopIteration:
                self.close()
                raise
            if envelope._source_epoch > self._initial_epoch:
                envelope.release()
                self._epoch_complete = True
                self.close()
                raise StopIteration
            if envelope.cursor_after.epoch > self._initial_epoch:
                self._epoch_complete = True
            return envelope

    def commit(self, cursor_after: SwagCursor) -> None:
        if not isinstance(cursor_after, SwagCursor):
            raise TypeError("cursor_after must be a SwagCursor")
        with self._state_lock:
            self._committed_cursor = cursor_after

    def close(self) -> None:
        if self._closed:
            return
        self._stop.set()
        self._closed = True
        producer = self._producer
        if producer is not None and producer is not threading.current_thread():
            producer.join()
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(item, SwagBatchEnvelope):
                item.release()
        if self._owns_bundle:
            self._bundle.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exception_type: object,
        exception: object,
        traceback: object,
    ) -> None:
        try:
            self.close()
        except BaseException as cleanup_error:
            if isinstance(exception, BaseException):
                raise exception from cleanup_error
            raise


__all__ = [
    "BOS_POLICY_V1",
    "EOS_POLICY_V1",
    "JOIN_POLICY_V1",
    "OVERLENGTH_POLICY_V1",
    "SWAG_IDENTITY_FIELDS",
    "HuggingFaceDatasetsSwagProvider",
    "ResolvedSwagSource",
    "SwagBatch",
    "SwagBatchEnvelope",
    "SwagBatchStream",
    "SwagBucket",
    "SwagCursor",
    "SwagDataBundle",
    "SwagPreparationConfig",
    "SwagProvider",
    "SwagSourceConfig",
    "load_swag_bundle",
    "prepare_swag_bundle",
]
