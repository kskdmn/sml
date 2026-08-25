"""Closed, immutable records for reproducible evaluation results."""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal

import numpy as np

from sml.artifacts.manifest import (
    VerificationLevel,
    canonical_json_bytes,
    parse_logical_path,
    structured_identity,
)
from sml.errors import SMLArtifactError, SMLRuntimeError
from sml.inference import ModelIdentity

type JsonScalar = None | bool | int | float | str
type JsonValue = JsonScalar | tuple[JsonValue, ...] | Mapping[str, JsonValue]
type JsonObject = Mapping[str, JsonValue]

_IDENTITY_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")


def _require_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _require_optional_string(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _require_string(value, name)


def _require_identity(value: object, name: str) -> str:
    if not isinstance(value, str) or _IDENTITY_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must match sha256:[0-9a-f]{{64}}")
    return value


def _require_optional_identity(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _require_identity(value, name)


def _require_tuple(value: object, name: str) -> tuple[object, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be a tuple")
    return value


def _normalize_mapping(value: object, name: str) -> JsonObject:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    normalized = normalize_json_value(value, context=name)
    if not isinstance(normalized, Mapping):  # pragma: no cover - guarded above
        raise TypeError(f"{name} must be a mapping")
    return normalized


def normalize_json_value(value: object, *, context: str) -> JsonValue:
    """Return a finite, JSON-compatible value with immutable containers."""
    if isinstance(value, np.generic):
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            normalized_float = float(value)
            if not bool(np.isfinite(value)) or not math.isfinite(normalized_float):
                raise ValueError(f"{context} contains a non-finite number")
            return normalized_float
        raise TypeError(
            f"{context} contains unsupported NumPy scalar {type(value).__name__}"
        )
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{context} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError(f"{context} object keys must be strings")
        return MappingProxyType(
            {
                key: normalize_json_value(value[key], context=f"{context}.{key}")
                for key in sorted(value)
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(
            normalize_json_value(item, context=f"{context}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(f"{context} contains unsupported value {type(value).__name__}")


def _validate_model_identity(model: ModelIdentity) -> None:
    _require_string(model.artifact_kind, "model artifact_kind")
    _require_optional_identity(model.run_identity, "model run_identity")
    if model.step is not None and (
        isinstance(model.step, bool)
        or not isinstance(model.step, int)
        or model.step < 0
    ):
        raise ValueError("model step must be a non-negative integer or null")
    _require_optional_identity(model.checkpoint_identity, "model checkpoint_identity")
    _require_optional_identity(model.run_step_identity, "model run_step_identity")
    _require_identity(model.tokenizer_identity, "model tokenizer_identity")
    if not isinstance(model.verification, VerificationLevel):
        raise TypeError("model verification must be a VerificationLevel")


@dataclass(frozen=True, slots=True)
class EvaluationSourceIdentity:
    logical_name: str
    content_identity: str

    def __post_init__(self) -> None:
        _require_string(self.logical_name, "logical_name")
        parse_logical_path(self.logical_name)
        _require_identity(self.content_identity, "content_identity")


@dataclass(frozen=True, slots=True)
class EvaluationProviderVersion:
    name: str
    version: str

    def __post_init__(self) -> None:
        _require_string(self.name, "provider name")
        _require_string(self.version, "provider version")


@dataclass(frozen=True, slots=True)
class EvaluationTaskRecord:
    task_name: str
    task_identity: str
    task_yaml: EvaluationSourceIdentity
    include_template_closure: tuple[EvaluationSourceIdentity, ...]
    task_metadata_version: str
    prompt_config: JsonObject
    few_shot_config: JsonObject
    generation_config: JsonObject
    metric_normalization_config: JsonObject
    seeds: JsonObject
    limit: int | None
    ordered_request_identity: str
    lm_eval_package_version: str
    lm_eval_source_commit: str | None
    dataset_revision: str
    dataset_fingerprint: str
    provider_versions: tuple[EvaluationProviderVersion, ...]
    metric_payload: JsonObject

    def __post_init__(self) -> None:
        _require_string(self.task_name, "task_name")
        _require_identity(self.task_identity, "task_identity")
        if not isinstance(self.task_yaml, EvaluationSourceIdentity):
            raise TypeError("task_yaml must be an EvaluationSourceIdentity")
        closure = _require_tuple(
            self.include_template_closure, "include_template_closure"
        )
        if not all(isinstance(item, EvaluationSourceIdentity) for item in closure):
            raise TypeError(
                "include_template_closure must contain EvaluationSourceIdentity values"
            )
        _require_string(self.task_metadata_version, "task_metadata_version")
        for name in (
            "prompt_config",
            "few_shot_config",
            "generation_config",
            "metric_normalization_config",
            "seeds",
            "metric_payload",
        ):
            object.__setattr__(
                self, name, _normalize_mapping(getattr(self, name), name)
            )
        if self.limit is not None and (
            isinstance(self.limit, bool)
            or not isinstance(self.limit, int)
            or self.limit <= 0
        ):
            raise ValueError("limit must be a positive integer or null")
        _require_identity(self.ordered_request_identity, "ordered_request_identity")
        _require_string(self.lm_eval_package_version, "lm_eval_package_version")
        _require_optional_string(self.lm_eval_source_commit, "lm_eval_source_commit")
        _require_string(self.dataset_revision, "dataset_revision")
        _require_string(self.dataset_fingerprint, "dataset_fingerprint")
        providers = _require_tuple(self.provider_versions, "provider_versions")
        if not all(isinstance(item, EvaluationProviderVersion) for item in providers):
            raise TypeError(
                "provider_versions must contain EvaluationProviderVersion values"
            )
        provider_names = tuple(item.name for item in providers)
        if provider_names != tuple(sorted(provider_names)) or len(
            provider_names
        ) != len(set(provider_names)):
            raise ValueError("provider_versions names must be unique and sorted")


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    kind: Literal["evaluation-result"]
    version: Literal[1]
    identity: str
    model: ModelIdentity
    tasks: tuple[EvaluationTaskRecord, ...]
    provider_result: JsonObject

    def __post_init__(self) -> None:
        if self.kind != "evaluation-result":
            raise ValueError("kind must be 'evaluation-result'")
        if (
            not isinstance(self.version, int)
            or isinstance(self.version, bool)
            or self.version != 1
        ):
            raise ValueError("version must be 1")
        _require_identity(self.identity, "identity")
        if not isinstance(self.model, ModelIdentity):
            raise TypeError("model must be a ModelIdentity")
        _validate_model_identity(self.model)
        tasks = _require_tuple(self.tasks, "tasks")
        if not tasks:
            raise ValueError("tasks must contain at least one task")
        if not all(isinstance(task, EvaluationTaskRecord) for task in tasks):
            raise TypeError("tasks must contain EvaluationTaskRecord values")
        task_names = tuple(task.task_name for task in tasks)
        if len(task_names) != len(set(task_names)):
            raise ValueError("task names must be unique")
        object.__setattr__(
            self,
            "provider_result",
            _normalize_mapping(self.provider_result, "provider_result"),
        )


def _source_projection(source: EvaluationSourceIdentity) -> dict[str, str]:
    return {
        "logical_name": source.logical_name,
        "content_identity": source.content_identity,
    }


def _model_projection(model: ModelIdentity) -> dict[str, object]:
    return {
        "artifact_kind": model.artifact_kind,
        "run_identity": model.run_identity,
        "step": model.step,
        "checkpoint_identity": model.checkpoint_identity,
        "run_step_identity": model.run_step_identity,
        "tokenizer_identity": model.tokenizer_identity,
        "verification": model.verification.value,
    }


def _task_projection(
    record: EvaluationTaskRecord, *, include_identity: bool, include_metrics: bool
) -> dict[str, object]:
    projection: dict[str, object] = {
        "task_name": record.task_name,
        "task_yaml": _source_projection(record.task_yaml),
        "include_template_closure": [
            _source_projection(source) for source in record.include_template_closure
        ],
        "task_metadata_version": record.task_metadata_version,
        "prompt_config": record.prompt_config,
        "few_shot_config": record.few_shot_config,
        "generation_config": record.generation_config,
        "metric_normalization_config": record.metric_normalization_config,
        "seeds": record.seeds,
        "limit": record.limit,
        "ordered_request_identity": record.ordered_request_identity,
        "lm_eval_package_version": record.lm_eval_package_version,
        "lm_eval_source_commit": record.lm_eval_source_commit,
        "dataset_revision": record.dataset_revision,
        "dataset_fingerprint": record.dataset_fingerprint,
        "provider_versions": [
            {"name": provider.name, "version": provider.version}
            for provider in record.provider_versions
        ],
    }
    if include_identity:
        projection["task_identity"] = record.task_identity
    if include_metrics:
        projection["metric_payload"] = record.metric_payload
    return projection


def evaluation_task_identity(record: EvaluationTaskRecord) -> str:
    if not isinstance(record, EvaluationTaskRecord):
        raise TypeError("record must be an EvaluationTaskRecord")
    return structured_identity(
        "sml-evaluation-task-v1",
        _task_projection(record, include_identity=False, include_metrics=False),
    )


def evaluation_result_identity(result: EvaluationResult) -> str:
    if not isinstance(result, EvaluationResult):
        raise TypeError("result must be an EvaluationResult")
    return structured_identity(
        "sml-evaluation-result-v1",
        {
            "kind": result.kind,
            "version": result.version,
            "model": _model_projection(result.model),
            "tasks": [
                _task_projection(task, include_identity=True, include_metrics=True)
                for task in result.tasks
            ],
            "provider_result": result.provider_result,
        },
    )


def _source_payload(source: EvaluationSourceIdentity) -> dict[str, str]:
    return {
        "logical_name": source.logical_name,
        "content_identity": source.content_identity,
    }


def _provider_payload(provider: EvaluationProviderVersion) -> dict[str, str]:
    return {"name": provider.name, "version": provider.version}


def _task_payload(record: EvaluationTaskRecord) -> dict[str, object]:
    return {
        "task_name": record.task_name,
        "task_identity": record.task_identity,
        "task_yaml": _source_payload(record.task_yaml),
        "include_template_closure": [
            _source_payload(source) for source in record.include_template_closure
        ],
        "task_metadata_version": record.task_metadata_version,
        "prompt_config": record.prompt_config,
        "few_shot_config": record.few_shot_config,
        "generation_config": record.generation_config,
        "metric_normalization_config": record.metric_normalization_config,
        "seeds": record.seeds,
        "limit": record.limit,
        "ordered_request_identity": record.ordered_request_identity,
        "lm_eval_package_version": record.lm_eval_package_version,
        "lm_eval_source_commit": record.lm_eval_source_commit,
        "dataset_revision": record.dataset_revision,
        "dataset_fingerprint": record.dataset_fingerprint,
        "provider_versions": [
            _provider_payload(provider) for provider in record.provider_versions
        ],
        "metric_payload": record.metric_payload,
    }


def _result_payload(result: EvaluationResult) -> dict[str, object]:
    return {
        "kind": result.kind,
        "version": result.version,
        "identity": result.identity,
        "model": _model_projection(result.model),
        "tasks": [_task_payload(task) for task in result.tasks],
        "provider_result": result.provider_result,
    }


def evaluation_result_bytes(result: EvaluationResult) -> bytes:
    """Return the canonical persisted representation of a self-identifying result."""
    if not isinstance(result, EvaluationResult):
        raise TypeError("result must be an EvaluationResult")
    for task in result.tasks:
        if evaluation_task_identity(task) != task.task_identity:
            raise ValueError("evaluation task identity mismatch")
    if evaluation_result_identity(result) != result.identity:
        raise ValueError("evaluation result identity mismatch")
    return canonical_json_bytes(_result_payload(result)) + b"\n"


def _json_object_no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise SMLArtifactError(f"duplicate JSON object key: {key}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> object:
    raise SMLArtifactError(f"evaluation result contains non-finite number: {value}")


def _require_field_set(
    payload: object, expected: frozenset[str], context: str
) -> dict[str, object]:
    if not isinstance(payload, dict) or set(payload) != expected:
        raise SMLArtifactError(f"{context} has invalid fields")
    return payload


def _source_from_payload(payload: object) -> EvaluationSourceIdentity:
    raw = _require_field_set(
        payload,
        frozenset({"logical_name", "content_identity"}),
        "evaluation source",
    )
    return EvaluationSourceIdentity(
        logical_name=raw["logical_name"],  # type: ignore[arg-type]
        content_identity=raw["content_identity"],  # type: ignore[arg-type]
    )


def _provider_from_payload(payload: object) -> EvaluationProviderVersion:
    raw = _require_field_set(
        payload, frozenset({"name", "version"}), "evaluation provider"
    )
    return EvaluationProviderVersion(
        name=raw["name"],  # type: ignore[arg-type]
        version=raw["version"],  # type: ignore[arg-type]
    )


def _model_from_payload(payload: object) -> ModelIdentity:
    raw = _require_field_set(
        payload,
        frozenset(
            {
                "artifact_kind",
                "run_identity",
                "step",
                "checkpoint_identity",
                "run_step_identity",
                "tokenizer_identity",
                "verification",
            }
        ),
        "evaluation model",
    )
    verification = raw["verification"]
    if not isinstance(verification, str):
        raise TypeError("model verification must be a string")
    return ModelIdentity(
        artifact_kind=raw["artifact_kind"],  # type: ignore[arg-type]
        run_identity=raw["run_identity"],  # type: ignore[arg-type]
        step=raw["step"],  # type: ignore[arg-type]
        checkpoint_identity=raw["checkpoint_identity"],  # type: ignore[arg-type]
        run_step_identity=raw["run_step_identity"],  # type: ignore[arg-type]
        tokenizer_identity=raw["tokenizer_identity"],  # type: ignore[arg-type]
        verification=VerificationLevel(verification),
    )


def _task_from_payload(payload: object) -> EvaluationTaskRecord:
    raw = _require_field_set(
        payload,
        frozenset(
            {
                "task_name",
                "task_identity",
                "task_yaml",
                "include_template_closure",
                "task_metadata_version",
                "prompt_config",
                "few_shot_config",
                "generation_config",
                "metric_normalization_config",
                "seeds",
                "limit",
                "ordered_request_identity",
                "lm_eval_package_version",
                "lm_eval_source_commit",
                "dataset_revision",
                "dataset_fingerprint",
                "provider_versions",
                "metric_payload",
            }
        ),
        "evaluation task",
    )
    closure = raw["include_template_closure"]
    providers = raw["provider_versions"]
    if not isinstance(closure, list):
        raise TypeError("include_template_closure must be a JSON array")
    if not isinstance(providers, list):
        raise TypeError("provider_versions must be a JSON array")
    task = EvaluationTaskRecord(
        task_name=raw["task_name"],  # type: ignore[arg-type]
        task_identity=raw["task_identity"],  # type: ignore[arg-type]
        task_yaml=_source_from_payload(raw["task_yaml"]),
        include_template_closure=tuple(_source_from_payload(item) for item in closure),
        task_metadata_version=raw["task_metadata_version"],  # type: ignore[arg-type]
        prompt_config=raw["prompt_config"],  # type: ignore[arg-type]
        few_shot_config=raw["few_shot_config"],  # type: ignore[arg-type]
        generation_config=raw["generation_config"],  # type: ignore[arg-type]
        metric_normalization_config=raw["metric_normalization_config"],  # type: ignore[arg-type]
        seeds=raw["seeds"],  # type: ignore[arg-type]
        limit=raw["limit"],  # type: ignore[arg-type]
        ordered_request_identity=raw["ordered_request_identity"],  # type: ignore[arg-type]
        lm_eval_package_version=raw["lm_eval_package_version"],  # type: ignore[arg-type]
        lm_eval_source_commit=raw["lm_eval_source_commit"],  # type: ignore[arg-type]
        dataset_revision=raw["dataset_revision"],  # type: ignore[arg-type]
        dataset_fingerprint=raw["dataset_fingerprint"],  # type: ignore[arg-type]
        provider_versions=tuple(_provider_from_payload(item) for item in providers),
        metric_payload=raw["metric_payload"],  # type: ignore[arg-type]
    )
    if evaluation_task_identity(task) != task.task_identity:
        raise SMLArtifactError("evaluation task identity mismatch")
    return task


def _result_from_payload(payload: object) -> EvaluationResult:
    raw = _require_field_set(
        payload,
        frozenset({"kind", "version", "identity", "model", "tasks", "provider_result"}),
        "evaluation result",
    )
    tasks = raw["tasks"]
    if not isinstance(tasks, list):
        raise TypeError("evaluation tasks must be a JSON array")
    return EvaluationResult(
        kind=raw["kind"],  # type: ignore[arg-type]
        version=raw["version"],  # type: ignore[arg-type]
        identity=raw["identity"],  # type: ignore[arg-type]
        model=_model_from_payload(raw["model"]),
        tasks=tuple(_task_from_payload(task) for task in tasks),
        provider_result=raw["provider_result"],  # type: ignore[arg-type]
    )


def read_evaluation_result(path: Path) -> EvaluationResult:
    """Strictly read one canonical evaluation result file."""
    if not isinstance(path, Path):
        raise TypeError("path must be a Path")
    try:
        raw_bytes = path.read_bytes()
        raw = json.loads(
            raw_bytes.decode("utf-8"),
            object_pairs_hook=_json_object_no_duplicates,
            parse_constant=_reject_json_constant,
        )
        result = _result_from_payload(raw)
        if evaluation_result_bytes(result) != raw_bytes:
            raise SMLArtifactError("evaluation result is not canonical")
        if evaluation_result_identity(result) != result.identity:
            raise SMLArtifactError("evaluation result identity mismatch")
        return result
    except SMLArtifactError:
        raise
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ) as error:
        raise SMLArtifactError("invalid evaluation result") from error


def _fsync_parent(path: Path) -> None:
    directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def publish_evaluation_result(path: Path, result: EvaluationResult) -> None:
    """Durably publish *result* once, accepting only exact idempotent reuse."""
    if not isinstance(path, Path):
        raise TypeError("path must be a Path")
    payload = evaluation_result_bytes(result)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            try:
                existing = read_evaluation_result(path)
            except SMLArtifactError:
                raise SMLRuntimeError(
                    "evaluation output collision: " + str(path)
                ) from None
            if existing != result:
                raise SMLRuntimeError("evaluation output collision: " + str(path))
        _fsync_parent(path)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = (
    "EvaluationProviderVersion",
    "EvaluationResult",
    "EvaluationSourceIdentity",
    "EvaluationTaskRecord",
    "evaluation_result_bytes",
    "evaluation_result_identity",
    "evaluation_task_identity",
    "normalize_json_value",
    "publish_evaluation_result",
    "read_evaluation_result",
)
