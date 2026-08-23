from __future__ import annotations

# Serialized quality schemas report malformed external content uniformly as ValueError.
# ruff: noqa: TRY004
import argparse
import ast
import dataclasses
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import mlx.core as mx
import numpy as np
from mlx import nn
from mlx.utils import tree_flatten, tree_map
from sml.model.config import ModelConfig
from sml.model.language_model import SMLLanguageModel, causal_lm_loss
from sml.model.layers import keyed_dropout
from sml.training.common import (
    LoaderConfig,
    OptimizerConfig,
    PrecisionConfig,
    PretrainingConfig,
    TrainerState,
    WeightDecayPolicy,
    accumulate_fp32,
    adamw_mixed_precision_update_tree,
    build_weight_decay_tree,
    initialize_adam_state,
    initialize_base_parameter_state,
    learning_rate_at,
    normalize_and_clip,
)
from sml.training.pretrain import build_pretraining_kernels

from v2.benchmarks.workload import (
    canonical_json_bytes,
    file_identity,
    semantic_row_content_identity,
    structured_identity,
)

CANONICAL_STEPS = 1_000
CHECKPOINT_STEPS = (0, 10, 100, CANONICAL_STEPS)
QUALITY_WALL_TIME_BUDGET_SECONDS = 12 * 60 * 60
HARNESS_COMPONENTS = (
    Path("v2/benchmarks/quality.py"),
    Path("v2/tests/unit/test_pretraining_quality.py"),
)
PRODUCTION_SOURCE_TREE = Path("v2/src/sml")
PRODUCTION_MODULE_ROOT = Path("v2/src")
PRODUCTION_DEPENDENCY_FIXED_COMPONENTS = (
    Path("v2/src/sml.py"),
    Path("v2/benchmarks/schema.py"),
    Path("v2/benchmarks/workload.py"),
)
PRODUCTION_IMPORT_ENTRYPOINTS = (HARNESS_COMPONENTS[0],)
TRAINING_FIXTURE = Path("v2/benchmarks/fixtures/pretraining-quality-train-v1.npy")
VALIDATION_FIXTURE = Path(
    "v2/benchmarks/fixtures/pretraining-quality-validation-v1.npy"
)
TRAINING_SHAPE = (32, 1_025)
VALIDATION_SHAPE = (8, 1_025)
CANONICAL_MANIFEST_PATH = Path("v2/benchmarks/manifests/pretraining-quality-v1.json")
CANONICAL_RAW_PATH = Path("v2/benchmarks/results/pretraining-quality-v1.jsonl")
CANONICAL_REPORT_PATH = Path("v2/benchmarks/results/pretraining-quality-v1.json")
RECOVERY_PATH = Path("v2/benchmarks/results/.pretraining-quality-v1.recording")
MEASUREMENT_BOUNDARIES = {
    "clock": "time.monotonic",
    "start": "record-handler-entry-before-path-worktree-and-workload-setup",
    "manifest_freeze": (
        "after-setup-both-runtime-setup-compilation-execution-validation-"
        "and-non-manifest-serialization"
    ),
    "excluded_after_manifest_freeze": [
        "manifest-self-serialization-and-all-durable-staging",
        "create-only-final-artifact-links-and-parent-fsyncs",
        "completion-marker-link-and-recovery-cleanup",
    ],
}
_PHASE_NAMES = (
    "setup",
    "candidate",
    "oracle",
    "validation_serialization",
)
_PUBLICATION_STRATEGY = "owned-create-only-hard-link-set-v1"
_IDENTITY_RE = re.compile(r"sha256:[0-9a-f]{64}")
_PLACEHOLDER_IDENTITY = "sha256:" + "0" * 64
_MASTER_TREE_IDENTITY_DOMAIN = "sml-pretraining-quality-master-parameters-v2"
_WORKING_TREE_IDENTITY_DOMAIN = "sml-pretraining-quality-working-parameters-v2"

RuntimeName = Literal["candidate", "oracle"]
ComputeDtype = Literal["bfloat16", "float32"]


@dataclass(frozen=True, slots=True)
class _EvidenceDestinations:
    manifest: Path
    raw_output: Path
    report: Path

    def ordered(self) -> tuple[tuple[str, Path], ...]:
        return (
            ("raw", self.raw_output),
            ("manifest", self.manifest),
            ("report", self.report),
        )


@dataclass(frozen=True, slots=True)
class _EvidencePublication:
    recovery_directory: Path
    destinations: _EvidenceDestinations
    owner: dict[str, object]
    staged: bool = False


def _payload_identity(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _require_identity(value: object, name: str) -> str:
    if not isinstance(value, str) or _IDENTITY_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a SHA-256 identity")
    return value


def _require_plain_int(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}")
    return value


def _require_finite(value: object, name: str, *, minimum: float = 0.0) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < minimum:
        raise ValueError(f"{name} must be finite and at least {minimum}")
    return normalized


def _require_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a bool")
    return value


def harness_content_identity(root: Path) -> str:
    if not isinstance(root, Path):
        raise TypeError("root must be a Path")
    digest = hashlib.sha256()
    for relative_path in HARNESS_COMPONENTS:
        path = root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"missing quality harness component: {path}")
        digest.update(path.read_bytes())
    return f"sha256:{digest.hexdigest()}"


def _module_parts(component: Path) -> tuple[str, ...] | None:
    try:
        relative = component.relative_to(PRODUCTION_MODULE_ROOT)
    except ValueError:
        return None
    if relative.name == "__init__.py":
        return relative.parent.parts
    if relative.suffix != ".py":
        return None
    return relative.with_suffix("").parts


def _local_import_names(
    component: Path,
    payload: bytes,
) -> tuple[tuple[str, bool], ...]:
    try:
        tree = ast.parse(payload.decode("utf-8"), filename=component.as_posix())
    except (SyntaxError, UnicodeError) as error:
        raise ValueError(
            f"quality production dependency is not valid Python: {component}"
        ) from error
    module_parts = _module_parts(component)
    if module_parts is None:
        package_parts: tuple[str, ...] = ()
    elif component.name == "__init__.py":
        package_parts = module_parts
    else:
        package_parts = module_parts[:-1]

    imports: set[tuple[str, bool]] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(
                (alias.name, True)
                for alias in node.names
                if alias.name == "sml" or alias.name.startswith("sml.")
            )
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level:
            retained = len(package_parts) - (node.level - 1)
            if retained < 0:
                raise ValueError(
                    f"quality production dependency has an invalid import: {component}"
                )
            base_parts = (*package_parts[:retained], *(node.module or "").split("."))
        else:
            base_parts = tuple((node.module or "").split("."))
        base_parts = tuple(part for part in base_parts if part)
        if not base_parts or base_parts[0] != "sml":
            continue
        base = ".".join(base_parts)
        imports.add((base, True))
        imports.update(
            (f"{base}.{alias.name}", False) for alias in node.names if alias.name != "*"
        )
    return tuple(sorted(imports))


def _local_module_components(
    module_name: str,
    available: set[Path],
) -> tuple[Path, ...]:
    parts = tuple(module_name.split("."))
    if not parts or parts[0] != "sml":
        return ()
    package_components = [
        PRODUCTION_MODULE_ROOT.joinpath(*parts[:depth], "__init__.py")
        for depth in range(1, len(parts) + 1)
    ]
    module_component = PRODUCTION_MODULE_ROOT.joinpath(*parts).with_suffix(".py")
    package_component = package_components[-1]
    if package_component in available:
        target = package_component
    elif module_component in available:
        target = module_component
    else:
        return ()
    return tuple(
        component
        for component in (*package_components[:-1], target)
        if component in available
    )


def _production_dependency_closure(
    available: set[Path],
    read_component: Callable[[Path], bytes],
) -> tuple[Path, ...]:
    required = {
        *PRODUCTION_DEPENDENCY_FIXED_COMPONENTS,
        *PRODUCTION_IMPORT_ENTRYPOINTS,
    }
    missing_required = required - available
    if missing_required:
        raise FileNotFoundError(
            "missing quality production entry components: "
            f"{sorted(path.as_posix() for path in missing_required)!r}"
        )
    components = set(PRODUCTION_DEPENDENCY_FIXED_COMPONENTS)
    pending = sorted(required, key=lambda path: path.as_posix())
    scanned: set[Path] = set()
    while pending:
        component = pending.pop()
        if component in scanned:
            continue
        scanned.add(component)
        for module_name, import_required in _local_import_names(
            component, read_component(component)
        ):
            dependencies = _local_module_components(module_name, available)
            if import_required and not dependencies:
                raise ValueError(
                    "quality production import cannot be resolved: "
                    f"{component}:{module_name}"
                )
            for dependency in dependencies:
                if dependency not in components:
                    components.add(dependency)
                    pending.append(dependency)
    return tuple(sorted(components, key=lambda path: path.as_posix()))


def _available_production_components(root: Path) -> set[Path]:
    source_tree = root / PRODUCTION_SOURCE_TREE
    if source_tree.is_symlink() or not source_tree.is_dir():
        raise FileNotFoundError(
            f"missing quality production source tree: {source_tree}"
        )
    return {
        *PRODUCTION_DEPENDENCY_FIXED_COMPONENTS,
        *PRODUCTION_IMPORT_ENTRYPOINTS,
        *(path.relative_to(root) for path in source_tree.rglob("*.py")),
    }


def _read_current_production_component(root: Path, relative_path: Path) -> bytes:
    path = root / relative_path
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"missing quality production dependency: {path}")
    return path.read_bytes()


def production_dependency_components(root: Path) -> tuple[Path, ...]:
    if not isinstance(root, Path):
        raise TypeError("root must be a Path")
    return _production_dependency_closure(
        _available_production_components(root),
        lambda component: _read_current_production_component(root, component),
    )


def _validate_production_dependency_closure(
    root: Path,
    components: Sequence[Path],
) -> None:
    expected = production_dependency_components(root)
    actual = tuple(components)
    if actual != expected:
        missing = sorted(set(expected) - set(actual), key=lambda path: path.as_posix())
        extra = sorted(set(actual) - set(expected), key=lambda path: path.as_posix())
        raise ValueError(
            "quality production dependency closure is incomplete or overbroad: "
            f"omitted={[path.as_posix() for path in missing]!r}, "
            f"unrelated={[path.as_posix() for path in extra]!r}"
        )


def _production_dependency_identity(
    components: Sequence[Path],
    read_component: Callable[[Path], bytes],
) -> str:
    digest = hashlib.sha256()
    digest.update(b"sml-pretraining-quality-production-import-closure-v1\0")
    for relative_path in components:
        encoded_path = relative_path.as_posix().encode("utf-8")
        payload = read_component(relative_path)
        if not isinstance(payload, bytes):
            raise TypeError("production dependency reader must return bytes")
        digest.update(len(encoded_path).to_bytes(8, "little"))
        digest.update(encoded_path)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return f"sha256:{digest.hexdigest()}"


def _current_production_dependency_identity(
    root: Path, components: Sequence[Path]
) -> str:
    def read_component(relative_path: Path) -> bytes:
        return _read_current_production_component(root, relative_path)

    return _production_dependency_identity(components, read_component)


def production_dependency_content_identity(root: Path) -> str:
    if not isinstance(root, Path):
        raise TypeError("root must be a Path")
    components = production_dependency_components(root)
    return _current_production_dependency_identity(root, components)


@dataclass(frozen=True, slots=True)
class QualityFixture:
    logical_path: str
    shape: tuple[int, int]
    dtype: Literal["int32"]
    byte_size: int
    file_identity: str
    semantic_identity: str
    source_identity: str

    def to_dict(self) -> dict[str, object]:
        return {
            "logical_path": self.logical_path,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "byte_size": self.byte_size,
            "file_identity": self.file_identity,
            "semantic_identity": self.semantic_identity,
            "source_identity": self.source_identity,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> QualityFixture:
        if set(raw) != {
            "logical_path",
            "shape",
            "dtype",
            "byte_size",
            "file_identity",
            "semantic_identity",
            "source_identity",
        }:
            raise ValueError("quality fixture has an invalid field set")
        logical_path = raw["logical_path"]
        shape = raw["shape"]
        if not isinstance(logical_path, str) or not logical_path:
            raise ValueError("quality fixture logical path must be non-empty")
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or any(type(value) is not int or value <= 0 for value in shape)
        ):
            raise ValueError("quality fixture shape must have two positive integers")
        if raw["dtype"] != "int32":
            raise ValueError("quality fixture dtype must be int32")
        return cls(
            logical_path=logical_path,
            shape=(shape[0], shape[1]),
            dtype="int32",
            byte_size=_require_plain_int(
                raw["byte_size"], "fixture byte size", minimum=1
            ),
            file_identity=_require_identity(
                raw["file_identity"], "fixture file identity"
            ),
            semantic_identity=_require_identity(
                raw["semantic_identity"], "fixture semantic identity"
            ),
            source_identity=_require_identity(
                raw["source_identity"], "fixture source identity"
            ),
        )


@dataclass(frozen=True, slots=True)
class ParameterLeafSpec:
    path: str
    shape: tuple[int, ...]
    value_count: int
    initial_bf16_identity: str
    initial_fp32_identity: str

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "shape": list(self.shape),
            "value_count": self.value_count,
            "initial_bf16_identity": self.initial_bf16_identity,
            "initial_fp32_identity": self.initial_fp32_identity,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> ParameterLeafSpec:
        if set(raw) != {
            "path",
            "shape",
            "value_count",
            "initial_bf16_identity",
            "initial_fp32_identity",
        }:
            raise ValueError("parameter leaf spec has an invalid field set")
        path = raw["path"]
        shape = raw["shape"]
        if not isinstance(path, str) or not path:
            raise ValueError("parameter leaf path must be non-empty")
        if (
            not isinstance(shape, list)
            or not shape
            or any(type(dimension) is not int or dimension <= 0 for dimension in shape)
        ):
            raise ValueError("parameter leaf shape must contain positive integers")
        normalized_shape = tuple(shape)
        value_count = _require_plain_int(
            raw["value_count"], "parameter leaf value count", minimum=1
        )
        if value_count != math.prod(normalized_shape):
            raise ValueError("parameter leaf value count does not match its shape")
        return cls(
            path=path,
            shape=normalized_shape,
            value_count=value_count,
            initial_bf16_identity=_require_identity(
                raw["initial_bf16_identity"], "initial BF16 leaf identity"
            ),
            initial_fp32_identity=_require_identity(
                raw["initial_fp32_identity"], "initial FP32 leaf identity"
            ),
        )


@dataclass(frozen=True, slots=True)
class PretrainingQualityWorkload:
    kind: Literal["pretraining-quality-workload"]
    version: Literal[2]
    identity: str
    training_fixture: QualityFixture
    validation_fixture: QualityFixture
    initial_bf16_parameter_identity: str
    parameter_leaf_names: tuple[str, ...]
    parameter_leaves: tuple[ParameterLeafSpec, ...]
    model: dict[str, object]
    optimizer: dict[str, object]
    loader: dict[str, object]
    precision: dict[str, object]
    ordered_batches: tuple[tuple[int, ...], ...]
    batch_prefix_identities: dict[int, str]
    checkpoint_steps: tuple[int, ...]
    evaluation_row_indices: tuple[int, ...]
    evaluation_request_identity: str
    model_seed: int
    training_seed: int
    loader_seed: int
    harness_components: tuple[str, ...]
    harness_identity: str
    production_dependency_components: tuple[str, ...]
    production_dependency_identity: str

    def _body(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "version": self.version,
            "training_fixture": self.training_fixture.to_dict(),
            "validation_fixture": self.validation_fixture.to_dict(),
            "initial_bf16_parameter_identity": self.initial_bf16_parameter_identity,
            "parameter_leaf_names": list(self.parameter_leaf_names),
            "parameter_leaves": [leaf.to_dict() for leaf in self.parameter_leaves],
            "model": self.model,
            "optimizer": self.optimizer,
            "loader": self.loader,
            "precision": self.precision,
            "ordered_batches": [list(batch) for batch in self.ordered_batches],
            "batch_prefix_identities": {
                str(step): identity
                for step, identity in sorted(self.batch_prefix_identities.items())
            },
            "checkpoint_steps": list(self.checkpoint_steps),
            "evaluation_row_indices": list(self.evaluation_row_indices),
            "evaluation_request_identity": self.evaluation_request_identity,
            "model_seed": self.model_seed,
            "training_seed": self.training_seed,
            "loader_seed": self.loader_seed,
            "harness_components": list(self.harness_components),
            "harness_identity": self.harness_identity,
            "production_dependency_components": list(
                self.production_dependency_components
            ),
            "production_dependency_identity": self.production_dependency_identity,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._body(), "identity": self.identity}

    def recompute_identity(self) -> str:
        return structured_identity("sml-pretraining-quality-workload-v2", self._body())

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> PretrainingQualityWorkload:
        expected = {
            "kind",
            "version",
            "identity",
            "training_fixture",
            "validation_fixture",
            "initial_bf16_parameter_identity",
            "parameter_leaf_names",
            "parameter_leaves",
            "model",
            "optimizer",
            "loader",
            "precision",
            "ordered_batches",
            "batch_prefix_identities",
            "checkpoint_steps",
            "evaluation_row_indices",
            "evaluation_request_identity",
            "model_seed",
            "training_seed",
            "loader_seed",
            "harness_components",
            "harness_identity",
            "production_dependency_components",
            "production_dependency_identity",
        }
        if set(raw) != expected:
            raise ValueError("pretraining quality workload has an invalid field set")
        if raw["kind"] != "pretraining-quality-workload" or raw["version"] != 2:
            raise ValueError("unsupported pretraining quality workload")
        training_raw = raw["training_fixture"]
        validation_raw = raw["validation_fixture"]
        if not isinstance(training_raw, dict) or not isinstance(validation_raw, dict):
            raise ValueError("quality workload fixtures must be objects")
        for name in (
            "model",
            "optimizer",
            "loader",
            "precision",
            "batch_prefix_identities",
        ):
            if not isinstance(raw[name], dict):
                raise ValueError(f"quality workload {name} must be an object")
        for name in (
            "parameter_leaf_names",
            "parameter_leaves",
            "ordered_batches",
            "checkpoint_steps",
            "evaluation_row_indices",
            "harness_components",
            "production_dependency_components",
        ):
            if not isinstance(raw[name], list):
                raise ValueError(f"quality workload {name} must be a list")
        parameter_names = raw["parameter_leaf_names"]
        if (
            not parameter_names
            or any(not isinstance(name, str) or not name for name in parameter_names)
            or parameter_names != sorted(set(parameter_names))
        ):
            raise ValueError("parameter leaf names must be unique and sorted")
        leaf_values = raw["parameter_leaves"]
        if any(not isinstance(value, dict) for value in leaf_values):
            raise ValueError("parameter leaves must be objects")
        parameter_leaves = tuple(
            ParameterLeafSpec.from_dict(value) for value in leaf_values
        )
        if tuple(leaf.path for leaf in parameter_leaves) != tuple(parameter_names):
            raise ValueError("parameter leaf specs must match the ordered leaf names")
        batches: list[tuple[int, ...]] = []
        for batch in raw["ordered_batches"]:
            if (
                not isinstance(batch, list)
                or not batch
                or any(type(index) is not int or index < 0 for index in batch)
            ):
                raise ValueError("ordered batches must contain row-index lists")
            batches.append(tuple(batch))
        prefixes: dict[int, str] = {}
        for step_text, identity in raw["batch_prefix_identities"].items():
            if not isinstance(step_text, str) or not step_text.isdecimal():
                raise ValueError("batch prefix steps must be decimal strings")
            prefixes[int(step_text)] = _require_identity(
                identity, "batch prefix identity"
            )
        checkpoint_steps = tuple(raw["checkpoint_steps"])
        evaluation_rows = tuple(raw["evaluation_row_indices"])
        components = tuple(raw["harness_components"])
        production_components = tuple(raw["production_dependency_components"])
        if any(type(step) is not int for step in checkpoint_steps):
            raise ValueError("checkpoint steps must be integers")
        if any(type(index) is not int or index < 0 for index in evaluation_rows):
            raise ValueError("evaluation row indices must be nonnegative integers")
        if any(not isinstance(value, str) or not value for value in components):
            raise ValueError("harness components must be non-empty strings")
        if any(
            not isinstance(value, str) or not value for value in production_components
        ):
            raise ValueError(
                "production dependency components must be non-empty strings"
            )
        if components != tuple(path.as_posix() for path in HARNESS_COMPONENTS):
            raise ValueError("quality harness component order changed")
        production_paths = tuple(Path(value) for value in production_components)
        if (
            production_components != tuple(sorted(set(production_components)))
            or any(
                path.is_absolute() or path.as_posix() != value or ".." in path.parts
                for path, value in zip(
                    production_paths, production_components, strict=True
                )
            )
            or not set(PRODUCTION_DEPENDENCY_FIXED_COMPONENTS).issubset(
                production_paths
            )
        ):
            raise ValueError("quality production dependency order changed")
        for path in production_paths:
            if path in PRODUCTION_DEPENDENCY_FIXED_COMPONENTS:
                continue
            if path.suffix != ".py" or not path.is_relative_to(PRODUCTION_SOURCE_TREE):
                raise ValueError("quality production import closure is incomplete")
        workload = cls(
            kind="pretraining-quality-workload",
            version=2,
            identity=_require_identity(raw["identity"], "workload identity"),
            training_fixture=QualityFixture.from_dict(training_raw),
            validation_fixture=QualityFixture.from_dict(validation_raw),
            initial_bf16_parameter_identity=_require_identity(
                raw["initial_bf16_parameter_identity"], "initial parameter identity"
            ),
            parameter_leaf_names=tuple(parameter_names),
            parameter_leaves=parameter_leaves,
            model=dict(raw["model"]),
            optimizer=dict(raw["optimizer"]),
            loader=dict(raw["loader"]),
            precision=dict(raw["precision"]),
            ordered_batches=tuple(batches),
            batch_prefix_identities=prefixes,
            checkpoint_steps=checkpoint_steps,
            evaluation_row_indices=evaluation_rows,
            evaluation_request_identity=_require_identity(
                raw["evaluation_request_identity"], "evaluation request identity"
            ),
            model_seed=_require_plain_int(raw["model_seed"], "model seed"),
            training_seed=_require_plain_int(raw["training_seed"], "training seed"),
            loader_seed=_require_plain_int(raw["loader_seed"], "loader seed"),
            harness_components=components,
            harness_identity=_require_identity(
                raw["harness_identity"], "harness identity"
            ),
            production_dependency_components=production_components,
            production_dependency_identity=_require_identity(
                raw["production_dependency_identity"],
                "production dependency identity",
            ),
        )
        if workload.checkpoint_steps != CHECKPOINT_STEPS:
            raise ValueError("quality workload checkpoint steps are not canonical")
        if set(workload.batch_prefix_identities) != set(CHECKPOINT_STEPS):
            raise ValueError("quality workload batch prefixes are incomplete")
        if (
            workload.training_fixture.source_identity
            == workload.validation_fixture.source_identity
        ):
            raise ValueError("quality fixtures must be source-disjoint")
        expected_initial_identity = _tree_identity_from_leaf_identities(
            "sml-pretraining-quality-initial-bf16-parameters-v1",
            [
                (leaf.path, leaf.shape, leaf.initial_bf16_identity)
                for leaf in workload.parameter_leaves
            ],
        )
        if workload.initial_bf16_parameter_identity != expected_initial_identity:
            raise ValueError("initial BF16 tree identity does not match leaf specs")
        if workload.identity != workload.recompute_identity():
            raise ValueError("pretraining quality workload identity mismatch")
        return workload


@dataclass(frozen=True, slots=True)
class ParameterUpdateStatistics:
    path: str
    shape: tuple[int, ...]
    value_count: int
    before_master_identity: str
    after_master_identity: str
    before_working_identity: str
    after_working_identity: str
    before_bf16_working_identity: str
    after_bf16_working_identity: str
    nonzero_update_count: int
    sub_bf16_ulp_update_count: int
    survived_sub_bf16_ulp_count: int
    changed_bf16_working_count: int
    first_sub_bf16_survival_step: int | None
    minimum_update_to_bf16_ulp: float | None
    mean_update_to_bf16_ulp: float | None
    maximum_update_to_bf16_ulp: float | None

    def to_dict(self) -> dict[str, object]:
        return {**dataclasses.asdict(self), "shape": list(self.shape)}

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> ParameterUpdateStatistics:
        if set(raw) != {field.name for field in dataclasses.fields(cls)}:
            raise ValueError("update statistics have an invalid field set")
        path = raw["path"]
        shape = raw["shape"]
        if not isinstance(path, str) or not path:
            raise ValueError("update statistics path must be non-empty")
        if (
            not isinstance(shape, list)
            or not shape
            or any(type(dimension) is not int or dimension <= 0 for dimension in shape)
        ):
            raise ValueError("update statistics shape must contain positive integers")
        normalized_shape = tuple(shape)
        counts = {
            name: _require_plain_int(raw[name], name)
            for name in (
                "value_count",
                "nonzero_update_count",
                "sub_bf16_ulp_update_count",
                "survived_sub_bf16_ulp_count",
                "changed_bf16_working_count",
            )
        }
        if counts["value_count"] != math.prod(normalized_shape):
            raise ValueError("update statistics value count does not match its shape")
        if not (
            counts["survived_sub_bf16_ulp_count"]
            <= counts["sub_bf16_ulp_update_count"]
            <= counts["nonzero_update_count"]
            <= counts["value_count"]
        ):
            raise ValueError("update statistics counts are inconsistent")
        if counts["changed_bf16_working_count"] > counts["value_count"]:
            raise ValueError("changed BF16 working count exceeds the leaf size")
        if (
            counts["survived_sub_bf16_ulp_count"]
            > counts["value_count"] - counts["changed_bf16_working_count"]
        ):
            raise ValueError("surviving sub-BF16 updates require unchanged BF16 values")
        identities = {
            name: _require_identity(raw[name], name)
            for name in (
                "before_master_identity",
                "after_master_identity",
                "before_working_identity",
                "after_working_identity",
                "before_bf16_working_identity",
                "after_bf16_working_identity",
            )
        }
        first_survival_raw = raw["first_sub_bf16_survival_step"]
        first_survival = (
            None
            if first_survival_raw is None
            else _require_plain_int(
                first_survival_raw,
                "first sub-BF16 survival step",
                minimum=1,
            )
        )
        ratios = []
        for name in (
            "minimum_update_to_bf16_ulp",
            "mean_update_to_bf16_ulp",
            "maximum_update_to_bf16_ulp",
        ):
            value = raw[name]
            ratios.append(None if value is None else _require_finite(value, name))
        if counts["nonzero_update_count"] == 0 and any(
            value is not None for value in ratios
        ):
            raise ValueError("zero-update statistics must not report ratios")
        if counts["nonzero_update_count"] > 0 and (
            any(value is None for value in ratios)
            or not ratios[0] <= ratios[1] <= ratios[2]
        ):
            raise ValueError("nonzero-update statistics require ordered ratios")
        return cls(
            path=path,
            shape=normalized_shape,
            value_count=counts["value_count"],
            before_master_identity=identities["before_master_identity"],
            after_master_identity=identities["after_master_identity"],
            before_working_identity=identities["before_working_identity"],
            after_working_identity=identities["after_working_identity"],
            before_bf16_working_identity=identities["before_bf16_working_identity"],
            after_bf16_working_identity=identities["after_bf16_working_identity"],
            nonzero_update_count=counts["nonzero_update_count"],
            sub_bf16_ulp_update_count=counts["sub_bf16_ulp_update_count"],
            survived_sub_bf16_ulp_count=counts["survived_sub_bf16_ulp_count"],
            changed_bf16_working_count=counts["changed_bf16_working_count"],
            first_sub_bf16_survival_step=first_survival,
            minimum_update_to_bf16_ulp=ratios[0],
            mean_update_to_bf16_ulp=ratios[1],
            maximum_update_to_bf16_ulp=ratios[2],
        )


def _telemetry_tree_identity(
    domain: str,
    statistics: Sequence[ParameterUpdateStatistics],
    identity_field: str,
) -> str:
    allowed_fields = {
        "before_master_identity",
        "after_master_identity",
        "before_working_identity",
        "after_working_identity",
        "before_bf16_working_identity",
        "after_bf16_working_identity",
    }
    if identity_field not in allowed_fields:
        raise ValueError("telemetry identity field is invalid")
    return _tree_identity_from_leaf_identities(
        domain,
        [(item.path, item.shape, getattr(item, identity_field)) for item in statistics],
    )


@dataclass(frozen=True, slots=True)
class PretrainingQualityCheckpoint:
    kind: Literal["pretraining-quality-checkpoint"]
    version: Literal[1]
    identity: str
    runtime: RuntimeName
    compute_dtype: ComputeDtype
    step: int
    workload_identity: str
    real_work_identity: str
    initial_bf16_parameter_identity: str
    ordered_batch_prefix_identity: str
    evaluation_request_identity: str
    master_parameter_identity: str
    working_parameter_identity: str
    trainer_key_identity: str
    train_loss: float
    validation_nll: float
    finite_state_value_count: int
    nonfinite_state_value_count: int
    finite: bool
    update_statistics: tuple[ParameterUpdateStatistics, ...]
    changed_bf16_working_fraction: float
    rms_norm_master_moved: bool
    sub_bf16_update_survived: bool

    def _body(self) -> dict[str, object]:
        return {
            field.name: (
                [item.to_dict() for item in self.update_statistics]
                if field.name == "update_statistics"
                else getattr(self, field.name)
            )
            for field in dataclasses.fields(self)
            if field.name != "identity"
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._body(), "identity": self.identity}

    def recompute_identity(self) -> str:
        return structured_identity(
            "sml-pretraining-quality-checkpoint-v1", self._body()
        )

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> PretrainingQualityCheckpoint:
        if set(raw) != {field.name for field in dataclasses.fields(cls)}:
            raise ValueError("pretraining quality checkpoint has an invalid field set")
        if raw["kind"] != "pretraining-quality-checkpoint" or raw["version"] != 1:
            raise ValueError("unsupported pretraining quality checkpoint")
        runtime = raw["runtime"]
        if runtime not in ("candidate", "oracle"):
            raise ValueError("quality checkpoint runtime is invalid")
        expected_dtype = "bfloat16" if runtime == "candidate" else "float32"
        if raw["compute_dtype"] != expected_dtype:
            raise ValueError("quality checkpoint compute dtype is invalid")
        statistics_raw = raw["update_statistics"]
        if not isinstance(statistics_raw, list) or any(
            not isinstance(value, dict) for value in statistics_raw
        ):
            raise ValueError("quality checkpoint update statistics must be objects")
        statistics = tuple(
            ParameterUpdateStatistics.from_dict(value) for value in statistics_raw
        )
        paths = [item.path for item in statistics]
        if not paths or paths != sorted(set(paths)):
            raise ValueError(
                "quality checkpoint update paths must be unique and sorted"
            )
        checkpoint = cls(
            kind="pretraining-quality-checkpoint",
            version=1,
            identity=_require_identity(raw["identity"], "checkpoint identity"),
            runtime=runtime,
            compute_dtype=expected_dtype,
            step=_require_plain_int(raw["step"], "checkpoint step"),
            workload_identity=_require_identity(
                raw["workload_identity"], "workload identity"
            ),
            real_work_identity=_require_identity(
                raw["real_work_identity"], "real-work identity"
            ),
            initial_bf16_parameter_identity=_require_identity(
                raw["initial_bf16_parameter_identity"], "initial parameter identity"
            ),
            ordered_batch_prefix_identity=_require_identity(
                raw["ordered_batch_prefix_identity"], "batch-prefix identity"
            ),
            evaluation_request_identity=_require_identity(
                raw["evaluation_request_identity"], "evaluation identity"
            ),
            master_parameter_identity=_require_identity(
                raw["master_parameter_identity"], "master identity"
            ),
            working_parameter_identity=_require_identity(
                raw["working_parameter_identity"], "working identity"
            ),
            trainer_key_identity=_require_identity(
                raw["trainer_key_identity"], "trainer-key identity"
            ),
            train_loss=_require_finite(raw["train_loss"], "train loss"),
            validation_nll=_require_finite(raw["validation_nll"], "validation NLL"),
            finite_state_value_count=_require_plain_int(
                raw["finite_state_value_count"],
                "finite-state value count",
                minimum=1,
            ),
            nonfinite_state_value_count=_require_plain_int(
                raw["nonfinite_state_value_count"],
                "nonfinite-state value count",
            ),
            finite=_require_bool(raw["finite"], "finite"),
            update_statistics=statistics,
            changed_bf16_working_fraction=_require_finite(
                raw["changed_bf16_working_fraction"], "changed working fraction"
            ),
            rms_norm_master_moved=_require_bool(
                raw["rms_norm_master_moved"], "RMSNorm master movement"
            ),
            sub_bf16_update_survived=_require_bool(
                raw["sub_bf16_update_survived"], "sub-BF16 update survival"
            ),
        )
        if checkpoint.changed_bf16_working_fraction > 1.0:
            raise ValueError("changed working fraction must be at most one")
        if checkpoint.nonfinite_state_value_count > checkpoint.finite_state_value_count:
            raise ValueError("nonfinite-state count exceeds finite-state value count")
        if checkpoint.identity != checkpoint.recompute_identity():
            raise ValueError("pretraining quality checkpoint identity mismatch")
        return checkpoint


@dataclass(frozen=True, slots=True)
class PretrainingQualityReport:
    candidate_validation_nll: float
    oracle_validation_nll: float
    candidate_finite: bool
    oracle_finite: bool
    rms_norm_master_moved: bool
    sub_bf16_update_survived: bool
    matching_work_identity: bool


def decide_pretraining_quality(
    report: PretrainingQualityReport,
) -> Literal["pass", "fail"]:
    if not isinstance(report, PretrainingQualityReport):
        raise TypeError("report must be a PretrainingQualityReport")
    if not (
        report.candidate_finite
        and report.oracle_finite
        and report.rms_norm_master_moved
        and report.sub_bf16_update_survived
        and report.matching_work_identity
        and math.isfinite(report.candidate_validation_nll)
        and math.isfinite(report.oracle_validation_nll)
        and report.oracle_validation_nll >= 0.0
        and report.candidate_validation_nll <= 1.01 * report.oracle_validation_nll
    ):
        return "fail"
    return "pass"


def _fixture_source_identity(split: str, seed: int) -> str:
    return structured_identity(
        "sml-pretraining-quality-source-v1",
        {
            "split": split,
            "generator": "deterministic-token-stream-v1",
            "seed": seed,
        },
    )


def _load_fixture(
    root: Path,
    relative_path: Path,
    expected_shape: tuple[int, int],
    *,
    split: str,
    source_seed: int,
) -> tuple[np.ndarray, QualityFixture]:
    path = root / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"missing quality fixture: {path}")
    rows = np.load(path, allow_pickle=False, mmap_mode="r")
    if (
        rows.dtype != np.dtype("<i4")
        or tuple(rows.shape) != expected_shape
        or not rows.flags.c_contiguous
    ):
        raise ValueError(
            f"quality {split} fixture must be C-order <i4 with shape {expected_shape}"
        )
    if int(rows.min()) < 0 or int(rows.max()) >= ModelConfig().vocab_size:
        raise ValueError(f"quality {split} fixture has out-of-vocabulary tokens")
    return rows, QualityFixture(
        logical_path=relative_path.as_posix(),
        shape=expected_shape,
        dtype="int32",
        byte_size=path.stat().st_size,
        file_identity=file_identity(path),
        semantic_identity=semantic_row_content_identity(rows),
        source_identity=_fixture_source_identity(split, source_seed),
    )


def _require_source_disjoint(training: np.ndarray, validation: np.ndarray) -> None:
    training_rows = {
        np.ascontiguousarray(row, dtype=np.dtype("<i4")).tobytes(order="C")
        for row in training
    }
    if any(
        np.ascontiguousarray(row, dtype=np.dtype("<i4")).tobytes(order="C")
        in training_rows
        for row in validation
    ):
        raise ValueError("quality training and validation rows must be source-disjoint")


def _dtype_name(array: mx.array) -> str:
    names = {
        mx.bfloat16: "bfloat16",
        mx.float32: "float32",
        mx.int32: "int32",
        mx.uint32: "uint32",
        mx.bool_: "bool",
    }
    try:
        return names[array.dtype]
    except KeyError as error:
        raise ValueError(f"unsupported quality array dtype: {array.dtype}") from error


def _mlx_array_bytes(array: mx.array) -> bytes:
    if array.dtype == mx.bfloat16:
        value = np.asarray(array.view(mx.uint16)).astype("<u2", copy=False)
    else:
        value = np.asarray(array)
    return np.ascontiguousarray(value).tobytes(order="C")


def _array_leaf_identity(path: str, array: mx.array) -> str:
    digest = hashlib.sha256()
    digest.update(b"sml-pretraining-quality-parameter-leaf-v1\0")
    encoded_path = path.encode("utf-8")
    encoded_dtype = _dtype_name(array).encode("ascii")
    digest.update(len(encoded_path).to_bytes(4, "little"))
    digest.update(encoded_path)
    digest.update(len(encoded_dtype).to_bytes(4, "little"))
    digest.update(encoded_dtype)
    digest.update(len(array.shape).to_bytes(4, "little"))
    for dimension in array.shape:
        digest.update(int(dimension).to_bytes(8, "little"))
    digest.update(_mlx_array_bytes(array))
    return f"sha256:{digest.hexdigest()}"


def _tree_identity_from_leaf_identities(
    domain: str,
    leaves: Sequence[tuple[str, tuple[int, ...], str]],
) -> str:
    return structured_identity(
        domain,
        [
            {"path": path, "shape": list(shape), "identity": identity}
            for path, shape, identity in leaves
        ],
    )


def _array_tree_identity(domain: str, tree: object) -> str:
    leaves = sorted(tree_flatten(tree))
    if not leaves:
        raise ValueError("quality array tree must not be empty")
    mx.eval(*(array for _path, array in leaves))
    identities = []
    for path, array in leaves:
        if not isinstance(array, mx.array):
            raise ValueError("quality tree leaves must be MLX arrays")
        identities.append((path, tuple(array.shape), _array_leaf_identity(path, array)))
    return _tree_identity_from_leaf_identities(domain, identities)


def _batch_prefix_identity(
    training_semantic_identity: str,
    ordered_batches: Sequence[Sequence[int]],
) -> str:
    return structured_identity(
        "sml-pretraining-quality-batch-prefix-v1",
        {
            "training_semantic_identity": training_semantic_identity,
            "ordered_batches": [list(batch) for batch in ordered_batches],
        },
    )


def build_pretraining_quality_workload(root: Path) -> PretrainingQualityWorkload:
    if not isinstance(root, Path):
        raise TypeError("root must be a Path")
    training, training_fixture = _load_fixture(
        root,
        TRAINING_FIXTURE,
        TRAINING_SHAPE,
        split="training",
        source_seed=1729,
    )
    validation, validation_fixture = _load_fixture(
        root,
        VALIDATION_FIXTURE,
        VALIDATION_SHAPE,
        split="validation",
        source_seed=2718,
    )
    _require_source_disjoint(training, validation)
    model_config = ModelConfig()
    optimizer_config = OptimizerConfig()
    loader_config = LoaderConfig()
    model_key, _trainer_key = mx.random.split(mx.random.key(42))
    model = SMLLanguageModel(model_config, key=model_key)
    initial_working = model.parameters()
    mx.eval(initial_working)
    initial_identity = _array_tree_identity(
        "sml-pretraining-quality-initial-bf16-parameters-v1", initial_working
    )
    initial_leaves = tuple(sorted(tree_flatten(initial_working)))
    parameter_names = tuple(path for path, _value in initial_leaves)
    parameter_leaves = tuple(
        ParameterLeafSpec(
            path=path,
            shape=tuple(value.shape),
            value_count=math.prod(value.shape),
            initial_bf16_identity=_array_leaf_identity(path, value),
            initial_fp32_identity=_array_leaf_identity(path, value.astype(mx.float32)),
        )
        for path, value in initial_leaves
    )
    microstep_count = CANONICAL_STEPS * loader_config.gradient_accumulation_steps
    ordered_batches = tuple(
        tuple(
            (microstep * loader_config.microbatch_size + offset) % training.shape[0]
            for offset in range(loader_config.microbatch_size)
        )
        for microstep in range(microstep_count)
    )
    prefixes = {
        step: _batch_prefix_identity(
            training_fixture.semantic_identity,
            ordered_batches[: step * loader_config.gradient_accumulation_steps],
        )
        for step in CHECKPOINT_STEPS
    }
    evaluation_rows = tuple(range(validation.shape[0]))
    evaluation_identity = structured_identity(
        "sml-pretraining-quality-evaluation-request-v1",
        {
            "validation_semantic_identity": validation_fixture.semantic_identity,
            "ordered_row_indices": list(evaluation_rows),
            "compute_dtype": "float32",
            "loss": "shifted-causal-negative-log-likelihood-v1",
            "pad_token_id": model_config.pad_token_id,
        },
    )
    production_components = production_dependency_components(root)
    workload = PretrainingQualityWorkload(
        kind="pretraining-quality-workload",
        version=2,
        identity=_PLACEHOLDER_IDENTITY,
        training_fixture=training_fixture,
        validation_fixture=validation_fixture,
        initial_bf16_parameter_identity=initial_identity,
        parameter_leaf_names=parameter_names,
        parameter_leaves=parameter_leaves,
        model=dataclasses.asdict(model_config),
        optimizer=dataclasses.asdict(optimizer_config),
        loader=dataclasses.asdict(loader_config),
        precision={
            "candidate": {
                **dataclasses.asdict(PrecisionConfig()),
                "compute_dtype": "bfloat16",
            },
            "oracle": {
                "master_parameter_dtype": "float32",
                "working_parameter_dtype": "float32",
                "gradient_accumulator_dtype": "float32",
                "optimizer_state_dtype": "float32",
                "update_dtype": "float32",
                "master_weights": True,
                "dynamic_loss_scaling": False,
                "compute_dtype": "float32",
            },
        },
        ordered_batches=ordered_batches,
        batch_prefix_identities=prefixes,
        checkpoint_steps=CHECKPOINT_STEPS,
        evaluation_row_indices=evaluation_rows,
        evaluation_request_identity=evaluation_identity,
        model_seed=42,
        training_seed=42,
        loader_seed=loader_config.epoch_seed,
        harness_components=tuple(path.as_posix() for path in HARNESS_COMPONENTS),
        harness_identity=harness_content_identity(root),
        production_dependency_components=tuple(
            path.as_posix() for path in production_components
        ),
        production_dependency_identity=_current_production_dependency_identity(
            root, production_components
        ),
    )
    return replace(workload, identity=workload.recompute_identity())


def real_work_identity(workload: PretrainingQualityWorkload, step: int) -> str:
    if not isinstance(workload, PretrainingQualityWorkload):
        raise TypeError("workload must be a PretrainingQualityWorkload")
    if step not in CHECKPOINT_STEPS:
        raise ValueError("real-work identity step is not a reporting checkpoint")
    return structured_identity(
        "sml-pretraining-quality-real-work-v1",
        {
            "workload_identity": workload.identity,
            "initial_bf16_parameter_identity": workload.initial_bf16_parameter_identity,
            "ordered_batch_prefix_identity": workload.batch_prefix_identities[step],
            "evaluation_request_identity": workload.evaluation_request_identity,
            "step": step,
        },
    )


def _fp32_rms_norm(weight: mx.array, x: mx.array, epsilon: float) -> mx.array:
    x = x.astype(mx.float32)
    mean_square = mx.mean(mx.square(x), axis=-1, keepdims=True)
    return (x * mx.rsqrt(mean_square + epsilon) * weight.astype(mx.float32)).astype(
        mx.float32
    )


def _rotate_half(x: mx.array) -> mx.array:
    half = x.shape[-1] // 2
    return mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def _fp32_attention(
    model: SMLLanguageModel,
    layer_index: int,
    parameters: dict[str, object],
    x: mx.array,
    attention_mask: mx.array,
    positions: mx.array,
) -> mx.array:
    layer = model.layers[layer_index]
    config = model.config
    batch_size, query_length, _hidden_size = x.shape

    def linear(inputs: mx.array, projection: object) -> mx.array:
        return inputs.astype(mx.float32) @ projection["weight"].astype(mx.float32).T

    q = linear(x, parameters["q_proj"])
    k = linear(x, parameters["k_proj"])
    v = linear(x, parameters["v_proj"])
    q = q.reshape(
        (batch_size, query_length, config.num_q_heads, config.head_dim)
    ).swapaxes(1, 2)
    k = k.reshape(
        (batch_size, query_length, config.num_kv_heads, config.head_dim)
    ).swapaxes(1, 2)
    v = v.reshape(
        (batch_size, query_length, config.num_kv_heads, config.head_dim)
    ).swapaxes(1, 2)
    in_bounds = (positions >= 0) & (positions < config.effective_context_length)
    safe_positions = mx.where(in_bounds, positions, mx.zeros_like(positions))
    cos = layer.self_attn.rope.cos_cached[safe_positions][:, None, :, :]
    sin = layer.self_attn.rope.sin_cached[safe_positions][:, None, :, :]
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin
    valid = in_bounds[:, None, :, None]
    q = mx.where(valid, q, mx.full(q.shape, float("nan"), dtype=mx.float32))
    k = mx.where(valid, k, mx.full(k.shape, float("nan"), dtype=mx.float32))
    causal_mask = positions[:, None, None, :] <= positions[:, None, :, None]
    boolean_mask = causal_mask & attention_mask[:, None, None, :]
    output = mx.fast.scaled_dot_product_attention(
        q,
        k,
        v,
        scale=1.0 / math.sqrt(config.head_dim),
        mask=boolean_mask,
    ).astype(mx.float32)
    output = output.swapaxes(1, 2).reshape(
        (batch_size, query_length, config.hidden_size)
    )
    output = linear(output, parameters["o_proj"])
    return mx.where(attention_mask[:, :, None], output, 0.0).astype(mx.float32)


def _fp32_forward_arrays(
    model: SMLLanguageModel,
    parameters: dict[str, object],
    input_ids: mx.array,
    *,
    training: bool,
    key: mx.array | None,
) -> tuple[mx.array, mx.array | None]:
    batch_size, query_length = input_ids.shape
    attention_mask = mx.ones((batch_size, query_length), dtype=mx.bool_)
    positions = mx.broadcast_to(
        mx.arange(query_length, dtype=mx.int32)[None, :],
        (batch_size, query_length),
    )
    hidden = parameters["embed_tokens"]["weight"][input_ids].astype(mx.float32)
    for layer_index, layer in enumerate(model.layers):
        layer_parameters = parameters["layers"][layer_index]
        attention_input = _fp32_rms_norm(
            layer_parameters["input_norm"]["weight"],
            hidden,
            layer.input_norm.epsilon,
        )
        attention_output = _fp32_attention(
            model,
            layer_index,
            layer_parameters["self_attn"],
            attention_input,
            attention_mask,
            positions,
        )
        hidden = (hidden + attention_output).astype(mx.float32)
        mlp_input = _fp32_rms_norm(
            layer_parameters["post_attn_norm"]["weight"],
            hidden,
            layer.post_attn_norm.epsilon,
        )
        mlp_parameters = layer_parameters["mlp"]
        gate = mlp_input @ mlp_parameters["gate_proj"]["weight"].astype(mx.float32).T
        up = mlp_input @ mlp_parameters["up_proj"]["weight"].astype(mx.float32).T
        mlp_output = nn.silu(gate) * up
        mlp_output = (
            mlp_output @ mlp_parameters["down_proj"]["weight"].astype(mx.float32).T
        ).astype(mx.float32)
        if training and layer.mlp.dropout_probability > 0.0:
            if key is None:
                raise ValueError("FP32 training requires an explicit dropout key")
            mlp_output, key = keyed_dropout(
                mlp_output, layer.mlp.dropout_probability, key
            )
        hidden = (hidden + mlp_output).astype(mx.float32)
    hidden = _fp32_rms_norm(parameters["norm"]["weight"], hidden, model.norm.epsilon)
    if model.config.tie_word_embeddings:
        vocabulary_weight = parameters["embed_tokens"]["weight"]
    else:
        vocabulary_weight = parameters["lm_head"]["weight"]
    logits = hidden @ vocabulary_weight.astype(mx.float32).T
    return logits.astype(mx.float32), key


@dataclass(frozen=True, slots=True)
class _QualityKernels:
    microstep_core: object
    optimizer_step_core: object


def _build_candidate_kernels(
    model: SMLLanguageModel,
    config: PretrainingConfig,
    weight_decay_tree: dict,
) -> _QualityKernels:
    production = build_pretraining_kernels(model, config, weight_decay_tree)
    if config.compile:
        return _QualityKernels(
            production.compiled_microstep_core,
            production.compiled_optimizer_step_core,
        )
    return _QualityKernels(
        production.eager_microstep_core,
        production.eager_optimizer_step_core,
    )


def _build_oracle_kernels(
    model: SMLLanguageModel,
    config: PretrainingConfig,
    weight_decay_tree: dict,
) -> _QualityKernels:
    def loss_with_key(
        working_parameters: dict,
        input_ids: mx.array,
        labels: mx.array,
        key: mx.array,
    ) -> tuple[mx.array, mx.array]:
        logits, next_key = _fp32_forward_arrays(
            model,
            working_parameters,
            input_ids,
            training=True,
            key=key,
        )
        if next_key is None:
            raise RuntimeError("FP32 training forward did not return a PRNG key")
        valid_mask = labels != model.config.pad_token_id
        return causal_lm_loss(logits, labels, valid_mask), next_key

    loss_and_grad = mx.value_and_grad(loss_with_key)

    def microstep_core(
        working_parameters: dict,
        trainer_tree: tuple[dict, mx.array, mx.array, mx.array],
        input_ids: mx.array,
        labels: mx.array,
    ) -> tuple[dict, tuple[dict, mx.array, mx.array, mx.array]]:
        accumulators, count, key, loss_numerator = trainer_tree
        (loss, next_key), gradients = loss_and_grad(
            working_parameters, input_ids, labels, key
        )
        return working_parameters, (
            accumulate_fp32(accumulators, gradients),
            (count + mx.array(1, dtype=mx.int32)).astype(mx.int32),
            next_key,
            (loss_numerator + loss.astype(mx.float32)).astype(mx.float32),
        )

    def optimizer_step_core(
        master_parameters: dict,
        working_parameters: dict,
        adam_tree: tuple[mx.array, dict, dict],
        trainer_tree: tuple[dict, mx.array, mx.array, mx.array],
    ) -> tuple[dict, dict, tuple, tuple, dict]:
        del working_parameters
        accumulators, count, next_key, loss_numerator = trainer_tree
        gradients = normalize_and_clip(
            accumulators,
            count,
            gradient_clip_norm=config.optimizer.gradient_clip_norm,
        )
        next_masters, _derived_bf16, next_adam = adamw_mixed_precision_update_tree(
            master_parameters,
            gradients,
            adam_tree,
            config.optimizer,
            weight_decay_tree,
        )
        reset = tree_map(lambda value: value - value, accumulators)
        safe_count = mx.maximum(
            count.astype(mx.float32), mx.array(1.0, dtype=mx.float32)
        )
        metrics = {
            "learning_rate": learning_rate_at(adam_tree[0], config.optimizer),
            "accumulation_count": count,
            "loss_numerator": loss_numerator,
            "loss": (loss_numerator / safe_count).astype(mx.float32),
        }
        return (
            next_masters,
            next_masters,
            next_adam,
            (
                reset,
                (count - count).astype(mx.int32),
                next_key,
                (loss_numerator - loss_numerator).astype(mx.float32),
            ),
            metrics,
        )

    if config.compile:
        return _QualityKernels(
            mx.compile(microstep_core), mx.compile(optimizer_step_core)
        )
    return _QualityKernels(microstep_core, optimizer_step_core)


@dataclass(frozen=True, slots=True)
class _LeafUpdateObservation:
    path: str
    shape: tuple[int, ...]
    value_count: int
    before_master: mx.array
    after_master: mx.array
    before_working: mx.array
    after_working: mx.array
    before_bf16_working: mx.array
    after_bf16_working: mx.array
    nonzero_count: mx.array
    sub_ulp_count: mx.array
    survived_count: mx.array
    minimum_ratio: mx.array
    ratio_sum: mx.array
    maximum_ratio: mx.array
    changed_working_count: mx.array
    first_survival_step: mx.array


def _observe_update(
    previous_masters: dict,
    updated_masters: dict,
    previous_working: dict,
    updated_working: dict,
    cumulative_survival: Mapping[str, mx.array],
    *,
    step: int,
) -> tuple[_LeafUpdateObservation, ...]:
    if type(step) is not int or not 1 <= step <= CANONICAL_STEPS:
        raise ValueError("update observation step is invalid")
    previous_master_leaves = dict(tree_flatten(previous_masters))
    updated_master_leaves = dict(tree_flatten(updated_masters))
    previous_working_leaves = dict(tree_flatten(previous_working))
    updated_working_leaves = dict(tree_flatten(updated_working))
    if not (
        previous_master_leaves.keys()
        == updated_master_leaves.keys()
        == previous_working_leaves.keys()
        == updated_working_leaves.keys()
        == cumulative_survival.keys()
    ):
        raise ValueError("update observation trees must have exact matching paths")
    observations = []
    for path in sorted(previous_master_leaves):
        previous_master = previous_master_leaves[path].astype(mx.float32)
        updated_master = updated_master_leaves[path].astype(mx.float32)
        previous_bf16 = previous_working_leaves[path].astype(mx.bfloat16)
        updated_bf16 = updated_working_leaves[path].astype(mx.bfloat16)
        update = mx.abs(updated_master - previous_master)
        magnitude = mx.abs(previous_bf16.astype(mx.float32))
        minimum_normal = mx.array(2.0**-126, dtype=mx.float32)
        exponent = mx.floor(mx.log2(mx.maximum(magnitude, minimum_normal)))
        normal_ulp = mx.power(mx.array(2.0, dtype=mx.float32), exponent - 7.0)
        ulp = mx.where(
            magnitude < minimum_normal,
            mx.array(2.0**-133, dtype=mx.float32),
            normal_ulp,
        )
        nonzero = update > 0.0
        ratio = update / ulp
        sub_ulp = nonzero & (ratio < 1.0)
        survived = sub_ulp & (previous_bf16 == updated_bf16)
        prior_first_survival = cumulative_survival[path]
        first_survival = mx.where(
            (prior_first_survival == 0) & mx.any(survived),
            mx.array(step, dtype=mx.int32),
            prior_first_survival,
        ).astype(mx.int32)
        observations.append(
            _LeafUpdateObservation(
                path=path,
                shape=tuple(previous_master.shape),
                value_count=math.prod(previous_master.shape),
                before_master=previous_master,
                after_master=updated_master,
                before_working=previous_working_leaves[path],
                after_working=updated_working_leaves[path],
                before_bf16_working=previous_bf16,
                after_bf16_working=updated_bf16,
                nonzero_count=mx.sum(nonzero).astype(mx.int32),
                sub_ulp_count=mx.sum(sub_ulp).astype(mx.int32),
                survived_count=mx.sum(survived).astype(mx.int32),
                minimum_ratio=mx.min(
                    mx.where(nonzero, ratio, mx.array(float("inf"), mx.float32))
                ),
                ratio_sum=mx.sum(mx.where(nonzero, ratio, 0.0)).astype(mx.float32),
                maximum_ratio=mx.max(mx.where(nonzero, ratio, 0.0)),
                changed_working_count=mx.sum(previous_bf16 != updated_bf16).astype(
                    mx.int32
                ),
                first_survival_step=first_survival,
            )
        )
    return tuple(observations)


def _materialize_update_observation(
    observations: Sequence[_LeafUpdateObservation],
) -> tuple[tuple[ParameterUpdateStatistics, ...], bool]:
    arrays = [
        array
        for observation in observations
        for array in (
            observation.nonzero_count,
            observation.sub_ulp_count,
            observation.survived_count,
            observation.minimum_ratio,
            observation.ratio_sum,
            observation.maximum_ratio,
            observation.changed_working_count,
            observation.first_survival_step,
        )
    ]
    mx.eval(*arrays)
    statistics = []
    survived_any = False
    for observation in observations:
        count = int(observation.nonzero_count.item())
        if count:
            minimum = float(observation.minimum_ratio.item())
            mean = float(observation.ratio_sum.item()) / count
            maximum = float(observation.maximum_ratio.item())
        else:
            minimum = mean = maximum = None
        first_survival_step = int(observation.first_survival_step.item())
        statistics.append(
            ParameterUpdateStatistics(
                path=observation.path,
                shape=observation.shape,
                value_count=observation.value_count,
                before_master_identity=_array_leaf_identity(
                    observation.path, observation.before_master
                ),
                after_master_identity=_array_leaf_identity(
                    observation.path, observation.after_master
                ),
                before_working_identity=_array_leaf_identity(
                    observation.path, observation.before_working
                ),
                after_working_identity=_array_leaf_identity(
                    observation.path, observation.after_working
                ),
                before_bf16_working_identity=_array_leaf_identity(
                    observation.path, observation.before_bf16_working
                ),
                after_bf16_working_identity=_array_leaf_identity(
                    observation.path, observation.after_bf16_working
                ),
                nonzero_update_count=count,
                sub_bf16_ulp_update_count=int(observation.sub_ulp_count.item()),
                survived_sub_bf16_ulp_count=int(observation.survived_count.item()),
                changed_bf16_working_count=int(
                    observation.changed_working_count.item()
                ),
                first_sub_bf16_survival_step=(first_survival_step or None),
                minimum_update_to_bf16_ulp=minimum,
                mean_update_to_bf16_ulp=mean,
                maximum_update_to_bf16_ulp=maximum,
            )
        )
        survived_any = survived_any or first_survival_step > 0
    return tuple(statistics), survived_any


def _empty_update_statistics(
    masters: dict, working: dict
) -> tuple[ParameterUpdateStatistics, ...]:
    master_leaves = dict(tree_flatten(masters))
    working_leaves = dict(tree_flatten(working))
    if master_leaves.keys() != working_leaves.keys():
        raise ValueError("empty update statistic trees must match")
    statistics = []
    for path in sorted(master_leaves):
        master = master_leaves[path]
        actual_working = working_leaves[path]
        bf16_working = actual_working.astype(mx.bfloat16)
        master_identity = _array_leaf_identity(path, master)
        working_identity = _array_leaf_identity(path, actual_working)
        bf16_identity = _array_leaf_identity(path, bf16_working)
        statistics.append(
            ParameterUpdateStatistics(
                path=path,
                shape=tuple(master.shape),
                value_count=math.prod(master.shape),
                before_master_identity=master_identity,
                after_master_identity=master_identity,
                before_working_identity=working_identity,
                after_working_identity=working_identity,
                before_bf16_working_identity=bf16_identity,
                after_bf16_working_identity=bf16_identity,
                nonzero_update_count=0,
                sub_bf16_ulp_update_count=0,
                survived_sub_bf16_ulp_count=0,
                changed_bf16_working_count=0,
                first_sub_bf16_survival_step=None,
                minimum_update_to_bf16_ulp=None,
                mean_update_to_bf16_ulp=None,
                maximum_update_to_bf16_ulp=None,
            )
        )
    return tuple(statistics)


def _changed_working_fraction(observations: Sequence[_LeafUpdateObservation]) -> float:
    if not observations:
        return 0.0
    mx.eval(*(item.changed_working_count for item in observations))
    changed = sum(int(item.changed_working_count.item()) for item in observations)
    total = sum(item.value_count for item in observations)
    return changed / total


def _is_rms_norm_path(path: str) -> bool:
    return (
        ".input_norm." in path or ".post_attn_norm." in path or path.startswith("norm.")
    )


def _derived_checkpoint_telemetry(
    workload: PretrainingQualityWorkload,
    statistics: Sequence[ParameterUpdateStatistics],
) -> tuple[float, bool, bool]:
    leaves = dict(zip(workload.parameter_leaf_names, workload.parameter_leaves))
    total_values = sum(item.value_count for item in statistics)
    if total_values <= 0:
        raise ValueError("quality telemetry has no parameter values")
    changed_fraction = (
        sum(item.changed_bf16_working_count for item in statistics) / total_values
    )
    rms_norm_moved = any(
        _is_rms_norm_path(item.path)
        and item.after_master_identity != leaves[item.path].initial_fp32_identity
        for item in statistics
    )
    sub_bf16_survived = any(
        item.first_sub_bf16_survival_step is not None for item in statistics
    )
    return changed_fraction, rms_norm_moved, sub_bf16_survived


def _validate_checkpoint_telemetry(
    workload: PretrainingQualityWorkload,
    record: PretrainingQualityCheckpoint,
    previous_record: PretrainingQualityCheckpoint | None,
) -> None:
    statistics = record.update_statistics
    specs = dict(zip(workload.parameter_leaf_names, workload.parameter_leaves))
    if tuple(item.path for item in statistics) != workload.parameter_leaf_names:
        raise ValueError("quality checkpoint parameter statistics are incomplete")
    previous_statistics = (
        {}
        if previous_record is None
        else {item.path: item for item in previous_record.update_statistics}
    )
    for item in statistics:
        spec = specs[item.path]
        if item.shape != spec.shape or item.value_count != spec.value_count:
            raise ValueError("quality telemetry leaf shape or value count changed")
        if (item.nonzero_update_count == 0) != (
            item.before_master_identity == item.after_master_identity
        ):
            raise ValueError("quality master before/after identities contradict counts")
        if (item.changed_bf16_working_count == 0) != (
            item.before_bf16_working_identity == item.after_bf16_working_identity
        ):
            raise ValueError("quality BF16 working identities contradict changed count")
        if record.runtime == "candidate" and (
            item.before_working_identity != item.before_bf16_working_identity
            or item.after_working_identity != item.after_bf16_working_identity
        ):
            raise ValueError("candidate working evidence must be BF16")
        if record.runtime == "oracle" and (
            item.before_working_identity != item.before_master_identity
            or item.after_working_identity != item.after_master_identity
        ):
            raise ValueError("oracle working evidence must match FP32 masters")
        first_survival = item.first_sub_bf16_survival_step
        if first_survival is not None and first_survival > record.step:
            raise ValueError("cumulative sub-BF16 survival exceeds checkpoint step")
        if item.survived_sub_bf16_ulp_count > 0 and first_survival is None:
            raise ValueError("cumulative sub-BF16 survival evidence is missing")
        if first_survival == record.step and item.survived_sub_bf16_ulp_count == 0:
            raise ValueError(
                "first cumulative sub-BF16 survival lacks current evidence"
            )
        if previous_record is not None:
            previous_first = previous_statistics[item.path].first_sub_bf16_survival_step
            if previous_first is not None and first_survival != previous_first:
                raise ValueError(
                    "cumulative sub-BF16 survival changed after first proof"
                )
            if (
                previous_first is None
                and first_survival is not None
                and not (previous_record.step < first_survival <= record.step)
            ):
                raise ValueError("cumulative sub-BF16 survival step is inconsistent")
        if record.step == 0:
            if (
                item.nonzero_update_count != 0
                or item.sub_bf16_ulp_update_count != 0
                or item.survived_sub_bf16_ulp_count != 0
                or item.changed_bf16_working_count != 0
                or first_survival is not None
                or item.minimum_update_to_bf16_ulp is not None
                or item.mean_update_to_bf16_ulp is not None
                or item.maximum_update_to_bf16_ulp is not None
            ):
                raise ValueError("step-zero telemetry must contain only zero counts")
            if (
                item.before_master_identity != spec.initial_fp32_identity
                or item.after_master_identity != spec.initial_fp32_identity
                or item.before_bf16_working_identity != spec.initial_bf16_identity
                or item.after_bf16_working_identity != spec.initial_bf16_identity
            ):
                raise ValueError("step-zero telemetry does not match initialization")
    expected_master_identity = _telemetry_tree_identity(
        _MASTER_TREE_IDENTITY_DOMAIN, statistics, "after_master_identity"
    )
    if record.master_parameter_identity != expected_master_identity:
        raise ValueError("quality master tree identity does not match leaf telemetry")
    expected_working_identity = _telemetry_tree_identity(
        _WORKING_TREE_IDENTITY_DOMAIN, statistics, "after_working_identity"
    )
    if record.working_parameter_identity != expected_working_identity:
        raise ValueError("quality working tree identity does not match leaf telemetry")
    changed_fraction, rms_norm_moved, sub_bf16_survived = _derived_checkpoint_telemetry(
        workload, statistics
    )
    if record.changed_bf16_working_fraction != changed_fraction:
        raise ValueError("quality changed working fraction is not derived")
    if record.rms_norm_master_moved is not rms_norm_moved:
        raise ValueError("quality RMSNorm master movement is not derived")
    if record.sub_bf16_update_survived is not sub_bf16_survived:
        raise ValueError("quality sub-BF16 update survival is not derived")
    expected_state_value_count = (
        5 * sum(spec.value_count for spec in workload.parameter_leaves) + 1
    )
    expected_finite = (
        record.nonfinite_state_value_count == 0
        and math.isfinite(record.train_loss)
        and math.isfinite(record.validation_nll)
    )
    if (
        record.finite_state_value_count != expected_state_value_count
        or record.finite is not expected_finite
    ):
        raise ValueError("quality finite state status is not derived")
    if record.step == 0 and (
        changed_fraction != 0.0 or rms_norm_moved or sub_bf16_survived
    ):
        raise ValueError("step-zero quality checkpoint has update evidence")


def validate_pretraining_quality_records(
    workload: PretrainingQualityWorkload,
    records: Sequence[PretrainingQualityCheckpoint],
) -> PretrainingQualityReport:
    if len(records) != 8:
        raise ValueError("pretraining quality evidence requires exactly eight records")
    expected_order = tuple(
        (runtime, step)
        for runtime in ("candidate", "oracle")
        for step in CHECKPOINT_STEPS
    )
    if tuple((record.runtime, record.step) for record in records) != expected_order:
        raise ValueError(
            "pretraining quality records have an invalid runtime/step order"
        )
    previous_by_runtime: dict[str, PretrainingQualityCheckpoint] = {}
    for record in records:
        if PretrainingQualityCheckpoint.from_dict(record.to_dict()) != record:
            raise ValueError("quality checkpoint does not round-trip canonically")
        if record.identity != record.recompute_identity():
            raise ValueError("quality checkpoint identity mismatch")
        if record.workload_identity != workload.identity:
            raise ValueError("quality checkpoint workload identity mismatch")
        if record.real_work_identity != real_work_identity(workload, record.step):
            raise ValueError("quality checkpoint real-work identity mismatch")
        if (
            record.initial_bf16_parameter_identity
            != workload.initial_bf16_parameter_identity
        ):
            raise ValueError("quality checkpoint initial parameter identity mismatch")
        if (
            record.ordered_batch_prefix_identity
            != workload.batch_prefix_identities[record.step]
        ):
            raise ValueError("quality checkpoint batch-prefix identity mismatch")
        if record.evaluation_request_identity != workload.evaluation_request_identity:
            raise ValueError("quality checkpoint evaluation identity mismatch")
        _validate_checkpoint_telemetry(
            workload, record, previous_by_runtime.get(record.runtime)
        )
        previous_by_runtime[record.runtime] = record
    candidate = [record for record in records if record.runtime == "candidate"]
    oracle = [record for record in records if record.runtime == "oracle"]
    for candidate_record, oracle_record in zip(candidate, oracle, strict=True):
        if candidate_record.trainer_key_identity != oracle_record.trainer_key_identity:
            raise ValueError("candidate/oracle PRNG key identities do not match")
    if candidate[0].master_parameter_identity != oracle[0].master_parameter_identity:
        raise ValueError("candidate/oracle step-zero master identities do not match")
    matching_work = all(
        left.real_work_identity == right.real_work_identity
        for left, right in zip(candidate, oracle, strict=True)
    )
    return PretrainingQualityReport(
        candidate_validation_nll=candidate[-1].validation_nll,
        oracle_validation_nll=oracle[-1].validation_nll,
        candidate_finite=all(record.finite for record in candidate),
        oracle_finite=all(record.finite for record in oracle),
        rms_norm_master_moved=candidate[-1].rms_norm_master_moved,
        sub_bf16_update_survived=candidate[-1].sub_bf16_update_survived,
        matching_work_identity=matching_work,
    )


def _canonical_steps(value: str) -> int:
    try:
        steps = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("--steps must be exactly 1000") from error
    if steps != CANONICAL_STEPS:
        raise argparse.ArgumentTypeError("--steps must be exactly 1000")
    return steps


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Controlled pretraining-quality gate")
    commands = parser.add_subparsers(dest="command", required=True)
    record = commands.add_parser("record")
    record.add_argument("--steps", type=_canonical_steps, required=True)
    record.add_argument("--manifest", type=Path, required=True)
    record.add_argument("--raw-output", type=Path, required=True)
    record.add_argument("--output", type=Path, required=True)
    record.set_defaults(handler=_record)
    validate = commands.add_parser("validate")
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--raw-input", type=Path, required=True)
    validate.add_argument("--report", type=Path, required=True)
    validate.set_defaults(handler=_validate)
    return parser


def _state_finiteness_counts(*trees: object) -> tuple[int, int]:
    floating_leaves = [
        value
        for tree in trees
        for _path, value in tree_flatten(tree)
        if value.dtype not in (mx.uint32, mx.int32, mx.bool_)
    ]
    nonfinite_counts = [mx.sum(~mx.isfinite(value)) for value in floating_leaves]
    mx.eval(*nonfinite_counts)
    return (
        sum(math.prod(value.shape) for value in floating_leaves),
        sum(int(value.item()) for value in nonfinite_counts),
    )


def _validation_nll(
    model: SMLLanguageModel,
    masters: dict,
    rows: np.ndarray,
) -> float:
    losses = []
    for row in rows:
        row_array = mx.array(row[None, :], dtype=mx.int32)
        logits, _key = _fp32_forward_arrays(
            model,
            masters,
            row_array[:, :-1],
            training=False,
            key=None,
        )
        labels = row_array[:, 1:]
        loss = causal_lm_loss(
            logits,
            labels,
            labels != model.config.pad_token_id,
        )
        mx.eval(loss)
        losses.append(float(loss.item()))
    return math.fsum(losses) / len(losses)


def _trainer_key_identity(trainer_tree: tuple) -> str:
    key = trainer_tree[2]
    mx.eval(key)
    return structured_identity(
        "sml-pretraining-quality-trainer-key-v1",
        [int(value) for value in np.asarray(key).tolist()],
    )


def _quality_config(
    root: Path, workload: PretrainingQualityWorkload
) -> PretrainingConfig:
    optimizer_values = dict(workload.optimizer)
    weight_decay_values = optimizer_values.pop("weight_decay")
    if not isinstance(weight_decay_values, dict):
        raise ValueError("quality optimizer weight decay must be an object")
    return PretrainingConfig(
        data=root / workload.training_fixture.logical_path,
        output_run=root / ".quality-runtime-not-published",
        model=ModelConfig(**workload.model),
        optimizer=OptimizerConfig(
            **optimizer_values,
            weight_decay=WeightDecayPolicy(**weight_decay_values),
        ),
        loader=LoaderConfig(**workload.loader),
        maximum_steps=CANONICAL_STEPS,
        maximum_epochs=None,
        log_interval=10,
        seed=workload.training_seed,
        compile=True,
    )


def _checkpoint_record(
    *,
    runtime: RuntimeName,
    step: int,
    workload: PretrainingQualityWorkload,
    masters: dict,
    working: dict,
    adam_tree: tuple,
    trainer_tree: tuple,
    train_loss: float,
    validation_nll: float,
    observations: Sequence[_LeafUpdateObservation],
) -> PretrainingQualityCheckpoint:
    if observations:
        statistics, _survived = _materialize_update_observation(observations)
    else:
        statistics = _empty_update_statistics(masters, working)
    changed_fraction, rms_norm_moved, survived = _derived_checkpoint_telemetry(
        workload, statistics
    )
    finite_state_value_count, nonfinite_state_value_count = _state_finiteness_counts(
        masters, working, adam_tree, trainer_tree
    )
    finite = (
        nonfinite_state_value_count == 0
        and math.isfinite(train_loss)
        and math.isfinite(validation_nll)
    )
    checkpoint = PretrainingQualityCheckpoint(
        kind="pretraining-quality-checkpoint",
        version=1,
        identity=_PLACEHOLDER_IDENTITY,
        runtime=runtime,
        compute_dtype="bfloat16" if runtime == "candidate" else "float32",
        step=step,
        workload_identity=workload.identity,
        real_work_identity=real_work_identity(workload, step),
        initial_bf16_parameter_identity=workload.initial_bf16_parameter_identity,
        ordered_batch_prefix_identity=workload.batch_prefix_identities[step],
        evaluation_request_identity=workload.evaluation_request_identity,
        master_parameter_identity=_telemetry_tree_identity(
            _MASTER_TREE_IDENTITY_DOMAIN,
            statistics,
            "after_master_identity",
        ),
        working_parameter_identity=_telemetry_tree_identity(
            _WORKING_TREE_IDENTITY_DOMAIN,
            statistics,
            "after_working_identity",
        ),
        trainer_key_identity=_trainer_key_identity(trainer_tree),
        train_loss=train_loss,
        validation_nll=validation_nll,
        finite_state_value_count=finite_state_value_count,
        nonfinite_state_value_count=nonfinite_state_value_count,
        finite=finite,
        update_statistics=statistics,
        changed_bf16_working_fraction=changed_fraction,
        rms_norm_master_moved=rms_norm_moved,
        sub_bf16_update_survived=survived,
    )
    return replace(checkpoint, identity=checkpoint.recompute_identity())


@dataclass(frozen=True, slots=True)
class _RuntimeLoopState:
    masters: dict
    working: dict
    adam_tree: tuple
    trainer_tree: tuple
    cumulative_survival: dict[str, mx.array]
    observations: tuple[_LeafUpdateObservation, ...]
    metrics: dict[str, mx.array]
    microstep_index: int


def _execute_training_steps(
    *,
    kernels: _QualityKernels,
    runtime: RuntimeName,
    gradient_accumulation_steps: int,
    ordered_batches: Sequence[Sequence[int]],
    training_rows: np.ndarray,
    start_step: int,
    stop_step: int,
    state: _RuntimeLoopState,
) -> _RuntimeLoopState:
    if runtime not in ("candidate", "oracle"):
        raise ValueError("quality runtime must be candidate or oracle")
    if not 0 <= start_step < stop_step <= CANONICAL_STEPS:
        raise ValueError("quality execution step range is invalid")
    masters = state.masters
    working = state.working
    adam_tree = state.adam_tree
    trainer_tree = state.trainer_tree
    cumulative_survival = state.cumulative_survival
    observations = state.observations
    metrics = state.metrics
    microstep_index = state.microstep_index
    for step in range(start_step + 1, stop_step + 1):
        for _microstep in range(gradient_accumulation_steps):
            batch_indices = ordered_batches[microstep_index]
            batch = mx.array(training_rows[list(batch_indices)], dtype=mx.int32)
            working, trainer_tree = kernels.microstep_core(
                working,
                trainer_tree,
                batch[:, :-1],
                batch[:, 1:],
            )
            microstep_index += 1
        previous_masters = masters
        previous_working = working
        masters, working, adam_tree, trainer_tree, metrics = (
            kernels.optimizer_step_core(
                masters,
                working,
                adam_tree,
                trainer_tree,
            )
        )
        updated_working = working
        observations = _observe_update(
            previous_masters,
            masters,
            previous_working,
            updated_working,
            cumulative_survival,
            step=step,
        )
        cumulative_survival = {
            item.path: item.first_survival_step for item in observations
        }
        # Submit each dependent transition without a host synchronization. This
        # bounds the lazy graph while preserving synchronization only at reports.
        mx.async_eval(
            masters,
            working,
            adam_tree,
            trainer_tree,
            metrics,
            cumulative_survival,
        )
        if step % 25 == 0 and step not in CHECKPOINT_STEPS:
            print(
                f"quality {runtime}: submitted optimizer step {step}/{CANONICAL_STEPS}",
                file=sys.stderr,
                flush=True,
            )
    return _RuntimeLoopState(
        masters,
        working,
        adam_tree,
        trainer_tree,
        cumulative_survival,
        observations,
        metrics,
        microstep_index,
    )


def _run_runtime(
    root: Path,
    workload: PretrainingQualityWorkload,
    runtime: RuntimeName,
    training_rows: np.ndarray,
    validation_rows: np.ndarray,
) -> tuple[tuple[PretrainingQualityCheckpoint, ...], float]:
    started = time.monotonic()
    config = _quality_config(root, workload)
    model_key, trainer_key = mx.random.split(mx.random.key(workload.training_seed))
    model = SMLLanguageModel(config.model, key=model_key)
    initial_working = model.parameters()
    mx.eval(initial_working)
    parameters = initialize_base_parameter_state(initial_working)
    if (
        _array_tree_identity(
            "sml-pretraining-quality-initial-bf16-parameters-v1",
            parameters.working_parameters,
        )
        != workload.initial_bf16_parameter_identity
    ):
        raise RuntimeError("runtime initial BF16 parameter identity changed")
    masters = parameters.master_parameters
    working = parameters.working_parameters if runtime == "candidate" else masters
    adam_tree = initialize_adam_state(masters).to_tree()
    trainer_tree = TrainerState(
        accumulators=tree_map(mx.zeros_like, masters),
        accumulation_count=mx.array(0, dtype=mx.int32),
        next_key=trainer_key,
        loss_numerator=mx.array(0.0, dtype=mx.float32),
    ).to_tree()
    decay = build_weight_decay_tree(
        parameters.working_parameters, config.optimizer.weight_decay
    )
    kernels = (
        _build_candidate_kernels(model, config, decay)
        if runtime == "candidate"
        else _build_oracle_kernels(model, config, decay)
    )
    cumulative_survival = {
        path: mx.array(0, dtype=mx.int32) for path, _value in tree_flatten(masters)
    }
    records = []
    initial_train_loss = _validation_nll(model, masters, training_rows[:1])
    initial_validation_nll = _validation_nll(model, masters, validation_rows)
    records.append(
        _checkpoint_record(
            runtime=runtime,
            step=0,
            workload=workload,
            masters=masters,
            working=working,
            adam_tree=adam_tree,
            trainer_tree=trainer_tree,
            train_loss=initial_train_loss,
            validation_nll=initial_validation_nll,
            observations=(),
        )
    )
    print(
        f"quality {runtime}: checkpoint 0/{CANONICAL_STEPS} "
        f"validation_nll={initial_validation_nll:.6f}",
        file=sys.stderr,
        flush=True,
    )
    state = _RuntimeLoopState(
        masters,
        working,
        adam_tree,
        trainer_tree,
        cumulative_survival,
        (),
        {"loss": mx.array(initial_train_loss, dtype=mx.float32)},
        0,
    )
    previous_step = 0
    for step in CHECKPOINT_STEPS[1:]:
        state = _execute_training_steps(
            kernels=kernels,
            runtime=runtime,
            gradient_accumulation_steps=config.loader.gradient_accumulation_steps,
            ordered_batches=workload.ordered_batches,
            training_rows=training_rows,
            start_step=previous_step,
            stop_step=step,
            state=state,
        )
        mx.eval(
            state.masters,
            state.working,
            state.adam_tree,
            state.trainer_tree,
            state.metrics,
        )
        train_loss = float(state.metrics["loss"].item())
        validation_nll = _validation_nll(model, state.masters, validation_rows)
        records.append(
            _checkpoint_record(
                runtime=runtime,
                step=step,
                workload=workload,
                masters=state.masters,
                working=state.working,
                adam_tree=state.adam_tree,
                trainer_tree=state.trainer_tree,
                train_loss=train_loss,
                validation_nll=validation_nll,
                observations=state.observations,
            )
        )
        print(
            f"quality {runtime}: checkpoint {step}/{CANONICAL_STEPS} "
            f"train_loss={train_loss:.6f} validation_nll={validation_nll:.6f} "
            f"elapsed_seconds={time.monotonic() - started:.1f}",
            file=sys.stderr,
            flush=True,
        )
        previous_step = step
    return tuple(records), time.monotonic() - started


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _manifest_document(
    *,
    workload: PretrainingQualityWorkload,
    source_commit: str,
    recording_command: Mapping[str, object],
    phase_times: Mapping[str, float],
    peak_memory: int,
    raw_identity: str,
    raw_file_identity: str,
    raw_bytes: int,
    report_identity: str,
    report_file_identity: str,
    report_bytes: int,
    recording_session_identity: str,
) -> dict[str, object]:
    normalized_phases = {
        name: _require_finite(phase_times.get(name), f"quality {name} phase")
        for name in _PHASE_NAMES
    }
    if set(phase_times) != set(_PHASE_NAMES) or any(
        value <= 0.0 for value in normalized_phases.values()
    ):
        raise ValueError("quality phase timing fields are invalid")
    measured_wall_time = math.fsum(normalized_phases.values())
    command = dict(recording_command)
    destinations = _destinations_from_recording_command(command)
    if recording_session_identity != _recording_session_identity(
        source_commit, workload.identity, command
    ):
        raise ValueError("quality recording session identity is not derived")
    owner = _publication_owner_document(
        session_identity=recording_session_identity,
        source_commit=source_commit,
        workload_identity=workload.identity,
        destinations=destinations,
    )
    manifest_size = 0
    metadata_size = 0
    temporary_high_water = raw_bytes + report_bytes
    result: dict[str, object] | None = None
    for _iteration in range(20):
        artifact_sizes = {
            "raw": raw_bytes,
            "manifest": manifest_size,
            "report": report_bytes,
        }
        body = {
            "kind": "pretraining-quality-manifest",
            "version": 2,
            "source_commit": source_commit,
            "harness_commit": source_commit,
            "harness_clean": True,
            "harness_identity": workload.harness_identity,
            "production_dependency_components": list(
                workload.production_dependency_components
            ),
            "production_dependency_identity": (workload.production_dependency_identity),
            "workload": workload.to_dict(),
            "workload_identity": workload.identity,
            "recording_command": command,
            "measurement_boundaries": dict(MEASUREMENT_BOUNDARIES),
            "phase_wall_time_seconds": normalized_phases,
            "measured_wall_time_seconds": measured_wall_time,
            "wall_time_budget_seconds": QUALITY_WALL_TIME_BUDGET_SECONDS,
            "peak_metal_memory_bytes": peak_memory,
            "publication_strategy": _PUBLICATION_STRATEGY,
            "recording_session_identity": recording_session_identity,
            "temporary_disk_high_water_bytes": temporary_high_water,
            "artifact_byte_sizes": artifact_sizes,
            "publication_metadata_bytes": metadata_size,
            "fixture_bytes": (
                workload.training_fixture.byte_size
                + workload.validation_fixture.byte_size
            ),
            "training_cardinality": workload.training_fixture.shape[0],
            "validation_cardinality": workload.validation_fixture.shape[0],
            "ordered_work_count": len(workload.ordered_batches),
            "record_count": 8,
            "raw_identity": raw_identity,
            "raw_file_identity": raw_file_identity,
            "report_identity": report_identity,
            "report_file_identity": report_file_identity,
        }
        result = {
            **body,
            "identity": structured_identity(
                "sml-pretraining-quality-manifest-v3", body
            ),
        }
        next_manifest_size = len(canonical_json_bytes(result))
        next_artifact_sizes = {
            **artifact_sizes,
            "manifest": next_manifest_size,
        }
        next_metadata_size = _publication_metadata_size(owner, next_artifact_sizes)
        next_high_water = sum(next_artifact_sizes.values()) + next_metadata_size
        if (
            next_manifest_size,
            next_metadata_size,
            next_high_water,
        ) == (manifest_size, metadata_size, temporary_high_water):
            return result
        manifest_size = next_manifest_size
        metadata_size = next_metadata_size
        temporary_high_water = next_high_water
    raise RuntimeError("quality manifest byte-size fixed point did not converge")


def _report_document(
    workload_identity: str,
    raw_identity: str,
    report: PretrainingQualityReport,
) -> dict[str, object]:
    body = {
        "kind": "pretraining-quality-report",
        "version": 1,
        "workload_identity": workload_identity,
        "raw_identity": raw_identity,
        "report": dataclasses.asdict(report),
        "decision": decide_pretraining_quality(report),
    }
    return {
        **body,
        "identity": structured_identity("sml-pretraining-quality-report-v1", body),
    }


def _canonical_evidence_destinations(
    root: Path,
    manifest: Path,
    raw_output: Path,
    report: Path,
) -> _EvidenceDestinations:
    expected = _EvidenceDestinations(
        manifest=(root / CANONICAL_MANIFEST_PATH).resolve(),
        raw_output=(root / CANONICAL_RAW_PATH).resolve(),
        report=(root / CANONICAL_REPORT_PATH).resolve(),
    )
    actual = _EvidenceDestinations(
        manifest=manifest.resolve(),
        raw_output=raw_output.resolve(),
        report=report.resolve(),
    )
    if actual != expected:
        raise ValueError("record requires the exact canonical evidence destinations")
    return actual


def _recording_command_document(
    root: Path, destinations: _EvidenceDestinations
) -> dict[str, object]:
    canonical_destinations = {
        "manifest": CANONICAL_MANIFEST_PATH.as_posix(),
        "raw_output": CANONICAL_RAW_PATH.as_posix(),
        "report": CANONICAL_REPORT_PATH.as_posix(),
    }
    resolved = {
        "manifest": destinations.manifest.resolve().as_posix(),
        "raw_output": destinations.raw_output.resolve().as_posix(),
        "report": destinations.report.resolve().as_posix(),
    }
    expected_resolved = {
        name: (root / path).resolve().as_posix()
        for name, path in canonical_destinations.items()
    }
    if resolved != expected_resolved:
        raise ValueError("record requires the exact canonical evidence destinations")
    return {
        "module": "v2.benchmarks.quality",
        "subcommand": "record",
        "steps": CANONICAL_STEPS,
        "argv": [
            "record",
            "--steps",
            str(CANONICAL_STEPS),
            "--manifest",
            canonical_destinations["manifest"],
            "--raw-output",
            canonical_destinations["raw_output"],
            "--output",
            canonical_destinations["report"],
        ],
        "destinations": canonical_destinations,
    }


def _destinations_from_recording_command(
    command: Mapping[str, object],
) -> _EvidenceDestinations:
    if set(command) != {
        "module",
        "subcommand",
        "steps",
        "argv",
        "destinations",
    }:
        raise ValueError("quality recording command has an invalid field set")
    destinations = command["destinations"]
    if not isinstance(destinations, dict):
        raise ValueError("quality recording destinations must be an object")
    expected_destinations = {
        "manifest": CANONICAL_MANIFEST_PATH.as_posix(),
        "raw_output": CANONICAL_RAW_PATH.as_posix(),
        "report": CANONICAL_REPORT_PATH.as_posix(),
    }
    if destinations != expected_destinations:
        raise ValueError("quality recording destinations are not canonical")
    expected_argv = [
        "record",
        "--steps",
        str(CANONICAL_STEPS),
        "--manifest",
        expected_destinations["manifest"],
        "--raw-output",
        expected_destinations["raw_output"],
        "--output",
        expected_destinations["report"],
    ]
    if (
        command["module"] != "v2.benchmarks.quality"
        or command["subcommand"] != "record"
        or command["steps"] != CANONICAL_STEPS
        or command["argv"] != expected_argv
    ):
        raise ValueError("quality recording argv is not canonical")
    return _EvidenceDestinations(
        manifest=Path(expected_destinations["manifest"]),
        raw_output=Path(expected_destinations["raw_output"]),
        report=Path(expected_destinations["report"]),
    )


def _recording_session_identity(
    source_commit: str,
    workload_identity: str,
    recording_command: Mapping[str, object],
) -> str:
    return structured_identity(
        "sml-pretraining-quality-recording-session-v2",
        {
            "source_commit": source_commit,
            "workload_identity": workload_identity,
            "recording_command": dict(recording_command),
        },
    )


def _signed_document(domain: str, body: dict[str, object]) -> dict[str, object]:
    return {**body, "identity": structured_identity(domain, body)}


def _publication_owner_document(
    *,
    session_identity: str,
    source_commit: str,
    workload_identity: str,
    destinations: _EvidenceDestinations,
) -> dict[str, object]:
    body = {
        "kind": "pretraining-quality-publication-owner",
        "version": 2,
        "session_identity": _require_identity(
            session_identity, "publication session identity"
        ),
        "source_commit": source_commit,
        "workload_identity": _require_identity(
            workload_identity, "publication workload identity"
        ),
        "destinations": {
            "raw": CANONICAL_RAW_PATH.as_posix(),
            "manifest": CANONICAL_MANIFEST_PATH.as_posix(),
            "report": CANONICAL_REPORT_PATH.as_posix(),
        },
    }
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("publication source commit must be a full Git commit")
    if not isinstance(destinations, _EvidenceDestinations):
        raise TypeError("destinations must be evidence destinations")
    return _signed_document("sml-pretraining-quality-publication-owner-v2", body)


def _publication_plan_document(
    owner: Mapping[str, object], artifacts: Mapping[str, Mapping[str, object]]
) -> dict[str, object]:
    body = {
        "kind": "pretraining-quality-publication-plan",
        "version": 1,
        "owner_identity": owner["identity"],
        "session_identity": owner["session_identity"],
        "artifacts": {
            name: dict(artifacts[name]) for name in ("raw", "manifest", "report")
        },
    }
    return _signed_document("sml-pretraining-quality-publication-plan-v1", body)


def _completion_document(
    owner: Mapping[str, object], plan: Mapping[str, object]
) -> dict[str, object]:
    body = {
        "kind": "pretraining-quality-publication-completion",
        "version": 1,
        "owner_identity": owner["identity"],
        "session_identity": owner["session_identity"],
        "plan_identity": plan["identity"],
    }
    return _signed_document("sml-pretraining-quality-publication-completion-v1", body)


def _publication_metadata_size(
    owner: Mapping[str, object], artifact_sizes: Mapping[str, int]
) -> int:
    artifacts = {
        name: {
            "staged_name": f"{name}.payload",
            "byte_size": artifact_sizes[name],
            "file_identity": _PLACEHOLDER_IDENTITY,
        }
        for name in ("raw", "manifest", "report")
    }
    plan = _publication_plan_document(owner, artifacts)
    completion = _completion_document(owner, plan)
    return sum(
        len(canonical_json_bytes(document)) for document in (owner, plan, completion)
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _durable_create(path: Path, payload: bytes) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as destination:
            destination.write(payload)
            destination.flush()
            os.fsync(destination.fileno())
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _read_publication_payload(path: Path, *, label: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"quality {label} must be a regular file") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"quality {label} must be a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as source:
            return source.read()
    finally:
        os.close(descriptor)


def _read_publication_document(path: Path) -> dict[str, object]:
    return _decode_json_object(
        _read_publication_payload(path, label=f"publication {path.name}"),
        label=f"publication {path.name}",
    )


def _validate_publication_owner(
    actual: Mapping[str, object], expected: Mapping[str, object]
) -> None:
    if actual != expected:
        raise ValueError("quality recovery directory belongs to another session")


def _validated_publication_plan(
    plan: Mapping[str, object], owner: Mapping[str, object]
) -> dict[str, dict[str, object]]:
    if set(plan) != {
        "kind",
        "version",
        "identity",
        "owner_identity",
        "session_identity",
        "artifacts",
    }:
        raise ValueError("quality publication plan has an invalid field set")
    body = {key: value for key, value in plan.items() if key != "identity"}
    if (
        plan["kind"] != "pretraining-quality-publication-plan"
        or plan["version"] != 1
        or plan["identity"]
        != structured_identity("sml-pretraining-quality-publication-plan-v1", body)
        or plan["owner_identity"] != owner["identity"]
        or plan["session_identity"] != owner["session_identity"]
    ):
        raise ValueError("quality publication plan ownership is invalid")
    artifacts = plan["artifacts"]
    if not isinstance(artifacts, dict) or set(artifacts) != {
        "raw",
        "manifest",
        "report",
    }:
        raise ValueError("quality publication plan artifacts are incomplete")
    normalized = {}
    for name in ("raw", "manifest", "report"):
        artifact = artifacts[name]
        if not isinstance(artifact, dict) or set(artifact) != {
            "staged_name",
            "byte_size",
            "file_identity",
        }:
            raise ValueError("quality publication artifact has invalid fields")
        if artifact["staged_name"] != f"{name}.payload":
            raise ValueError("quality publication staged name changed")
        normalized[name] = {
            "staged_name": artifact["staged_name"],
            "byte_size": _require_plain_int(
                artifact["byte_size"], f"quality staged {name} byte size", minimum=1
            ),
            "file_identity": _require_identity(
                artifact["file_identity"], f"quality staged {name} identity"
            ),
        }
    return normalized


def _remove_owned_recovery(publication: _EvidencePublication) -> None:
    owner_path = publication.recovery_directory / "owner.json"
    if owner_path.is_symlink() or not owner_path.is_file():
        raise ValueError("publication owner must be a regular file")
    _validate_publication_owner(
        _read_publication_document(owner_path), publication.owner
    )
    allowed = {"owner.json"}
    validated_metadata: dict[str, tuple[int, int]] = {}
    owner_metadata = owner_path.stat(follow_symlinks=False)
    validated_metadata["owner.json"] = (owner_metadata.st_dev, owner_metadata.st_ino)
    plan_path = publication.recovery_directory / "plan.payload"
    if plan_path.exists() or plan_path.is_symlink():
        if plan_path.is_symlink() or not plan_path.is_file():
            raise ValueError("quality recovery plan is not a regular file")
        plan = _read_publication_document(plan_path)
        artifacts = _validated_publication_plan(plan, publication.owner)
        allowed.update(
            {
                "plan.payload",
                "completion.payload",
                "completed.json",
                *(artifact["staged_name"] for artifact in artifacts.values()),
            }
        )
        for name, artifact in artifacts.items():
            path = publication.recovery_directory / artifact["staged_name"]
            if not path.exists() and not path.is_symlink():
                continue
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"quality staged {name} is not a regular file")
            payload = _read_publication_payload(path, label=f"staged {name}")
            if (
                len(payload) != artifact["byte_size"]
                or _payload_identity(payload) != artifact["file_identity"]
            ):
                raise ValueError(f"quality staged {name} changed before cleanup")
            metadata = path.stat(follow_symlinks=False)
            validated_metadata[path.name] = (metadata.st_dev, metadata.st_ino)
        completion_path = publication.recovery_directory / "completion.payload"
        if completion_path.exists() or completion_path.is_symlink():
            if completion_path.is_symlink() or not completion_path.is_file():
                raise ValueError("quality completion payload is not a regular file")
            expected_completion = canonical_json_bytes(
                _completion_document(publication.owner, plan)
            )
            if (
                _read_publication_payload(completion_path, label="completion payload")
                != expected_completion
            ):
                raise ValueError("quality completion payload changed before cleanup")
            metadata = completion_path.stat(follow_symlinks=False)
            validated_metadata[completion_path.name] = (
                metadata.st_dev,
                metadata.st_ino,
            )
        completed_path = publication.recovery_directory / "completed.json"
        if completed_path.exists() or completed_path.is_symlink():
            if (
                completed_path.is_symlink()
                or not completed_path.is_file()
                or not completion_path.is_file()
            ):
                raise ValueError("quality completion marker is invalid")
            completed_metadata = completed_path.stat(follow_symlinks=False)
            completion_metadata = completion_path.stat(follow_symlinks=False)
            if (completed_metadata.st_dev, completed_metadata.st_ino) != (
                completion_metadata.st_dev,
                completion_metadata.st_ino,
            ):
                raise ValueError("quality completion marker is not owned")
            validated_metadata[completed_path.name] = (
                completed_metadata.st_dev,
                completed_metadata.st_ino,
            )
        metadata = plan_path.stat(follow_symlinks=False)
        validated_metadata[plan_path.name] = (metadata.st_dev, metadata.st_ino)
    entries = {path.name: path for path in publication.recovery_directory.iterdir()}
    unexpected = set(entries) - allowed
    if unexpected:
        raise ValueError(
            f"unexpected recovery entry blocks cleanup: {sorted(unexpected)!r}"
        )
    if set(entries) - set(validated_metadata):
        raise ValueError("unvalidated recovery entry blocks cleanup")
    for name in (
        "completed.json",
        "completion.payload",
        "report.payload",
        "manifest.payload",
        "raw.payload",
        "plan.payload",
        "owner.json",
    ):
        path = entries.get(name)
        if path is None:
            continue
        current = path.stat(follow_symlinks=False)
        if (current.st_dev, current.st_ino) != validated_metadata[name]:
            raise ValueError("quality recovery entry changed during cleanup")
        path.unlink()
    _fsync_directory(publication.recovery_directory)
    publication.recovery_directory.rmdir()
    _fsync_directory(publication.recovery_directory.parent)


def _prepare_evidence_publication(
    recovery_directory: Path,
    destinations: _EvidenceDestinations,
    owner: dict[str, object],
) -> _EvidencePublication:
    _validate_publication_owner(owner, owner)
    for _name, destination in destinations.ordered():
        destination.parent.mkdir(parents=True, exist_ok=True)
    recovery_directory.parent.mkdir(parents=True, exist_ok=True)
    publication = _EvidencePublication(recovery_directory, destinations, owner)
    if recovery_directory.exists() or recovery_directory.is_symlink():
        if recovery_directory.is_symlink() or not recovery_directory.is_dir():
            raise FileExistsError("quality recovery path is not an owned directory")
        _validate_publication_owner(
            _read_publication_document(recovery_directory / "owner.json"), owner
        )
        plan_path = recovery_directory / "plan.payload"
        if plan_path.is_file():
            try:
                _load_staged_publication(publication)
            except (FileNotFoundError, ValueError):
                if any(
                    path.exists() or path.is_symlink()
                    for _name, path in destinations.ordered()
                ):
                    raise FileExistsError(
                        "quality artifacts exist with incomplete staged publication"
                    )
                _remove_owned_recovery(publication)
            else:
                return replace(publication, staged=True)
        else:
            if any(
                path.exists() or path.is_symlink()
                for _name, path in destinations.ordered()
            ):
                raise FileExistsError(
                    "quality artifacts exist without a durable publication plan"
                )
            _remove_owned_recovery(publication)
    os.mkdir(recovery_directory, 0o700)
    _fsync_directory(recovery_directory.parent)
    _durable_create(recovery_directory / "owner.json", canonical_json_bytes(owner))
    return publication


def _stage_named_payload(
    publication: _EvidencePublication, name: str, payload: bytes
) -> None:
    path = publication.recovery_directory / f"{name}.payload"
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise FileExistsError(f"conflicting staged quality {name} payload")
        return
    _durable_create(path, payload)


def _stage_evidence_publication(
    publication: _EvidencePublication, payloads: Mapping[str, bytes]
) -> _EvidencePublication:
    if set(payloads) != {"manifest", "raw", "report"}:
        raise ValueError("quality publication requires exactly three payloads")
    _validate_publication_owner(
        _read_publication_document(publication.recovery_directory / "owner.json"),
        publication.owner,
    )
    artifacts = {
        name: {
            "staged_name": f"{name}.payload",
            "byte_size": len(payloads[name]),
            "file_identity": _payload_identity(payloads[name]),
        }
        for name in ("raw", "manifest", "report")
    }
    plan = _publication_plan_document(publication.owner, artifacts)
    _stage_named_payload(publication, "plan", canonical_json_bytes(plan))
    for name in ("raw", "report", "manifest"):
        _stage_named_payload(publication, name, payloads[name])
    completion = _completion_document(publication.owner, plan)
    _stage_named_payload(publication, "completion", canonical_json_bytes(completion))
    return replace(publication, staged=True)


def _load_staged_publication(
    publication: _EvidencePublication,
) -> tuple[dict[str, object], dict[str, bytes]]:
    plan = _read_publication_document(publication.recovery_directory / "plan.payload")
    payloads = {
        name: _read_publication_payload(
            publication.recovery_directory / f"{name}.payload",
            label=f"staged {name}",
        )
        for name in ("raw", "manifest", "report")
    }
    artifacts = {
        name: {
            "staged_name": f"{name}.payload",
            "byte_size": len(payloads[name]),
            "file_identity": _payload_identity(payloads[name]),
        }
        for name in ("raw", "manifest", "report")
    }
    if plan != _publication_plan_document(publication.owner, artifacts):
        raise ValueError("staged quality publication plan does not match payloads")
    expected_completion = canonical_json_bytes(
        _completion_document(publication.owner, plan)
    )
    if (
        _read_publication_payload(
            publication.recovery_directory / "completion.payload",
            label="completion payload",
        )
        != expected_completion
    ):
        raise ValueError("staged quality completion payload changed")
    return plan, payloads


def _link_create_only_or_identical(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination, follow_symlinks=False)
    except FileExistsError as error:
        if (
            destination.is_symlink()
            or not destination.is_file()
            or destination.stat().st_size != source.stat().st_size
            or _payload_identity(destination.read_bytes())
            != _payload_identity(source.read_bytes())
        ):
            raise FileExistsError(
                f"conflicting quality evidence target: {destination}"
            ) from error


def _publish_staged_evidence(publication: _EvidencePublication) -> None:
    _validate_publication_owner(
        _read_publication_document(publication.recovery_directory / "owner.json"),
        publication.owner,
    )
    _plan, payloads = _load_staged_publication(publication)
    for name, destination in publication.destinations.ordered():
        source = publication.recovery_directory / f"{name}.payload"
        _link_create_only_or_identical(source, destination)
    for parent in sorted(
        {path.parent.resolve() for _name, path in publication.destinations.ordered()},
        key=lambda path: path.as_posix(),
    ):
        _fsync_directory(parent)
    for name, destination in publication.destinations.ordered():
        if destination.read_bytes() != payloads[name]:
            raise ValueError("published quality artifact changed before completion")
    _link_create_only_or_identical(
        publication.recovery_directory / "completion.payload",
        publication.recovery_directory / "completed.json",
    )
    _fsync_directory(publication.recovery_directory)
    _remove_owned_recovery(publication)


def _temporary_tree_bytes(path: Path) -> int:
    seen: set[tuple[int, int]] = set()
    total = 0
    for candidate in sorted(path.iterdir()):
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError("quality recovery directory contains an invalid entry")
        metadata = candidate.stat()
        inode = (metadata.st_dev, metadata.st_ino)
        if inode not in seen:
            seen.add(inode)
            total += metadata.st_size
    return total


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _preflight_output_paths(*paths: Path) -> None:
    if len(paths) != 3 or any(not isinstance(path, Path) for path in paths):
        raise TypeError("quality recording requires three output Paths")
    resolved = [path.resolve(strict=False) for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("quality output paths must be distinct")
    for path in paths:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"quality evidence already exists: {path}")


def _require_clean_recording_checkout(
    root: Path,
    destinations: _EvidenceDestinations,
    recovery_directory: Path,
) -> None:
    for arguments in (("diff", "--quiet"), ("diff", "--cached", "--quiet")):
        result = subprocess.run(
            ["git", *arguments], cwd=root, check=False, capture_output=True
        )
        if result.returncode != 0:
            raise RuntimeError("quality recording requires a clean tracked checkout")
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout.split(b"\0")
    allowed_files = {
        path.resolve().relative_to(root.resolve()).as_posix()
        for _name, path in destinations.ordered()
    }
    recovery_relative = recovery_directory.resolve().relative_to(root.resolve())
    unexpected = []
    for encoded in untracked:
        if not encoded:
            continue
        relative = encoded.decode("utf-8")
        candidate = Path(relative)
        if (
            relative in allowed_files
            or candidate == recovery_relative
            or (recovery_relative in candidate.parents)
        ):
            continue
        unexpected.append(relative)
    if unexpected:
        raise RuntimeError(
            f"quality recording checkout has unrelated untracked paths: {unexpected!r}"
        )


def _all_or_no_destinations(
    destinations: _EvidenceDestinations,
) -> tuple[bool, bool]:
    present = tuple(
        path.exists() or path.is_symlink() for _name, path in destinations.ordered()
    )
    return all(present), any(present)


def _record(args: argparse.Namespace) -> int:
    recording_started = time.monotonic()
    root = _root()
    destinations = _canonical_evidence_destinations(
        root,
        Path(args.manifest),
        Path(args.raw_output),
        Path(args.output),
    )
    recovery_directory = (root / RECOVERY_PATH).resolve()
    _require_clean_recording_checkout(root, destinations, recovery_directory)
    source_commit = _git(root, "rev-parse", "HEAD")
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise RuntimeError("quality recording requires a full Git source commit")
    mx.clear_cache()
    mx.reset_peak_memory()
    workload = build_pretraining_quality_workload(root)
    recording_command = _recording_command_document(root, destinations)
    session_identity = _recording_session_identity(
        source_commit, workload.identity, recording_command
    )
    owner = _publication_owner_document(
        session_identity=session_identity,
        source_commit=source_commit,
        workload_identity=workload.identity,
        destinations=destinations,
    )
    all_destinations, any_destination = _all_or_no_destinations(destinations)
    publication: _EvidencePublication | None = None
    if recovery_directory.exists() or recovery_directory.is_symlink():
        publication = _prepare_evidence_publication(
            recovery_directory, destinations, owner
        )
        if publication.staged:
            _publish_staged_evidence(publication)
            decision = _validate_evidence_files(
                root, destinations, recorded_workload=workload
            )
            print(decision)
            return 0 if decision == "pass" else 1
    elif all_destinations:
        decision = _validate_evidence_files(
            root, destinations, recorded_workload=workload
        )
        print(decision)
        return 0 if decision == "pass" else 1
    elif any_destination:
        raise FileExistsError(
            "partial quality evidence exists without an owned recovery directory"
        )
    if publication is None:
        publication = _prepare_evidence_publication(
            recovery_directory, destinations, owner
        )
    _preflight_output_paths(
        destinations.manifest, destinations.raw_output, destinations.report
    )
    training_rows = np.load(root / TRAINING_FIXTURE, allow_pickle=False, mmap_mode="r")
    validation_rows = np.load(
        root / VALIDATION_FIXTURE, allow_pickle=False, mmap_mode="r"
    )
    setup_elapsed = time.monotonic() - recording_started
    records = []
    runtime_times: dict[str, float] = {}
    for runtime in ("candidate", "oracle"):
        runtime_started = time.monotonic()
        runtime_records, _internal_elapsed = _run_runtime(
            root, workload, runtime, training_rows, validation_rows
        )
        records.extend(runtime_records)
        mx.clear_cache()
        runtime_times[runtime] = time.monotonic() - runtime_started
    validation_started = time.monotonic()
    report = validate_pretraining_quality_records(workload, records)
    raw_documents = [record.to_dict() for record in records]
    raw_identity = structured_identity("sml-pretraining-quality-raw-v1", raw_documents)
    raw_bytes = b"".join(
        canonical_json_bytes(document) + b"\n" for document in raw_documents
    )
    report_document = _report_document(workload.identity, raw_identity, report)
    report_bytes = canonical_json_bytes(report_document)
    validation_elapsed = time.monotonic() - validation_started
    phase_times = {
        "setup": setup_elapsed,
        "candidate": runtime_times["candidate"],
        "oracle": runtime_times["oracle"],
        "validation_serialization": validation_elapsed,
    }
    measured_wall_time = math.fsum(phase_times.values())
    if measured_wall_time > QUALITY_WALL_TIME_BUDGET_SECONDS:
        raise RuntimeError("pretraining quality workload exceeded its 12-hour budget")
    manifest = _manifest_document(
        workload=workload,
        source_commit=source_commit,
        recording_command=recording_command,
        phase_times=phase_times,
        peak_memory=int(mx.get_peak_memory()),
        raw_identity=raw_identity,
        raw_file_identity=_payload_identity(raw_bytes),
        raw_bytes=len(raw_bytes),
        report_identity=report_document["identity"],
        report_file_identity=_payload_identity(report_bytes),
        report_bytes=len(report_bytes),
        recording_session_identity=session_identity,
    )
    manifest_bytes = canonical_json_bytes(manifest)
    total_size = (
        workload.training_fixture.byte_size
        + workload.validation_fixture.byte_size
        + len(raw_bytes)
        + len(manifest_bytes)
        + len(report_bytes)
    )
    if total_size > 64 * 1024 * 1024:
        raise RuntimeError("quality fixtures and evidence exceed 64 MiB")
    publication = _stage_evidence_publication(
        publication,
        {"raw": raw_bytes, "manifest": manifest_bytes, "report": report_bytes},
    )
    measured_temporary_bytes = _temporary_tree_bytes(recovery_directory)
    if measured_temporary_bytes != manifest["temporary_disk_high_water_bytes"]:
        raise RuntimeError("quality temporary-disk high-water measurement changed")
    _publish_staged_evidence(publication)
    validated_decision = _validate_evidence_files(
        root, destinations, recorded_workload=workload
    )
    decision = decide_pretraining_quality(report)
    if decision != validated_decision:
        raise RuntimeError("published quality decision changed during validation")
    print(decision)
    return 0 if decision == "pass" else 1


def _json_object_no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path: Path) -> dict[str, object]:
    return _decode_json_object(path.read_bytes(), label=str(path))


def _decode_json_object(payload: bytes, *, label: str) -> dict[str, object]:
    value = json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=_json_object_no_duplicates,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"invalid JSON constant: {token}")
        ),
    )
    if not isinstance(value, dict) or canonical_json_bytes(value) != payload:
        raise ValueError(f"quality JSON is not a canonical object: {label}")
    return value


def _validate_manifest_fields(
    raw: Mapping[str, object],
    expected_workload: PretrainingQualityWorkload,
    expected_command: Mapping[str, object],
) -> dict[str, object]:
    expected_fields = {
        "kind",
        "version",
        "identity",
        "source_commit",
        "harness_commit",
        "harness_clean",
        "harness_identity",
        "production_dependency_components",
        "production_dependency_identity",
        "workload",
        "workload_identity",
        "recording_command",
        "measurement_boundaries",
        "phase_wall_time_seconds",
        "measured_wall_time_seconds",
        "wall_time_budget_seconds",
        "peak_metal_memory_bytes",
        "publication_strategy",
        "recording_session_identity",
        "temporary_disk_high_water_bytes",
        "artifact_byte_sizes",
        "publication_metadata_bytes",
        "fixture_bytes",
        "training_cardinality",
        "validation_cardinality",
        "ordered_work_count",
        "record_count",
        "raw_identity",
        "raw_file_identity",
        "report_identity",
        "report_file_identity",
    }
    if set(raw) != expected_fields:
        raise ValueError("pretraining quality manifest has an invalid field set")
    if raw["kind"] != "pretraining-quality-manifest" or raw["version"] != 2:
        raise ValueError("unsupported pretraining quality manifest")
    body = {key: value for key, value in raw.items() if key != "identity"}
    if raw["identity"] != structured_identity(
        "sml-pretraining-quality-manifest-v3", body
    ):
        raise ValueError("pretraining quality manifest identity mismatch")
    workload_raw = raw["workload"]
    if not isinstance(workload_raw, dict):
        raise ValueError("pretraining quality manifest workload must be an object")
    workload = PretrainingQualityWorkload.from_dict(workload_raw)
    if workload != expected_workload or raw["workload_identity"] != workload.identity:
        raise ValueError("pretraining quality manifest workload changed")
    if raw["harness_identity"] != expected_workload.harness_identity:
        raise ValueError("pretraining quality manifest harness changed")
    if raw["production_dependency_components"] != list(
        expected_workload.production_dependency_components
    ):
        raise ValueError("pretraining quality production dependency order changed")
    if raw["production_dependency_identity"] != (
        expected_workload.production_dependency_identity
    ):
        raise ValueError("pretraining quality production dependencies changed")
    if raw["harness_clean"] is not True:
        raise ValueError("pretraining quality manifest requires a clean harness")
    if raw["source_commit"] != raw["harness_commit"]:
        raise ValueError("quality source and harness commits must match")
    source_commit = raw["source_commit"]
    if (
        not isinstance(source_commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None
    ):
        raise ValueError("quality source commit must be a full Git commit")
    command = raw["recording_command"]
    if not isinstance(command, dict):
        raise ValueError("quality recording command must be an object")
    _destinations_from_recording_command(command)
    if command != expected_command:
        raise ValueError("quality recording command or canonical destinations changed")
    if raw["measurement_boundaries"] != MEASUREMENT_BOUNDARIES:
        raise ValueError("quality measurement boundaries changed")
    phases = raw["phase_wall_time_seconds"]
    if not isinstance(phases, dict) or set(phases) != set(_PHASE_NAMES):
        raise ValueError("quality phase timing fields changed")
    normalized_phases = {
        name: _require_finite(phases[name], f"quality {name} phase")
        for name in _PHASE_NAMES
    }
    if any(value <= 0.0 for value in normalized_phases.values()):
        raise ValueError("quality phase timings must be positive")
    measured_wall_time = _require_finite(
        raw["measured_wall_time_seconds"], "quality measured wall time"
    )
    if (
        measured_wall_time != math.fsum(normalized_phases.values())
        or measured_wall_time > QUALITY_WALL_TIME_BUDGET_SECONDS
    ):
        raise ValueError("quality measured wall time is inconsistent")
    if raw["wall_time_budget_seconds"] != QUALITY_WALL_TIME_BUDGET_SECONDS:
        raise ValueError("quality manifest wall-time budget changed")
    if (
        type(raw["peak_metal_memory_bytes"]) is not int
        or raw["peak_metal_memory_bytes"] < 0
    ):
        raise ValueError("quality peak Metal memory must be nonnegative")
    if raw["publication_strategy"] != _PUBLICATION_STRATEGY:
        raise ValueError("quality publication strategy changed")
    session_identity = _require_identity(
        raw["recording_session_identity"], "quality recording session identity"
    )
    if session_identity != _recording_session_identity(
        source_commit, workload.identity, command
    ):
        raise ValueError("quality recording session identity is not derived")
    artifact_sizes = raw["artifact_byte_sizes"]
    if not isinstance(artifact_sizes, dict) or set(artifact_sizes) != {
        "raw",
        "manifest",
        "report",
    }:
        raise ValueError("quality artifact byte sizes are incomplete")
    normalized_artifact_sizes = {
        name: _require_plain_int(
            artifact_sizes[name], f"quality {name} byte size", minimum=1
        )
        for name in ("raw", "manifest", "report")
    }
    if normalized_artifact_sizes["manifest"] != len(canonical_json_bytes(dict(raw))):
        raise ValueError("quality manifest byte size is not self-consistent")
    destinations = _destinations_from_recording_command(command)
    owner = _publication_owner_document(
        session_identity=session_identity,
        source_commit=source_commit,
        workload_identity=workload.identity,
        destinations=destinations,
    )
    metadata_size = _require_plain_int(
        raw["publication_metadata_bytes"],
        "quality publication metadata bytes",
        minimum=1,
    )
    if metadata_size != _publication_metadata_size(owner, normalized_artifact_sizes):
        raise ValueError("quality publication metadata byte size changed")
    temporary_high_water = _require_plain_int(
        raw["temporary_disk_high_water_bytes"],
        "quality temporary disk high-water bytes",
        minimum=1,
    )
    if temporary_high_water != sum(normalized_artifact_sizes.values()) + metadata_size:
        raise ValueError("quality temporary disk high-water measurement changed")
    expected_fixture_bytes = (
        expected_workload.training_fixture.byte_size
        + expected_workload.validation_fixture.byte_size
    )
    if raw["fixture_bytes"] != expected_fixture_bytes:
        raise ValueError("quality manifest fixture bytes changed")
    if raw["training_cardinality"] != expected_workload.training_fixture.shape[0]:
        raise ValueError("quality manifest training cardinality changed")
    if raw["validation_cardinality"] != expected_workload.validation_fixture.shape[0]:
        raise ValueError("quality manifest validation cardinality changed")
    if raw["ordered_work_count"] != len(expected_workload.ordered_batches):
        raise ValueError("quality manifest ordered work count changed")
    if raw["record_count"] != 8:
        raise ValueError("quality manifest record count must be eight")
    for name in (
        "raw_identity",
        "raw_file_identity",
        "report_identity",
        "report_file_identity",
    ):
        _require_identity(raw[name], f"quality {name}")
    return dict(raw)


def _git_bytes(root: Path, *arguments: str) -> bytes:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout


def _git_production_dependency_components(root: Path, commit: str) -> tuple[Path, ...]:
    scopes = (
        PRODUCTION_SOURCE_TREE,
        *PRODUCTION_DEPENDENCY_FIXED_COMPONENTS,
        *PRODUCTION_IMPORT_ENTRYPOINTS,
    )
    entries = _git_bytes(
        root,
        "ls-tree",
        "-r",
        "-z",
        commit,
        "--",
        *(path.as_posix() for path in scopes),
    ).split(b"\0")
    available: set[Path] = set()
    modes: dict[Path, tuple[bytes, bytes]] = {}
    for entry in entries:
        if not entry:
            continue
        metadata, encoded_path = entry.split(b"\t", 1)
        mode, object_type, _object_identity = metadata.split(b" ", 2)
        path = Path(encoded_path.decode("utf-8"))
        is_source = path.suffix == ".py" and path.is_relative_to(PRODUCTION_SOURCE_TREE)
        if (
            not is_source
            and path not in PRODUCTION_DEPENDENCY_FIXED_COMPONENTS
            and path not in PRODUCTION_IMPORT_ENTRYPOINTS
        ):
            continue
        available.add(path)
        modes[path] = (mode, object_type)
    components = _production_dependency_closure(
        available,
        lambda component: _git_bytes(root, "show", f"{commit}:{component.as_posix()}"),
    )
    if any(modes[component] != (b"100644", b"blob") for component in components):
        raise ValueError("recorded production import closure contains a non-file")
    return components


def _validate_harness_commit(
    root: Path,
    commit: str,
    workload: PretrainingQualityWorkload,
) -> None:
    try:
        subprocess.run(
            ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
            cwd=root,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
        )
        digest = hashlib.sha256()
        for component in HARNESS_COMPONENTS:
            digest.update(_git_bytes(root, "show", f"{commit}:{component.as_posix()}"))
        if f"sha256:{digest.hexdigest()}" != workload.harness_identity:
            raise ValueError("recorded harness commit does not contain harness bytes")
        production_components = _git_production_dependency_components(root, commit)
        if tuple(path.as_posix() for path in production_components) != (
            workload.production_dependency_components
        ):
            raise ValueError("recorded production import closure component set changed")
        production_identity = _production_dependency_identity(
            production_components,
            lambda component: _git_bytes(
                root, "show", f"{commit}:{component.as_posix()}"
            ),
        )
        if production_identity != workload.production_dependency_identity:
            raise ValueError(
                "recorded harness commit does not contain production import closure bytes"
            )
        for fixture in (workload.training_fixture, workload.validation_fixture):
            payload = _git_bytes(root, "show", f"{commit}:{fixture.logical_path}")
            identity = f"sha256:{hashlib.sha256(payload).hexdigest()}"
            if identity != fixture.file_identity:
                raise ValueError(
                    "recorded harness commit does not contain fixture bytes"
                )
    except subprocess.CalledProcessError as error:
        raise ValueError(
            "recorded harness commit is unavailable or not an ancestor"
        ) from error


def _validate_manifest(
    raw: Mapping[str, object],
    expected_workload: PretrainingQualityWorkload,
    root: Path,
    expected_command: Mapping[str, object],
) -> dict[str, object]:
    manifest = _validate_manifest_fields(raw, expected_workload, expected_command)
    _validate_harness_commit(root, manifest["harness_commit"], expected_workload)
    return manifest


def _read_raw(path: Path) -> tuple[PretrainingQualityCheckpoint, ...]:
    payload = path.read_bytes()
    if not payload.endswith(b"\n"):
        raise ValueError("quality raw JSONL must end with a newline")
    records = []
    framed = []
    for line in payload.splitlines():
        raw = json.loads(
            line.decode("utf-8"),
            object_pairs_hook=_json_object_no_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant: {token}")
            ),
        )
        if not isinstance(raw, dict) or canonical_json_bytes(raw) != line:
            raise ValueError("quality raw record is not a canonical JSON object")
        records.append(PretrainingQualityCheckpoint.from_dict(raw))
        framed.append(line + b"\n")
    if b"".join(framed) != payload:
        raise ValueError("quality raw JSONL has invalid framing")
    return tuple(records)


def _validate_evidence_files(
    root: Path,
    destinations: _EvidenceDestinations,
    *,
    recorded_workload: PretrainingQualityWorkload | None = None,
) -> Literal["pass", "fail"]:
    for _name, path in destinations.ordered():
        if path.is_symlink() or not path.is_file():
            raise ValueError("quality evidence paths must be regular files")
    expected_command = _recording_command_document(root, destinations)
    manifest_payload = destinations.manifest.read_bytes()
    raw_manifest = _decode_json_object(
        manifest_payload, label=str(destinations.manifest)
    )
    workload_raw = raw_manifest.get("workload")
    if not isinstance(workload_raw, dict):
        raise ValueError("pretraining quality manifest workload must be an object")
    workload = PretrainingQualityWorkload.from_dict(workload_raw)
    if recorded_workload is not None and recorded_workload != workload:
        raise ValueError("pretraining quality manifest workload changed")
    manifest = _validate_manifest(
        raw_manifest,
        workload,
        root,
        expected_command,
    )
    artifact_sizes = manifest["artifact_byte_sizes"]
    if len(manifest_payload) != artifact_sizes["manifest"]:
        raise ValueError("quality manifest byte size changed")
    raw_payload = destinations.raw_output.read_bytes()
    if (
        len(raw_payload) != artifact_sizes["raw"]
        or _payload_identity(raw_payload) != manifest["raw_file_identity"]
    ):
        raise ValueError("pretraining quality raw file identity mismatch")
    records = _read_raw(destinations.raw_output)
    raw_identity = structured_identity(
        "sml-pretraining-quality-raw-v1",
        [record.to_dict() for record in records],
    )
    if raw_identity != manifest["raw_identity"]:
        raise ValueError("pretraining quality raw identity mismatch")
    report = validate_pretraining_quality_records(workload, records)
    expected_report = _report_document(workload.identity, raw_identity, report)
    report_payload = destinations.report.read_bytes()
    if (
        len(report_payload) != artifact_sizes["report"]
        or _payload_identity(report_payload) != manifest["report_file_identity"]
        or expected_report["identity"] != manifest["report_identity"]
        or _decode_json_object(report_payload, label=str(destinations.report))
        != expected_report
    ):
        raise ValueError("pretraining quality report does not match raw evidence")
    total_size = (
        workload.training_fixture.byte_size
        + workload.validation_fixture.byte_size
        + len(manifest_payload)
        + len(raw_payload)
        + len(report_payload)
    )
    if total_size > 64 * 1024 * 1024:
        raise ValueError("quality fixtures and evidence exceed 64 MiB")
    return decide_pretraining_quality(report)


def _validate(args: argparse.Namespace) -> int:
    root = _root()
    destinations = _canonical_evidence_destinations(
        root,
        Path(args.manifest),
        Path(args.raw_input),
        Path(args.report),
    )
    decision = _validate_evidence_files(root, destinations)
    print(decision)
    return 0 if decision == "pass" else 1


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return args.handler(args)


__all__ = (
    "CANONICAL_STEPS",
    "CHECKPOINT_STEPS",
    "ParameterLeafSpec",
    "ParameterUpdateStatistics",
    "PretrainingQualityCheckpoint",
    "PretrainingQualityReport",
    "PretrainingQualityWorkload",
    "build_pretraining_quality_workload",
    "decide_pretraining_quality",
    "harness_content_identity",
    "production_dependency_components",
    "production_dependency_content_identity",
    "real_work_identity",
    "validate_pretraining_quality_records",
)


if __name__ == "__main__":
    raise SystemExit(main())
