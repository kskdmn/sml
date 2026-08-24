from __future__ import annotations

# Serialized quality schemas report malformed external content uniformly as ValueError.
# ruff: noqa: TRY004
import argparse
import ast
import dataclasses
import hashlib
import io
import json
import math
import os
import re
import subprocess
import sys
import time
import unicodedata
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_map
from sml.data.swag import SwagBatch, SwagCursor
from sml.model.config import ModelConfig
from sml.model.language_model import SMLLanguageModel
from sml.training.common import (
    LoaderConfig,
    build_weight_decay_tree,
    initialize_adam_state,
)
from sml.training.lora import LoRAConfig, apply_lora, split_adapter_parameters
from sml.training.swag import (
    SwagTrainingConfig,
    build_swag_kernels,
    default_swag_optimizer_config,
    initial_swag_trainer_state,
)

from v2.benchmarks.workload import (
    BENCHMARK_CORPUS,
    canonical_json_bytes,
    file_identity,
    semantic_array_identity,
    structured_identity,
)

CANONICAL_STEPS = 256
QUALITY_WALL_TIME_BUDGET_SECONDS = 4 * 60 * 60
HARNESS_COMPONENTS = (
    Path("v2/benchmarks/swag_quality.py"),
    Path("v2/tests/unit/test_swag_quality.py"),
)
PRODUCTION_SOURCE_TREE = Path("v2/src/sml")
PRODUCTION_MODULE_ROOT = Path("v2/src")
PRODUCTION_DEPENDENCY_FIXED_COMPONENTS = (
    Path("v2/src/sml.py"),
    Path("v2/benchmarks/schema.py"),
    Path("v2/benchmarks/workload.py"),
)
PRODUCTION_IMPORT_ENTRYPOINTS = (HARNESS_COMPONENTS[0],)
TRAINING_FIXTURE = Path("v2/benchmarks/fixtures/swag-quality-train-v1.npz")
VALIDATION_FIXTURE = Path("v2/benchmarks/fixtures/swag-quality-validation-v1.npz")
CANONICAL_MANIFEST_PATH = Path("v2/benchmarks/manifests/swag-quality-v1.json")
CANONICAL_RAW_PATH = Path("v2/benchmarks/results/swag-quality-v1.jsonl")
CANONICAL_REPORT_PATH = Path("v2/benchmarks/results/swag-quality-v1.json")
TRAINING_EXAMPLE_COUNT = 255
VALIDATION_EXAMPLE_COUNT = 16
SEQUENCE_LENGTH = 64
MICROBATCH_SIZE = 2
SCORE_POLICY = "fp32-mean-continuation-including-eos-v1"
MODEL_SEED = 42
VALIDATION_SEED = 7
ARRAY_NAMES = ("input_ids", "valid_token_mask", "score_mask", "labels")
ENDINGS = ("on the mat", "in the car", "by the door", "near a tree")
MEASUREMENT_BOUNDARIES = {
    "clock": "time.monotonic",
    "start": "record-handler-entry-before-path-worktree-and-workload-setup",
    "manifest_freeze": (
        "after-setup-both-runtime-setup-compilation-execution-validation-"
        "and-non-manifest-serialization"
    ),
    "excluded_after_manifest_freeze": [
        "manifest-self-serialization-and-durable-create-only-writes",
        "parent-directory-fsyncs",
    ],
}
_PHASE_NAMES = (
    "setup",
    "candidate",
    "oracle",
    "validation_serialization",
)
_IDENTITY_RE = re.compile(r"sha256:[0-9a-f]{64}")
_PLACEHOLDER_IDENTITY = "sha256:" + "0" * 64
_INT32 = np.dtype("<i4")
_BOOL = np.dtype("?")

RuntimeName = Literal["candidate", "oracle"]


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


def _require_finite(value: object, name: str, *, minimum: float | None = None) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and normalized < minimum:
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


def _production_dependency_identity(
    components: Sequence[Path],
    read_component: Callable[[Path], bytes],
) -> str:
    digest = hashlib.sha256()
    digest.update(b"sml-swag-quality-production-import-closure-v1\0")
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
    return _production_dependency_identity(
        components,
        lambda relative_path: _read_current_production_component(root, relative_path),
    )


def production_dependency_content_identity(root: Path) -> str:
    if not isinstance(root, Path):
        raise TypeError("root must be a Path")
    components = production_dependency_components(root)
    return _current_production_dependency_identity(root, components)


@dataclass(frozen=True, slots=True)
class SwagQualityFixture:
    logical_path: str
    example_count: int
    candidate_count: int
    sequence_length: int
    byte_size: int
    file_identity: str
    semantic_identity: str
    source_identity: str

    def to_dict(self) -> dict[str, object]:
        return {
            "logical_path": self.logical_path,
            "example_count": self.example_count,
            "candidate_count": self.candidate_count,
            "sequence_length": self.sequence_length,
            "byte_size": self.byte_size,
            "file_identity": self.file_identity,
            "semantic_identity": self.semantic_identity,
            "source_identity": self.source_identity,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> SwagQualityFixture:
        expected = {
            "logical_path",
            "example_count",
            "candidate_count",
            "sequence_length",
            "byte_size",
            "file_identity",
            "semantic_identity",
            "source_identity",
        }
        if set(raw) != expected:
            raise ValueError("swag quality fixture has an invalid field set")
        logical_path = raw["logical_path"]
        if not isinstance(logical_path, str) or not logical_path:
            raise ValueError("quality fixture logical path must be non-empty")
        return cls(
            logical_path=logical_path,
            example_count=_require_plain_int(
                raw["example_count"], "fixture example count", minimum=1
            ),
            candidate_count=_require_plain_int(
                raw["candidate_count"], "fixture candidate count", minimum=4
            ),
            sequence_length=_require_plain_int(
                raw["sequence_length"], "fixture sequence length", minimum=2
            ),
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
class SwagQualityWorkload:
    kind: Literal["swag-quality-workload"]
    version: Literal[1]
    identity: str
    training_fixture: SwagQualityFixture
    validation_fixture: SwagQualityFixture
    fp32_master_identity: str
    frozen_bf16_base_identity: str
    initial_fp32_adapter_identity: str
    model: dict[str, object]
    lora: dict[str, object]
    optimizer: dict[str, object]
    loader: dict[str, object]
    score_policy: str
    ordered_batches: tuple[tuple[int, ...], ...]
    optimizer_steps: int
    model_seed: int
    validation_seed: int
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
            "fp32_master_identity": self.fp32_master_identity,
            "frozen_bf16_base_identity": self.frozen_bf16_base_identity,
            "initial_fp32_adapter_identity": self.initial_fp32_adapter_identity,
            "model": self.model,
            "lora": self.lora,
            "optimizer": self.optimizer,
            "loader": self.loader,
            "score_policy": self.score_policy,
            "ordered_batches": [list(batch) for batch in self.ordered_batches],
            "optimizer_steps": self.optimizer_steps,
            "model_seed": self.model_seed,
            "validation_seed": self.validation_seed,
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
        return structured_identity("sml-swag-quality-workload-v1", self._body())

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> SwagQualityWorkload:
        expected = {field.name for field in dataclasses.fields(cls)}
        if set(raw) != expected:
            raise ValueError("swag quality workload has an invalid field set")
        if raw["kind"] != "swag-quality-workload" or raw["version"] != 1:
            raise ValueError("unsupported swag quality workload")
        training_raw = raw["training_fixture"]
        validation_raw = raw["validation_fixture"]
        ordered = raw["ordered_batches"]
        if not isinstance(training_raw, dict) or not isinstance(validation_raw, dict):
            raise ValueError("swag quality fixtures must be objects")
        if not isinstance(ordered, list) or any(
            not isinstance(batch, list) or not batch for batch in ordered
        ):
            raise ValueError("ordered batches must be a nonempty list of index lists")
        components = raw["harness_components"]
        production = raw["production_dependency_components"]
        if not isinstance(components, list) or not isinstance(production, list):
            raise ValueError("quality component lists must be arrays")
        model = raw["model"]
        lora = raw["lora"]
        optimizer = raw["optimizer"]
        loader = raw["loader"]
        if not all(
            isinstance(value, dict) for value in (model, lora, optimizer, loader)
        ):
            raise ValueError("quality configuration mappings must be objects")
        workload = cls(
            kind="swag-quality-workload",
            version=1,
            identity=_require_identity(raw["identity"], "workload identity"),
            training_fixture=SwagQualityFixture.from_dict(training_raw),
            validation_fixture=SwagQualityFixture.from_dict(validation_raw),
            fp32_master_identity=_require_identity(
                raw["fp32_master_identity"], "fp32 master identity"
            ),
            frozen_bf16_base_identity=_require_identity(
                raw["frozen_bf16_base_identity"], "frozen bf16 base identity"
            ),
            initial_fp32_adapter_identity=_require_identity(
                raw["initial_fp32_adapter_identity"], "initial adapter identity"
            ),
            model=dict(model),
            lora=dict(lora),
            optimizer=dict(optimizer),
            loader=dict(loader),
            score_policy=_require_string(raw["score_policy"], "score policy"),
            ordered_batches=tuple(
                tuple(_require_plain_int(index, "batch index") for index in batch)
                for batch in ordered
            ),
            optimizer_steps=_require_plain_int(
                raw["optimizer_steps"], "optimizer steps", minimum=1
            ),
            model_seed=_require_plain_int(raw["model_seed"], "model seed"),
            validation_seed=_require_plain_int(
                raw["validation_seed"], "validation seed"
            ),
            loader_seed=_require_plain_int(raw["loader_seed"], "loader seed"),
            harness_components=tuple(
                _require_string(item, "harness component") for item in components
            ),
            harness_identity=_require_identity(
                raw["harness_identity"], "harness identity"
            ),
            production_dependency_components=tuple(
                _require_string(item, "production component") for item in production
            ),
            production_dependency_identity=_require_identity(
                raw["production_dependency_identity"],
                "production dependency identity",
            ),
        )
        if workload.identity != workload.recompute_identity():
            raise ValueError("swag quality workload identity mismatch")
        return workload


@dataclass(frozen=True, slots=True)
class SwagQualityRecord:
    kind: Literal["swag-quality-record"]
    version: Literal[1]
    identity: str
    runtime: RuntimeName
    step: int
    workload_identity: str
    train_loss: float
    validation_loss: float
    validation_accuracy: float
    real_example_count: int
    finite: bool
    frozen_base_identity: str
    adapter_identity: str

    def _body(self) -> dict[str, object]:
        return {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
            if field.name != "identity"
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._body(), "identity": self.identity}

    def recompute_identity(self) -> str:
        return structured_identity("sml-swag-quality-record-v1", self._body())

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> SwagQualityRecord:
        if set(raw) != {field.name for field in dataclasses.fields(cls)}:
            raise ValueError("swag quality record has an invalid field set")
        if raw["kind"] != "swag-quality-record" or raw["version"] != 1:
            raise ValueError("unsupported swag quality record")
        runtime = raw["runtime"]
        if runtime not in ("candidate", "oracle"):
            raise ValueError("swag quality record runtime is invalid")
        record = cls(
            kind="swag-quality-record",
            version=1,
            identity=_require_identity(raw["identity"], "record identity"),
            runtime=runtime,
            step=_require_plain_int(raw["step"], "record step"),
            workload_identity=_require_identity(
                raw["workload_identity"], "record workload identity"
            ),
            train_loss=_require_finite(raw["train_loss"], "train loss"),
            validation_loss=_require_finite(raw["validation_loss"], "validation loss"),
            validation_accuracy=_require_finite(
                raw["validation_accuracy"], "validation accuracy"
            ),
            real_example_count=_require_plain_int(
                raw["real_example_count"], "real example count"
            ),
            finite=_require_bool(raw["finite"], "finite"),
            frozen_base_identity=_require_identity(
                raw["frozen_base_identity"], "frozen base identity"
            ),
            adapter_identity=_require_identity(
                raw["adapter_identity"], "adapter identity"
            ),
        )
        if record.identity != record.recompute_identity():
            raise ValueError("swag quality record identity mismatch")
        return record


@dataclass(frozen=True, slots=True)
class SwagQualityReport:
    candidate_validation_loss: float
    oracle_validation_loss: float
    candidate_accuracy: float
    oracle_accuracy: float
    candidate_examples: int
    oracle_examples: int
    candidate_finite: bool
    oracle_finite: bool


def decide_swag_quality(report: SwagQualityReport) -> Literal["pass", "fail"]:
    if not isinstance(report, SwagQualityReport):
        raise TypeError("report must be a SwagQualityReport")
    if not (
        report.candidate_finite
        and report.oracle_finite
        and report.candidate_examples == report.oracle_examples
        and math.isfinite(report.candidate_validation_loss)
        and math.isfinite(report.oracle_validation_loss)
        and math.isfinite(report.candidate_accuracy)
        and math.isfinite(report.oracle_accuracy)
        and abs(report.candidate_validation_loss - report.oracle_validation_loss)
        / max(abs(report.oracle_validation_loss), 1e-12)
        <= 0.01
        and abs(report.candidate_accuracy - report.oracle_accuracy) <= 0.01
    ):
        return "fail"
    return "pass"


def _to_json_value(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _to_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_value(item) for item in value]
    return value


def _json_mapping(value: object) -> dict[str, object]:
    loaded = json.loads(canonical_json_bytes(_to_json_value(value)).decode("utf-8"))
    if not isinstance(loaded, dict):
        raise TypeError("canonical mapping must decode as an object")
    return loaded


def _require_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _fixture_source_identity(split: str) -> str:
    return structured_identity(
        "sml-swag-quality-source-v1",
        {
            "split": split,
            "generator": "local-fake-swag-encoder-v1",
            "endings": list(ENDINGS),
        },
    )


def _encode_text(text: str, vocab_size: int) -> tuple[int, ...]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    pieces = re.findall(r"\w+|[^\w\s]", normalized, flags=re.UNICODE)
    if not pieces:
        return (4,)
    ids = []
    for piece in pieces:
        digest = hashlib.sha256(piece.encode("utf-8")).digest()
        ids.append(4 + int.from_bytes(digest[:4], "little") % (vocab_size - 4))
    return tuple(ids)


def _pad_candidate(
    token_ids: tuple[int, ...],
    continuation: tuple[int, int],
    *,
    length: int,
    pad_token_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    input_ids = np.full(length, pad_token_id, dtype=_INT32)
    valid_token_mask = np.zeros(length, dtype=_BOOL)
    score_mask = np.zeros(length, dtype=_BOOL)
    token_length = len(token_ids)
    input_ids[:token_length] = np.asarray(token_ids, dtype=_INT32)
    valid_token_mask[:token_length] = True
    start, end = continuation
    if end > start:
        score_mask[start:end] = True
    score_mask[0] = False
    if not bool(score_mask.any()):
        raise ValueError("encoded SWAG candidate is missing scored tokens")
    return input_ids, valid_token_mask, score_mask


def _fit_candidate(
    context_ids: tuple[int, ...],
    ending_ids: tuple[int, ...],
    *,
    bos_token_id: int,
    eos_token_id: int,
    length: int,
) -> tuple[tuple[int, ...], tuple[int, int]]:
    ending = ending_ids[: max(1, length - 3)]
    budget = max(0, length - 2 - len(ending))
    context = context_ids[:budget]
    token_ids = (bos_token_id, *context, *ending, eos_token_id)
    if len(token_ids) > length:
        raise ValueError("encoded SWAG example exceeds the pinned length")
    continuation_start = 1 + len(context)
    return token_ids, (continuation_start, len(token_ids))


def _encode_example(
    context: str,
    label: int,
    *,
    model_config: ModelConfig,
    length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.int32]:
    context_ids = _encode_text(context, model_config.vocab_size)
    encoded_endings = [
        _encode_text(ending, model_config.vocab_size) or (4,) for ending in ENDINGS
    ]
    candidates = [
        _fit_candidate(
            context_ids,
            ending_ids,
            bos_token_id=model_config.bos_token_id,
            eos_token_id=model_config.eos_token_id,
            length=length,
        )
        for ending_ids in encoded_endings
    ]
    input_ids = np.empty((4, length), dtype=_INT32)
    valid_token_mask = np.empty((4, length), dtype=_BOOL)
    score_mask = np.empty((4, length), dtype=_BOOL)
    for index, (token_ids, continuation) in enumerate(candidates):
        padded = _pad_candidate(
            token_ids,
            continuation,
            length=length,
            pad_token_id=model_config.pad_token_id,
        )
        input_ids[index], valid_token_mask[index], score_mask[index] = padded
    return input_ids, valid_token_mask, score_mask, np.int32(label)


def _encode_split(
    *,
    split: str,
    count: int,
    model_config: ModelConfig,
    length: int,
) -> dict[str, np.ndarray]:
    prefix = "Train scenario" if split == "training" else "Validation scene"
    input_ids = np.empty((count, 4, length), dtype=_INT32)
    valid_token_mask = np.empty((count, 4, length), dtype=_BOOL)
    score_mask = np.empty((count, 4, length), dtype=_BOOL)
    labels = np.empty((count,), dtype=_INT32)
    for index in range(count):
        corpus = BENCHMARK_CORPUS[index % len(BENCHMARK_CORPUS)]
        context = (
            f"{prefix} {index}: {corpus} A local tokenizer owned by this "
            "harness encodes the row without downloading SWAG."
        )
        label = hashlib.sha256(f"{split}:{index}:{corpus}".encode()).digest()[0] % 4
        encoded = _encode_example(
            context,
            label,
            model_config=model_config,
            length=length,
        )
        input_ids[index], valid_token_mask[index], score_mask[index], labels[index] = (
            encoded
        )
    return {
        "input_ids": np.ascontiguousarray(input_ids, dtype=_INT32),
        "valid_token_mask": np.ascontiguousarray(valid_token_mask, dtype=_BOOL),
        "score_mask": np.ascontiguousarray(score_mask, dtype=_BOOL),
        "labels": np.ascontiguousarray(labels, dtype=_INT32),
    }


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        path, "w", compression=zipfile.ZIP_STORED, allowZip64=False
    ) as archive:
        for name in ARRAY_NAMES:
            buffer = io.BytesIO()
            np.save(buffer, arrays[name], allow_pickle=False)
            info = zipfile.ZipInfo(
                filename=f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0)
            )
            archive.writestr(info, buffer.getvalue())


def write_canonical_fixtures(root: Path) -> None:
    model_config = ModelConfig()
    for relative, split, count in (
        (TRAINING_FIXTURE, "training", TRAINING_EXAMPLE_COUNT),
        (VALIDATION_FIXTURE, "validation", VALIDATION_EXAMPLE_COUNT),
    ):
        _write_npz(
            root / relative,
            _encode_split(
                split=split,
                count=count,
                model_config=model_config,
                length=SEQUENCE_LENGTH,
            ),
        )


def _load_encoded_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        arrays = {name: np.ascontiguousarray(loaded[name]) for name in ARRAY_NAMES}
    if arrays["input_ids"].dtype != _INT32 or arrays["labels"].dtype != _INT32:
        raise ValueError(f"swag quality fixture token arrays must be <i4: {path}")
    if arrays["valid_token_mask"].dtype != _BOOL or arrays["score_mask"].dtype != _BOOL:
        raise ValueError(f"swag quality fixture masks must be bool: {path}")
    return arrays


def _example_bytes(arrays: Mapping[str, np.ndarray], index: int) -> bytes:
    return b"".join(
        np.ascontiguousarray(arrays[name][index]).tobytes(order="C")
        for name in ARRAY_NAMES
    )


def _require_source_disjoint(
    training: Mapping[str, np.ndarray], validation: Mapping[str, np.ndarray]
) -> None:
    training_rows = {
        _example_bytes(training, index)
        for index in range(training["input_ids"].shape[0])
    }
    for index in range(validation["input_ids"].shape[0]):
        if _example_bytes(validation, index) in training_rows:
            raise ValueError("swag quality splits must be source-disjoint")


def _load_fixture(
    root: Path,
    relative_path: Path,
    *,
    split: str,
    example_count: int,
) -> tuple[dict[str, np.ndarray], SwagQualityFixture]:
    path = root / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"missing quality fixture: {path}")
    arrays = _load_encoded_arrays(path)
    shape = arrays["input_ids"].shape
    if shape != (example_count, 4, SEQUENCE_LENGTH):
        raise ValueError(
            f"quality {split} fixture must have shape "
            f"{(example_count, 4, SEQUENCE_LENGTH)}"
        )
    for name in ("valid_token_mask", "score_mask"):
        if arrays[name].shape != shape:
            raise ValueError(f"quality {split} {name} shape mismatch")
    if arrays["labels"].shape != (example_count,):
        raise ValueError(f"quality {split} labels shape mismatch")
    if int(arrays["input_ids"].min()) < 0 or int(arrays["input_ids"].max()) >= (
        ModelConfig().vocab_size
    ):
        raise ValueError(f"quality {split} fixture has out-of-vocabulary tokens")
    return arrays, SwagQualityFixture(
        logical_path=relative_path.as_posix(),
        example_count=example_count,
        candidate_count=4,
        sequence_length=SEQUENCE_LENGTH,
        byte_size=path.stat().st_size,
        file_identity=file_identity(path),
        semantic_identity=semantic_array_identity(
            "sml-swag-quality-encoded-v1", arrays
        ),
        source_identity=_fixture_source_identity(split),
    )


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
    digest.update(b"sml-swag-quality-parameter-leaf-v1\0")
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


def _array_tree_identity(domain: str, tree: object) -> str:
    leaves = sorted(tree_flatten(tree))
    if not leaves:
        raise ValueError("quality array tree must not be empty")
    mx.eval(*(array for _path, array in leaves))
    identities = []
    for path, array in leaves:
        if not isinstance(array, mx.array):
            raise ValueError("quality tree leaves must be MLX arrays")
        identities.append(
            {
                "path": path,
                "shape": list(array.shape),
                "identity": _array_leaf_identity(path, array),
            }
        )
    return structured_identity(domain, identities)


def _clone_tree(tree: object) -> object:
    return tree_map(lambda value: mx.array(value), tree)


def _require_tree_dtype(tree: object, dtype, name: str) -> None:
    for path, array in tree_flatten(tree):
        if array.dtype != dtype:
            raise ValueError(f"{name} leaf {path} must have dtype {dtype}")


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


def _ordered_batches(
    example_count: int, microbatch_size: int, steps: int
) -> tuple[tuple[int, ...], ...]:
    pattern: list[tuple[int, ...]] = []
    index = 0
    while index < example_count:
        take = min(microbatch_size, example_count - index)
        pattern.append(tuple(range(index, index + take)))
        index += take
    if not pattern or steps % len(pattern) != 0:
        raise ValueError("optimizer steps must cover complete encoded epochs")
    return tuple(pattern * (steps // len(pattern)))


def _loader_config() -> LoaderConfig:
    return LoaderConfig(
        microbatch_size=MICROBATCH_SIZE,
        gradient_accumulation_steps=1,
        prefetch_depth=1,
        epoch_seed=42,
    )


def _verified_source_snapshot(
    model_config: ModelConfig, lora_config: LoRAConfig, seed: int
) -> tuple[object, dict, dict, mx.array]:
    model_key, trainer_key = mx.random.split(mx.random.key(seed))
    model_key, adapter_key = mx.random.split(model_key)
    model = SMLLanguageModel(model_config, key=model_key)
    fp32_master = tree_map(lambda value: value.astype(mx.float32), model.parameters())
    mx.eval(fp32_master)
    _require_tree_dtype(fp32_master, mx.float32, "fp32 master")
    _, master_nonfinite = _state_finiteness_counts(fp32_master)
    if master_nonfinite:
        raise ValueError("fp32 master snapshot is not finite")
    bf16_working = tree_map(lambda value: value.astype(mx.bfloat16), fp32_master)
    mx.eval(bf16_working)
    _require_tree_dtype(bf16_working, mx.bfloat16, "bf16 working")
    model.update(bf16_working)
    apply_lora(model, lora_config, key=adapter_key)
    adapters, frozen_base = split_adapter_parameters(model.parameters())
    if not isinstance(adapters, dict) or not isinstance(frozen_base, dict):
        raise ValueError("adapter parameter split must return dictionaries")
    mx.eval(adapters, frozen_base)
    _require_tree_dtype(adapters, mx.float32, "adapters")
    _require_tree_dtype(frozen_base, mx.bfloat16, "frozen base")
    _, adapter_nonfinite = _state_finiteness_counts(adapters, frozen_base)
    if adapter_nonfinite:
        raise ValueError("verified LoRA snapshot is not finite")
    return model, fp32_master, frozen_base, adapters, trainer_key


def build_swag_quality_workload(root: Path) -> SwagQualityWorkload:
    if not isinstance(root, Path):
        raise TypeError("root must be a Path")
    training, training_fixture = _load_fixture(
        root, TRAINING_FIXTURE, split="training", example_count=TRAINING_EXAMPLE_COUNT
    )
    validation, validation_fixture = _load_fixture(
        root,
        VALIDATION_FIXTURE,
        split="validation",
        example_count=VALIDATION_EXAMPLE_COUNT,
    )
    _require_source_disjoint(training, validation)
    model_config = ModelConfig()
    lora_config = LoRAConfig()
    optimizer_config = default_swag_optimizer_config()
    loader_config = _loader_config()
    _model, fp32_master, frozen_base, adapters, _trainer_key = (
        _verified_source_snapshot(model_config, lora_config, MODEL_SEED)
    )
    fp32_master_identity = _array_tree_identity(
        "sml-swag-quality-fp32-master-v1", fp32_master
    )
    frozen_identity = _array_tree_identity(
        "sml-swag-quality-frozen-bf16-base-v1", frozen_base
    )
    adapter_identity = _array_tree_identity(
        "sml-swag-quality-fp32-adapters-v1", adapters
    )
    production_components = production_dependency_components(root)
    workload = SwagQualityWorkload(
        kind="swag-quality-workload",
        version=1,
        identity=_PLACEHOLDER_IDENTITY,
        training_fixture=training_fixture,
        validation_fixture=validation_fixture,
        fp32_master_identity=fp32_master_identity,
        frozen_bf16_base_identity=frozen_identity,
        initial_fp32_adapter_identity=adapter_identity,
        model=_json_mapping(dataclasses.asdict(model_config)),
        lora=_json_mapping(dataclasses.asdict(lora_config)),
        optimizer=_json_mapping(dataclasses.asdict(optimizer_config)),
        loader=_json_mapping(dataclasses.asdict(loader_config)),
        score_policy=SCORE_POLICY,
        ordered_batches=_ordered_batches(
            TRAINING_EXAMPLE_COUNT, MICROBATCH_SIZE, CANONICAL_STEPS
        ),
        optimizer_steps=CANONICAL_STEPS,
        model_seed=MODEL_SEED,
        validation_seed=VALIDATION_SEED,
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


def validate_swag_quality_records(
    workload: SwagQualityWorkload,
    records: Sequence[SwagQualityRecord],
) -> SwagQualityReport:
    if len(records) != 2:
        raise ValueError("swag quality evidence requires exactly two records")
    expected_order = ("candidate", "oracle")
    if tuple(record.runtime for record in records) != expected_order:
        raise ValueError("swag quality records have an invalid runtime order")
    for record in records:
        if SwagQualityRecord.from_dict(record.to_dict()) != record:
            raise ValueError("swag quality record does not round-trip canonically")
        if record.identity != record.recompute_identity():
            raise ValueError("swag quality record identity mismatch")
        if record.workload_identity != workload.identity:
            raise ValueError("swag quality record workload identity mismatch")
        if record.step != CANONICAL_STEPS:
            raise ValueError("swag quality record step must be 256")
        if record.frozen_base_identity != workload.frozen_bf16_base_identity:
            raise ValueError("swag quality record frozen base identity changed")
    candidate, oracle = records
    if candidate.real_example_count != oracle.real_example_count:
        raise ValueError("swag quality real-example counts do not match")
    return SwagQualityReport(
        candidate_validation_loss=candidate.validation_loss,
        oracle_validation_loss=oracle.validation_loss,
        candidate_accuracy=candidate.validation_accuracy,
        oracle_accuracy=oracle.validation_accuracy,
        candidate_examples=candidate.real_example_count,
        oracle_examples=oracle.real_example_count,
        candidate_finite=candidate.finite
        and math.isfinite(candidate.validation_loss)
        and math.isfinite(candidate.validation_accuracy),
        oracle_finite=oracle.finite
        and math.isfinite(oracle.validation_loss)
        and math.isfinite(oracle.validation_accuracy),
    )


def _canonical_steps(value: str) -> int:
    try:
        steps = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("--steps must be exactly 256") from error
    if steps != CANONICAL_STEPS:
        raise argparse.ArgumentTypeError("--steps must be exactly 256")
    return steps


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Controlled SWAG-quality gate")
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


def _synthetic_candidates(
    length: int, *, pad_token_id: int, bos_token_id: int, eos_token_id: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    input_ids = np.full((4, length), pad_token_id, dtype=_INT32)
    input_ids[:, 0] = bos_token_id
    input_ids[:, 1] = eos_token_id
    valid_token_mask = np.zeros((4, length), dtype=_BOOL)
    valid_token_mask[:, :2] = True
    score_mask = np.zeros((4, length), dtype=_BOOL)
    score_mask[:, 1] = True
    return input_ids, valid_token_mask, score_mask


def _assemble_batch(
    encoded: Mapping[str, np.ndarray],
    row_indices: Sequence[int],
    *,
    batch_size: int,
    model_config: ModelConfig,
) -> SwagBatch:
    length = encoded["input_ids"].shape[-1]
    input_ids = np.empty((batch_size, 4, length), dtype=_INT32)
    valid_token_mask = np.empty((batch_size, 4, length), dtype=_BOOL)
    score_mask = np.empty((batch_size, 4, length), dtype=_BOOL)
    labels = np.empty((batch_size,), dtype=_INT32)
    example_mask = np.zeros((batch_size,), dtype=_BOOL)
    for slot, row_index in enumerate(row_indices):
        input_ids[slot] = encoded["input_ids"][row_index]
        valid_token_mask[slot] = encoded["valid_token_mask"][row_index]
        score_mask[slot] = encoded["score_mask"][row_index]
        labels[slot] = int(encoded["labels"][row_index])
        example_mask[slot] = True
    if len(row_indices) < batch_size:
        syn_ids, syn_valid, syn_score = _synthetic_candidates(
            length,
            pad_token_id=model_config.pad_token_id,
            bos_token_id=model_config.bos_token_id,
            eos_token_id=model_config.eos_token_id,
        )
        for slot in range(len(row_indices), batch_size):
            input_ids[slot] = syn_ids
            valid_token_mask[slot] = syn_valid
            score_mask[slot] = syn_score
            labels[slot] = 0
            example_mask[slot] = False
    return SwagBatch(
        mx.array(input_ids),
        mx.array(score_mask),
        mx.array(labels),
        mx.array(example_mask),
        mx.array(valid_token_mask),
        SwagCursor.initial(),
    )


def _training_config(*, compile: bool) -> SwagTrainingConfig:
    return SwagTrainingConfig(
        base_checkpoint=Path("/unused-swag-quality-base"),
        data=Path("/unused-swag-quality-data"),
        output_run=Path("/unused-swag-quality-run"),
        lora=LoRAConfig(),
        optimizer=default_swag_optimizer_config(),
        loader=_loader_config(),
        maximum_steps=CANONICAL_STEPS,
        seed=MODEL_SEED,
        compile=compile,
    )


def _evaluate_validation(
    kernels,
    adapters: dict,
    frozen_base: dict,
    encoded: Mapping[str, np.ndarray],
    *,
    model_config: ModelConfig,
    key: mx.array,
) -> tuple[float, float, int]:
    trainer = initial_swag_trainer_state(adapters, key=key)
    example_count = int(encoded["input_ids"].shape[0])
    for index in range(example_count):
        batch = _assemble_batch(
            encoded, (index,), batch_size=1, model_config=model_config
        )
        trainer = kernels.ranking_microstep(adapters, frozen_base, trainer, batch)
    mx.eval(trainer.to_tree())
    valid = int(trainer.valid_count.item())
    if valid != example_count:
        raise ValueError("validation real-example count does not match the fixture")
    loss = float(trainer.loss_numerator.item()) / float(valid)
    accuracy = float(trainer.correct_count.item()) / float(valid)
    return loss, accuracy, valid


def _make_record(
    *,
    runtime: RuntimeName,
    workload: SwagQualityWorkload,
    train_loss: float,
    validation_loss: float,
    validation_accuracy: float,
    real_example_count: int,
    finite: bool,
    frozen_base_identity: str,
    adapter_identity: str,
) -> SwagQualityRecord:
    record = SwagQualityRecord(
        kind="swag-quality-record",
        version=1,
        identity=_PLACEHOLDER_IDENTITY,
        runtime=runtime,
        step=CANONICAL_STEPS,
        workload_identity=workload.identity,
        train_loss=train_loss,
        validation_loss=validation_loss,
        validation_accuracy=validation_accuracy,
        real_example_count=real_example_count,
        finite=finite,
        frozen_base_identity=frozen_base_identity,
        adapter_identity=adapter_identity,
    )
    return replace(record, identity=record.recompute_identity())


def _run_runtime(
    *,
    runtime: RuntimeName,
    compile: bool,
    workload: SwagQualityWorkload,
    model,
    frozen_base: dict,
    adapters: dict,
    trainer_key: mx.array,
    training: Mapping[str, np.ndarray],
    validation: Mapping[str, np.ndarray],
    model_config: ModelConfig,
) -> SwagQualityRecord:
    config = _training_config(compile=compile)
    weight_decay_tree = build_weight_decay_tree(adapters, config.optimizer.weight_decay)
    kernels = build_swag_kernels(model, config, weight_decay_tree)
    optimizer = initialize_adam_state(adapters)
    trainer = initial_swag_trainer_state(adapters, key=trainer_key)
    last_train_numerator = mx.array(0.0, dtype=mx.float32)
    last_train_valid = mx.array(0, dtype=mx.int32)
    for step, indices in enumerate(workload.ordered_batches, start=1):
        batch = _assemble_batch(
            training,
            indices,
            batch_size=MICROBATCH_SIZE,
            model_config=model_config,
        )
        trainer = kernels.ranking_microstep(adapters, frozen_base, trainer, batch)
        last_train_numerator = trainer.loss_numerator
        last_train_valid = trainer.valid_count
        adapters, optimizer, trainer = kernels.optimizer_step(
            adapters, optimizer, trainer
        )
        mx.eval(adapters, optimizer.to_tree(), trainer.to_tree(), frozen_base)
        if step % 32 == 0:
            print(
                f"swag quality {runtime}: optimizer step {step}/{CANONICAL_STEPS}",
                file=sys.stderr,
                flush=True,
            )
    mx.eval(last_train_numerator, last_train_valid)
    last_train_loss = float(last_train_numerator.item()) / float(
        last_train_valid.item()
    )
    val_loss, val_accuracy, val_examples = _evaluate_validation(
        kernels,
        adapters,
        frozen_base,
        validation,
        model_config=model_config,
        key=mx.random.key(workload.validation_seed),
    )
    frozen_identity = _array_tree_identity(
        "sml-swag-quality-frozen-bf16-base-v1", frozen_base
    )
    adapter_identity = _array_tree_identity(
        "sml-swag-quality-fp32-adapters-v1", adapters
    )
    _, nonfinite = _state_finiteness_counts(
        adapters, frozen_base, optimizer.to_tree(), trainer.to_tree()
    )
    finite = (
        nonfinite == 0
        and math.isfinite(last_train_loss)
        and math.isfinite(val_loss)
        and math.isfinite(val_accuracy)
        and frozen_identity == workload.frozen_bf16_base_identity
    )
    return _make_record(
        runtime=runtime,
        workload=workload,
        train_loss=last_train_loss,
        validation_loss=val_loss,
        validation_accuracy=val_accuracy,
        real_example_count=val_examples,
        finite=finite,
        frozen_base_identity=frozen_identity,
        adapter_identity=adapter_identity,
    )


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_bytes(root: Path, *arguments: str) -> bytes:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


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
        "module": "v2.benchmarks.swag_quality",
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


def _recording_session_identity(
    source_commit: str,
    workload_identity: str,
    recording_command: Mapping[str, object],
) -> str:
    return structured_identity(
        "sml-swag-quality-recording-session-v1",
        {
            "source_commit": source_commit,
            "workload_identity": workload_identity,
            "recording_command": dict(recording_command),
        },
    )


def _manifest_document(
    *,
    workload: SwagQualityWorkload,
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
    if recording_session_identity != _recording_session_identity(
        source_commit, workload.identity, command
    ):
        raise ValueError("quality recording session identity is not derived")
    manifest_size = 0
    result: dict[str, object] | None = None
    for _iteration in range(20):
        artifact_sizes = {
            "raw": raw_bytes,
            "manifest": manifest_size,
            "report": report_bytes,
        }
        body = {
            "kind": "swag-quality-manifest",
            "version": 1,
            "source_commit": source_commit,
            "harness_commit": source_commit,
            "harness_clean": True,
            "harness_identity": workload.harness_identity,
            "production_dependency_components": list(
                workload.production_dependency_components
            ),
            "production_dependency_identity": workload.production_dependency_identity,
            "workload": workload.to_dict(),
            "workload_identity": workload.identity,
            "recording_command": command,
            "measurement_boundaries": dict(MEASUREMENT_BOUNDARIES),
            "phase_wall_time_seconds": normalized_phases,
            "measured_wall_time_seconds": measured_wall_time,
            "wall_time_budget_seconds": QUALITY_WALL_TIME_BUDGET_SECONDS,
            "peak_metal_memory_bytes": peak_memory,
            "recording_session_identity": recording_session_identity,
            "artifact_byte_sizes": artifact_sizes,
            "fixture_bytes": (
                workload.training_fixture.byte_size
                + workload.validation_fixture.byte_size
            ),
            "training_cardinality": workload.training_fixture.example_count,
            "validation_cardinality": workload.validation_fixture.example_count,
            "ordered_work_count": len(workload.ordered_batches),
            "optimizer_steps": CANONICAL_STEPS,
            "record_count": 2,
            "raw_identity": raw_identity,
            "raw_file_identity": raw_file_identity,
            "report_identity": report_identity,
            "report_file_identity": report_file_identity,
        }
        result = {
            **body,
            "identity": structured_identity("sml-swag-quality-manifest-v1", body),
        }
        next_manifest_size = len(canonical_json_bytes(result))
        if next_manifest_size == manifest_size:
            return result
        manifest_size = next_manifest_size
    raise RuntimeError("quality manifest byte-size fixed point did not converge")


def _report_document(
    workload_identity: str,
    raw_identity: str,
    report: SwagQualityReport,
) -> dict[str, object]:
    body = {
        "kind": "swag-quality-report",
        "version": 1,
        "workload_identity": workload_identity,
        "raw_identity": raw_identity,
        "report": dataclasses.asdict(report),
        "decision": decide_swag_quality(report),
    }
    return {
        **body,
        "identity": structured_identity("sml-swag-quality-report-v1", body),
    }


def _validate_manifest_fields(
    raw: Mapping[str, object],
    expected_workload: SwagQualityWorkload,
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
        "recording_session_identity",
        "artifact_byte_sizes",
        "fixture_bytes",
        "training_cardinality",
        "validation_cardinality",
        "ordered_work_count",
        "optimizer_steps",
        "record_count",
        "raw_identity",
        "raw_file_identity",
        "report_identity",
        "report_file_identity",
    }
    if set(raw) != expected_fields:
        raise ValueError("swag quality manifest has an invalid field set")
    if raw["kind"] != "swag-quality-manifest" or raw["version"] != 1:
        raise ValueError("unsupported swag quality manifest")
    if raw["source_commit"] != raw["harness_commit"]:
        raise ValueError("source_commit must equal harness_commit")
    if raw["harness_clean"] is not True:
        raise ValueError("quality recording checkout must be clean")
    if raw["optimizer_steps"] != CANONICAL_STEPS or raw["record_count"] != 2:
        raise ValueError("swag quality manifest step or record count is invalid")
    if raw["recording_command"] != dict(expected_command):
        raise ValueError("quality recording command is not canonical")
    if raw["measurement_boundaries"] != dict(MEASUREMENT_BOUNDARIES):
        raise ValueError("quality measurement boundaries changed")
    if raw["wall_time_budget_seconds"] != QUALITY_WALL_TIME_BUDGET_SECONDS:
        raise ValueError("quality wall-time budget changed")
    workload_raw = raw["workload"]
    if not isinstance(workload_raw, dict):
        raise ValueError("swag quality manifest workload must be an object")
    workload = SwagQualityWorkload.from_dict(workload_raw)
    if workload != expected_workload or raw["workload_identity"] != workload.identity:
        raise ValueError("swag quality manifest workload mismatch")
    if raw["harness_identity"] != workload.harness_identity:
        raise ValueError("swag quality manifest harness identity mismatch")
    if raw["production_dependency_identity"] != workload.production_dependency_identity:
        raise ValueError("swag quality manifest production identity mismatch")
    if raw["production_dependency_components"] != list(
        workload.production_dependency_components
    ):
        raise ValueError("swag quality manifest production components mismatch")
    phases = raw["phase_wall_time_seconds"]
    if not isinstance(phases, dict) or set(phases) != set(_PHASE_NAMES):
        raise ValueError("quality phase timing fields are invalid")
    measured = math.fsum(float(phases[name]) for name in _PHASE_NAMES)
    if raw["measured_wall_time_seconds"] != measured:
        raise ValueError("measured wall time is not the sum of phases")
    sizes = raw["artifact_byte_sizes"]
    if not isinstance(sizes, dict) or set(sizes) != {"raw", "manifest", "report"}:
        raise ValueError("quality artifact sizes are invalid")
    if raw["fixture_bytes"] != (
        workload.training_fixture.byte_size + workload.validation_fixture.byte_size
    ):
        raise ValueError("quality fixture byte total changed")
    if raw["training_cardinality"] != workload.training_fixture.example_count:
        raise ValueError("training cardinality changed")
    if raw["validation_cardinality"] != workload.validation_fixture.example_count:
        raise ValueError("validation cardinality changed")
    if raw["ordered_work_count"] != len(workload.ordered_batches):
        raise ValueError("ordered work count changed")
    peak_memory = raw["peak_metal_memory_bytes"]
    if type(peak_memory) is not int or peak_memory < 0:
        raise ValueError("peak metal memory must be a nonnegative integer")
    body = {key: value for key, value in raw.items() if key != "identity"}
    if raw["identity"] != structured_identity("sml-swag-quality-manifest-v1", body):
        raise ValueError("swag quality manifest identity mismatch")
    if raw["recording_session_identity"] != _recording_session_identity(
        str(raw["source_commit"]), workload.identity, dict(expected_command)
    ):
        raise ValueError("quality recording session identity is not derived")
    return dict(raw)


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
    workload: SwagQualityWorkload,
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


def _json_object_no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


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


def _read_raw(path: Path) -> tuple[SwagQualityRecord, ...]:
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
        records.append(SwagQualityRecord.from_dict(raw))
        framed.append(line + b"\n")
    if b"".join(framed) != payload:
        raise ValueError("quality raw JSONL has invalid framing")
    return tuple(records)


def _validate_evidence_files(
    root: Path,
    destinations: _EvidenceDestinations,
    *,
    recorded_workload: SwagQualityWorkload | None = None,
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
        raise ValueError("swag quality manifest workload must be an object")
    workload = SwagQualityWorkload.from_dict(workload_raw)
    if recorded_workload is not None and recorded_workload != workload:
        raise ValueError("swag quality manifest workload changed")
    manifest = _validate_manifest_fields(raw_manifest, workload, expected_command)
    _validate_harness_commit(root, str(manifest["harness_commit"]), workload)
    artifact_sizes = manifest["artifact_byte_sizes"]
    if len(manifest_payload) != artifact_sizes["manifest"]:
        raise ValueError("quality manifest byte size changed")
    raw_payload = destinations.raw_output.read_bytes()
    if (
        len(raw_payload) != artifact_sizes["raw"]
        or _payload_identity(raw_payload) != manifest["raw_file_identity"]
    ):
        raise ValueError("swag quality raw file identity mismatch")
    records = _read_raw(destinations.raw_output)
    raw_identity = structured_identity(
        "sml-swag-quality-raw-v1",
        [record.to_dict() for record in records],
    )
    if raw_identity != manifest["raw_identity"]:
        raise ValueError("swag quality raw identity mismatch")
    report = validate_swag_quality_records(workload, records)
    expected_report = _report_document(workload.identity, raw_identity, report)
    report_payload = destinations.report.read_bytes()
    if (
        len(report_payload) != artifact_sizes["report"]
        or _payload_identity(report_payload) != manifest["report_file_identity"]
        or expected_report["identity"] != manifest["report_identity"]
        or _decode_json_object(report_payload, label=str(destinations.report))
        != expected_report
    ):
        raise ValueError("swag quality report does not match raw evidence")
    return decide_swag_quality(report)


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


def _require_clean_recording_checkout(
    root: Path, destinations: _EvidenceDestinations
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
    unexpected = []
    for encoded in untracked:
        if not encoded:
            continue
        relative = encoded.decode("utf-8")
        if relative in allowed_files:
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
    _require_clean_recording_checkout(root, destinations)
    source_commit = _git(root, "rev-parse", "HEAD")
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise RuntimeError("quality recording requires a full Git source commit")
    all_destinations, any_destination = _all_or_no_destinations(destinations)
    if all_destinations:
        decision = _validate_evidence_files(root, destinations)
        print(decision)
        return 0 if decision == "pass" else 1
    if any_destination:
        raise FileExistsError("partial quality evidence exists; discard it and retry")
    mx.clear_cache()
    mx.reset_peak_memory()
    workload = build_swag_quality_workload(root)
    recording_command = _recording_command_document(root, destinations)
    session_identity = _recording_session_identity(
        source_commit, workload.identity, recording_command
    )
    training = _load_encoded_arrays(root / TRAINING_FIXTURE)
    validation = _load_encoded_arrays(root / VALIDATION_FIXTURE)
    model_config = ModelConfig()
    lora_config = LoRAConfig()
    model, _fp32_master, frozen_base, adapters, trainer_key = _verified_source_snapshot(
        model_config, lora_config, MODEL_SEED
    )
    if (
        _array_tree_identity("sml-swag-quality-frozen-bf16-base-v1", frozen_base)
        != workload.frozen_bf16_base_identity
        or _array_tree_identity("sml-swag-quality-fp32-adapters-v1", adapters)
        != workload.initial_fp32_adapter_identity
    ):
        raise RuntimeError("verified SWAG source snapshot drifted from the workload")
    setup_elapsed = time.monotonic() - recording_started
    records = []
    runtime_times: dict[str, float] = {}
    for runtime, compile in (("candidate", True), ("oracle", False)):
        runtime_started = time.monotonic()
        records.append(
            _run_runtime(
                runtime=runtime,
                compile=compile,
                workload=workload,
                model=model,
                frozen_base=_clone_tree(frozen_base),
                adapters=_clone_tree(adapters),
                trainer_key=mx.array(trainer_key),
                training=training,
                validation=validation,
                model_config=model_config,
            )
        )
        mx.clear_cache()
        runtime_times[runtime] = time.monotonic() - runtime_started
    validation_started = time.monotonic()
    report = validate_swag_quality_records(workload, records)
    raw_documents = [record.to_dict() for record in records]
    raw_identity = structured_identity("sml-swag-quality-raw-v1", raw_documents)
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
        raise RuntimeError("swag quality workload exceeded its 4-hour budget")
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
    try:
        _durable_create(destinations.raw_output, raw_bytes)
        _durable_create(destinations.manifest, manifest_bytes)
        _durable_create(destinations.report, report_bytes)
    except Exception:
        for _name, path in destinations.ordered():
            path.unlink(missing_ok=True)
        raise
    validated_decision = _validate_evidence_files(
        root, destinations, recorded_workload=workload
    )
    decision = decide_swag_quality(report)
    if decision != validated_decision:
        raise RuntimeError("published quality decision changed during validation")
    print(decision)
    return 0 if decision == "pass" else 1


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
    "SwagQualityFixture",
    "SwagQualityRecord",
    "SwagQualityReport",
    "SwagQualityWorkload",
    "build_swag_quality_workload",
    "decide_swag_quality",
    "harness_content_identity",
    "production_dependency_components",
    "production_dependency_content_identity",
    "validate_swag_quality_records",
    "write_canonical_fixtures",
)


if __name__ == "__main__":
    raise SystemExit(main())
