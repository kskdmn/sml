from __future__ import annotations

# Serialized quality schemas report malformed external content uniformly as ValueError.
# ruff: noqa: TRY004
import argparse
import dataclasses
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
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
TRAINING_FIXTURE = Path("v2/benchmarks/fixtures/pretraining-quality-train-v1.npy")
VALIDATION_FIXTURE = Path(
    "v2/benchmarks/fixtures/pretraining-quality-validation-v1.npy"
)
TRAINING_SHAPE = (32, 1_025)
VALIDATION_SHAPE = (8, 1_025)
_IDENTITY_RE = re.compile(r"sha256:[0-9a-f]{64}")
_PLACEHOLDER_IDENTITY = "sha256:" + "0" * 64

RuntimeName = Literal["candidate", "oracle"]
ComputeDtype = Literal["bfloat16", "float32"]


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
class PretrainingQualityWorkload:
    kind: Literal["pretraining-quality-workload"]
    version: Literal[1]
    identity: str
    training_fixture: QualityFixture
    validation_fixture: QualityFixture
    initial_bf16_parameter_identity: str
    parameter_leaf_names: tuple[str, ...]
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

    def _body(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "version": self.version,
            "training_fixture": self.training_fixture.to_dict(),
            "validation_fixture": self.validation_fixture.to_dict(),
            "initial_bf16_parameter_identity": self.initial_bf16_parameter_identity,
            "parameter_leaf_names": list(self.parameter_leaf_names),
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
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._body(), "identity": self.identity}

    def recompute_identity(self) -> str:
        return structured_identity("sml-pretraining-quality-workload-v1", self._body())

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
        }
        if set(raw) != expected:
            raise ValueError("pretraining quality workload has an invalid field set")
        if raw["kind"] != "pretraining-quality-workload" or raw["version"] != 1:
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
            "ordered_batches",
            "checkpoint_steps",
            "evaluation_row_indices",
            "harness_components",
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
        if any(type(step) is not int for step in checkpoint_steps):
            raise ValueError("checkpoint steps must be integers")
        if any(type(index) is not int or index < 0 for index in evaluation_rows):
            raise ValueError("evaluation row indices must be nonnegative integers")
        if any(not isinstance(value, str) or not value for value in components):
            raise ValueError("harness components must be non-empty strings")
        workload = cls(
            kind="pretraining-quality-workload",
            version=1,
            identity=_require_identity(raw["identity"], "workload identity"),
            training_fixture=QualityFixture.from_dict(training_raw),
            validation_fixture=QualityFixture.from_dict(validation_raw),
            initial_bf16_parameter_identity=_require_identity(
                raw["initial_bf16_parameter_identity"], "initial parameter identity"
            ),
            parameter_leaf_names=tuple(parameter_names),
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
        if workload.identity != workload.recompute_identity():
            raise ValueError("pretraining quality workload identity mismatch")
        return workload


@dataclass(frozen=True, slots=True)
class ParameterUpdateStatistics:
    path: str
    value_count: int
    nonzero_update_count: int
    sub_bf16_ulp_update_count: int
    survived_sub_bf16_ulp_count: int
    minimum_update_to_bf16_ulp: float | None
    mean_update_to_bf16_ulp: float | None
    maximum_update_to_bf16_ulp: float | None

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> ParameterUpdateStatistics:
        if set(raw) != {field.name for field in dataclasses.fields(cls)}:
            raise ValueError("update statistics have an invalid field set")
        path = raw["path"]
        if not isinstance(path, str) or not path:
            raise ValueError("update statistics path must be non-empty")
        counts = {
            name: _require_plain_int(raw[name], name)
            for name in (
                "value_count",
                "nonzero_update_count",
                "sub_bf16_ulp_update_count",
                "survived_sub_bf16_ulp_count",
            )
        }
        if not (
            counts["survived_sub_bf16_ulp_count"]
            <= counts["sub_bf16_ulp_update_count"]
            <= counts["nonzero_update_count"]
            <= counts["value_count"]
        ):
            raise ValueError("update statistics counts are inconsistent")
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
        return cls(path, *counts.values(), *ratios)


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


def _array_tree_identity(domain: str, tree: object) -> str:
    leaves = sorted(tree_flatten(tree))
    if not leaves:
        raise ValueError("quality array tree must not be empty")
    mx.eval(*(array for _path, array in leaves))
    digest = hashlib.sha256()
    digest.update(domain.encode("ascii"))
    digest.update(b"\0")
    for path, array in leaves:
        if not isinstance(array, mx.array):
            raise ValueError("quality tree leaves must be MLX arrays")
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
    parameter_names = tuple(
        sorted(path for path, _value in tree_flatten(initial_working))
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
    workload = PretrainingQualityWorkload(
        kind="pretraining-quality-workload",
        version=1,
        identity=_PLACEHOLDER_IDENTITY,
        training_fixture=training_fixture,
        validation_fixture=validation_fixture,
        initial_bf16_parameter_identity=initial_identity,
        parameter_leaf_names=parameter_names,
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
    value_count: int
    nonzero_count: mx.array
    sub_ulp_count: mx.array
    survived_count: mx.array
    minimum_ratio: mx.array
    ratio_sum: mx.array
    maximum_ratio: mx.array
    changed_working_count: mx.array
    cumulative_survived: mx.array


def _observe_update(
    previous_masters: dict,
    updated_masters: dict,
    previous_working: dict,
    updated_working: dict,
    cumulative_survival: Mapping[str, mx.array],
) -> tuple[_LeafUpdateObservation, ...]:
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
        cumulative = cumulative_survival[path] | mx.any(survived)
        observations.append(
            _LeafUpdateObservation(
                path=path,
                value_count=math.prod(previous_master.shape),
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
                cumulative_survived=cumulative,
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
            observation.cumulative_survived,
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
        statistics.append(
            ParameterUpdateStatistics(
                observation.path,
                observation.value_count,
                count,
                int(observation.sub_ulp_count.item()),
                int(observation.survived_count.item()),
                minimum,
                mean,
                maximum,
            )
        )
        survived_any = survived_any or bool(observation.cumulative_survived.item())
    return tuple(statistics), survived_any


def _empty_update_statistics(parameters: dict) -> tuple[ParameterUpdateStatistics, ...]:
    return tuple(
        ParameterUpdateStatistics(
            path, math.prod(value.shape), 0, 0, 0, None, None, None
        )
        for path, value in sorted(tree_flatten(parameters))
    )


def _changed_working_fraction(observations: Sequence[_LeafUpdateObservation]) -> float:
    if not observations:
        return 0.0
    mx.eval(*(item.changed_working_count for item in observations))
    changed = sum(int(item.changed_working_count.item()) for item in observations)
    total = sum(item.value_count for item in observations)
    return changed / total


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
    for record in records:
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
        if (
            tuple(item.path for item in record.update_statistics)
            != workload.parameter_leaf_names
        ):
            raise ValueError("quality checkpoint parameter statistics are incomplete")
        if record.step == 0 and (
            record.rms_norm_master_moved
            or record.sub_bf16_update_survived
            or record.changed_bf16_working_fraction != 0.0
        ):
            raise ValueError("step-zero quality checkpoint has update evidence")
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


def _state_is_finite(*trees: object) -> bool:
    checks = [
        mx.all(mx.isfinite(value))
        for tree in trees
        for _path, value in tree_flatten(tree)
        if value.dtype not in (mx.uint32, mx.int32, mx.bool_)
    ]
    combined = mx.all(mx.stack(checks)) if checks else mx.array(True)
    mx.eval(combined)
    return bool(combined.item())


def _rms_norm_moved(initial_masters: dict, masters: dict) -> bool:
    initial = dict(tree_flatten(initial_masters))
    current = dict(tree_flatten(masters))
    norm_paths = [
        path
        for path in sorted(initial)
        if ".input_norm." in path
        or ".post_attn_norm." in path
        or path.startswith("norm.")
    ]
    moved = mx.any(
        mx.stack([mx.any(initial[path] != current[path]) for path in norm_paths])
    )
    mx.eval(moved)
    return bool(moved.item())


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
    initial_masters: dict,
    masters: dict,
    working: dict,
    adam_tree: tuple,
    trainer_tree: tuple,
    train_loss: float,
    validation_nll: float,
    observations: Sequence[_LeafUpdateObservation],
) -> PretrainingQualityCheckpoint:
    if observations:
        statistics, survived = _materialize_update_observation(observations)
        changed_fraction = _changed_working_fraction(observations)
    else:
        statistics = _empty_update_statistics(masters)
        survived = False
        changed_fraction = 0.0
    finite = (
        _state_is_finite(masters, working, adam_tree, trainer_tree)
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
        master_parameter_identity=_array_tree_identity(
            "sml-pretraining-quality-master-parameters-v1", masters
        ),
        working_parameter_identity=_array_tree_identity(
            "sml-pretraining-quality-working-parameters-v1", working
        ),
        trainer_key_identity=_trainer_key_identity(trainer_tree),
        train_loss=train_loss,
        validation_nll=validation_nll,
        finite=finite,
        update_statistics=statistics,
        changed_bf16_working_fraction=changed_fraction,
        rms_norm_master_moved=_rms_norm_moved(initial_masters, masters),
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
        previous_working = (
            working
            if runtime == "candidate"
            else tree_map(lambda value: value.astype(mx.bfloat16), masters)
        )
        masters, working, adam_tree, trainer_tree, metrics = (
            kernels.optimizer_step_core(
                masters,
                working,
                adam_tree,
                trainer_tree,
            )
        )
        updated_working = (
            working
            if runtime == "candidate"
            else tree_map(lambda value: value.astype(mx.bfloat16), masters)
        )
        observations = _observe_update(
            previous_masters,
            masters,
            previous_working,
            updated_working,
            cumulative_survival,
        )
        cumulative_survival = {
            item.path: item.cumulative_survived for item in observations
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
        path: mx.array(False, dtype=mx.bool_) for path, _value in tree_flatten(masters)
    }
    records = []
    started = time.monotonic()
    initial_train_loss = _validation_nll(model, masters, training_rows[:1])
    initial_validation_nll = _validation_nll(model, masters, validation_rows)
    records.append(
        _checkpoint_record(
            runtime=runtime,
            step=0,
            workload=workload,
            initial_masters=parameters.master_parameters,
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
                initial_masters=parameters.master_parameters,
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
    command: str,
    wall_time: float,
    runtime_times: Mapping[str, float],
    peak_memory: int,
    raw_identity: str,
) -> dict[str, object]:
    body = {
        "kind": "pretraining-quality-manifest",
        "version": 1,
        "source_commit": source_commit,
        "harness_commit": source_commit,
        "harness_clean": True,
        "harness_identity": workload.harness_identity,
        "workload": workload.to_dict(),
        "workload_identity": workload.identity,
        "command": command,
        "wall_time_seconds": wall_time,
        "runtime_wall_time_seconds": dict(runtime_times),
        "wall_time_budget_seconds": QUALITY_WALL_TIME_BUDGET_SECONDS,
        "peak_metal_memory_bytes": peak_memory,
        "temporary_disk_bytes": 0,
        "fixture_bytes": (
            workload.training_fixture.byte_size + workload.validation_fixture.byte_size
        ),
        "training_cardinality": workload.training_fixture.shape[0],
        "validation_cardinality": workload.validation_fixture.shape[0],
        "ordered_work_count": len(workload.ordered_batches),
        "record_count": 8,
        "raw_identity": raw_identity,
    }
    return {
        **body,
        "identity": structured_identity("sml-pretraining-quality-manifest-v1", body),
    }


def _report_document(
    manifest: Mapping[str, object],
    raw_identity: str,
    report: PretrainingQualityReport,
) -> dict[str, object]:
    body = {
        "kind": "pretraining-quality-report",
        "version": 1,
        "manifest_identity": manifest["identity"],
        "workload_identity": manifest["workload_identity"],
        "raw_identity": raw_identity,
        "report": dataclasses.asdict(report),
        "decision": decide_pretraining_quality(report),
    }
    return {
        **body,
        "identity": structured_identity("sml-pretraining-quality-report-v1", body),
    }


def _write_immutable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"quality evidence already exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(payload)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


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


def _record(args: argparse.Namespace) -> int:
    root = _root()
    _preflight_output_paths(
        Path(args.manifest), Path(args.raw_output), Path(args.output)
    )
    if _git(root, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("quality recording requires a clean checkout")
    source_commit = _git(root, "rev-parse", "HEAD")
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise RuntimeError("quality recording requires a full Git source commit")
    workload = build_pretraining_quality_workload(root)
    training_rows = np.load(root / TRAINING_FIXTURE, allow_pickle=False, mmap_mode="r")
    validation_rows = np.load(
        root / VALIDATION_FIXTURE, allow_pickle=False, mmap_mode="r"
    )
    mx.clear_cache()
    mx.reset_peak_memory()
    started = time.monotonic()
    records = []
    runtime_times = {}
    for runtime in ("candidate", "oracle"):
        runtime_records, elapsed = _run_runtime(
            root, workload, runtime, training_rows, validation_rows
        )
        records.extend(runtime_records)
        runtime_times[runtime] = elapsed
        mx.clear_cache()
    wall_time = time.monotonic() - started
    if wall_time > QUALITY_WALL_TIME_BUDGET_SECONDS:
        raise RuntimeError("pretraining quality workload exceeded its 12-hour budget")
    report = validate_pretraining_quality_records(workload, records)
    raw_documents = [record.to_dict() for record in records]
    raw_identity = structured_identity("sml-pretraining-quality-raw-v1", raw_documents)
    manifest = _manifest_document(
        workload=workload,
        source_commit=source_commit,
        command=shlex.join(sys.argv),
        wall_time=wall_time,
        runtime_times=runtime_times,
        peak_memory=int(mx.get_peak_memory()),
        raw_identity=raw_identity,
    )
    report_document = _report_document(manifest, raw_identity, report)
    raw_bytes = b"".join(
        canonical_json_bytes(document) + b"\n" for document in raw_documents
    )
    total_size = (
        workload.training_fixture.byte_size
        + workload.validation_fixture.byte_size
        + len(raw_bytes)
        + len(canonical_json_bytes(report_document))
    )
    if total_size > 64 * 1024 * 1024:
        raise RuntimeError("quality fixtures and evidence exceed 64 MiB")
    _write_immutable(Path(args.raw_output), raw_bytes)
    _write_immutable(Path(args.manifest), canonical_json_bytes(manifest))
    _write_immutable(Path(args.output), canonical_json_bytes(report_document))
    decision = decide_pretraining_quality(report)
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
    payload = path.read_bytes()
    value = json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=_json_object_no_duplicates,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"invalid JSON constant: {token}")
        ),
    )
    if not isinstance(value, dict) or canonical_json_bytes(value) != payload:
        raise ValueError(f"quality JSON is not a canonical object: {path}")
    return value


def _validate_manifest_fields(
    raw: Mapping[str, object], expected_workload: PretrainingQualityWorkload
) -> dict[str, object]:
    expected_fields = {
        "kind",
        "version",
        "identity",
        "source_commit",
        "harness_commit",
        "harness_clean",
        "harness_identity",
        "workload",
        "workload_identity",
        "command",
        "wall_time_seconds",
        "runtime_wall_time_seconds",
        "wall_time_budget_seconds",
        "peak_metal_memory_bytes",
        "temporary_disk_bytes",
        "fixture_bytes",
        "training_cardinality",
        "validation_cardinality",
        "ordered_work_count",
        "record_count",
        "raw_identity",
    }
    if set(raw) != expected_fields:
        raise ValueError("pretraining quality manifest has an invalid field set")
    if raw["kind"] != "pretraining-quality-manifest" or raw["version"] != 1:
        raise ValueError("unsupported pretraining quality manifest")
    body = {key: value for key, value in raw.items() if key != "identity"}
    if raw["identity"] != structured_identity(
        "sml-pretraining-quality-manifest-v1", body
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
    if raw["harness_clean"] is not True:
        raise ValueError("pretraining quality manifest requires a clean harness")
    if raw["source_commit"] != raw["harness_commit"]:
        raise ValueError("quality source and harness commits must match")
    if (
        not isinstance(raw["source_commit"], str)
        or re.fullmatch(r"[0-9a-f]{40}", raw["source_commit"]) is None
    ):
        raise ValueError("quality source commit must be a full Git commit")
    if raw["record_count"] != 8:
        raise ValueError("quality manifest record count must be eight")
    if raw["wall_time_budget_seconds"] != QUALITY_WALL_TIME_BUDGET_SECONDS:
        raise ValueError("quality manifest wall-time budget changed")
    wall_time = _require_finite(raw["wall_time_seconds"], "quality wall time")
    if wall_time <= 0.0 or wall_time > QUALITY_WALL_TIME_BUDGET_SECONDS:
        raise ValueError("quality manifest exceeds its wall-time budget")
    runtime_times = raw["runtime_wall_time_seconds"]
    if not isinstance(runtime_times, dict) or set(runtime_times) != {
        "candidate",
        "oracle",
    }:
        raise ValueError("quality runtime wall times must name candidate and oracle")
    normalized_runtime_times = {
        runtime: _require_finite(value, f"{runtime} wall time")
        for runtime, value in runtime_times.items()
    }
    if (
        any(value <= 0.0 for value in normalized_runtime_times.values())
        or math.fsum(normalized_runtime_times.values()) > wall_time
    ):
        raise ValueError("quality runtime wall times are inconsistent")
    command = raw["command"]
    if not isinstance(command, str) or not command:
        raise ValueError("quality command must be non-empty")
    try:
        command_parts = shlex.split(command)
        steps_index = command_parts.index("--steps")
    except (ValueError, IndexError) as error:
        raise ValueError("quality command must record exactly 1000 steps") from error
    if (
        "record" not in command_parts
        or steps_index + 1 >= len(command_parts)
        or command_parts[steps_index + 1] != str(CANONICAL_STEPS)
    ):
        raise ValueError("quality command must record exactly 1000 steps")
    if (
        type(raw["peak_metal_memory_bytes"]) is not int
        or raw["peak_metal_memory_bytes"] < 0
    ):
        raise ValueError("quality peak Metal memory must be nonnegative")
    if raw["temporary_disk_bytes"] != 0:
        raise ValueError("quality temporary disk bytes must be zero")
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
    _require_identity(raw["raw_identity"], "quality raw identity")
    return dict(raw)


def _git_bytes(root: Path, *arguments: str) -> bytes:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout


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
) -> dict[str, object]:
    manifest = _validate_manifest_fields(raw, expected_workload)
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


def _validate(args: argparse.Namespace) -> int:
    root = _root()
    workload = build_pretraining_quality_workload(root)
    manifest = _validate_manifest(_read_json(Path(args.manifest)), workload, root)
    records = _read_raw(Path(args.raw_input))
    raw_identity = structured_identity(
        "sml-pretraining-quality-raw-v1",
        [record.to_dict() for record in records],
    )
    if raw_identity != manifest["raw_identity"]:
        raise ValueError("pretraining quality raw identity mismatch")
    report = validate_pretraining_quality_records(workload, records)
    expected_report = _report_document(manifest, raw_identity, report)
    if _read_json(Path(args.report)) != expected_report:
        raise ValueError("pretraining quality report does not match raw evidence")
    decision = decide_pretraining_quality(report)
    print(decision)
    return 0 if decision == "pass" else 1


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return args.handler(args)


__all__ = (
    "CANONICAL_STEPS",
    "CHECKPOINT_STEPS",
    "ParameterUpdateStatistics",
    "PretrainingQualityCheckpoint",
    "PretrainingQualityReport",
    "PretrainingQualityWorkload",
    "build_pretraining_quality_workload",
    "decide_pretraining_quality",
    "harness_content_identity",
    "real_work_identity",
    "validate_pretraining_quality_records",
)


if __name__ == "__main__":
    raise SystemExit(main())
