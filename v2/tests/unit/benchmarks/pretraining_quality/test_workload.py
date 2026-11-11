from __future__ import annotations

import dataclasses
import hashlib
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from sml.model.config import ModelConfig
from sml.training.common import (
    LoaderConfig,
    OptimizerConfig,
)

import v2.benchmarks.quality as quality_module
from v2.benchmarks.quality import (
    CANONICAL_STEPS,
    CHECKPOINT_STEPS,
    ParameterLeafSpec,
    PretrainingQualityReport,
    PretrainingQualityWorkload,
    build_pretraining_quality_workload,
    decide_pretraining_quality,
    harness_content_identity,
    production_dependency_components,
    production_dependency_content_identity,
)

from . import ROOT


@pytest.mark.parametrize(
    "changes,decision",
    [
        ({}, "pass"),
        ({"sub_bf16_update_survived": False}, "fail"),
        ({"candidate_finite": False}, "fail"),
        ({"oracle_finite": False}, "fail"),
        ({"rms_norm_master_moved": False}, "fail"),
        ({"matching_work_identity": False}, "fail"),
        ({"candidate_validation_nll": float("nan")}, "fail"),
        ({"oracle_validation_nll": float("inf")}, "fail"),
    ],
)
def test_quality_gate_requires_every_acceptance_input(changes, decision):
    report = PretrainingQualityReport(
        candidate_validation_nll=2.01,
        oracle_validation_nll=2.0,
        candidate_finite=True,
        oracle_finite=True,
        rms_norm_master_moved=True,
        sub_bf16_update_survived=True,
        matching_work_identity=True,
    )
    assert decide_pretraining_quality(replace(report, **changes)) == decision


def test_workload_binds_checked_in_source_disjoint_rows_and_exact_work(
    canonical_workload,
):
    workload = canonical_workload

    assert workload.checkpoint_steps == CHECKPOINT_STEPS == (0, 10, 100, 1_000)
    assert workload.model == dataclasses.asdict(ModelConfig())
    assert workload.optimizer == dataclasses.asdict(
        OptimizerConfig(schedule_steps=268_000)
    )
    assert workload.optimizer["schedule_steps"] == 268_000
    assert workload.loader == dataclasses.asdict(LoaderConfig())
    assert len(workload.ordered_batches) == (
        CANONICAL_STEPS * LoaderConfig().gradient_accumulation_steps
    )
    assert workload.training_fixture.logical_path.endswith("train-v1.npy")
    assert workload.validation_fixture.logical_path.endswith("validation-v1.npy")
    assert workload.training_fixture.shape == (32, 1_025)
    assert workload.validation_fixture.shape == (8, 1_025)
    assert workload.training_fixture.dtype == "int32"
    assert workload.validation_fixture.dtype == "int32"
    assert workload.training_fixture.source_identity != (
        workload.validation_fixture.source_identity
    )
    assert workload.training_fixture.semantic_identity != (
        workload.validation_fixture.semantic_identity
    )
    assert workload.ordered_batches[:3] == ((0,), (1,), (2,))
    assert workload.ordered_batches[31:34] == ((31,), (0,), (1,))
    assert workload.evaluation_row_indices == tuple(range(8))
    assert tuple(leaf.path for leaf in workload.parameter_leaves) == (
        workload.parameter_leaf_names
    )
    assert all(
        leaf.value_count == int(np.prod(leaf.shape))
        and leaf.initial_bf16_identity.startswith("sha256:")
        and leaf.initial_fp32_identity.startswith("sha256:")
        for leaf in workload.parameter_leaves
    )
    assert all(
        isinstance(leaf, ParameterLeafSpec) for leaf in workload.parameter_leaves
    )
    assert workload.identity == workload.recompute_identity()
    assert (
        PretrainingQualityWorkload.from_dict(workload.to_dict()).to_dict()
        == workload.to_dict()
    )

    for fixture in (workload.training_fixture, workload.validation_fixture):
        path = ROOT / fixture.logical_path
        assert path.stat().st_size == fixture.byte_size
        rows = np.load(path, allow_pickle=False)
        assert rows.dtype == np.dtype("<i4")
        assert tuple(rows.shape) == fixture.shape
        assert int(rows.min()) >= 0
        assert int(rows.max()) < ModelConfig().vocab_size


def test_harness_identity_binds_quality_and_publication_implementation_in_order(
    tmp_path,
):
    expected = hashlib.sha256()
    assert all((ROOT / path).is_file() for path in quality_module.HARNESS_COMPONENTS)
    for index, relative in enumerate(quality_module.HARNESS_COMPONENTS):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = f"component-{index}\n".encode()
        path.write_bytes(payload)
        expected.update(payload)
    assert harness_content_identity(tmp_path) == f"sha256:{expected.hexdigest()}"


def test_workload_binds_an_exact_closed_base_training_import_surface(
    canonical_workload,
    tmp_path,
):
    expected_components = tuple(
        path.as_posix() for path in production_dependency_components(ROOT)
    )
    assert canonical_workload.production_dependency_components == expected_components
    assert "v2/src/sml/training/pretrain.py" in expected_components
    assert "v2/src/sml/model/language_model.py" in expected_components
    assert "v2/src/sml/artifacts/checkpoint.py" in expected_components
    assert "v2/src/sml/__init__.py" in expected_components
    assert "v2/src/sml/cli.py" not in expected_components
    quality_module._validate_production_dependency_closure(
        ROOT,
        tuple(Path(path) for path in expected_components),
    )
    omitted = tuple(
        Path(path)
        for path in expected_components
        if path != "v2/src/sml/model/layers.py"
    )
    with pytest.raises(ValueError, match="closure|omitted|incomplete"):
        quality_module._validate_production_dependency_closure(ROOT, omitted)
    assert (
        tuple(path.as_posix() for path in production_dependency_components(ROOT))
        == expected_components
    )
    assert canonical_workload.production_dependency_identity == (
        production_dependency_content_identity(ROOT)
    )

    for relative in (
        *quality_module.HARNESS_COMPONENTS,
        *(Path(path) for path in expected_components),
    ):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    original_harness = harness_content_identity(tmp_path)
    dependency = tmp_path / "v2/src/sml/artifacts/manifest.py"
    dependency.write_bytes(dependency.read_bytes() + b"\n# provenance mutation\n")

    assert harness_content_identity(tmp_path) == original_harness
    assert production_dependency_content_identity(tmp_path) != (
        canonical_workload.production_dependency_identity
    )


def test_production_provenance_ignores_an_unrelated_future_sml_module(tmp_path):
    components = production_dependency_components(ROOT)
    for relative in (
        *quality_module.HARNESS_COMPONENTS,
        *quality_module.PRODUCTION_DEPENDENCY_FIXED_COMPONENTS,
        *components,
    ):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)

    before_components = production_dependency_components(tmp_path)
    before_identity = production_dependency_content_identity(tmp_path)
    unrelated = tmp_path / "v2/src/sml/future_workflow.py"
    unrelated.write_text("UNRELATED_FUTURE_MODULE = True\n")

    assert production_dependency_components(tmp_path) == before_components
    assert production_dependency_content_identity(tmp_path) == before_identity


def test_recording_command_and_session_identity_are_checkout_portable(tmp_path):
    left_root = tmp_path / "left"
    right_root = tmp_path / "a-much-longer-relocated-checkout"
    left_destinations = quality_module._canonical_evidence_destinations(
        left_root,
        left_root / quality_module.RECORD_MANIFEST_PATH,
        left_root / quality_module.RECORD_RAW_PATH,
        left_root / quality_module.RECORD_REPORT_PATH,
    )
    right_destinations = quality_module._canonical_evidence_destinations(
        right_root,
        right_root / quality_module.RECORD_MANIFEST_PATH,
        right_root / quality_module.RECORD_RAW_PATH,
        right_root / quality_module.RECORD_REPORT_PATH,
    )

    left = quality_module._recording_command_document(left_root, left_destinations)
    right = quality_module._recording_command_document(right_root, right_destinations)

    assert left == right
    assert "resolved_destinations" not in left
    assert quality_module._recording_session_identity(
        "a" * 40,
        "sha256:" + "b" * 64,
        left,
    ) == quality_module._recording_session_identity(
        "a" * 40,
        "sha256:" + "b" * 64,
        right,
    )


def test_workload_rejects_a_validation_row_copied_from_training(tmp_path):
    for relative in (
        Path("v2/benchmarks/quality.py"),
        Path("v2/tests/unit/benchmarks/pretraining_quality/test_workload.py"),
        Path("v2/benchmarks/fixtures/pretraining-quality-train-v1.npy"),
        Path("v2/benchmarks/fixtures/pretraining-quality-validation-v1.npy"),
    ):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    training = np.load(
        tmp_path / "v2/benchmarks/fixtures/pretraining-quality-train-v1.npy",
        allow_pickle=False,
    )
    validation_path = (
        tmp_path / "v2/benchmarks/fixtures/pretraining-quality-validation-v1.npy"
    )
    validation = np.load(validation_path, allow_pickle=False)
    validation[0] = training[0]
    np.save(validation_path, validation, allow_pickle=False)

    with pytest.raises(ValueError, match="source-disjoint"):
        build_pretraining_quality_workload(tmp_path)


@pytest.mark.parametrize("unsupported_version", [0, 2, 4])
def test_workload_accepts_only_current_version(canonical_workload, unsupported_version):
    raw = canonical_workload.to_dict()
    raw["version"] = unsupported_version
    with pytest.raises(ValueError, match="unsupported"):
        type(canonical_workload).from_dict(raw)
