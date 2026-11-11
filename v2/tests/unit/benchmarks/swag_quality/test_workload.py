from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from v2.benchmarks import swag_quality
from v2.benchmarks.swag_quality import (
    CANONICAL_STEPS,
    SwagQualityReport,
    SwagQualityWorkload,
    build_swag_quality_workload,
    decide_swag_quality,
    harness_content_identity,
)

from . import ROOT


@pytest.mark.parametrize(
    "changes,decision",
    [
        ({}, "pass"),
        ({"candidate_validation_loss": 1.011}, "fail"),
        ({"candidate_accuracy": 0.789}, "fail"),
        ({"candidate_examples": 511}, "fail"),
        ({"candidate_finite": False}, "fail"),
        ({"oracle_finite": False}, "fail"),
        ({"candidate_validation_loss": float("nan")}, "fail"),
        ({"oracle_validation_loss": float("inf")}, "fail"),
        ({"candidate_accuracy": float("nan")}, "fail"),
        ({"oracle_examples": 511}, "fail"),
    ],
)
def test_swag_quality_gate_requires_valid_inputs_and_passing_thresholds(
    changes, decision
):
    report = SwagQualityReport(
        candidate_validation_loss=1.005,
        oracle_validation_loss=1.0,
        candidate_accuracy=0.795,
        oracle_accuracy=0.80,
        candidate_examples=512,
        oracle_examples=512,
        candidate_finite=True,
        oracle_finite=True,
    )
    assert decide_swag_quality(replace(report, **changes)) == decision


def test_workload_pins_256_steps_disjoint_encoded_examples_and_identities(
    canonical_workload,
):
    workload = canonical_workload

    assert CANONICAL_STEPS == 256
    assert workload.optimizer_steps == 256
    assert len(workload.ordered_batches) == 256
    assert workload.score_policy == "fp32-mean-continuation-including-eos-v1"
    assert workload.training_fixture.source_identity != (
        workload.validation_fixture.source_identity
    )
    assert workload.training_fixture.semantic_identity != (
        workload.validation_fixture.semantic_identity
    )
    assert workload.frozen_bf16_base_identity.startswith("sha256:")
    assert workload.fp32_master_identity.startswith("sha256:")
    assert workload.frozen_bf16_base_identity != workload.fp32_master_identity
    assert workload.initial_fp32_adapter_identity.startswith("sha256:")
    assert workload.identity == workload.recompute_identity()
    assert (
        SwagQualityWorkload.from_dict(workload.to_dict()).to_dict()
        == workload.to_dict()
    )

    for fixture in (workload.training_fixture, workload.validation_fixture):
        path = ROOT / fixture.logical_path
        assert path.stat().st_size == fixture.byte_size
        arrays = np.load(path, allow_pickle=False)
        input_ids = arrays["input_ids"]
        assert input_ids.dtype == np.dtype("<i4")
        assert input_ids.ndim == 3
        assert input_ids.shape[1] == 4
        assert int(input_ids.min()) >= 0


def test_workload_rejects_a_validation_example_copied_from_training(
    canonical_workload, tmp_path
):
    for relative in (
        Path("v2/benchmarks/swag_quality.py"),
        Path("v2/tests/unit/benchmarks/swag_quality/test_workload.py"),
        Path(canonical_workload.training_fixture.logical_path),
        Path(canonical_workload.validation_fixture.logical_path),
        *(Path(path) for path in canonical_workload.production_dependency_components),
    ):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        source = ROOT / relative
        if source.is_file():
            destination.write_bytes(source.read_bytes())

    training = np.load(
        tmp_path / canonical_workload.training_fixture.logical_path,
        allow_pickle=False,
    )
    validation_path = tmp_path / canonical_workload.validation_fixture.logical_path
    validation = dict(np.load(validation_path, allow_pickle=False))
    validation["input_ids"][0] = np.array(training["input_ids"][0])
    validation["valid_token_mask"][0] = np.array(training["valid_token_mask"][0])
    validation["score_mask"][0] = np.array(training["score_mask"][0])
    validation["labels"][0] = np.array(training["labels"][0])
    np.savez(validation_path, **validation)

    with pytest.raises(ValueError, match="source-disjoint"):
        build_swag_quality_workload(tmp_path)


@pytest.mark.parametrize("unsupported_version", [0, 1, 3])
def test_workload_accepts_only_current_version(canonical_workload, unsupported_version):
    raw = canonical_workload.to_dict()
    raw["version"] = unsupported_version
    with pytest.raises(ValueError, match="unsupported"):
        type(canonical_workload).from_dict(raw)


def test_harness_identity_includes_publication_dependencies_in_order(tmp_path):
    expected = hashlib.sha256()
    assert all((ROOT / path).is_file() for path in swag_quality.HARNESS_COMPONENTS)
    for index, relative in enumerate(swag_quality.HARNESS_COMPONENTS):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = f"component-{index}\n".encode()
        path.write_bytes(payload)
        expected.update(payload)
    assert harness_content_identity(tmp_path) == f"sha256:{expected.hexdigest()}"
