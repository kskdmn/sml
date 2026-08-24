from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from v2.benchmarks import swag_quality
from v2.benchmarks.swag_quality import (
    CANONICAL_STEPS,
    SwagQualityRecord,
    SwagQualityReport,
    SwagQualityWorkload,
    build_swag_quality_workload,
    decide_swag_quality,
    harness_content_identity,
    validate_swag_quality_records,
)

ROOT = Path(__file__).parents[3]


def test_swag_quality_gate_enforces_loss_accuracy_and_example_count():
    passing = SwagQualityReport(
        candidate_validation_loss=1.005,
        oracle_validation_loss=1.0,
        candidate_accuracy=0.795,
        oracle_accuracy=0.80,
        candidate_examples=512,
        oracle_examples=512,
        candidate_finite=True,
        oracle_finite=True,
    )
    assert decide_swag_quality(passing) == "pass"
    assert (
        decide_swag_quality(
            replace(passing, candidate_validation_loss=1.011),
        )
        == "fail"
    )
    assert (
        decide_swag_quality(
            replace(passing, candidate_accuracy=0.789),
        )
        == "fail"
    )
    assert (
        decide_swag_quality(
            replace(passing, candidate_examples=511),
        )
        == "fail"
    )


@pytest.mark.parametrize(
    ("change", "value"),
    [
        ("candidate_finite", False),
        ("oracle_finite", False),
        ("candidate_validation_loss", float("nan")),
        ("oracle_validation_loss", float("inf")),
        ("candidate_accuracy", float("nan")),
        ("oracle_examples", 511),
    ],
)
def test_swag_quality_gate_fails_closed_before_thresholds(change, value):
    passing = SwagQualityReport(
        candidate_validation_loss=1.005,
        oracle_validation_loss=1.0,
        candidate_accuracy=0.795,
        oracle_accuracy=0.80,
        candidate_examples=512,
        oracle_examples=512,
        candidate_finite=True,
        oracle_finite=True,
    )

    assert decide_swag_quality(replace(passing, **{change: value})) == "fail"


@pytest.fixture(scope="module")
def canonical_workload() -> SwagQualityWorkload:
    return build_swag_quality_workload(ROOT)


def test_harness_identity_hashes_only_the_two_reviewed_files_in_order():
    expected = hashlib.sha256()
    for relative in (
        Path("v2/benchmarks/swag_quality.py"),
        Path("v2/tests/unit/test_swag_quality.py"),
    ):
        expected.update((ROOT / relative).read_bytes())

    assert harness_content_identity(ROOT) == f"sha256:{expected.hexdigest()}"


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
        Path("v2/tests/unit/test_swag_quality.py"),
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


def _record(
    workload: SwagQualityWorkload,
    runtime: str,
    *,
    validation_loss: float,
    accuracy: float,
    examples: int,
    finite: bool = True,
    frozen_base_identity: str | None = None,
    train_loss: float = 1.0,
) -> SwagQualityRecord:
    record = SwagQualityRecord(
        kind="swag-quality-record",
        version=1,
        identity="sha256:" + "0" * 64,
        runtime=runtime,
        step=CANONICAL_STEPS,
        workload_identity=workload.identity,
        train_loss=train_loss,
        validation_loss=validation_loss,
        validation_accuracy=accuracy,
        real_example_count=examples,
        finite=finite,
        frozen_base_identity=(
            workload.frozen_bf16_base_identity
            if frozen_base_identity is None
            else frozen_base_identity
        ),
        adapter_identity="sha256:" + ("a" if runtime == "candidate" else "b") * 64,
    )
    return replace(record, identity=record.recompute_identity())


def test_raw_validation_rejects_missing_extra_changed_base_and_counts(
    canonical_workload,
):
    records = (
        _record(
            canonical_workload,
            "candidate",
            validation_loss=1.005,
            accuracy=0.80,
            examples=8,
        ),
        _record(
            canonical_workload,
            "oracle",
            validation_loss=1.0,
            accuracy=0.80,
            examples=8,
        ),
    )

    report = validate_swag_quality_records(canonical_workload, records)
    assert report.candidate_examples == report.oracle_examples == 8
    assert decide_swag_quality(report) == "pass"

    with pytest.raises(ValueError, match="exactly two"):
        validate_swag_quality_records(canonical_workload, records[:1])
    with pytest.raises(ValueError, match="exactly two"):
        validate_swag_quality_records(canonical_workload, (*records, records[0]))

    changed_base = _record(
        canonical_workload,
        "candidate",
        validation_loss=1.005,
        accuracy=0.80,
        examples=8,
        frozen_base_identity="sha256:" + "f" * 64,
    )
    with pytest.raises(ValueError, match="base"):
        validate_swag_quality_records(canonical_workload, (changed_base, records[1]))

    mismatched = _record(
        canonical_workload, "oracle", validation_loss=1.0, accuracy=0.80, examples=7
    )
    with pytest.raises(ValueError, match="real-example"):
        validate_swag_quality_records(canonical_workload, (records[0], mismatched))

    nonfinite = _record(
        canonical_workload,
        "candidate",
        validation_loss=1.005,
        accuracy=0.80,
        examples=8,
        finite=False,
    )
    failed = validate_swag_quality_records(canonical_workload, (nonfinite, records[1]))
    assert decide_swag_quality(failed) == "fail"


def test_public_record_accepts_only_exactly_256_steps():
    parser = swag_quality._build_parser()
    common = [
        "--manifest",
        "manifest.json",
        "--raw-output",
        "raw.jsonl",
        "--output",
        "report.json",
    ]

    assert parser.parse_args(["record", "--steps", "256", *common]).steps == 256
    for invalid in ("1", "255", "257", "1000"):
        with pytest.raises(SystemExit):
            parser.parse_args(["record", "--steps", invalid, *common])


def test_manifest_fields_and_output_paths_fail_closed(canonical_workload):
    destinations = swag_quality._canonical_evidence_destinations(
        ROOT,
        ROOT / "v2/benchmarks/manifests/swag-quality-v1.json",
        ROOT / "v2/benchmarks/results/swag-quality-v1.jsonl",
        ROOT / "v2/benchmarks/results/swag-quality-v1.json",
    )
    command = swag_quality._recording_command_document(ROOT, destinations)
    session_identity = swag_quality._recording_session_identity(
        "a" * 40, canonical_workload.identity, command
    )
    manifest = swag_quality._manifest_document(
        workload=canonical_workload,
        source_commit="a" * 40,
        recording_command=command,
        phase_times={
            "setup": 1.0,
            "candidate": 3.0,
            "oracle": 4.0,
            "validation_serialization": 2.0,
        },
        peak_memory=123,
        raw_identity="sha256:" + "b" * 64,
        raw_file_identity="sha256:" + "c" * 64,
        raw_bytes=1_000,
        report_identity="sha256:" + "d" * 64,
        report_file_identity="sha256:" + "e" * 64,
        report_bytes=500,
        recording_session_identity=session_identity,
    )

    assert swag_quality._validate_manifest_fields(
        manifest, canonical_workload, command
    ) == dict(manifest)
    assert manifest["source_commit"] == manifest["harness_commit"] == "a" * 40
    assert manifest["optimizer_steps"] == 256
    assert manifest["record_count"] == 2

    with pytest.raises(ValueError, match="canonical evidence destinations"):
        swag_quality._canonical_evidence_destinations(
            ROOT,
            ROOT / "v2/benchmarks/manifests/swag-quality-v1.json",
            ROOT / "v2/benchmarks/results/swag-quality-v1.jsonl",
            ROOT / "v2/benchmarks/results/forged.json",
        )
