from __future__ import annotations

import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from v2.benchmarks import swag_quality
from v2.benchmarks.swag_quality import (
    CANONICAL_STEPS,
    SwagQualityRecord,
    SwagQualityWorkload,
    decide_swag_quality,
    validate_swag_quality_records,
)
from v2.benchmarks.workload import structured_identity

from . import ROOT


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

    with pytest.raises(ValueError, match="real-example"):
        validate_swag_quality_records(canonical_workload, records)

    passing = (
        _record(
            canonical_workload,
            "candidate",
            validation_loss=1.005,
            accuracy=0.80,
            examples=canonical_workload.validation_fixture.example_count,
        ),
        _record(
            canonical_workload,
            "oracle",
            validation_loss=1.0,
            accuracy=0.80,
            examples=canonical_workload.validation_fixture.example_count,
        ),
    )
    report = validate_swag_quality_records(canonical_workload, passing)
    assert (
        report.candidate_examples
        == report.oracle_examples
        == canonical_workload.validation_fixture.example_count
    )
    assert decide_swag_quality(report) == "pass"

    with pytest.raises(ValueError, match="exactly two"):
        validate_swag_quality_records(canonical_workload, passing[:1])
    with pytest.raises(ValueError, match="exactly two"):
        validate_swag_quality_records(canonical_workload, (*passing, passing[0]))

    changed_base = _record(
        canonical_workload,
        "candidate",
        validation_loss=1.005,
        accuracy=0.80,
        examples=canonical_workload.validation_fixture.example_count,
        frozen_base_identity="sha256:" + "f" * 64,
    )
    with pytest.raises(ValueError, match="base"):
        validate_swag_quality_records(canonical_workload, (changed_base, passing[1]))

    mismatched = _record(
        canonical_workload,
        "oracle",
        validation_loss=1.0,
        accuracy=0.80,
        examples=canonical_workload.validation_fixture.example_count - 1,
    )
    with pytest.raises(ValueError, match="real-example"):
        validate_swag_quality_records(canonical_workload, (passing[0], mismatched))

    nonfinite = _record(
        canonical_workload,
        "candidate",
        validation_loss=1.005,
        accuracy=0.80,
        examples=canonical_workload.validation_fixture.example_count,
        finite=False,
    )
    failed = validate_swag_quality_records(canonical_workload, (nonfinite, passing[1]))
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
        ROOT / swag_quality.RECORD_MANIFEST_PATH,
        ROOT / swag_quality.RECORD_RAW_PATH,
        ROOT / swag_quality.RECORD_REPORT_PATH,
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
            ROOT / swag_quality.RECORD_MANIFEST_PATH,
            ROOT / swag_quality.RECORD_RAW_PATH,
            ROOT / "v2/benchmarks/results/forged.json",
        )


def test_swag_recording_accepts_only_current_complete_destination_triple():
    destinations = swag_quality._canonical_evidence_destinations(
        ROOT,
        ROOT / swag_quality.RECORD_MANIFEST_PATH,
        ROOT / swag_quality.RECORD_RAW_PATH,
        ROOT / swag_quality.RECORD_REPORT_PATH,
    )
    command = swag_quality._recording_command_document(ROOT, destinations)
    assert command["destinations"] == {
        "manifest": swag_quality.RECORD_MANIFEST_PATH.as_posix(),
        "raw_output": swag_quality.RECORD_RAW_PATH.as_posix(),
        "report": swag_quality.RECORD_REPORT_PATH.as_posix(),
    }


def _recompute_manifest_identity(manifest: dict[str, object]) -> dict[str, object]:
    body = {key: value for key, value in manifest.items() if key != "identity"}
    return {
        **body,
        "identity": structured_identity("sml-swag-quality-manifest-v1", body),
    }


def test_standalone_validation_rejects_over_budget_and_nonpositive_phases(
    canonical_workload,
):
    destinations = swag_quality._canonical_evidence_destinations(
        ROOT,
        ROOT / swag_quality.RECORD_MANIFEST_PATH,
        ROOT / swag_quality.RECORD_RAW_PATH,
        ROOT / swag_quality.RECORD_REPORT_PATH,
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

    over_budget = _recompute_manifest_identity(
        {
            **manifest,
            "phase_wall_time_seconds": {
                "setup": 1.0,
                "candidate": 10_000.0,
                "oracle": 5_000.0,
                "validation_serialization": 1.0,
            },
            "measured_wall_time_seconds": 15_002.0,
        }
    )
    with pytest.raises(ValueError, match="budget"):
        swag_quality._validate_manifest_fields(over_budget, canonical_workload, command)

    nonpositive = _recompute_manifest_identity(
        {
            **manifest,
            "phase_wall_time_seconds": {
                "setup": 0.0,
                "candidate": 3.0,
                "oracle": 4.0,
                "validation_serialization": 2.0,
            },
            "measured_wall_time_seconds": 9.0,
        }
    )
    with pytest.raises(ValueError, match="phase"):
        swag_quality._validate_manifest_fields(nonpositive, canonical_workload, command)


@pytest.fixture(scope="module")
def current_recorded_evidence(tmp_path_factory, canonical_workload):
    """Build validator inputs from current source and bounded synthetic records."""
    root = tmp_path_factory.mktemp("swag-quality-current-evidence")
    workload = canonical_workload
    copied = {
        *swag_quality.HARNESS_COMPONENTS,
        *(Path(path) for path in workload.production_dependency_components),
        swag_quality.TRAINING_FIXTURE,
        swag_quality.VALIDATION_FIXTURE,
    }
    for relative in copied:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)

    def git(*arguments):
        return subprocess.run(
            ["git", *arguments], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()

    git("init", "--quiet")
    git("config", "user.name", "Quality Evidence Test")
    git("config", "user.email", "quality-evidence@example.invalid")
    git("add", ".")
    git("commit", "--quiet", "-m", "current quality source")
    commit = git("rev-parse", "HEAD")
    destinations = swag_quality._canonical_evidence_destinations(
        root,
        root / swag_quality.RECORD_MANIFEST_PATH,
        root / swag_quality.RECORD_RAW_PATH,
        root / swag_quality.RECORD_REPORT_PATH,
    )
    command = swag_quality._recording_command_document(root, destinations)
    records = tuple(
        _record(
            workload,
            runtime,
            validation_loss=1.005 if runtime == "candidate" else 1.0,
            accuracy=0.75,
            examples=workload.validation_fixture.example_count,
        )
        for runtime in ("candidate", "oracle")
    )
    documents = [record.to_dict() for record in records]
    raw = b"".join(
        swag_quality.canonical_json_bytes(document) + b"\n" for document in documents
    )
    raw_identity = swag_quality.structured_identity(
        "sml-swag-quality-raw-v1", documents
    )
    report = swag_quality._report_document(
        workload.identity,
        raw_identity,
        validate_swag_quality_records(workload, records),
    )
    report_bytes = swag_quality._canonical_json_file_bytes(report)
    manifest = swag_quality._manifest_document(
        workload=workload,
        source_commit=commit,
        recording_command=command,
        phase_times={
            "setup": 1.0,
            "candidate": 1.0,
            "oracle": 1.0,
            "validation_serialization": 1.0,
        },
        peak_memory=1,
        raw_identity=raw_identity,
        raw_file_identity=swag_quality._payload_identity(raw),
        raw_bytes=len(raw),
        report_identity=report["identity"],
        report_file_identity=swag_quality._payload_identity(report_bytes),
        report_bytes=len(report_bytes),
        recording_session_identity=swag_quality._recording_session_identity(
            commit, workload.identity, command
        ),
    )
    for path in (destinations.manifest, destinations.raw_output, destinations.report):
        path.parent.mkdir(parents=True, exist_ok=True)
    destinations.manifest.write_bytes(swag_quality._canonical_json_file_bytes(manifest))
    destinations.raw_output.write_bytes(raw)
    destinations.report.write_bytes(report_bytes)
    return root, destinations, manifest, workload


def test_current_standalone_evidence_validation(current_recorded_evidence, monkeypatch):
    root, destinations, manifest, _workload = current_recorded_evidence
    assert manifest["recording_command"] == swag_quality._recording_command_document(
        root, destinations
    )
    monkeypatch.setattr(swag_quality, "_root", lambda: root)
    assert (
        swag_quality._validate(
            SimpleNamespace(
                manifest=destinations.manifest,
                raw_input=destinations.raw_output,
                report=destinations.report,
            )
        )
        == 0
    )


@pytest.mark.parametrize("changed_input", [None, "production", "harness", "fixture"])
def test_record_reuses_only_current_workload_evidence(
    current_recorded_evidence, tmp_path, monkeypatch, changed_input
):
    recorded_root, _destinations, _manifest, _workload = current_recorded_evidence
    root = tmp_path / "recording"
    shutil.copytree(recorded_root, root)
    destinations = swag_quality._canonical_evidence_destinations(
        root,
        root / swag_quality.RECORD_MANIFEST_PATH,
        root / swag_quality.RECORD_RAW_PATH,
        root / swag_quality.RECORD_REPORT_PATH,
    )
    original_payloads = {
        path: path.read_bytes() for _name, path in destinations.ordered()
    }
    if changed_input is not None:
        relative = {
            "production": Path("v2/src/sml/training/swag.py"),
            "harness": Path("v2/benchmarks/swag_quality.py"),
            "fixture": swag_quality.TRAINING_FIXTURE,
        }[changed_input]
        source = root / relative
        if changed_input == "fixture":
            arrays = swag_quality._load_encoded_arrays(source)
            arrays["labels"][0] = (arrays["labels"][0] + 1) % 4
            swag_quality._write_npz(source, arrays)
        else:
            source.write_bytes(source.read_bytes() + b"\n# changed recording input\n")
        for arguments in (
            ("add", str(source)),
            ("commit", "--quiet", "-m", "change recording input"),
        ):
            subprocess.run(
                ["git", *arguments], cwd=root, check=True, capture_output=True
            )

    def forbid_quality_execution(*_args, **_kwargs):
        raise AssertionError(
            "existing evidence must be checked before quality execution"
        )

    monkeypatch.setattr(swag_quality, "_root", lambda: root)
    monkeypatch.setattr(swag_quality, "_run_runtime", forbid_quality_execution)
    monkeypatch.setattr(
        swag_quality, "_verified_source_snapshot", forbid_quality_execution
    )
    args = SimpleNamespace(
        manifest=destinations.manifest,
        raw_output=destinations.raw_output,
        output=destinations.report,
    )
    if changed_input is not None:
        with pytest.raises(ValueError, match="recording .* changed"):
            swag_quality._record(args)
    else:
        assert swag_quality._record(args) == 0

    assert all(
        path.read_bytes() == payload for path, payload in original_payloads.items()
    )
    assert (
        swag_quality._validate(
            SimpleNamespace(
                manifest=destinations.manifest,
                raw_input=destinations.raw_output,
                report=destinations.report,
            )
        )
        == 0
    )


def test_recorded_validator_rejects_required_component_omission(
    current_recorded_evidence,
):
    root, _destinations, manifest, workload = current_recorded_evidence
    retained = tuple(
        path
        for path in workload.production_dependency_components
        if path != "v2/src/sml/model/layers.py"
    )
    assert len(retained) + 1 == len(workload.production_dependency_components)
    identity = swag_quality._production_dependency_identity(
        tuple(Path(path) for path in retained),
        lambda path: swag_quality._git_bytes(
            root, "show", f"{manifest['source_commit']}:{path.as_posix()}"
        ),
    )
    tampered = replace(
        workload,
        production_dependency_components=retained,
        production_dependency_identity=identity,
    )
    tampered = replace(tampered, identity=tampered.recompute_identity())
    with pytest.raises(ValueError, match="component set changed"):
        swag_quality._validate_harness_commit(root, manifest["source_commit"], tampered)
