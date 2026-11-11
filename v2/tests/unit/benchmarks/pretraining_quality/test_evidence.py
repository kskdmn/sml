from __future__ import annotations

import hashlib
import shutil
import subprocess
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import v2.benchmarks.quality as quality_module
from v2.benchmarks.quality import (
    CHECKPOINT_STEPS,
    ParameterUpdateStatistics,
    PretrainingQualityCheckpoint,
    PretrainingQualityReport,
    production_dependency_content_identity,
    real_work_identity,
    validate_pretraining_quality_records,
)

from . import ROOT


@contextmanager
def _temporary_current_tree_bytes(path: Path, payload: bytes | None):
    original = path.read_bytes()
    try:
        if payload is None:
            path.unlink()
        else:
            path.write_bytes(payload)
        yield
    finally:
        path.write_bytes(original)


@pytest.mark.parametrize(
    "source_mutation",
    ["content", "executable_mode"],
)
def test_re_signed_descendant_cannot_reuse_evidence_after_artifact_source_change(
    canonical_workload,
    tmp_path,
    source_mutation,
):
    changed_source = Path("v2/src/sml/artifacts/manifest.py")
    fixtures = (
        Path(canonical_workload.training_fixture.logical_path),
        Path(canonical_workload.validation_fixture.logical_path),
    )
    copied = {
        *quality_module.HARNESS_COMPONENTS,
        *(Path(path) for path in canonical_workload.production_dependency_components),
        *fixtures,
        changed_source,
    }
    for relative in copied:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)

    def git(*arguments):
        return subprocess.run(
            ["git", *arguments],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    git("init", "--quiet")
    git("config", "user.name", "Quality Provenance Test")
    git("config", "user.email", "quality-provenance@example.invalid")
    git("add", ".")
    git("commit", "--quiet", "-m", "quality source")
    changed_path = tmp_path / changed_source
    if source_mutation == "content":
        changed_path.write_bytes(
            changed_path.read_bytes() + b"\n# descendant mutation\n"
        )
    else:
        changed_path.chmod(changed_path.stat().st_mode | 0o111)
    git("add", changed_source.as_posix())
    git("commit", "--quiet", "-m", "change executed artifact source")
    descendant = git("rev-parse", "HEAD")

    destinations = quality_module._canonical_evidence_destinations(
        tmp_path,
        tmp_path / quality_module.RECORD_MANIFEST_PATH,
        tmp_path / quality_module.RECORD_RAW_PATH,
        tmp_path / quality_module.RECORD_REPORT_PATH,
    )
    command = quality_module._recording_command_document(tmp_path, destinations)
    session_identity = quality_module._recording_session_identity(
        descendant, canonical_workload.identity, command
    )
    re_signed = quality_module._manifest_document(
        workload=canonical_workload,
        source_commit=descendant,
        recording_command=command,
        phase_times={
            "setup": 1.0,
            "candidate": 2.0,
            "oracle": 3.0,
            "validation_serialization": 1.0,
        },
        peak_memory=1,
        raw_identity="sha256:" + "1" * 64,
        raw_file_identity="sha256:" + "2" * 64,
        raw_bytes=100,
        report_identity="sha256:" + "3" * 64,
        report_file_identity="sha256:" + "4" * 64,
        report_bytes=100,
        recording_session_identity=session_identity,
    )

    with pytest.raises(ValueError, match="production import closure"):
        quality_module._validate_manifest(
            re_signed,
            canonical_workload,
            tmp_path,
            command,
        )


def _signed_quality_manifest(tmp_path, canonical_workload, source_commit):
    destinations = quality_module._canonical_evidence_destinations(
        tmp_path,
        tmp_path / quality_module.RECORD_MANIFEST_PATH,
        tmp_path / quality_module.RECORD_RAW_PATH,
        tmp_path / quality_module.RECORD_REPORT_PATH,
    )
    command = quality_module._recording_command_document(tmp_path, destinations)
    session_identity = quality_module._recording_session_identity(
        source_commit, canonical_workload.identity, command
    )
    return (
        quality_module._manifest_document(
            workload=canonical_workload,
            source_commit=source_commit,
            recording_command=command,
            phase_times={
                "setup": 1.0,
                "candidate": 2.0,
                "oracle": 3.0,
                "validation_serialization": 1.0,
            },
            peak_memory=1,
            raw_identity="sha256:" + "1" * 64,
            raw_file_identity="sha256:" + "2" * 64,
            raw_bytes=100,
            report_identity="sha256:" + "3" * 64,
            report_file_identity="sha256:" + "4" * 64,
            report_bytes=100,
            recording_session_identity=session_identity,
        ),
        command,
    )


@pytest.mark.parametrize(
    ("mismatch", "match"),
    [
        ("harness", "harness"),
        ("fixture", "fixture"),
        ("embedded_identity", "identity"),
    ],
)
def test_validate_rejects_recorded_source_bytes_that_do_not_match_the_manifest(
    canonical_workload,
    tmp_path,
    mismatch,
    match,
):
    fixtures = (
        Path(canonical_workload.training_fixture.logical_path),
        Path(canonical_workload.validation_fixture.logical_path),
    )
    copied = {
        *quality_module.HARNESS_COMPONENTS,
        *(Path(path) for path in canonical_workload.production_dependency_components),
        *fixtures,
    }
    for relative in copied:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)

    def git(*arguments):
        return subprocess.run(
            ["git", *arguments],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    git("init", "--quiet")
    git("config", "user.name", "Quality Provenance Test")
    git("config", "user.email", "quality-provenance@example.invalid")
    git("add", ".")
    git("commit", "--quiet", "-m", "quality source")
    recorded = git("rev-parse", "HEAD")
    workload = canonical_workload
    if mismatch == "harness":
        harness = tmp_path / quality_module.HARNESS_COMPONENTS[0]
        harness.write_bytes(harness.read_bytes() + b"\n# harness mutation\n")
        git("add", quality_module.HARNESS_COMPONENTS[0].as_posix())
        git("commit", "--quiet", "-m", "change recorded harness bytes")
        recorded = git("rev-parse", "HEAD")
    elif mismatch == "fixture":
        fixture = tmp_path / fixtures[0]
        fixture.write_bytes(fixture.read_bytes() + b"\x00")
        git("add", fixtures[0].as_posix())
        git("commit", "--quiet", "-m", "change recorded fixture bytes")
        recorded = git("rev-parse", "HEAD")
    else:
        workload = replace(canonical_workload, identity="sha256:" + "f" * 64)

    re_signed, command = _signed_quality_manifest(tmp_path, workload, recorded)

    with pytest.raises(ValueError, match=match):
        quality_module._validate_manifest(
            re_signed,
            workload,
            tmp_path,
            command,
        )


def _fake_identity(*parts: object) -> str:
    digest = hashlib.sha256("/".join(map(str, parts)).encode()).hexdigest()
    return f"sha256:{digest}"


def _statistics(workload, runtime, step, survived):
    statistics = []
    survival_path = workload.parameter_leaves[0].path
    for leaf in workload.parameter_leaves:
        if step == 0:
            before_master = after_master = leaf.initial_fp32_identity
            before_bf16 = after_bf16 = leaf.initial_bf16_identity
            before_working = after_working = (
                before_bf16 if runtime == "candidate" else before_master
            )
            nonzero = sub_ulp = survived_count = changed = 0
            first_survival = None
            minimum = mean = maximum = None
        else:
            before_master = _fake_identity(runtime, step, leaf.path, "before-master")
            after_master = _fake_identity(runtime, step, leaf.path, "after-master")
            before_bf16 = _fake_identity(runtime, step, leaf.path, "before-bf16")
            after_bf16 = _fake_identity(runtime, step, leaf.path, "after-bf16")
            before_working = before_bf16 if runtime == "candidate" else before_master
            after_working = after_bf16 if runtime == "candidate" else after_master
            is_survival_leaf = survived and leaf.path == survival_path
            nonzero = 1
            sub_ulp = int(is_survival_leaf)
            survived_count = int(is_survival_leaf)
            changed = 1
            first_survival = 1 if is_survival_leaf else None
            minimum = mean = maximum = 0.5 if is_survival_leaf else 1.0
        statistics.append(
            ParameterUpdateStatistics(
                path=leaf.path,
                shape=leaf.shape,
                value_count=leaf.value_count,
                before_master_identity=before_master,
                after_master_identity=after_master,
                before_working_identity=before_working,
                after_working_identity=after_working,
                before_bf16_working_identity=before_bf16,
                after_bf16_working_identity=after_bf16,
                nonzero_update_count=nonzero,
                sub_bf16_ulp_update_count=sub_ulp,
                survived_sub_bf16_ulp_count=survived_count,
                changed_bf16_working_count=changed,
                first_sub_bf16_survival_step=first_survival,
                minimum_update_to_bf16_ulp=minimum,
                mean_update_to_bf16_ulp=mean,
                maximum_update_to_bf16_ulp=maximum,
            )
        )
    return tuple(statistics)


def _checkpoint(workload, runtime, step, *, validation_nll, survived=True):
    statistics = _statistics(workload, runtime, step, survived)
    changed_fraction = sum(
        item.changed_bf16_working_count for item in statistics
    ) / sum(item.value_count for item in statistics)
    finite_state_value_count = (
        5 * sum(leaf.value_count for leaf in workload.parameter_leaves) + 1
    )
    checkpoint = PretrainingQualityCheckpoint(
        kind="pretraining-quality-checkpoint",
        version=1,
        identity="sha256:" + "0" * 64,
        runtime=runtime,
        compute_dtype="bfloat16" if runtime == "candidate" else "float32",
        step=step,
        workload_identity=workload.identity,
        real_work_identity=real_work_identity(workload, step),
        initial_bf16_parameter_identity=workload.initial_bf16_parameter_identity,
        ordered_batch_prefix_identity=workload.batch_prefix_identities[step],
        evaluation_request_identity=workload.evaluation_request_identity,
        master_parameter_identity=quality_module._telemetry_tree_identity(
            quality_module._MASTER_TREE_IDENTITY_DOMAIN,
            statistics,
            "after_master_identity",
        ),
        working_parameter_identity=quality_module._telemetry_tree_identity(
            quality_module._WORKING_TREE_IDENTITY_DOMAIN,
            statistics,
            "after_working_identity",
        ),
        trainer_key_identity="sha256:" + "e" * 64,
        train_loss=2.0,
        validation_nll=validation_nll,
        finite_state_value_count=finite_state_value_count,
        nonfinite_state_value_count=0,
        finite=True,
        update_statistics=statistics,
        changed_bf16_working_fraction=changed_fraction,
        rms_norm_master_moved=step > 0,
        sub_bf16_update_survived=survived and step > 0,
    )
    return replace(checkpoint, identity=checkpoint.recompute_identity())


def test_raw_validation_requires_exact_eight_records_and_matching_real_work(
    canonical_workload,
):
    records = tuple(
        _checkpoint(
            canonical_workload,
            runtime,
            step,
            validation_nll=(2.01 if runtime == "candidate" else 2.0),
        )
        for runtime in ("candidate", "oracle")
        for step in CHECKPOINT_STEPS
    )

    report = validate_pretraining_quality_records(canonical_workload, records)

    assert report == PretrainingQualityReport(2.01, 2.0, True, True, True, True, True)
    with pytest.raises(ValueError, match="exactly eight"):
        validate_pretraining_quality_records(canonical_workload, records[:-1])
    forged = replace(records[4], real_work_identity="sha256:" + "f" * 64)
    forged = replace(forged, identity=forged.recompute_identity())
    with pytest.raises(ValueError, match="real-work identity"):
        validate_pretraining_quality_records(
            canonical_workload,
            (records[0], records[1], records[2], records[3], forged, *records[5:]),
        )

    forged_key = replace(records[4], trainer_key_identity="sha256:" + "f" * 64)
    forged_key = replace(forged_key, identity=forged_key.recompute_identity())
    with pytest.raises(ValueError, match="PRNG"):
        validate_pretraining_quality_records(
            canonical_workload,
            (*records[:4], forged_key, *records[5:]),
        )
    forged_master = replace(records[4], master_parameter_identity="sha256:" + "f" * 64)
    forged_master = replace(forged_master, identity=forged_master.recompute_identity())
    with pytest.raises(ValueError, match="master tree identity|step-zero master"):
        validate_pretraining_quality_records(
            canonical_workload,
            (*records[:4], forged_master, *records[5:]),
        )


def test_raw_validation_reconstructs_every_accepting_telemetry_summary(
    canonical_workload,
):
    records = tuple(
        _checkpoint(
            canonical_workload,
            runtime,
            step,
            validation_nll=(2.01 if runtime == "candidate" else 2.0),
        )
        for runtime in ("candidate", "oracle")
        for step in CHECKPOINT_STEPS
    )

    def resign_record(index, **changes):
        forged = replace(records[index], **changes)
        return replace(forged, identity=forged.recompute_identity())

    step_zero_statistic = records[0].update_statistics[0]
    bad_step_zero_statistic = replace(
        step_zero_statistic,
        nonzero_update_count=1,
        minimum_update_to_bf16_ulp=1.0,
        mean_update_to_bf16_ulp=1.0,
        maximum_update_to_bf16_ulp=1.0,
    )
    forged_step_zero = resign_record(
        0,
        update_statistics=(
            bad_step_zero_statistic,
            *records[0].update_statistics[1:],
        ),
    )
    with pytest.raises(ValueError, match="step-zero|before/after"):
        validate_pretraining_quality_records(
            canonical_workload, (forged_step_zero, *records[1:])
        )

    bad_shape = replace(
        records[1].update_statistics[0],
        shape=(records[1].update_statistics[0].shape[0] + 1,),
    )
    forged_shape = resign_record(
        1,
        update_statistics=(bad_shape, *records[1].update_statistics[1:]),
    )
    with pytest.raises(ValueError, match="shape|value count"):
        validate_pretraining_quality_records(
            canonical_workload, (records[0], forged_shape, *records[2:])
        )

    forged_fraction = resign_record(1, changed_bf16_working_fraction=0.75)
    with pytest.raises(ValueError, match="changed working fraction"):
        validate_pretraining_quality_records(
            canonical_workload, (records[0], forged_fraction, *records[2:])
        )

    forged_movement = resign_record(1, rms_norm_master_moved=False)
    with pytest.raises(ValueError, match="RMSNorm master movement"):
        validate_pretraining_quality_records(
            canonical_workload, (records[0], forged_movement, *records[2:])
        )

    forged_survival = resign_record(1, sub_bf16_update_survived=False)
    with pytest.raises(ValueError, match="sub-BF16 update survival"):
        validate_pretraining_quality_records(
            canonical_workload, (records[0], forged_survival, *records[2:])
        )

    forged_finite = resign_record(1, finite=False)
    with pytest.raises(ValueError, match="finite state status"):
        validate_pretraining_quality_records(
            canonical_workload, (records[0], forged_finite, *records[2:])
        )
    forged_nonfinite_count = resign_record(1, nonfinite_state_value_count=1)
    with pytest.raises(ValueError, match="finite state status"):
        validate_pretraining_quality_records(
            canonical_workload, (records[0], forged_nonfinite_count, *records[2:])
        )

    changed_statistic = replace(
        records[1].update_statistics[0], changed_bf16_working_count=0
    )
    changed_statistics = (changed_statistic, *records[1].update_statistics[1:])
    changed_fraction = sum(
        item.changed_bf16_working_count for item in changed_statistics
    ) / sum(item.value_count for item in changed_statistics)
    forged_before_after = resign_record(
        1,
        update_statistics=changed_statistics,
        changed_bf16_working_fraction=changed_fraction,
    )
    with pytest.raises(ValueError, match="BF16 working identities"):
        validate_pretraining_quality_records(
            canonical_workload,
            (records[0], forged_before_after, *records[2:]),
        )

    survival_statistic = replace(
        records[1].update_statistics[0], first_sub_bf16_survival_step=None
    )
    forged_cumulative = resign_record(
        1,
        update_statistics=(survival_statistic, *records[1].update_statistics[1:]),
        sub_bf16_update_survived=False,
    )
    with pytest.raises(ValueError, match="cumulative sub-BF16"):
        validate_pretraining_quality_records(
            canonical_workload, (records[0], forged_cumulative, *records[2:])
        )


def test_manifest_fields_and_output_paths_fail_closed(canonical_workload, tmp_path):
    destinations = quality_module._canonical_evidence_destinations(
        ROOT,
        ROOT / quality_module.RECORD_MANIFEST_PATH,
        ROOT / quality_module.RECORD_RAW_PATH,
        ROOT / quality_module.RECORD_REPORT_PATH,
    )
    command = quality_module._recording_command_document(ROOT, destinations)
    session_identity = quality_module._recording_session_identity(
        "a" * 40, canonical_workload.identity, command
    )
    manifest = quality_module._manifest_document(
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

    assert quality_module._validate_manifest_fields(
        manifest, canonical_workload, command
    ) == dict(manifest)
    assert manifest["measurement_boundaries"] == (quality_module.MEASUREMENT_BOUNDARIES)
    assert manifest["measured_wall_time_seconds"] == 10.0
    assert manifest["temporary_disk_high_water_bytes"] == (
        sum(manifest["artifact_byte_sizes"].values())
        + manifest["publication_metadata_bytes"]
    )
    assert manifest["artifact_byte_sizes"]["manifest"] == len(
        quality_module.canonical_json_bytes(manifest)
    )
    changed_destinations = {
        **command["destinations"],
        "report": "v2/benchmarks/results/forged-report.json",
    }
    for name, value in (
        ("recording_command", {**command, "argv": ["record", "--steps", "10"]}),
        (
            "recording_command",
            {
                **command,
                "destinations": changed_destinations,
            },
        ),
        (
            "measurement_boundaries",
            {**quality_module.MEASUREMENT_BOUNDARIES, "clock": "wall-clock"},
        ),
        ("phase_wall_time_seconds", {"candidate": 10.0}),
        ("measured_wall_time_seconds", 9.0),
        ("peak_metal_memory_bytes", -1),
        (
            "temporary_disk_high_water_bytes",
            manifest["temporary_disk_high_water_bytes"] + 1,
        ),
        (
            "artifact_byte_sizes",
            {**manifest["artifact_byte_sizes"], "raw": 1},
        ),
        ("fixture_bytes", 1),
        ("training_cardinality", 1),
        ("validation_cardinality", 1),
        ("ordered_work_count", 1),
    ):
        forged = {**manifest, name: value}
        body = {key: item for key, item in forged.items() if key != "identity"}
        forged["identity"] = quality_module.structured_identity(
            "sml-pretraining-quality-manifest-v3", body
        )
        with pytest.raises(ValueError):
            quality_module._validate_manifest_fields(
                forged, canonical_workload, command
            )

    first = tmp_path / "manifest.json"
    second = tmp_path / "raw.jsonl"
    third = tmp_path / "report.json"
    quality_module._preflight_output_paths(first, second, third)
    with pytest.raises(ValueError, match="distinct"):
        quality_module._preflight_output_paths(first, first, third)
    second.write_text("occupied")
    with pytest.raises(FileExistsError, match="already exists"):
        quality_module._preflight_output_paths(first, second, third)
    with pytest.raises(ValueError, match="canonical evidence destinations"):
        quality_module._canonical_evidence_destinations(
            ROOT, first, tmp_path / "raw.jsonl", tmp_path / "report.json"
        )


def test_public_record_accepts_only_the_exact_canonical_step_count():
    parser = quality_module._build_parser()
    common = [
        "--manifest",
        "manifest.json",
        "--raw-output",
        "raw.jsonl",
        "--output",
        "report.json",
    ]

    assert parser.parse_args(["record", "--steps", "1000", *common]).steps == 1_000
    for invalid in ("1", "10", "999", "1001"):
        with pytest.raises(SystemExit):
            parser.parse_args(["record", "--steps", invalid, *common])


@pytest.fixture(scope="module")
def current_recorded_evidence(tmp_path_factory, canonical_workload):
    """Build validator inputs from current source and bounded synthetic records."""
    root = tmp_path_factory.mktemp("pretraining-quality-current-evidence")
    workload = canonical_workload
    copied = {
        *quality_module.HARNESS_COMPONENTS,
        *(Path(path) for path in workload.production_dependency_components),
        quality_module.TRAINING_FIXTURE,
        quality_module.VALIDATION_FIXTURE,
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
    destinations = quality_module._canonical_evidence_destinations(
        root,
        root / quality_module.RECORD_MANIFEST_PATH,
        root / quality_module.RECORD_RAW_PATH,
        root / quality_module.RECORD_REPORT_PATH,
    )
    command = quality_module._recording_command_document(root, destinations)
    records = tuple(
        _checkpoint(
            workload,
            runtime,
            step,
            validation_nll=2.01 if runtime == "candidate" else 2.0,
        )
        for runtime in ("candidate", "oracle")
        for step in CHECKPOINT_STEPS
    )
    documents = [record.to_dict() for record in records]
    raw = b"".join(
        quality_module.canonical_json_bytes(document) + b"\n" for document in documents
    )
    raw_identity = quality_module.structured_identity(
        "sml-pretraining-quality-raw-v1", documents
    )
    report = quality_module._report_document(
        workload.identity,
        raw_identity,
        validate_pretraining_quality_records(workload, records),
    )
    report_bytes = quality_module.canonical_json_bytes(report)
    manifest = quality_module._manifest_document(
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
        raw_file_identity=quality_module._payload_identity(raw),
        raw_bytes=len(raw),
        report_identity=report["identity"],
        report_file_identity=quality_module._payload_identity(report_bytes),
        report_bytes=len(report_bytes),
        recording_session_identity=quality_module._recording_session_identity(
            commit, workload.identity, command
        ),
    )
    for path in (destinations.manifest, destinations.raw_output, destinations.report):
        path.parent.mkdir(parents=True, exist_ok=True)
    destinations.manifest.write_bytes(quality_module.canonical_json_bytes(manifest))
    destinations.raw_output.write_bytes(raw)
    destinations.report.write_bytes(report_bytes)
    return root, destinations, manifest, workload


def test_current_standalone_evidence_validation(current_recorded_evidence, monkeypatch):
    root, destinations, manifest, _workload = current_recorded_evidence
    assert manifest["recording_command"] == quality_module._recording_command_document(
        root, destinations
    )
    monkeypatch.setattr(quality_module, "_root", lambda: root)
    assert (
        quality_module._validate(
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
    identity = quality_module._production_dependency_identity(
        tuple(Path(path) for path in retained),
        lambda path: quality_module._git_bytes(
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
        quality_module._validate_harness_commit(
            root, manifest["source_commit"], tampered
        )


def test_recorded_evidence_validation_uses_recorded_source(
    current_recorded_evidence, monkeypatch
):
    root, destinations, manifest, _workload = current_recorded_evidence
    path = root / "v2/src/sml/artifacts/checkpoint.py"
    with _temporary_current_tree_bytes(
        path, path.read_bytes() + b"\n# current unrecorded source edit\n"
    ):
        assert (
            production_dependency_content_identity(root)
            != manifest["production_dependency_identity"]
        )

        def forbid_current_workload(*args, **kwargs):
            raise AssertionError("validation must use the recorded workload")

        monkeypatch.setattr(
            quality_module,
            "build_pretraining_quality_workload",
            forbid_current_workload,
        )
        assert quality_module._validate_evidence_files(root, destinations) == "pass"
