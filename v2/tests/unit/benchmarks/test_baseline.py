import json
from copy import deepcopy
from dataclasses import replace

import pytest

import v2.benchmarks.runner as benchmark_runner
from v2.benchmarks.runner import (
    build_baseline_manifest,
    validate_baseline_manifest,
    validate_baseline_trial,
)
from v2.benchmarks.schema import RawTrial
from v2.benchmarks.workload import (
    build_canonical_workload,
    canonical_workload_identity,
    structured_identity,
)

from .support import (
    _baseline_fixture,
    _valid_baseline_trials,
    _valid_raw_trial,
    _valid_recovered_raw_trial,
    _with_observation_changes,
    _with_trial_payload,
)


def test_baseline_manifest_binds_raw_trials_to_clean_source_and_harness():
    workload = build_canonical_workload()
    workload_identity = canonical_workload_identity(workload)
    trials = _valid_baseline_trials(workload)
    trial = trials[0]
    manifest = build_baseline_manifest(
        trials=trials,
        workload=workload,
        workload_identity=workload_identity,
        source_commit=trial.source_commit,
        harness_commit=trial.harness_commit,
        harness_identity=trial.harness_identity,
        command=f"record-baseline --source-commit {'c' * 40}",
        pairs=5,
        warmup_units=5,
        measured_units=20,
    )

    validate_baseline_manifest(manifest, trials)
    assert manifest["identity"].startswith("sha256:")
    mutated = _with_trial_payload(trial, harness_identity="sha256:" + "e" * 64)
    with pytest.raises(ValueError, match="harness identity"):
        validate_baseline_manifest(manifest, (mutated, *trials[1:]))


def test_baseline_rejects_raw_trial_without_valid_post_exit_evidence():
    workload = build_canonical_workload()
    trial = _valid_raw_trial(workload)
    raw = trial.to_dict()
    raw["post_exit_observation"]["child_measurement_identity"] = "sha256:" + "0" * 64

    with pytest.raises(ValueError, match="post-exit observation identity"):
        validate_baseline_trial(
            RawTrial.from_dict(raw),
            workload=workload,
            source_commit=trial.source_commit,
            harness_commit=trial.harness_commit,
            harness_identity=trial.harness_identity,
            expected_hardware=trial.hardware,
            expected_software_versions=trial.software_versions,
        )


def test_baseline_manifest_construction_rejects_tampered_embedded_evidence():
    workload = build_canonical_workload()
    trials = _valid_baseline_trials(workload)
    tampered = replace(trials[0], value=trials[0].value + 1.0)

    with pytest.raises(ValueError, match="embedded evidence"):
        build_baseline_manifest(
            trials=(tampered, *trials[1:]),
            workload=workload,
            workload_identity=canonical_workload_identity(workload),
            source_commit=trials[0].source_commit,
            harness_commit=trials[0].harness_commit,
            harness_identity=trials[0].harness_identity,
            command="record-baseline",
            pairs=5,
            warmup_units=5,
            measured_units=20,
        )


def _resign_baseline(manifest):
    body = {key: value for key, value in manifest.items() if key != "identity"}
    manifest["identity"] = structured_identity(
        benchmark_runner._baseline_identity_domain(manifest["version"]), body
    )


@pytest.mark.parametrize("version", (1, 2, True, 3.0))
def test_baseline_validator_rejects_unsupported_versions_and_numeric_aliases(version):
    manifest, trials = _baseline_fixture()
    tampered = deepcopy(manifest)
    tampered["version"] = version
    body = {key: value for key, value in tampered.items() if key != "identity"}
    tampered["identity"] = structured_identity("sml-performance-baseline-v3", body)

    with pytest.raises(ValueError, match="unsupported baseline"):
        validate_baseline_manifest(tampered, trials)


@pytest.mark.parametrize(
    ("field", "value"),
    (("measured_units", 20.0), ("prepared_data_measured_units", 100.0)),
)
def test_baseline_validator_rejects_float_protocol_aliases(field, value):
    manifest, trials = _baseline_fixture()
    tampered = deepcopy(manifest)
    tampered["protocol"][field] = value
    _resign_baseline(tampered)

    with pytest.raises(ValueError, match="baseline protocol"):
        validate_baseline_manifest(tampered, trials)


def test_baseline_validator_rejects_partial_or_weakened_protocols():
    workload = build_canonical_workload()
    trials = _valid_baseline_trials(workload)
    manifest = build_baseline_manifest(
        trials=trials,
        workload=workload,
        workload_identity=canonical_workload_identity(workload),
        source_commit=trials[0].source_commit,
        harness_commit=trials[0].harness_commit,
        harness_identity=trials[0].harness_identity,
        command=f"record-baseline --source-commit {'c' * 40}",
        pairs=5,
        warmup_units=5,
        measured_units=20,
    )

    wrong_source = json.loads(json.dumps(manifest))
    wrong_source["source"]["commit"] = "1" * 40
    _resign_baseline(wrong_source)
    with pytest.raises(ValueError, match="source commit"):
        validate_baseline_manifest(wrong_source, trials)

    missing_metric = json.loads(json.dumps(manifest))
    missing_metric["metrics"].pop("inference-decode")
    _resign_baseline(missing_metric)
    without_decode = tuple(
        trial for trial in trials if trial.metric != "inference-decode"
    )
    with pytest.raises(ValueError, match="every benchmark metric"):
        validate_baseline_manifest(missing_metric, without_decode)

    wrong_warmup = _with_trial_payload(trials[0], warmup_units=20)
    with pytest.raises(ValueError, match="warmup or measured"):
        validate_baseline_manifest(manifest, (wrong_warmup, *trials[1:]))

    wrong_units = _with_trial_payload(trials[0], measured_units=99)
    with pytest.raises(ValueError, match="warmup or measured"):
        validate_baseline_manifest(manifest, (wrong_units, *trials[1:]))

    old_protocol = json.loads(json.dumps(manifest))
    old_protocol["protocol"]["warmup_units"] = 20
    old_protocol["protocol"]["measured_units"] = 100
    _resign_baseline(old_protocol)
    with pytest.raises(ValueError, match="pinned protocol"):
        validate_baseline_manifest(old_protocol, trials)

    with pytest.raises(ValueError, match="duplicate, missing, or extra"):
        validate_baseline_manifest(manifest, (*trials, trials[0]))


@pytest.mark.parametrize(
    ("environment_change", "software_change", "message"),
    [
        ({"power_connected": False}, {}, "power_connected"),
        ({"power_mode": "changed"}, {}, "power_mode"),
        ({"low_power_mode": True}, {}, "low_power_mode"),
        (
            {"memory_pressure": "warning"},
            {},
            "non-normal-start-memory-pressure",
        ),
        ({"competing_gpu_workload": True}, {}, "competing_gpu_workload"),
        ({}, {"python": "3.12.12"}, "Python version"),
    ],
)
def test_baseline_rejects_invalid_environment_or_software(
    environment_change, software_change, message
):
    workload = build_canonical_workload()
    trials = _valid_baseline_trials(workload)
    manifest = build_baseline_manifest(
        trials=trials,
        workload=workload,
        workload_identity=canonical_workload_identity(workload),
        source_commit=trials[0].source_commit,
        harness_commit=trials[0].harness_commit,
        harness_identity=trials[0].harness_identity,
        command="record-baseline",
        pairs=5,
        warmup_units=5,
        measured_units=20,
    )
    changed = _with_observation_changes(
        trials[0],
        environment_changes=environment_change,
        software_changes=software_change,
    )

    with pytest.raises(ValueError, match=message):
        validate_baseline_manifest(manifest, (changed, *trials[1:]))


@pytest.mark.parametrize(
    ("field", "build_invalid", "message"),
    (
        ("pair_index", lambda trial: replace(trial, pair_index=False), "pair_index"),
        (
            "attempt_index",
            lambda trial: replace(trial, attempt_index=False),
            "attempt_index",
        ),
        ("warmup_units", lambda trial: replace(trial, warmup_units=False), "warmup"),
        (
            "measured_units",
            lambda trial: replace(trial, measured_units=True),
            "measured",
        ),
        (
            "rope_scaling_factor",
            lambda trial: replace(
                trial,
                native_configuration={
                    **trial.native_configuration,
                    "rope_scaling_factor": 1,
                },
            ),
            "rope_scaling_factor",
        ),
        ("value", lambda trial: replace(trial, value=True), "embedded evidence"),
        (
            "power_connected",
            lambda trial: replace(
                trial,
                environment_status={
                    **trial.environment_status,
                    "power_connected": 1,
                },
            ),
            "power_connected",
        ),
        (
            "low_power_mode",
            lambda trial: replace(
                trial,
                environment_status={
                    **trial.environment_status,
                    "low_power_mode": 0,
                },
            ),
            "low_power_mode",
        ),
        (
            "competing_gpu_workload",
            lambda trial: replace(
                trial,
                environment_status={
                    **trial.environment_status,
                    "competing_gpu_workload": 0,
                },
            ),
            "competing_gpu_workload",
        ),
    ),
)
def test_single_baseline_trial_validation_rejects_boolean_numeric_substitutes(
    field, build_invalid, message
):
    workload = build_canonical_workload()
    valid = _valid_raw_trial(workload, metric="compile-cold-start")
    invalid = build_invalid(valid)

    with pytest.raises(ValueError, match=message):
        benchmark_runner.validate_baseline_trial(
            invalid,
            workload=workload,
            source_commit=valid.source_commit,
            harness_commit=valid.harness_commit,
            harness_identity=valid.harness_identity,
            expected_hardware=valid.hardware,
            expected_software_versions=valid.software_versions,
            allow_rejected_environment=True,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("power_connected", 1),
        ("low_power_mode", 0),
        ("competing_gpu_workload", 0),
    ),
)
def test_baseline_preflight_rejects_boolean_integer_substitutes(field, value):
    workload = build_canonical_workload()
    trial = _valid_raw_trial(workload)
    status = {**trial.environment_status, field: value}

    with pytest.raises(ValueError, match=field):
        benchmark_runner._validate_baseline_preflight(
            workload=workload,
            hardware=trial.hardware,
            status=status,
            software_versions=trial.software_versions,
            expected_hardware=trial.hardware,
            expected_software_versions=trial.software_versions,
        )


@pytest.mark.parametrize(
    "mutate",
    (
        lambda raw: raw["post_exit_recovery"].update(identity="sha256:" + "0" * 64),
        lambda raw: raw["post_exit_recovery_samples"][0].update(elapsed_seconds=31.0),
        lambda raw: raw["post_exit_recovery_samples"][0].update(
            previous_sample_identity="sha256:" + "0" * 64
        ),
    ),
)
def test_baseline_rejects_tampered_recovery_evidence(mutate):
    workload = build_canonical_workload()
    raw = _valid_recovered_raw_trial(workload).to_dict()
    mutate(raw)

    with pytest.raises(ValueError, match="evidence|recovery|identity"):
        validate_baseline_trial(
            RawTrial.from_dict(raw),
            workload=workload,
            source_commit=raw["source_commit"],
            harness_commit=raw["harness_commit"],
            harness_identity=raw["harness_identity"],
            expected_hardware=raw["hardware"],
            expected_software_versions=raw["software_versions"],
        )
