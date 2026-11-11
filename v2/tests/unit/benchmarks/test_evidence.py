import json
from dataclasses import replace

import pytest

import v2.benchmarks.runner as benchmark_runner
from v2.benchmarks.evidence import (
    build_post_exit_observation,
    build_post_exit_recovery,
    build_post_exit_recovery_sample,
    finalize_raw_trial,
    validate_child_trial_measurement,
    validate_post_exit_observation,
    validate_post_exit_recovery,
    validate_post_exit_recovery_sample,
    validate_raw_trial_evidence,
)
from v2.benchmarks.schema import RawTrial
from v2.benchmarks.workload import (
    build_canonical_workload,
    post_exit_recovery_policy,
    structured_identity,
)

from .support import (
    _recovery_evidence,
    _valid_child_measurement,
    _valid_observation,
    _valid_post_exit_observation,
    _valid_post_exit_recovery,
    _valid_raw_trial,
    _valid_recovery_policy,
    _validate_trial_metadata,
    _with_trial_payload,
)


def _valid_recovered_evidence():
    return _recovery_evidence(
        build_canonical_workload(),
        immediate_pressure="warning",
        sample_pressures=("normal",) * 7,
        sample_elapsed=(5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0),
        outcome="recovered",
    )[:4]


def _warning_recovery_sample_chain(workload, pressures, elapsed, status_changes=()):
    measurement = _valid_child_measurement(workload)
    immediate = _valid_observation("2026-08-09T00:00:02+00:00")
    immediate["environment_status"]["memory_pressure"] = "warning"
    post_exit = build_post_exit_observation(measurement=measurement, **immediate)
    changes = status_changes or ({},) * len(pressures)
    samples = []
    previous_identity = None
    for index, (pressure, elapsed_seconds, changes_for_sample) in enumerate(
        zip(pressures, elapsed, changes, strict=True)
    ):
        observation = _valid_observation("2026-08-09T00:00:03+00:00")
        observation["environment_status"].update(
            memory_pressure=pressure, **changes_for_sample
        )
        sample = build_post_exit_recovery_sample(
            measurement=measurement,
            post_exit=post_exit,
            sample_index=index,
            previous_sample_identity=previous_identity,
            elapsed_seconds=elapsed_seconds,
            **observation,
        )
        samples.append(sample)
        previous_identity = sample["identity"]
    return measurement, post_exit, tuple(samples)


def _evidence_for_recovery_outcome(outcome, pressure, failure_fields):
    kwargs = {
        "immediate_pressure": pressure,
        "outcome": outcome,
        "failure_fields": failure_fields,
    }
    if outcome == "recovered":
        kwargs.update(
            sample_pressures=("normal",) * 7,
            sample_elapsed=(5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0),
        )
    elif outcome == "timeout":
        kwargs.update(sample_pressures=("warning",), sample_elapsed=(300.0,))
    elif outcome == "environment-failure":
        kwargs.update(
            sample_pressures=("normal",),
            sample_elapsed=(5.0,),
            sample_status_changes=({"power_connected": False},),
        )
    return _recovery_evidence(build_canonical_workload(), **kwargs)[:3]


@pytest.mark.parametrize("protocol", ("baseline", "comparison", "predecessor"))
@pytest.mark.parametrize(
    "field,value,message",
    (
        ("canonical_projection_identity", None, "projection identity"),
        ("rope_scaling_factor", 1, "rope_scaling_factor"),
        ("parameter_precision_policy", None, "precision policy"),
    ),
)
def test_all_trial_validators_enforce_native_metadata(protocol, field, value, message):
    workload = build_canonical_workload()
    reference = _valid_raw_trial(workload, metric="inference-decode")
    changed = _with_trial_payload(
        reference,
        native_configuration={**reference.native_configuration, field: value},
    )

    with pytest.raises(ValueError, match=message):
        _validate_trial_metadata(protocol, workload, changed)


def test_child_and_post_exit_documents_are_exactly_identity_bound():
    measurement = _valid_child_measurement(build_canonical_workload())
    post_exit = _valid_post_exit_observation(measurement)

    assert validate_child_trial_measurement(measurement) == measurement
    assert (
        validate_post_exit_observation(post_exit, measurement=measurement) == post_exit
    )

    changed = json.loads(json.dumps(post_exit))
    changed["environment_status"]["memory_free_percentage"] -= 1
    with pytest.raises(ValueError, match="post-exit observation identity"):
        validate_post_exit_observation(changed, measurement=measurement)


@pytest.mark.parametrize(
    "document_kind", ["measurement", "post-exit", "sample", "recovery"]
)
@pytest.mark.parametrize("alias_type", ["boolean", "float"])
def test_evidence_rejects_numeric_version_aliases(document_kind, alias_type):
    measurement, post_exit, samples, recovery = _valid_recovered_evidence()
    documents = {
        "measurement": (
            measurement,
            validate_child_trial_measurement,
            "sml-child-trial-measurement-v1",
            {},
        ),
        "post-exit": (
            post_exit,
            validate_post_exit_observation,
            "sml-parent-post-exit-observation-v1",
            {"measurement": measurement},
        ),
        "sample": (
            samples[0],
            validate_post_exit_recovery_sample,
            "sml-parent-post-exit-recovery-sample-v1",
            {
                "measurement": measurement,
                "post_exit": post_exit,
                "previous_sample": None,
            },
        ),
        "recovery": (
            recovery,
            validate_post_exit_recovery,
            "sml-parent-post-exit-recovery-v2",
            {"measurement": measurement, "post_exit": post_exit, "samples": samples},
        ),
    }
    original, validate, domain, context = documents[document_kind]
    changed = dict(original)
    changed["version"] = True if alias_type == "boolean" else float(original["version"])
    body = {key: value for key, value in changed.items() if key != "identity"}
    changed["identity"] = structured_identity(domain, body)

    with pytest.raises(ValueError, match="version"):
        validate(changed, **context)


def test_post_exit_observation_rejects_a_boolean_pair_index():
    measurement = _valid_child_measurement(build_canonical_workload(), pair_index=1)
    post_exit = _valid_post_exit_observation(measurement)
    post_exit["pair_index"] = True
    post_exit_body = {
        key: value for key, value in post_exit.items() if key != "identity"
    }
    post_exit["identity"] = structured_identity(
        "sml-parent-post-exit-observation-v1", post_exit_body
    )

    with pytest.raises(ValueError, match="pair_index"):
        validate_post_exit_observation(post_exit, measurement=measurement)


def test_raw_trial_v3_embeds_and_revalidates_the_recovery_chain():
    workload = build_canonical_workload()
    measurement = _valid_child_measurement(workload)
    post_exit = _valid_post_exit_observation(measurement)
    recovery = _valid_post_exit_recovery(measurement, post_exit)
    trial = finalize_raw_trial(measurement, post_exit, (), recovery)

    assert trial.schema_version == 3
    assert trial.post_exit_recovery_samples == ()
    assert trial.post_exit_recovery == recovery
    validate_raw_trial_evidence(trial)


@pytest.mark.parametrize("version", (1, 2, True, 3.0))
def test_raw_trial_rejects_unsupported_versions_and_numeric_aliases(version):
    raw = _valid_raw_trial(build_canonical_workload()).to_dict()
    raw["schema_version"] = version

    with pytest.raises(ValueError, match="schema.version"):
        RawTrial.from_dict(raw)


def test_recovery_summary_v2_requires_identity_bound_completion_source():
    measurement = _valid_child_measurement(build_canonical_workload())
    post_exit = _valid_post_exit_observation(measurement)
    recovery = _valid_post_exit_recovery(measurement, post_exit)

    assert recovery["version"] == 2
    assert recovery["completion_source"] == "live"
    recovery_body = {key: value for key, value in recovery.items() if key != "identity"}
    assert recovery["identity"] == structured_identity(
        "sml-parent-post-exit-recovery-v2", recovery_body
    )

    changed_source = {**recovery, "completion_source": "crash-reconstruction"}
    with pytest.raises(ValueError, match="recovery identity"):
        validate_post_exit_recovery(
            changed_source,
            measurement=measurement,
            post_exit=post_exit,
            samples=(),
        )

    version_one_body = {
        key: value
        for key, value in recovery.items()
        if key not in {"completion_source", "identity"}
    }
    version_one_body["version"] = 1
    version_one = {
        **version_one_body,
        "identity": structured_identity(
            "sml-parent-post-exit-recovery-v1", version_one_body
        ),
    }
    with pytest.raises(ValueError, match="field set|kind or version"):
        validate_post_exit_recovery(
            version_one,
            measurement=measurement,
            post_exit=post_exit,
            samples=(),
        )


def test_recovery_summary_rejects_invalid_completion_source_outcome_pairs():
    workload = build_canonical_workload()
    measurement, post_exit, samples, _recovery = _valid_recovered_evidence()
    policy = post_exit_recovery_policy(workload)

    with pytest.raises(ValueError, match="recovery outcome"):
        build_post_exit_recovery(
            measurement=measurement,
            post_exit=post_exit,
            samples=samples,
            policy=policy,
            outcome="recovered",
            duration_seconds=35.0,
            completion_source="crash-reconstruction",
        )
    with pytest.raises(ValueError, match="recovery outcome"):
        build_post_exit_recovery(
            measurement=measurement,
            post_exit=post_exit,
            samples=(),
            policy=policy,
            outcome="interrupted",
            duration_seconds=0.0,
            completion_source="live",
        )


def test_recovery_samples_form_an_exact_ordered_identity_chain():
    measurement, post_exit, samples, recovery = _valid_recovered_evidence()

    previous = None
    for index, sample in enumerate(samples):
        assert (
            validate_post_exit_recovery_sample(
                sample,
                measurement=measurement,
                post_exit=post_exit,
                previous_sample=previous,
            )
            == sample
        )
        assert sample["sample_index"] == index
        previous = sample
    assert (
        validate_post_exit_recovery(
            recovery,
            measurement=measurement,
            post_exit=post_exit,
            samples=samples,
        )
        == recovery
    )

    changed = json.loads(json.dumps(samples))
    changed[1]["previous_sample_identity"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="previous sample identity"):
        validate_post_exit_recovery(
            recovery,
            measurement=measurement,
            post_exit=post_exit,
            samples=changed,
        )


@pytest.mark.parametrize(
    ("outcome", "pressure", "failure_fields"),
    (
        ("not-required", "normal", ()),
        ("recovered", "warning", ()),
        ("timeout", "warning", ()),
        ("critical", "critical", ()),
        ("environment-failure", "warning", ("power_connected",)),
        ("interrupted", "warning", ()),
    ),
)
def test_recovery_summary_rejects_incompatible_outcome_evidence(
    outcome, pressure, failure_fields
):
    measurement, post_exit, samples = _evidence_for_recovery_outcome(
        outcome, pressure, failure_fields
    )
    recovery = build_post_exit_recovery(
        measurement=measurement,
        post_exit=post_exit,
        samples=samples,
        policy=_valid_recovery_policy(build_canonical_workload()),
        outcome=outcome,
        duration_seconds=0.0 if not samples else samples[-1]["elapsed_seconds"],
        failure_fields=failure_fields,
        completion_source=(
            "crash-reconstruction" if outcome == "interrupted" else "live"
        ),
    )
    changed = {
        **recovery,
        "outcome": "timeout" if outcome == "recovered" else "recovered",
    }

    with pytest.raises(ValueError, match="recovery identity|recovery outcome"):
        validate_post_exit_recovery(
            changed,
            measurement=measurement,
            post_exit=post_exit,
            samples=samples,
        )


def test_critical_recovery_sample_requires_a_critical_outcome():
    workload = build_canonical_workload()
    measurement, post_exit, samples = _warning_recovery_sample_chain(
        workload,
        pressures=("critical",),
        elapsed=(5.0,),
    )
    policy = post_exit_recovery_policy(workload)

    with pytest.raises(ValueError, match="recovery outcome"):
        build_post_exit_recovery(
            measurement=measurement,
            post_exit=post_exit,
            samples=samples,
            policy=policy,
            outcome="recovered",
            duration_seconds=5.0,
        )

    critical = build_post_exit_recovery(
        measurement=measurement,
        post_exit=post_exit,
        samples=samples,
        policy=policy,
        outcome="critical",
        duration_seconds=5.0,
    )
    assert critical["outcome"] == "critical"
    with pytest.raises(ValueError, match="recovery outcome"):
        validate_post_exit_recovery(
            {**critical, "outcome": "recovered"},
            measurement=measurement,
            post_exit=post_exit,
            samples=samples,
        )


def test_thermal_recovery_sample_requires_environment_failure_evidence():
    workload = build_canonical_workload()
    measurement, post_exit, samples = _warning_recovery_sample_chain(
        workload,
        pressures=("normal",),
        elapsed=(5.0,),
        status_changes=({"thermal_state": "fair", "thermal_state_raw_value": 1},),
    )
    policy = post_exit_recovery_policy(workload)

    with pytest.raises(ValueError, match="recovery outcome"):
        build_post_exit_recovery(
            measurement=measurement,
            post_exit=post_exit,
            samples=samples,
            policy=policy,
            outcome="recovered",
            duration_seconds=5.0,
        )

    environment_failure = build_post_exit_recovery(
        measurement=measurement,
        post_exit=post_exit,
        samples=samples,
        policy=policy,
        outcome="environment-failure",
        duration_seconds=5.0,
        failure_fields=("thermal_state",),
    )
    assert environment_failure["failure_fields"] == ["thermal_state"]
    with pytest.raises(ValueError, match="recovery failure_fields"):
        build_post_exit_recovery(
            measurement=measurement,
            post_exit=post_exit,
            samples=samples,
            policy=policy,
            outcome="environment-failure",
            duration_seconds=5.0,
            failure_fields=("power_connected",),
        )


def test_timeout_uses_the_deadline_bound_stable_recovery_window():
    workload = build_canonical_workload()
    policy = post_exit_recovery_policy(workload)
    measurement, post_exit, late_recovery_samples = _warning_recovery_sample_chain(
        workload,
        pressures=("warning", "normal", "normal", "normal", "normal"),
        elapsed=(280.0, 285.0, 290.0, 295.0, 300.0),
    )

    timeout = build_post_exit_recovery(
        measurement=measurement,
        post_exit=post_exit,
        samples=late_recovery_samples,
        policy=policy,
        outcome="timeout",
        duration_seconds=300.0,
    )
    assert timeout["outcome"] == "timeout"

    measurement, post_exit, stable_samples = _warning_recovery_sample_chain(
        workload,
        pressures=("normal",) * 4,
        elapsed=(270.0, 280.0, 290.0, 300.0),
    )
    with pytest.raises(ValueError, match="recovery outcome"):
        build_post_exit_recovery(
            measurement=measurement,
            post_exit=post_exit,
            samples=stable_samples,
            policy=policy,
            outcome="timeout",
            duration_seconds=300.0,
        )


@pytest.mark.parametrize(
    ("pressure", "status_changes"),
    (
        ("critical", {}),
        ("normal", {"power_connected": False}),
    ),
)
def test_timeout_cannot_override_an_earlier_terminal_sample(pressure, status_changes):
    workload = build_canonical_workload()
    measurement, post_exit, samples = _warning_recovery_sample_chain(
        workload,
        pressures=(pressure,),
        elapsed=(5.0,),
        status_changes=(status_changes,),
    )

    with pytest.raises(ValueError, match="recovery outcome"):
        build_post_exit_recovery(
            measurement=measurement,
            post_exit=post_exit,
            samples=samples,
            policy=post_exit_recovery_policy(workload),
            outcome="timeout",
            duration_seconds=300.0,
        )


@pytest.mark.parametrize(
    "elapsed",
    (
        (4.0, 9.0, 14.0, 19.0, 24.0, 29.0, 34.0),
        (5.0, 10.0, 14.0, 19.0, 24.0, 29.0, 35.0),
    ),
)
def test_recovery_summary_rejects_cadence_infeasible_samples(elapsed):
    workload = build_canonical_workload()
    measurement, post_exit, samples = _warning_recovery_sample_chain(
        workload,
        pressures=("normal",) * 7,
        elapsed=elapsed,
    )

    with pytest.raises(ValueError, match="recovery sample cadence"):
        build_post_exit_recovery(
            measurement=measurement,
            post_exit=post_exit,
            samples=samples,
            policy=post_exit_recovery_policy(workload),
            outcome="recovered",
            duration_seconds=elapsed[-1],
        )


@pytest.mark.parametrize(
    ("pressures", "elapsed", "status_changes", "outcome", "failure_fields"),
    (
        (("critical", "normal"), (5.0, 10.0), (), "critical", ()),
        (
            ("normal", "normal"),
            (5.0, 10.0),
            ({"power_connected": False}, {}),
            "environment-failure",
            ("power_connected",),
        ),
        (
            ("normal",) * 8,
            tuple(float(value) for value in range(5, 45, 5)),
            (),
            "recovered",
            (),
        ),
        (
            ("normal",) * 7 + ("warning",),
            (5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 300.0),
            (),
            "timeout",
            (),
        ),
    ),
)
def test_recovery_summary_rejects_samples_after_first_terminal_event(
    pressures, elapsed, status_changes, outcome, failure_fields
):
    workload = build_canonical_workload()
    measurement, post_exit, samples = _warning_recovery_sample_chain(
        workload,
        pressures=pressures,
        elapsed=elapsed,
        status_changes=status_changes,
    )

    with pytest.raises(ValueError, match="samples after the first terminal event"):
        build_post_exit_recovery(
            measurement=measurement,
            post_exit=post_exit,
            samples=samples,
            policy=post_exit_recovery_policy(workload),
            outcome=outcome,
            duration_seconds=elapsed[-1],
            failure_fields=failure_fields,
        )


def test_recovery_summary_uses_nonmemory_failure_before_critical_pressure():
    workload = build_canonical_workload()
    measurement, post_exit, samples = _warning_recovery_sample_chain(
        workload,
        pressures=("critical",),
        elapsed=(5.0,),
        status_changes=({"power_connected": False},),
    )
    policy = post_exit_recovery_policy(workload)

    recovery = build_post_exit_recovery(
        measurement=measurement,
        post_exit=post_exit,
        samples=samples,
        policy=policy,
        outcome="environment-failure",
        duration_seconds=5.0,
        failure_fields=("power_connected",),
    )

    assert recovery["outcome"] == "environment-failure"
    assert recovery["failure_fields"] == ["power_connected"]
    with pytest.raises(ValueError, match="recovery outcome"):
        build_post_exit_recovery(
            measurement=measurement,
            post_exit=post_exit,
            samples=samples,
            policy=policy,
            outcome="critical",
            duration_seconds=5.0,
        )


@pytest.mark.parametrize("pressure", ("normal", "critical"))
def test_immediate_identity_drift_is_a_zero_sample_environment_failure(pressure):
    workload = build_canonical_workload()
    measurement = _valid_child_measurement(workload)
    immediate = _valid_observation("2026-08-09T00:00:02+00:00")
    immediate["hardware"] = {**immediate["hardware"], "chip": "Other"}
    immediate["environment_status"]["memory_pressure"] = pressure
    post_exit = build_post_exit_observation(measurement=measurement, **immediate)
    policy = post_exit_recovery_policy(workload)

    recovery = build_post_exit_recovery(
        measurement=measurement,
        post_exit=post_exit,
        samples=(),
        policy=policy,
        outcome="environment-failure",
        duration_seconds=0.0,
        failure_fields=("hardware",),
    )

    assert recovery["outcome"] == "environment-failure"
    assert recovery["failure_fields"] == ["hardware"]
    assert (
        validate_post_exit_recovery(
            recovery,
            measurement=measurement,
            post_exit=post_exit,
            samples=(),
        )
        == recovery
    )
    for failure_fields in ((), ("software_versions",), ("hardware", "thermal_state")):
        with pytest.raises(ValueError, match="recovery failure_fields"):
            build_post_exit_recovery(
                measurement=measurement,
                post_exit=post_exit,
                samples=(),
                policy=policy,
                outcome="environment-failure",
                duration_seconds=0.0,
                failure_fields=failure_fields,
            )
    with pytest.raises(ValueError, match="recovery outcome"):
        build_post_exit_recovery(
            measurement=measurement,
            post_exit=post_exit,
            samples=(),
            policy=policy,
            outcome="critical" if pressure == "critical" else "not-required",
            duration_seconds=0.0,
        )


def test_raw_trial_round_trip_is_strict():
    workload = build_canonical_workload()
    trial = _valid_raw_trial(workload)

    assert RawTrial.from_dict(trial.to_dict()) == trial
    invalid = trial.to_dict()
    invalid["unexpected"] = True
    with pytest.raises(ValueError, match="field set"):
        RawTrial.from_dict(invalid)


@pytest.mark.parametrize(
    "field",
    (
        "schema_version",
        "pair_index",
        "attempt_index",
        "process_order",
        "warmup_units",
        "measured_units",
        "elapsed_seconds",
        "value",
        "startup_verification_seconds",
        "compilation_seconds",
        "peak_memory_bytes",
    ),
)
def test_raw_trial_rejects_boolean_numeric_scalars_before_coercion(field):
    raw = _valid_raw_trial(build_canonical_workload()).to_dict()
    raw[field] = True

    with pytest.raises(ValueError, match=field):
        RawTrial.from_dict(raw)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("elapsed_seconds", 2),
        ("elapsed_seconds", 2.5),
        ("value", 50),
        ("value", 50.5),
        ("startup_verification_seconds", 1),
        ("startup_verification_seconds", 1.5),
        ("compilation_seconds", 1),
        ("compilation_seconds", 1.5),
    ),
)
def test_raw_trial_accepts_real_integer_and_float_seconds_and_values(field, value):
    raw = _valid_raw_trial(build_canonical_workload()).to_dict()
    raw[field] = value

    parsed = RawTrial.from_dict(raw)

    assert getattr(parsed, field) == float(value)
    assert type(getattr(parsed, field)) is float


def test_raw_trial_evidence_rejects_a_cached_only_environment_change():
    trial = _valid_raw_trial(build_canonical_workload())
    mismatched = replace(
        trial,
        environment_status={
            **trial.environment_status,
            "thermal_state": "fair",
            "thermal_state_raw_value": 1,
        },
    )

    with pytest.raises(ValueError, match="embedded evidence"):
        validate_raw_trial_evidence(mismatched)


def test_raw_trial_identity_uses_version_three_domain():
    trial = _valid_raw_trial(build_canonical_workload())

    assert benchmark_runner._raw_trial_identity(trial) == structured_identity(
        "sml-raw-benchmark-trial-v3", trial.to_dict()
    )


def test_raw_trial_identity_rejects_tampered_embedded_evidence():
    trial = _valid_raw_trial(build_canonical_workload())
    tampered = replace(trial, value=trial.value + 1.0)

    with pytest.raises(ValueError, match="embedded evidence"):
        benchmark_runner._raw_trial_identity(tampered)


def test_raw_trial_jsonl_rejects_tampered_embedded_evidence(tmp_path):
    trial = _valid_raw_trial(build_canonical_workload())
    raw = replace(trial, value=trial.value + 1.0).to_dict()
    path = tmp_path / "baseline.jsonl"
    path.write_text(json.dumps(raw) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="embedded evidence"):
        benchmark_runner._read_trials(path)
