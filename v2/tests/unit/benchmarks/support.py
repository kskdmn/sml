import json

import v2.benchmarks.runner as benchmark_runner
from v2.benchmarks.evidence import (
    build_child_trial_measurement,
    build_post_exit_observation,
    build_post_exit_recovery,
    build_post_exit_recovery_sample,
    finalize_raw_trial,
)
from v2.benchmarks.journal import (
    atomic_write_json,
    build_session_document,
)
from v2.benchmarks.runner import (
    build_baseline_manifest,
    build_comparison_report,
    process_order,
    validate_baseline_trial,
)
from v2.benchmarks.schema import METRIC_NAMES
from v2.benchmarks.workload import (
    PRECISION_POLICY,
    build_canonical_workload,
    canonical_execution_order_identity,
    canonical_input_identity,
    canonical_metric_projection,
    canonical_workload_identity,
    post_exit_recovery_policy,
    structured_identity,
)


class _RecoveryClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def _valid_observation(observed_at_utc):
    return {
        "observed_at_utc": observed_at_utc,
        "hardware": {
            "chip": "Apple M5",
            "cpu_cores": 10,
            "gpu_cores": 10,
            "unified_memory_bytes": 24 * 1024**3,
            "macos_build": "25F84",
        },
        "environment_status": {
            "power_connected": True,
            "power_mode": "automatic",
            "low_power_mode": False,
            "thermal_state": "nominal",
            "thermal_state_raw_value": 0,
            "memory_pressure": "normal",
            "memory_free_percentage": 50,
            "competing_gpu_workload": False,
        },
        "software_versions": {
            "python": "3.12.16",
            "mlx": "0.32.4",
            "numpy": "2.5.4",
            "sentencepiece": "0.2.2",
        },
    }


def _valid_trial_payload(workload, metric="prepared-data", pair_index=0):
    work_unit = next(unit for unit in workload.work_units if unit.metric == metric)
    representation_suffix = {
        "swag-end-to-end": "e",
        "inference-prefill": "f",
        "inference-decode": "f",
    }.get(metric, "d")
    return {
        "metric": metric,
        "side": "reference",
        "attempt_index": 0,
        "pair_index": pair_index,
        "process_order": 0,
        "source_commit": "c" * 40,
        "source_clean": True,
        "harness_commit": "a" * 40,
        "harness_clean": True,
        "harness_identity": "sha256:" + "b" * 64,
        "canonical_workload_identity": canonical_workload_identity(workload),
        "native_configuration": {
            "rope_scaling_factor": 1.0,
            "canonical_projection_identity": structured_identity(
                "sml-benchmark-metric-projection-v1",
                canonical_metric_projection(metric, workload),
            ),
            "parameter_precision_policy": PRECISION_POLICY,
        },
        "native_representation_identity": "sha256:" + representation_suffix * 64,
        "canonical_row_identity": workload.semantic_identities[
            "canonical_training_rows"
        ],
        "canonical_input_identity": canonical_input_identity(metric, workload),
        "canonical_projection": canonical_metric_projection(metric, workload),
        "execution_order_identity": canonical_execution_order_identity(
            metric, workload
        ),
        "initial_parameter_identity": "sha256:" + "c" * 64,
        "comparison_target": "baseline",
        "warmup_units": 0 if metric == "compile-cold-start" else 5,
        "measured_units": work_unit.measured_units,
        "elapsed_seconds": 2.0,
        "value": 50.0,
        "startup_verification_seconds": 0.1,
        "compilation_seconds": None,
        "peak_memory_bytes": 1_024,
        "synchronization_boundaries": list(workload.synchronization_boundaries),
    }


def _valid_child_measurement(
    workload,
    metric="prepared-data",
    pair_index=0,
    session_identity="sha256:" + "9" * 64,
    journal_attempt_index=0,
):
    return build_child_trial_measurement(
        session_identity=session_identity,
        journal_attempt_index=journal_attempt_index,
        trial=_valid_trial_payload(workload, metric, pair_index),
        start=_valid_observation("2026-08-08T00:00:00+00:00"),
        end=_valid_observation("2026-08-08T00:00:01+00:00"),
    )


def _valid_post_exit_observation(measurement):
    observation = _valid_observation("2026-08-08T00:00:02+00:00")
    return build_post_exit_observation(measurement=measurement, **observation)


def _valid_recovery_policy(workload):
    return post_exit_recovery_policy(workload)


def _recovery_evidence(
    workload,
    *,
    immediate_pressure="normal",
    sample_pressures=(),
    sample_elapsed=(),
    sample_status_changes=(),
    outcome=None,
    failure_fields=(),
):
    if len(sample_pressures) != len(sample_elapsed):
        raise ValueError("sample pressure and elapsed fixtures must have equal length")
    if sample_status_changes and len(sample_status_changes) != len(sample_pressures):
        raise ValueError("sample status changes must match the sample count")
    measurement = _valid_child_measurement(workload)
    immediate = _valid_observation("2026-08-09T00:00:02+00:00")
    immediate["environment_status"]["memory_pressure"] = immediate_pressure
    post_exit = build_post_exit_observation(measurement=measurement, **immediate)
    samples = []
    previous_identity = None
    changes = sample_status_changes or ({},) * len(sample_pressures)
    for index, (pressure, elapsed, status_changes) in enumerate(
        zip(sample_pressures, sample_elapsed, changes, strict=True)
    ):
        observation = _valid_observation("2026-08-09T00:00:03+00:00")
        observation["environment_status"].update(
            memory_pressure=pressure, **status_changes
        )
        sample = build_post_exit_recovery_sample(
            measurement=measurement,
            post_exit=post_exit,
            sample_index=index,
            previous_sample_identity=previous_identity,
            elapsed_seconds=elapsed,
            **observation,
        )
        samples.append(sample)
        previous_identity = sample["identity"]
    resolved_outcome = outcome or (
        "not-required" if immediate_pressure == "normal" else "recovered"
    )
    recovery = build_post_exit_recovery(
        measurement=measurement,
        post_exit=post_exit,
        samples=samples,
        policy=post_exit_recovery_policy(workload),
        outcome=resolved_outcome,
        duration_seconds=0.0 if not samples else samples[-1]["elapsed_seconds"],
        failure_fields=failure_fields,
        completion_source=(
            "crash-reconstruction" if resolved_outcome == "interrupted" else "live"
        ),
    )
    trial = finalize_raw_trial(measurement, post_exit, samples, recovery)
    return measurement, post_exit, tuple(samples), recovery, trial


def _valid_post_exit_recovery(measurement, post_exit, samples=()):
    workload = build_canonical_workload()
    start = measurement["start"]
    status = post_exit["environment_status"]
    failure_fields = set()
    if post_exit["hardware"] != start["hardware"]:
        failure_fields.add("hardware")
    if post_exit["software_versions"] != start["software_versions"]:
        failure_fields.add("software_versions")
    for field in (
        "power_connected",
        "power_mode",
        "low_power_mode",
        "thermal_state",
        "competing_gpu_workload",
    ):
        if status[field] != workload.required_environment[field]:
            failure_fields.add(field)
    immediate_pressure = post_exit["environment_status"]["memory_pressure"]
    outcome = (
        "environment-failure"
        if failure_fields
        else {
            "normal": "not-required",
            "warning": "interrupted",
            "critical": "critical",
        }[immediate_pressure]
    )
    return build_post_exit_recovery(
        measurement=measurement,
        post_exit=post_exit,
        samples=samples,
        policy=post_exit_recovery_policy(workload),
        outcome=outcome if not samples else "recovered",
        duration_seconds=0.0 if not samples else samples[-1]["elapsed_seconds"],
        failure_fields=tuple(sorted(failure_fields)),
        completion_source=(
            "crash-reconstruction" if outcome == "interrupted" else "live"
        ),
    )


def _valid_recovered_raw_trial(workload):
    return _recovery_evidence(
        workload,
        immediate_pressure="warning",
        sample_pressures=("normal",) * 7,
        sample_elapsed=(5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0),
        outcome="recovered",
    )[4]


def _valid_raw_trial(workload, metric="prepared-data", pair_index=0):
    measurement = _valid_child_measurement(workload, metric, pair_index)
    post_exit = _valid_post_exit_observation(measurement)
    recovery = _valid_post_exit_recovery(measurement, post_exit)
    return finalize_raw_trial(measurement, post_exit, (), recovery)


def _with_trial_payload(trial, **changes):
    measurement = json.loads(json.dumps(trial.child_measurement))
    measurement["trial"].update(changes)
    measurement_body = {
        key: value for key, value in measurement.items() if key != "identity"
    }
    measurement["identity"] = structured_identity(
        "sml-child-trial-measurement-v1", measurement_body
    )
    post_exit = json.loads(json.dumps(trial.post_exit_observation))
    post_exit["metric"] = measurement["trial"]["metric"]
    post_exit["pair_index"] = measurement["trial"]["pair_index"]
    post_exit["child_measurement_identity"] = measurement["identity"]
    post_exit_body = {
        key: value for key, value in post_exit.items() if key != "identity"
    }
    post_exit["identity"] = structured_identity(
        "sml-parent-post-exit-observation-v1", post_exit_body
    )
    samples = []
    previous_identity = None
    for raw_sample in trial.post_exit_recovery_samples:
        sample = build_post_exit_recovery_sample(
            measurement=measurement,
            post_exit=post_exit,
            sample_index=raw_sample["sample_index"],
            previous_sample_identity=previous_identity,
            observed_at_utc=raw_sample["observed_at_utc"],
            elapsed_seconds=raw_sample["elapsed_seconds"],
            hardware=raw_sample["hardware"],
            environment_status=raw_sample["environment_status"],
            software_versions=raw_sample["software_versions"],
        )
        samples.append(sample)
        previous_identity = sample["identity"]
    raw_recovery = trial.post_exit_recovery
    recovery = build_post_exit_recovery(
        measurement=measurement,
        post_exit=post_exit,
        samples=samples,
        policy=raw_recovery["policy"],
        outcome=raw_recovery["outcome"],
        duration_seconds=raw_recovery["duration_seconds"],
        failure_fields=raw_recovery["failure_fields"],
        completion_source=raw_recovery["completion_source"],
    )
    return finalize_raw_trial(measurement, post_exit, samples, recovery)


def _validate_trial_metadata(protocol, workload, reference):
    if protocol == "baseline":
        validate_baseline_trial(
            reference,
            workload=workload,
            source_commit=reference.source_commit,
            harness_commit=reference.harness_commit,
            harness_identity=reference.harness_identity,
            expected_hardware=reference.hardware,
            expected_software_versions=reference.software_versions,
        )
        return
    candidate = _with_trial_payload(
        reference, side="candidate", process_order=1, source_commit="e" * 40
    )
    validator = {
        "comparison": benchmark_runner._validate_comparison_trial_pair,
        "predecessor": benchmark_runner._validate_predecessor_trial_pair,
    }[protocol]
    validator(
        reference_trial=reference,
        candidate_trial=candidate,
        pair_index=0,
        attempt_index=0,
        baseline={
            "harness": {
                "commit": reference.harness_commit,
                "content_identity": reference.harness_identity,
            },
            "canonical_workload_identity": canonical_workload_identity(workload),
        },
        workload=workload,
        expected_projection=canonical_metric_projection(reference.metric, workload),
        expected_input_identity=canonical_input_identity(reference.metric, workload),
        expected_warmup=reference.warmup_units,
        expected_units=reference.measured_units,
    )


def _with_observation_changes(
    trial, *, environment_changes=None, software_changes=None, hardware_changes=None
):
    environment_changes = environment_changes or {}
    software_changes = software_changes or {}
    hardware_changes = hardware_changes or {}
    measurement = json.loads(json.dumps(trial.child_measurement))
    for endpoint in ("start", "end"):
        measurement[endpoint]["environment_status"].update(environment_changes)
        measurement[endpoint]["software_versions"].update(software_changes)
        measurement[endpoint]["hardware"].update(hardware_changes)
    measurement_body = {
        key: value for key, value in measurement.items() if key != "identity"
    }
    measurement["identity"] = structured_identity(
        "sml-child-trial-measurement-v1", measurement_body
    )
    post_exit = json.loads(json.dumps(trial.post_exit_observation))
    post_exit["environment_status"].update(environment_changes)
    post_exit["software_versions"].update(software_changes)
    post_exit["hardware"].update(hardware_changes)
    updated_post_exit = build_post_exit_observation(
        measurement=measurement,
        observed_at_utc=post_exit["observed_at_utc"],
        hardware=post_exit["hardware"],
        environment_status=post_exit["environment_status"],
        software_versions=post_exit["software_versions"],
    )
    recovery = _valid_post_exit_recovery(measurement, updated_post_exit)
    return finalize_raw_trial(measurement, updated_post_exit, (), recovery)


def _with_environment_observations(trial, *, start=None, end=None, post_exit=None):
    measurement = json.loads(json.dumps(trial.child_measurement))
    parent = json.loads(json.dumps(trial.post_exit_observation))
    if start is not None:
        measurement["start"]["environment_status"] = start
    if end is not None:
        measurement["end"]["environment_status"] = end
    measurement_body = {
        key: value for key, value in measurement.items() if key != "identity"
    }
    measurement["identity"] = structured_identity(
        "sml-child-trial-measurement-v1", measurement_body
    )
    parent["child_measurement_identity"] = measurement["identity"]
    if post_exit is not None:
        parent["environment_status"] = post_exit
    parent_body = {key: value for key, value in parent.items() if key != "identity"}
    parent["identity"] = structured_identity(
        "sml-parent-post-exit-observation-v1", parent_body
    )
    recovery = _valid_post_exit_recovery(measurement, parent)
    return finalize_raw_trial(measurement, parent, (), recovery)


def _with_thermal_state(trial, state, raw_value):
    return _with_observation_changes(
        trial,
        environment_changes={
            "thermal_state": state,
            "thermal_state_raw_value": raw_value,
        },
    )


def _valid_baseline_trials(workload):
    return tuple(
        _valid_raw_trial(workload, metric=metric, pair_index=pair_index)
        for metric in METRIC_NAMES
        for pair_index in range(5)
    )


def _baseline_fixture():
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
    return manifest, trials


def _comparison_trials(
    workload,
    candidate_commit,
    candidate_values,
    attempt_index,
    metric="prepared-data",
):
    trials = []
    for pair_index, candidate_value in enumerate(candidate_values):
        order = process_order(pair_index)
        reference = _valid_raw_trial(workload, metric=metric, pair_index=pair_index)
        trials.extend(
            (
                _with_trial_payload(
                    reference,
                    attempt_index=attempt_index,
                    process_order=order.index("reference"),
                    value=100.0,
                ),
                _with_trial_payload(
                    reference,
                    side="candidate",
                    attempt_index=attempt_index,
                    process_order=order.index("candidate"),
                    source_commit=candidate_commit,
                    native_configuration={
                        **reference.native_configuration,
                        "parameter_precision_policy": PRECISION_POLICY,
                    },
                    value=candidate_value,
                ),
            )
        )
    return trials


def _valid_prepared_comparison():
    workload = build_canonical_workload()
    baseline_trials = _valid_baseline_trials(workload)
    baseline = build_baseline_manifest(
        trials=baseline_trials,
        workload=workload,
        workload_identity=canonical_workload_identity(workload),
        source_commit=baseline_trials[0].source_commit,
        harness_commit=baseline_trials[0].harness_commit,
        harness_identity=baseline_trials[0].harness_identity,
        command="record-baseline",
        pairs=5,
        warmup_units=5,
        measured_units=20,
    )
    candidate_commit = "e" * 40
    trials = _comparison_trials(workload, candidate_commit, [100.0] * 5, 0)
    report = build_comparison_report(
        baseline=baseline,
        trials=trials,
        candidate_commit=candidate_commit,
        minimum_ratio=0.97,
        pretraining_minimum_ratio=None,
        maximum_dispersion=0.02,
        require_lower_bound=False,
        bootstrap_resamples=10_000,
        predecessor_metrics={},
    )
    return workload, baseline, report


def _nominal_cooldown_evidence():
    status = {
        "power_connected": True,
        "power_mode": "automatic",
        "low_power_mode": False,
        "thermal_state": "nominal",
        "memory_pressure": "normal",
        "competing_gpu_workload": False,
    }
    return {
        "duration_seconds": 900.0,
        "sample_interval_seconds": 60.0,
        "samples": [
            {"elapsed_seconds": float(elapsed), "environment_status": status}
            for elapsed in range(0, 901, 60)
        ],
    }


def _session_document(
    tmp_path,
    *,
    harness_commit="a" * 40,
    protocol=None,
    hardware=None,
    software_versions=None,
    manifest_name="baseline.json",
    raw_output_name="baseline.jsonl",
):
    workload = build_canonical_workload()
    return build_session_document(
        harness_commit=harness_commit,
        harness_identity="sha256:" + "b" * 64,
        source_commit="c" * 40,
        canonical_workload=workload,
        canonical_workload_identity=canonical_workload_identity(workload),
        protocol=protocol or {"pairs": 5, "warmup_units": 5, "measured_units": 20},
        hardware=hardware or {"chip": "Apple M5"},
        software_versions=software_versions or {"python": "3.12.16", "mlx": "0.32.4"},
        manifest_path=tmp_path / manifest_name,
        raw_output_path=tmp_path / raw_output_name,
    )


def _persist_journal_trial(journal, attempt, trial):
    return _persist_journal_evidence(journal, attempt, trial)[4]


def _preflight_from_trial(trial):
    return (
        trial.hardware,
        trial.environment_status["start"],
        trial.software_versions,
    )


def _baseline_validator(workload):
    expected = _valid_raw_trial(workload)

    def validate(trial, *, allow_rejected_environment):
        validate_baseline_trial(
            trial,
            workload=workload,
            source_commit=expected.source_commit,
            harness_commit=expected.harness_commit,
            harness_identity=expected.harness_identity,
            expected_hardware=expected.hardware,
            expected_software_versions=expected.software_versions,
            allow_rejected_environment=allow_rejected_environment,
        )

    return validate


def _accept_trial(_trial):
    return benchmark_runner.TrialEnvironmentDisposition("accept", None)


def _journal_trial_evidence(journal, attempt, trial=None):
    workload = build_canonical_workload()
    if trial is None:
        trial = _valid_recovered_raw_trial(workload)
    measurement = json.loads(json.dumps(trial.child_measurement))
    measurement["session_identity"] = journal.session["identity"]
    measurement["journal_attempt_index"] = attempt.journal_attempt_index
    measurement_body = {
        key: value for key, value in measurement.items() if key != "identity"
    }
    measurement["identity"] = structured_identity(
        "sml-child-trial-measurement-v1", measurement_body
    )
    post_exit = json.loads(json.dumps(trial.post_exit_observation))
    post_exit["session_identity"] = journal.session["identity"]
    post_exit["journal_attempt_index"] = attempt.journal_attempt_index
    post_exit["child_measurement_identity"] = measurement["identity"]
    post_exit_body = {
        key: value for key, value in post_exit.items() if key != "identity"
    }
    post_exit["identity"] = structured_identity(
        "sml-parent-post-exit-observation-v1", post_exit_body
    )
    samples = []
    previous_identity = None
    for raw_sample in trial.post_exit_recovery_samples:
        sample = build_post_exit_recovery_sample(
            measurement=measurement,
            post_exit=post_exit,
            sample_index=raw_sample["sample_index"],
            previous_sample_identity=previous_identity,
            observed_at_utc=raw_sample["observed_at_utc"],
            elapsed_seconds=raw_sample["elapsed_seconds"],
            hardware=raw_sample["hardware"],
            environment_status=raw_sample["environment_status"],
            software_versions=raw_sample["software_versions"],
        )
        samples.append(sample)
        previous_identity = sample["identity"]
    raw_recovery = trial.post_exit_recovery
    recovery = build_post_exit_recovery(
        measurement=measurement,
        post_exit=post_exit,
        samples=samples,
        policy=raw_recovery["policy"],
        outcome=raw_recovery["outcome"],
        duration_seconds=raw_recovery["duration_seconds"],
        failure_fields=raw_recovery["failure_fields"],
        completion_source=raw_recovery["completion_source"],
    )
    return (
        measurement,
        post_exit,
        tuple(samples),
        recovery,
        finalize_raw_trial(measurement, post_exit, samples, recovery),
    )


def _persist_journal_evidence(
    journal,
    attempt,
    trial=None,
    *,
    measurement=True,
    post_exit=True,
    recovery=True,
    inflight=True,
):
    child, parent, samples, summary, persisted = _journal_trial_evidence(
        journal, attempt, trial
    )
    if measurement:
        atomic_write_json(
            journal.measurement_path(attempt.slot, attempt.journal_attempt_index),
            child,
            create_only=True,
        )
    if post_exit:
        atomic_write_json(
            journal.post_exit_path(attempt.slot, attempt.journal_attempt_index),
            parent,
            create_only=True,
        )
    if post_exit and recovery:
        for index, sample in enumerate(samples):
            atomic_write_json(
                journal.recovery_sample_path(
                    attempt.slot, attempt.journal_attempt_index, index
                ),
                sample,
                create_only=True,
            )
        atomic_write_json(
            journal.recovery_path(attempt.slot, attempt.journal_attempt_index),
            summary,
            create_only=True,
        )
    if inflight:
        atomic_write_json(attempt.path, persisted.to_dict(), create_only=True)
    return child, parent, samples, summary, persisted


def tiny_training_workload(*, row_count=6):
    return build_canonical_workload(
        model_overrides={
            "vocab_size": 32,
            "hidden_size": 16,
            "num_layers": 1,
            "num_q_heads": 4,
            "num_kv_heads": 2,
            "intermediate_size": 32,
            "original_context_length": 32,
            "hidden_dropout": 0.0,
        },
        optimizer_overrides={
            "gradient_accumulation_steps": 2,
            "swag": {"gradient_accumulation_steps": 2},
        },
        loader_overrides={
            "sequence_length": 4,
            "microbatch_size": 2,
            "swag": {"sequence_length": 8, "batch_size": 1},
        },
        generation_overrides={
            "request_count": 2,
            "prompt_tokens": 4,
            "decode_chunk_size": 2,
        },
        row_count=row_count,
    )
