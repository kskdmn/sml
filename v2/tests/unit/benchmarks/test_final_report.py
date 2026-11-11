import json
from types import SimpleNamespace

import pytest

import v2.benchmarks.runner as benchmark_runner
from v2.benchmarks.journal import (
    read_json_object,
)
from v2.benchmarks.runner import (
    _resolve_predecessor_mapping,
    build_baseline_manifest,
    build_comparison_report,
    validate_comparison_report,
    validate_final_report,
)
from v2.benchmarks.schema import METRIC_NAMES
from v2.benchmarks.workload import (
    build_canonical_workload,
    canonical_workload_identity,
    structured_identity,
)

from .support import (
    _baseline_fixture,
    _comparison_trials,
    _nominal_cooldown_evidence,
    _valid_baseline_trials,
    _valid_prepared_comparison,
    _valid_raw_trial,
)


def test_predecessors_are_resolved_as_an_explicit_per_metric_mapping(tmp_path):
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

    def write_report(metric, commit, path):
        trials = _comparison_trials(workload, commit, [100.0] * 5, 0, metric=metric)
        report = build_comparison_report(
            baseline=baseline,
            trials=trials,
            candidate_commit=commit,
            minimum_ratio=0.97,
            pretraining_minimum_ratio=None,
            maximum_dispersion=0.02,
            require_lower_bound=False,
            bootstrap_resamples=10_000,
            predecessor_metrics={},
        )
        path.write_text(json.dumps(report), encoding="utf-8")
        return report

    prepared_path = tmp_path / "prepared.json"
    inference_path = tmp_path / "inference.json"
    prepared = write_report("prepared-data", "e" * 40, prepared_path)
    inference = write_report("inference-prefill", "f" * 40, inference_path)

    reports, lineages, proof = _resolve_predecessor_mapping(
        json.dumps(
            {
                "prepared-data": str(prepared_path),
                "inference-prefill": str(inference_path),
            }
        ),
        tmp_path,
        baseline,
        ("prepared-data", "inference-prefill"),
    )

    assert reports["prepared-data"]["identity"] == prepared["identity"]
    assert reports["inference-prefill"]["identity"] == inference["identity"]
    assert lineages["prepared-data"]["metric"] == "prepared-data"
    assert proof["inference-prefill"]["report_identity"] == inference["identity"]

    forged = json.loads(json.dumps(prepared))
    forged["raw_trials"][0]["value"] = 1.0
    body = {key: value for key, value in forged.items() if key != "identity"}
    forged["identity"] = structured_identity("sml-performance-comparison-v1", body)
    forged_path = tmp_path / "forged.json"
    forged_path.write_text(json.dumps(forged), encoding="utf-8")
    with pytest.raises(ValueError, match="embedded evidence"):
        _resolve_predecessor_mapping(
            json.dumps(
                {
                    "prepared-data": str(forged_path),
                    "inference-prefill": None,
                }
            ),
            tmp_path,
            baseline,
            ("prepared-data", "inference-prefill"),
        )

    forged_lineage = json.loads(json.dumps(prepared))
    forged_lineage["latest_metrics"]["prepared-data"]["source_commit"] = "1" * 40
    body = {key: value for key, value in forged_lineage.items() if key != "identity"}
    forged_lineage["identity"] = structured_identity(
        "sml-performance-comparison-v1", body
    )
    forged_lineage_path = tmp_path / "forged-lineage.json"
    forged_lineage_path.write_text(json.dumps(forged_lineage), encoding="utf-8")
    with pytest.raises(ValueError, match="latest-metric lineage"):
        _resolve_predecessor_mapping(
            json.dumps(
                {
                    "prepared-data": str(forged_lineage_path),
                    "inference-prefill": None,
                }
            ),
            tmp_path,
            baseline,
            ("prepared-data", "inference-prefill"),
        )

    reports, lineages, proof = _resolve_predecessor_mapping(
        json.dumps({"prepared-data": str(prepared_path), "inference-prefill": None}),
        tmp_path,
        baseline,
        ("prepared-data", "inference-prefill"),
    )
    assert reports["inference-prefill"] is None
    assert "inference-prefill" not in lineages
    assert proof["inference-prefill"] is None

    with pytest.raises(ValueError, match="wrong-metric or stale"):
        _resolve_predecessor_mapping(
            json.dumps(
                {
                    "prepared-data": str(prepared_path),
                    "inference-prefill": str(prepared_path),
                }
            ),
            tmp_path,
            baseline,
            ("prepared-data", "inference-prefill"),
        )


def _with_tampered_report_post_exit_evidence(report):
    invalid = json.loads(json.dumps(report))
    invalid["raw_trials"][0]["post_exit_observation"]["child_measurement_identity"] = (
        "sha256:" + "0" * 64
    )
    body = {key: value for key, value in invalid.items() if key != "identity"}
    invalid["identity"] = structured_identity("sml-performance-comparison-v1", body)
    return invalid


def test_validate_phase_rejects_embedded_evidence_before_predecessor_lookup(
    tmp_path,
):
    _workload, baseline, report = _valid_prepared_comparison()
    baseline_path = tmp_path / "baseline.json"
    results_path = tmp_path / "phase.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    results_path.write_text(
        json.dumps(_with_tampered_report_post_exit_evidence(report)),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        baseline=baseline_path,
        results=results_path,
        predecessors=json.dumps(
            {"prepared-data": str(tmp_path / "missing-predecessor.json")}
        ),
        phase=2,
        output=None,
    )

    with pytest.raises(ValueError, match="post-exit observation identity"):
        benchmark_runner._validate_phase(args)


def test_validate_final_rejects_embedded_evidence_before_predecessor_lookup(
    tmp_path,
):
    _workload, baseline, report = _valid_prepared_comparison()
    report["predecessors"] = {
        metric: (
            {
                "report_identity": "sha256:" + "0" * 64,
                "result_identity": "sha256:" + "1" * 64,
            }
            if metric in benchmark_runner.FINAL_PREDECESSOR_METRICS
            else None
        )
        for metric in benchmark_runner.FINAL_METRICS
    }
    invalid = _with_tampered_report_post_exit_evidence(report)
    baseline_path = tmp_path / "baseline.json"
    report_path = tmp_path / "final.json"
    raw_path = tmp_path / "final.jsonl"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    report_path.write_text(json.dumps(invalid), encoding="utf-8")
    raw_path.write_text(
        "".join(json.dumps(raw) + "\n" for raw in invalid["raw_trials"]),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        baseline=baseline_path,
        report=report_path,
        raw_input=raw_path,
    )

    with pytest.raises(ValueError, match="post-exit observation identity"):
        benchmark_runner._validate_final(args)


def test_final_validation_requires_complete_raw_input_and_passing_gates(monkeypatch):
    workload = build_canonical_workload()
    trial = _valid_raw_trial(workload)
    with pytest.raises(ValueError, match="complete report"):
        validate_final_report({"raw_trials": []}, {}, {}, (trial,))

    proof = {
        "report_identity": "sha256:" + "a" * 64,
        "result_identity": "sha256:" + "b" * 64,
    }
    metrics = {
        metric: {
            "baseline_comparison": {
                "decision": "fail" if metric == "prepared-data" else "pass"
            },
            "previous_comparison": None,
        }
        for metric in benchmark_runner.FINAL_METRICS
    }
    report = {
        "raw_trials": [],
        "metrics": metrics,
        "comparison_mode": benchmark_runner.COMPARISON_FINAL,
        "predecessors": {
            metric: proof
            if metric in benchmark_runner.FINAL_PREDECESSOR_METRICS
            else None
            for metric in benchmark_runner.FINAL_METRICS
        },
    }
    monkeypatch.setattr(benchmark_runner, "validate_comparison_report", lambda *_: None)

    with pytest.raises(ValueError, match="final acceptance baseline gate failed"):
        validate_final_report(
            report,
            {},
            {metric: None for metric in benchmark_runner.FINAL_METRICS},
            (),
        )


@pytest.mark.parametrize(
    ("rejected_metric", "candidate_values", "decision"),
    (
        ("pretraining-end-to-end", [102.0] * 10, "fail"),
        (
            "pretraining-end-to-end",
            [102.0, 102.5, 102.8, 102.9, 103.0, 103.0, 103.2, 103.5, 104.0, 104.5],
            "inconclusive",
        ),
        ("peak-metal-memory", [95.0, 100.0, 103.0, 106.0, 110.0] * 2, "too-noisy"),
    ),
)
def test_rejected_final_comparison_publishes_complete_valid_evidence(
    tmp_path, monkeypatch, rejected_metric, candidate_values, decision
):
    workload = build_canonical_workload()
    baseline, _baseline_trials = _baseline_fixture()
    baseline_commit = baseline["source"]["commit"]
    candidate_commit = "e" * 40
    metrics = benchmark_runner.FINAL_METRICS
    predecessor_metrics = benchmark_runner.FINAL_PREDECESSOR_METRICS
    predecessor = build_comparison_report(
        baseline=baseline,
        trials=[
            trial
            for metric in metrics
            if metric in predecessor_metrics
            for trial in _comparison_trials(
                workload, baseline_commit, [100.0] * 5, 0, metric=metric
            )
        ],
        candidate_commit=baseline_commit,
        minimum_ratio=0.97,
        pretraining_minimum_ratio=None,
        maximum_dispersion=0.02,
        require_lower_bound=False,
        bootstrap_resamples=10_000,
        predecessor_metrics={},
    )
    baseline_path = tmp_path / "baseline.json"
    predecessor_path = tmp_path / "predecessor.json"
    benchmark_runner._write_json(baseline_path, baseline)
    benchmark_runner._write_json(predecessor_path, predecessor)
    args = SimpleNamespace(
        baseline=baseline_path,
        candidate=candidate_commit,
        metrics=metrics,
        mode="final",
        pairs=10,
        warmup=5,
        measure=20,
        prepared_data_measure=100,
        bootstrap_resamples=10_000,
        minimum_ratio=0.97,
        pretraining_minimum_ratio=1.03,
        maximum_dispersion=0.015,
        lower_bound_report_only=False,
        predecessors=json.dumps(
            {
                metric: str(predecessor_path) if metric in predecessor_metrics else None
                for metric in metrics
            }
        ),
        raw_output=tmp_path / "final.jsonl",
        output=tmp_path / "final.json",
    )
    collected_trials = []

    def collect_attempt(*, attempt_index, **_kwargs):
        trials = []
        for metric in metrics:
            values = (
                candidate_values
                if metric == rejected_metric
                else [104.0 if metric == "pretraining-end-to-end" else 100.0] * 10
            )
            trials.extend(
                _comparison_trials(
                    workload, candidate_commit, values, attempt_index, metric=metric
                )
            )
        collected_trials.extend(trials)
        return trials, {}

    monkeypatch.setattr(benchmark_runner, "_git_root", lambda _path: tmp_path)
    monkeypatch.setattr(
        benchmark_runner, "_require_clean_checkout", lambda *_a, **_k: None
    )
    monkeypatch.setattr(benchmark_runner, "_git_commit", lambda *_a: candidate_commit)
    monkeypatch.setattr(
        benchmark_runner,
        "harness_content_identity",
        lambda _root: baseline["harness"]["content_identity"],
    )
    monkeypatch.setattr(
        benchmark_runner, "_collect_comparison_attempt", collect_attempt
    )
    monkeypatch.setattr(
        benchmark_runner, "perform_cooldown", _nominal_cooldown_evidence
    )

    with pytest.raises(ValueError, match=f"baseline gate failed for {rejected_metric}"):
        benchmark_runner._compare(args)

    report = benchmark_runner._read_json_object(args.output, label="final report")
    raw_trials = benchmark_runner._read_trials(args.raw_output)
    predecessors = {
        metric: predecessor if metric in predecessor_metrics else None
        for metric in metrics
    }
    validate_comparison_report(report, baseline, predecessors)
    assert raw_trials == tuple(collected_trials)
    assert report["raw_trials"] == [trial.to_dict() for trial in collected_trials]
    assert (
        report["metrics"][rejected_metric]["baseline_comparison"]["decision"]
        == decision
    )
    attempts = 2 if decision == "too-noisy" else 1
    assert all(
        len(record["attempts"]) == attempts for record in report["metrics"].values()
    )
    assert report["cooldown_evidence"] == (
        _nominal_cooldown_evidence() if attempts == 2 else None
    )
    with pytest.raises(ValueError, match=f"baseline gate failed for {rejected_metric}"):
        validate_final_report(report, baseline, predecessors, raw_trials)


@pytest.mark.parametrize("phase", (1, 3, 4, "final"))
@pytest.mark.parametrize("mutation", (None, "missing", "extra"))
def test_phase_and_final_metric_contract_survives_sorted_json_publication(
    tmp_path, monkeypatch, phase, mutation
):
    baseline, _trials = _baseline_fixture()
    final = phase == "final"
    expected = (
        benchmark_runner.FINAL_METRICS
        if final
        else benchmark_runner.PHASE_METRICS[phase]
    )
    required_predecessors = (
        benchmark_runner.FINAL_PREDECESSOR_METRICS
        if final
        else benchmark_runner.PHASE_PREDECESSOR_METRICS[phase]
    )
    metric_names = list(expected)
    if mutation == "missing":
        metric_names.pop()
    elif mutation == "extra":
        metric_names.append(
            next(metric for metric in METRIC_NAMES if metric not in expected)
        )
    report = {
        "raw_trials": [],
        "metrics": {
            metric: {
                "baseline_comparison": {"decision": "pass"},
                "previous_comparison": None,
            }
            for metric in metric_names
        },
        "comparison_mode": "final" if final else "screen",
        "predecessors": {
            metric: {"report_identity": "sha256:" + "a" * 64}
            if metric in required_predecessors
            else None
            for metric in expected
        },
    }
    baseline_path = tmp_path / "baseline.json"
    report_path = tmp_path / "report.json"
    benchmark_runner._write_json(baseline_path, baseline)
    benchmark_runner._write_json(report_path, report)
    saved = read_json_object(report_path, label="report")
    if mutation is None:
        assert tuple(saved["metrics"]) != tuple(expected)
    # Isolate the phase/final metric boundary; comparison evidence and lineage
    # validation have their own complete-report behavioral tests above.
    monkeypatch.setattr(benchmark_runner, "validate_comparison_report", lambda *_: None)
    monkeypatch.setattr(benchmark_runner, "_git_root", lambda _: tmp_path)
    monkeypatch.setattr(
        benchmark_runner,
        "_resolve_predecessor_mapping",
        lambda *_: (report["predecessors"], {}, {}),
    )

    def validate():
        if final:
            return validate_final_report(saved, baseline, report["predecessors"], ())
        return benchmark_runner._validate_phase(
            SimpleNamespace(
                baseline=baseline_path,
                results=report_path,
                predecessors="{}",
                phase=phase,
                output=None,
            )
        )

    if mutation is None:
        validate()
    else:
        with pytest.raises(ValueError, match="wrong metric set"):
            validate()
