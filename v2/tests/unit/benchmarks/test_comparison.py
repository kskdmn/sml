import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

import v2.benchmarks.runner as benchmark_runner
from v2.benchmarks.runner import (
    build_baseline_manifest,
    build_comparison_report,
    build_parser,
    comparison_has_noise,
    process_order,
    validate_comparison_report,
    validate_throughput_gates,
)
from v2.benchmarks.schema import METRIC_NAMES, CanonicalWorkload, RawTrial
from v2.benchmarks.workload import (
    PRECISION_POLICY,
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
    _valid_recovered_raw_trial,
    _with_trial_payload,
)


def test_comparison_report_pins_pairs_decisions_and_metric_lineage():
    workload = build_canonical_workload()
    baseline_trials = _valid_baseline_trials(workload)
    baseline_trial = baseline_trials[0]
    baseline_manifest = build_baseline_manifest(
        trials=baseline_trials,
        workload=workload,
        workload_identity=canonical_workload_identity(workload),
        source_commit=baseline_trial.source_commit,
        harness_commit=baseline_trial.harness_commit,
        harness_identity=baseline_trial.harness_identity,
        command="record-baseline",
        pairs=5,
        warmup_units=5,
        measured_units=20,
    )
    trials = []
    candidate_commit = "e" * 40
    for pair_index in range(5):
        order = process_order(pair_index)
        reference_template = _valid_raw_trial(
            workload, metric="pretraining-end-to-end", pair_index=pair_index
        )
        trials.extend(
            (
                _with_trial_payload(
                    reference_template,
                    process_order=order.index("reference"),
                    value=100.0,
                ),
                _with_trial_payload(
                    reference_template,
                    side="candidate",
                    process_order=order.index("candidate"),
                    source_commit=candidate_commit,
                    native_configuration={
                        **reference_template.native_configuration,
                        "parameter_precision_policy": PRECISION_POLICY,
                    },
                    value=103.0,
                ),
            )
        )

    report = build_comparison_report(
        baseline=baseline_manifest,
        trials=tuple(trials),
        candidate_commit=candidate_commit,
        minimum_ratio=0.97,
        pretraining_minimum_ratio=1.03,
        maximum_dispersion=0.02,
        require_lower_bound=False,
        bootstrap_resamples=10_000,
        predecessor_metrics={},
    )

    validate_comparison_report(report, baseline_manifest, predecessor_reports=None)
    validate_comparison_report(
        json.loads(json.dumps(report)),
        baseline_manifest,
        predecessor_reports=None,
    )
    metric = report["metrics"]["pretraining-end-to-end"]
    assert metric["baseline_comparison"]["median_ratio"] == 1.03
    assert metric["baseline_comparison"]["decision"] == "pass"
    assert metric["previous_comparison"] is None
    assert (
        report["latest_metrics"]["pretraining-end-to-end"]["result_identity"]
        == metric["result_identity"]
    )

    invalid = json.loads(json.dumps(report))
    invalid["raw_trials"][1] = _with_trial_payload(
        RawTrial.from_dict(invalid["raw_trials"][1]), process_order=0
    ).to_dict()
    invalid_body = {key: value for key, value in invalid.items() if key != "identity"}
    invalid["identity"] = structured_identity(
        "sml-performance-comparison-v1", invalid_body
    )
    with pytest.raises(ValueError, match="alternating process order"):
        validate_comparison_report(invalid, baseline_manifest, predecessor_reports=None)


def _resign_comparison(report):
    body = {key: value for key, value in report.items() if key != "identity"}
    report["identity"] = structured_identity("sml-performance-comparison-v1", body)


@pytest.mark.parametrize("matching_harness", [False, True])
def test_compare_requires_the_executing_parent_harness_identity(
    tmp_path, monkeypatch, matching_harness
):
    _workload, baseline, original_report = _valid_prepared_comparison()
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline))
    output = tmp_path / "report.json"
    args = build_parser().parse_args(
        [
            "compare",
            "--baseline",
            str(baseline_path),
            "--candidate",
            "HEAD",
            "--metrics",
            "prepared-data",
            "--mode",
            "screen",
            "--lower-bound-report-only",
            "--predecessors",
            '{"prepared-data":null}',
            "--output",
            str(output),
        ]
    )
    monkeypatch.setattr(benchmark_runner, "_git_root", lambda _path: tmp_path)
    monkeypatch.setattr(
        benchmark_runner, "_require_clean_checkout", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        benchmark_runner,
        "_git_commit",
        lambda *_args: original_report["candidate_commit"],
    )
    verified_roots = []

    def parent_harness_identity(path):
        verified_roots.append(path)
        return (
            baseline["harness"]["content_identity"]
            if matching_harness
            else "sha256:" + "0" * 64
        )

    monkeypatch.setattr(
        benchmark_runner, "harness_content_identity", parent_harness_identity
    )
    captures = []

    def capture(**_kwargs):
        captures.append(True)
        return [RawTrial.from_dict(raw) for raw in original_report["raw_trials"]], {}

    monkeypatch.setattr(benchmark_runner, "_collect_comparison_attempt", capture)
    if matching_harness:
        assert benchmark_runner._compare(args) == 0
        assert captures == [True]
        validate_comparison_report(json.loads(output.read_text()), baseline, None)
    else:
        with pytest.raises(
            ValueError, match="executing comparison harness.*new baseline"
        ):
            benchmark_runner._compare(args)
        assert not captures
        assert not output.exists()
    assert verified_roots == [Path(benchmark_runner.__file__).resolve().parents[2]]


@pytest.mark.parametrize(
    ("protected", "output_field", "alias"),
    (
        ("outputs", "raw_output", "same"),
        ("outputs", "raw_output", "normalized"),
        ("outputs", "raw_output", "casefold"),
        ("outputs", "raw_output", "unicode"),
        ("outputs", "raw_output", "descendant"),
        ("outputs", "output", "descendant"),
        ("outputs", "raw_output", "symlink-parent"),
        ("outputs", "raw_output", "hardlink"),
        ("baseline", "output", "same"),
        ("baseline", "raw_output", "symlink"),
        ("baseline", "output", "hardlink"),
        ("predecessor", "output", "same"),
        ("predecessor", "raw_output", "normalized"),
        ("predecessor-identity", "output", "same"),
        ("mapping", "output", "same"),
        ("mapping", "raw_output", "same"),
    ),
)
def test_compare_rejects_output_aliases_before_measurement(
    tmp_path, monkeypatch, protected, output_field, alias
):
    _workload, baseline, predecessor = _valid_prepared_comparison()
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline))
    predecessor_path = tmp_path / "v2" / "benchmarks" / "results" / "predecessor.json"
    predecessor_path.parent.mkdir(parents=True)
    predecessor_path.write_text(json.dumps(predecessor))
    requested_predecessor = (
        predecessor["identity"]
        if protected == "predecessor-identity"
        else str(predecessor_path)
    )
    mapping = json.dumps({"prepared-data": requested_predecessor})
    mapping_path = tmp_path / "predecessors.json"
    mapping_path.write_text(mapping)
    output_path = tmp_path / ("café.json" if alias == "unicode" else "report.json")
    args = build_parser().parse_args(
        [
            "compare",
            "--baseline",
            str(baseline_path),
            "--candidate",
            "HEAD",
            "--metrics",
            "prepared-data",
            "--mode",
            "screen",
            "--lower-bound-report-only",
            "--predecessors",
            str(mapping_path) if protected == "mapping" else mapping,
            "--output",
            str(output_path),
            "--raw-output",
            str(tmp_path / "raw.jsonl"),
        ]
    )
    protected_path = {
        "outputs": output_path,
        "baseline": baseline_path,
        "predecessor": predecessor_path,
        "predecessor-identity": predecessor_path,
        "mapping": mapping_path,
    }[protected]
    if alias == "same":
        aliased_path = protected_path
    elif alias == "normalized":
        intermediate = protected_path.parent / "intermediate"
        intermediate.mkdir()
        aliased_path = intermediate / ".." / protected_path.name
    elif alias == "casefold":
        aliased_path = protected_path.with_name(protected_path.name.upper())
    elif alias == "unicode":
        aliased_path = protected_path.with_name("cafe\u0301.json")
    elif alias == "descendant":
        if output_field == "output":
            args.raw_output = protected_path
        aliased_path = protected_path / "nested.json"
    elif alias == "symlink-parent":
        linked_parent = tmp_path / "linked-parent"
        linked_parent.symlink_to(protected_path.parent, target_is_directory=True)
        aliased_path = linked_parent / protected_path.name
    else:
        aliased_path = tmp_path / "aliased-output.json"
        if alias == "symlink":
            aliased_path.symlink_to(protected_path)
        else:
            if not protected_path.exists():
                protected_path.write_text("previous output\n")
            os.link(protected_path, aliased_path)
    setattr(args, output_field, aliased_path)
    saved_files = {
        path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()
    }

    def reject_measurement(**_kwargs):
        pytest.fail("measurement started before validating output paths")

    monkeypatch.setattr(benchmark_runner, "_git_root", lambda _path: tmp_path)
    monkeypatch.setattr(
        benchmark_runner, "_require_clean_checkout", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        benchmark_runner, "_git_commit", lambda *_a: predecessor["candidate_commit"]
    )
    monkeypatch.setattr(
        benchmark_runner,
        "harness_content_identity",
        lambda _root: baseline["harness"]["content_identity"],
    )
    monkeypatch.setattr(
        benchmark_runner, "_collect_comparison_attempt", reject_measurement
    )

    with pytest.raises(ValueError, match="comparison output paths must be distinct"):
        benchmark_runner._compare(args)

    assert {
        path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()
    } == saved_files


@pytest.mark.parametrize("version", (True, 1.0))
def test_comparison_validator_rejects_boolean_and_float_version_aliases(version):
    _workload, baseline, report = _valid_prepared_comparison()
    report["version"] = version
    _resign_comparison(report)

    with pytest.raises(ValueError, match="unsupported comparison"):
        validate_comparison_report(report, baseline, None)


def test_comparison_construction_preserves_the_baseline_protocol():
    baseline, _trials = _baseline_fixture()
    workload = CanonicalWorkload.from_dict(baseline["canonical_workload"])
    candidate_commit = "e" * 40
    report = build_comparison_report(
        baseline=baseline,
        trials=_comparison_trials(workload, candidate_commit, [100.0] * 5, 0),
        candidate_commit=candidate_commit,
        minimum_ratio=0.97,
        pretraining_minimum_ratio=None,
        maximum_dispersion=0.02,
        require_lower_bound=False,
        bootstrap_resamples=10_000,
        predecessor_metrics={},
    )

    validate_comparison_report(report, baseline, None)
    expected = {
        "pairs",
        "compilation_passes",
        "warmup_units",
        "measured_units",
        "bootstrap_seed",
        "bootstrap_resamples",
        "minimum_ratio",
        "pretraining_minimum_ratio",
        "maximum_dispersion",
        "require_lower_bound",
    }
    expected.add("prepared_data_measured_units")
    assert set(report["protocol"]) == expected


@pytest.mark.parametrize("mutation", ("omit", "extra"))
def test_comparison_validator_rejects_serialized_protocol_field_changes(
    mutation,
):
    _workload, baseline, report = _valid_prepared_comparison()
    if mutation == "omit":
        report["protocol"].pop("prepared_data_measured_units")
    else:
        report["protocol"]["unexpected"] = 100
    _resign_comparison(report)

    with pytest.raises(ValueError, match="protocol"):
        validate_comparison_report(report, baseline, None)


def test_comparison_validator_rejects_float_prepared_data_protocol_alias():
    _workload, baseline, report = _valid_prepared_comparison()
    report["protocol"]["prepared_data_measured_units"] = 100.0
    body = {key: value for key, value in report.items() if key != "identity"}
    report["identity"] = structured_identity("sml-performance-comparison-v1", body)

    with pytest.raises(ValueError, match="prepared_data_measured_units"):
        validate_comparison_report(report, baseline, None)


def test_comparison_rejects_raw_trial_without_valid_post_exit_evidence():
    _workload, baseline, report = _valid_prepared_comparison()
    report["raw_trials"][0]["post_exit_observation"]["child_measurement_identity"] = (
        "sha256:" + "0" * 64
    )
    body = {key: value for key, value in report.items() if key != "identity"}
    report["identity"] = structured_identity("sml-performance-comparison-v1", body)

    with pytest.raises(ValueError, match="post-exit observation identity"):
        validate_comparison_report(report, baseline, None)


@pytest.mark.parametrize(
    "mutate",
    (
        lambda raw: raw["post_exit_recovery"].update(identity="sha256:" + "0" * 64),
        lambda raw: raw["post_exit_recovery_samples"][0].update(
            previous_sample_identity="sha256:" + "0" * 64
        ),
    ),
)
def test_comparison_rejects_tampered_recovery_evidence_after_resigning(mutate):
    workload, baseline, prepared = _valid_prepared_comparison()
    trials = _comparison_trials(
        workload,
        prepared["candidate_commit"],
        [100.0] * 5,
        attempt_index=0,
    )
    trials[0] = _with_trial_payload(
        _valid_recovered_raw_trial(workload),
        metric="prepared-data",
        pair_index=0,
        attempt_index=0,
        side="reference",
        process_order=0,
        comparison_target="baseline",
        value=100.0,
    )
    report = build_comparison_report(
        baseline=baseline,
        trials=trials,
        candidate_commit=prepared["candidate_commit"],
        minimum_ratio=0.97,
        pretraining_minimum_ratio=None,
        maximum_dispersion=0.02,
        require_lower_bound=False,
        bootstrap_resamples=10_000,
        predecessor_metrics={},
    )
    mutate(report["raw_trials"][0])
    body = {key: value for key, value in report.items() if key != "identity"}
    report["identity"] = structured_identity("sml-performance-comparison-v1", body)

    with pytest.raises(ValueError, match="evidence|recovery|identity"):
        validate_comparison_report(report, baseline, None)


def test_comparison_construction_revalidates_evidence_before_using_values():
    _workload, baseline, report = _valid_prepared_comparison()
    trials = tuple(RawTrial.from_dict(raw) for raw in report["raw_trials"])
    tampered = replace(trials[0], value=float("nan"))

    with pytest.raises(ValueError, match="embedded evidence"):
        build_comparison_report(
            baseline=baseline,
            trials=(tampered, *trials[1:]),
            candidate_commit=report["candidate_commit"],
            minimum_ratio=0.97,
            pretraining_minimum_ratio=None,
            maximum_dispersion=0.02,
            require_lower_bound=False,
            bootstrap_resamples=10_000,
            predecessor_metrics={},
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("pairs", 4),
        ("warmup_units", 20),
        ("measured_units", 100),
        ("bootstrap_resamples", 9_999),
        ("minimum_ratio", 0.90),
        ("maximum_dispersion", 0.03),
        ("require_lower_bound", True),
    ],
)
def test_comparison_validator_rejects_weakened_screen_protocol(field, value):
    _workload, baseline, report = _valid_prepared_comparison()
    invalid = json.loads(json.dumps(report))
    invalid["protocol"][field] = value
    body = {key: item for key, item in invalid.items() if key != "identity"}
    invalid["identity"] = structured_identity("sml-performance-comparison-v1", body)

    with pytest.raises(ValueError, match="protocol|phase-screen"):
        validate_comparison_report(invalid, baseline, predecessor_reports=None)


def test_comparison_validator_rejects_unreferenced_raw_trials():
    _workload, baseline, report = _valid_prepared_comparison()
    invalid = json.loads(json.dumps(report))
    extra = _with_trial_payload(
        RawTrial.from_dict(invalid["raw_trials"][0]),
        comparison_target="unreferenced-extra",
    )
    invalid["raw_trials"].append(extra.to_dict())
    body = {key: item for key, item in invalid.items() if key != "identity"}
    invalid["identity"] = structured_identity("sml-performance-comparison-v1", body)

    with pytest.raises(ValueError, match="unreferenced or missing raw trials"):
        validate_comparison_report(invalid, baseline, predecessor_reports=None)


def test_comparison_validator_rejects_false_round_trip():
    _workload, baseline, report = _valid_prepared_comparison()
    invalid_projection = json.loads(json.dumps(report))
    trial = RawTrial.from_dict(invalid_projection["raw_trials"][1])
    projection = json.loads(json.dumps(trial.canonical_projection))
    projection["model"]["hidden_size"] = 1
    invalid_projection["raw_trials"][1] = _with_trial_payload(
        trial, canonical_projection=projection
    ).to_dict()
    body = {key: item for key, item in invalid_projection.items() if key != "identity"}
    invalid_projection["identity"] = structured_identity(
        "sml-performance-comparison-v1", body
    )
    with pytest.raises(ValueError, match="canonical round trip"):
        validate_comparison_report(
            invalid_projection, baseline, predecessor_reports=None
        )


def test_noisy_comparison_retains_exactly_one_cooled_retry():
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
    trials = _comparison_trials(
        workload, candidate_commit, [95.0, 100.0, 103.0, 106.0, 110.0], 0
    )
    trials.extend(_comparison_trials(workload, candidate_commit, [100.0] * 5, 1))
    cooldown = _nominal_cooldown_evidence()
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
        cooldown_evidence=cooldown,
    )

    validate_comparison_report(report, baseline, predecessor_reports=None)
    assert comparison_has_noise(report, attempt_index=0)
    assert len(report["metrics"]["prepared-data"]["attempts"]) == 2
    assert report["metrics"]["prepared-data"]["baseline_comparison"]["decision"] == (
        "pass"
    )


@pytest.mark.parametrize("metric", METRIC_NAMES)
def test_persistent_noise_blocks_acceptance_after_the_single_retry(metric):
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
    noisy = [95.0, 100.0, 103.0, 106.0, 110.0]
    trials = _comparison_trials(workload, candidate_commit, noisy, 0, metric=metric)
    trials.extend(
        _comparison_trials(workload, candidate_commit, noisy, 1, metric=metric)
    )
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
        cooldown_evidence=_nominal_cooldown_evidence(),
    )

    validate_comparison_report(report, baseline, predecessor_reports=None)
    with pytest.raises(ValueError, match="baseline gate failed"):
        validate_throughput_gates(report, label="phase")


@pytest.mark.parametrize(
    "metric", ("checkpoint-pause", "compile-cold-start", "peak-metal-memory")
)
@pytest.mark.parametrize("decision", ("fail", "inconclusive"))
def test_secondary_metric_ratios_remain_report_only(metric, decision):
    report = {
        "metrics": {
            metric: {
                "baseline_comparison": {"decision": decision},
                "previous_comparison": {"analysis": {"decision": decision}},
            }
        }
    }

    validate_throughput_gates(report, label="final acceptance")
