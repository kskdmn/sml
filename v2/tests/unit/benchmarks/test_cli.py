import json
from types import SimpleNamespace

import pytest

import v2.benchmarks.runner as benchmark_runner
from v2.benchmarks.runner import (
    _resolve_comparison_mode,
    build_parser,
    parse_metrics,
)

from .support import (
    _baseline_fixture,
)


def test_metric_parser_rejects_unknown_or_duplicate_names():
    assert parse_metrics("prepared-data,inference-decode") == (
        "prepared-data",
        "inference-decode",
    )
    with pytest.raises(ValueError, match="unsupported"):
        parse_metrics("prepared-data,unknown")
    with pytest.raises(ValueError, match="duplicate"):
        parse_metrics("prepared-data,prepared-data")


@pytest.mark.parametrize(
    "argv",
    [
        [
            "record-baseline",
            "--source-commit",
            "c" * 40,
            "--manifest",
            "manifest.json",
            "--raw-output",
            "raw.jsonl",
            "--state-directory",
            "state",
            "--prepared-data-measure",
            "100",
        ],
        [
            "compare",
            "--baseline",
            "manifest.json",
            "--candidate",
            "HEAD",
            "--metrics",
            "prepared-data",
            "--predecessors",
            '{"prepared-data":null}',
            "--output",
            "report.json",
        ],
        [
            "validate",
            "--manifest",
            "manifest.json",
            "--raw-input",
            "raw.jsonl",
        ],
    ],
)
def test_runner_parser_accepts_supported_operations(argv):
    assert build_parser().parse_args(argv).operation == argv[0]


@pytest.mark.parametrize(
    "required_option", ["--state-directory", "--prepared-data-measure"]
)
def test_record_baseline_parser_requires_each_explicit_control(required_option, capsys):
    arguments = [
        "record-baseline",
        "--source-commit",
        "c" * 40,
        "--manifest",
        "manifest.json",
        "--raw-output",
        "raw.jsonl",
        "--state-directory",
        "state",
        "--prepared-data-measure",
        "100",
    ]
    index = arguments.index(required_option)
    del arguments[index : index + 2]
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args(arguments)
    assert error.value.code == 2
    assert required_option in capsys.readouterr().err


def test_benchmark_parser_defaults_to_the_shorter_protocol():
    baseline = build_parser().parse_args(
        [
            "record-baseline",
            "--source-commit",
            "c" * 40,
            "--manifest",
            "manifest.json",
            "--raw-output",
            "raw.jsonl",
            "--state-directory",
            "state",
            "--prepared-data-measure",
            "100",
        ]
    )
    comparison = build_parser().parse_args(
        [
            "compare",
            "--baseline",
            "manifest.json",
            "--candidate",
            "HEAD",
            "--metrics",
            "prepared-data",
            "--predecessors",
            '{"prepared-data":null}',
            "--output",
            "report.json",
        ]
    )

    assert (baseline.pairs, baseline.warmup, baseline.measure) == (5, 5, 20)
    assert baseline.prepared_data_measure == 100
    assert (comparison.pairs, comparison.warmup, comparison.measure) == (5, 5, 20)
    assert comparison.prepared_data_measure is None


def test_final_mode_is_inferred_only_for_the_strict_ten_pair_protocol():
    metrics = (
        "prepared-data,pretraining-end-to-end,swag-end-to-end,"
        "inference-prefill,inference-decode,checkpoint-pause,"
        "compile-cold-start,peak-metal-memory"
    )
    predecessors = json.dumps({metric: None for metric in metrics.split(",")})
    args = build_parser().parse_args(
        [
            "compare",
            "--baseline",
            "baseline.json",
            "--candidate",
            "HEAD",
            "--metrics",
            metrics,
            "--pairs",
            "10",
            "--pretraining-minimum-ratio",
            "1.03",
            "--maximum-dispersion",
            "0.015",
            "--predecessors",
            predecessors,
            "--output",
            "final.json",
        ]
    )

    assert _resolve_comparison_mode(args) == "final"
    args.pairs = 5
    with pytest.raises(ValueError, match="final-acceptance"):
        args.mode = "final"
        _resolve_comparison_mode(args)


@pytest.mark.parametrize(
    ("explicit", "expected"),
    [(None, 100), (100, 100)],
)
def test_compare_resolves_prepared_data_count_from_baseline(explicit, expected):
    baseline, _ = _baseline_fixture()

    assert (
        benchmark_runner._resolve_prepared_data_measure(
            SimpleNamespace(prepared_data_measure=explicit), baseline
        )
        == expected
    )


@pytest.mark.parametrize(
    "explicit",
    [20, 99, True],
)
def test_compare_rejects_prepared_data_count_that_disagrees_with_baseline(explicit):
    baseline, _ = _baseline_fixture()

    with pytest.raises(ValueError, match="prepared-data measured units"):
        benchmark_runner._resolve_prepared_data_measure(
            SimpleNamespace(prepared_data_measure=explicit), baseline
        )


def test_parser_exposes_final_acceptance_validation():
    arguments = build_parser().parse_args(
        [
            "validate-final",
            "--baseline",
            "baseline.json",
            "--raw-input",
            "final.jsonl",
            "--report",
            "final.json",
        ]
    )

    assert arguments.operation == "validate-final"
