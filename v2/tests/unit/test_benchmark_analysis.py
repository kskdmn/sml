import hashlib
import inspect
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import v2.benchmarks.runner as benchmark_runner
import v2.benchmarks.journal as baseline_journal
from v2.benchmarks.adapters.replacement import (
    METRIC_OWNER_IMPORTS,
    ReplacementNativeWorkload,
    UnavailableNativeWorkload,
    resolve_native_workload,
    run_measured as run_replacement_measured,
    run_warmup as run_replacement_warmup,
)
from v2.benchmarks.adapters import legacy
from v2.benchmarks.analysis import analyze_pairs
from v2.benchmarks.journal import (
    BaselineSlot,
    BaselineJournal,
    JournalAttempt,
    atomic_write_json,
    atomic_write_text,
    build_session_document,
    read_json_object,
    require_external_state_directory,
)
from v2.benchmarks.recovery import (
    ThermalRecoveryResult,
    ThermalRecoveryTimeout,
    wait_for_nominal_thermal_window,
)
from v2.benchmarks.runner import (
    _resolve_predecessor_mapping,
    _resolve_comparison_mode,
    build_baseline_manifest,
    build_comparison_report,
    build_parser,
    capture_baseline_trials,
    comparison_has_noise,
    detect_competing_gpu_workload,
    decode_thermal_state,
    decode_memory_pressure_level,
    measure_native_process,
    merge_environment_status,
    parse_metrics,
    parse_power_status,
    perform_cooldown,
    process_order,
    publish_baseline_from_journal,
    validate_baseline_manifest,
    validate_checkout_status,
    validate_comparison_report,
    validate_cooldown_evidence,
    validate_final_report,
    validate_thermal_observation,
    validate_throughput_gates,
)
from v2.benchmarks.schema import METRIC_NAMES, CanonicalWorkload, RawTrial
from v2.benchmarks.workload import (
    BENCHMARK_CORPUS,
    HARNESS_COMPONENTS,
    REPLACEMENT_PRECISION_POLICY,
    build_canonical_workload,
    canonical_execution_order_identity,
    canonical_execution_order,
    canonical_input_identity,
    canonical_metric_projection,
    canonical_workload_identity,
    fixed_canonical_rows,
    fixed_inference_requests,
    fixed_swag_examples,
    harness_content_identity,
    semantic_row_content_identity,
    structured_identity,
    write_paired_pretraining_representations,
)


class _RecoveryClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def _recovery_collect(states):
    remaining = iter(states)
    last = [states[-1]]

    def collect():
        try:
            last[0] = next(remaining)
        except StopIteration:
            pass
        status = {
            "power_connected": True,
            "power_mode": "automatic",
            "low_power_mode": False,
            "thermal_state": last[0],
            "thermal_state_raw_value": {
                "nominal": 0,
                "fair": 1,
                "serious": 2,
                "critical": 3,
            }[last[0]],
            "memory_pressure": "normal",
            "memory_free_percentage": 60,
            "competing_gpu_workload": False,
        }
        return {"chip": "Apple M5"}, status, {"python": "3.12.13"}

    return collect


def test_throughput_ratio_and_bound_are_direction_normalized():
    report = analyze_pairs(
        reference=[100.0, 100.0, 100.0, 100.0, 100.0],
        candidate=[103.0, 103.0, 103.0, 103.0, 103.0],
        direction="higher-is-better",
        bootstrap_seed=1729,
        resamples=10_000,
        minimum_ratio=1.03,
        maximum_dispersion=0.02,
        require_lower_bound=True,
    )
    assert report.paired_ratios == (1.03, 1.03, 1.03, 1.03, 1.03)
    assert report.median_ratio == 1.03
    assert report.lower_confidence_bound == 1.03
    assert report.decision == "pass"


def test_noise_and_confidence_fail_closed():
    noisy = analyze_pairs(
        reference=[100.0, 100.0, 100.0, 100.0, 100.0],
        candidate=[95.0, 100.0, 103.0, 106.0, 110.0],
        direction="higher-is-better",
        bootstrap_seed=1729,
        resamples=10_000,
        minimum_ratio=0.97,
        maximum_dispersion=0.02,
        require_lower_bound=True,
    )
    assert noisy.decision == "too-noisy"


def test_bootstrap_is_reproducible_and_point_only_pass_is_inconclusive():
    arguments = dict(
        reference=[100.0, 100.0, 100.0, 100.0, 100.0],
        candidate=[96.0, 98.0, 100.0, 103.0, 108.0],
        direction="higher-is-better",
        bootstrap_seed=1729,
        resamples=10_000,
        minimum_ratio=0.99,
        maximum_dispersion=0.10,
        require_lower_bound=True,
    )
    first = analyze_pairs(**arguments)
    second = analyze_pairs(**arguments)
    assert first.lower_confidence_bound == second.lower_confidence_bound
    assert first.median_ratio >= 0.99
    assert first.lower_confidence_bound < 0.99
    assert first.decision == "inconclusive"


def test_latency_ratios_reverse_direction():
    report = analyze_pairs(
        reference=[10.0] * 5,
        candidate=[8.0] * 5,
        direction="lower-is-better",
        bootstrap_seed=1729,
        resamples=10_000,
        minimum_ratio=1.20,
        maximum_dispersion=0.02,
        require_lower_bound=True,
    )
    assert report.paired_ratios == (1.25,) * 5
    assert report.decision == "pass"


@pytest.mark.parametrize(
    ("reference", "candidate"),
    [([], []), ([1.0], []), ([0.0], [1.0]), ([float("nan")], [1.0])],
)
def test_analysis_rejects_invalid_pairs(reference, candidate):
    with pytest.raises(ValueError):
        analyze_pairs(
            reference,
            candidate,
            direction="higher-is-better",
            bootstrap_seed=1729,
            resamples=10_000,
            minimum_ratio=0.97,
            maximum_dispersion=0.02,
            require_lower_bound=False,
        )


def test_canonical_workload_round_trip_pins_complete_benchmark_contract():
    workload = build_canonical_workload()

    assert CanonicalWorkload.from_dict(workload.to_dict()) == workload
    assert workload.model["vocab_size"] == 28_672
    assert workload.model["rope_scaling_factor"] == 1.0
    assert workload.optimizer["gradient_accumulation_steps"] == 8
    assert workload.precision["compute_dtype"] == "bfloat16"
    assert workload.loader["sequence_length"] == 1_024
    assert workload.loader["swag"]["sequence_length"] == 256
    assert workload.compilation["compilation_passes"] == 1
    assert workload.generation["decode_chunk_size"] == 8
    assert tuple(unit.metric for unit in workload.work_units) == METRIC_NAMES
    units = {unit.metric: unit.measured_units for unit in workload.work_units}
    assert units["inference-prefill"] == 32
    assert units["inference-decode"] == 32
    assert units["compile-cold-start"] == 1


def test_canonical_rows_are_derived_from_pinned_tokenizer_and_corpus():
    workload = build_canonical_workload(row_count=2)
    rows = fixed_canonical_rows(row_count=2)

    assert BENCHMARK_CORPUS
    assert workload.semantic_identities["benchmark_tokenizer"].startswith("sha256:")
    assert workload.semantic_identities["source_corpus_sample"].startswith("sha256:")
    assert (
        semantic_row_content_identity(rows)
        == workload.semantic_identities["canonical_training_rows"]
    )
    assert rows.dtype == np.int32
    assert np.all((rows >= 0) & (rows < 28_672))


def test_swag_and_inference_identities_hash_the_executed_ordered_arrays():
    workload = build_canonical_workload(row_count=32)
    rows = fixed_canonical_rows(row_count=32)
    swag = fixed_swag_examples(workload, rows)
    requests = fixed_inference_requests(workload, rows)

    assert swag.example_ids == tuple(range(128))
    assert requests.request_ids == tuple(range(32))
    assert np.all(np.any(swag.labels != workload.model["pad_token_id"], axis=-1))
    for example_index in range(128):
        for candidate_index in range(4):
            scored = swag.labels[example_index, candidate_index]
            scored = scored[scored != workload.model["pad_token_id"]]
            assert scored[-1] == workload.model["eos_token_id"]
    assert canonical_input_identity("swag-end-to-end", workload) == swag.identity
    assert canonical_input_identity("inference-prefill", workload) == requests.identity
    assert canonical_input_identity("inference-decode", workload) == requests.identity

    reordered = fixed_inference_requests(
        workload, rows, order=tuple(reversed(range(32)))
    )
    changed = build_canonical_workload(
        generation_overrides={"prompt_tokens": 64}, row_count=32
    )
    changed_swag = build_canonical_workload(
        loader_overrides={"swag": {"sequence_length": 128}}, row_count=32
    )
    assert reordered.identity != requests.identity
    assert canonical_input_identity("inference-prefill", changed) != requests.identity
    assert canonical_input_identity("swag-end-to-end", changed_swag) != swag.identity


def test_metric_projection_changes_when_semantic_configuration_changes():
    workload = build_canonical_workload(row_count=32)
    changed = build_canonical_workload(
        loader_overrides={"sequence_length": 512}, row_count=32
    )

    assert canonical_metric_projection("pretraining-compute", workload) != (
        canonical_metric_projection("pretraining-compute", changed)
    )


def test_semantic_row_identity_uses_ordered_little_endian_int32_values():
    rows = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint16)

    first = semantic_row_content_identity(rows)
    second = semantic_row_content_identity(rows.astype(np.int32))
    reordered = semantic_row_content_identity(rows[::-1])

    assert first == second
    assert first.startswith("sha256:")
    assert reordered != first


def test_paired_native_pretraining_representations_preserve_canonical_rows(tmp_path):
    rows = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)

    first_path = tmp_path / "first"
    second_path = tmp_path / "second"
    representations = write_paired_pretraining_representations(rows, first_path)
    repeated = write_paired_pretraining_representations(rows, second_path)

    with np.load(first_path / "legacy-pretraining.npz") as archive:
        legacy_rows = archive["tokens"]
    replacement_rows = np.load(first_path / "replacement-pretraining.npy")
    assert legacy_rows.dtype == np.uint16
    assert replacement_rows.dtype == np.int32
    assert (
        semantic_row_content_identity(legacy_rows)
        == representations["canonical_row_identity"]
    )
    assert (
        semantic_row_content_identity(replacement_rows)
        == representations["canonical_row_identity"]
    )
    assert (
        representations["legacy_file_identity"]
        != representations["replacement_file_identity"]
    )
    assert repeated["legacy_file_identity"] == representations["legacy_file_identity"]
    assert (
        repeated["replacement_file_identity"]
        == representations["replacement_file_identity"]
    )


def test_harness_identity_hashes_every_component_in_fixed_order(tmp_path: Path):
    expected = hashlib.sha256()
    for index, relative_path in enumerate(HARNESS_COMPONENTS):
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = f"component-{index}\n".encode()
        path.write_bytes(payload)
        expected.update(payload)

    assert harness_content_identity(tmp_path) == f"sha256:{expected.hexdigest()}"


def test_runner_alternates_fresh_process_order_by_pair():
    assert process_order(0) == ("reference", "candidate")
    assert process_order(1) == ("candidate", "reference")
    assert process_order(2) == ("reference", "candidate")


def test_replacement_adapter_has_frozen_lazy_owner_map_before_modules_exist():
    assert set(METRIC_OWNER_IMPORTS) == set(METRIC_NAMES)
    assert METRIC_OWNER_IMPORTS == {
        "prepared-data": "sml.data.pretraining",
        "pretraining-compute": "sml.model.language_model",
        "pretraining-end-to-end": "sml.training.pretrain",
        "swag-end-to-end": "sml.training.swag",
        "inference-prefill": "sml.inference",
        "inference-decode": "sml.inference",
        "checkpoint-pause": "sml.artifacts.checkpoint",
        "compile-cold-start": "sml.model.language_model",
        "peak-metal-memory": "sml.training.pretrain",
    }

    native = resolve_native_workload(
        "prepared-data", build_canonical_workload(), Path.cwd()
    )
    assert isinstance(native, UnavailableNativeWorkload)
    assert native.owner_import == "sml.data.pretraining"


def test_replacement_adapter_abi_and_future_owner_transition(tmp_path, monkeypatch):
    from v2.benchmarks.adapters import replacement

    assert tuple(inspect.signature(replacement.resolve_native_workload).parameters) == (
        "metric",
        "canonical_workload",
        "source_root",
    )
    assert tuple(inspect.signature(replacement.run_warmup).parameters) == (
        "metric",
        "native_workload",
        "units",
    )
    assert tuple(inspect.signature(replacement.run_measured).parameters) == (
        "metric",
        "native_workload",
        "units",
    )

    workload = build_canonical_workload(row_count=32)
    assert isinstance(
        resolve_native_workload("prepared-data", workload, tmp_path),
        UnavailableNativeWorkload,
    )
    package = tmp_path / "v2" / "src" / "sml" / "data"
    package.mkdir(parents=True)
    (package.parent / "__init__.py").write_text("", encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "pretraining.py").write_text(
        "from v2.benchmarks.workload import (\n"
        "    REPLACEMENT_PRECISION_POLICY, canonical_input_identity,\n"
        "    canonical_metric_projection, structured_identity,\n"
        ")\n"
        "class Runtime:\n"
        "    verification_level = 'full'\n"
        "    def __init__(self, metric, workload):\n"
        "        self.native_configuration = {\n"
        "            'parameter_precision_policy': REPLACEMENT_PRECISION_POLICY\n"
        "        }\n"
        "        self.native_representation_identity = structured_identity(\n"
        "            'test-native', {'metric': metric}\n"
        "        )\n"
        "        self.canonical_row_identity = workload.semantic_identities[\n"
        "            'canonical_training_rows'\n"
        "        ]\n"
        "        self.canonical_input_identity = canonical_input_identity(metric, workload)\n"
        "        self.canonical_projection = canonical_metric_projection(metric, workload)\n"
        "        from v2.benchmarks.workload import canonical_execution_order_identity\n"
        "        self.execution_order_identity = canonical_execution_order_identity(metric, workload)\n"
        "        self.initial_parameter_identity = workload.semantic_identities[\n"
        "            'initial_bf16_parameters'\n"
        "        ]\n"
        "        self.calls = []\n"
        "    def run(self, units):\n"
        "        self.calls.append(units)\n"
        "        return float(units)\n"
        "def build_benchmark_workload(metric, canonical_workload):\n"
        "    return Runtime(metric, canonical_workload)\n"
        "",
        encoding="utf-8",
    )
    for module_name in ("sml.data.pretraining", "sml.data", "sml"):
        monkeypatch.delitem(sys.modules, module_name, raising=False)

    native = resolve_native_workload("prepared-data", workload, tmp_path)

    assert isinstance(native, ReplacementNativeWorkload)
    assert native.canonical_projection == canonical_metric_projection(
        "prepared-data", workload
    )
    run_replacement_warmup("prepared-data", native, 2)
    assert run_replacement_measured("prepared-data", native, 3) == 3.0
    assert native.runtime.calls == [2, 3]
    owner = sys.modules["sml.data.pretraining"]
    original_factory = owner.build_benchmark_workload

    def wrong_sequence_length(metric, canonical_workload):
        runtime = original_factory(metric, canonical_workload)
        runtime.canonical_projection = json.loads(
            json.dumps(runtime.canonical_projection)
        )
        runtime.canonical_projection["loader"]["sequence_length"] = 1
        return runtime

    monkeypatch.setattr(owner, "build_benchmark_workload", wrong_sequence_length)
    with pytest.raises(RuntimeError, match="canonical workload round trip"):
        resolve_native_workload("prepared-data", workload, tmp_path)
    for module_name in ("sml.data.pretraining", "sml.data", "sml"):
        sys.modules.pop(module_name, None)
    sys.path.remove(str(tmp_path / "v2" / "src"))


def test_measurement_protocol_compiles_warms_and_times_with_explicit_syncs():
    events = []
    adapter = SimpleNamespace(
        run_warmup=lambda metric, native, units: events.append(
            ("warmup", metric, native, units)
        ),
        run_measured=lambda metric, native, units: (
            events.append(("measured", metric, native, units)) or 8_192.0
        ),
    )
    clock_values = iter((1.0, 3.0, 10.0, 12.0))

    measurement = measure_native_process(
        adapter=adapter,
        metric="pretraining-end-to-end",
        native_workload="native",
        warmup_units=2,
        measured_units=1,
        synchronize=lambda: events.append(("synchronize",)),
        clock=lambda: next(clock_values),
        peak_memory=lambda: 123,
        reset_peak_memory=lambda: events.append(("reset-peak-memory",)),
    )

    assert events == [
        ("synchronize",),
        ("warmup", "pretraining-end-to-end", "native", 1),
        ("synchronize",),
        ("warmup", "pretraining-end-to-end", "native", 1),
        ("synchronize",),
        ("warmup", "pretraining-end-to-end", "native", 1),
        ("synchronize",),
        ("reset-peak-memory",),
        ("synchronize",),
        ("measured", "pretraining-end-to-end", "native", 1),
        ("synchronize",),
    ]
    assert measurement.elapsed_seconds == 2.0
    assert measurement.compilation_seconds == 2.0
    assert measurement.value == 4_096.0
    assert measurement.peak_memory_bytes == 123


def test_latency_and_peak_memory_metrics_use_their_pinned_values():
    adapter = SimpleNamespace(
        run_warmup=lambda *_args: None,
        run_measured=lambda *_args: 3.0,
    )

    latency = measure_native_process(
        adapter=adapter,
        metric="checkpoint-pause",
        native_workload=object(),
        warmup_units=0,
        measured_units=3,
        synchronize=lambda: None,
        clock=iter((0.0, 1.0, 2.0, 8.0)).__next__,
        peak_memory=lambda: 700,
        reset_peak_memory=lambda: None,
    )
    memory = measure_native_process(
        adapter=adapter,
        metric="peak-metal-memory",
        native_workload=object(),
        warmup_units=0,
        measured_units=3,
        synchronize=lambda: None,
        clock=iter((0.0, 1.0, 2.0, 8.0)).__next__,
        peak_memory=lambda: 700,
        reset_peak_memory=lambda: None,
    )

    assert latency.value == 2.0
    assert memory.value == 700.0


def test_environment_status_fails_closed_when_run_degrades():
    start = {
        "power_connected": True,
        "power_mode": "automatic",
        "low_power_mode": False,
        "thermal_state": "nominal",
        "thermal_state_raw_value": 0,
        "memory_pressure": "normal",
        "memory_free_percentage": 50,
        "competing_gpu_workload": False,
    }
    end = {
        **start,
        "power_connected": False,
        "thermal_state": "serious",
        "thermal_state_raw_value": 2,
        "memory_pressure": "warning",
        "memory_free_percentage": 7,
    }

    merged = merge_environment_status(start, end)

    assert merged["power_connected"] is False
    assert merged["thermal_state"] == "serious"
    assert merged["memory_pressure"] == "warning"
    assert merged["memory_free_percentage"] == 7
    assert merged["start"] == start
    assert merged["end"] == end


@pytest.mark.parametrize(
    ("raw_value", "state"),
    [(0, "nominal"), (1, "fair"), (2, "serious"), (3, "critical")],
)
def test_thermal_state_retains_foundation_raw_value(raw_value, state):
    assert decode_thermal_state(raw_value) == state
    validate_thermal_observation(
        {"thermal_state": state, "thermal_state_raw_value": raw_value}
    )


def test_thermal_merge_retains_the_worse_matching_raw_value():
    start = {
        "power_connected": True,
        "power_mode": "automatic",
        "low_power_mode": False,
        "thermal_state": "nominal",
        "thermal_state_raw_value": 0,
        "memory_pressure": "normal",
        "memory_free_percentage": 60,
        "competing_gpu_workload": False,
    }
    end = {**start, "thermal_state": "serious", "thermal_state_raw_value": 2}

    merged = merge_environment_status(start, end)

    assert merged["thermal_state"] == "serious"
    assert merged["thermal_state_raw_value"] == 2
    validate_thermal_observation(merged)


def test_thermal_observation_rejects_a_mismatched_string_and_raw_value():
    with pytest.raises(ValueError, match="thermal state and raw value disagree"):
        validate_thermal_observation(
            {"thermal_state": "nominal", "thermal_state_raw_value": 1}
        )


def test_thermal_observation_rejects_a_merged_value_that_is_not_the_worst_endpoint():
    nominal = {"thermal_state": "nominal", "thermal_state_raw_value": 0}
    fair = {"thermal_state": "fair", "thermal_state_raw_value": 1}
    with pytest.raises(ValueError, match="merged thermal state"):
        validate_thermal_observation(
            {
                "thermal_state": "nominal",
                "thermal_state_raw_value": 0,
                "start": nominal,
                "end": fair,
            }
        )


def test_gpu_workload_detection_ignores_system_metal_and_unrelated_python():
    process_table = """
       10 1 /System/Library/Frameworks/Metal.framework/MTLCompilerService
       20 1 .venv/bin/python scripts/nego_golden_eval.py --suite nego
       30 1 python -m mlx_lm.server --model local
       40 1 uv run python v2/src/train_sml.py
    """

    assert detect_competing_gpu_workload(process_table, current_pid=30, parent_pid=1)
    without_gpu_jobs = "\n".join(process_table.splitlines()[:3])
    assert not detect_competing_gpu_workload(
        without_gpu_jobs, current_pid=999, parent_pid=998
    )


def test_native_memory_pressure_levels_fail_closed():
    assert decode_memory_pressure_level(1) == "normal"
    assert decode_memory_pressure_level(2) == "warning"
    assert decode_memory_pressure_level(4) == "critical"
    with pytest.raises(RuntimeError, match="unsupported"):
        decode_memory_pressure_level(0)


def test_power_status_selects_the_active_labeled_pmset_section():
    custom = """Battery Power:
 lowpowermode 1
 sleep 1
AC Power:
 lowpowermode 0
 sleep 0
"""

    assert parse_power_status("Now drawing from 'AC Power'\n", custom) == (
        True,
        "automatic",
        False,
    )
    assert parse_power_status("Now drawing from 'Battery Power'\n", custom) == (
        False,
        "low-power",
        True,
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
            "3687f8b",
            "--manifest",
            "manifest.json",
            "--raw-output",
            "raw.jsonl",
            "--state-directory",
            "state",
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
def test_runner_parser_accepts_the_planned_operations(argv):
    assert build_parser().parse_args(argv).operation == argv[0]


def test_record_baseline_parser_requires_state_directory():
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "record-baseline",
                "--source-commit",
                "3687f8b",
                "--manifest",
                "manifest.json",
                "--raw-output",
                "raw.jsonl",
            ]
        )


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


def test_legacy_adapter_executes_every_metric_against_real_tiny_mlx_workload(
    tmp_path, monkeypatch
):
    import mlx.core as mx

    workload = build_canonical_workload(
        model_overrides={
            "vocab_size": 32,
            "hidden_size": 16,
            "num_layers": 1,
            "num_q_heads": 4,
            "num_kv_heads": 2,
            "intermediate_size": 32,
            "original_max_position_embeddings": 32,
            "hidden_dropout": 0.0,
        },
        optimizer_overrides={
            "gradient_accumulation_steps": 2,
            "swag": {"gradient_accumulation_steps": 2},
        },
        loader_overrides={
            "sequence_length": 8,
            "microbatch_size": 1,
            "swag": {"sequence_length": 8, "batch_size": 1},
        },
        generation_overrides={
            "request_count": 2,
            "prompt_tokens": 4,
            "decode_chunk_size": 2,
        },
        row_count=32,
    )
    expected_work = {
        "prepared-data": 2.0,
        "pretraining-compute": 32.0,
        "pretraining-end-to-end": 32.0,
        "swag-end-to-end": 4.0,
        "inference-prefill": 8.0,
        "inference-decode": 4.0,
        "checkpoint-pause": 2.0,
        "compile-cold-start": 32.0,
        "peak-metal-memory": 32.0,
    }
    paired = write_paired_pretraining_representations(
        fixed_canonical_rows(row_count=32, row_width=9, vocab_size=32),
        tmp_path,
    )

    for metric in METRIC_NAMES:
        native = legacy.resolve_native_workload(metric, workload, Path.cwd())
        assert native.native_configuration["rope_scaling_factor"] == 1.0
        if metric == "swag-end-to-end":
            assert native.native_configuration["sequence_length"] == 8
            assert native.native_configuration["gradient_accumulation_steps"] == 2
        assert (
            native.canonical_row_identity
            == workload.semantic_identities["canonical_training_rows"]
        )
        if metric not in {
            "swag-end-to-end",
            "inference-prefill",
            "inference-decode",
        }:
            assert (
                native.native_representation_identity == paired["legacy_file_identity"]
            )
        assert native.canonical_input_identity == canonical_input_identity(
            metric, workload
        )
        assert native.canonical_projection == canonical_metric_projection(
            metric, workload
        )
        assert native.execution_order_identity == canonical_execution_order_identity(
            metric, workload
        )
        if metric == "pretraining-compute":
            events = []
            runtime = native.runtime
            with monkeypatch.context() as patch:
                for name, label in (
                    ("clip_gradients_by_global_norm", "clip"),
                    ("apply_decoupled_weight_decay", "decay"),
                    ("_retie_embeddings_if_needed", "retie"),
                ):
                    original = getattr(runtime.train, name)

                    def recording_helper(
                        *args, _original=original, _label=label, **kwargs
                    ):
                        events.append(_label)
                        return _original(*args, **kwargs)

                    patch.setattr(runtime.train, name, recording_helper)
                original_update = runtime.optimizer.update

                def recording_update(*args, **kwargs):
                    events.append("update")
                    return original_update(*args, **kwargs)

                patch.setattr(runtime.optimizer, "update", recording_update)
                measured_work = legacy.run_measured(metric, native, 2)
            assert (
                events
                == [
                    "clip",
                    "decay",
                    "retie",
                    "update",
                    "retie",
                ]
                * 2
            )
        else:
            measured_work = legacy.run_measured(metric, native, 2)
        assert measured_work == expected_work[metric]
        if metric in {
            "swag-end-to-end",
            "inference-prefill",
            "inference-decode",
        }:
            expected_order = canonical_execution_order(metric, workload)
            if metric == "swag-end-to-end":
                expected_order = expected_order[:4]
            else:
                expected_order = expected_order[:2]
            assert native.runtime.measured_work_ids == list(expected_order)
        if metric == "swag-end-to-end":
            assert native.runtime.fine_config.sequence_length == 8
            assert native.runtime.fine_config.learning_rate == 1e-4
            assert native.runtime.fine_config.lr_total_steps == 8_192
            assert native.runtime.fine_config.lora.rank == 16
            assert native.runtime.fine_config.lora.target_modules == (
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
            )
        if metric == "checkpoint-pause":
            assert [
                path.name for path in native.runtime.output_directory.iterdir()
            ] == ["step-2"]
        mx.synchronize()


def _valid_raw_trial(workload, metric="prepared-data", pair_index=0):
    work_unit = next(unit for unit in workload.work_units if unit.metric == metric)
    representation_suffix = {
        "swag-end-to-end": "e",
        "inference-prefill": "f",
        "inference-decode": "f",
    }.get(metric, "d")
    return RawTrial(
        schema_version=1,
        metric=metric,
        side="reference",
        attempt_index=0,
        pair_index=pair_index,
        process_order=0,
        source_commit="3687f8b3214a44c675ae67af52e4997762f6c634",
        source_clean=True,
        harness_commit="a" * 40,
        harness_clean=True,
        harness_identity="sha256:" + "b" * 64,
        canonical_workload_identity=canonical_workload_identity(workload),
        native_configuration={
            "rope_scaling_factor": 1.0,
            "canonical_projection_identity": structured_identity(
                "sml-benchmark-metric-projection-v1",
                canonical_metric_projection(metric, workload),
            ),
            "parameter_precision_policy": (
                "legacy BF16 persistent parameters and BF16 Adam moments without "
                "authoritative master parameters"
            ),
        },
        native_representation_identity="sha256:" + representation_suffix * 64,
        canonical_row_identity=workload.semantic_identities["canonical_training_rows"],
        canonical_input_identity=canonical_input_identity(metric, workload),
        canonical_projection=canonical_metric_projection(metric, workload),
        execution_order_identity=canonical_execution_order_identity(metric, workload),
        initial_parameter_identity="sha256:" + "c" * 64,
        comparison_target="baseline",
        warmup_units=0 if metric == "compile-cold-start" else 20,
        measured_units=work_unit.measured_units,
        elapsed_seconds=2.0,
        value=50.0,
        startup_verification_seconds=0.1,
        compilation_seconds=None,
        peak_memory_bytes=1_024,
        synchronization_boundaries=workload.synchronization_boundaries,
        software_versions={
            "python": "3.12.13",
            "mlx": "0.32.0",
            "numpy": "2.4.6",
            "sentencepiece": "0.2.1",
        },
        hardware={
            "chip": "Apple M5",
            "cpu_cores": 10,
            "gpu_cores": 10,
            "unified_memory_bytes": 24 * 1024**3,
            "macos_build": "25F84",
        },
        environment_status={
            "power_connected": True,
            "power_mode": "automatic",
            "low_power_mode": False,
            "thermal_state": "nominal",
            "thermal_state_raw_value": 0,
            "memory_pressure": "normal",
            "competing_gpu_workload": False,
            "start": {
                "thermal_state": "nominal",
                "thermal_state_raw_value": 0,
            },
            "end": {
                "thermal_state": "nominal",
                "thermal_state_raw_value": 0,
            },
        },
    )


def _with_thermal_state(trial, state, raw_value):
    status = {
        **trial.environment_status,
        "thermal_state": state,
        "thermal_state_raw_value": raw_value,
    }
    for endpoint in ("start", "end"):
        if endpoint in status:
            status[endpoint] = {
                **status[endpoint],
                "thermal_state": state,
                "thermal_state_raw_value": raw_value,
            }
    return replace(trial, environment_status=status)


def _valid_baseline_trials(workload):
    return tuple(
        _valid_raw_trial(workload, metric=metric, pair_index=pair_index)
        for metric in METRIC_NAMES
        for pair_index in range(5)
    )


def _valid_paired_representations(workload):
    return {
        "canonical_row_identity": workload.semantic_identities[
            "canonical_training_rows"
        ],
        "row_count": workload.loader["row_count"],
        "row_width": workload.loader["sequence_length"] + 1,
        "legacy_format": "npz",
        "legacy_dtype": "uint16",
        "legacy_file_identity": "sha256:" + "d" * 64,
        "legacy_byte_size": 1,
        "replacement_format": "npy",
        "replacement_dtype": "int32",
        "replacement_file_identity": "sha256:" + "f" * 64,
        "replacement_byte_size": 1,
    }


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


def test_baseline_manifest_binds_raw_trials_to_clean_source_and_harness():
    workload = build_canonical_workload()
    workload_identity = canonical_workload_identity(workload)
    trials = _valid_baseline_trials(workload)
    trial = trials[0]
    paired_representations = _valid_paired_representations(workload)
    manifest = build_baseline_manifest(
        trials=trials,
        workload=workload,
        workload_identity=workload_identity,
        source_commit=trial.source_commit,
        harness_commit=trial.harness_commit,
        harness_identity=trial.harness_identity,
        command="record-baseline --source-commit 3687f8b",
        pairs=5,
        warmup_units=20,
        measured_units=100,
        paired_representations=paired_representations,
    )

    validate_baseline_manifest(manifest, trials)
    assert manifest["paired_pretraining_representations"] == paired_representations
    assert manifest["identity"].startswith("sha256:")
    mutated = replace(trial, harness_identity="sha256:" + "e" * 64)
    with pytest.raises(ValueError, match="harness identity"):
        validate_baseline_manifest(manifest, (mutated, *trials[1:]))


def _resign_baseline(manifest):
    body = {key: value for key, value in manifest.items() if key != "identity"}
    manifest["identity"] = structured_identity("sml-performance-baseline-v1", body)


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
        command="record-baseline --source-commit 3687f8b",
        pairs=5,
        warmup_units=20,
        measured_units=100,
        paired_representations=_valid_paired_representations(workload),
    )

    wrong_source = json.loads(json.dumps(manifest))
    wrong_source["source"]["commit"] = "1" * 40
    _resign_baseline(wrong_source)
    with pytest.raises(ValueError, match="pinned 3687f8b"):
        validate_baseline_manifest(wrong_source, trials)

    missing_metric = json.loads(json.dumps(manifest))
    missing_metric["metrics"].pop("inference-decode")
    _resign_baseline(missing_metric)
    without_decode = tuple(
        trial for trial in trials if trial.metric != "inference-decode"
    )
    with pytest.raises(ValueError, match="every benchmark metric"):
        validate_baseline_manifest(missing_metric, without_decode)

    wrong_warmup = replace(trials[0], warmup_units=19)
    with pytest.raises(ValueError, match="warmup or measured"):
        validate_baseline_manifest(manifest, (wrong_warmup, *trials[1:]))

    wrong_units = replace(trials[0], measured_units=99)
    with pytest.raises(ValueError, match="warmup or measured"):
        validate_baseline_manifest(manifest, (wrong_units, *trials[1:]))

    with pytest.raises(ValueError, match="duplicate, missing, or extra"):
        validate_baseline_manifest(manifest, (*trials, trials[0]))


@pytest.mark.parametrize(
    ("environment_change", "software_change", "message"),
    [
        ({"power_connected": False}, {}, "power_connected"),
        ({"power_mode": "changed"}, {}, "power_mode"),
        ({"low_power_mode": True}, {}, "low_power_mode"),
        ({"memory_pressure": "warning"}, {}, "memory_pressure"),
        ({"competing_gpu_workload": True}, {}, "competing_gpu_workload"),
        ({}, {"python": "3.12.12"}, "software-version"),
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
        warmup_units=20,
        measured_units=100,
        paired_representations=_valid_paired_representations(workload),
    )
    changed = replace(
        trials[0],
        environment_status={
            **trials[0].environment_status,
            **environment_change,
        },
        software_versions={**trials[0].software_versions, **software_change},
    )

    with pytest.raises(ValueError, match=message):
        validate_baseline_manifest(manifest, (changed, *trials[1:]))


@pytest.mark.parametrize(
    ("field", "build_invalid"),
    (
        ("pair_index", lambda trial: replace(trial, pair_index=False)),
        ("attempt_index", lambda trial: replace(trial, attempt_index=False)),
        ("warmup_units", lambda trial: replace(trial, warmup_units=False)),
        ("measured_units", lambda trial: replace(trial, measured_units=True)),
        (
            "rope_scaling_factor",
            lambda trial: replace(
                trial,
                native_configuration={
                    **trial.native_configuration,
                    "rope_scaling_factor": 1,
                },
            ),
        ),
        ("value", lambda trial: replace(trial, value=True)),
        (
            "power_connected",
            lambda trial: replace(
                trial,
                environment_status={
                    **trial.environment_status,
                    "power_connected": 1,
                },
            ),
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
        ),
    ),
)
def test_single_baseline_trial_validation_rejects_boolean_numeric_substitutes(
    field, build_invalid
):
    workload = build_canonical_workload()
    valid = _valid_raw_trial(workload, metric="compile-cold-start")
    invalid = build_invalid(valid)

    with pytest.raises(ValueError, match=field):
        benchmark_runner.validate_baseline_trial(
            invalid,
            workload=workload,
            source_commit=valid.source_commit,
            harness_commit=valid.harness_commit,
            harness_identity=valid.harness_identity,
            expected_hardware=valid.hardware,
            expected_software_versions=valid.software_versions,
            allow_non_nominal_thermal=True,
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
        warmup_units=20,
        measured_units=100,
        paired_representations=_valid_paired_representations(workload),
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
                replace(
                    reference_template,
                    process_order=order.index("reference"),
                    value=100.0,
                ),
                replace(
                    reference_template,
                    side="candidate",
                    process_order=order.index("candidate"),
                    source_commit=candidate_commit,
                    native_configuration={
                        **baseline_trial.native_configuration,
                        "parameter_precision_policy": REPLACEMENT_PRECISION_POLICY,
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
    invalid["raw_trials"][1]["process_order"] = 0
    invalid_body = {key: value for key, value in invalid.items() if key != "identity"}
    invalid["identity"] = structured_identity(
        "sml-performance-comparison-v1", invalid_body
    )
    with pytest.raises(ValueError, match="alternating process order"):
        validate_comparison_report(invalid, baseline_manifest, predecessor_reports=None)


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
                replace(
                    reference,
                    attempt_index=attempt_index,
                    process_order=order.index("reference"),
                    value=100.0,
                ),
                replace(
                    reference,
                    side="candidate",
                    attempt_index=attempt_index,
                    process_order=order.index("candidate"),
                    source_commit=candidate_commit,
                    native_configuration={
                        **reference.native_configuration,
                        "parameter_precision_policy": REPLACEMENT_PRECISION_POLICY,
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
        warmup_units=20,
        measured_units=100,
        paired_representations=_valid_paired_representations(workload),
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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("pairs", 4),
        ("warmup_units", 19),
        ("measured_units", 99),
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
    extra = replace(
        RawTrial.from_dict(invalid["raw_trials"][0]),
        comparison_target="unreferenced-extra",
    )
    invalid["raw_trials"].append(extra.to_dict())
    body = {key: item for key, item in invalid.items() if key != "identity"}
    invalid["identity"] = structured_identity("sml-performance-comparison-v1", body)

    with pytest.raises(ValueError, match="unreferenced or missing raw trials"):
        validate_comparison_report(invalid, baseline, predecessor_reports=None)


def test_comparison_validator_rejects_false_round_trip_and_precision_annotation():
    workload, baseline, report = _valid_prepared_comparison()
    invalid_projection = json.loads(json.dumps(report))
    invalid_projection["raw_trials"][1]["canonical_projection"]["model"][
        "hidden_size"
    ] = 1
    body = {key: item for key, item in invalid_projection.items() if key != "identity"}
    invalid_projection["identity"] = structured_identity(
        "sml-performance-comparison-v1", body
    )
    with pytest.raises(ValueError, match="canonical round trip"):
        validate_comparison_report(
            invalid_projection, baseline, predecessor_reports=None
        )

    candidate_commit = "f" * 40
    trials = []
    for pair_index in range(5):
        order = process_order(pair_index)
        reference = _valid_raw_trial(
            workload, metric="pretraining-end-to-end", pair_index=pair_index
        )
        trials.extend(
            (
                replace(reference, process_order=order.index("reference"), value=100.0),
                replace(
                    reference,
                    side="candidate",
                    process_order=order.index("candidate"),
                    source_commit=candidate_commit,
                    native_configuration={
                        **reference.native_configuration,
                        "parameter_precision_policy": REPLACEMENT_PRECISION_POLICY,
                    },
                    value=103.0,
                ),
            )
        )
    pretraining = build_comparison_report(
        baseline=baseline,
        trials=trials,
        candidate_commit=candidate_commit,
        minimum_ratio=0.97,
        pretraining_minimum_ratio=1.03,
        maximum_dispersion=0.02,
        require_lower_bound=False,
        bootstrap_resamples=10_000,
        predecessor_metrics={},
    )
    pretraining["metrics"]["pretraining-end-to-end"]["precision_policy"][
        "trajectory_equivalent"
    ] = True
    body = {key: item for key, item in pretraining.items() if key != "identity"}
    pretraining["identity"] = structured_identity("sml-performance-comparison-v1", body)
    with pytest.raises(ValueError, match="precision annotation"):
        validate_comparison_report(pretraining, baseline, predecessor_reports=None)


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
        warmup_units=20,
        measured_units=100,
        paired_representations=_valid_paired_representations(workload),
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


def test_persistent_noise_blocks_acceptance_after_the_single_retry():
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
        warmup_units=20,
        measured_units=100,
        paired_representations=_valid_paired_representations(workload),
    )
    candidate_commit = "e" * 40
    noisy = [95.0, 100.0, 103.0, 106.0, 110.0]
    trials = _comparison_trials(workload, candidate_commit, noisy, 0)
    trials.extend(_comparison_trials(workload, candidate_commit, noisy, 1))
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


def test_cooldown_uses_fake_clock_and_proves_final_nominal_window():
    now = [0.0]
    status = _nominal_cooldown_evidence()["samples"][0]["environment_status"]

    evidence = perform_cooldown(
        collect=lambda: ({}, status, {}),
        clock=lambda: now[0],
        sleep=lambda seconds: now.__setitem__(0, now[0] + seconds),
    )

    validate_cooldown_evidence(
        evidence, build_canonical_workload().required_environment
    )
    assert evidence["duration_seconds"] == 900.0
    assert len(evidence["samples"]) >= 16


def test_generated_cooldown_evidence_survives_a_realistically_advancing_clock():
    now = [0.0]
    status = _nominal_cooldown_evidence()["samples"][0]["environment_status"]

    def clock():
        observed = now[0]
        now[0] += 0.001
        return observed

    evidence = perform_cooldown(
        collect=lambda: ({}, status, {}),
        clock=clock,
        sleep=lambda seconds: now.__setitem__(0, now[0] + seconds),
    )

    assert evidence["duration_seconds"] == evidence["samples"][-1]["elapsed_seconds"]
    validate_cooldown_evidence(
        evidence, build_canonical_workload().required_environment
    )


def test_cooldown_rejects_a_gap_larger_than_the_declared_interval():
    evidence = _nominal_cooldown_evidence()
    evidence["samples"].pop(1)

    with pytest.raises(ValueError, match="sampling gap"):
        validate_cooldown_evidence(
            evidence, build_canonical_workload().required_environment
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
        warmup_units=20,
        measured_units=100,
        paired_representations=_valid_paired_representations(workload),
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
    with pytest.raises(ValueError, match="raw pairs"):
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


def _session_document(
    tmp_path,
    *,
    harness_commit="a" * 40,
    protocol=None,
    hardware=None,
    software_versions=None,
    paired_representations=None,
    manifest_name="baseline.json",
    raw_output_name="baseline.jsonl",
):
    workload = build_canonical_workload()
    return build_session_document(
        harness_commit=harness_commit,
        harness_identity="sha256:" + "b" * 64,
        source_commit="3687f8b3214a44c675ae67af52e4997762f6c634",
        canonical_workload=workload,
        canonical_workload_identity=canonical_workload_identity(workload),
        protocol=protocol or {"pairs": 5, "warmup_units": 20, "measured_units": 100},
        hardware=hardware or {"chip": "Apple M5"},
        software_versions=software_versions or {"python": "3.12.13", "mlx": "0.32.0"},
        paired_representations=paired_representations
        or {"canonical_row_identity": "sha256:" + "c" * 64},
        manifest_path=tmp_path / manifest_name,
        raw_output_path=tmp_path / raw_output_name,
    )


def test_capture_retries_only_the_thermal_slot_and_resumes_accepted_slots(tmp_path):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slots = (
        BaselineSlot("prepared-data", 0),
        BaselineSlot("prepared-data", 1),
    )
    accepted_first = _valid_raw_trial(workload, pair_index=0)
    first_attempt = journal.next_attempt(slots[0])
    atomic_write_json(first_attempt.path, accepted_first.to_dict(), create_only=True)
    journal.accept_inflight(first_attempt, accepted_first)
    launches = []

    def launch(slot, attempt):
        launches.append(slot)
        base = _valid_raw_trial(workload, pair_index=slot.pair_index)
        if len(launches) == 1:
            return replace(
                base,
                environment_status={
                    **base.environment_status,
                    "thermal_state": "fair",
                    "thermal_state_raw_value": 1,
                },
            )
        return base

    recovered = []
    trials = capture_baseline_trials(
        journal=journal,
        slots=slots,
        launch_trial=launch,
        preflight=lambda: (
            accepted_first.hardware,
            accepted_first.environment_status,
            accepted_first.software_versions,
        ),
        validate_preflight=lambda hardware, status, software: None,
        recover=lambda slot, recovery_index, deadline, trigger: recovered.append(slot),
        validate_trial=lambda trial, allow_non_nominal_thermal: None,
        clock=lambda: 0.0,
        utc_now=lambda: "2026-08-05T00:00:00+00:00",
        progress=lambda message: None,
    )

    assert launches == [slots[1], slots[1]]
    assert recovered == [slots[1]]
    assert [(trial.metric, trial.pair_index) for trial in trials] == [
        ("prepared-data", 0),
        ("prepared-data", 1),
    ]
    rejected = read_json_object(
        journal.rejected_path(slots[1], 0), label="rejected trial"
    )
    assert rejected["trial"]["environment_status"]["thermal_state"] == "fair"


def test_capture_reports_resumed_slots_in_supplied_order(tmp_path):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slots = (
        BaselineSlot("prepared-data", 1),
        BaselineSlot("prepared-data", 0),
    )
    for slot in reversed(slots):
        trial = _valid_raw_trial(workload, pair_index=slot.pair_index)
        attempt = journal.next_attempt(slot)
        atomic_write_json(attempt.path, trial.to_dict(), create_only=True)
        journal.accept_inflight(attempt, trial)
    messages = []

    capture_baseline_trials(
        journal=journal,
        slots=slots,
        launch_trial=lambda slot, attempt: pytest.fail("accepted slot was launched"),
        preflight=lambda: pytest.fail("accepted slot reached preflight"),
        validate_preflight=lambda hardware, status, software: None,
        recover=lambda slot, recovery_index, deadline, trigger: None,
        validate_trial=lambda trial, allow_non_nominal_thermal: None,
        progress=messages.append,
    )

    assert messages == [
        "resumed accepted prepared-data pair 1",
        "resumed accepted prepared-data pair 0",
    ]


def test_capture_passes_the_thermal_policy_to_trial_validation_by_keyword(tmp_path):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    trial = _valid_raw_trial(workload)
    attempt = journal.next_attempt(slot)
    atomic_write_json(attempt.path, trial.to_dict(), create_only=True)
    policies = []

    def validate(current_trial, *, allow_non_nominal_thermal):
        policies.append(allow_non_nominal_thermal)

    capture_baseline_trials(
        journal=journal,
        slots=(slot,),
        launch_trial=lambda current_slot, current_attempt: pytest.fail(
            "complete in-flight trial was relaunched"
        ),
        preflight=lambda: pytest.fail("accepted in-flight slot reached preflight"),
        validate_preflight=lambda hardware, status, software: None,
        recover=lambda current_slot, recovery_index, deadline, trigger: None,
        validate_trial=validate,
        progress=lambda message: None,
    )

    assert policies == [True]


def test_capture_records_preflight_thermal_trigger_before_launch(tmp_path):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    nominal_trial = _valid_raw_trial(workload)
    fair = {
        **nominal_trial.environment_status,
        "thermal_state": "fair",
        "thermal_state_raw_value": 1,
    }
    preflights = iter(
        [
            (nominal_trial.hardware, fair, nominal_trial.software_versions),
            (
                nominal_trial.hardware,
                nominal_trial.environment_status,
                nominal_trial.software_versions,
            ),
        ]
    )
    triggers = []
    launches = []

    capture_baseline_trials(
        journal=journal,
        slots=(slot,),
        launch_trial=lambda current_slot, attempt: (
            launches.append(current_slot) or nominal_trial
        ),
        preflight=lambda: next(preflights),
        validate_preflight=lambda hardware, status, software: None,
        recover=lambda current_slot, recovery_index, deadline, trigger: triggers.append(
            trigger
        ),
        validate_trial=lambda trial, allow_non_nominal_thermal: None,
        clock=lambda: 0.0,
        utc_now=lambda: "2026-08-05T00:00:00+00:00",
        progress=lambda message: None,
    )

    assert launches == [slot]
    assert triggers[0]["source"] == "preflight"
    assert (
        triggers[0]["preflight"]["environment_status"]["thermal_state_raw_value"] == 1
    )
    assert journal.preflight_path(slot, 0).is_file()


def test_capture_classifies_complete_inflight_before_launching(tmp_path):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    trial = _valid_raw_trial(workload)
    atomic_write_json(attempt.path, trial.to_dict(), create_only=True)

    trials = capture_baseline_trials(
        journal=journal,
        slots=(slot,),
        launch_trial=lambda current_slot, current_attempt: pytest.fail(
            "resume launched a replacement for a complete in-flight trial"
        ),
        preflight=lambda: pytest.fail("accepted in-flight slot reached preflight"),
        validate_preflight=lambda hardware, status, software: None,
        recover=lambda current_slot, recovery_index, deadline, trigger: None,
        validate_trial=lambda current_trial, allow_non_nominal_thermal: None,
        clock=lambda: 0.0,
        utc_now=lambda: "2026-08-05T00:00:00+00:00",
        progress=lambda message: None,
    )

    assert trials == (trial,)
    assert journal.load_accepted((slot,)) == {slot: trial}


def test_capture_timeout_preserves_prior_acceptance_and_rejected_attempt(tmp_path):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    first = BaselineSlot("prepared-data", 0)
    second = BaselineSlot("prepared-data", 1)
    accepted = _valid_raw_trial(workload, pair_index=0)
    attempt = journal.next_attempt(first)
    atomic_write_json(attempt.path, accepted.to_dict(), create_only=True)
    journal.accept_inflight(attempt, accepted)
    fair_trial = replace(
        _valid_raw_trial(workload, pair_index=1),
        environment_status={
            **_valid_raw_trial(workload, pair_index=1).environment_status,
            "thermal_state": "fair",
            "thermal_state_raw_value": 1,
        },
    )

    def timeout_recovery(current_slot, recovery_index, deadline, trigger):
        raise ThermalRecoveryTimeout(ThermalRecoveryResult(7_200.0, 241))

    with pytest.raises(ThermalRecoveryTimeout):
        capture_baseline_trials(
            journal=journal,
            slots=(first, second),
            launch_trial=lambda current_slot, current_attempt: fair_trial,
            preflight=lambda: (
                accepted.hardware,
                accepted.environment_status,
                accepted.software_versions,
            ),
            validate_preflight=lambda hardware, status, software: None,
            recover=timeout_recovery,
            validate_trial=lambda current_trial, allow_non_nominal_thermal: None,
            clock=lambda: 0.0,
            utc_now=lambda: "2026-08-05T00:00:00+00:00",
            progress=lambda message: None,
        )

    assert journal.load_accepted((first, second)) == {first: accepted}
    assert journal.rejected_path(second, 0).is_file()


@pytest.mark.parametrize(
    "persisted_state",
    (
        "untriggered-preflight",
        "unrecovered-rejected",
        "unfinished-recovery",
        "timeout-recovery",
    ),
)
def test_capture_replays_persisted_thermal_state_before_preflight(
    tmp_path, persisted_state
):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    nominal = _valid_raw_trial(workload)
    fair = _with_thermal_state(nominal, "fair", 1)
    observation = {
        "observed_at_utc": "2026-08-05T00:00:00+00:00",
        "hardware": fair.hardware,
        "environment_status": fair.environment_status,
        "software_versions": fair.software_versions,
    }

    if persisted_state == "unrecovered-rejected":
        attempt = journal.next_attempt(slot)
        atomic_write_json(attempt.path, fair.to_dict(), create_only=True)
        journal.reject_inflight(attempt, fair, reason="non-nominal-thermal")
        rejected = read_json_object(
            journal.rejected_path(slot, 0), label="rejected trial"
        )
        expected_trigger = {
            "source": "rejected-trial",
            "rejected_trial_identity": rejected["identity"],
        }
        expected_recovery_index = 0
    else:
        preflight_document = journal.record_preflight(slot, 0, observation)
        expected_trigger = {
            "source": "preflight",
            "preflight": preflight_document,
        }
        expected_recovery_index = 0
        if persisted_state in ("unfinished-recovery", "timeout-recovery"):
            journal.record_recovery_trigger(slot, 0, expected_trigger)
            expected_recovery_index = 1
        if persisted_state == "timeout-recovery":
            journal.record_recovery_summary(
                slot,
                0,
                {
                    "outcome": "timeout",
                    "duration_seconds": 7_200.0,
                    "sample_count": 0,
                },
            )

    events = []
    recovered = []
    clock = SimpleNamespace(now=0.0)

    def progress(message):
        clock.now += 100.0

    capture_baseline_trials(
        journal=journal,
        slots=(slot,),
        launch_trial=lambda current_slot, attempt: events.append("launch") or nominal,
        preflight=lambda: (
            events.append("preflight")
            or (nominal.hardware, nominal.environment_status, nominal.software_versions)
        ),
        validate_preflight=lambda hardware, status, software: None,
        recover=lambda current_slot, recovery_index, deadline, trigger: (
            events.append("recover"),
            recovered.append((recovery_index, deadline, trigger)),
        ),
        validate_trial=lambda trial, allow_non_nominal_thermal: None,
        clock=lambda: clock.now,
        utc_now=lambda: "2026-08-05T00:01:00+00:00",
        progress=progress,
    )

    assert events == ["recover", "preflight", "launch"]
    assert recovered == [(expected_recovery_index, 7_200.0, expected_trigger)]


def test_capture_anchors_persisted_deadline_before_later_history_reads():
    workload = build_canonical_workload()
    slot = BaselineSlot("prepared-data", 0)
    fair = _with_thermal_state(_valid_raw_trial(workload), "fair", 1)
    preflight_document = {
        "identity": "sha256:" + "a" * 64,
        "environment_status": fair.environment_status,
    }
    clock = SimpleNamespace(now=0.0)

    class SlowHistoryJournal:
        def load_accepted(self, slots):
            return {}

        def _preflight_records(self):
            return ((slot, 0, Path("preflight.json"), preflight_document),)

        def _attempt_records(self, category):
            clock.now += 100.0
            return ()

        def _thermal_recovery_records(self):
            clock.now += 100.0
            return ()

        def load_inflight(self, slots):
            return ()

        def _validate_preflight_history(self):
            clock.now += 100.0
            return {slot: ((0, preflight_document),)}

        def _validate_thermal_recovery_history(self):
            clock.now += 100.0
            return {}

    recovered_deadlines = []

    def stop_after_recovery(current_slot, recovery_index, deadline, trigger):
        recovered_deadlines.append(deadline)
        raise RuntimeError("stop after recovery")

    with pytest.raises(RuntimeError, match="stop after recovery"):
        capture_baseline_trials(
            journal=SlowHistoryJournal(),
            slots=(slot,),
            launch_trial=lambda current_slot, attempt: pytest.fail(
                "trial launched before persisted recovery"
            ),
            preflight=lambda: pytest.fail("preflight ran before persisted recovery"),
            validate_preflight=lambda hardware, status, software: None,
            recover=stop_after_recovery,
            validate_trial=lambda trial, allow_non_nominal_thermal: None,
            clock=lambda: clock.now,
            progress=lambda message: None,
        )

    assert recovered_deadlines == [7_200.0]


def test_persisted_recovery_reuses_first_recognition_deadline_for_slot():
    workload = build_canonical_workload()
    slot = BaselineSlot("prepared-data", 0)
    fair = _with_thermal_state(_valid_raw_trial(workload), "fair", 1)
    first = {
        "identity": "sha256:" + "a" * 64,
        "environment_status": fair.environment_status,
    }
    second = {
        "identity": "sha256:" + "b" * 64,
        "environment_status": fair.environment_status,
    }

    class MultiplePreflightJournal:
        def _preflight_records(self):
            return (
                (slot, 0, Path("0.json"), first),
                (slot, 1, Path("1.json"), second),
            )

        def _attempt_records(self, category):
            return ()

        def _thermal_recovery_records(self):
            return ()

    clock = SimpleNamespace(now=0.0)

    def advancing_clock():
        observed = clock.now
        clock.now += 100.0
        return observed

    persisted = benchmark_runner._persisted_pending_thermal_triggers(
        MultiplePreflightJournal(), (slot,), advancing_clock
    )

    assert persisted[slot] == (
        {"source": "preflight", "preflight": second},
        7_200.0,
    )


def test_capture_does_not_replay_thermal_state_closed_by_nominal_window(tmp_path):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    nominal = _valid_raw_trial(workload)
    fair = _with_thermal_state(nominal, "fair", 1)
    preflight_document = journal.record_preflight(
        slot,
        0,
        {
            "observed_at_utc": "2026-08-05T00:00:00+00:00",
            "hardware": fair.hardware,
            "environment_status": fair.environment_status,
            "software_versions": fair.software_versions,
        },
    )
    journal.record_recovery_trigger(
        slot, 0, {"source": "preflight", "preflight": preflight_document}
    )
    journal.record_thermal_sample(
        slot,
        0,
        0,
        {
            "schema_version": 1,
            "observed_at_utc": "2026-08-05T00:05:00+00:00",
            "elapsed_seconds": 300.0,
            "hardware": nominal.hardware,
            "environment_status": nominal.environment_status,
            "software_versions": nominal.software_versions,
        },
    )
    journal.record_recovery_summary(
        slot,
        0,
        {
            "outcome": "nominal-window",
            "duration_seconds": 300.0,
            "sample_count": 1,
        },
    )

    capture_baseline_trials(
        journal=journal,
        slots=(slot,),
        launch_trial=lambda current_slot, attempt: nominal,
        preflight=lambda: (
            nominal.hardware,
            nominal.environment_status,
            nominal.software_versions,
        ),
        validate_preflight=lambda hardware, status, software: None,
        recover=lambda current_slot, recovery_index, deadline, trigger: pytest.fail(
            "completed nominal recovery was replayed"
        ),
        validate_trial=lambda trial, allow_non_nominal_thermal: None,
        progress=lambda message: None,
    )


@pytest.mark.parametrize("violation_source", ("preflight", "rejected-trial"))
def test_capture_deadline_starts_before_slow_thermal_transition(
    tmp_path, monkeypatch, violation_source
):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    nominal = _valid_raw_trial(workload)
    fair = _with_thermal_state(nominal, "fair", 1)
    clock = SimpleNamespace(now=0.0)
    deadlines = []

    if violation_source == "preflight":
        preflights = iter(
            (
                (fair.hardware, fair.environment_status, fair.software_versions),
                (
                    nominal.hardware,
                    nominal.environment_status,
                    nominal.software_versions,
                ),
            )
        )
        original = BaselineJournal.record_preflight

        def slow_record_preflight(self, current_slot, preflight_index, observation):
            clock.now += 100.0
            return original(self, current_slot, preflight_index, observation)

        monkeypatch.setattr(BaselineJournal, "record_preflight", slow_record_preflight)

        def launch(current_slot, attempt):
            return nominal

    else:
        preflights = iter(
            (
                (
                    nominal.hardware,
                    nominal.environment_status,
                    nominal.software_versions,
                ),
                (
                    nominal.hardware,
                    nominal.environment_status,
                    nominal.software_versions,
                ),
            )
        )
        launches = []
        original = BaselineJournal.reject_inflight

        def slow_reject(self, attempt, trial, *, reason):
            clock.now += 100.0
            return original(self, attempt, trial, reason=reason)

        monkeypatch.setattr(BaselineJournal, "reject_inflight", slow_reject)

        def launch(current_slot, attempt):
            launches.append(current_slot)
            if len(launches) == 1:
                clock.now = 0.0
                return fair
            return nominal

    def progress(message):
        clock.now += 100.0

    capture_baseline_trials(
        journal=journal,
        slots=(slot,),
        launch_trial=launch,
        preflight=lambda: next(preflights),
        validate_preflight=lambda hardware, status, software: None,
        recover=lambda current_slot, recovery_index, deadline, trigger: (
            deadlines.append(deadline)
        ),
        validate_trial=lambda trial, allow_non_nominal_thermal: None,
        clock=lambda: clock.now,
        utc_now=lambda: "2026-08-05T00:00:00+00:00",
        progress=progress,
    )

    assert deadlines == [7_200.0]


def test_checkout_status_allows_only_bound_untracked_final_outputs():
    allowed = frozenset(
        {
            "v2/benchmarks/manifests/baseline-3687f8b.json",
            "v2/benchmarks/results/baseline-3687f8b.jsonl",
        }
    )
    validate_checkout_status(
        "?? v2/benchmarks/manifests/baseline-3687f8b.json\n"
        "?? v2/benchmarks/results/baseline-3687f8b.jsonl\n",
        allowed_untracked_paths=allowed,
    )
    with pytest.raises(ValueError, match="checkout must be clean"):
        validate_checkout_status(
            "?? v2/benchmarks/manifests/baseline-3687f8b.json\n?? unexpected.txt\n",
            allowed_untracked_paths=allowed,
        )
    with pytest.raises(ValueError, match="checkout must be clean"):
        validate_checkout_status(
            " M v2/benchmarks/runner.py\n",
            allowed_untracked_paths=allowed,
        )


def _accepted_complete_journal(
    tmp_path,
    *,
    manifest_name="baseline.json",
    raw_output_name="baseline.jsonl",
):
    workload = build_canonical_workload()
    paired = _valid_paired_representations(workload)
    session = _session_document(
        tmp_path,
        paired_representations=paired,
        manifest_name=manifest_name,
        raw_output_name=raw_output_name,
    )
    journal = BaselineJournal.open(tmp_path / "state", session)
    trials = _valid_baseline_trials(workload)
    for trial in trials:
        slot = BaselineSlot(trial.metric, trial.pair_index)
        attempt = journal.next_attempt(slot)
        atomic_write_json(attempt.path, trial.to_dict(), create_only=True)
        journal.accept_inflight(attempt, trial)
    return workload, paired, journal, trials


def test_final_publication_uses_exactly_the_45_accepted_trials(tmp_path):
    workload, paired, journal, trials = _accepted_complete_journal(tmp_path)
    manifest_path = tmp_path / "baseline.json"
    raw_path = tmp_path / "baseline.jsonl"

    manifest = publish_baseline_from_journal(
        journal=journal,
        trials=trials,
        workload=workload,
        workload_identity=canonical_workload_identity(workload),
        source_commit=trials[0].source_commit,
        harness_commit=trials[0].harness_commit,
        harness_identity=trials[0].harness_identity,
        command="record-baseline --state-directory state",
        paired_representations=paired,
        manifest_path=manifest_path,
        raw_output_path=raw_path,
    )

    raw_trials = tuple(
        RawTrial.from_dict(json.loads(line))
        for line in raw_path.read_text(encoding="utf-8").splitlines()
    )
    validate_baseline_manifest(manifest, raw_trials)
    assert len(raw_trials) == 45
    assert [(trial.metric, trial.pair_index) for trial in raw_trials] == [
        (metric, pair_index) for metric in METRIC_NAMES for pair_index in range(5)
    ]
    completed = read_json_object(journal.completed_path, label="completion")
    assert completed["baseline_identity"] == manifest["identity"]
    assert completed["raw_trial_identities"] == [
        benchmark_runner._raw_trial_identity(trial) for trial in raw_trials
    ]


def test_final_publication_never_overwrites_different_existing_output(tmp_path):
    workload, paired, journal, trials = _accepted_complete_journal(tmp_path)
    raw_path = tmp_path / "baseline.jsonl"
    raw_path.write_text("different existing content\n", encoding="utf-8")

    with pytest.raises(
        ValueError, match="final artifact already exists with different content"
    ):
        publish_baseline_from_journal(
            journal=journal,
            trials=trials,
            workload=workload,
            workload_identity=canonical_workload_identity(workload),
            source_commit=trials[0].source_commit,
            harness_commit=trials[0].harness_commit,
            harness_identity=trials[0].harness_identity,
            command="record-baseline --state-directory state",
            paired_representations=paired,
            manifest_path=tmp_path / "baseline.json",
            raw_output_path=raw_path,
        )

    assert raw_path.read_text(encoding="utf-8") == "different existing content\n"
    assert not journal.completed_path.exists()


@pytest.mark.parametrize(
    ("manifest_name", "raw_output_name"),
    (
        ("same.json", "same.json"),
        ("baseline.json", "state/completed.json"),
        ("state/final-manifest.json", "baseline.jsonl"),
        ("out", "out/raw.jsonl"),
        ("out/manifest.json", "out"),
        (".", "baseline.jsonl"),
    ),
)
def test_final_publication_rejects_colliding_or_journal_contained_outputs(
    tmp_path, manifest_name, raw_output_name
):
    workload, paired, journal, trials = _accepted_complete_journal(
        tmp_path,
        manifest_name=manifest_name,
        raw_output_name=raw_output_name,
    )
    manifest_path = tmp_path / manifest_name
    raw_path = tmp_path / raw_output_name
    output_existence_before = {
        manifest_path: manifest_path.exists(),
        raw_path: raw_path.exists(),
    }
    journal_files_before = {
        path.relative_to(journal.root): path.read_bytes()
        for path in journal.root.rglob("*")
        if path.is_file()
    }

    with pytest.raises(ValueError, match="final output paths"):
        publish_baseline_from_journal(
            journal=journal,
            trials=trials,
            workload=workload,
            workload_identity=canonical_workload_identity(workload),
            source_commit=trials[0].source_commit,
            harness_commit=trials[0].harness_commit,
            harness_identity=trials[0].harness_identity,
            command="record-baseline --state-directory state",
            paired_representations=paired,
            manifest_path=manifest_path,
            raw_output_path=raw_path,
        )

    assert not journal.completed_path.exists()
    for path in (manifest_path, raw_path):
        if not output_existence_before[path]:
            assert not path.exists()
    assert {
        path.relative_to(journal.root): path.read_bytes()
        for path in journal.root.rglob("*")
        if path.is_file()
    } == journal_files_before
    assert (
        BaselineJournal.open(journal.root, journal.session).session == journal.session
    )


def test_detached_worktree_cleans_registration_after_verification_error(
    tmp_path, monkeypatch
):
    repository = tmp_path / "repository"
    destination = tmp_path / "source"
    commit = "a" * 40
    registered = set()
    commands = []

    def run(command, *, cwd, check):
        commands.append(command)
        if command[:3] == ("git", "worktree", "add"):
            registered.add(Path(command[-2]))
        elif command[:3] == ("git", "worktree", "remove"):
            registered.remove(Path(command[-1]))
        return SimpleNamespace()

    monkeypatch.setattr(benchmark_runner.subprocess, "run", run)
    monkeypatch.setattr(
        benchmark_runner,
        "_require_clean_checkout",
        lambda path, label: (_ for _ in ()).throw(RuntimeError("verification failed")),
    )

    with pytest.raises(RuntimeError, match="verification failed"):
        benchmark_runner._create_detached_worktree(repository, commit, destination)

    assert registered == set()
    assert [command[:3] for command in commands] == [
        ("git", "worktree", "add"),
        ("git", "worktree", "remove"),
    ]


def test_detached_worktree_preserves_verification_error_when_cleanup_fails(
    tmp_path, monkeypatch
):
    repository = tmp_path / "repository"
    repository.mkdir()
    destination = tmp_path / "source"
    commit = "a" * 40
    original_error = RuntimeError("original verification failure")
    commands = []

    def run(command, *, cwd, check):
        commands.append(command)
        if command[:3] == ("git", "worktree", "add"):
            destination.mkdir()
            (destination / "tracked.py").write_text("content\n", encoding="utf-8")
        return SimpleNamespace()

    monkeypatch.setattr(benchmark_runner.subprocess, "run", run)
    monkeypatch.setattr(
        benchmark_runner,
        "_require_clean_checkout",
        lambda path, label: (_ for _ in ()).throw(original_error),
    )
    monkeypatch.setattr(
        benchmark_runner,
        "_remove_worktree",
        lambda repository, destination: (_ for _ in ()).throw(
            RuntimeError("primary cleanup failure")
        ),
    )

    with pytest.raises(RuntimeError, match="original verification failure") as caught:
        benchmark_runner._create_detached_worktree(repository, commit, destination)

    assert caught.value is original_error
    assert not destination.exists()
    assert ("git", "worktree", "prune") in commands
    assert any("primary cleanup failure" in note for note in caught.value.__notes__)


def test_baseline_journal_resumes_only_an_identical_session(tmp_path):
    state = tmp_path / "state"
    expected = _session_document(tmp_path)

    first = BaselineJournal.open(state, expected)
    resumed = BaselineJournal.open(state, expected)

    assert first.session == expected
    assert resumed.session == expected
    changed = _session_document(tmp_path, harness_commit="d" * 40)
    with pytest.raises(ValueError, match="session does not match"):
        BaselineJournal.open(state, changed)


@pytest.mark.parametrize(
    "changed",
    [
        {"protocol": {"pairs": 4, "warmup_units": 20, "measured_units": 100}},
        {"hardware": {"chip": "Apple M4"}},
        {"software_versions": {"python": "3.12.12", "mlx": "0.32.0"}},
        {"manifest_name": "other.json"},
        {"raw_output_name": "other.jsonl"},
    ],
)
def test_baseline_journal_rejects_every_session_compatibility_change(tmp_path, changed):
    state = tmp_path / "state"
    BaselineJournal.open(state, _session_document(tmp_path))
    with pytest.raises(ValueError, match="session does not match"):
        BaselineJournal.open(state, _session_document(tmp_path, **changed))


def test_state_directory_must_be_outside_measured_checkouts(tmp_path):
    harness = tmp_path / "harness"
    harness.mkdir()
    with pytest.raises(ValueError, match="outside measured checkouts"):
        require_external_state_directory(harness / "state", (harness,))

    external = tmp_path / "external"
    assert require_external_state_directory(external, (harness,)) == external.resolve()


def test_journal_never_adopts_a_nonempty_directory_without_a_session(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    (state / "orphan.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty state directory has no session"):
        BaselineJournal.open(state, _session_document(tmp_path))


def test_journal_rejects_an_orphan_written_during_session_initialization(
    tmp_path, monkeypatch
):
    state = tmp_path / "state"
    expected = _session_document(tmp_path)
    original_write = baseline_journal.atomic_write_json

    def write_session_after_orphan(path, value, *, create_only=False):
        (state / "orphan.json").parent.mkdir(parents=True, exist_ok=True)
        (state / "orphan.json").write_text("{}\n", encoding="utf-8")
        return original_write(path, value, create_only=create_only)

    monkeypatch.setattr(
        baseline_journal, "atomic_write_json", write_session_after_orphan
    )

    with pytest.raises(ValueError, match="unexpected content"):
        BaselineJournal.open(state, expected)

    assert (state / "orphan.json").is_file()
    assert (state / "session.json").is_file()
    assert {path.name for path in state.iterdir()} == {
        ".baseline-session-initializing",
        "orphan.json",
        "session.json",
    }
    with pytest.raises(ValueError, match="unexpected content"):
        BaselineJournal.open(state, expected)


def test_journal_rejects_later_resume_after_orphan_follows_publication_scan(
    tmp_path, monkeypatch
):
    state = tmp_path / "state"
    expected = _session_document(tmp_path)
    original_iterdir = Path.iterdir

    def entries_before_late_orphan(path):
        entries = tuple(original_iterdir(path))
        if path == state and (state / "session.json") in entries:
            (state / "orphan.json").write_text("{}\n", encoding="utf-8")
        return iter(entries)

    monkeypatch.setattr(Path, "iterdir", entries_before_late_orphan)

    first = BaselineJournal.open(state, expected)

    assert first.session == expected
    assert (state / "orphan.json").is_file()
    with pytest.raises(ValueError, match="unexpected content"):
        BaselineJournal.open(state, expected)


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda session: session.__setitem__("unexpected", True),
        lambda session: session.pop("protocol"),
        lambda session: session.__setitem__("identity", "sha256:" + "0" * 64),
    ],
)
def test_journal_rejects_corrupt_persisted_session_documents(tmp_path, corrupt):
    state = tmp_path / "state"
    expected = _session_document(tmp_path)
    BaselineJournal.open(state, expected)
    persisted = read_json_object(state / "session.json", label="test session")
    corrupt(persisted)
    (state / "session.json").write_text(json.dumps(persisted), encoding="utf-8")

    with pytest.raises(ValueError, match="session does not match"):
        BaselineJournal.open(state, expected)


def test_atomic_write_text_replaces_content_and_cleans_up_temporary_files(tmp_path):
    path = tmp_path / "record.txt"
    path.write_text("old\n", encoding="utf-8")

    atomic_write_text(path, "new\n")

    assert path.read_text(encoding="utf-8") == "new\n"
    assert not list(tmp_path.glob(".record.txt.*"))


def test_atomic_write_json_create_only_never_overwrites_and_cleans_up(tmp_path):
    path = tmp_path / "record.json"
    atomic_write_json(path, {"generation": 1}, create_only=True)

    with pytest.raises(FileExistsError):
        atomic_write_json(path, {"generation": 2}, create_only=True)

    assert read_json_object(path, label="record") == {"generation": 1}
    assert not list(tmp_path.glob(".record.json.*"))


def test_journal_promotes_an_inflight_trial_and_resumes_the_slot(tmp_path):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    trial = _valid_raw_trial(workload)
    atomic_write_json(attempt.path, trial.to_dict(), create_only=True)

    journal.accept_inflight(attempt, trial)

    assert journal.load_accepted((slot,)) == {slot: trial}
    assert not attempt.path.exists()
    with pytest.raises(ValueError, match="accepted slot is immutable"):
        journal.accept_inflight(
            JournalAttempt(slot, 1, journal.inflight_path(slot, 1)),
            replace(trial, value=trial.value + 1.0),
        )


def test_journal_preserves_rejected_trial_and_uses_a_new_attempt_number(tmp_path):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    trial = replace(
        _valid_raw_trial(workload),
        environment_status={
            **_valid_raw_trial(workload).environment_status,
            "thermal_state": "fair",
            "thermal_state_raw_value": 1,
        },
    )
    atomic_write_json(attempt.path, trial.to_dict(), create_only=True)

    journal.reject_inflight(attempt, trial, reason="non-nominal-thermal")

    rejected = read_json_object(journal.rejected_path(slot, 0), label="rejected trial")
    assert rejected["journal_attempt_index"] == 0
    assert rejected["reason"] == "non-nominal-thermal"
    assert rejected["trial"]["environment_status"]["thermal_state_raw_value"] == 1
    assert journal.next_attempt(slot).journal_attempt_index == 1


def test_journal_records_every_preflight_observation(tmp_path):
    workload = build_canonical_workload()
    trial = _valid_raw_trial(workload)
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    observation = {
        "observed_at_utc": "2026-08-05T00:00:00+00:00",
        "hardware": trial.hardware,
        "environment_status": trial.environment_status,
        "software_versions": trial.software_versions,
    }

    document = journal.record_preflight(slot, 0, observation)

    assert document["identity"].startswith("sha256:")
    assert (
        read_json_object(journal.preflight_path(slot, 0), label="preflight") == document
    )


def test_journal_rejects_malformed_and_unexpected_accepted_state(tmp_path):
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    atomic_write_json(journal.accepted_path(slot), {}, create_only=True)
    with pytest.raises(ValueError, match="raw trial has an invalid field set"):
        journal.load_accepted((slot,))

    journal.accepted_path(slot).unlink()
    unexpected = BaselineSlot("prepared-data", 1)
    atomic_write_json(journal.accepted_path(unexpected), {}, create_only=True)
    with pytest.raises(ValueError, match="unexpected accepted slot"):
        journal.load_accepted((slot,))


def test_journal_refuses_a_symlinked_preflight_ancestor(tmp_path):
    workload = build_canonical_workload()
    trial = _valid_raw_trial(workload)
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    outside = tmp_path / "outside"
    outside.mkdir()
    preflight = journal.root / "preflight"
    preflight.mkdir()
    (preflight / slot.metric).symlink_to(outside, target_is_directory=True)
    observation = {
        "observed_at_utc": "2026-08-05T00:00:00+00:00",
        "hardware": trial.hardware,
        "environment_status": trial.environment_status,
        "software_versions": trial.software_versions,
    }

    with pytest.raises(ValueError):
        journal.record_preflight(slot, 0, observation)

    assert not list(outside.rglob("*"))


def test_journal_replays_an_accepted_inflight_crash_split(tmp_path):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    trial = _valid_raw_trial(workload)
    atomic_write_json(attempt.path, trial.to_dict(), create_only=True)
    accepted_path = journal.accepted_path(slot)
    accepted_path.parent.mkdir(parents=True)
    os.link(attempt.path, accepted_path)

    journal.accept_inflight(attempt, trial)

    assert not attempt.path.exists()
    assert journal.load_inflight((slot,)) == ()


def test_journal_load_inflight_replays_an_accepted_crash_split(tmp_path):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    trial = _valid_raw_trial(workload)
    atomic_write_json(attempt.path, trial.to_dict(), create_only=True)
    accepted_path = journal.accepted_path(slot)
    accepted_path.parent.mkdir(parents=True)
    os.link(attempt.path, accepted_path)

    assert journal.load_inflight((slot,)) == ()
    assert not attempt.path.exists()


def test_journal_replays_a_rejected_inflight_crash_split(tmp_path):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    trial = _valid_raw_trial(workload)
    atomic_write_json(attempt.path, trial.to_dict(), create_only=True)
    journal.reject_inflight(attempt, trial, reason="non-nominal-thermal")
    atomic_write_json(attempt.path, trial.to_dict(), create_only=True)

    resumed = journal.next_attempt(slot)

    assert resumed.journal_attempt_index == 1
    assert not attempt.path.exists()


@pytest.mark.parametrize("operation", ["accept", "reject"])
def test_journal_rejects_a_mutation_that_skips_an_attempt_index(tmp_path, operation):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    trial = _valid_raw_trial(workload)
    forged = JournalAttempt(slot, 1, journal.inflight_path(slot, 1))
    atomic_write_json(forged.path, trial.to_dict(), create_only=True)

    with pytest.raises(ValueError, match="attempt indices have a gap"):
        if operation == "accept":
            journal.accept_inflight(forged, trial)
        else:
            journal.reject_inflight(forged, trial, reason="non-nominal-thermal")

    assert forged.path.exists()


def test_journal_rejects_a_preflight_index_gap_before_writing(tmp_path):
    workload = build_canonical_workload()
    trial = _valid_raw_trial(workload)
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    observation = {
        "observed_at_utc": "2026-08-05T00:00:00+00:00",
        "hardware": trial.hardware,
        "environment_status": trial.environment_status,
        "software_versions": trial.software_versions,
    }

    with pytest.raises(ValueError, match="preflight indices have a gap"):
        journal.record_preflight(slot, 1, observation)

    assert not journal.preflight_path(slot, 1).exists()


def test_journal_rejects_a_recovery_index_gap_before_writing(tmp_path):
    workload = build_canonical_workload()
    trial = _valid_raw_trial(workload)
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    preflight = journal.record_preflight(
        slot,
        0,
        {
            "observed_at_utc": "2026-08-05T00:00:00+00:00",
            "hardware": trial.hardware,
            "environment_status": trial.environment_status,
            "software_versions": trial.software_versions,
        },
    )

    with pytest.raises(ValueError, match="thermal recovery indices have a gap"):
        journal.record_recovery_trigger(
            slot, 1, {"source": "preflight", "preflight": preflight}
        )


def test_journal_requires_a_recovery_trigger_preflight_from_its_slot(tmp_path):
    workload = build_canonical_workload()
    trial = _valid_raw_trial(workload)
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    other_slot = BaselineSlot("prepared-data", 1)
    observation = {
        "observed_at_utc": "2026-08-05T00:00:00+00:00",
        "hardware": trial.hardware,
        "environment_status": trial.environment_status,
        "software_versions": trial.software_versions,
    }
    journal.record_preflight(slot, 0, observation)
    other_preflight = journal.record_preflight(other_slot, 0, observation)

    with pytest.raises(ValueError, match="does not match a persisted preflight"):
        journal.record_recovery_trigger(
            slot, 0, {"source": "preflight", "preflight": other_preflight}
        )


def test_journal_requires_a_recovery_trigger_rejected_identity_from_its_slot(tmp_path):
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)

    with pytest.raises(ValueError, match="does not match a persisted rejected trial"):
        journal.record_recovery_trigger(
            slot,
            0,
            {
                "source": "rejected-trial",
                "rejected_trial_identity": "sha256:" + "0" * 64,
            },
        )


def test_journal_rejects_boolean_rejected_attempt_index(tmp_path):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    trial = _valid_raw_trial(workload)
    zero_body = {
        "kind": "sml-baseline-rejected-trial",
        "version": 1,
        "journal_attempt_index": 0,
        "reason": "non-nominal-thermal",
        "trial": trial.to_dict(),
    }
    one_body = {**zero_body, "journal_attempt_index": True}
    atomic_write_json(
        journal.rejected_path(slot, 0),
        {
            **zero_body,
            "identity": structured_identity(
                "sml-baseline-rejected-trial-v1", zero_body
            ),
        },
        create_only=True,
    )
    atomic_write_json(
        journal.rejected_path(slot, 1),
        {
            **one_body,
            "identity": structured_identity("sml-baseline-rejected-trial-v1", one_body),
        },
        create_only=True,
    )

    with pytest.raises(ValueError, match="journal attempt index must be non-negative"):
        journal.next_attempt(slot)


def test_journal_load_accepted_replays_only_a_hard_linked_crash_split(tmp_path):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    trial = _valid_raw_trial(workload)
    atomic_write_json(attempt.path, trial.to_dict(), create_only=True)
    accepted_path = journal.accepted_path(slot)
    accepted_path.parent.mkdir(parents=True)
    os.link(attempt.path, accepted_path)

    assert journal.load_accepted((slot,)) == {slot: trial}
    assert not attempt.path.exists()


def test_journal_does_not_replay_independently_serialized_accepted_content(tmp_path):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    trial = _valid_raw_trial(workload)
    atomic_write_json(attempt.path, trial.to_dict(), create_only=True)
    accepted_path = journal.accepted_path(slot)
    accepted_path.parent.mkdir(parents=True)
    accepted_path.write_text(
        json.dumps(trial.to_dict(), separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="accepted transition"):
        journal.load_accepted((slot,))

    assert attempt.path.exists()


def test_journal_preserves_a_gapped_inflight_split_before_reconciliation(tmp_path):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    forged = JournalAttempt(slot, 1, journal.inflight_path(slot, 1))
    trial = _valid_raw_trial(workload)
    atomic_write_json(forged.path, trial.to_dict(), create_only=True)
    accepted_path = journal.accepted_path(slot)
    accepted_path.parent.mkdir(parents=True)
    os.link(forged.path, accepted_path)

    with pytest.raises(ValueError, match="attempt indices have a gap"):
        journal.load_inflight((slot,))

    assert forged.path.exists()


def test_journal_preserves_multiple_inflight_split_candidates(tmp_path):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    first = JournalAttempt(slot, 0, journal.inflight_path(slot, 0))
    second = JournalAttempt(slot, 1, journal.inflight_path(slot, 1))
    trial = _valid_raw_trial(workload)
    atomic_write_json(first.path, trial.to_dict(), create_only=True)
    atomic_write_json(second.path, trial.to_dict(), create_only=True)
    accepted_path = journal.accepted_path(slot)
    accepted_path.parent.mkdir(parents=True)
    os.link(first.path, accepted_path)

    with pytest.raises(ValueError, match="multiple accepted transition candidates"):
        journal.load_inflight((slot,))

    assert first.path.exists()
    assert second.path.exists()


def test_journal_rejects_a_foreign_persisted_preflight_trigger_before_sampling(
    tmp_path,
):
    workload = build_canonical_workload()
    trial = _valid_raw_trial(workload)
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    other_slot = BaselineSlot("prepared-data", 1)
    observation = {
        "observed_at_utc": "2026-08-05T00:00:00+00:00",
        "hardware": trial.hardware,
        "environment_status": trial.environment_status,
        "software_versions": trial.software_versions,
    }
    other_preflight = journal.record_preflight(other_slot, 0, observation)
    body = {
        "kind": "sml-baseline-thermal-recovery-trigger",
        "version": 1,
        "source": "preflight",
        "preflight": other_preflight,
    }
    trigger = {
        **body,
        "identity": structured_identity(
            "sml-baseline-thermal-recovery-trigger-v1", body
        ),
    }
    trigger_path = (
        journal.root / "thermal-waits" / slot.metric / "0" / "0" / "trigger.json"
    )
    atomic_write_json(trigger_path, trigger, create_only=True)

    with pytest.raises(ValueError, match="does not match a persisted preflight"):
        journal.record_thermal_sample(
            slot,
            0,
            0,
            {
                "schema_version": 1,
                "observed_at_utc": "2026-08-05T00:00:01+00:00",
                "elapsed_seconds": 0.0,
                "hardware": trial.hardware,
                "environment_status": trial.environment_status,
                "software_versions": trial.software_versions,
            },
        )

    assert not (trigger_path.parent / "0.json").exists()


def test_journal_rejects_a_stale_persisted_rejected_trigger_before_summary(tmp_path):
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    body = {
        "kind": "sml-baseline-thermal-recovery-trigger",
        "version": 1,
        "source": "rejected-trial",
        "rejected_trial_identity": "sha256:" + "0" * 64,
    }
    trigger = {
        **body,
        "identity": structured_identity(
            "sml-baseline-thermal-recovery-trigger-v1", body
        ),
    }
    trigger_path = (
        journal.root / "thermal-waits" / slot.metric / "0" / "0" / "trigger.json"
    )
    atomic_write_json(trigger_path, trigger, create_only=True)

    with pytest.raises(ValueError, match="does not match a persisted rejected trial"):
        journal.record_recovery_summary(
            slot,
            0,
            {"outcome": "timeout", "duration_seconds": 0.0, "sample_count": 0},
        )

    assert not (trigger_path.parent / "summary.json").exists()


def test_thermal_recovery_requires_five_continuous_nominal_minutes():
    clock = _RecoveryClock()
    samples = []
    result = wait_for_nominal_thermal_window(
        collect=_recovery_collect(["nominal"] * 4 + ["fair"] + ["nominal"] * 20),
        expected_hardware={"chip": "Apple M5"},
        expected_software_versions={"python": "3.12.13"},
        required_environment=build_canonical_workload().required_environment,
        record_sample=lambda index, sample: samples.append((index, sample)),
        deadline=clock() + 7_200,
        clock=clock,
        sleep=clock.sleep,
        utc_now=lambda: "2026-08-05T00:00:00+00:00",
    )

    nominal_times = [
        sample["elapsed_seconds"]
        for _index, sample in samples
        if sample["environment_status"]["thermal_state"] == "nominal"
    ]
    assert result.duration_seconds >= 300
    assert nominal_times[-1] - nominal_times[4] >= 300


def test_thermal_recovery_times_out_without_losing_samples():
    clock = _RecoveryClock()
    samples = []
    with pytest.raises(ThermalRecoveryTimeout, match="two-hour deadline"):
        wait_for_nominal_thermal_window(
            collect=_recovery_collect(["fair"]),
            expected_hardware={"chip": "Apple M5"},
            expected_software_versions={"python": "3.12.13"},
            required_environment=build_canonical_workload().required_environment,
            record_sample=lambda index, sample: samples.append((index, sample)),
            deadline=120.0,
            clock=clock,
            sleep=clock.sleep,
            utc_now=lambda: "2026-08-05T00:00:00+00:00",
        )
    assert samples[-1][1]["environment_status"]["thermal_state"] == "fair"
    assert samples[-1][1]["elapsed_seconds"] >= 120


def test_thermal_recovery_enforces_a_deadline_at_a_nominal_window_boundary():
    clock = _RecoveryClock()
    samples = []

    with pytest.raises(ThermalRecoveryTimeout, match="two-hour deadline") as raised:
        wait_for_nominal_thermal_window(
            collect=_recovery_collect(["nominal"]),
            expected_hardware={"chip": "Apple M5"},
            expected_software_versions={"python": "3.12.13"},
            required_environment=build_canonical_workload().required_environment,
            record_sample=lambda index, sample: samples.append((index, sample)),
            deadline=324.0,
            clock=clock,
            sleep=clock.sleep,
            utc_now=lambda: "2026-08-05T00:00:00+00:00",
        )

    assert raised.value.result.duration_seconds == 324.0
    assert samples[-1][1]["elapsed_seconds"] == 324.0


def test_recovery_import_does_not_eagerly_import_runner():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import v2.benchmarks.recovery; "
            "assert 'v2.benchmarks.runner' not in sys.modules",
        ],
        cwd=Path(__file__).resolve().parents[3],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("target", "field", "value"),
    [
        ("status", "power_connected", False),
        ("status", "power_connected", 1),
        ("status", "low_power_mode", True),
        ("status", "low_power_mode", 0),
        ("status", "memory_pressure", "warning"),
        ("status", "competing_gpu_workload", True),
        ("status", "competing_gpu_workload", 0),
        ("hardware", "chip", "Apple M4"),
        ("software", "python", "3.12.12"),
    ],
)
def test_thermal_recovery_stops_on_nonthermal_changes(target, field, value):
    clock = _RecoveryClock()
    hardware = {"chip": "Apple M5"}
    status = _recovery_collect(["fair"])()[1]
    software = {"python": "3.12.13"}
    if target == "hardware":
        hardware[field] = value
    elif target == "software":
        software[field] = value
    else:
        status[field] = value

    with pytest.raises(ValueError, match=field):
        wait_for_nominal_thermal_window(
            collect=lambda: (hardware, status, software),
            expected_hardware={"chip": "Apple M5"},
            expected_software_versions={"python": "3.12.13"},
            required_environment=build_canonical_workload().required_environment,
            record_sample=lambda index, sample: None,
            deadline=120.0,
            clock=clock,
            sleep=clock.sleep,
            utc_now=lambda: "2026-08-05T00:00:00+00:00",
        )
