import hashlib
import inspect
import json
import os
import shlex
import stat
import subprocess
import sys
import threading
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import v2.benchmarks.journal as baseline_journal
import v2.benchmarks.runner as benchmark_runner
from v2.benchmarks.adapters import legacy
from v2.benchmarks.adapters.replacement import (
    METRIC_OWNER_IMPORTS,
    ReplacementNativeWorkload,
    UnavailableNativeWorkload,
    resolve_native_workload,
)
from v2.benchmarks.adapters.replacement import (
    run_measured as run_replacement_measured,
)
from v2.benchmarks.adapters.replacement import (
    run_warmup as run_replacement_warmup,
)
from v2.benchmarks.analysis import analyze_pairs
from v2.benchmarks.evidence import (
    build_child_trial_measurement,
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
from v2.benchmarks.journal import (
    BaselineJournal,
    BaselineSlot,
    JournalAttempt,
    atomic_write_json,
    atomic_write_text,
    build_session_document,
    read_json_object,
    require_external_state_directory,
)
from v2.benchmarks.recovery import (
    PostExitMemoryRecoveryResult,
    ThermalRecoveryResult,
    ThermalRecoveryTimeout,
    wait_for_nominal_thermal_window,
    wait_for_post_exit_memory_recovery,
)
from v2.benchmarks.runner import (
    _resolve_comparison_mode,
    _resolve_predecessor_mapping,
    build_baseline_manifest,
    build_comparison_report,
    build_parser,
    capture_baseline_trials,
    classify_trial_environment,
    comparison_has_noise,
    decode_memory_pressure_level,
    decode_thermal_state,
    detect_competing_gpu_workload,
    measure_native_process,
    parse_metrics,
    parse_power_status,
    perform_cooldown,
    process_order,
    publish_baseline_from_journal,
    validate_baseline_manifest,
    validate_baseline_trial,
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
    canonical_execution_order,
    canonical_execution_order_identity,
    canonical_input_identity,
    canonical_metric_projection,
    canonical_workload_identity,
    fixed_canonical_rows,
    fixed_inference_requests,
    fixed_swag_examples,
    harness_content_identity,
    post_exit_recovery_policy,
    semantic_row_content_identity,
    structured_identity,
    write_paired_pretraining_representations,
)

_LEGACY_MODULE_NAMES = frozenset(
    {
        "config",
        "evaluate_sml",
        "ft_swag",
        "infer_sml",
        "lora",
        "prepare_pretraining_data",
        "pretraining_format",
        "sml",
        "tokenizer",
        "train_sml",
        "train_tokenizer",
        "utils",
    }
)


@contextmanager
def _isolated_legacy_modules():
    def is_legacy_name(name):
        return name in _LEGACY_MODULE_NAMES or name.startswith("sml.")

    saved = {
        name: module for name, module in sys.modules.items() if is_legacy_name(name)
    }
    original_path = list(sys.path)
    for name in saved:
        sys.modules.pop(name)
    try:
        yield
    finally:
        for name in tuple(sys.modules):
            if is_legacy_name(name):
                sys.modules.pop(name)
        sys.modules.update(saved)
        sys.path[:] = original_path


class _RecoveryClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
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
    arguments = {
        "reference": [100.0, 100.0, 100.0, 100.0, 100.0],
        "candidate": [96.0, 98.0, 100.0, 103.0, 108.0],
        "direction": "higher-is-better",
        "bootstrap_seed": 1729,
        "resamples": 10_000,
        "minimum_ratio": 0.99,
        "maximum_dispersion": 0.10,
        "require_lower_bound": True,
    }
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
    assert workload.compilation == {
        "compilation_passes": 1,
        "warmup_units": 5,
        "measured_units": 20,
        "fresh_processes": True,
        "state_reset_policy": "fresh-native-workload-per-process",
    }
    assert workload.generation["request_count"] == 32
    assert workload.generation["decode_chunk_size"] == 8
    assert tuple(unit.metric for unit in workload.work_units) == METRIC_NAMES
    assert {unit.metric: unit.measured_units for unit in workload.work_units} == {
        "prepared-data": 100,
        "pretraining-compute": 20,
        "pretraining-end-to-end": 20,
        "swag-end-to-end": 20,
        "inference-prefill": 32,
        "inference-decode": 32,
        "checkpoint-pause": 20,
        "compile-cold-start": 1,
        "peak-metal-memory": 1,
    }

    protocol_neutral = workload.to_dict()
    protocol_neutral["compilation"].pop("warmup_units")
    protocol_neutral["compilation"].pop("measured_units")
    for unit in protocol_neutral["work_units"]:
        unit.pop("measured_units")
    assert (
        structured_identity(
            "sml-benchmark-protocol-neutral-workload-v1", protocol_neutral
        )
        == "sha256:90981b91ce14a96a5b44f40258f44b762b11e96dc82f57586348c84370bf41b2"
    )


def test_canonical_workload_changes_only_prepared_data_to_one_hundred_units():
    legacy = build_canonical_workload(prepared_data_measured_units=20)
    current = build_canonical_workload()
    legacy_raw = legacy.to_dict()
    current_raw = current.to_dict()

    legacy_units = {
        unit["metric"]: unit.pop("measured_units") for unit in legacy_raw["work_units"]
    }
    current_units = {
        unit["metric"]: unit.pop("measured_units") for unit in current_raw["work_units"]
    }

    assert legacy_raw == current_raw
    assert legacy_units == {
        "prepared-data": 20,
        "pretraining-compute": 20,
        "pretraining-end-to-end": 20,
        "swag-end-to-end": 20,
        "inference-prefill": 32,
        "inference-decode": 32,
        "checkpoint-pause": 20,
        "compile-cold-start": 1,
        "peak-metal-memory": 1,
    }
    assert current_units == {**legacy_units, "prepared-data": 100}


def test_benchmark_parser_exposes_explicit_prepared_data_count():
    baseline = build_parser().parse_args(
        [
            "record-baseline",
            "--source-commit",
            benchmark_runner.PINNED_BASELINE_SOURCE_COMMIT,
            "--manifest",
            "manifest.json",
            "--raw-output",
            "raw.jsonl",
            "--state-directory",
            "state",
            "--prepared-data-measure",
            "100",
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

    assert (baseline.measure, baseline.prepared_data_measure) == (20, 100)
    assert (comparison.measure, comparison.prepared_data_measure) == (20, None)


@pytest.mark.parametrize("metric", ("compile-cold-start", "peak-metal-memory"))
def test_canonical_workload_rejects_boolean_measured_units(metric):
    raw = build_canonical_workload().to_dict()
    work_unit = next(unit for unit in raw["work_units"] if unit["metric"] == metric)
    work_unit["measured_units"] = True

    with pytest.raises(ValueError, match="measured_units must be a positive integer"):
        CanonicalWorkload.from_dict(raw)


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
    assert HARNESS_COMPONENTS == (
        Path("v2/benchmarks/schema.py"),
        Path("v2/benchmarks/evidence.py"),
        Path("v2/benchmarks/workload.py"),
        Path("v2/benchmarks/runner.py"),
        Path("v2/benchmarks/journal.py"),
        Path("v2/benchmarks/recovery.py"),
        Path("v2/benchmarks/analysis.py"),
        Path("v2/benchmarks/adapters/legacy.py"),
        Path("v2/benchmarks/adapters/replacement.py"),
        Path("v2/tests/unit/test_benchmark_analysis.py"),
    )
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


def test_replacement_adapter_enables_real_prepared_data_owner():
    workload = build_canonical_workload(
        model_overrides={"vocab_size": 32},
        loader_overrides={"sequence_length": 8},
        row_count=32,
    )

    native = resolve_native_workload("prepared-data", workload, Path.cwd())

    assert isinstance(native, ReplacementNativeWorkload)
    assert native.owner_import == "sml.data.pretraining"
    assert (
        native.canonical_row_identity
        == workload.semantic_identities["canonical_training_rows"]
    )
    assert native.canonical_projection == canonical_metric_projection(
        "prepared-data", workload
    )
    native.runtime.close()


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


def test_peak_memory_runs_one_measured_step_after_five_warmups_and_peak_reset():
    workload = build_canonical_workload()
    peak_unit = next(
        unit for unit in workload.work_units if unit.metric == "peak-metal-memory"
    )
    events = []
    adapter = SimpleNamespace(
        run_warmup=lambda _metric, _native, units: events.append(("warmup", units)),
        run_measured=lambda _metric, _native, units: (
            events.append(("measured", units)) or 8_192.0
        ),
    )

    measurement = measure_native_process(
        adapter=adapter,
        metric="peak-metal-memory",
        native_workload="native",
        warmup_units=workload.compilation["warmup_units"],
        measured_units=peak_unit.measured_units,
        synchronize=lambda: events.append(("synchronize",)),
        clock=iter((0.0, 1.0, 2.0, 3.0)).__next__,
        peak_memory=lambda: 7_875_602_848,
        reset_peak_memory=lambda: events.append(("reset-peak-memory",)),
    )

    operations = [event for event in events if event != ("synchronize",)]
    assert operations == [
        ("warmup", 1),
        ("warmup", 1),
        ("warmup", 1),
        ("warmup", 1),
        ("warmup", 1),
        ("warmup", 1),
        ("reset-peak-memory",),
        ("measured", 1),
    ]
    assert measurement.value == 7_875_602_848.0


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


def test_canonical_workload_binds_the_post_exit_recovery_policy():
    required = build_canonical_workload().required_environment

    assert required["memory_pressure"] == "normal"
    assert required["measurement_end_memory_pressure_allowed"] == [
        "normal",
        "warning",
    ]
    assert required["post_exit_memory_pressure_allowed"] == ["normal", "warning"]
    assert required["post_exit_memory_pressure"] == "normal"
    assert required["post_exit_recovery_required_for_warning"] is True
    assert required["post_exit_recovery_sample_interval_seconds"] == 5.0
    assert required["post_exit_recovery_timeout_seconds"] == 300.0
    assert required["post_exit_recovery_stability_seconds"] == 30.0
    assert required["post_exit_recovery_evidence_required"] is True


@pytest.mark.parametrize(
    ("raw_value", "state"),
    [(0, "nominal"), (1, "fair"), (2, "serious"), (3, "critical")],
)
def test_thermal_state_retains_foundation_raw_value(raw_value, state):
    assert decode_thermal_state(raw_value) == state
    validate_thermal_observation(
        {"thermal_state": state, "thermal_state_raw_value": raw_value}
    )


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


def test_record_baseline_parser_requires_prepared_data_measure():
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
                "--state-directory",
                "state",
            ]
        )


def test_benchmark_parser_defaults_to_the_shorter_protocol():
    baseline = build_parser().parse_args(
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
    tmp_path, monkeypatch, request
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
    cleanup = ExitStack()
    request.addfinalizer(cleanup.close)
    source_root = tmp_path / "legacy-source"
    cleanup.enter_context(
        benchmark_runner._managed_detached_worktree(
            Path.cwd(),
            benchmark_runner.PINNED_BASELINE_SOURCE_COMMIT,
            source_root,
        )
    )
    cleanup.enter_context(_isolated_legacy_modules())

    for metric in METRIC_NAMES:
        native = legacy.resolve_native_workload(metric, workload, source_root)
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
                        *args,
                        _events=events,
                        _original=original,
                        _label=label,
                        **kwargs,
                    ):
                        _events.append(_label)
                        return _original(*args, **kwargs)

                    patch.setattr(runtime.train, name, recording_helper)
                original_update = runtime.optimizer.update

                def recording_update(
                    *args,
                    _events=events,
                    _original_update=original_update,
                    **kwargs,
                ):
                    _events.append("update")
                    return _original_update(*args, **kwargs)

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
            "python": "3.12.13",
            "mlx": "0.32.0",
            "numpy": "2.4.6",
            "sentencepiece": "0.2.1",
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
        "source_commit": "3687f8b3214a44c675ae67af52e4997762f6c634",
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
            "parameter_precision_policy": (
                "legacy BF16 persistent parameters and BF16 Adam moments without "
                "authoritative master parameters"
            ),
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


def _valid_recovered_evidence():
    return _recovery_evidence(
        build_canonical_workload(),
        immediate_pressure="warning",
        sample_pressures=("normal",) * 7,
        sample_elapsed=(5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0),
        outcome="recovered",
    )[:4]


def _valid_recovered_raw_trial(workload):
    return _recovery_evidence(
        workload,
        immediate_pressure="warning",
        sample_pressures=("normal",) * 7,
        sample_elapsed=(5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0),
        outcome="recovered",
    )[4]


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


def test_child_measurement_rejects_a_boolean_version():
    measurement = _valid_child_measurement(build_canonical_workload())
    measurement["version"] = True

    with pytest.raises(ValueError, match="version"):
        validate_child_trial_measurement(measurement)


def test_post_exit_observation_rejects_a_boolean_version():
    measurement = _valid_child_measurement(build_canonical_workload())
    post_exit = _valid_post_exit_observation(measurement)
    post_exit["version"] = True

    with pytest.raises(ValueError, match="version"):
        validate_post_exit_observation(post_exit, measurement=measurement)


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

    version_two = trial.to_dict()
    version_two["schema_version"] = 2
    with pytest.raises(ValueError, match="schema version"):
        RawTrial.from_dict(version_two)


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


def _raw_trial_for_recovery_signal(workload, signal):
    if signal == "accepted-not-required":
        return _valid_raw_trial(workload)
    if signal == "accepted-recovered":
        return _valid_recovered_raw_trial(workload)
    if signal in {"start-warning", "end-critical"}:
        trial = _valid_raw_trial(workload)
        start = dict(trial.environment_status["start"])
        end = dict(trial.environment_status["end"])
        if signal == "start-warning":
            start["memory_pressure"] = "warning"
        else:
            end["memory_pressure"] = "critical"
        return _with_environment_observations(trial, start=start, end=end)
    if signal == "recovery-timeout":
        return _recovery_evidence(
            workload,
            immediate_pressure="warning",
            sample_pressures=("warning",),
            sample_elapsed=(300.0,),
            outcome="timeout",
        )[4]
    if signal == "recovery-critical":
        return _recovery_evidence(
            workload, immediate_pressure="critical", outcome="critical"
        )[4]
    if signal == "recovery-interrupted":
        return _recovery_evidence(
            workload, immediate_pressure="warning", outcome="interrupted"
        )[4]
    if signal == "recovery-thermal":
        return _recovery_evidence(
            workload,
            immediate_pressure="warning",
            sample_pressures=("normal",),
            sample_elapsed=(5.0,),
            sample_status_changes=(
                {"thermal_state": "fair", "thermal_state_raw_value": 1},
            ),
            outcome="environment-failure",
            failure_fields=("thermal_state",),
        )[4]
    if signal == "recovery-power":
        return _recovery_evidence(
            workload,
            immediate_pressure="warning",
            sample_pressures=("normal",),
            sample_elapsed=(5.0,),
            sample_status_changes=({"power_connected": False},),
            outcome="environment-failure",
            failure_fields=("power_connected",),
        )[4]
    raise AssertionError(f"unknown recovery signal: {signal}")


@pytest.mark.parametrize(
    ("signal", "outcome", "reason"),
    (
        ("accepted-not-required", "accept", None),
        ("accepted-recovered", "accept", None),
        ("start-warning", "memory-reject", "non-normal-start-memory-pressure"),
        ("end-critical", "memory-reject", "critical-measurement-memory-pressure"),
        ("recovery-timeout", "memory-reject", "persistent-post-exit-memory-pressure"),
        ("recovery-critical", "memory-reject", "critical-post-exit-memory-pressure"),
        ("recovery-interrupted", "memory-reject", "interrupted-post-exit-recovery"),
        ("recovery-thermal", "thermal-reject", "non-nominal-thermal"),
        (
            "recovery-power",
            "environment-reject",
            "post-exit-recovery-environment-violation",
        ),
    ),
)
def test_trial_disposition_uses_the_complete_recovery_chain(signal, outcome, reason):
    workload = build_canonical_workload()
    trial = _raw_trial_for_recovery_signal(workload, signal)

    disposition = classify_trial_environment(workload, trial)

    assert (disposition.outcome, disposition.reason) == (outcome, reason)


def test_trial_disposition_is_performance_independent():
    workload = build_canonical_workload()
    trial = _raw_trial_for_recovery_signal(workload, "recovery-timeout")
    changed = _with_trial_payload(
        trial,
        value=123.0,
        elapsed_seconds=17.0,
        compilation_seconds=0.25,
        peak_memory_bytes=9_999,
    )

    assert classify_trial_environment(workload, changed) == (
        classify_trial_environment(workload, trial)
    )


@pytest.mark.parametrize("endpoint", ("start", "end", "post_exit"))
def test_trial_thermal_disposition_rejects_each_non_nominal_observation(endpoint):
    workload = build_canonical_workload()
    trial = _valid_raw_trial(workload)
    statuses = {
        name: dict(trial.environment_status[name])
        for name in ("start", "end", "post_exit")
    }
    statuses[endpoint].update(thermal_state="fair", thermal_state_raw_value=1)
    changed = _with_environment_observations(trial, **statuses)

    disposition = classify_trial_environment(workload, changed)

    assert (disposition.outcome, disposition.reason) == (
        "thermal-reject",
        "non-nominal-thermal",
    )


@pytest.mark.parametrize(
    ("endpoint", "change", "message"),
    (
        ("start", {"power_connected": False}, "power_connected"),
        ("end", {"power_mode": "changed"}, "power_mode"),
        ("start", {"competing_gpu_workload": True}, "competing_gpu_workload"),
    ),
)
def test_rejected_environment_override_never_allows_power_policy_failures(
    endpoint, change, message
):
    workload = build_canonical_workload()
    trial = _valid_raw_trial(workload)
    statuses = {
        name: dict(trial.environment_status[name])
        for name in ("start", "end", "post_exit")
    }
    statuses[endpoint].update(change)
    changed = _with_environment_observations(trial, **statuses)

    with pytest.raises(ValueError, match=message):
        benchmark_runner._validate_acceptance_environment(
            workload, changed, allow_rejected_environment=True
        )


def test_rejected_environment_override_allows_named_immediate_failure_only():
    workload = build_canonical_workload()
    trial = _valid_raw_trial(workload)
    post_exit = {
        **trial.environment_status["post_exit"],
        "low_power_mode": True,
    }
    changed = _with_environment_observations(trial, post_exit=post_exit)

    with pytest.raises(ValueError, match="low_power_mode"):
        benchmark_runner._validate_acceptance_environment(workload, changed)
    benchmark_runner._validate_acceptance_environment(
        workload, changed, allow_rejected_environment=True
    )


@pytest.mark.parametrize(
    ("signal", "message"),
    (
        ("recovery-power", "power_connected"),
        ("recovery-thermal", "thermal_state"),
    ),
)
def test_rejected_environment_override_validates_nested_recovery_samples(
    signal, message
):
    workload = build_canonical_workload()
    trial = _raw_trial_for_recovery_signal(workload, signal)

    with pytest.raises(ValueError, match=message):
        benchmark_runner._validate_acceptance_environment(workload, trial)
    benchmark_runner._validate_acceptance_environment(
        workload, trial, allow_rejected_environment=True
    )


@pytest.mark.parametrize(
    ("change", "message"),
    (
        ({"hardware_changes": {"chip": "unexpected"}}, "required chip"),
        ({"software_changes": {"python": "3.12.12"}}, "Python version"),
        ({"identity_mismatch": True}, "embedded evidence"),
    ),
)
def test_rejected_environment_override_never_allows_proof_failures(change, message):
    workload = build_canonical_workload()
    trial = _valid_raw_trial(workload)
    if change.get("identity_mismatch"):
        changed = replace(
            trial,
            evidence_session_identity="sha256:" + "e" * 64,
        )
    else:
        changed = _with_observation_changes(trial, **change)

    with pytest.raises(ValueError, match=message):
        benchmark_runner._validate_acceptance_environment(
            workload, changed, allow_rejected_environment=True
        )


def test_rejected_environment_override_allows_only_retryable_dispositions():
    workload = build_canonical_workload()
    trial = _valid_raw_trial(workload)
    statuses = {
        name: dict(trial.environment_status[name])
        for name in ("start", "end", "post_exit")
    }
    statuses["post_exit"]["memory_pressure"] = "warning"
    changed = _with_environment_observations(trial, **statuses)

    with pytest.raises(ValueError, match="interrupted-post-exit-recovery"):
        benchmark_runner._validate_acceptance_environment(workload, changed)
    benchmark_runner._validate_acceptance_environment(
        workload, changed, allow_rejected_environment=True
    )


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
        warmup_units=5,
        measured_units=20,
        paired_representations=paired_representations,
        baseline_version=2,
    )

    validate_baseline_manifest(manifest, trials)
    assert manifest["paired_pretraining_representations"] == paired_representations
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
            paired_representations=_valid_paired_representations(workload),
            baseline_version=2,
        )


def _resign_baseline(manifest):
    body = {key: value for key, value in manifest.items() if key != "identity"}
    manifest["identity"] = structured_identity(
        benchmark_runner._baseline_identity_domain(manifest["version"]), body
    )


def _baseline_fixture(version):
    prepared_units = 20 if version == 1 else 100
    workload = build_canonical_workload(prepared_data_measured_units=prepared_units)
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
        paired_representations=_valid_paired_representations(workload),
        baseline_version=version,
    )
    return manifest, trials


@pytest.mark.parametrize(
    ("version", "prepared_units", "domain", "has_explicit_field"),
    [
        (1, 20, "sml-performance-baseline-v1", False),
        (2, 100, "sml-performance-baseline-v2", True),
    ],
)
def test_baseline_versions_bind_their_exact_prepared_data_protocol(
    version, prepared_units, domain, has_explicit_field
):
    manifest, trials = _baseline_fixture(version)

    validate_baseline_manifest(manifest, trials)
    assert manifest["version"] == version
    assert manifest["identity"] == structured_identity(
        domain, {key: value for key, value in manifest.items() if key != "identity"}
    )
    assert (
        "prepared_data_measured_units" in manifest["protocol"]
    ) is has_explicit_field
    assert (
        next(
            unit["measured_units"]
            for unit in manifest["canonical_workload"]["work_units"]
            if unit["metric"] == "prepared-data"
        )
        == prepared_units
    )


def test_committed_version_one_baseline_remains_valid():
    manifest_path = Path("v2/benchmarks/manifests/baseline-3687f8b.json")
    raw_path = Path("v2/benchmarks/results/baseline-3687f8b.jsonl")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    trials = tuple(
        RawTrial.from_dict(json.loads(line))
        for line in raw_path.read_text(encoding="utf-8").splitlines()
    )

    validate_baseline_manifest(manifest, trials)
    assert manifest["version"] == 1


@pytest.mark.parametrize(("version", "prepared_units"), [(1, 100), (2, 20), (2, 99)])
def test_baseline_validator_rejects_cross_version_prepared_data_counts(
    version, prepared_units
):
    manifest, trials = _baseline_fixture(version=version)
    tampered = deepcopy(manifest)
    next(
        unit
        for unit in tampered["canonical_workload"]["work_units"]
        if unit["metric"] == "prepared-data"
    )["measured_units"] = prepared_units
    _resign_baseline(tampered)

    with pytest.raises(ValueError, match="pinned workload"):
        validate_baseline_manifest(tampered, trials)


@pytest.mark.parametrize("version", (True, 1.0))
def test_baseline_validator_rejects_boolean_and_float_version_aliases(version):
    manifest, trials = _baseline_fixture(version=1)
    tampered = deepcopy(manifest)
    tampered["version"] = version
    body = {key: value for key, value in tampered.items() if key != "identity"}
    tampered["identity"] = structured_identity("sml-performance-baseline-v1", body)

    with pytest.raises(ValueError, match="unsupported baseline"):
        validate_baseline_manifest(tampered, trials)


@pytest.mark.parametrize(
    ("field", "value"),
    (("measured_units", 20.0), ("prepared_data_measured_units", 100.0)),
)
def test_baseline_validator_rejects_float_protocol_aliases(field, value):
    manifest, trials = _baseline_fixture(version=2)
    tampered = deepcopy(manifest)
    tampered["protocol"][field] = value
    _resign_baseline(tampered)

    with pytest.raises(ValueError, match="baseline protocol"):
        validate_baseline_manifest(tampered, trials)


@pytest.mark.parametrize(
    ("version", "explicit", "expected"),
    [(1, None, 20), (1, 20, 20), (2, None, 100), (2, 100, 100)],
)
def test_compare_resolves_prepared_data_count_from_baseline(
    version, explicit, expected
):
    baseline, _ = _baseline_fixture(version=version)

    assert (
        benchmark_runner._resolve_prepared_data_measure(
            SimpleNamespace(prepared_data_measure=explicit), baseline
        )
        == expected
    )


@pytest.mark.parametrize(
    ("version", "explicit"),
    [(1, 100), (2, 20), (2, 99), (2, True)],
)
def test_compare_rejects_prepared_data_count_that_disagrees_with_baseline(
    version, explicit
):
    baseline, _ = _baseline_fixture(version=version)

    with pytest.raises(ValueError, match="prepared-data measured units"):
        benchmark_runner._resolve_prepared_data_measure(
            SimpleNamespace(prepared_data_measure=explicit), baseline
        )


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
        warmup_units=5,
        measured_units=20,
        paired_representations=_valid_paired_representations(workload),
        baseline_version=2,
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
        paired_representations=_valid_paired_representations(workload),
        baseline_version=2,
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
        paired_representations=_valid_paired_representations(workload),
        baseline_version=2,
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
    invalid["raw_trials"][1] = _with_trial_payload(
        RawTrial.from_dict(invalid["raw_trials"][1]), process_order=0
    ).to_dict()
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
                        "parameter_precision_policy": REPLACEMENT_PRECISION_POLICY,
                    },
                    value=candidate_value,
                ),
            )
        )
    return trials


def test_evidence_resigned_comparison_and_thermal_variants_are_valid():
    workload = build_canonical_workload()
    comparison_trials = _comparison_trials(
        workload,
        "e" * 40,
        [101.0],
        attempt_index=1,
    )
    thermal_trial = _with_thermal_state(_valid_raw_trial(workload), "fair", 1)

    for trial in (*comparison_trials, thermal_trial):
        validate_raw_trial_evidence(trial)


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


def test_raw_trial_jsonl_rejects_tampered_embedded_evidence(tmp_path):
    trial = _valid_raw_trial(build_canonical_workload())
    raw = replace(trial, value=trial.value + 1.0).to_dict()
    path = tmp_path / "baseline.jsonl"
    path.write_text(json.dumps(raw) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="embedded evidence"):
        benchmark_runner._read_trials(path)


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
        paired_representations=_valid_paired_representations(workload),
        baseline_version=2,
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


def _resign_comparison(report):
    body = {key: value for key, value in report.items() if key != "identity"}
    report["identity"] = structured_identity("sml-performance-comparison-v1", body)


@pytest.mark.parametrize("version", (True, 1.0))
def test_comparison_validator_rejects_boolean_and_float_version_aliases(version):
    _workload, baseline, report = _valid_prepared_comparison()
    report["version"] = version
    _resign_comparison(report)

    with pytest.raises(ValueError, match="unsupported comparison"):
        validate_comparison_report(report, baseline, None)


@pytest.mark.parametrize("baseline_version", (1, 2))
def test_comparison_construction_preserves_the_baseline_version_protocol(
    baseline_version,
):
    baseline, _trials = _baseline_fixture(baseline_version)
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
    if baseline_version == 2:
        expected.add("prepared_data_measured_units")
    assert set(report["protocol"]) == expected


@pytest.mark.parametrize("mutation", ("omit", "extra"))
def test_comparison_validator_rejects_serialized_version_two_protocol_field_changes(
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


def test_comparison_validator_rejects_added_prepared_data_field_for_version_one():
    baseline, _trials = _baseline_fixture(1)
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
    report["protocol"]["prepared_data_measured_units"] = 100
    _resign_comparison(report)

    with pytest.raises(ValueError, match="field set"):
        validate_comparison_report(report, baseline, None)


def test_validate_phase_rejects_version_one_report_against_version_two_baseline(
    tmp_path, monkeypatch
):
    legacy_baseline, _trials = _baseline_fixture(1)
    workload = CanonicalWorkload.from_dict(legacy_baseline["canonical_workload"])
    report = build_comparison_report(
        baseline=legacy_baseline,
        trials=_comparison_trials(workload, "e" * 40, [100.0] * 5, 0),
        candidate_commit="e" * 40,
        minimum_ratio=0.97,
        pretraining_minimum_ratio=None,
        maximum_dispersion=0.02,
        require_lower_bound=False,
        bootstrap_resamples=10_000,
        predecessor_metrics={},
    )
    current_baseline, _trials = _baseline_fixture(2)
    baseline_path = tmp_path / "baseline-v2.json"
    report_path = tmp_path / "report-v1.json"
    baseline_path.write_text(json.dumps(current_baseline), encoding="utf-8")
    report_path.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(benchmark_runner, "_git_root", lambda _path: tmp_path)
    args = SimpleNamespace(
        baseline=baseline_path,
        results=report_path,
        predecessors=json.dumps({"prepared-data": None}),
        phase=2,
        output=None,
    )

    with pytest.raises(ValueError, match="wrong baseline"):
        benchmark_runner._validate_phase(args)


def test_validate_final_rejects_version_one_report_against_version_two_baseline(
    tmp_path, monkeypatch
):
    legacy_baseline, _trials = _baseline_fixture(1)
    workload = CanonicalWorkload.from_dict(legacy_baseline["canonical_workload"])
    report = build_comparison_report(
        baseline=legacy_baseline,
        trials=_comparison_trials(workload, "e" * 40, [100.0] * 5, 0),
        candidate_commit="e" * 40,
        minimum_ratio=0.97,
        pretraining_minimum_ratio=None,
        maximum_dispersion=0.02,
        require_lower_bound=False,
        bootstrap_resamples=10_000,
        predecessor_metrics={},
    )
    proof = {
        "report_identity": "sha256:" + "a" * 64,
        "result_identity": "sha256:" + "b" * 64,
    }
    report["metrics"] = {metric: {} for metric in benchmark_runner.FINAL_METRICS}
    report["comparison_mode"] = benchmark_runner.COMPARISON_FINAL
    report["predecessors"] = {
        metric: proof if metric in benchmark_runner.FINAL_PREDECESSOR_METRICS else None
        for metric in benchmark_runner.FINAL_METRICS
    }
    _resign_comparison(report)
    current_baseline, _trials = _baseline_fixture(2)
    baseline_path = tmp_path / "baseline-v2.json"
    report_path = tmp_path / "report-v1.json"
    raw_path = tmp_path / "report-v1.jsonl"
    baseline_path.write_text(json.dumps(current_baseline), encoding="utf-8")
    report_path.write_text(json.dumps(report), encoding="utf-8")
    raw_path.write_text(
        "".join(json.dumps(raw) + "\n" for raw in report["raw_trials"]),
        encoding="utf-8",
    )
    monkeypatch.setattr(benchmark_runner, "_git_root", lambda _path: tmp_path)
    monkeypatch.setattr(
        benchmark_runner,
        "_resolve_predecessor_mapping",
        lambda *_args, **_kwargs: (
            {metric: None for metric in benchmark_runner.FINAL_METRICS},
            {},
            {},
        ),
    )
    args = SimpleNamespace(
        baseline=baseline_path,
        report=report_path,
        raw_input=raw_path,
    )

    with pytest.raises(ValueError, match="wrong baseline"):
        benchmark_runner._validate_final(args)


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


def test_comparison_validator_rejects_false_round_trip_and_precision_annotation():
    workload, baseline, report = _valid_prepared_comparison()
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

    candidate_commit = "f" * 40
    trials = []
    for pair_index in range(5):
        order = process_order(pair_index)
        reference = _valid_raw_trial(
            workload, metric="pretraining-end-to-end", pair_index=pair_index
        )
        trials.extend(
            (
                _with_trial_payload(
                    reference,
                    process_order=order.index("reference"),
                    value=100.0,
                ),
                _with_trial_payload(
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
        warmup_units=5,
        measured_units=20,
        paired_representations=_valid_paired_representations(workload),
        baseline_version=2,
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
        warmup_units=5,
        measured_units=20,
        paired_representations=_valid_paired_representations(workload),
        baseline_version=2,
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
        warmup_units=5,
        measured_units=20,
        paired_representations=_valid_paired_representations(workload),
        baseline_version=2,
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
        protocol=protocol or {"pairs": 5, "warmup_units": 5, "measured_units": 20},
        hardware=hardware or {"chip": "Apple M5"},
        software_versions=software_versions or {"python": "3.12.13", "mlx": "0.32.0"},
        paired_representations=paired_representations
        or {"canonical_row_identity": "sha256:" + "c" * 64},
        manifest_path=tmp_path / manifest_name,
        raw_output_path=tmp_path / raw_output_name,
    )


def test_version_two_baseline_session_and_replay_bind_prepared_data_units(tmp_path):
    workload = build_canonical_workload()
    protocol = benchmark_runner._baseline_protocol(2, workload)
    session = _session_document(tmp_path, protocol=protocol)
    journal = BaselineJournal.open(tmp_path / "state", session)

    assert journal.session["protocol"]["measured_units"] == 20
    assert journal.session["protocol"]["prepared_data_measured_units"] == 100
    command = benchmark_runner.canonical_baseline_command(journal)
    assert "--measure 20" in command
    assert "--prepared-data-measure 100" in command


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


def test_interrupted_memory_rejection_stops_then_manual_resume_fills_only_missing_slot(
    tmp_path,
):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    first = BaselineSlot("prepared-data", 0)
    second = BaselineSlot("prepared-data", 1)
    first_attempt = journal.next_attempt(first)
    accepted = _persist_journal_trial(
        journal,
        first_attempt,
        _valid_raw_trial(workload, pair_index=0),
    )
    journal.accept_inflight(first_attempt, accepted)
    normal = _valid_raw_trial(workload, pair_index=1)
    warning = dict(normal.environment_status["post_exit"])
    warning["memory_pressure"] = "warning"
    rejected = _with_environment_observations(normal, post_exit=warning)
    first_run_launches = []

    with pytest.raises(
        benchmark_runner.MemoryPressureTrialRejected,
        match="interrupted-post-exit-recovery",
    ):
        capture_baseline_trials(
            journal=journal,
            slots=(first, second),
            launch_trial=lambda slot, attempt: (
                first_run_launches.append(slot)
                or _persist_journal_trial(journal, attempt, rejected)
            ),
            preflight=lambda: _preflight_from_trial(normal),
            validate_preflight=lambda hardware, status, software: None,
            recover=lambda slot, index, deadline, trigger: pytest.fail(
                "memory rejection entered thermal recovery"
            ),
            validate_trial=_baseline_validator(workload),
            classify_trial=lambda trial: classify_trial_environment(workload, trial),
            progress=lambda message: None,
        )

    assert first_run_launches == [second]
    assert journal.load_accepted((first, second)) == {first: accepted}
    assert (
        read_json_object(journal.rejected_path(second, 0), label="memory rejection")[
            "reason"
        ]
        == "interrupted-post-exit-recovery"
    )

    second_run_launches = []
    trials = capture_baseline_trials(
        journal=journal,
        slots=(first, second),
        launch_trial=lambda slot, attempt: (
            second_run_launches.append(slot)
            or _persist_journal_trial(journal, attempt, normal)
        ),
        preflight=lambda: _preflight_from_trial(normal),
        validate_preflight=lambda hardware, status, software: None,
        recover=lambda slot, index, deadline, trigger: pytest.fail(
            "memory rejection created a thermal trigger"
        ),
        validate_trial=_baseline_validator(workload),
        classify_trial=lambda trial: classify_trial_environment(workload, trial),
        progress=lambda message: None,
    )

    assert second_run_launches == [second]
    assert trials[0] == accepted
    assert (trials[1].metric, trials[1].pair_index, trials[1].value) == (
        normal.metric,
        normal.pair_index,
        normal.value,
    )


@pytest.mark.parametrize(
    ("endpoint", "pressure", "reason"),
    (
        ("start", "warning", "non-normal-start-memory-pressure"),
        ("end", "critical", "critical-measurement-memory-pressure"),
    ),
)
def test_capture_stops_immediately_for_rejected_child_memory_observations(
    tmp_path, endpoint, pressure, reason
):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    normal = _valid_raw_trial(workload)
    status = dict(normal.environment_status[endpoint])
    status["memory_pressure"] = pressure
    rejected = _with_environment_observations(normal, **{endpoint: status})
    launches = []

    with pytest.raises(benchmark_runner.MemoryPressureTrialRejected) as raised:
        capture_baseline_trials(
            journal=journal,
            slots=(slot,),
            launch_trial=lambda current_slot, attempt: (
                launches.append(current_slot)
                or _persist_journal_trial(journal, attempt, rejected)
            ),
            preflight=lambda: _preflight_from_trial(normal),
            validate_preflight=lambda hardware, status, software: None,
            recover=lambda *args: pytest.fail(
                "child memory rejection entered thermal recovery"
            ),
            validate_trial=_baseline_validator(workload),
            classify_trial=lambda trial: classify_trial_environment(workload, trial),
            progress=lambda message: None,
        )

    assert raised.value.slot == slot
    assert raised.value.reason == reason
    assert launches == [slot]
    assert (
        read_json_object(journal.rejected_path(slot, 0), label="memory rejection")[
            "reason"
        ]
        == reason
    )


def test_capture_accepts_child_end_warning_when_post_exit_is_normal(tmp_path):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    normal = _valid_raw_trial(workload)
    warning = dict(normal.environment_status["end"])
    warning["memory_pressure"] = "warning"
    diagnostic_warning = _with_environment_observations(normal, end=warning)
    launches = []

    trials = capture_baseline_trials(
        journal=journal,
        slots=(slot,),
        launch_trial=lambda current_slot, attempt: (
            launches.append(current_slot)
            or _persist_journal_trial(journal, attempt, diagnostic_warning)
        ),
        preflight=lambda: _preflight_from_trial(normal),
        validate_preflight=lambda hardware, status, software: None,
        recover=lambda *args: pytest.fail("accepted memory warning entered recovery"),
        validate_trial=_baseline_validator(workload),
        classify_trial=lambda trial: classify_trial_environment(workload, trial),
        progress=lambda message: None,
    )

    assert launches == [slot]
    assert trials[0].environment_status["measurement_memory_pressure"] == "warning"
    assert journal.load_accepted((slot,)) == {slot: trials[0]}


def test_capture_rejects_measurement_only_crash_then_launches_a_new_attempt(tmp_path):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    normal = _valid_raw_trial(workload)
    attempt = journal.next_attempt(slot)
    measurement, _post_exit, _samples, _recovery, _persisted = _journal_trial_evidence(
        journal, attempt, normal
    )
    atomic_write_json(journal.measurement_path(slot, 0), measurement, create_only=True)
    events = []

    trials = capture_baseline_trials(
        journal=journal,
        slots=(slot,),
        launch_trial=lambda current_slot, current_attempt: (
            events.append(("launch", current_attempt.journal_attempt_index))
            or _persist_journal_trial(journal, current_attempt, normal)
        ),
        preflight=lambda: (
            events.append(("preflight", 1)) or _preflight_from_trial(normal)
        ),
        validate_preflight=lambda hardware, status, software: None,
        recover=lambda *args: pytest.fail("missing evidence entered thermal recovery"),
        validate_trial=_baseline_validator(workload),
        classify_trial=lambda trial: classify_trial_environment(workload, trial),
        progress=lambda message: None,
    )

    assert events == [("preflight", 1), ("launch", 1)]
    assert (
        read_json_object(
            journal.rejected_path(slot, 0), label="missing evidence rejection"
        )["reason"]
        == "missing-immediate-post-exit-evidence"
    )
    assert trials[0].pair_index == 0

    resumed = capture_baseline_trials(
        journal=journal,
        slots=(slot,),
        launch_trial=lambda current_slot, current_attempt: pytest.fail(
            "accepted slot was relaunched after missing-evidence rejection"
        ),
        preflight=lambda: pytest.fail(
            "accepted slot reached preflight after missing-evidence rejection"
        ),
        validate_preflight=lambda hardware, status, software: None,
        recover=lambda *args: pytest.fail(
            "missing-evidence rejection created a thermal trigger on resume"
        ),
        validate_trial=_baseline_validator(workload),
        classify_trial=lambda trial: classify_trial_environment(workload, trial),
        progress=lambda message: None,
    )

    assert resumed == trials


def test_capture_never_creates_raw_evidence_for_an_unstaged_launcher_result(tmp_path):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    normal = _valid_raw_trial(workload)

    with pytest.raises(FileNotFoundError):
        capture_baseline_trials(
            journal=journal,
            slots=(slot,),
            launch_trial=lambda current_slot, current_attempt: normal,
            preflight=lambda: _preflight_from_trial(normal),
            validate_preflight=lambda hardware, status, software: None,
            recover=lambda *args: pytest.fail("unstaged launch entered recovery"),
            validate_trial=_baseline_validator(workload),
            classify_trial=lambda trial: classify_trial_environment(workload, trial),
            progress=lambda message: None,
        )

    assert not journal.inflight_path(slot, 0).exists()


def test_capture_reconstructs_measurement_and_post_exit_before_replacement_launch(
    tmp_path,
):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    measurement, post_exit, _samples, _recovery, expected = _journal_trial_evidence(
        journal, attempt, _valid_raw_trial(workload)
    )
    expected_recovery = build_post_exit_recovery(
        measurement=measurement,
        post_exit=post_exit,
        samples=(),
        policy=post_exit_recovery_policy(workload),
        outcome="not-required",
        duration_seconds=0.0,
        completion_source="crash-reconstruction",
    )
    expected = finalize_raw_trial(measurement, post_exit, (), expected_recovery)
    atomic_write_json(journal.measurement_path(slot, 0), measurement, create_only=True)
    atomic_write_json(journal.post_exit_path(slot, 0), post_exit, create_only=True)

    trials = capture_baseline_trials(
        journal=journal,
        slots=(slot,),
        launch_trial=lambda current_slot, current_attempt: pytest.fail(
            "complete staged evidence launched a replacement"
        ),
        preflight=lambda: pytest.fail(
            "complete staged evidence reached replacement preflight"
        ),
        validate_preflight=lambda hardware, status, software: None,
        recover=lambda *args: pytest.fail(
            "complete staged evidence entered thermal recovery"
        ),
        validate_trial=_baseline_validator(workload),
        classify_trial=lambda trial: classify_trial_environment(workload, trial),
        progress=lambda message: None,
    )

    assert trials == (expected,)
    assert not attempt.path.exists()
    assert journal.load_accepted((slot,)) == {slot: expected}


def test_capture_reconstructs_not_required_summary_before_any_launch(tmp_path):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    measurement, post_exit, _samples, _recovery, expected = _journal_trial_evidence(
        journal, attempt, _valid_raw_trial(workload)
    )
    expected_recovery = build_post_exit_recovery(
        measurement=measurement,
        post_exit=post_exit,
        samples=(),
        policy=post_exit_recovery_policy(workload),
        outcome="not-required",
        duration_seconds=0.0,
        completion_source="crash-reconstruction",
    )
    expected = finalize_raw_trial(measurement, post_exit, (), expected_recovery)
    atomic_write_json(journal.measurement_path(slot, 0), measurement, create_only=True)
    atomic_write_json(journal.post_exit_path(slot, 0), post_exit, create_only=True)
    launches = []

    trials = capture_baseline_trials(
        journal=journal,
        slots=(slot,),
        launch_trial=lambda slot, next_attempt: launches.append(next_attempt),
        preflight=lambda: _preflight_from_trial(expected),
        validate_preflight=lambda hardware, status, software: None,
        recover=lambda slot, index, deadline, trigger: pytest.fail(
            "normal pending evidence entered thermal recovery"
        ),
        validate_trial=_baseline_validator(workload),
        classify_trial=lambda trial: classify_trial_environment(workload, trial),
        progress=lambda message: None,
    )

    assert launches == []
    assert trials == (expected,)
    assert journal.recovery_path(slot, 0).is_file()
    assert (
        read_json_object(journal.recovery_path(slot, 0), label="recovery")["outcome"]
        == "not-required"
    )


def test_capture_rejects_partial_warning_recovery_before_any_launch(tmp_path):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    interrupted = _recovery_evidence(
        workload,
        immediate_pressure="warning",
        sample_pressures=("normal",),
        sample_elapsed=(5.0,),
        outcome="interrupted",
    )[4]
    measurement, post_exit, samples, _recovery, rebound = _journal_trial_evidence(
        journal, attempt, interrupted
    )
    atomic_write_json(journal.measurement_path(slot, 0), measurement, create_only=True)
    atomic_write_json(journal.post_exit_path(slot, 0), post_exit, create_only=True)
    atomic_write_json(
        journal.recovery_sample_path(slot, 0, 0), samples[0], create_only=True
    )
    launches = []

    with pytest.raises(
        benchmark_runner.MemoryPressureTrialRejected,
        match="interrupted-post-exit-recovery",
    ):
        capture_baseline_trials(
            journal=journal,
            slots=(slot,),
            launch_trial=lambda slot, next_attempt: launches.append(next_attempt),
            preflight=lambda: _preflight_from_trial(rebound),
            validate_preflight=lambda hardware, status, software: None,
            recover=lambda slot, index, deadline, trigger: pytest.fail(
                "interrupted memory recovery entered thermal recovery"
            ),
            validate_trial=_baseline_validator(workload),
            classify_trial=lambda trial: classify_trial_environment(workload, trial),
            progress=lambda message: None,
        )

    assert launches == []
    assert (
        read_json_object(journal.recovery_path(slot, 0), label="recovery")["outcome"]
        == "interrupted"
    )


@pytest.mark.parametrize(
    "boundary",
    ("critical", "environment-failure", "stable-completion", "deadline"),
)
def test_capture_reconstructs_decisive_warning_crash_as_interrupted(tmp_path, boundary):
    workload = build_canonical_workload()
    if boundary == "critical":
        terminal = _recovery_evidence(
            workload,
            immediate_pressure="warning",
            sample_pressures=("critical",),
            sample_elapsed=(5.0,),
            outcome="critical",
        )[4]
    elif boundary == "environment-failure":
        terminal = _recovery_evidence(
            workload,
            immediate_pressure="warning",
            sample_pressures=("critical",),
            sample_elapsed=(5.0,),
            sample_status_changes=({"power_connected": False},),
            outcome="environment-failure",
            failure_fields=("power_connected",),
        )[4]
    elif boundary == "stable-completion":
        terminal = _valid_recovered_raw_trial(workload)
    else:
        terminal = _recovery_evidence(
            workload,
            immediate_pressure="warning",
            sample_pressures=("warning",),
            sample_elapsed=(300.0,),
            outcome="timeout",
        )[4]

    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    measurement, post_exit, samples, _summary, rebound = _journal_trial_evidence(
        journal, attempt, terminal
    )
    atomic_write_json(journal.measurement_path(slot, 0), measurement, create_only=True)
    atomic_write_json(journal.post_exit_path(slot, 0), post_exit, create_only=True)
    for index, sample in enumerate(samples):
        atomic_write_json(
            journal.recovery_sample_path(slot, 0, index), sample, create_only=True
        )

    with pytest.raises(
        benchmark_runner.MemoryPressureTrialRejected,
        match="interrupted-post-exit-recovery",
    ):
        capture_baseline_trials(
            journal=journal,
            slots=(slot,),
            launch_trial=lambda *args: pytest.fail(
                "crash reconstruction launched a replacement"
            ),
            preflight=lambda: _preflight_from_trial(rebound),
            validate_preflight=lambda *args: None,
            recover=lambda *args: pytest.fail("memory rejection entered recovery"),
            validate_trial=_baseline_validator(workload),
            classify_trial=lambda trial: classify_trial_environment(workload, trial),
            progress=lambda message: None,
        )

    recovery_path = journal.recovery_path(slot, 0)
    rejection_path = journal.rejected_path(slot, 0)
    recovery = read_json_object(recovery_path, label="recovery")
    rejected = read_json_object(rejection_path, label="rejection")
    assert (
        recovery["version"],
        recovery["completion_source"],
        recovery["outcome"],
    ) == (
        2,
        "crash-reconstruction",
        "interrupted",
    )
    assert rejected["reason"] == "interrupted-post-exit-recovery"
    retained_paths = tuple(
        [journal.recovery_sample_path(slot, 0, index) for index in range(len(samples))]
        + [recovery_path, rejection_path]
    )
    retained = tuple(path.read_bytes() for path in retained_paths)
    launches = []
    normal = _valid_raw_trial(workload)

    capture_baseline_trials(
        journal=journal,
        slots=(slot,),
        launch_trial=lambda current_slot, next_attempt: (
            launches.append(next_attempt.journal_attempt_index)
            or _persist_journal_trial(journal, next_attempt, normal)
        ),
        preflight=lambda: _preflight_from_trial(normal),
        validate_preflight=lambda *args: None,
        recover=lambda *args: pytest.fail("memory rejection created thermal recovery"),
        validate_trial=_baseline_validator(workload),
        classify_trial=lambda trial: classify_trial_environment(workload, trial),
        progress=lambda message: None,
    )

    assert launches == [1]
    assert tuple(path.read_bytes() for path in retained_paths) == retained


@pytest.mark.parametrize(
    ("trial", "exception_name", "reason"),
    (
        (
            lambda workload: _recovery_evidence(
                workload, immediate_pressure="critical", outcome="critical"
            )[4],
            "MemoryPressureTrialRejected",
            "critical-post-exit-memory-pressure",
        ),
        (
            lambda workload: _with_environment_observations(
                _valid_raw_trial(workload),
                post_exit={
                    **_valid_raw_trial(workload).environment_status["post_exit"],
                    "power_connected": False,
                },
            ),
            "EnvironmentTrialRejected",
            "post-exit-recovery-environment-violation",
        ),
    ),
)
def test_capture_reconstructs_terminal_immediate_failure_before_any_launch(
    tmp_path, trial, exception_name, reason
):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    rejected = trial(workload)
    measurement, post_exit, _samples, _recovery, rebound = _journal_trial_evidence(
        journal, attempt, rejected
    )
    atomic_write_json(journal.measurement_path(slot, 0), measurement, create_only=True)
    atomic_write_json(journal.post_exit_path(slot, 0), post_exit, create_only=True)
    launches = []

    exception = getattr(benchmark_runner, exception_name)
    with pytest.raises(exception, match=reason):
        capture_baseline_trials(
            journal=journal,
            slots=(slot,),
            launch_trial=lambda slot, next_attempt: launches.append(next_attempt),
            preflight=lambda: _preflight_from_trial(rebound),
            validate_preflight=lambda hardware, status, software: None,
            recover=lambda *args: pytest.fail(
                "terminal immediate failure entered thermal recovery"
            ),
            validate_trial=_baseline_validator(workload),
            classify_trial=lambda trial: classify_trial_environment(workload, trial),
            progress=lambda message: None,
        )

    assert launches == []
    assert read_json_object(journal.recovery_path(slot, 0), label="recovery")[
        "outcome"
    ] in {"critical", "environment-failure"}
    assert (
        read_json_object(journal.rejected_path(slot, 0), label="rejection")["reason"]
        == reason
    )


def test_capture_resumes_after_interruption_with_only_the_next_attempt(tmp_path):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    interrupted = _recovery_evidence(
        workload,
        immediate_pressure="warning",
        sample_pressures=("normal",),
        sample_elapsed=(5.0,),
        outcome="interrupted",
    )[4]
    measurement, post_exit, samples, _recovery, rebound = _journal_trial_evidence(
        journal, attempt, interrupted
    )
    atomic_write_json(journal.measurement_path(slot, 0), measurement, create_only=True)
    atomic_write_json(journal.post_exit_path(slot, 0), post_exit, create_only=True)
    atomic_write_json(
        journal.recovery_sample_path(slot, 0, 0), samples[0], create_only=True
    )

    with pytest.raises(benchmark_runner.MemoryPressureTrialRejected):
        capture_baseline_trials(
            journal=journal,
            slots=(slot,),
            launch_trial=lambda *args: pytest.fail(
                "interrupted attempt was relaunched"
            ),
            preflight=lambda: _preflight_from_trial(rebound),
            validate_preflight=lambda *args: None,
            recover=lambda *args: pytest.fail("memory rejection entered recovery"),
            validate_trial=_baseline_validator(workload),
            classify_trial=lambda trial: classify_trial_environment(workload, trial),
            progress=lambda message: None,
        )

    retained_paths = (
        journal.recovery_sample_path(slot, 0, 0),
        journal.recovery_path(slot, 0),
        journal.rejected_path(slot, 0),
    )
    retained = tuple(path.read_bytes() for path in retained_paths)
    rejected_document = read_json_object(retained_paths[-1], label="rejection")
    raw_trial = rejected_document["trial"]
    launches = []
    normal = _valid_raw_trial(workload)

    trials = capture_baseline_trials(
        journal=journal,
        slots=(slot,),
        launch_trial=lambda current_slot, next_attempt: (
            launches.append(next_attempt.journal_attempt_index)
            or _persist_journal_trial(journal, next_attempt, normal)
        ),
        preflight=lambda: _preflight_from_trial(normal),
        validate_preflight=lambda *args: None,
        recover=lambda *args: pytest.fail("memory rejection created thermal recovery"),
        validate_trial=_baseline_validator(workload),
        classify_trial=lambda trial: classify_trial_environment(workload, trial),
        progress=lambda message: None,
    )

    assert launches == [1]
    assert len(trials) == 1
    assert tuple(path.read_bytes() for path in retained_paths) == retained
    assert read_json_object(retained_paths[-1], label="rejection")["trial"] == raw_trial


def test_capture_retries_only_the_thermal_slot_and_resumes_accepted_slots(tmp_path):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slots = (
        BaselineSlot("prepared-data", 0),
        BaselineSlot("prepared-data", 1),
    )
    accepted_first = _valid_raw_trial(workload, pair_index=0)
    first_attempt = journal.next_attempt(slots[0])
    accepted_first = _persist_journal_trial(journal, first_attempt, accepted_first)
    journal.accept_inflight(first_attempt, accepted_first)
    launches = []

    def launch(slot, attempt):
        launches.append(slot)
        base = _valid_raw_trial(workload, pair_index=slot.pair_index)
        if len(launches) == 1:
            base = _with_thermal_state(base, "fair", 1)
        return _persist_journal_trial(journal, attempt, base)

    recovered = []
    trials = capture_baseline_trials(
        journal=journal,
        slots=slots,
        launch_trial=launch,
        preflight=lambda: _preflight_from_trial(accepted_first),
        validate_preflight=lambda hardware, status, software: None,
        recover=lambda slot, recovery_index, deadline, trigger: recovered.append(slot),
        validate_trial=lambda trial, allow_rejected_environment: None,
        classify_trial=lambda trial: classify_trial_environment(workload, trial),
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
        trial = _persist_journal_trial(journal, attempt, trial)
        journal.accept_inflight(attempt, trial)
    messages = []

    capture_baseline_trials(
        journal=journal,
        slots=slots,
        launch_trial=lambda slot, attempt: pytest.fail("accepted slot was launched"),
        preflight=lambda: pytest.fail("accepted slot reached preflight"),
        validate_preflight=lambda hardware, status, software: None,
        recover=lambda slot, recovery_index, deadline, trigger: None,
        validate_trial=lambda trial, allow_rejected_environment: None,
        classify_trial=_accept_trial,
        progress=messages.append,
    )

    assert messages == [
        "resumed accepted prepared-data pair 1",
        "resumed accepted prepared-data pair 0",
    ]


def test_capture_passes_the_environment_policy_to_trial_validation_by_keyword(
    tmp_path,
):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    trial = _valid_raw_trial(workload)
    attempt = journal.next_attempt(slot)
    trial = _persist_journal_trial(journal, attempt, trial)
    policies = []

    def validate(current_trial, *, allow_rejected_environment):
        policies.append(allow_rejected_environment)

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
        classify_trial=_accept_trial,
        progress=lambda message: None,
    )

    assert policies == [True]


def test_capture_records_preflight_thermal_trigger_before_launch(tmp_path):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    nominal_trial = _valid_raw_trial(workload)
    fair = {
        **nominal_trial.environment_status["start"],
        "thermal_state": "fair",
        "thermal_state_raw_value": 1,
    }
    preflights = iter(
        [
            (nominal_trial.hardware, fair, nominal_trial.software_versions),
            (
                nominal_trial.hardware,
                nominal_trial.environment_status["start"],
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
            launches.append(current_slot)
            or _persist_journal_trial(journal, attempt, nominal_trial)
        ),
        preflight=lambda: next(preflights),
        validate_preflight=lambda hardware, status, software: None,
        recover=lambda current_slot, recovery_index, deadline, trigger: triggers.append(
            trigger
        ),
        validate_trial=lambda trial, allow_rejected_environment: None,
        classify_trial=_accept_trial,
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
    trial = _persist_journal_trial(journal, attempt, trial)

    trials = capture_baseline_trials(
        journal=journal,
        slots=(slot,),
        launch_trial=lambda current_slot, current_attempt: pytest.fail(
            "resume launched a replacement for a complete in-flight trial"
        ),
        preflight=lambda: pytest.fail("accepted in-flight slot reached preflight"),
        validate_preflight=lambda hardware, status, software: None,
        recover=lambda current_slot, recovery_index, deadline, trigger: None,
        validate_trial=lambda current_trial, allow_rejected_environment: None,
        classify_trial=_accept_trial,
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
    accepted = _persist_journal_trial(journal, attempt, accepted)
    journal.accept_inflight(attempt, accepted)
    fair_trial = _with_thermal_state(
        _valid_raw_trial(workload, pair_index=1), "fair", 1
    )

    def timeout_recovery(current_slot, recovery_index, deadline, trigger):
        raise ThermalRecoveryTimeout(ThermalRecoveryResult(7_200.0, 241))

    with pytest.raises(ThermalRecoveryTimeout):
        capture_baseline_trials(
            journal=journal,
            slots=(first, second),
            launch_trial=lambda current_slot, current_attempt: _persist_journal_trial(
                journal, current_attempt, fair_trial
            ),
            preflight=lambda: _preflight_from_trial(accepted),
            validate_preflight=lambda hardware, status, software: None,
            recover=timeout_recovery,
            validate_trial=lambda current_trial, allow_rejected_environment: None,
            classify_trial=lambda trial: classify_trial_environment(workload, trial),
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
        fair = _persist_journal_trial(journal, attempt, fair)
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
        launch_trial=lambda current_slot, attempt: (
            events.append("launch") or _persist_journal_trial(journal, attempt, nominal)
        ),
        preflight=lambda: events.append("preflight") or _preflight_from_trial(nominal),
        validate_preflight=lambda hardware, status, software: None,
        recover=lambda current_slot, recovery_index, deadline, trigger: (
            events.append("recover"),
            recovered.append((recovery_index, deadline, trigger)),
        ),
        validate_trial=lambda trial, allow_rejected_environment: None,
        classify_trial=lambda trial: classify_trial_environment(workload, trial),
        clock=lambda: clock.now,
        utc_now=lambda: "2026-08-05T00:01:00+00:00",
        progress=progress,
    )

    assert events == ["recover", "preflight", "launch"]
    assert recovered == [(expected_recovery_index, 7_200.0, expected_trigger)]


def test_capture_replays_post_exit_only_thermal_rejection_before_preflight(tmp_path):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    nominal = _valid_raw_trial(workload)
    post_exit_fair = dict(nominal.environment_status["post_exit"])
    post_exit_fair.update(thermal_state="fair", thermal_state_raw_value=1)
    rejected = _with_environment_observations(nominal, post_exit=post_exit_fair)
    attempt = journal.next_attempt(slot)
    _measurement, _post_exit, _samples, _recovery, rejected = _persist_journal_evidence(
        journal, attempt, rejected
    )
    journal.reject_inflight(attempt, rejected, reason="non-nominal-thermal")
    persisted_document = read_json_object(
        journal.rejected_path(slot, 0), label="rejected trial"
    )
    recovered = []

    def stop_after_recovery(current_slot, recovery_index, deadline, trigger):
        recovered.append((current_slot, recovery_index, deadline, trigger))
        raise RuntimeError("stop after persisted recovery")

    with pytest.raises(RuntimeError, match="stop after persisted recovery"):
        capture_baseline_trials(
            journal=journal,
            slots=(slot,),
            launch_trial=lambda current_slot, current_attempt: pytest.fail(
                "trial launched before persisted recovery"
            ),
            preflight=lambda: pytest.fail("preflight ran before persisted recovery"),
            validate_preflight=lambda hardware, status, software: None,
            recover=stop_after_recovery,
            validate_trial=lambda trial, allow_rejected_environment: None,
            classify_trial=lambda trial: classify_trial_environment(workload, trial),
            clock=lambda: 0.0,
            progress=lambda message: None,
        )

    assert recovered == [
        (
            slot,
            0,
            7_200.0,
            {
                "source": "rejected-trial",
                "rejected_trial_identity": persisted_document["identity"],
            },
        )
    ]


def _session_bound_preflight_validator(workload, trial):
    return lambda hardware, status, software: (
        benchmark_runner._validate_baseline_preflight(
            workload=workload,
            hardware=hardware,
            status=status,
            software_versions=software,
            expected_hardware=trial.hardware,
            expected_software_versions=trial.software_versions,
        )
    )


def test_capture_rejects_an_invalid_persisted_preflight_before_any_action(tmp_path):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    trial = _valid_raw_trial(workload)
    journal.record_preflight(
        slot,
        0,
        {
            "observed_at_utc": "2026-08-05T00:00:00+00:00",
            "hardware": trial.hardware,
            "environment_status": {
                **trial.environment_status,
                "power_connected": False,
            },
            "software_versions": trial.software_versions,
        },
    )

    with pytest.raises(ValueError, match="power_connected"):
        capture_baseline_trials(
            journal=journal,
            slots=(slot,),
            launch_trial=lambda current_slot, attempt: pytest.fail(
                "launch ran before persisted preflight validation"
            ),
            preflight=lambda: pytest.fail(
                "new preflight ran before persisted preflight validation"
            ),
            validate_preflight=_session_bound_preflight_validator(workload, trial),
            recover=lambda current_slot, recovery_index, deadline, trigger: pytest.fail(
                "recovery ran before persisted preflight validation"
            ),
            validate_trial=lambda current_trial, allow_rejected_environment: None,
            classify_trial=_accept_trial,
            progress=lambda message: None,
        )


def test_capture_rejects_every_persisted_recovery_sample_before_any_action(tmp_path):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    nominal = _valid_raw_trial(workload)
    fair = _with_thermal_state(nominal, "fair", 1)
    preflight = journal.record_preflight(
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
        slot, 0, {"source": "preflight", "preflight": preflight}
    )
    journal.record_thermal_sample(
        slot,
        0,
        0,
        {
            "schema_version": 1,
            "observed_at_utc": "2026-08-05T00:00:01+00:00",
            "elapsed_seconds": 300.0,
            "hardware": nominal.hardware,
            "environment_status": nominal.environment_status,
            "software_versions": {
                **nominal.software_versions,
                "python": "3.12.12",
            },
        },
    )
    journal.record_recovery_summary(
        slot,
        0,
        {"outcome": "nominal-window", "duration_seconds": 300.0, "sample_count": 1},
    )

    with pytest.raises(ValueError, match="software versions"):
        capture_baseline_trials(
            journal=journal,
            slots=(slot,),
            launch_trial=lambda current_slot, attempt: pytest.fail(
                "launch ran before persisted recovery validation"
            ),
            preflight=lambda: pytest.fail(
                "new preflight ran before persisted recovery validation"
            ),
            validate_preflight=_session_bound_preflight_validator(workload, nominal),
            recover=lambda current_slot, recovery_index, deadline, trigger: pytest.fail(
                "recovery ran before persisted recovery validation"
            ),
            validate_trial=lambda current_trial, allow_rejected_environment: None,
            classify_trial=_accept_trial,
            progress=lambda message: None,
        )


def test_capture_anchors_persisted_deadline_before_later_history_reads():
    workload = build_canonical_workload()
    slot = BaselineSlot("prepared-data", 0)
    fair = _with_thermal_state(_valid_raw_trial(workload), "fair", 1)
    preflight_document = {
        "identity": "sha256:" + "a" * 64,
        "hardware": fair.hardware,
        "environment_status": fair.environment_status,
        "software_versions": fair.software_versions,
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

        def load_pending_attempts(self, slots):
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
            validate_trial=lambda trial, allow_rejected_environment: None,
            classify_trial=_accept_trial,
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
        "hardware": fair.hardware,
        "environment_status": fair.environment_status,
        "software_versions": fair.software_versions,
    }
    second = {
        "identity": "sha256:" + "b" * 64,
        "hardware": fair.hardware,
        "environment_status": fair.environment_status,
        "software_versions": fair.software_versions,
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
        MultiplePreflightJournal(),
        (slot,),
        advancing_clock,
        lambda hardware, status, software: None,
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
        launch_trial=lambda current_slot, attempt: _persist_journal_trial(
            journal, attempt, nominal
        ),
        preflight=lambda: _preflight_from_trial(nominal),
        validate_preflight=lambda hardware, status, software: None,
        recover=lambda current_slot, recovery_index, deadline, trigger: pytest.fail(
            "completed nominal recovery was replayed"
        ),
        validate_trial=lambda trial, allow_rejected_environment: None,
        classify_trial=_accept_trial,
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
            return _persist_journal_trial(journal, attempt, nominal)

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
                trial = fair
            else:
                trial = nominal
            return _persist_journal_trial(journal, attempt, trial)

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
        validate_trial=lambda trial, allow_rejected_environment: None,
        classify_trial=lambda trial: classify_trial_environment(workload, trial),
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
    persisted_trials = []
    for trial in trials:
        slot = BaselineSlot(trial.metric, trial.pair_index)
        attempt = journal.next_attempt(slot)
        _measurement, _post_exit, _samples, _recovery, persisted = (
            _persist_journal_evidence(journal, attempt, trial)
        )
        journal.accept_inflight(attempt, persisted)
        persisted_trials.append(persisted)
    return workload, paired, journal, tuple(persisted_trials)


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
            paired_representations=paired,
            manifest_path=tmp_path / "baseline.json",
            raw_output_path=raw_path,
        )

    assert raw_path.read_text(encoding="utf-8") == "different existing content\n"
    assert not journal.completed_path.exists()


def test_interrupted_manifest_publication_resumes_byte_identically_across_cli_spellings(
    tmp_path, monkeypatch
):
    workload, paired, journal, trials = _accepted_complete_journal(tmp_path)
    manifest_path = tmp_path / "baseline.json"
    raw_path = tmp_path / "baseline.jsonl"
    original_publish_completed = BaselineJournal.publish_completed

    monkeypatch.setattr(sys, "executable", "python-relative")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "v2/benchmarks/runner.py",
            "record-baseline",
            "--manifest",
            "baseline.json",
            "--raw-output",
            "baseline.jsonl",
        ],
    )
    first_command = benchmark_runner.canonical_baseline_command(journal)
    command_parts = shlex.split(first_command)
    assert command_parts[command_parts.index("--pairs") + 1] == "5"
    assert command_parts[command_parts.index("--warmup") + 1] == "5"
    assert command_parts[command_parts.index("--measure") + 1] == "20"
    relocated = BaselineJournal(tmp_path / "equivalent-state", journal.session)
    assert benchmark_runner.canonical_baseline_command(relocated) == first_command
    monkeypatch.setattr(
        BaselineJournal,
        "publish_completed",
        lambda self, document: (_ for _ in ()).throw(
            RuntimeError("interrupted before completion")
        ),
    )
    with pytest.raises(RuntimeError, match="interrupted before completion"):
        publish_baseline_from_journal(
            journal=journal,
            trials=trials,
            workload=workload,
            workload_identity=canonical_workload_identity(workload),
            source_commit=trials[0].source_commit,
            harness_commit=trials[0].harness_commit,
            harness_identity=trials[0].harness_identity,
            paired_representations=paired,
            manifest_path=manifest_path,
            raw_output_path=raw_path,
        )
    first_manifest_bytes = manifest_path.read_bytes()

    monkeypatch.setattr(sys, "executable", "/absolute/venv/bin/python3")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "/absolute/checkout/v2/benchmarks/runner.py",
            "record-baseline",
            "--raw-output",
            str(raw_path),
            "--manifest",
            str(manifest_path),
        ],
    )
    monkeypatch.setattr(
        BaselineJournal, "publish_completed", original_publish_completed
    )
    resumed = BaselineJournal.open(journal.root, journal.session)
    second_command = benchmark_runner.canonical_baseline_command(resumed)
    manifest = publish_baseline_from_journal(
        journal=resumed,
        trials=trials,
        workload=workload,
        workload_identity=canonical_workload_identity(workload),
        source_commit=trials[0].source_commit,
        harness_commit=trials[0].harness_commit,
        harness_identity=trials[0].harness_identity,
        paired_representations=paired,
        manifest_path=manifest_path,
        raw_output_path=raw_path,
    )

    assert second_command == first_command
    assert manifest["command"] == first_command
    assert manifest_path.read_bytes() == first_manifest_bytes
    assert resumed.completed_path.is_file()


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


def test_managed_worktree_preserves_capture_error_when_final_cleanup_fails(
    tmp_path, monkeypatch
):
    repository = tmp_path / "repository"
    repository.mkdir()
    destination = tmp_path / "source"
    destination.mkdir()
    (destination / "tracked.py").write_text("content\n", encoding="utf-8")
    original_error = RuntimeError("outer capture failure")
    commands = []

    monkeypatch.setattr(
        benchmark_runner,
        "_create_detached_worktree",
        lambda current_repository, commit, current_destination: None,
    )
    monkeypatch.setattr(
        benchmark_runner,
        "_remove_worktree",
        lambda current_repository, current_destination: (_ for _ in ()).throw(
            RuntimeError("final cleanup failure")
        ),
    )
    monkeypatch.setattr(
        benchmark_runner.subprocess,
        "run",
        lambda command, *, cwd, check: commands.append(command) or SimpleNamespace(),
    )

    with (
        pytest.raises(RuntimeError, match="outer capture failure") as caught,
        benchmark_runner._managed_detached_worktree(repository, "a" * 40, destination),
    ):
        raise original_error

    assert caught.value is original_error
    assert not destination.exists()
    assert ("git", "worktree", "prune") in commands
    assert any("final cleanup failure" in note for note in caught.value.__notes__)


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


def test_baseline_journal_rejects_old_journal_session_without_post_exit_policy(
    tmp_path,
):
    state = tmp_path / "state"
    current = _session_document(tmp_path)
    old = json.loads(json.dumps(current))
    required = old["canonical_workload"]["required_environment"]
    for key in (
        "measurement_end_memory_pressure_allowed",
        "post_exit_memory_pressure_allowed",
        "post_exit_memory_pressure",
        "post_exit_recovery_required_for_warning",
        "post_exit_recovery_sample_interval_seconds",
        "post_exit_recovery_timeout_seconds",
        "post_exit_recovery_stability_seconds",
        "post_exit_recovery_evidence_required",
    ):
        required.pop(key)
    old["canonical_workload_identity"] = structured_identity(
        "sml-canonical-benchmark-workload-v1", old["canonical_workload"]
    )
    old_body = {key: value for key, value in old.items() if key != "identity"}
    old["identity"] = structured_identity("sml-baseline-journal-session-v1", old_body)
    BaselineJournal.open(state, old)

    with pytest.raises(ValueError, match="session does not match expected session"):
        BaselineJournal.open(state, current)


@pytest.mark.parametrize(
    "changed",
    [
        {"protocol": {"pairs": 4, "warmup_units": 5, "measured_units": 20}},
        {"protocol": {"pairs": 5, "warmup_units": 20, "measured_units": 100}},
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


def _orphan_atomic_temporary(destination, token="a" * 32):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / (f".{destination.name}.sml-atomic-{token}.tmp")
    temporary.write_text("orphan\n", encoding="utf-8")
    return temporary


def test_session_lock_is_same_thread_reentrant_and_excludes_a_concurrent_invocation(
    tmp_path,
):
    state = tmp_path / "state"
    outcomes = []

    with baseline_journal.baseline_session_lock(state):
        with baseline_journal.baseline_session_lock(state):
            pass

        def contend():
            try:
                with baseline_journal.baseline_session_lock(state):
                    outcomes.append("entered")
            except RuntimeError as error:
                outcomes.append(str(error))

        thread = threading.Thread(target=contend)
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive()

    assert outcomes == ["baseline session is already locked"]
    assert (state / ".baseline-session.lock").is_file()


def test_session_lock_is_released_when_an_owner_process_crashes(tmp_path):
    state = tmp_path / "state"
    script = (
        "import os, pathlib; "
        "from v2.benchmarks.journal import baseline_session_lock; "
        f"state = pathlib.Path({str(state)!r}); "
        "lock = baseline_session_lock(state); lock.__enter__(); os._exit(0)"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[3],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    with baseline_journal.baseline_session_lock(state):
        pass


def test_output_lock_excludes_a_process_with_a_different_temporary_directory(
    tmp_path,
):
    manifest = tmp_path / "outputs" / "baseline.json"
    raw_output = tmp_path / "outputs" / "baseline.jsonl"
    alternate_temporary_directory = tmp_path / "alternate-tmp"
    alternate_temporary_directory.mkdir()
    script = f"""
from pathlib import Path

from v2.benchmarks.journal import baseline_output_lock

manifest = Path({str(manifest)!r})
raw_output = Path({str(raw_output)!r})
try:
    with baseline_output_lock(manifest, raw_output):
        raise SystemExit("entered")
except RuntimeError as error:
    print(error)
"""

    with baseline_journal.baseline_output_lock(manifest, raw_output):
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[3],
            env={**os.environ, "TMPDIR": str(alternate_temporary_directory)},
            check=False,
            capture_output=True,
            text=True,
        )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "baseline final outputs are already locked"


@pytest.mark.parametrize(
    ("first_name", "alias_name"),
    [
        ("Case/Manifest.json", "case/manifest.JSON"),
        (
            "caf\N{LATIN SMALL LETTER E WITH ACUTE}.json",
            "cafe\N{COMBINING ACUTE ACCENT}.json",
        ),
    ],
)
def test_output_lock_conservatively_serializes_case_and_unicode_aliases(
    tmp_path, first_name, alias_name
):
    outcomes = []
    first_manifest = tmp_path / first_name
    alias_manifest = tmp_path / alias_name

    with baseline_journal.baseline_output_lock(
        first_manifest, tmp_path / "first.jsonl"
    ):

        def contend():
            try:
                with baseline_journal.baseline_output_lock(
                    alias_manifest, tmp_path / "second.jsonl"
                ):
                    outcomes.append("entered")
            except RuntimeError as error:
                outcomes.append(str(error))

        thread = threading.Thread(target=contend)
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive()

    assert outcomes == ["baseline final outputs are already locked"]


def test_output_lock_root_is_private_and_rejects_unsafe_permissions(
    tmp_path, monkeypatch
):
    secure_root = tmp_path / "secure-locks"
    monkeypatch.setattr(
        baseline_journal, "_baseline_output_lock_root", lambda: secure_root
    )

    with baseline_journal.baseline_output_lock(
        tmp_path / "manifest.json", tmp_path / "raw.jsonl"
    ):
        pass

    assert stat.S_IMODE(secure_root.stat().st_mode) == 0o700

    unsafe_root = tmp_path / "unsafe-locks"
    unsafe_root.mkdir(mode=0o700)
    unsafe_root.chmod(0o777)
    monkeypatch.setattr(
        baseline_journal, "_baseline_output_lock_root", lambda: unsafe_root
    )

    with (
        pytest.raises(ValueError, match="lock directory permissions"),
        baseline_journal.baseline_output_lock(
            tmp_path / "other-manifest.json", tmp_path / "other-raw.jsonl"
        ),
    ):
        pass


def test_durable_directory_creation_tolerates_a_concurrent_creator(
    tmp_path, monkeypatch
):
    target = tmp_path / "parent" / "child"
    original_mkdir = baseline_journal.os.mkdir
    raced = False

    def mkdir_after_competitor(path, mode=0o777):
        nonlocal raced
        if Path(path) == target and not raced:
            raced = True
            original_mkdir(path, mode)
            raise FileExistsError
        return original_mkdir(path, mode)

    monkeypatch.setattr(baseline_journal.os, "mkdir", mkdir_after_competitor)

    baseline_journal._create_durable_directory(target)

    assert target.is_dir()


def test_journal_never_treats_a_symlink_as_the_persistent_lock_inode(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    outside = tmp_path / "outside-lock"
    outside.write_text("outside\n", encoding="utf-8")
    (state / ".baseline-session.lock").symlink_to(outside)

    with pytest.raises(ValueError, match="lock"):
        BaselineJournal.open(state, _session_document(tmp_path))

    assert outside.read_text(encoding="utf-8") == "outside\n"


def test_locked_resume_removes_every_recognized_journal_orphan_and_fsyncs_parents(
    tmp_path, monkeypatch
):
    state = tmp_path / "state"
    expected_session = _session_document(tmp_path)
    BaselineJournal.open(state, expected_session)
    destinations = (
        state / ".baseline-session-initializing",
        state / "session.json",
        state / "completed.json",
        state / "accepted" / "prepared-data" / "0.json",
        state / "measurements" / "prepared-data" / "0" / "0.json",
        state / "post-exit" / "prepared-data" / "0" / "0.json",
        state / "recovery-samples" / "prepared-data" / "0" / "0" / "0.json",
        state / "recovery" / "prepared-data" / "0" / "0.json",
        state / "inflight" / "prepared-data" / "0" / "0.json",
        state / "rejected" / "prepared-data" / "0" / "0.json",
        state / "preflight" / "prepared-data" / "0" / "0.json",
        state / "thermal-waits" / "prepared-data" / "0" / "0" / "trigger.json",
        state / "thermal-waits" / "prepared-data" / "0" / "0" / "0.json",
        state / "thermal-waits" / "prepared-data" / "0" / "0" / "summary.json",
    )
    temporaries = tuple(_orphan_atomic_temporary(path) for path in destinations)
    marker = state / ".baseline-session-initializing"
    marker.write_text("", encoding="utf-8")
    synced = []
    original_fsync = baseline_journal._fsync_directory

    with baseline_journal.baseline_session_lock(state):
        monkeypatch.setattr(
            baseline_journal,
            "_fsync_directory",
            lambda path: synced.append(path.resolve()),
        )
        baseline_journal.cleanup_orphaned_journal_temporaries(state)
        monkeypatch.setattr(baseline_journal, "_fsync_directory", original_fsync)

    assert not marker.exists()
    assert not any(path.exists() for path in temporaries)
    assert {path.parent.resolve() for path in temporaries} <= set(synced)
    resumed = BaselineJournal.open(state, expected_session)
    assert (
        resumed.next_attempt(BaselineSlot("prepared-data", 0)).journal_attempt_index
        == 0
    )


def test_locked_resume_recovers_an_interrupted_root_initialization(tmp_path):
    state = tmp_path / "state"
    marker = state / ".baseline-session-initializing"
    marker_temp = _orphan_atomic_temporary(marker)
    session_temp = _orphan_atomic_temporary(state / "session.json", "b" * 32)
    marker.write_text("", encoding="utf-8")

    with baseline_journal.baseline_session_lock(state):
        baseline_journal.cleanup_orphaned_journal_temporaries(state)

    assert not marker.exists()
    assert not marker_temp.exists()
    assert not session_temp.exists()
    BaselineJournal.open(state, _session_document(tmp_path))


def test_orphan_cleanup_preserves_hidden_wrong_pattern_and_symlink_nodes(tmp_path):
    state = tmp_path / "state"
    arbitrary = state / ".arbitrary-hidden"
    wrong_pattern = state / ".session.json.sml-atomic-not-a-token.tmp"
    wrong_destination = state / f".foreign.json.sml-atomic-{'b' * 32}.tmp"
    arbitrary.parent.mkdir(parents=True)
    arbitrary.write_text("keep\n", encoding="utf-8")
    wrong_pattern.write_text("keep\n", encoding="utf-8")
    wrong_destination.write_text("keep\n", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.write_text("outside\n", encoding="utf-8")
    symlink = state / f".session.json.sml-atomic-{'c' * 32}.tmp"
    symlink.symlink_to(outside)

    with baseline_journal.baseline_session_lock(state):
        baseline_journal.cleanup_orphaned_journal_temporaries(state)

    assert arbitrary.read_text(encoding="utf-8") == "keep\n"
    assert wrong_pattern.read_text(encoding="utf-8") == "keep\n"
    assert wrong_destination.read_text(encoding="utf-8") == "keep\n"
    assert symlink.is_symlink()
    assert outside.read_text(encoding="utf-8") == "outside\n"
    with pytest.raises(ValueError, match="unexpected content|non-empty"):
        BaselineJournal.open(state, _session_document(tmp_path))


def test_a_live_atomic_temporary_cannot_be_cleaned_by_a_concurrent_invocation(
    tmp_path,
):
    state = tmp_path / "state"
    destination = state / "session.json"
    outcomes = []

    with baseline_journal.baseline_session_lock(state):
        live = _orphan_atomic_temporary(destination)

        def contend():
            try:
                with baseline_journal.baseline_session_lock(state):
                    baseline_journal.cleanup_orphaned_journal_temporaries(state)
            except RuntimeError as error:
                outcomes.append(str(error))

        thread = threading.Thread(target=contend)
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert live.is_file()

    assert outcomes == ["baseline session is already locked"]
    with baseline_journal.baseline_session_lock(state):
        baseline_journal.cleanup_orphaned_journal_temporaries(state)
    assert not live.exists()


def test_locked_final_output_cleanup_removes_only_exact_regular_temporaries(tmp_path):
    state = tmp_path / "state"
    manifest = tmp_path / "checkout" / "baseline.json"
    raw = tmp_path / "results" / "baseline.jsonl"
    manifest_temp = _orphan_atomic_temporary(manifest, "d" * 32)
    raw_temp = _orphan_atomic_temporary(raw, "e" * 32)
    wrong = manifest.parent / f".other.json.sml-atomic-{'f' * 32}.tmp"
    wrong.write_text("keep\n", encoding="utf-8")
    outside = tmp_path / "outside-final"
    outside.write_text("outside\n", encoding="utf-8")
    linked = raw.parent / f".{raw.name}.sml-atomic-{'0' * 32}.tmp"
    linked.symlink_to(outside)

    with baseline_journal.baseline_session_lock(state):
        baseline_journal.cleanup_orphaned_atomic_temporaries((manifest, raw))

    assert not manifest_temp.exists()
    assert not raw_temp.exists()
    assert wrong.read_text(encoding="utf-8") == "keep\n"
    assert linked.is_symlink()
    assert outside.read_text(encoding="utf-8") == "outside\n"


def test_record_baseline_holds_the_session_lock_across_cleanup_and_capture_entry(
    tmp_path, monkeypatch
):
    harness = tmp_path / "harness"
    harness.mkdir()
    args = SimpleNamespace(
        state_directory=tmp_path / "state",
        manifest=tmp_path / "outputs" / "baseline.json",
        raw_output=tmp_path / "outputs" / "baseline.jsonl",
    )
    manifest_temp = _orphan_atomic_temporary(args.manifest)
    raw_temp = _orphan_atomic_temporary(args.raw_output, "b" * 32)
    entered = threading.Event()
    release = threading.Event()
    calls = []
    failures = []

    monkeypatch.setattr(benchmark_runner, "_git_root", lambda path: harness)

    def capture(current_args, **paths):
        calls.append(paths)
        assert not manifest_temp.exists()
        assert not raw_temp.exists()
        entered.set()
        assert release.wait(timeout=5)
        return 0

    monkeypatch.setattr(benchmark_runner, "_record_baseline_locked", capture)

    def first_invocation():
        try:
            benchmark_runner._record_baseline(args)
        except Exception as error:  # noqa: BLE001 - surface thread failures to test
            failures.append(error)

    thread = threading.Thread(target=first_invocation)
    thread.start()
    assert entered.wait(timeout=5)
    with pytest.raises(RuntimeError, match="already locked"):
        benchmark_runner._record_baseline(args)
    release.set()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert failures == []
    assert len(calls) == 1


@pytest.mark.parametrize("second_raw_name", ["baseline.jsonl", "other.jsonl"])
def test_different_state_roots_cannot_clean_a_live_shared_output_temporary(
    tmp_path, monkeypatch, second_raw_name
):
    harness = tmp_path / "harness"
    harness.mkdir()
    manifest = tmp_path / "outputs" / "baseline.json"
    raw_output = tmp_path / "outputs" / "baseline.jsonl"
    first_args = SimpleNamespace(
        state_directory=tmp_path / "first-state",
        manifest=manifest,
        raw_output=raw_output,
    )
    second_args = SimpleNamespace(
        state_directory=tmp_path / "second-state",
        manifest=manifest,
        raw_output=tmp_path / "outputs" / second_raw_name,
    )
    entered = threading.Event()
    release = threading.Event()
    failures = []
    live_temporary = []

    monkeypatch.setattr(benchmark_runner, "_git_root", lambda path: harness)

    def capture(current_args, **paths):
        if paths["state_root"] == first_args.state_directory.resolve():
            live_temporary.append(_orphan_atomic_temporary(manifest))
            entered.set()
            assert release.wait(timeout=5)
        return 0

    monkeypatch.setattr(benchmark_runner, "_record_baseline_locked", capture)

    def first_invocation():
        try:
            benchmark_runner._record_baseline(first_args)
        except Exception as error:  # noqa: BLE001 - surface thread failures to test
            failures.append(error)

    thread = threading.Thread(target=first_invocation)
    thread.start()
    assert entered.wait(timeout=5)
    try:
        with pytest.raises(RuntimeError, match="final outputs are already locked"):
            benchmark_runner._record_baseline(second_args)
        assert live_temporary[0].is_file()
    finally:
        release.set()
        thread.join(timeout=5)

    assert not thread.is_alive()
    assert failures == []


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


def _single_process_arguments(tmp_path):
    return SimpleNamespace(
        harness_root=tmp_path / "harness",
        source_root=tmp_path / "source",
        source_commit="3687f8b3214a44c675ae67af52e4997762f6c634",
        harness_commit="a" * 40,
        harness_identity="sha256:" + "b" * 64,
        adapter="legacy",
        metric="prepared-data",
        side="reference",
        attempt_index=0,
        pair_index=0,
        process_order=0,
        warmup=5,
        measure=100,
        prepared_data_measure=100,
        comparison_target="baseline",
        evidence_session_identity="sha256:" + "9" * 64,
        journal_attempt_index=0,
        measurement_output=(
            tmp_path / "state" / "measurements" / "prepared-data" / "0" / "0.json"
        ),
    )


def _launch_trial_arguments(tmp_path):
    return {
        "harness_root": tmp_path / "harness",
        "source_root": tmp_path / "source",
        "source_commit": "3687f8b3214a44c675ae67af52e4997762f6c634",
        "harness_commit": "a" * 40,
        "harness_identity": "sha256:" + "b" * 64,
        "adapter": "legacy",
        "metric": "prepared-data",
        "side": "reference",
        "attempt_index": 0,
        "pair_index": 0,
        "order": 0,
        "warmup": 5,
        "measure": 100,
        "prepared_data_measure": 100,
        "comparison_target": "baseline",
    }


def _stub_single_process_measurement(monkeypatch, args, captured=None):
    args.harness_root.mkdir()
    args.source_root.mkdir()
    workload = build_canonical_workload()
    trial = _valid_raw_trial(workload, metric=args.metric)
    status = {
        "power_connected": True,
        "power_mode": "automatic",
        "low_power_mode": False,
        "thermal_state": "nominal",
        "thermal_state_raw_value": 0,
        "memory_pressure": "normal",
        "memory_free_percentage": 60,
        "competing_gpu_workload": False,
    }
    native = SimpleNamespace(
        native_configuration=trial.native_configuration,
        native_representation_identity=trial.native_representation_identity,
        canonical_row_identity=trial.canonical_row_identity,
        canonical_input_identity=trial.canonical_input_identity,
        canonical_projection=trial.canonical_projection,
        execution_order_identity=trial.execution_order_identity,
        initial_parameter_identity=trial.initial_parameter_identity,
        startup_verification_seconds=trial.startup_verification_seconds,
    )
    monkeypatch.setattr(benchmark_runner, "_git_root", lambda path: path.resolve())
    monkeypatch.setattr(
        benchmark_runner, "_require_clean_checkout", lambda path, *, label: None
    )
    monkeypatch.setattr(
        benchmark_runner,
        "_git_commit",
        lambda path: (
            args.harness_commit
            if path.resolve() == args.harness_root.resolve()
            else args.source_commit
        ),
    )
    monkeypatch.setattr(
        benchmark_runner,
        "harness_content_identity",
        lambda path: args.harness_identity,
    )
    monkeypatch.setattr(legacy, "resolve_native_workload", lambda *unused: native)
    monkeypatch.setattr(
        benchmark_runner,
        "collect_environment",
        lambda: (trial.hardware, status, trial.software_versions),
    )

    def measure(**kwargs):
        if captured is not None:
            captured.update(kwargs)
        return benchmark_runner.ProcessMeasurement(
            elapsed_seconds=trial.elapsed_seconds,
            value=trial.value,
            work_count=100.0,
            compilation_seconds=trial.compilation_seconds,
            peak_memory_bytes=trial.peak_memory_bytes,
        )

    monkeypatch.setattr(benchmark_runner, "measure_native_process", measure)


@pytest.mark.parametrize(
    ("metric", "expected_warmup", "expected_measured"),
    [
        ("prepared-data", 5, 100),
        ("inference-prefill", 5, 32),
        ("compile-cold-start", 0, 1),
        ("peak-metal-memory", 5, 1),
    ],
)
def test_child_process_forwards_each_canonical_metric_count(
    tmp_path, monkeypatch, metric, expected_warmup, expected_measured
):
    args = _single_process_arguments(tmp_path)
    args.metric = metric
    args.measure = expected_measured
    args.measurement_output = (
        tmp_path / "state" / "measurements" / metric / "0" / "0.json"
    )
    captured = {}
    _stub_single_process_measurement(monkeypatch, args, captured)

    benchmark_runner._run_single_process(args)

    assert captured["warmup_units"] == expected_warmup
    assert captured["measured_units"] == expected_measured
    measurement = validate_child_trial_measurement(
        read_json_object(args.measurement_output, label="child measurement")
    )
    assert measurement["trial"]["warmup_units"] == expected_warmup
    assert measurement["trial"]["measured_units"] == expected_measured


@pytest.mark.parametrize(
    ("metric", "supplied"),
    (("prepared-data", 20), ("prepared-data", 99), ("pretraining-compute", 100)),
)
def test_child_rejects_noncanonical_count_before_adapter_import(
    tmp_path, monkeypatch, metric, supplied
):
    args = _single_process_arguments(tmp_path)
    args.metric = metric
    args.measure = supplied
    resolved = []
    _stub_single_process_measurement(monkeypatch, args)
    monkeypatch.setattr(
        legacy, "resolve_native_workload", lambda *args: resolved.append(args)
    )

    with pytest.raises(ValueError, match="child measured units do not match"):
        benchmark_runner._run_single_process(args)

    assert resolved == []


def test_completed_child_output_never_overwrites_an_existing_attempt(
    tmp_path, monkeypatch
):
    args = _single_process_arguments(tmp_path)
    _stub_single_process_measurement(monkeypatch, args)
    args.measurement_output.parent.mkdir(parents=True)
    args.measurement_output.write_bytes(b"existing immutable measurement\n")

    with pytest.raises(FileExistsError):
        benchmark_runner._run_single_process(args)

    assert args.measurement_output.read_bytes() == b"existing immutable measurement\n"


def test_completed_child_output_crosses_the_parent_fsync_boundary(
    tmp_path, monkeypatch
):
    args = _single_process_arguments(tmp_path)
    _stub_single_process_measurement(monkeypatch, args)
    synced_directories = []
    monkeypatch.setattr(
        baseline_journal,
        "_fsync_directory",
        lambda path: synced_directories.append(path.resolve()),
    )

    benchmark_runner._run_single_process(args)

    assert args.measurement_output.parent.resolve() in synced_directories
    assert (
        validate_child_trial_measurement(
            read_json_object(
                args.measurement_output, label="completed child measurement"
            )
        )["trial"]["attempt_index"]
        == 0
    )


def _observation_with_memory(pressure, free_percentage):
    observation = _valid_observation("2026-08-09T00:00:03+00:00")
    observation["environment_status"].update(
        memory_pressure=pressure,
        memory_free_percentage=free_percentage,
    )
    return observation


def test_collect_post_exit_environment_records_monotonic_time_before_memory_probe(
    monkeypatch,
):
    events = []
    observation = _valid_observation("2026-08-09T00:00:03+00:00")

    monkeypatch.setattr(
        benchmark_runner,
        "_memory_pressure",
        lambda: events.append("memory") or ("normal", 67),
    )
    monkeypatch.setattr(
        benchmark_runner,
        "_utc_now_iso",
        lambda: events.append("timestamp") or observation["observed_at_utc"],
    )
    monkeypatch.setattr(
        benchmark_runner,
        "collect_environment",
        lambda *, memory_sample: (
            events.append(("environment", memory_sample))
            or (
                observation["hardware"],
                observation["environment_status"],
                observation["software_versions"],
            )
        ),
    )

    started_at, collected = benchmark_runner.collect_post_exit_environment(
        clock=lambda: events.append("clock") or 42.0,
        deadline=42.0,
    )

    assert started_at == 42.0
    assert events == ["clock", "memory", "timestamp", ("environment", ("normal", 67))]
    assert collected == observation


def test_recovery_failure_fields_ignores_memory_and_reports_identity_and_environment_drift():
    workload = build_canonical_workload()
    observation = _observation_with_memory("warning", 1)
    observation["hardware"] = {**observation["hardware"], "chip": "Other"}
    observation["software_versions"] = {
        **observation["software_versions"],
        "python": "3.12.12",
    }
    observation["environment_status"]["power_connected"] = False
    observation["environment_status"]["thermal_state"] = "fair"
    observation["environment_status"]["thermal_state_raw_value"] = 1

    assert benchmark_runner._recovery_failure_fields(
        observation,
        expected_hardware=_valid_observation("2026-08-09T00:00:00+00:00")["hardware"],
        expected_software_versions=_valid_observation("2026-08-09T00:00:00+00:00")[
            "software_versions"
        ],
        required_environment=workload.required_environment,
    ) == ("hardware", "power_connected", "software_versions", "thermal_state")


def test_recovery_failure_fields_rejects_malformed_thermal_evidence():
    observation = _valid_observation("2026-08-09T00:00:03+00:00")
    observation["environment_status"]["thermal_state"] = "fair"

    with pytest.raises(ValueError, match="thermal state and raw value disagree"):
        benchmark_runner._recovery_failure_fields(
            observation,
            expected_hardware=observation["hardware"],
            expected_software_versions=observation["software_versions"],
            required_environment=build_canonical_workload().required_environment,
        )


def test_parent_records_immediate_warning_then_every_recovery_sample_before_finalizing(
    tmp_path, monkeypatch
):
    workload = build_canonical_workload()
    measurement = _valid_child_measurement(workload)
    measurement_path = tmp_path / "measurement.json"
    post_exit_path = tmp_path / "post-exit.json"
    recovery_samples_directory = tmp_path / "recovery-samples"
    recovery_path = tmp_path / "recovery.json"
    trial_path = tmp_path / "trial.json"
    clock = _RecoveryClock()
    events = []
    observations = [
        _observation_with_memory("warning", 36),
        *[_observation_with_memory("normal", 67) for _ in range(7)],
    ]

    def run_child(command, *, cwd, check):
        events.append("child-exited")
        atomic_write_json(measurement_path, measurement, create_only=True)

    monkeypatch.setattr(benchmark_runner.subprocess, "run", run_child)
    monkeypatch.setattr(
        benchmark_runner,
        "collect_post_exit_environment",
        lambda **kwargs: (
            events.append("memory-first") or clock(),
            observations.pop(0),
        ),
    )
    original_atomic_write_json = benchmark_runner.atomic_write_json

    def record_atomic_write(path, value, *, create_only):
        events.append(f"write:{path.name}")
        return original_atomic_write_json(path, value, create_only=create_only)

    monkeypatch.setattr(benchmark_runner, "atomic_write_json", record_atomic_write)

    trial = benchmark_runner._launch_trial(
        **_launch_trial_arguments(tmp_path),
        evidence_session_identity=measurement["session_identity"],
        journal_attempt_index=0,
        measurement_output=measurement_path,
        post_exit_output=post_exit_path,
        recovery_samples_directory=recovery_samples_directory,
        recovery_output=recovery_path,
        output=trial_path,
        clock=clock,
        sleep=clock.sleep,
    )

    assert events[:2] == ["child-exited", "memory-first"]
    assert trial.schema_version == 3
    assert (
        trial.post_exit_observation["environment_status"]["memory_pressure"]
        == "warning"
    )
    assert trial.post_exit_recovery["outcome"] == "recovered"
    assert len(trial.post_exit_recovery_samples) == 7
    assert sorted(path.name for path in recovery_samples_directory.iterdir()) == [
        f"{index}.json" for index in range(7)
    ]
    assert recovery_path.is_file()
    assert events.index("write:post-exit.json") < events.index("write:0.json")
    assert events.index("write:6.json") < events.index("write:recovery.json")
    assert events.index("write:recovery.json") < events.index("write:trial.json")
    assert trial == RawTrial.from_dict(read_json_object(trial_path, label="trial"))


def test_parent_finalizes_and_rejects_timeout_when_deadline_sleep_overshoots(
    tmp_path, monkeypatch
):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    measurement = _valid_child_measurement(
        workload,
        session_identity=journal.session["identity"],
        journal_attempt_index=0,
    )
    immediate = _observation_with_memory("warning", 36)

    class OversleepingDeadlineClock(_RecoveryClock):
        def sleep(self, seconds):
            super().sleep(seconds)
            if self.now == 300.0:
                self.now += 0.001

    clock = OversleepingDeadlineClock()
    collection_starts = []

    monkeypatch.setattr(
        benchmark_runner.subprocess,
        "run",
        lambda command, *, cwd, check: atomic_write_json(
            journal.measurement_path(slot, 0), measurement, create_only=True
        ),
    )

    def collect_post_exit_environment(**kwargs):
        collection_starts.append(clock())
        return clock(), immediate

    monkeypatch.setattr(
        benchmark_runner,
        "collect_post_exit_environment",
        collect_post_exit_environment,
    )

    def launch(current_slot, attempt):
        assert current_slot == slot
        return benchmark_runner._launch_trial(
            **_launch_trial_arguments(tmp_path),
            evidence_session_identity=journal.session["identity"],
            journal_attempt_index=attempt.journal_attempt_index,
            measurement_output=journal.measurement_path(slot, 0),
            post_exit_output=journal.post_exit_path(slot, 0),
            recovery_samples_directory=journal.recovery_samples_path(slot, 0),
            recovery_output=journal.recovery_path(slot, 0),
            output=attempt.path,
            clock=clock,
            sleep=clock.sleep,
        )

    with pytest.raises(
        benchmark_runner.MemoryPressureTrialRejected,
        match="persistent-post-exit-memory-pressure",
    ):
        capture_baseline_trials(
            journal=journal,
            slots=(slot,),
            launch_trial=launch,
            preflight=lambda: _preflight_from_trial(_valid_raw_trial(workload)),
            validate_preflight=lambda hardware, status, software: None,
            recover=lambda *args: pytest.fail("memory rejection entered recovery"),
            validate_trial=_baseline_validator(workload),
            classify_trial=lambda trial: classify_trial_environment(workload, trial),
            progress=lambda message: None,
            clock=clock,
        )

    recovery = read_json_object(journal.recovery_path(slot, 0), label="recovery")
    rejected = read_json_object(journal.rejected_path(slot, 0), label="rejection")
    assert recovery["outcome"] == "timeout"
    assert recovery["duration_seconds"] == 300.0
    assert recovery["terminal_environment_status"] == immediate["environment_status"]
    assert collection_starts == [float(value) for value in range(0, 300, 5)]
    assert rejected["reason"] == "persistent-post-exit-memory-pressure"
    assert rejected["trial"]["post_exit_recovery"] == recovery


def test_parent_aborts_recovery_collection_when_probe_start_crosses_deadline(
    tmp_path, monkeypatch
):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    measurement = _valid_child_measurement(
        workload,
        session_identity=journal.session["identity"],
        journal_attempt_index=0,
    )
    immediate = _observation_with_memory("warning", 36)

    class CollectorRaceClock(_RecoveryClock):
        def __init__(self):
            super().__init__()
            self.deadline_reads = 0

        def __call__(self):
            if self.now == 300.0:
                self.deadline_reads += 1
                if self.deadline_reads == 2:
                    self.now += 0.001
            return self.now

        def sleep(self, seconds):
            self.sleeps.append(seconds)
            self.now = 300.0
            self.deadline_reads = 0

    clock = CollectorRaceClock()
    memory_probe_calls = []
    environment_collection_calls = []

    monkeypatch.setattr(
        benchmark_runner.subprocess,
        "run",
        lambda command, *, cwd, check: atomic_write_json(
            journal.measurement_path(slot, 0), measurement, create_only=True
        ),
    )
    monkeypatch.setattr(
        benchmark_runner,
        "_memory_pressure",
        lambda: memory_probe_calls.append(True) or ("warning", 36),
    )
    monkeypatch.setattr(
        benchmark_runner,
        "_utc_now_iso",
        lambda: immediate["observed_at_utc"],
    )

    def collect_environment(*, memory_sample):
        environment_collection_calls.append(memory_sample)
        return (
            immediate["hardware"],
            immediate["environment_status"],
            immediate["software_versions"],
        )

    monkeypatch.setattr(benchmark_runner, "collect_environment", collect_environment)

    def launch(current_slot, attempt):
        assert current_slot == slot
        return benchmark_runner._launch_trial(
            **_launch_trial_arguments(tmp_path),
            evidence_session_identity=journal.session["identity"],
            journal_attempt_index=attempt.journal_attempt_index,
            measurement_output=journal.measurement_path(slot, 0),
            post_exit_output=journal.post_exit_path(slot, 0),
            recovery_samples_directory=journal.recovery_samples_path(slot, 0),
            recovery_output=journal.recovery_path(slot, 0),
            output=attempt.path,
            clock=clock,
            sleep=clock.sleep,
        )

    with pytest.raises(
        benchmark_runner.MemoryPressureTrialRejected,
        match="persistent-post-exit-memory-pressure",
    ):
        capture_baseline_trials(
            journal=journal,
            slots=(slot,),
            launch_trial=launch,
            preflight=lambda: _preflight_from_trial(_valid_raw_trial(workload)),
            validate_preflight=lambda hardware, status, software: None,
            recover=lambda *args: pytest.fail("memory rejection entered recovery"),
            validate_trial=_baseline_validator(workload),
            classify_trial=lambda trial: classify_trial_environment(workload, trial),
            progress=lambda message: None,
            clock=clock,
        )

    recovery = read_json_object(journal.recovery_path(slot, 0), label="recovery")
    rejected = read_json_object(journal.rejected_path(slot, 0), label="rejection")
    assert memory_probe_calls == [True]
    assert environment_collection_calls == [("warning", 36)]
    assert not journal.recovery_samples_path(slot, 0).exists()
    assert recovery["outcome"] == "timeout"
    assert recovery["duration_seconds"] == 300.0
    assert recovery["completion_source"] == "live"
    assert rejected["reason"] == "persistent-post-exit-memory-pressure"
    assert rejected["trial"]["post_exit_recovery"] == recovery
    validate_raw_trial_evidence(RawTrial.from_dict(rejected["trial"]))


def test_parent_finalizes_immediate_hardware_drift_without_recovery_samples(
    tmp_path, monkeypatch
):
    workload = build_canonical_workload()
    measurement = _valid_child_measurement(workload)
    measurement_path = tmp_path / "measurement.json"
    post_exit_path = tmp_path / "post-exit.json"
    recovery_samples_directory = tmp_path / "recovery-samples"
    recovery_path = tmp_path / "recovery.json"
    trial_path = tmp_path / "trial.json"
    immediate = _observation_with_memory("normal", 67)
    immediate["hardware"] = {**immediate["hardware"], "chip": "Other"}

    def run_child(command, *, cwd, check):
        atomic_write_json(measurement_path, measurement, create_only=True)

    monkeypatch.setattr(benchmark_runner.subprocess, "run", run_child)
    monkeypatch.setattr(
        benchmark_runner,
        "collect_post_exit_environment",
        lambda **kwargs: (0.0, immediate),
    )

    trial = benchmark_runner._launch_trial(
        **_launch_trial_arguments(tmp_path),
        evidence_session_identity=measurement["session_identity"],
        journal_attempt_index=0,
        measurement_output=measurement_path,
        post_exit_output=post_exit_path,
        recovery_samples_directory=recovery_samples_directory,
        recovery_output=recovery_path,
        output=trial_path,
        clock=lambda: 0.0,
        sleep=lambda seconds: pytest.fail("immediate failure must not sleep"),
    )

    assert trial.schema_version == 3
    assert trial.post_exit_recovery_samples == ()
    assert trial.post_exit_recovery["outcome"] == "environment-failure"
    assert trial.post_exit_recovery["duration_seconds"] == 0.0
    assert trial.post_exit_recovery["failure_fields"] == ["hardware"]
    assert not recovery_samples_directory.exists()
    validate_raw_trial_evidence(trial)


def _stub_parent_observation(monkeypatch, observation):
    monkeypatch.setattr(
        benchmark_runner,
        "collect_post_exit_environment",
        lambda **kwargs: (0.0, observation),
    )


def test_parent_rejects_wrong_child_count_before_post_exit_publication(
    tmp_path, monkeypatch
):
    workload = build_canonical_workload()
    measurement = build_child_trial_measurement(
        session_identity="sha256:" + "9" * 64,
        journal_attempt_index=0,
        trial={**_valid_trial_payload(workload), "measured_units": 20},
        start=_valid_observation("2026-08-08T00:00:00+00:00"),
        end=_valid_observation("2026-08-08T00:00:01+00:00"),
    )
    measurement_path = tmp_path / "measurement.json"
    post_exit_path = tmp_path / "post-exit.json"

    monkeypatch.setattr(
        benchmark_runner.subprocess,
        "run",
        lambda command, **unused: atomic_write_json(
            measurement_path, measurement, create_only=True
        ),
    )

    with pytest.raises(ValueError, match="child trial does not match"):
        benchmark_runner._launch_trial(
            **_launch_trial_arguments(tmp_path),
            evidence_session_identity=measurement["session_identity"],
            journal_attempt_index=0,
            measurement_output=measurement_path,
            post_exit_output=post_exit_path,
            recovery_samples_directory=tmp_path / "recovery-samples",
            recovery_output=tmp_path / "recovery.json",
            output=tmp_path / "trial.json",
        )

    assert not post_exit_path.exists()


def test_parent_post_exit_output_never_overwrites_existing_evidence(
    tmp_path, monkeypatch
):
    measurement = _valid_child_measurement(build_canonical_workload())
    measurement_path = tmp_path / "measurement.json"
    post_exit_path = tmp_path / "post-exit.json"
    recovery_samples_directory = tmp_path / "recovery-samples"
    recovery_path = tmp_path / "recovery.json"
    trial_path = tmp_path / "trial.json"
    post_exit_path.write_bytes(b"existing immutable post-exit evidence\n")

    def run_child(command, *, cwd, check):
        atomic_write_json(measurement_path, measurement, create_only=True)

    monkeypatch.setattr(benchmark_runner.subprocess, "run", run_child)
    _stub_parent_observation(
        monkeypatch, _valid_observation("2026-08-08T00:00:02+00:00")
    )

    with pytest.raises(FileExistsError):
        benchmark_runner._launch_trial(
            **_launch_trial_arguments(tmp_path),
            evidence_session_identity=measurement["session_identity"],
            journal_attempt_index=0,
            measurement_output=measurement_path,
            post_exit_output=post_exit_path,
            recovery_samples_directory=recovery_samples_directory,
            recovery_output=recovery_path,
            output=trial_path,
        )

    assert post_exit_path.read_bytes() == b"existing immutable post-exit evidence\n"
    assert not recovery_samples_directory.exists()
    assert not recovery_path.exists()
    assert not trial_path.exists()


def test_parent_trial_output_never_overwrites_existing_evidence(tmp_path, monkeypatch):
    measurement = _valid_child_measurement(build_canonical_workload())
    measurement_path = tmp_path / "measurement.json"
    post_exit_path = tmp_path / "post-exit.json"
    recovery_samples_directory = tmp_path / "recovery-samples"
    recovery_path = tmp_path / "recovery.json"
    trial_path = tmp_path / "trial.json"
    trial_path.write_bytes(b"existing immutable trial evidence\n")

    def run_child(command, *, cwd, check):
        atomic_write_json(measurement_path, measurement, create_only=True)

    monkeypatch.setattr(benchmark_runner.subprocess, "run", run_child)
    _stub_parent_observation(
        monkeypatch, _valid_observation("2026-08-08T00:00:02+00:00")
    )

    with pytest.raises(FileExistsError):
        benchmark_runner._launch_trial(
            **_launch_trial_arguments(tmp_path),
            evidence_session_identity=measurement["session_identity"],
            journal_attempt_index=0,
            measurement_output=measurement_path,
            post_exit_output=post_exit_path,
            recovery_samples_directory=recovery_samples_directory,
            recovery_output=recovery_path,
            output=trial_path,
        )

    assert post_exit_path.is_file()
    assert recovery_path.is_file()
    assert trial_path.read_bytes() == b"existing immutable trial evidence\n"


def test_parent_first_recovery_sample_never_overwrites_existing_evidence(
    tmp_path, monkeypatch
):
    measurement = _valid_child_measurement(build_canonical_workload())
    measurement_path = tmp_path / "measurement.json"
    post_exit_path = tmp_path / "post-exit.json"
    recovery_samples_directory = tmp_path / "recovery-samples"
    recovery_sample_path = recovery_samples_directory / "0.json"
    recovery_path = tmp_path / "recovery.json"
    trial_path = tmp_path / "trial.json"
    recovery_samples_directory.mkdir()
    recovery_sample_path.write_bytes(b"existing immutable recovery sample\n")

    monkeypatch.setattr(
        benchmark_runner.subprocess,
        "run",
        lambda command, *, cwd, check: atomic_write_json(
            measurement_path, measurement, create_only=True
        ),
    )
    _stub_parent_observation(monkeypatch, _observation_with_memory("warning", 36))

    with pytest.raises(FileExistsError):
        benchmark_runner._launch_trial(
            **_launch_trial_arguments(tmp_path),
            evidence_session_identity=measurement["session_identity"],
            journal_attempt_index=0,
            measurement_output=measurement_path,
            post_exit_output=post_exit_path,
            recovery_samples_directory=recovery_samples_directory,
            recovery_output=recovery_path,
            output=trial_path,
            clock=_RecoveryClock(),
            sleep=lambda seconds: None,
        )

    assert recovery_sample_path.read_bytes() == b"existing immutable recovery sample\n"
    assert post_exit_path.is_file()
    assert not recovery_path.exists()
    assert not trial_path.exists()


def test_parent_recovery_output_never_overwrites_existing_evidence(
    tmp_path, monkeypatch
):
    measurement = _valid_child_measurement(build_canonical_workload())
    measurement_path = tmp_path / "measurement.json"
    post_exit_path = tmp_path / "post-exit.json"
    recovery_samples_directory = tmp_path / "recovery-samples"
    recovery_path = tmp_path / "recovery.json"
    trial_path = tmp_path / "trial.json"
    recovery_path.write_bytes(b"existing immutable recovery summary\n")

    monkeypatch.setattr(
        benchmark_runner.subprocess,
        "run",
        lambda command, *, cwd, check: atomic_write_json(
            measurement_path, measurement, create_only=True
        ),
    )
    _stub_parent_observation(
        monkeypatch, _valid_observation("2026-08-08T00:00:02+00:00")
    )

    with pytest.raises(FileExistsError):
        benchmark_runner._launch_trial(
            **_launch_trial_arguments(tmp_path),
            evidence_session_identity=measurement["session_identity"],
            journal_attempt_index=0,
            measurement_output=measurement_path,
            post_exit_output=post_exit_path,
            recovery_samples_directory=recovery_samples_directory,
            recovery_output=recovery_path,
            output=trial_path,
        )

    assert recovery_path.read_bytes() == b"existing immutable recovery summary\n"
    assert post_exit_path.is_file()
    assert not recovery_samples_directory.exists()
    assert not trial_path.exists()


def test_nonzero_child_exit_does_not_publish_parent_evidence(tmp_path, monkeypatch):
    measurement_path = tmp_path / "measurement.json"
    post_exit_path = tmp_path / "post-exit.json"
    recovery_samples_directory = tmp_path / "recovery-samples"
    recovery_path = tmp_path / "recovery.json"
    trial_path = tmp_path / "trial.json"

    def fail_child(command, *, cwd, check):
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(benchmark_runner.subprocess, "run", fail_child)

    with pytest.raises(subprocess.CalledProcessError):
        benchmark_runner._launch_trial(
            **_launch_trial_arguments(tmp_path),
            evidence_session_identity="sha256:" + "9" * 64,
            journal_attempt_index=0,
            measurement_output=measurement_path,
            post_exit_output=post_exit_path,
            recovery_samples_directory=recovery_samples_directory,
            recovery_output=recovery_path,
            output=trial_path,
        )

    assert not measurement_path.exists()
    assert not post_exit_path.exists()
    assert not recovery_samples_directory.exists()
    assert not recovery_path.exists()
    assert not trial_path.exists()


def test_paired_trials_stops_before_the_next_process_on_interrupted_memory(
    tmp_path, monkeypatch
):
    workload = build_canonical_workload()
    trial = _valid_raw_trial(workload)
    warning = dict(trial.environment_status["post_exit"])
    warning["memory_pressure"] = "warning"
    rejected = _with_environment_observations(trial, post_exit=warning)
    launches = []

    def launch(**kwargs):
        launches.append(kwargs)
        return rejected

    monkeypatch.setattr(benchmark_runner, "_launch_trial", launch)

    with pytest.raises(ValueError, match="interrupted-post-exit-recovery"):
        benchmark_runner._run_paired_trials(
            harness_root=tmp_path / "harness",
            reference_root=tmp_path / "reference",
            candidate_root=tmp_path / "candidate",
            reference_commit=rejected.source_commit,
            candidate_commit="c" * 40,
            harness_commit=rejected.harness_commit,
            harness_identity=rejected.harness_identity,
            reference_adapter="legacy",
            metrics=("prepared-data",),
            pairs=2,
            warmup=5,
            measure=20,
            comparison_target="baseline",
            attempt_index=0,
            output_directory=tmp_path,
        )

    assert len(launches) == 1


@pytest.mark.parametrize("outcome", ("timeout", "critical", "environment-failure"))
def test_paired_trials_rejects_every_terminal_recovery_outcome_before_next_launch(
    tmp_path, monkeypatch, outcome
):
    workload = build_canonical_workload()
    if outcome == "timeout":
        trial = _recovery_evidence(
            workload,
            immediate_pressure="warning",
            sample_pressures=("warning",),
            sample_elapsed=(300.0,),
            outcome=outcome,
        )[4]
    elif outcome == "critical":
        trial = _recovery_evidence(
            workload,
            immediate_pressure="critical",
            outcome=outcome,
        )[4]
    else:
        trial = _recovery_evidence(
            workload,
            immediate_pressure="warning",
            sample_pressures=("normal",),
            sample_elapsed=(5.0,),
            sample_status_changes=({"power_connected": False},),
            outcome=outcome,
            failure_fields=("power_connected",),
        )[4]
    launches = []
    validated = []
    original_validate = benchmark_runner._validate_acceptance_environment

    def launch(**kwargs):
        launches.append(kwargs)
        return trial

    def validate_at_boundary(current_workload, current_trial):
        validated.append(current_trial)
        original_validate(current_workload, current_trial)

    monkeypatch.setattr(benchmark_runner, "_launch_trial", launch)
    monkeypatch.setattr(
        benchmark_runner, "_validate_acceptance_environment", validate_at_boundary
    )

    with pytest.raises(ValueError, match=r"raw .*environment"):
        benchmark_runner._run_paired_trials(
            harness_root=tmp_path / "harness",
            reference_root=tmp_path / "reference",
            candidate_root=tmp_path / "candidate",
            reference_commit=trial.source_commit,
            candidate_commit="c" * 40,
            harness_commit=trial.harness_commit,
            harness_identity=trial.harness_identity,
            reference_adapter="legacy",
            metrics=("prepared-data",),
            pairs=2,
            warmup=5,
            measure=20,
            comparison_target="baseline",
            attempt_index=0,
            output_directory=tmp_path,
        )

    assert len(launches) == 1
    assert validated == [trial]
    assert validated[0].schema_version == 3
    assert validated[0].post_exit_recovery["outcome"] == outcome


def test_paired_trials_share_one_identity_bound_evidence_session(tmp_path, monkeypatch):
    workload = build_canonical_workload()
    launches = []

    def launch(**kwargs):
        launches.append(kwargs)
        measurement = _valid_child_measurement(
            workload,
            session_identity=kwargs["evidence_session_identity"],
            journal_attempt_index=kwargs["journal_attempt_index"],
        )
        post_exit = _valid_post_exit_observation(measurement)
        recovery = _valid_post_exit_recovery(measurement, post_exit)
        return finalize_raw_trial(measurement, post_exit, (), recovery)

    monkeypatch.setattr(benchmark_runner, "_launch_trial", launch)

    trials = benchmark_runner._run_paired_trials(
        harness_root=tmp_path / "harness",
        reference_root=tmp_path / "reference",
        candidate_root=tmp_path / "candidate",
        reference_commit="1" * 40,
        candidate_commit="2" * 40,
        harness_commit="3" * 40,
        harness_identity="sha256:" + "4" * 64,
        reference_adapter="legacy",
        metrics=("prepared-data",),
        pairs=1,
        warmup=5,
        measure=20,
        comparison_target="baseline:sha256:" + "5" * 64,
        attempt_index=1,
        output_directory=tmp_path,
    )

    assert len(trials) == 2
    assert {launch["evidence_session_identity"] for launch in launches} == {
        launches[0]["evidence_session_identity"]
    }
    assert {launch["journal_attempt_index"] for launch in launches} == {1}
    assert {
        tuple(launch["measurement_output"].suffixes[-2:]) for launch in launches
    } == {(".measurement", ".json")}
    assert {tuple(launch["post_exit_output"].suffixes[-2:]) for launch in launches} == {
        (".post-exit", ".json")
    }
    assert {tuple(launch["recovery_output"].suffixes[-2:]) for launch in launches} == {
        (".recovery", ".json")
    }
    assert len({launch["recovery_output"] for launch in launches}) == 2
    assert len({launch["recovery_samples_directory"] for launch in launches}) == 2
    assert all(
        path.name.endswith(".recovery-samples")
        for path in (launch["recovery_samples_directory"] for launch in launches)
    )
    assert {tuple(launch["output"].suffixes[-2:]) for launch in launches} == {
        (".trial", ".json")
    }


def test_parent_launches_each_metric_with_its_canonical_count(tmp_path, monkeypatch):
    workload = build_canonical_workload()
    launches = []
    monkeypatch.setattr(
        benchmark_runner,
        "_launch_trial",
        lambda **kwargs: (
            launches.append(kwargs)
            or _valid_raw_trial(workload, metric=kwargs["metric"])
        ),
    )
    monkeypatch.setattr(
        benchmark_runner, "_validate_acceptance_environment", lambda *_: None
    )
    monkeypatch.setattr(
        benchmark_runner, "_validate_software_versions", lambda *_: None
    )

    benchmark_runner._run_paired_trials(
        harness_root=tmp_path / "harness",
        reference_root=tmp_path / "reference",
        candidate_root=tmp_path / "candidate",
        reference_commit="1" * 40,
        candidate_commit="2" * 40,
        harness_commit="3" * 40,
        harness_identity="sha256:" + "4" * 64,
        reference_adapter="legacy",
        metrics=METRIC_NAMES,
        pairs=1,
        warmup=5,
        measure=20,
        prepared_data_measure=100,
        comparison_target="baseline:sha256:" + "5" * 64,
        attempt_index=0,
        output_directory=tmp_path,
    )

    assert {launch["metric"]: launch["measure"] for launch in launches} == {
        unit.metric: unit.measured_units for unit in workload.work_units
    }


def test_baseline_launch_routes_recovery_paths_when_journal_supports_them(
    tmp_path, monkeypatch
):
    slot = BaselineSlot("prepared-data", 0)
    attempt = SimpleNamespace(journal_attempt_index=0, path=tmp_path / "trial.json")
    journal = SimpleNamespace(
        session={"identity": "sha256:" + "9" * 64},
        measurement_path=lambda current_slot, index: (
            tmp_path / f"{current_slot.metric}-{index}.measurement.json"
        ),
        post_exit_path=lambda current_slot, index: (
            tmp_path / f"{current_slot.metric}-{index}.post-exit.json"
        ),
        recovery_samples_path=lambda current_slot, index: (
            tmp_path / f"{current_slot.metric}-{index}.recovery-samples"
        ),
        recovery_path=lambda current_slot, index: (
            tmp_path / f"{current_slot.metric}-{index}.recovery.json"
        ),
    )
    captured = {}
    observation = _valid_observation("2026-08-09T00:00:00+00:00")
    arguments = SimpleNamespace(
        source_commit=benchmark_runner.PINNED_BASELINE_SOURCE_COMMIT,
        metrics=METRIC_NAMES,
        pairs=benchmark_runner.SCREEN_PAIRS,
        warmup=benchmark_runner.WARMUP_UNITS,
        measure=benchmark_runner.DEFAULT_MEASURED_UNITS,
        prepared_data_measure=benchmark_runner.PREPARED_DATA_MEASURED_UNITS,
    )

    class FakeBaselineJournal:
        @staticmethod
        def open(*unused):
            return journal

        @staticmethod
        def expected_slots(*unused):
            return (slot,)

    @contextmanager
    def managed_worktree(*unused):
        yield tmp_path / "source"

    monkeypatch.setattr(benchmark_runner, "BaselineJournal", FakeBaselineJournal)
    monkeypatch.setattr(benchmark_runner, "_run_command", lambda *unused, **kwargs: "")
    monkeypatch.setattr(
        benchmark_runner, "_git_commit", lambda *unused: arguments.source_commit
    )
    monkeypatch.setattr(
        benchmark_runner, "harness_content_identity", lambda path: "harness"
    )
    monkeypatch.setattr(
        benchmark_runner,
        "collect_environment",
        lambda: (
            observation["hardware"],
            observation["environment_status"],
            observation["software_versions"],
        ),
    )
    monkeypatch.setattr(
        benchmark_runner,
        "write_paired_pretraining_representations",
        lambda *unused: {},
    )
    monkeypatch.setattr(
        benchmark_runner,
        "build_session_document",
        lambda **unused: journal.session,
    )
    monkeypatch.setattr(
        benchmark_runner, "_managed_detached_worktree", managed_worktree
    )
    monkeypatch.setattr(
        benchmark_runner,
        "capture_baseline_trials",
        lambda **kwargs: (kwargs["launch_trial"](slot, attempt),),
    )
    monkeypatch.setattr(
        benchmark_runner, "_launch_trial", lambda **kwargs: captured.update(kwargs)
    )
    monkeypatch.setattr(
        benchmark_runner, "publish_baseline_from_journal", lambda **kwargs: {}
    )

    assert (
        benchmark_runner._record_baseline_locked(
            arguments,
            harness_root=tmp_path / "harness",
            state_root=tmp_path / "state",
            manifest_path=tmp_path / "baseline.json",
            raw_output_path=tmp_path / "baseline.jsonl",
        )
        == 0
    )
    assert captured["recovery_samples_directory"] == (
        tmp_path / "prepared-data-0.recovery-samples"
    )
    assert captured["recovery_output"] == tmp_path / "prepared-data-0.recovery.json"


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


def test_journal_retains_the_complete_recovery_chain_after_acceptance(tmp_path):
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    measurement, post_exit, samples, recovery, trial = _persist_journal_evidence(
        journal, attempt
    )

    journal.accept_inflight(attempt, trial)

    assert journal.measurement_path(slot, 0).is_file()
    assert journal.post_exit_path(slot, 0).is_file()
    assert journal.recovery_path(slot, 0).is_file()
    assert tuple(
        journal.recovery_sample_path(slot, 0, index).is_file()
        for index in range(len(samples))
    ) == (True,) * len(samples)
    assert journal.load_accepted((slot,)) == {slot: trial}
    assert read_json_object(journal.measurement_path(slot, 0), label="measurement") == (
        measurement
    )
    assert read_json_object(journal.post_exit_path(slot, 0), label="post-exit") == (
        post_exit
    )
    assert read_json_object(journal.recovery_path(slot, 0), label="recovery") == (
        recovery
    )


def test_journal_reports_measurement_only_as_pending_immediate_evidence(tmp_path):
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    measurement, _post_exit, _samples, _recovery, _trial = _persist_journal_evidence(
        journal, attempt, post_exit=False, inflight=False
    )

    pending = journal.load_pending_attempts((slot,))

    assert pending == (
        baseline_journal.JournalAttemptEvidence(
            attempt=attempt,
            measurement=measurement,
            post_exit=None,
            recovery_samples=(),
            recovery=None,
            trial=None,
        ),
    )


def test_journal_rejects_post_exit_without_its_exact_measurement(tmp_path):
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    _measurement, post_exit, _samples, _recovery, _trial = _journal_trial_evidence(
        journal, attempt
    )
    atomic_write_json(journal.post_exit_path(slot, 0), post_exit, create_only=True)

    with pytest.raises(ValueError, match="post-exit evidence has no measurement"):
        journal.load_pending_attempts((slot,))


def test_journal_reports_post_exit_without_recovery_as_pending(tmp_path):
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    measurement, post_exit, _samples, _recovery, _trial = _persist_journal_evidence(
        journal, attempt, recovery=False, inflight=False
    )

    assert journal.load_pending_attempts((slot,)) == (
        baseline_journal.JournalAttemptEvidence(
            attempt=attempt,
            measurement=measurement,
            post_exit=post_exit,
            recovery_samples=(),
            recovery=None,
            trial=None,
        ),
    )


def test_journal_rejects_recovery_sample_without_post_exit(tmp_path):
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    measurement, _post_exit, samples, _recovery, _trial = _journal_trial_evidence(
        journal, attempt
    )
    atomic_write_json(journal.measurement_path(slot, 0), measurement, create_only=True)
    atomic_write_json(
        journal.recovery_sample_path(slot, 0, 0), samples[0], create_only=True
    )

    with pytest.raises(ValueError, match="recovery sample has no post-exit evidence"):
        journal.load_pending_attempts((slot,))


def test_journal_rejects_recovery_sample_bound_to_another_immediate_identity(
    tmp_path,
):
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    measurement, post_exit, _samples, _recovery, _trial = _journal_trial_evidence(
        journal, attempt
    )
    other_post_exit = json.loads(json.dumps(post_exit))
    other_post_exit["observed_at_utc"] = "2026-08-09T00:00:04+00:00"
    other_body = {
        key: value for key, value in other_post_exit.items() if key != "identity"
    }
    other_post_exit["identity"] = structured_identity(
        "sml-parent-post-exit-observation-v1", other_body
    )
    observation = _valid_observation("2026-08-09T00:00:03+00:00")
    foreign_sample = build_post_exit_recovery_sample(
        measurement=measurement,
        post_exit=other_post_exit,
        sample_index=0,
        previous_sample_identity=None,
        elapsed_seconds=5.0,
        **observation,
    )
    atomic_write_json(journal.measurement_path(slot, 0), measurement, create_only=True)
    atomic_write_json(journal.post_exit_path(slot, 0), post_exit, create_only=True)
    atomic_write_json(
        journal.recovery_sample_path(slot, 0, 0), foreign_sample, create_only=True
    )

    with pytest.raises(ValueError, match="recovery .* does not match bound evidence"):
        journal.load_pending_attempts((slot,))


def test_journal_rejects_recovery_sample_index_gap(tmp_path):
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    measurement, post_exit, samples, _recovery, _trial = _journal_trial_evidence(
        journal, attempt
    )
    atomic_write_json(journal.measurement_path(slot, 0), measurement, create_only=True)
    atomic_write_json(journal.post_exit_path(slot, 0), post_exit, create_only=True)
    atomic_write_json(
        journal.recovery_sample_path(slot, 0, 1), samples[1], create_only=True
    )

    with pytest.raises(ValueError, match="recovery sample indices have a gap"):
        journal.load_pending_attempts((slot,))


def test_journal_rejects_recovery_sample_previous_identity_fork(tmp_path):
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    measurement, post_exit, samples, _recovery, _trial = _journal_trial_evidence(
        journal, attempt
    )
    forked = {
        **samples[1],
        "previous_sample_identity": "sha256:" + "0" * 64,
    }
    forked_body = {key: value for key, value in forked.items() if key != "identity"}
    forked["identity"] = structured_identity(
        "sml-parent-post-exit-recovery-sample-v1", forked_body
    )
    atomic_write_json(journal.measurement_path(slot, 0), measurement, create_only=True)
    atomic_write_json(journal.post_exit_path(slot, 0), post_exit, create_only=True)
    atomic_write_json(
        journal.recovery_sample_path(slot, 0, 0), samples[0], create_only=True
    )
    atomic_write_json(
        journal.recovery_sample_path(slot, 0, 1), forked, create_only=True
    )

    with pytest.raises(ValueError, match="previous sample identity"):
        journal.load_pending_attempts((slot,))


def test_journal_rejects_nonincreasing_pending_recovery_sample_elapsed_time(tmp_path):
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    measurement, post_exit, samples, _recovery, _trial = _journal_trial_evidence(
        journal, attempt
    )
    nonincreasing = {
        **samples[1],
        "elapsed_seconds": samples[0]["elapsed_seconds"],
    }
    nonincreasing_body = {
        key: value for key, value in nonincreasing.items() if key != "identity"
    }
    nonincreasing["identity"] = structured_identity(
        "sml-parent-post-exit-recovery-sample-v1", nonincreasing_body
    )
    atomic_write_json(journal.measurement_path(slot, 0), measurement, create_only=True)
    atomic_write_json(journal.post_exit_path(slot, 0), post_exit, create_only=True)
    atomic_write_json(
        journal.recovery_sample_path(slot, 0, 0), samples[0], create_only=True
    )
    atomic_write_json(
        journal.recovery_sample_path(slot, 0, 1), nonincreasing, create_only=True
    )

    with pytest.raises(ValueError, match="elapsed_seconds must be increasing"):
        journal.load_pending_attempts((slot,))


def test_journal_rejects_recovery_summary_without_its_exact_samples(tmp_path):
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    measurement, post_exit, samples, recovery, _trial = _journal_trial_evidence(
        journal, attempt
    )
    atomic_write_json(journal.measurement_path(slot, 0), measurement, create_only=True)
    atomic_write_json(journal.post_exit_path(slot, 0), post_exit, create_only=True)
    for index, sample in enumerate(samples[:-1]):
        atomic_write_json(
            journal.recovery_sample_path(slot, 0, index), sample, create_only=True
        )
    atomic_write_json(journal.recovery_path(slot, 0), recovery, create_only=True)

    with pytest.raises(ValueError, match="sample identities do not match samples"):
        journal.load_pending_attempts((slot,))


def test_journal_rejects_inflight_without_recovery_summary(tmp_path):
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    _persist_journal_evidence(journal, attempt, recovery=False)

    with pytest.raises(ValueError, match="inflight trial has no recovery summary"):
        journal.load_pending_attempts((slot,))


@pytest.mark.parametrize("outcome", ["accepted", "rejected"])
def test_journal_rejects_terminal_evidence_missing_recovery_summary(tmp_path, outcome):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    source_trial = _valid_raw_trial(workload)
    if outcome == "rejected":
        source_trial = _with_thermal_state(source_trial, "fair", 1)
    _measurement, _post_exit, _samples, _recovery, trial = _persist_journal_evidence(
        journal, attempt, source_trial
    )
    if outcome == "accepted":
        accepted_path = journal.accepted_path(slot)
        accepted_path.parent.mkdir(parents=True)
        os.link(attempt.path, accepted_path)
    else:
        journal.reject_inflight(attempt, trial, reason="non-nominal-thermal")
    journal.recovery_path(slot, 0).unlink()

    with pytest.raises(
        ValueError,
        match=(
            "incomplete staged evidence|no recovery summary|"
            "does not match its post-exit recovery"
        ),
    ):
        journal.load_pending_attempts((slot,))


def test_journal_rejects_measurement_from_another_session(tmp_path):
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    measurement = _valid_child_measurement(build_canonical_workload())
    atomic_write_json(journal.measurement_path(slot, 0), measurement, create_only=True)

    with pytest.raises(ValueError, match="does not match the journal session"):
        journal.load_pending_attempts((slot,))


def test_journal_rejects_post_exit_bound_to_another_measurement(tmp_path):
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    measurement, _post_exit, _samples, _recovery, _trial = _journal_trial_evidence(
        journal, attempt
    )
    other_measurement = json.loads(json.dumps(measurement))
    other_measurement["trial"]["value"] += 1.0
    other_body = {
        key: value for key, value in other_measurement.items() if key != "identity"
    }
    other_measurement["identity"] = structured_identity(
        "sml-child-trial-measurement-v1", other_body
    )
    other_post_exit = _valid_post_exit_observation(other_measurement)
    atomic_write_json(journal.measurement_path(slot, 0), measurement, create_only=True)
    atomic_write_json(
        journal.post_exit_path(slot, 0), other_post_exit, create_only=True
    )

    with pytest.raises(
        ValueError, match="child_measurement_identity does not match child measurement"
    ):
        journal.load_pending_attempts((slot,))


def test_journal_rejects_inflight_without_staged_evidence(tmp_path):
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    atomic_write_json(
        attempt.path,
        _valid_raw_trial(build_canonical_workload()).to_dict(),
        create_only=True,
    )

    with pytest.raises(ValueError, match="inflight trial has no post-exit evidence"):
        journal.load_pending_attempts((slot,))


def test_journal_rejects_measurement_only_and_preserves_its_evidence(tmp_path):
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    measurement, _post_exit, _samples, _recovery, _trial = _persist_journal_evidence(
        journal, attempt, post_exit=False, inflight=False
    )

    journal.reject_unfinalized(
        attempt,
        measurement,
        reason="missing-immediate-post-exit-evidence",
    )

    rejected = read_json_object(journal.rejected_path(slot, 0), label="rejected trial")
    assert rejected == {
        "kind": "sml-baseline-rejected-trial",
        "version": 3,
        "journal_attempt_index": 0,
        "reason": "missing-immediate-post-exit-evidence",
        "child_measurement_identity": measurement["identity"],
        "post_exit_observation_identity": None,
        "post_exit_recovery_identity": None,
        "trial": None,
        "identity": rejected["identity"],
    }
    assert journal.measurement_path(slot, 0).is_file()
    assert journal.load_pending_attempts((slot,)) == ()
    assert journal.next_attempt(slot).journal_attempt_index == 1


def test_journal_reject_unfinalized_requires_measurement_only_evidence(tmp_path):
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    measurement, _post_exit, _samples, _recovery, _trial = _persist_journal_evidence(
        journal, attempt, inflight=False
    )

    with pytest.raises(ValueError, match="measurement-only evidence"):
        journal.reject_unfinalized(
            attempt,
            measurement,
            reason="missing-immediate-post-exit-evidence",
        )


def test_journal_reject_unfinalized_requires_the_exact_reason(tmp_path):
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    measurement, _post_exit, _samples, _recovery, _trial = _persist_journal_evidence(
        journal, attempt, post_exit=False, inflight=False
    )

    with pytest.raises(ValueError, match="invalid reason"):
        journal.reject_unfinalized(attempt, measurement, reason="process-failed")


def test_journal_memory_rejection_needs_no_thermal_trigger(tmp_path):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    nominal = _valid_raw_trial(workload)
    warning = dict(nominal.environment_status["post_exit"])
    warning["memory_pressure"] = "warning"
    memory_trial = _with_environment_observations(nominal, post_exit=warning)
    measurement, post_exit, _samples, _recovery, persisted = _persist_journal_evidence(
        journal, attempt, memory_trial
    )

    journal.reject_inflight(attempt, persisted, reason="interrupted-post-exit-recovery")

    rejected = read_json_object(journal.rejected_path(slot, 0), label="rejected trial")
    assert rejected["version"] == 3
    assert rejected["child_measurement_identity"] == measurement["identity"]
    assert rejected["post_exit_observation_identity"] == post_exit["identity"]
    assert (
        rejected["post_exit_recovery_identity"]
        == persisted.post_exit_recovery["identity"]
    )
    assert journal.measurement_path(slot, 0).is_file()
    assert journal.post_exit_path(slot, 0).is_file()
    assert not (journal.root / "thermal-waits").exists()
    assert journal.next_attempt(slot).journal_attempt_index == 1


def test_journal_reject_inflight_rejects_an_unknown_reason_before_writing(tmp_path):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    fair = _with_thermal_state(_valid_raw_trial(workload), "fair", 1)
    _measurement, _post_exit, _samples, _recovery, fair = _persist_journal_evidence(
        journal, attempt, fair
    )

    with pytest.raises(ValueError, match="rejected trial reason is invalid"):
        journal.reject_inflight(attempt, fair, reason="retry-environment-later")

    assert attempt.path.is_file()
    assert not journal.rejected_path(slot, 0).exists()


@pytest.mark.parametrize(
    "reason",
    (
        "critical-post-exit-memory-pressure",
        "post-exit-recovery-environment-violation",
        "interrupted-post-exit-recovery",
    ),
)
def test_journal_recognizes_every_schema_v3_recovery_rejection_reason(tmp_path, reason):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    fair = _with_thermal_state(_valid_raw_trial(workload), "fair", 1)
    _measurement, _post_exit, _samples, _recovery, trial = _persist_journal_evidence(
        journal, attempt, fair
    )

    with pytest.raises(ValueError, match="expected non-nominal-thermal"):
        journal.reject_inflight(attempt, trial, reason=reason)

    assert attempt.path.is_file()
    assert not journal.rejected_path(slot, 0).exists()


def test_journal_reject_inflight_refuses_an_accept_disposition(tmp_path):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    _measurement, _post_exit, _samples, _recovery, nominal = _persist_journal_evidence(
        journal, attempt, _valid_raw_trial(workload)
    )

    with pytest.raises(ValueError, match="accepted trial cannot be rejected"):
        journal.reject_inflight(attempt, nominal, reason="non-nominal-thermal")

    assert attempt.path.is_file()
    assert not journal.rejected_path(slot, 0).exists()


@pytest.mark.parametrize(
    ("primary_signal", "supplied_reason", "expected_reason"),
    (
        (
            "start-memory",
            "critical-measurement-memory-pressure",
            "non-normal-start-memory-pressure",
        ),
        (
            "end-memory",
            "persistent-post-exit-memory-pressure",
            "critical-measurement-memory-pressure",
        ),
        (
            "post-exit-memory",
            "non-nominal-thermal",
            "interrupted-post-exit-recovery",
        ),
        (
            "thermal",
            "persistent-post-exit-memory-pressure",
            "non-nominal-thermal",
        ),
    ),
)
def test_journal_reject_inflight_requires_the_ordered_evidence_reason(
    tmp_path, primary_signal, supplied_reason, expected_reason
):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    nominal = _valid_raw_trial(workload)
    start = dict(nominal.environment_status["start"])
    end = dict(nominal.environment_status["end"])
    post_exit = dict(nominal.environment_status["post_exit"])
    if primary_signal == "start-memory":
        start["memory_pressure"] = "warning"
        end["memory_pressure"] = "critical"
        post_exit["memory_pressure"] = "warning"
        post_exit.update(thermal_state="fair", thermal_state_raw_value=1)
    elif primary_signal == "end-memory":
        end["memory_pressure"] = "critical"
        post_exit["memory_pressure"] = "warning"
        post_exit.update(thermal_state="fair", thermal_state_raw_value=1)
    elif primary_signal == "post-exit-memory":
        post_exit["memory_pressure"] = "warning"
    else:
        post_exit.update(thermal_state="fair", thermal_state_raw_value=1)
    rejected = _with_environment_observations(
        nominal,
        start=start,
        end=end,
        post_exit=post_exit,
    )
    attempt = journal.next_attempt(slot)
    _measurement, _post_exit, _samples, _recovery, rejected = _persist_journal_evidence(
        journal, attempt, rejected
    )

    with pytest.raises(ValueError, match=expected_reason):
        journal.reject_inflight(attempt, rejected, reason=supplied_reason)

    assert attempt.path.is_file()
    assert not journal.rejected_path(slot, 0).exists()


def _rewrite_rejected_reason(journal, slot, reason):
    path = journal.rejected_path(slot, 0)
    document = read_json_object(path, label="rejected trial")
    body = {
        **{key: value for key, value in document.items() if key != "identity"},
        "reason": reason,
    }
    path.unlink()
    atomic_write_json(
        path,
        {
            **body,
            "identity": structured_identity("sml-baseline-rejected-trial-v3", body),
        },
        create_only=True,
    )


@pytest.mark.parametrize(
    "mutated_reason",
    (
        "retry-environment-later",
        "missing-immediate-post-exit-evidence",
        "persistent-post-exit-memory-pressure",
    ),
)
def test_journal_load_rejects_a_resigned_wrong_finalized_reason(
    tmp_path, mutated_reason
):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    fair = _with_thermal_state(_valid_raw_trial(workload), "fair", 1)
    _measurement, _post_exit, _samples, _recovery, fair = _persist_journal_evidence(
        journal, attempt, fair
    )
    journal.reject_inflight(attempt, fair, reason="non-nominal-thermal")
    _rewrite_rejected_reason(journal, slot, mutated_reason)

    with pytest.raises(ValueError, match="rejected trial reason"):
        journal.next_attempt(slot)


def test_journal_load_refuses_a_rejected_accept_disposition(tmp_path):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    measurement, post_exit, _samples, recovery, nominal = _persist_journal_evidence(
        journal, attempt, _valid_raw_trial(workload)
    )
    body = {
        "kind": "sml-baseline-rejected-trial",
        "version": 3,
        "journal_attempt_index": 0,
        "reason": "non-nominal-thermal",
        "child_measurement_identity": measurement["identity"],
        "post_exit_observation_identity": post_exit["identity"],
        "post_exit_recovery_identity": recovery["identity"],
        "trial": nominal.to_dict(),
    }
    attempt.path.unlink()
    atomic_write_json(
        journal.rejected_path(slot, 0),
        {
            **body,
            "identity": structured_identity("sml-baseline-rejected-trial-v3", body),
        },
        create_only=True,
    )

    with pytest.raises(ValueError, match="accepted trial cannot be rejected"):
        journal.next_attempt(slot)


def test_journal_rejects_both_outcomes_for_one_attempt(tmp_path):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    measurement, post_exit, _samples, recovery, trial = _persist_journal_evidence(
        journal,
        attempt,
        _with_thermal_state(_valid_raw_trial(workload), "fair", 1),
    )
    accepted_path = journal.accepted_path(slot)
    accepted_path.parent.mkdir(parents=True)
    os.link(attempt.path, accepted_path)
    body = {
        "kind": "sml-baseline-rejected-trial",
        "version": 3,
        "journal_attempt_index": 0,
        "reason": "non-nominal-thermal",
        "child_measurement_identity": measurement["identity"],
        "post_exit_observation_identity": post_exit["identity"],
        "post_exit_recovery_identity": recovery["identity"],
        "trial": trial.to_dict(),
    }
    atomic_write_json(
        journal.rejected_path(slot, 0),
        {
            **body,
            "identity": structured_identity("sml-baseline-rejected-trial-v3", body),
        },
        create_only=True,
    )

    with pytest.raises(ValueError, match="both accepted and rejected outcomes"):
        journal.load_pending_attempts((slot,))

    assert attempt.path.is_file()


def test_journal_rejects_an_attempt_index_gap_in_staged_evidence(tmp_path):
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    forged = JournalAttempt(slot, 1, journal.inflight_path(slot, 1))
    _persist_journal_evidence(journal, forged, post_exit=False, inflight=False)

    with pytest.raises(ValueError, match="attempt indices have a gap"):
        journal.load_pending_attempts((slot,))


@pytest.mark.parametrize(
    "category", ["measurements", "post-exit", "recovery-samples", "recovery"]
)
def test_journal_rejects_unsupported_stage_paths(tmp_path, category):
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    unsupported = journal.root / category / "unsupported" / "0" / "0.json"
    atomic_write_json(unsupported, {}, create_only=True)

    with pytest.raises(ValueError, match="unsupported metric directory"):
        journal.load_pending_attempts((BaselineSlot("prepared-data", 0),))


@pytest.mark.parametrize(
    "category", ["measurements", "post-exit", "recovery-samples", "recovery"]
)
def test_journal_refuses_symlinked_stage_ancestors(tmp_path, category):
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    outside = tmp_path / f"outside-{category}"
    outside.mkdir()
    stage_root = journal.root / category
    stage_root.mkdir()
    (stage_root / "prepared-data").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError):
        journal.load_pending_attempts((BaselineSlot("prepared-data", 0),))


def test_journal_promotes_an_inflight_trial_and_resumes_the_slot(tmp_path):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    _measurement, _post_exit, _samples, _recovery, trial = _persist_journal_evidence(
        journal, attempt, _valid_raw_trial(workload)
    )

    journal.accept_inflight(attempt, trial)

    assert journal.load_accepted((slot,)) == {slot: trial}
    assert not attempt.path.exists()
    with pytest.raises(ValueError, match="accepted slot is immutable"):
        journal.accept_inflight(
            JournalAttempt(slot, 1, journal.inflight_path(slot, 1)),
            _with_trial_payload(trial, value=trial.value + 1.0),
        )


def test_journal_preserves_rejected_trial_and_uses_a_new_attempt_number(tmp_path):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    _measurement, _post_exit, _samples, _recovery, trial = _persist_journal_evidence(
        journal,
        attempt,
        _with_thermal_state(_valid_raw_trial(workload), "fair", 1),
    )

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
    _measurement, _post_exit, _samples, _recovery, trial = _persist_journal_evidence(
        journal, attempt, _valid_raw_trial(workload)
    )
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
    _measurement, _post_exit, _samples, _recovery, _trial = _persist_journal_evidence(
        journal, attempt, _valid_raw_trial(workload)
    )
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
    fair = _with_thermal_state(_valid_raw_trial(workload), "fair", 1)
    _measurement, _post_exit, _samples, _recovery, trial = _persist_journal_evidence(
        journal, attempt, fair
    )
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
    forged = JournalAttempt(slot, 1, journal.inflight_path(slot, 1))
    _measurement, _post_exit, _samples, _recovery, trial = _persist_journal_evidence(
        journal, forged, _valid_raw_trial(workload)
    )

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


def _persist_rejected_trial_for_recovery_test(journal, slot, reason):
    attempt = journal.next_attempt(slot)
    workload = build_canonical_workload()
    if reason == "missing-immediate-post-exit-evidence":
        measurement, _post_exit, _samples, _recovery, _trial = (
            _persist_journal_evidence(journal, attempt, post_exit=False, inflight=False)
        )
        journal.reject_unfinalized(attempt, measurement, reason=reason)
    else:
        trial = _valid_raw_trial(workload, pair_index=slot.pair_index)
        if reason == "interrupted-post-exit-recovery":
            warning = dict(trial.environment_status["post_exit"])
            warning["memory_pressure"] = "warning"
            trial = _with_environment_observations(trial, post_exit=warning)
        elif reason == "non-nominal-thermal":
            trial = _with_thermal_state(trial, "fair", 1)
        _measurement, _post_exit, _samples, _recovery, persisted = (
            _persist_journal_evidence(journal, attempt, trial)
        )
        journal.reject_inflight(attempt, persisted, reason=reason)
    return read_json_object(
        journal.rejected_path(slot, attempt.journal_attempt_index),
        label="rejected trial",
    )


@pytest.mark.parametrize(
    "reason",
    (
        "interrupted-post-exit-recovery",
        "missing-immediate-post-exit-evidence",
    ),
)
@pytest.mark.parametrize("boundary", ("record", "load"))
def test_journal_never_uses_a_nonthermal_rejection_as_a_thermal_trigger(
    tmp_path, reason, boundary
):
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    rejected = _persist_rejected_trial_for_recovery_test(journal, slot, reason)
    trigger_body = {
        "kind": "sml-baseline-thermal-recovery-trigger",
        "version": 1,
        "source": "rejected-trial",
        "rejected_trial_identity": rejected["identity"],
    }

    if boundary == "load":
        trigger = {
            **trigger_body,
            "identity": structured_identity(
                "sml-baseline-thermal-recovery-trigger-v1", trigger_body
            ),
        }
        atomic_write_json(
            journal.root
            / "thermal-waits"
            / slot.metric
            / str(slot.pair_index)
            / "0"
            / "trigger.json",
            trigger,
            create_only=True,
        )

    with pytest.raises(
        ValueError,
        match="requires a non-nominal-thermal rejected trial",
    ):
        if boundary == "record":
            journal.record_recovery_trigger(
                slot,
                0,
                {
                    "source": "rejected-trial",
                    "rejected_trial_identity": rejected["identity"],
                },
            )
        else:
            journal._validate_thermal_recovery_history()


def test_journal_accepts_a_non_nominal_thermal_rejection_as_a_recovery_trigger(
    tmp_path,
):
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    rejected = _persist_rejected_trial_for_recovery_test(
        journal, slot, "non-nominal-thermal"
    )

    journal.record_recovery_trigger(
        slot,
        0,
        {
            "source": "rejected-trial",
            "rejected_trial_identity": rejected["identity"],
        },
    )

    assert journal._validate_thermal_recovery_history() == {slot: (0,)}


def test_journal_rejects_boolean_rejected_attempt_index(tmp_path):
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    first = JournalAttempt(slot, 0, journal.inflight_path(slot, 0))
    second = JournalAttempt(slot, 1, journal.inflight_path(slot, 1))
    first_measurement, _post_exit, _samples, _recovery, _trial = (
        _persist_journal_evidence(journal, first, post_exit=False, inflight=False)
    )
    second_measurement, _post_exit, _samples, _recovery, _trial = (
        _persist_journal_evidence(journal, second, post_exit=False, inflight=False)
    )
    zero_body = {
        "kind": "sml-baseline-rejected-trial",
        "version": 3,
        "journal_attempt_index": 0,
        "reason": "missing-immediate-post-exit-evidence",
        "child_measurement_identity": first_measurement["identity"],
        "post_exit_observation_identity": None,
        "post_exit_recovery_identity": None,
        "trial": None,
    }
    one_body = {
        **zero_body,
        "journal_attempt_index": True,
        "child_measurement_identity": second_measurement["identity"],
    }
    atomic_write_json(
        journal.rejected_path(slot, 0),
        {
            **zero_body,
            "identity": structured_identity(
                "sml-baseline-rejected-trial-v3", zero_body
            ),
        },
        create_only=True,
    )
    atomic_write_json(
        journal.rejected_path(slot, 1),
        {
            **one_body,
            "identity": structured_identity("sml-baseline-rejected-trial-v3", one_body),
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
    _measurement, _post_exit, _samples, _recovery, trial = _persist_journal_evidence(
        journal, attempt, _valid_raw_trial(workload)
    )
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
    _measurement, _post_exit, _samples, _recovery, trial = _persist_journal_evidence(
        journal, attempt, _valid_raw_trial(workload)
    )
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
    _measurement, _post_exit, _samples, _recovery, _trial = _persist_journal_evidence(
        journal, forged, _valid_raw_trial(workload)
    )
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
    _measurement, _post_exit, _samples, _recovery, _first_trial = (
        _persist_journal_evidence(journal, first, _valid_raw_trial(workload))
    )
    _persist_journal_evidence(journal, second, _valid_raw_trial(workload))
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


def test_post_exit_recovery_returns_immediately_for_normal_memory():
    clock = _RecoveryClock()
    collected = []

    result = wait_for_post_exit_memory_recovery(
        immediate_observation=_valid_observation("2026-08-09T00:00:00+00:00"),
        immediate_started_at=clock(),
        recovery_policy=_valid_recovery_policy(build_canonical_workload()),
        collect=lambda _deadline: collected.append(True),
        classify_nonmemory=lambda observation: (),
        record_sample=lambda *args: pytest.fail("normal memory must not sample"),
        clock=clock,
        sleep=clock.sleep,
    )

    assert result == PostExitMemoryRecoveryResult("not-required", 0.0, (), ())
    assert collected == []
    assert clock.sleeps == []


@pytest.mark.parametrize(
    ("pressure", "failure_fields", "outcome"),
    (
        ("critical", (), "critical"),
        ("normal", ("hardware",), "environment-failure"),
    ),
)
def test_post_exit_recovery_returns_immediately_for_terminal_immediate_observation(
    pressure, failure_fields, outcome
):
    clock = _RecoveryClock()
    immediate = _valid_observation("2026-08-09T00:00:00+00:00")
    immediate["environment_status"]["memory_pressure"] = pressure

    result = wait_for_post_exit_memory_recovery(
        immediate_observation=immediate,
        immediate_started_at=clock(),
        recovery_policy=_valid_recovery_policy(build_canonical_workload()),
        collect=lambda _deadline: pytest.fail(
            "terminal immediate observation must not collect"
        ),
        classify_nonmemory=lambda observation: tuple(failure_fields),
        record_sample=lambda *args: pytest.fail(
            "terminal immediate observation must not sample"
        ),
        clock=clock,
        sleep=clock.sleep,
    )

    assert result == PostExitMemoryRecoveryResult(outcome, 0.0, (), failure_fields)
    assert clock.sleeps == []


def test_post_exit_recovery_requires_thirty_continuous_normal_seconds():
    clock = _RecoveryClock()
    immediate = _valid_observation("2026-08-09T00:00:00+00:00")
    immediate["environment_status"]["memory_pressure"] = "warning"
    observations = [
        _valid_observation(f"2026-08-09T00:00:{second:02d}+00:00")
        for second in (5, 10, 15, 20, 25, 30, 35)
    ]
    iter_observations = iter(observations)
    persisted = []

    result = wait_for_post_exit_memory_recovery(
        immediate_observation=immediate,
        immediate_started_at=0.0,
        recovery_policy=_valid_recovery_policy(build_canonical_workload()),
        collect=lambda _deadline: (clock(), next(iter_observations)),
        classify_nonmemory=lambda observation: (),
        record_sample=lambda index, elapsed, observation, previous: (
            persisted.append(
                {
                    "sample_index": index,
                    "elapsed_seconds": elapsed,
                    "identity": str(index),
                }
            )
            or persisted[-1]
        ),
        clock=clock,
        sleep=clock.sleep,
    )

    assert result.outcome == "recovered"
    assert result.duration_seconds == 35.0
    assert len(result.samples) == 7
    assert clock.sleeps == [5.0] * 7


def _run_scripted_memory_recovery(*, pressures, failure_fields=()):
    clock = _RecoveryClock()
    immediate = _valid_observation("2026-08-09T00:00:00+00:00")
    immediate["environment_status"]["memory_pressure"] = "warning"
    scripted = []
    for index, pressure in enumerate(pressures, 1):
        observation = _valid_observation(
            f"2026-08-09T00:00:{min(index * 5, 59):02d}+00:00"
        )
        observation["environment_status"]["memory_pressure"] = pressure
        scripted.append(observation)
    iter_scripted = iter(scripted)
    persisted = []

    result = wait_for_post_exit_memory_recovery(
        immediate_observation=immediate,
        immediate_started_at=0.0,
        recovery_policy=_valid_recovery_policy(build_canonical_workload()),
        collect=lambda _deadline: (clock(), next(iter_scripted)),
        classify_nonmemory=lambda observation: (
            () if observation is immediate else tuple(failure_fields)
        ),
        record_sample=lambda index, elapsed, observation, previous: (
            persisted.append(
                {
                    "sample_index": index,
                    "elapsed_seconds": elapsed,
                    "identity": f"sample-{index}",
                }
            )
            or persisted[-1]
        ),
        clock=clock,
        sleep=clock.sleep,
    )
    return result, persisted


def test_post_exit_recovery_warning_resets_the_normal_stability_window():
    result, persisted = _run_scripted_memory_recovery(
        pressures=("normal", "normal", "normal", "warning") + ("normal",) * 7
    )

    assert result.outcome == "recovered"
    assert result.duration_seconds == 55.0
    assert len(persisted) == 11


def test_post_exit_recovery_records_the_deadline_sample_without_extending_deadline():
    pressures = tuple(
        "warning" if elapsed % 30 == 25 else "normal" for elapsed in range(5, 301, 5)
    )

    result, persisted = _run_scripted_memory_recovery(pressures=pressures)

    assert result.outcome == "timeout"
    assert result.duration_seconds == 300.0
    assert len(persisted) == 60
    assert persisted[-1]["elapsed_seconds"] == 300.0


def test_post_exit_recovery_first_wake_overshoot_finalizes_zero_sample_timeout():
    clock = _RecoveryClock()
    immediate = _valid_observation("2026-08-09T00:00:00+00:00")
    immediate["environment_status"]["memory_pressure"] = "warning"

    def oversleep(_seconds):
        clock.now = 300.001

    result = wait_for_post_exit_memory_recovery(
        immediate_observation=immediate,
        immediate_started_at=0.0,
        recovery_policy=_valid_recovery_policy(build_canonical_workload()),
        collect=lambda _deadline: pytest.fail("deadline overshoot must not collect"),
        classify_nonmemory=lambda observation: (),
        record_sample=lambda *args: pytest.fail("deadline overshoot must not persist"),
        clock=clock,
        sleep=oversleep,
    )
    measurement = _valid_child_measurement(build_canonical_workload())
    post_exit = build_post_exit_observation(measurement=measurement, **immediate)
    recovery = build_post_exit_recovery(
        measurement=measurement,
        post_exit=post_exit,
        samples=result.samples,
        policy=post_exit_recovery_policy(build_canonical_workload()),
        outcome=result.outcome,
        duration_seconds=result.duration_seconds,
        completion_source="live",
    )
    trial = finalize_raw_trial(measurement, post_exit, result.samples, recovery)

    assert result == PostExitMemoryRecoveryResult("timeout", 300.0, (), ())
    assert recovery["terminal_sample_identity"] is None
    validate_raw_trial_evidence(trial)


@pytest.mark.parametrize(
    ("terminal_pressure", "failure_fields", "expected"),
    (
        ("critical", (), "critical"),
        ("normal", ("power_connected",), "environment-failure"),
        ("normal", ("thermal_state",), "environment-failure"),
        ("normal", ("hardware",), "environment-failure"),
        ("normal", ("software_versions",), "environment-failure"),
    ),
)
def test_post_exit_recovery_stops_on_terminal_failure(
    terminal_pressure, failure_fields, expected
):
    result, persisted = _run_scripted_memory_recovery(
        pressures=(terminal_pressure,), failure_fields=failure_fields
    )

    assert result.outcome == expected
    assert len(persisted) == 1
    assert result.failure_fields == failure_fields


def test_post_exit_recovery_persists_before_classifying_a_sample():
    clock = _RecoveryClock()
    immediate = _valid_observation("2026-08-09T00:00:00+00:00")
    immediate["environment_status"]["memory_pressure"] = "warning"
    observation = _valid_observation("2026-08-09T00:00:05+00:00")
    persisted = []

    def classify_nonmemory(candidate):
        if candidate is immediate:
            return ()
        assert persisted == [observation]
        return ("power_connected",)

    result = wait_for_post_exit_memory_recovery(
        immediate_observation=immediate,
        immediate_started_at=clock(),
        recovery_policy=_valid_recovery_policy(build_canonical_workload()),
        collect=lambda _deadline: (clock(), observation),
        classify_nonmemory=classify_nonmemory,
        record_sample=lambda index, elapsed, candidate, previous: (
            persisted.append(candidate) or {"identity": "sample-0"}
        ),
        clock=clock,
        sleep=clock.sleep,
    )

    assert result.outcome == "environment-failure"
    assert result.samples == ({"identity": "sample-0"},)


def test_post_exit_recovery_does_not_overlap_a_slow_collector():
    clock = _RecoveryClock()
    immediate = _valid_observation("2026-08-09T00:00:00+00:00")
    immediate["environment_status"]["memory_pressure"] = "warning"
    pressures = iter(("normal", "critical"))
    collection_starts = []

    def collect(_deadline):
        sample_started_at = clock()
        collection_starts.append(sample_started_at)
        clock.now += 7.0
        observation = _valid_observation("2026-08-09T00:00:05+00:00")
        observation["environment_status"]["memory_pressure"] = next(pressures)
        return sample_started_at, observation

    result = wait_for_post_exit_memory_recovery(
        immediate_observation=immediate,
        immediate_started_at=clock(),
        recovery_policy=_valid_recovery_policy(build_canonical_workload()),
        collect=collect,
        classify_nonmemory=lambda observation: (),
        record_sample=lambda index, elapsed, observation, previous: {
            "identity": f"sample-{index}"
        },
        clock=clock,
        sleep=clock.sleep,
    )

    assert result.outcome == "critical"
    assert collection_starts == [5.0, 12.0]
    assert clock.sleeps == [5.0, 0.0]


def test_post_exit_recovery_uses_memory_probe_start_as_sample_time():
    clock = _RecoveryClock()
    immediate = _valid_observation("2026-08-09T00:00:00+00:00")
    immediate["environment_status"]["memory_pressure"] = "warning"
    pressures = iter(("normal", "critical"))
    persisted_elapsed = []

    def collect(_deadline):
        clock.now += 1.0
        observation = _valid_observation("2026-08-09T00:00:05+00:00")
        observation["environment_status"]["memory_pressure"] = next(pressures)
        return clock(), observation

    result = wait_for_post_exit_memory_recovery(
        immediate_observation=immediate,
        immediate_started_at=0.0,
        recovery_policy=_valid_recovery_policy(build_canonical_workload()),
        collect=collect,
        classify_nonmemory=lambda observation: (),
        record_sample=lambda index, elapsed, observation, previous: (
            persisted_elapsed.append(elapsed) or {"identity": f"sample-{index}"}
        ),
        clock=clock,
        sleep=clock.sleep,
    )

    assert result.outcome == "critical"
    assert persisted_elapsed == [6.0, 12.0]
    assert clock.sleeps == [5.0, 5.0]


@pytest.mark.parametrize(
    ("policy_change", "message"),
    (
        (("sample_interval_seconds", 0.0), "sample interval"),
        (("timeout_seconds", float("inf")), "timeout"),
        (("stability_seconds", -1.0), "stability"),
        (("stability_seconds", 301.0), "must not exceed"),
    ),
)
def test_post_exit_recovery_validates_time_policy(policy_change, message):
    policy = _valid_recovery_policy(build_canonical_workload())
    policy[policy_change[0]] = policy_change[1]

    with pytest.raises(ValueError, match=message):
        wait_for_post_exit_memory_recovery(
            immediate_observation=_valid_observation("2026-08-09T00:00:00+00:00"),
            immediate_started_at=0.0,
            recovery_policy=policy,
            collect=lambda _deadline: pytest.fail("invalid policy must not collect"),
            classify_nonmemory=lambda observation: (),
            record_sample=lambda *args: pytest.fail("invalid policy must not sample"),
            clock=lambda: 0.0,
            sleep=lambda seconds: pytest.fail("invalid policy must not sleep"),
        )


@pytest.mark.parametrize("started_at", (-1.0, float("inf"), float("nan")))
def test_post_exit_recovery_validates_immediate_start_time(started_at):
    with pytest.raises(ValueError, match="immediate_started_at"):
        wait_for_post_exit_memory_recovery(
            immediate_observation=_valid_observation("2026-08-09T00:00:00+00:00"),
            immediate_started_at=started_at,
            recovery_policy=_valid_recovery_policy(build_canonical_workload()),
            collect=lambda _deadline: pytest.fail(
                "invalid start time must not collect"
            ),
            classify_nonmemory=lambda observation: (),
            record_sample=lambda *args: pytest.fail(
                "invalid start time must not sample"
            ),
            clock=lambda: 0.0,
            sleep=lambda seconds: pytest.fail("invalid start time must not sleep"),
        )


def test_recovery_import_does_not_eagerly_import_runner():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import v2.benchmarks.recovery; "
                "assert 'v2.benchmarks.runner' not in sys.modules"
            ),
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
