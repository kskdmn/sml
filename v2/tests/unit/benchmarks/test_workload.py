import hashlib
from pathlib import Path

import numpy as np
import pytest

from v2.benchmarks.schema import METRIC_NAMES, CanonicalWorkload
from v2.benchmarks.workload import (
    BENCHMARK_CORPUS,
    HARNESS_COMPONENTS,
    build_canonical_workload,
    canonical_input_identity,
    canonical_metric_projection,
    fixed_canonical_rows,
    fixed_inference_requests,
    fixed_swag_examples,
    harness_content_identity,
    semantic_row_content_identity,
    structured_identity,
)


def test_canonical_workload_round_trip_pins_complete_benchmark_contract():
    workload = build_canonical_workload()

    assert CanonicalWorkload.from_dict(workload.to_dict()) == workload
    assert workload.model["vocab_size"] == 28_672
    assert workload.model["rope_scaling_factor"] == 1.0
    assert workload.optimizer["gradient_accumulation_steps"] == 8
    assert workload.precision == {
        "compute_dtype": "bfloat16",
        "master_parameter_dtype": "float32",
        "working_parameter_dtype": "bfloat16",
        "moment_dtype": "float32",
    }
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
        == "sha256:6aee56e33d6ba7fd280f7512d843d2f45dd326fc66f5b3277bacb38669af853d"
    )


@pytest.mark.parametrize("version", (0, True, 1.0))
def test_canonical_workload_rejects_unsupported_versions_and_numeric_aliases(version):
    raw = build_canonical_workload().to_dict()
    raw["schema_version"] = version

    with pytest.raises(ValueError, match="schema version"):
        CanonicalWorkload.from_dict(raw)


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


def test_harness_identity_hashes_every_component_in_fixed_order(tmp_path: Path):
    checkout = Path(__file__).resolve().parents[4]
    assert len(HARNESS_COMPONENTS) == len(set(HARNESS_COMPONENTS))
    assert all((checkout / component).is_file() for component in HARNESS_COMPONENTS)
    expected = hashlib.sha256()
    for index, relative_path in enumerate(HARNESS_COMPONENTS):
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = f"component-{index}\n".encode()
        path.write_bytes(payload)
        expected.update(payload)

    assert harness_content_identity(tmp_path) == f"sha256:{expected.hexdigest()}"


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
