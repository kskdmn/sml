from __future__ import annotations

import dataclasses
import hashlib
import shutil
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_map
from sml.model.config import ModelConfig
from sml.model.language_model import SMLLanguageModel
from sml.training.common import (
    LoaderConfig,
    OptimizerConfig,
    PretrainingConfig,
    TrainerState,
    build_weight_decay_tree,
    initialize_adam_state,
    initialize_base_parameter_state,
)

import v2.benchmarks.quality as quality_module
from v2.benchmarks.quality import (
    CANONICAL_STEPS,
    CHECKPOINT_STEPS,
    ParameterUpdateStatistics,
    PretrainingQualityCheckpoint,
    PretrainingQualityReport,
    PretrainingQualityWorkload,
    build_pretraining_quality_workload,
    decide_pretraining_quality,
    harness_content_identity,
    real_work_identity,
    validate_pretraining_quality_records,
)

ROOT = Path(__file__).parents[3]


def test_quality_gate_requires_fp32_master_evidence_and_oracle_bound():
    passing = PretrainingQualityReport(
        candidate_validation_nll=2.01,
        oracle_validation_nll=2.00,
        candidate_finite=True,
        oracle_finite=True,
        rms_norm_master_moved=True,
        sub_bf16_update_survived=True,
        matching_work_identity=True,
    )

    assert decide_pretraining_quality(passing) == "pass"
    assert (
        decide_pretraining_quality(
            replace(passing, sub_bf16_update_survived=False),
        )
        == "fail"
    )


@pytest.mark.parametrize(
    ("change", "value"),
    [
        ("candidate_finite", False),
        ("oracle_finite", False),
        ("rms_norm_master_moved", False),
        ("matching_work_identity", False),
        ("candidate_validation_nll", float("nan")),
        ("oracle_validation_nll", float("inf")),
    ],
)
def test_quality_gate_fails_closed_for_every_acceptance_input(change, value):
    passing = PretrainingQualityReport(2.02, 2.0, True, True, True, True, True)

    assert decide_pretraining_quality(replace(passing, **{change: value})) == "fail"


@pytest.fixture(scope="module")
def canonical_workload() -> PretrainingQualityWorkload:
    return build_pretraining_quality_workload(ROOT)


def test_workload_binds_checked_in_source_disjoint_rows_and_exact_work(
    canonical_workload,
):
    workload = canonical_workload

    assert workload.checkpoint_steps == CHECKPOINT_STEPS == (0, 10, 100, 1_000)
    assert workload.model == dataclasses.asdict(ModelConfig())
    assert workload.optimizer == dataclasses.asdict(OptimizerConfig())
    assert workload.optimizer["schedule_steps"] == 268_000
    assert workload.loader == dataclasses.asdict(LoaderConfig())
    assert len(workload.ordered_batches) == (
        CANONICAL_STEPS * LoaderConfig().gradient_accumulation_steps
    )
    assert workload.training_fixture.logical_path.endswith("train-v1.npy")
    assert workload.validation_fixture.logical_path.endswith("validation-v1.npy")
    assert workload.training_fixture.shape == (32, 1_025)
    assert workload.validation_fixture.shape == (8, 1_025)
    assert workload.training_fixture.dtype == "int32"
    assert workload.validation_fixture.dtype == "int32"
    assert workload.training_fixture.source_identity != (
        workload.validation_fixture.source_identity
    )
    assert workload.training_fixture.semantic_identity != (
        workload.validation_fixture.semantic_identity
    )
    assert workload.ordered_batches[:3] == ((0,), (1,), (2,))
    assert workload.ordered_batches[31:34] == ((31,), (0,), (1,))
    assert workload.evaluation_row_indices == tuple(range(8))
    assert workload.identity == workload.recompute_identity()
    assert (
        PretrainingQualityWorkload.from_dict(workload.to_dict()).to_dict()
        == workload.to_dict()
    )

    for fixture in (workload.training_fixture, workload.validation_fixture):
        path = ROOT / fixture.logical_path
        assert path.stat().st_size == fixture.byte_size
        rows = np.load(path, allow_pickle=False)
        assert rows.dtype == np.dtype("<i4")
        assert tuple(rows.shape) == fixture.shape
        assert int(rows.min()) >= 0
        assert int(rows.max()) < ModelConfig().vocab_size


def test_harness_identity_hashes_only_the_two_reviewed_files_in_order():
    expected = hashlib.sha256()
    for relative in (
        Path("v2/benchmarks/quality.py"),
        Path("v2/tests/unit/test_pretraining_quality.py"),
    ):
        expected.update((ROOT / relative).read_bytes())

    assert harness_content_identity(ROOT) == f"sha256:{expected.hexdigest()}"


def test_workload_rejects_a_validation_row_copied_from_training(tmp_path):
    for relative in (
        Path("v2/benchmarks/quality.py"),
        Path("v2/tests/unit/test_pretraining_quality.py"),
        Path("v2/benchmarks/fixtures/pretraining-quality-train-v1.npy"),
        Path("v2/benchmarks/fixtures/pretraining-quality-validation-v1.npy"),
    ):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    training = np.load(
        tmp_path / "v2/benchmarks/fixtures/pretraining-quality-train-v1.npy",
        allow_pickle=False,
    )
    validation_path = (
        tmp_path / "v2/benchmarks/fixtures/pretraining-quality-validation-v1.npy"
    )
    validation = np.load(validation_path, allow_pickle=False)
    validation[0] = training[0]
    np.save(validation_path, validation, allow_pickle=False)

    with pytest.raises(ValueError, match="source-disjoint"):
        build_pretraining_quality_workload(tmp_path)


def test_fp32_oracle_is_real_fp32_compute_with_matching_dropout_key():
    config = ModelConfig(
        vocab_size=32,
        hidden_size=8,
        num_layers=1,
        num_q_heads=2,
        num_kv_heads=1,
        intermediate_size=16,
        original_context_length=4,
        hidden_dropout=0.2,
    )
    model = SMLLanguageModel(config, key=mx.random.key(7))
    bf16_parameters = model.parameters()
    fp32_parameters = tree_map(lambda value: value.astype(mx.float32), bf16_parameters)
    rows = mx.array([[1, 4, 5, 2]], dtype=mx.int32)
    key = mx.random.key(11)

    candidate_logits, candidate_cache, candidate_key = model.forward_arrays(
        bf16_parameters,
        rows,
        attention_mask=None,
        positions=None,
        cache_state=None,
        training=True,
        key=key,
    )
    oracle_logits, oracle_key = quality_module._fp32_forward_arrays(
        model,
        fp32_parameters,
        rows,
        training=True,
        key=key,
    )
    mx.eval(candidate_logits, oracle_logits, candidate_key, oracle_key)

    assert candidate_cache is None
    assert candidate_logits.dtype == mx.bfloat16
    assert oracle_logits.dtype == mx.float32
    assert bool(mx.array_equal(candidate_key, oracle_key))
    assert "lm_head" not in fp32_parameters
    assert bool(mx.all(mx.isfinite(oracle_logits)).item())


def _tiny_runtime(tmp_path: Path):
    model_config = ModelConfig(
        vocab_size=32,
        hidden_size=8,
        num_layers=1,
        num_q_heads=2,
        num_kv_heads=1,
        intermediate_size=16,
        original_context_length=4,
        hidden_dropout=0.2,
    )
    config = PretrainingConfig(
        data=tmp_path / "data",
        output_run=tmp_path / "run",
        model=model_config,
        optimizer=OptimizerConfig(
            learning_rate=0.01,
            beta1=0.5,
            beta2=0.5,
            schedule_steps=2,
            warmup_steps=0,
        ),
        loader=LoaderConfig(gradient_accumulation_steps=1),
        maximum_steps=2,
        maximum_epochs=None,
        compile=False,
    )
    model_key, trainer_key = mx.random.split(mx.random.key(13))
    model = SMLLanguageModel(model_config, key=model_key)
    parameters = initialize_base_parameter_state(model.parameters())
    optimizer = initialize_adam_state(parameters.master_parameters)
    trainer = TrainerState(
        accumulators=tree_map(mx.zeros_like, parameters.master_parameters),
        accumulation_count=mx.array(0, dtype=mx.int32),
        next_key=trainer_key,
        loss_numerator=mx.array(0.0, dtype=mx.float32),
    )
    decay = build_weight_decay_tree(
        parameters.working_parameters,
        config.optimizer.weight_decay,
    )
    return config, model, parameters, optimizer, trainer, decay


def test_candidate_transition_is_the_reviewed_production_kernel(tmp_path):
    config, model, parameters, optimizer, trainer, decay = _tiny_runtime(tmp_path)
    rows = mx.array([[1, 4, 5, 2, 6]], dtype=mx.int32)
    candidate = quality_module._build_candidate_kernels(model, config, decay)
    production = quality_module.build_pretraining_kernels(model, config, decay)

    candidate_working, candidate_trainer = candidate.microstep_core(
        parameters.working_parameters,
        trainer.to_tree(),
        rows[:, :-1],
        rows[:, 1:],
    )
    expected_working, expected_trainer = production.eager_microstep_core(
        parameters.working_parameters,
        trainer.to_tree(),
        rows[:, :-1],
        rows[:, 1:],
    )
    actual = candidate.optimizer_step_core(
        parameters.master_parameters,
        candidate_working,
        optimizer.to_tree(),
        candidate_trainer,
    )
    expected = production.eager_optimizer_step_core(
        parameters.master_parameters,
        expected_working,
        optimizer.to_tree(),
        expected_trainer,
    )
    mx.eval(actual, expected)

    for (_, actual_leaf), (_, expected_leaf) in zip(
        quality_module.tree_flatten(actual),
        quality_module.tree_flatten(expected),
        strict=True,
    ):
        assert bool(mx.array_equal(actual_leaf, expected_leaf))


def test_oracle_transition_keeps_fp32_working_state_and_matching_key(tmp_path):
    config, model, parameters, optimizer, trainer, decay = _tiny_runtime(tmp_path)
    rows = mx.array([[1, 4, 5, 2, 6]], dtype=mx.int32)
    candidate = quality_module._build_candidate_kernels(model, config, decay)
    oracle = quality_module._build_oracle_kernels(model, config, decay)

    candidate_working, candidate_trainer = candidate.microstep_core(
        parameters.working_parameters,
        trainer.to_tree(),
        rows[:, :-1],
        rows[:, 1:],
    )
    oracle_working, oracle_trainer = oracle.microstep_core(
        parameters.master_parameters,
        trainer.to_tree(),
        rows[:, :-1],
        rows[:, 1:],
    )
    oracle_result = oracle.optimizer_step_core(
        parameters.master_parameters,
        oracle_working,
        optimizer.to_tree(),
        oracle_trainer,
    )
    mx.eval(candidate_working, candidate_trainer, oracle_result)

    assert bool(mx.array_equal(candidate_trainer[2], oracle_trainer[2]))
    assert int(oracle_result[2][0].item()) == 1
    for (_master_path, master), (_working_path, working) in zip(
        quality_module.tree_flatten(oracle_result[0]),
        quality_module.tree_flatten(oracle_result[1]),
        strict=True,
    ):
        assert working.dtype == mx.float32
        assert bool(mx.array_equal(master, working))


def test_update_statistics_prove_sub_bf16_ulp_master_survival():
    previous_master = {"norm": {"weight": mx.array([1.0], dtype=mx.float32)}}
    previous_working = tree_map(
        lambda value: value.astype(mx.bfloat16), previous_master
    )
    updated_master = {"norm": {"weight": mx.array([1.0001], dtype=mx.float32)}}
    updated_working = tree_map(lambda value: value.astype(mx.bfloat16), updated_master)

    observation = quality_module._observe_update(
        previous_master,
        updated_master,
        previous_working,
        updated_working,
        {"norm.weight": mx.array(False)},
    )
    statistics, survived = quality_module._materialize_update_observation(observation)

    assert statistics == (
        ParameterUpdateStatistics(
            path="norm.weight",
            value_count=1,
            nonzero_update_count=1,
            sub_bf16_ulp_update_count=1,
            survived_sub_bf16_ulp_count=1,
            minimum_update_to_bf16_ulp=0.0128021240234375,
            mean_update_to_bf16_ulp=0.0128021240234375,
            maximum_update_to_bf16_ulp=0.0128021240234375,
        ),
    )
    assert survived is True


def _statistics(workload):
    return tuple(
        ParameterUpdateStatistics(path, 1, 1, 0, 0, 1.0, 1.0, 1.0)
        for path in workload.parameter_leaf_names
    )


def _checkpoint(workload, runtime, step, *, validation_nll, survived=True):
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
        master_parameter_identity="sha256:"
        + ("a" if step == 0 or runtime == "candidate" else "b") * 64,
        working_parameter_identity="sha256:"
        + ("c" if runtime == "candidate" else "d") * 64,
        trainer_key_identity="sha256:" + "e" * 64,
        train_loss=2.0,
        validation_nll=validation_nll,
        finite=True,
        update_statistics=_statistics(workload),
        changed_bf16_working_fraction=0.5 if step else 0.0,
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
    with pytest.raises(ValueError, match="step-zero master"):
        validate_pretraining_quality_records(
            canonical_workload,
            (*records[:4], forged_master, *records[5:]),
        )


def test_multi_step_runtime_submits_each_optimizer_boundary(monkeypatch, tmp_path):
    config, model, parameters, optimizer, trainer, decay = _tiny_runtime(tmp_path)
    kernels = quality_module._build_candidate_kernels(model, config, decay)
    rows = np.asarray([[1, 4, 5, 2, 6]], dtype=np.int32)
    state = quality_module._RuntimeLoopState(
        masters=parameters.master_parameters,
        working=parameters.working_parameters,
        adam_tree=optimizer.to_tree(),
        trainer_tree=trainer.to_tree(),
        cumulative_survival={
            path: mx.array(False, dtype=mx.bool_)
            for path, _value in quality_module.tree_flatten(
                parameters.master_parameters
            )
        },
        observations=(),
        metrics={"loss": mx.array(0.0, dtype=mx.float32)},
        microstep_index=0,
    )
    real_async_eval = mx.async_eval
    submissions = []

    def recording_async_eval(*trees):
        submissions.append(trees)
        real_async_eval(*trees)

    monkeypatch.setattr(mx, "async_eval", recording_async_eval)

    result = quality_module._execute_training_steps(
        kernels=kernels,
        runtime="candidate",
        gradient_accumulation_steps=1,
        ordered_batches=((0,), (0,), (0,)),
        training_rows=rows,
        start_step=0,
        stop_step=3,
        state=state,
    )
    mx.eval(result.masters, result.adam_tree, result.trainer_tree)

    assert len(submissions) == 3
    assert int(result.adam_tree[0].item()) == 3
    assert result.microstep_index == 3
    assert not bool(mx.array_equal(result.trainer_tree[2], trainer.next_key))


def test_manifest_fields_and_output_paths_fail_closed(canonical_workload, tmp_path):
    manifest = quality_module._manifest_document(
        workload=canonical_workload,
        source_commit="a" * 40,
        command="python -m v2.benchmarks.quality record --steps 1000",
        wall_time=10.0,
        runtime_times={"candidate": 4.0, "oracle": 6.0},
        peak_memory=123,
        raw_identity="sha256:" + "b" * 64,
    )

    assert quality_module._validate_manifest_fields(
        manifest, canonical_workload
    ) == dict(manifest)
    for name, value in (
        ("command", ""),
        ("runtime_wall_time_seconds", {"candidate": 4.0}),
        ("peak_metal_memory_bytes", -1),
        ("temporary_disk_bytes", 1),
        ("fixture_bytes", 1),
        ("training_cardinality", 1),
        ("validation_cardinality", 1),
        ("ordered_work_count", 1),
    ):
        forged = {**manifest, name: value}
        body = {key: item for key, item in forged.items() if key != "identity"}
        forged["identity"] = quality_module.structured_identity(
            "sml-pretraining-quality-manifest-v1", body
        )
        with pytest.raises(ValueError):
            quality_module._validate_manifest_fields(forged, canonical_workload)

    first = tmp_path / "manifest.json"
    second = tmp_path / "raw.jsonl"
    third = tmp_path / "report.json"
    quality_module._preflight_output_paths(first, second, third)
    with pytest.raises(ValueError, match="distinct"):
        quality_module._preflight_output_paths(first, first, third)
    second.write_text("occupied")
    with pytest.raises(FileExistsError, match="already exists"):
        quality_module._preflight_output_paths(first, second, third)


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
