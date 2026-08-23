from __future__ import annotations

import dataclasses
import hashlib
import os
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

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
    ParameterLeafSpec,
    ParameterUpdateStatistics,
    PretrainingQualityCheckpoint,
    PretrainingQualityReport,
    PretrainingQualityWorkload,
    build_pretraining_quality_workload,
    decide_pretraining_quality,
    harness_content_identity,
    production_dependency_components,
    production_dependency_content_identity,
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
    assert tuple(leaf.path for leaf in workload.parameter_leaves) == (
        workload.parameter_leaf_names
    )
    assert all(
        leaf.value_count == int(np.prod(leaf.shape))
        and leaf.initial_bf16_identity.startswith("sha256:")
        and leaf.initial_fp32_identity.startswith("sha256:")
        for leaf in workload.parameter_leaves
    )
    assert all(
        isinstance(leaf, ParameterLeafSpec) for leaf in workload.parameter_leaves
    )
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


def test_workload_separately_binds_the_complete_sml_production_source_tree(
    canonical_workload,
    tmp_path,
):
    expected_components = tuple(
        path.as_posix()
        for path in sorted(
            {
                Path("v2/src/sml.py"),
                Path("v2/benchmarks/schema.py"),
                Path("v2/benchmarks/workload.py"),
                *(
                    path.relative_to(ROOT)
                    for path in (ROOT / "v2/src/sml").rglob("*.py")
                ),
            },
            key=lambda path: path.as_posix(),
        )
    )
    assert canonical_workload.production_dependency_components == expected_components
    assert (
        tuple(path.as_posix() for path in production_dependency_components(ROOT))
        == expected_components
    )
    assert canonical_workload.production_dependency_identity == (
        production_dependency_content_identity(ROOT)
    )

    for relative in (
        *quality_module.HARNESS_COMPONENTS,
        *(Path(path) for path in expected_components),
    ):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    original_harness = harness_content_identity(tmp_path)
    dependency = tmp_path / "v2/src/sml/artifacts/manifest.py"
    dependency.write_bytes(dependency.read_bytes() + b"\n# provenance mutation\n")

    assert harness_content_identity(tmp_path) == original_harness
    assert production_dependency_content_identity(tmp_path) != (
        canonical_workload.production_dependency_identity
    )


@pytest.mark.parametrize(
    "source_mutation",
    ["content", "executable_mode"],
)
def test_re_signed_descendant_cannot_reuse_evidence_after_artifact_source_change(
    canonical_workload,
    tmp_path,
    source_mutation,
):
    changed_source = Path("v2/src/sml/artifacts/manifest.py")
    fixtures = (
        Path(canonical_workload.training_fixture.logical_path),
        Path(canonical_workload.validation_fixture.logical_path),
    )
    copied = {
        *quality_module.HARNESS_COMPONENTS,
        *(Path(path) for path in canonical_workload.production_dependency_components),
        *fixtures,
        changed_source,
    }
    for relative in copied:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)

    def git(*arguments):
        return subprocess.run(
            ["git", *arguments],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    git("init", "--quiet")
    git("config", "user.name", "Quality Provenance Test")
    git("config", "user.email", "quality-provenance@example.invalid")
    git("add", ".")
    git("commit", "--quiet", "-m", "quality source")
    changed_path = tmp_path / changed_source
    if source_mutation == "content":
        changed_path.write_bytes(
            changed_path.read_bytes() + b"\n# descendant mutation\n"
        )
    else:
        changed_path.chmod(changed_path.stat().st_mode | 0o111)
    git("add", changed_source.as_posix())
    git("commit", "--quiet", "-m", "change executed artifact source")
    descendant = git("rev-parse", "HEAD")

    destinations = quality_module._canonical_evidence_destinations(
        tmp_path,
        tmp_path / quality_module.CANONICAL_MANIFEST_PATH,
        tmp_path / quality_module.CANONICAL_RAW_PATH,
        tmp_path / quality_module.CANONICAL_REPORT_PATH,
    )
    command = quality_module._recording_command_document(tmp_path, destinations)
    session_identity = quality_module._recording_session_identity(
        descendant, canonical_workload.identity, command
    )
    re_signed = quality_module._manifest_document(
        workload=canonical_workload,
        source_commit=descendant,
        recording_command=command,
        phase_times={
            "setup": 1.0,
            "candidate": 2.0,
            "oracle": 3.0,
            "validation_serialization": 1.0,
        },
        peak_memory=1,
        raw_identity="sha256:" + "1" * 64,
        raw_file_identity="sha256:" + "2" * 64,
        raw_bytes=100,
        report_identity="sha256:" + "3" * 64,
        report_file_identity="sha256:" + "4" * 64,
        report_bytes=100,
        recording_session_identity=session_identity,
    )

    with pytest.raises(ValueError, match="production source tree"):
        quality_module._validate_manifest(
            re_signed,
            canonical_workload,
            tmp_path,
            command,
        )


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
        {"norm.weight": mx.array(0, dtype=mx.int32)},
        step=1,
    )
    statistics, survived = quality_module._materialize_update_observation(observation)

    assert len(statistics) == 1
    statistic = statistics[0]
    assert statistic.path == "norm.weight"
    assert statistic.shape == (1,)
    assert statistic.value_count == 1
    assert statistic.nonzero_update_count == 1
    assert statistic.sub_bf16_ulp_update_count == 1
    assert statistic.survived_sub_bf16_ulp_count == 1
    assert statistic.changed_bf16_working_count == 0
    assert statistic.first_sub_bf16_survival_step == 1
    assert statistic.minimum_update_to_bf16_ulp == 0.0128021240234375
    assert statistic.mean_update_to_bf16_ulp == 0.0128021240234375
    assert statistic.maximum_update_to_bf16_ulp == 0.0128021240234375
    assert statistic.before_master_identity != statistic.after_master_identity
    assert statistic.before_bf16_working_identity == (
        statistic.after_bf16_working_identity
    )
    assert survived is True


def _fake_identity(*parts: object) -> str:
    digest = hashlib.sha256("/".join(map(str, parts)).encode()).hexdigest()
    return f"sha256:{digest}"


def _statistics(workload, runtime, step, survived):
    statistics = []
    survival_path = workload.parameter_leaves[0].path
    for leaf in workload.parameter_leaves:
        if step == 0:
            before_master = after_master = leaf.initial_fp32_identity
            before_bf16 = after_bf16 = leaf.initial_bf16_identity
            before_working = after_working = (
                before_bf16 if runtime == "candidate" else before_master
            )
            nonzero = sub_ulp = survived_count = changed = 0
            first_survival = None
            minimum = mean = maximum = None
        else:
            before_master = _fake_identity(runtime, step, leaf.path, "before-master")
            after_master = _fake_identity(runtime, step, leaf.path, "after-master")
            before_bf16 = _fake_identity(runtime, step, leaf.path, "before-bf16")
            after_bf16 = _fake_identity(runtime, step, leaf.path, "after-bf16")
            before_working = before_bf16 if runtime == "candidate" else before_master
            after_working = after_bf16 if runtime == "candidate" else after_master
            is_survival_leaf = survived and leaf.path == survival_path
            nonzero = 1
            sub_ulp = int(is_survival_leaf)
            survived_count = int(is_survival_leaf)
            changed = 1
            first_survival = 1 if is_survival_leaf else None
            minimum = mean = maximum = 0.5 if is_survival_leaf else 1.0
        statistics.append(
            ParameterUpdateStatistics(
                path=leaf.path,
                shape=leaf.shape,
                value_count=leaf.value_count,
                before_master_identity=before_master,
                after_master_identity=after_master,
                before_working_identity=before_working,
                after_working_identity=after_working,
                before_bf16_working_identity=before_bf16,
                after_bf16_working_identity=after_bf16,
                nonzero_update_count=nonzero,
                sub_bf16_ulp_update_count=sub_ulp,
                survived_sub_bf16_ulp_count=survived_count,
                changed_bf16_working_count=changed,
                first_sub_bf16_survival_step=first_survival,
                minimum_update_to_bf16_ulp=minimum,
                mean_update_to_bf16_ulp=mean,
                maximum_update_to_bf16_ulp=maximum,
            )
        )
    return tuple(statistics)


def _checkpoint(workload, runtime, step, *, validation_nll, survived=True):
    statistics = _statistics(workload, runtime, step, survived)
    changed_fraction = sum(
        item.changed_bf16_working_count for item in statistics
    ) / sum(item.value_count for item in statistics)
    finite_state_value_count = (
        5 * sum(leaf.value_count for leaf in workload.parameter_leaves) + 1
    )
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
        master_parameter_identity=quality_module._telemetry_tree_identity(
            quality_module._MASTER_TREE_IDENTITY_DOMAIN,
            statistics,
            "after_master_identity",
        ),
        working_parameter_identity=quality_module._telemetry_tree_identity(
            quality_module._WORKING_TREE_IDENTITY_DOMAIN,
            statistics,
            "after_working_identity",
        ),
        trainer_key_identity="sha256:" + "e" * 64,
        train_loss=2.0,
        validation_nll=validation_nll,
        finite_state_value_count=finite_state_value_count,
        nonfinite_state_value_count=0,
        finite=True,
        update_statistics=statistics,
        changed_bf16_working_fraction=changed_fraction,
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
    with pytest.raises(ValueError, match="master tree identity|step-zero master"):
        validate_pretraining_quality_records(
            canonical_workload,
            (*records[:4], forged_master, *records[5:]),
        )


def test_raw_validation_reconstructs_every_accepting_telemetry_summary(
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

    def resign_record(index, **changes):
        forged = replace(records[index], **changes)
        return replace(forged, identity=forged.recompute_identity())

    step_zero_statistic = records[0].update_statistics[0]
    bad_step_zero_statistic = replace(
        step_zero_statistic,
        nonzero_update_count=1,
        minimum_update_to_bf16_ulp=1.0,
        mean_update_to_bf16_ulp=1.0,
        maximum_update_to_bf16_ulp=1.0,
    )
    forged_step_zero = resign_record(
        0,
        update_statistics=(
            bad_step_zero_statistic,
            *records[0].update_statistics[1:],
        ),
    )
    with pytest.raises(ValueError, match="step-zero|before/after"):
        validate_pretraining_quality_records(
            canonical_workload, (forged_step_zero, *records[1:])
        )

    bad_shape = replace(
        records[1].update_statistics[0],
        shape=(records[1].update_statistics[0].shape[0] + 1,),
    )
    forged_shape = resign_record(
        1,
        update_statistics=(bad_shape, *records[1].update_statistics[1:]),
    )
    with pytest.raises(ValueError, match="shape|value count"):
        validate_pretraining_quality_records(
            canonical_workload, (records[0], forged_shape, *records[2:])
        )

    forged_fraction = resign_record(1, changed_bf16_working_fraction=0.75)
    with pytest.raises(ValueError, match="changed working fraction"):
        validate_pretraining_quality_records(
            canonical_workload, (records[0], forged_fraction, *records[2:])
        )

    forged_movement = resign_record(1, rms_norm_master_moved=False)
    with pytest.raises(ValueError, match="RMSNorm master movement"):
        validate_pretraining_quality_records(
            canonical_workload, (records[0], forged_movement, *records[2:])
        )

    forged_survival = resign_record(1, sub_bf16_update_survived=False)
    with pytest.raises(ValueError, match="sub-BF16 update survival"):
        validate_pretraining_quality_records(
            canonical_workload, (records[0], forged_survival, *records[2:])
        )

    forged_finite = resign_record(1, finite=False)
    with pytest.raises(ValueError, match="finite state status"):
        validate_pretraining_quality_records(
            canonical_workload, (records[0], forged_finite, *records[2:])
        )
    forged_nonfinite_count = resign_record(1, nonfinite_state_value_count=1)
    with pytest.raises(ValueError, match="finite state status"):
        validate_pretraining_quality_records(
            canonical_workload, (records[0], forged_nonfinite_count, *records[2:])
        )

    changed_statistic = replace(
        records[1].update_statistics[0], changed_bf16_working_count=0
    )
    changed_statistics = (changed_statistic, *records[1].update_statistics[1:])
    changed_fraction = sum(
        item.changed_bf16_working_count for item in changed_statistics
    ) / sum(item.value_count for item in changed_statistics)
    forged_before_after = resign_record(
        1,
        update_statistics=changed_statistics,
        changed_bf16_working_fraction=changed_fraction,
    )
    with pytest.raises(ValueError, match="BF16 working identities"):
        validate_pretraining_quality_records(
            canonical_workload,
            (records[0], forged_before_after, *records[2:]),
        )

    survival_statistic = replace(
        records[1].update_statistics[0], first_sub_bf16_survival_step=None
    )
    forged_cumulative = resign_record(
        1,
        update_statistics=(survival_statistic, *records[1].update_statistics[1:]),
        sub_bf16_update_survived=False,
    )
    with pytest.raises(ValueError, match="cumulative sub-BF16"):
        validate_pretraining_quality_records(
            canonical_workload, (records[0], forged_cumulative, *records[2:])
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
    destinations = quality_module._canonical_evidence_destinations(
        ROOT,
        ROOT / "v2/benchmarks/manifests/pretraining-quality-v1.json",
        ROOT / "v2/benchmarks/results/pretraining-quality-v1.jsonl",
        ROOT / "v2/benchmarks/results/pretraining-quality-v1.json",
    )
    command = quality_module._recording_command_document(ROOT, destinations)
    session_identity = quality_module._recording_session_identity(
        "a" * 40, canonical_workload.identity, command
    )
    manifest = quality_module._manifest_document(
        workload=canonical_workload,
        source_commit="a" * 40,
        recording_command=command,
        phase_times={
            "setup": 1.0,
            "candidate": 3.0,
            "oracle": 4.0,
            "validation_serialization": 2.0,
        },
        peak_memory=123,
        raw_identity="sha256:" + "b" * 64,
        raw_file_identity="sha256:" + "c" * 64,
        raw_bytes=1_000,
        report_identity="sha256:" + "d" * 64,
        report_file_identity="sha256:" + "e" * 64,
        report_bytes=500,
        recording_session_identity=session_identity,
    )

    assert quality_module._validate_manifest_fields(
        manifest, canonical_workload, command
    ) == dict(manifest)
    assert manifest["measurement_boundaries"] == (quality_module.MEASUREMENT_BOUNDARIES)
    assert manifest["measured_wall_time_seconds"] == 10.0
    assert manifest["temporary_disk_high_water_bytes"] == (
        sum(manifest["artifact_byte_sizes"].values())
        + manifest["publication_metadata_bytes"]
    )
    assert manifest["artifact_byte_sizes"]["manifest"] == len(
        quality_module.canonical_json_bytes(manifest)
    )
    changed_resolved_destinations = {
        **command["resolved_destinations"],
        "report": (tmp_path / "forged-report.json").resolve().as_posix(),
    }
    for name, value in (
        ("recording_command", {**command, "argv": ["record", "--steps", "10"]}),
        (
            "recording_command",
            {
                **command,
                "resolved_destinations": changed_resolved_destinations,
            },
        ),
        (
            "measurement_boundaries",
            {**quality_module.MEASUREMENT_BOUNDARIES, "clock": "wall-clock"},
        ),
        ("phase_wall_time_seconds", {"candidate": 10.0}),
        ("measured_wall_time_seconds", 9.0),
        ("peak_metal_memory_bytes", -1),
        (
            "temporary_disk_high_water_bytes",
            manifest["temporary_disk_high_water_bytes"] + 1,
        ),
        (
            "artifact_byte_sizes",
            {**manifest["artifact_byte_sizes"], "raw": 1},
        ),
        ("fixture_bytes", 1),
        ("training_cardinality", 1),
        ("validation_cardinality", 1),
        ("ordered_work_count", 1),
    ):
        forged = {**manifest, name: value}
        body = {key: item for key, item in forged.items() if key != "identity"}
        forged["identity"] = quality_module.structured_identity(
            "sml-pretraining-quality-manifest-v2", body
        )
        with pytest.raises(ValueError):
            quality_module._validate_manifest_fields(
                forged, canonical_workload, command
            )

    first = tmp_path / "manifest.json"
    second = tmp_path / "raw.jsonl"
    third = tmp_path / "report.json"
    quality_module._preflight_output_paths(first, second, third)
    with pytest.raises(ValueError, match="distinct"):
        quality_module._preflight_output_paths(first, first, third)
    second.write_text("occupied")
    with pytest.raises(FileExistsError, match="already exists"):
        quality_module._preflight_output_paths(first, second, third)
    with pytest.raises(ValueError, match="canonical evidence destinations"):
        quality_module._canonical_evidence_destinations(
            ROOT, first, tmp_path / "raw.jsonl", tmp_path / "report.json"
        )


def _test_publication(tmp_path):
    destinations = quality_module._EvidenceDestinations(
        manifest=tmp_path / "manifests" / "manifest.json",
        raw_output=tmp_path / "results" / "raw.jsonl",
        report=tmp_path / "results" / "report.json",
    )
    owner = quality_module._publication_owner_document(
        session_identity="sha256:" + "1" * 64,
        source_commit="a" * 40,
        workload_identity="sha256:" + "2" * 64,
        destinations=destinations,
    )
    recovery = tmp_path / "results" / ".quality-recording"
    publication = quality_module._prepare_evidence_publication(
        recovery, destinations, owner
    )
    payloads = {
        "manifest": b'{"artifact":"manifest"}',
        "raw": b'{"artifact":"raw"}\n',
        "report": b'{"artifact":"report"}',
    }
    quality_module._stage_evidence_publication(publication, payloads)
    return publication, payloads, owner


def test_evidence_publication_never_replaces_a_raced_target(tmp_path, monkeypatch):
    publication, payloads, _owner = _test_publication(tmp_path)
    real_link = os.link
    raced = False

    def race_first_target(source, destination, **kwargs):
        nonlocal raced
        destination = Path(destination)
        if destination == publication.destinations.raw_output and not raced:
            raced = True
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"unrelated-raced-content")
        return real_link(source, destination, **kwargs)

    monkeypatch.setattr(os, "link", race_first_target)

    with pytest.raises(FileExistsError, match="conflicting"):
        quality_module._publish_staged_evidence(publication)

    assert publication.destinations.raw_output.read_bytes() == (
        b"unrelated-raced-content"
    )
    assert not publication.destinations.manifest.exists()
    assert not publication.destinations.report.exists()
    assert publication.recovery_directory.is_dir()
    assert not (publication.recovery_directory / "completed.json").exists()
    assert payloads["raw"] != publication.destinations.raw_output.read_bytes()


def test_interrupted_owned_publication_resumes_and_fsyncs_completion_last(
    tmp_path,
    monkeypatch,
):
    publication, payloads, owner = _test_publication(tmp_path)
    real_link = os.link
    link_count = 0

    def interrupt_second_target(source, destination, **kwargs):
        nonlocal link_count
        destination = Path(destination)
        if destination.name != "completed.json":
            link_count += 1
            if link_count == 2:
                raise RuntimeError("injected publication interruption")
        return real_link(source, destination, **kwargs)

    monkeypatch.setattr(os, "link", interrupt_second_target)
    with pytest.raises(RuntimeError, match="injected publication interruption"):
        quality_module._publish_staged_evidence(publication)

    assert publication.destinations.raw_output.read_bytes() == payloads["raw"]
    assert publication.recovery_directory.is_dir()
    assert not (publication.recovery_directory / "completed.json").exists()

    monkeypatch.setattr(os, "link", real_link)
    real_fsync_directory = quality_module._fsync_directory
    fsynced = []

    def record_fsync(path):
        fsynced.append(Path(path).resolve())
        real_fsync_directory(path)

    monkeypatch.setattr(quality_module, "_fsync_directory", record_fsync)
    resumed = quality_module._prepare_evidence_publication(
        publication.recovery_directory,
        publication.destinations,
        owner,
    )
    quality_module._publish_staged_evidence(resumed)

    assert publication.destinations.raw_output.read_bytes() == payloads["raw"]
    assert publication.destinations.manifest.read_bytes() == payloads["manifest"]
    assert publication.destinations.report.read_bytes() == payloads["report"]
    assert not publication.recovery_directory.exists()
    assert publication.destinations.raw_output.parent.resolve() in fsynced
    assert publication.destinations.manifest.parent.resolve() in fsynced
    assert publication.recovery_directory.parent.resolve() in fsynced


@pytest.mark.parametrize("completed_stage_count", [1, 2, 3, 4, 5])
def test_interrupted_staging_prefix_is_identity_bound_and_safely_retryable(
    tmp_path,
    monkeypatch,
    completed_stage_count,
):
    publication, payloads, owner = _test_publication(tmp_path)
    quality_module._remove_owned_recovery(publication)
    publication = quality_module._prepare_evidence_publication(
        publication.recovery_directory,
        publication.destinations,
        owner,
    )
    real_stage = quality_module._stage_named_payload
    staged = 0

    def interrupt_stage(publication, name, payload):
        nonlocal staged
        real_stage(publication, name, payload)
        staged += 1
        if staged == completed_stage_count:
            raise RuntimeError("injected staging interruption")

    monkeypatch.setattr(quality_module, "_stage_named_payload", interrupt_stage)
    with pytest.raises(RuntimeError, match="injected staging interruption"):
        quality_module._stage_evidence_publication(publication, payloads)
    assert (publication.recovery_directory / "plan.payload").is_file()

    monkeypatch.setattr(quality_module, "_stage_named_payload", real_stage)
    retried = quality_module._prepare_evidence_publication(
        publication.recovery_directory,
        publication.destinations,
        owner,
    )
    assert retried.staged is (completed_stage_count == 5)
    if not retried.staged:
        assert {path.name for path in retried.recovery_directory.iterdir()} == {
            "owner.json"
        }
        retried = quality_module._stage_evidence_publication(retried, payloads)
    quality_module._publish_staged_evidence(retried)
    assert not publication.recovery_directory.exists()


def test_record_resumes_after_all_artifact_links_before_completed_fast_path(
    tmp_path,
    monkeypatch,
):
    root = tmp_path
    destinations = quality_module._canonical_evidence_destinations(
        root,
        root / quality_module.CANONICAL_MANIFEST_PATH,
        root / quality_module.CANONICAL_RAW_PATH,
        root / quality_module.CANONICAL_REPORT_PATH,
    )
    source_commit = "a" * 40
    workload = SimpleNamespace(identity="sha256:" + "2" * 64)
    command = quality_module._recording_command_document(root, destinations)
    session_identity = quality_module._recording_session_identity(
        source_commit, workload.identity, command
    )
    owner = quality_module._publication_owner_document(
        session_identity=session_identity,
        source_commit=source_commit,
        workload_identity=workload.identity,
        destinations=destinations,
    )
    recovery = root / quality_module.RECOVERY_PATH
    publication = quality_module._prepare_evidence_publication(
        recovery, destinations, owner
    )
    payloads = {
        "manifest": b'{"artifact":"manifest"}',
        "raw": b'{"artifact":"raw"}\n',
        "report": b'{"artifact":"report"}',
    }
    publication = quality_module._stage_evidence_publication(publication, payloads)
    real_link = os.link

    def interrupt_completion(source, destination, **kwargs):
        if Path(destination).name == "completed.json":
            raise RuntimeError("interrupted before completion link")
        return real_link(source, destination, **kwargs)

    monkeypatch.setattr(os, "link", interrupt_completion)
    with pytest.raises(RuntimeError, match="before completion link"):
        quality_module._publish_staged_evidence(publication)
    assert all(path.is_file() for _name, path in destinations.ordered())
    assert recovery.is_dir()

    monkeypatch.setattr(os, "link", real_link)
    monkeypatch.setattr(quality_module, "_root", lambda: root)
    monkeypatch.setattr(
        quality_module, "_require_clean_recording_checkout", lambda *_args: None
    )
    monkeypatch.setattr(quality_module, "_git", lambda *_args: source_commit)
    monkeypatch.setattr(
        quality_module, "build_pretraining_quality_workload", lambda _root: workload
    )
    validated_after_cleanup = []

    def validate_after_cleanup(_root, _workload, _destinations):
        validated_after_cleanup.append(not recovery.exists())
        return "pass"

    monkeypatch.setattr(
        quality_module, "_validate_evidence_files", validate_after_cleanup
    )
    result = quality_module._record(
        SimpleNamespace(
            manifest=destinations.manifest,
            raw_output=destinations.raw_output,
            output=destinations.report,
            steps=CANONICAL_STEPS,
        )
    )

    assert result == 0
    assert validated_after_cleanup == [True]
    assert not recovery.exists()


@pytest.mark.parametrize("entry_kind", ["regular", "symlink"])
def test_owned_recovery_cleanup_preserves_and_rejects_unknown_entries(
    tmp_path,
    entry_kind,
):
    publication, _payloads, _owner = _test_publication(tmp_path)
    unknown = publication.recovery_directory / "unrelated"
    if entry_kind == "regular":
        unknown.write_bytes(b"unrelated")
    else:
        os.symlink(publication.recovery_directory / "raw.payload", unknown)

    with pytest.raises(ValueError, match="unexpected recovery entry"):
        quality_module._remove_owned_recovery(publication)

    assert unknown.exists() or unknown.is_symlink()
    assert publication.recovery_directory.is_dir()


def test_owned_recovery_cleanup_preserves_and_rejects_a_swapped_owner_symlink(
    tmp_path,
):
    publication, _payloads, _owner = _test_publication(tmp_path)
    owner_path = publication.recovery_directory / "owner.json"
    external_owner = tmp_path / "external-owner.json"
    external_owner.write_bytes(owner_path.read_bytes())
    owner_path.unlink()
    os.symlink(external_owner, owner_path)

    with pytest.raises(ValueError, match="owner.*regular file"):
        quality_module._remove_owned_recovery(publication)

    assert owner_path.is_symlink()
    assert external_owner.is_file()
    assert publication.recovery_directory.is_dir()


def test_owned_recovery_resume_preserves_and_rejects_a_swapped_payload_symlink(
    tmp_path,
):
    publication, payloads, owner = _test_publication(tmp_path)
    raw_path = publication.recovery_directory / "raw.payload"
    external_raw = tmp_path / "external-raw.jsonl"
    external_raw.write_bytes(payloads["raw"])
    raw_path.unlink()
    os.symlink(external_raw, raw_path)

    with pytest.raises(ValueError, match="staged raw.*regular file"):
        quality_module._prepare_evidence_publication(
            publication.recovery_directory,
            publication.destinations,
            owner,
        )

    assert raw_path.is_symlink()
    assert external_raw.read_bytes() == payloads["raw"]
    assert publication.recovery_directory.is_dir()
    assert not any(path.exists() for _name, path in publication.destinations.ordered())


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
