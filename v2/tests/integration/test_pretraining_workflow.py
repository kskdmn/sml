from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from sml.artifacts.checkpoint import (
    publish_immutable_bundle,
    resolve_latest_step,
    run_writer_lock,
)
from sml.artifacts.manifest import (
    ArrayPayloadRef,
    ArraySpec,
    CheckpointManifest,
    PayloadRef,
    PretrainingDataManifest,
    TokenizerManifest,
    VerificationLevel,
    canonical_json_bytes,
    file_identity,
    row_content_identity,
)
from sml.data.pretraining import PretrainingCursor
from sml.errors import SMLArtifactError
from sml.model.config import ModelConfig
from sml.training import common as training_common
from sml.training import pretrain
from sml.training.common import (
    CheckpointPolicy,
    LoaderConfig,
    OptimizerConfig,
    PretrainingConfig,
)


class InjectedFailure(RuntimeError):
    pass


def _overrides(**values):
    return training_common.ResumeOverrides(**values)


def _payload_ref(path: Path, logical_path: str) -> PayloadRef:
    with path.open("rb") as payload:
        identity = file_identity(payload)
    return PayloadRef(logical_path, identity, path.stat().st_size)


def _prepared_bundle(
    output: Path,
    *,
    partitions: tuple[tuple[int, ...], ...] = ((0, 1, 2), (3, 4, 5)),
) -> Path:
    width = 5

    def build(private: Path) -> PretrainingDataManifest:
        tokenizer = private / "tokenizer"
        tokenizer.mkdir()
        model = tokenizer / "tokenizer.model"
        vocab = tokenizer / "tokenizer.vocab"
        model.write_bytes(b"portable-test-tokenizer")
        vocab.write_bytes(b"<unk>\t0\n<s>\t0\n</s>\t0\n<pad>\t0\n")
        tokenizer_manifest = TokenizerManifest(
            kind="tokenizer",
            version=1,
            identity="sha256:" + "0" * 64,
            algorithm="bpe",
            training={},
            vocab_size=32,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=3,
            unk_token_id=0,
            model=_payload_ref(model, "tokenizer.model"),
            vocab=_payload_ref(vocab, "tokenizer.vocab"),
            diagnostic_source_locator=None,
        )
        tokenizer_manifest = replace(
            tokenizer_manifest,
            identity=tokenizer_manifest.recompute_identity(),
        )
        (tokenizer / "manifest.json").write_bytes(
            canonical_json_bytes(tokenizer_manifest)
        )

        shard_directory = private / "shards"
        shard_directory.mkdir()
        shard_paths: list[Path] = []
        shard_arrays: list[np.ndarray] = []
        for index, identifiers in enumerate(partitions):
            rows = np.stack(
                [
                    (np.arange(width, dtype="<i4") + identifier + 4) % 32
                    for identifier in identifiers
                ]
            )
            path = shard_directory / f"train-{index:06d}.npy"
            with path.open("wb") as payload:
                np.save(payload, rows, allow_pickle=False)
            shard_paths.append(path)
            shard_arrays.append(rows)
        counts = tuple(array.shape[0] for array in shard_arrays)
        manifest = PretrainingDataManifest(
            kind="pretraining-data",
            version=1,
            identity="sha256:" + "0" * 64,
            sequence_length=4,
            row_width=width,
            dtype="int32",
            shard_row_counts=counts,
            shards=tuple(
                _payload_ref(path, f"shards/{path.name}") for path in shard_paths
            ),
            preparation_seed=7,
            row_order_policy={
                "algorithm": "numpy-pcg64-windowed-row-shuffle-v1",
                "shuffle_window_rows": 3,
                "output_shard_rows": max(counts),
            },
            tokenizer_identity=tokenizer_manifest.identity,
            tokenizer_model=replace(
                tokenizer_manifest.model,
                logical_path="tokenizer/tokenizer.model",
            ),
            tokenizer_vocab=replace(
                tokenizer_manifest.vocab,
                logical_path="tokenizer/tokenizer.vocab",
            ),
            source_summary={},
            diagnostic_source_locator=None,
            row_content_identity=row_content_identity(
                (row for array in shard_arrays for row in array),
                sum(counts),
                width,
            ),
        )
        return replace(manifest, identity=manifest.recompute_identity())

    return publish_immutable_bundle(output, build).path


@pytest.fixture
def prepared_data(tmp_path: Path) -> Path:
    return _prepared_bundle(tmp_path / "prepared")


def _config(data: Path, run: Path, *, maximum_steps: int = 2) -> PretrainingConfig:
    return PretrainingConfig(
        data=data,
        output_run=run,
        model=ModelConfig(
            vocab_size=32,
            hidden_size=8,
            num_layers=1,
            num_q_heads=2,
            num_kv_heads=1,
            intermediate_size=16,
            original_context_length=4,
            rope_scaling_factor=1.0,
            hidden_dropout=0.2,
        ),
        optimizer=OptimizerConfig(
            learning_rate=0.01,
            beta1=0.5,
            beta2=0.5,
            schedule_steps=8,
            warmup_steps=0,
        ),
        loader=LoaderConfig(
            microbatch_size=1,
            gradient_accumulation_steps=2,
            prefetch_depth=3,
            epoch_seed=13,
        ),
        checkpoint=CheckpointPolicy(interval=2),
        maximum_steps=maximum_steps,
        maximum_epochs=4,
        log_interval=1,
        seed=19,
    )


def _loaded_groups(run: Path) -> tuple[dict, dict, dict, dict]:
    resolved = resolve_latest_step(
        run,
        writable=False,
        verification=VerificationLevel.FULL,
    )
    names = ("master", "model", "optimizer", "trainer")
    return tuple(
        mx.load(resolved.step_directory / f"{name}.safetensors") for name in names
    )


def _dtype_name(array: mx.array) -> str:
    return {
        mx.bfloat16: "bfloat16",
        mx.float32: "float32",
        mx.int32: "int32",
        mx.uint32: "uint32",
    }[array.dtype]


def _rewrite_array_group(
    resolved,
    logical_path: str,
    arrays: dict[str, mx.array],
) -> None:
    path = resolved.step_directory / logical_path
    mx.save_safetensors(path, arrays)
    replacement = ArrayPayloadRef(
        payload=_payload_ref(path, logical_path),
        arrays=tuple(
            ArraySpec(name, tuple(array.shape), _dtype_name(array))
            for name, array in sorted(arrays.items())
        ),
    )
    manifest = replace(
        resolved.checkpoint,
        arrays=tuple(
            replacement if ref.payload.logical_path == logical_path else ref
            for ref in resolved.checkpoint.arrays
        ),
    )
    manifest = replace(manifest, identity=manifest.recompute_identity())
    (resolved.step_directory / CheckpointManifest.MANIFEST_FILENAME).write_bytes(
        canonical_json_bytes(manifest)
    )


def _assert_run_states_equal(left: Path, right: Path) -> None:
    left_groups = _loaded_groups(left)
    right_groups = _loaded_groups(right)
    for left_group, right_group in zip(left_groups, right_groups, strict=True):
        assert left_group.keys() == right_group.keys()
        mx.eval(*left_group.values(), *right_group.values())
        for name in left_group:
            assert bool(mx.array_equal(left_group[name], right_group[name])), name
    left_resolved = resolve_latest_step(
        left, writable=False, verification=VerificationLevel.FULL
    )
    right_resolved = resolve_latest_step(
        right, writable=False, verification=VerificationLevel.FULL
    )
    assert pretrain.read_scalar_state(left_resolved) == pretrain.read_scalar_state(
        right_resolved
    )


def _fault_after_one_successful_microstep(real_builder):
    def build(*args, **kwargs):
        kernels = real_builder(*args, **kwargs)
        real_microstep = kernels.microstep
        calls = 0

        def microstep(*microstep_args, **microstep_kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise InjectedFailure("after one uncommitted microstep")
            return real_microstep(*microstep_args, **microstep_kwargs)

        class FaultingKernels:
            def optimizer_step(self, *step_args, **step_kwargs):
                return kernels.optimizer_step(*step_args, **step_kwargs)

        wrapper = FaultingKernels()
        wrapper.microstep = microstep
        return wrapper

    return build


def test_failure_after_atomic_creation_leaves_complete_step_zero(
    prepared_data: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_build(*_args, **_kwargs):
        raise InjectedFailure("after step zero")

    monkeypatch.setattr(pretrain, "build_pretraining_kernels", fail_build)
    run = tmp_path / "run"
    with pytest.raises(InjectedFailure, match="step zero"):
        pretrain.train(_config(prepared_data, run, maximum_steps=1))

    resolved = resolve_latest_step(
        run, writable=False, verification=VerificationLevel.FULL
    )
    assert resolved.step == 0
    assert pretrain.read_scalar_state(resolved).cursor == PretrainingCursor.initial()
    assert {path.name for path in resolved.step_directory.iterdir()} == {
        "checkpoint.json",
        "master.safetensors",
        "model.safetensors",
        "optimizer.safetensors",
        "state.json",
        "trainer.safetensors",
    }
    master, model, optimizer, trainer = _loaded_groups(run)
    assert master.keys() == model.keys()
    for name in master:
        assert master[name].dtype == mx.float32
        assert model[name].dtype == mx.bfloat16
        assert bool(mx.array_equal(model[name], master[name].astype(mx.bfloat16)))
    assert optimizer["step"].dtype == mx.int32
    assert all(
        value.dtype == mx.float32 for name, value in optimizer.items() if name != "step"
    )
    assert trainer["accumulation_count"].dtype == mx.int32
    assert trainer["next_key"].dtype == mx.uint32
    assert trainer["loss_numerator"].dtype == mx.float32


def test_interrupted_accumulation_replays_the_complete_window(
    prepared_data: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uninterrupted = pretrain.train(_config(prepared_data, tmp_path / "full"))
    real_builder = pretrain.build_pretraining_kernels
    monkeypatch.setattr(
        pretrain,
        "build_pretraining_kernels",
        _fault_after_one_successful_microstep(real_builder),
    )
    crashed = tmp_path / "crashed"
    with pytest.raises(InjectedFailure, match="uncommitted"):
        pretrain.train(_config(prepared_data, crashed))
    monkeypatch.setattr(pretrain, "build_pretraining_kernels", real_builder)

    resumed = pretrain.resume(
        crashed,
        data=prepared_data,
        overrides=_overrides(maximum_steps=uninterrupted.step),
    )

    assert resumed.step == uninterrupted.step == 2
    _assert_run_states_equal(resumed.run, uninterrupted.run)


def test_resume_accepts_relocation_rejects_resharding_and_prunes_to_latest(
    prepared_data: Path,
    tmp_path: Path,
) -> None:
    first = pretrain.train(_config(prepared_data, tmp_path / "source", maximum_steps=1))
    moved_run = tmp_path / "moved-run"
    first.run.rename(moved_run)
    immutable_run_bytes = (moved_run / "run.json").read_bytes()
    relocated = tmp_path / "relocated-data"
    shutil.copytree(prepared_data, relocated)

    resumed = pretrain.resume(
        moved_run,
        data=relocated,
        overrides=_overrides(maximum_steps=2, checkpoint_interval=1),
    )

    assert resumed.step == 2
    assert [path.name for path in (moved_run / "checkpoints").iterdir()] == [
        "step-000000002"
    ]
    assert (moved_run / "run.json").read_bytes() == immutable_run_bytes
    prepared_data.rename(tmp_path / "removed-original-data")
    assert (
        resolve_latest_step(
            moved_run, writable=False, verification=VerificationLevel.FULL
        ).step
        == 2
    )

    resharded = _prepared_bundle(
        tmp_path / "resharded",
        partitions=((0, 1), (2, 3, 4, 5)),
    )
    with pytest.raises(SMLArtifactError, match="prepared-data identity"):
        pretrain.resume(
            moved_run,
            data=resharded,
            overrides=_overrides(maximum_steps=3),
        )


def test_completed_limit_returns_before_stream_model_or_kernel_construction(
    prepared_data: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed = pretrain.train(
        _config(prepared_data, tmp_path / "run", maximum_steps=1)
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("runtime construction was reached")

    monkeypatch.setattr(pretrain, "PretrainingBatchStream", forbidden)
    monkeypatch.setattr(pretrain, "SMLLanguageModel", forbidden)
    monkeypatch.setattr(pretrain, "build_pretraining_kernels", forbidden)

    result = pretrain.resume(
        completed.run,
        data=prepared_data,
        overrides=_overrides(maximum_steps=1),
    )

    assert result == completed


def test_corrupt_inputs_fail_before_model_allocation(
    prepared_data: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corrupted_data = tmp_path / "corrupted-data"
    shutil.copytree(prepared_data, corrupted_data)
    shard = next((corrupted_data / "shards").iterdir())
    payload = bytearray(shard.read_bytes())
    payload[-1] ^= 1
    shard.write_bytes(payload)

    allocations = 0

    def record_allocation(*_args, **_kwargs):
        nonlocal allocations
        allocations += 1
        raise AssertionError("model allocated")

    monkeypatch.setattr(pretrain, "SMLLanguageModel", record_allocation)
    run = tmp_path / "fresh-corrupt"
    with pytest.raises(SMLArtifactError, match="payload identity"):
        pretrain.train(_config(corrupted_data, run, maximum_steps=1))
    assert allocations == 0
    assert not run.exists()

    monkeypatch.undo()
    completed = pretrain.train(
        _config(prepared_data, tmp_path / "resume-corrupt", maximum_steps=1)
    )
    resolved = resolve_latest_step(
        completed.run, writable=False, verification=VerificationLevel.FULL
    )
    model_file = resolved.step_directory / "model.safetensors"
    model_payload = bytearray(model_file.read_bytes())
    model_payload[-1] ^= 1
    model_file.write_bytes(model_payload)
    monkeypatch.setattr(pretrain, "SMLLanguageModel", record_allocation)

    with pytest.raises(SMLArtifactError, match="payload identity"):
        pretrain.resume(
            completed.run,
            data=prepared_data,
            overrides=_overrides(maximum_steps=2),
        )
    assert allocations == 0


def test_existing_target_and_writer_conflict_fail_without_runtime_allocation(
    prepared_data: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    monkeypatch.setattr(
        pretrain,
        "SMLLanguageModel",
        lambda *_args, **_kwargs: pytest.fail("allocated model for rejected target"),
    )
    with pytest.raises(SMLArtifactError, match="existing|already"):
        pretrain.train(_config(prepared_data, existing, maximum_steps=1))

    locked = tmp_path / "locked"
    with run_writer_lock(locked), pytest.raises(SMLArtifactError, match="held by|lock"):
        pretrain.train(_config(prepared_data, locked, maximum_steps=1))


def test_latest_is_recovered_before_completed_resume_returns(
    prepared_data: Path,
    tmp_path: Path,
) -> None:
    completed = pretrain.train(
        _config(prepared_data, tmp_path / "run", maximum_steps=1)
    )
    (completed.run / "latest.json").unlink()

    result = pretrain.resume(
        completed.run,
        data=prepared_data,
        overrides=_overrides(maximum_steps=1),
    )

    assert result.step == 1
    assert (completed.run / "latest.json").is_file()


def test_checkpoint_interval_counts_updates_and_final_state_is_committed(
    prepared_data: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    published: list[int] = []
    real_publish = pretrain.publish_checkpoint

    def record_publish(run, build):
        resolved = real_publish(run, build)
        published.append(resolved.step)
        return resolved

    monkeypatch.setattr(pretrain, "publish_checkpoint", record_publish)

    result = pretrain.train(_config(prepared_data, tmp_path / "run", maximum_steps=3))

    assert result.step == 3
    assert published == [0, 2, 3]
    assert [path.name for path in (result.run / "checkpoints").iterdir()] == [
        "step-000000003"
    ]


@pytest.mark.parametrize(
    "mutation",
    (
        "missing-master",
        "additional-master",
        "missing-working",
        "additional-working",
        "wrong-master-dtype",
        "wrong-working-dtype",
        "working-not-master-cast",
    ),
)
def test_structural_checkpoint_corruption_fails_before_allocation_or_retention(
    mutation: str,
    prepared_data: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed = pretrain.train(
        _config(prepared_data, tmp_path / "run", maximum_steps=1)
    )
    resolved = resolve_latest_step(
        completed.run, writable=False, verification=VerificationLevel.FULL
    )
    master = mx.load(resolved.step_directory / "master.safetensors")
    working = mx.load(resolved.step_directory / "model.safetensors")
    name = next(iter(master))
    if mutation == "missing-master":
        master.pop(name)
        _rewrite_array_group(resolved, "master.safetensors", master)
    elif mutation == "additional-master":
        master["unexpected.weight"] = master[name]
        _rewrite_array_group(resolved, "master.safetensors", master)
    elif mutation == "missing-working":
        working.pop(name)
        _rewrite_array_group(resolved, "model.safetensors", working)
    elif mutation == "additional-working":
        working["unexpected.weight"] = working[name]
        _rewrite_array_group(resolved, "model.safetensors", working)
    elif mutation == "wrong-master-dtype":
        master[name] = master[name].astype(mx.bfloat16)
        _rewrite_array_group(resolved, "master.safetensors", master)
    elif mutation == "wrong-working-dtype":
        working[name] = working[name].astype(mx.float32)
        _rewrite_array_group(resolved, "model.safetensors", working)
    else:
        working[name] = (master[name] + 1.0).astype(mx.bfloat16)
        _rewrite_array_group(resolved, "model.safetensors", working)

    allocations = 0
    retentions = 0

    def forbidden_allocation(*_args, **_kwargs):
        nonlocal allocations
        allocations += 1
        raise AssertionError("model allocation reached")

    def forbidden_retention(*_args, **_kwargs):
        nonlocal retentions
        retentions += 1
        raise AssertionError("retention reached")

    monkeypatch.setattr(pretrain, "SMLLanguageModel", forbidden_allocation)
    monkeypatch.setattr(pretrain, "apply_retention", forbidden_retention)
    with pytest.raises(SMLArtifactError, match="checkpoint|parameter|working|master"):
        pretrain.resume(
            completed.run,
            data=prepared_data,
            overrides=_overrides(maximum_steps=2),
        )
    assert allocations == 0
    assert retentions == 0


def test_dropped_epoch_tail_does_not_publish_duplicate_progress_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _prepared_bundle(
        tmp_path / "prepared",
        partitions=((0, 1, 2), (3, 4)),
    )
    config = replace(
        _config(data, tmp_path / "run", maximum_steps=8),
        loader=LoaderConfig(
            microbatch_size=2,
            gradient_accumulation_steps=2,
            prefetch_depth=3,
            epoch_seed=13,
        ),
        maximum_epochs=1,
        checkpoint=CheckpointPolicy(interval=1),
    )
    published: list[int] = []
    real_publish = pretrain.publish_checkpoint

    def record_publish(run, build):
        resolved = real_publish(run, build)
        published.append(resolved.step)
        return resolved

    monkeypatch.setattr(pretrain, "publish_checkpoint", record_publish)

    result = pretrain.train(config)

    assert result.step == 1
    assert result.rows == 4
    assert published == [0, 1]
