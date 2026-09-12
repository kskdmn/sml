from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
import weakref
from collections import Counter
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
import zstandard as zstd
from mlx.utils import tree_flatten, tree_unflatten
from sml.artifacts import checkpoint as checkpoint_module
from sml.artifacts import manifest as manifest_module
from sml.artifacts.checkpoint import (
    publish_immutable_bundle,
    resolve_latest_step,
    run_writer_lock,
)
from sml.artifacts.manifest import (
    ArrayPayloadRef,
    ArraySpec,
    ArtifactRoot,
    LatestIndex,
    PayloadRef,
    PretrainingCheckpointManifest,
    PretrainingDataManifest,
    PretrainingRunManifest,
    TokenizerManifest,
    VerificationLevel,
    canonical_json_bytes,
    file_identity,
    read_manifest,
    row_content_identity,
)
from sml.artifacts.semantics import expected_next_key
from sml.artifacts.verify import verify_artifact
from sml.data import pretraining as data_module
from sml.data.corpus import CorpusConfig
from sml.data.pretraining import (
    PretrainingCursor,
    PretrainingPreparationConfig,
    prepare_pretraining_bundle,
)
from sml.data.tokenizer import TokenizerTrainingConfig, train_tokenizer_bundle
from sml.errors import SMLArtifactError, SMLRuntimeError
from sml.inference import InferenceSession
from sml.model.config import ModelConfig
from sml.model.language_model import SMLLanguageModel
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


@pytest.mark.parametrize(
    "mode", ("fresh", "resume", "complete", "restore-failure", "initial-state-failure")
)
def test_training_retains_one_full_data_proof_and_closes_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    data = _prepared_bundle(
        tmp_path / "prepared",
        partitions=tuple(tuple(range(start, start + 3)) for start in range(0, 18, 3)),
    )
    run = tmp_path / "run"
    if mode not in {"fresh", "initial-state-failure"}:
        pretrain.train(_config(data, run, maximum_steps=1))

    hashes = Counter()
    scans = 0
    stores = []
    descriptors = []
    real_open = manifest_module._open_verified_payload
    real_identity = data_module._row_content_identity_blocks
    real_preflight = pretrain._open_validated_prepared_resources

    def record_open(root, reference, verification):
        if (
            reference.logical_path.startswith("shards/")
            and verification is VerificationLevel.FULL
        ):
            hashes[reference.logical_path] += 1
        return real_open(root, reference, verification)

    def record_identity(*args, **kwargs):
        nonlocal scans
        scans += 1
        return real_identity(*args, **kwargs)

    def record_preflight(*args, **kwargs):
        store = real_preflight(*args, **kwargs)
        stores.append(store)
        descriptors.append(store.artifact.root.fileno())
        return store

    def fail_restore(*_args, **_kwargs):
        raise InjectedFailure("restore failed")

    def fail_initial_state(*_args, **_kwargs):
        raise InjectedFailure("initial state failed")

    monkeypatch.setattr(manifest_module, "_open_verified_payload", record_open)
    monkeypatch.setattr(data_module, "_row_content_identity_blocks", record_identity)
    monkeypatch.setattr(
        pretrain, "_open_validated_prepared_resources", record_preflight
    )
    if mode == "initial-state-failure":
        monkeypatch.setattr(pretrain, "_initial_state", fail_initial_state)
        with pytest.raises(InjectedFailure, match="initial state failed"):
            pretrain.train(_config(data, run))
    elif mode == "restore-failure":
        monkeypatch.setattr(pretrain, "_restore_checkpoint", fail_restore)
        with pytest.raises(InjectedFailure, match="restore failed"):
            pretrain.resume(run, data=data, overrides=_overrides(maximum_steps=2))
    elif mode == "fresh":
        assert pretrain.train(_config(data, run)).step == 2
    else:
        limit = 1 if mode == "complete" else 2
        assert (
            pretrain.resume(
                run, data=data, overrides=_overrides(maximum_steps=limit)
            ).step
            == limit
        )

    assert hashes == {f"shards/train-{index:06d}.npy": 1 for index in range(6)}
    assert scans == 1
    assert len(stores) == 1 and stores[0]._closed
    for descriptor in descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_pretraining_logs_update_metrics_at_configured_interval(
    prepared_data, tmp_path, monkeypatch, capsys
) -> None:
    updates = []
    original_step = pretrain.PretrainingKernels.optimizer_step

    def record_step(kernels, *args):
        updated = original_step(kernels, *args)
        updates.append(updated.metrics)
        return updated

    monkeypatch.setattr(pretrain.PretrainingKernels, "optimizer_step", record_step)
    trained = pretrain.train(
        replace(
            _config(prepared_data, tmp_path / "logs-run", maximum_steps=3),
            log_interval=2,
        )
    )
    output = capsys.readouterr()
    assert trained.step == 3
    lines = output.err.splitlines()
    assert output.out == ""
    assert len(lines) == 1
    assert lines[0].startswith("pretrain step=2 epoch=0 rows=4 ")
    metrics = dict(field.split("=", 1) for field in lines[0].split()[1:])
    assert float(metrics["loss"]) == pytest.approx(
        float(updates[1]["loss"].item()), abs=1e-6
    )
    assert float(metrics["learning_rate"]) == pytest.approx(
        float(updates[1]["learning_rate"].item()), rel=1e-5
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
    assert isinstance(resolved.checkpoint, PretrainingCheckpointManifest)
    field_name = {
        "model.safetensors": "model",
        "master.safetensors": "master",
        "optimizer.safetensors": "optimizer",
        "trainer.safetensors": "trainer",
    }[logical_path]
    manifest = replace(resolved.checkpoint, **{field_name: replacement})
    manifest = replace(manifest, identity=manifest.recompute_identity())
    (
        resolved.step_directory / PretrainingCheckpointManifest.MANIFEST_FILENAME
    ).write_bytes(canonical_json_bytes(manifest))


def _rewrite_scalar_state(resolved, mutate) -> None:
    path = resolved.step_directory / "state.json"
    document = json.loads(path.read_bytes())
    mutate(document)
    path.write_bytes(canonical_json_bytes(document))
    manifest = replace(
        resolved.checkpoint,
        scalar_state=_payload_ref(path, "state.json"),
    )
    manifest = replace(manifest, identity=manifest.recompute_identity())
    (
        resolved.step_directory / PretrainingCheckpointManifest.MANIFEST_FILENAME
    ).write_bytes(canonical_json_bytes(manifest))


def _replace_latest_trainer_key(run: Path, key: mx.array) -> None:
    resolved = resolve_latest_step(
        run,
        writable=False,
        verification=VerificationLevel.FULL,
    )
    trainer_path = resolved.step_directory / "trainer.safetensors"
    trainer = dict(mx.load(trainer_path))
    mx.eval(*trainer.values())
    trainer["next_key"] = key
    mx.save_safetensors(trainer_path, trainer)
    assert isinstance(resolved.checkpoint, PretrainingCheckpointManifest)
    checkpoint = replace(
        resolved.checkpoint,
        trainer=replace(
            resolved.checkpoint.trainer,
            payload=_payload_ref(trainer_path, "trainer.safetensors"),
        ),
    )
    checkpoint = replace(checkpoint, identity=checkpoint.recompute_identity())
    (resolved.step_directory / "checkpoint.json").write_bytes(
        canonical_json_bytes(checkpoint)
    )
    latest = read_manifest(
        run,
        LatestIndex,
        VerificationLevel.MANIFEST_TRUSTED,
    ).manifest
    latest = replace(latest, checkpoint_identity=checkpoint.identity)
    latest = replace(latest, identity=latest.recompute_identity())
    (run / "latest.json").write_bytes(canonical_json_bytes(latest))


def _run_with_unpruned_latest(
    data: Path,
    run: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    real_retention = checkpoint_module._prune_to_latest

    def interrupt_retention(*_args, **_kwargs):
        raise InjectedFailure("before retention")

    monkeypatch.setattr(checkpoint_module, "_prune_to_latest", interrupt_retention)
    with pytest.raises(InjectedFailure, match="before retention"):
        pretrain.train(
            replace(
                _config(data, run, maximum_steps=1),
                checkpoint=CheckpointPolicy(interval=1),
            )
        )
    monkeypatch.setattr(checkpoint_module, "_prune_to_latest", real_retention)
    assert sorted(path.name for path in (run / "checkpoints").iterdir()) == [
        "step-000000000",
        "step-000000001",
    ]
    return resolve_latest_step(
        run,
        writable=False,
        verification=VerificationLevel.FULL,
    )


def _bind_run_to_token_invalid_data(
    run: Path,
    source_data: Path,
    invalid_data: Path,
    *,
    token_id: int = 32,
) -> None:
    shutil.copytree(source_data, invalid_data)
    data_manifest = read_manifest(
        invalid_data,
        PretrainingDataManifest,
        VerificationLevel.FULL,
    ).manifest
    shard_path = invalid_data / data_manifest.shards[0].logical_path
    rows = np.load(shard_path, allow_pickle=False)
    rows[0, 0] = token_id
    with shard_path.open("wb") as payload:
        np.save(payload, rows, allow_pickle=False)
    shards = (
        replace(
            data_manifest.shards[0],
            identity=_payload_ref(
                shard_path, data_manifest.shards[0].logical_path
            ).identity,
            byte_size=shard_path.stat().st_size,
        ),
        *data_manifest.shards[1:],
    )
    data_manifest = replace(
        data_manifest,
        shards=shards,
        row_content_identity=row_content_identity(
            (
                row
                for reference in shards
                for row in np.load(
                    invalid_data / reference.logical_path,
                    allow_pickle=False,
                )
            ),
            sum(data_manifest.shard_row_counts),
            data_manifest.row_width,
        ),
    )
    data_manifest = replace(data_manifest, identity=data_manifest.recompute_identity())
    (invalid_data / "manifest.json").write_bytes(canonical_json_bytes(data_manifest))

    resolved = resolve_latest_step(
        run, writable=False, verification=VerificationLevel.FULL
    )
    assert isinstance(resolved.run, PretrainingRunManifest)
    assert isinstance(resolved.checkpoint, PretrainingCheckpointManifest)
    run_manifest = replace(
        resolved.run,
        data_identity=data_manifest.identity,
        diagnostic_data_locator=str(invalid_data),
    )
    run_manifest = replace(run_manifest, identity=run_manifest.recompute_identity())
    (run / "run.json").write_bytes(canonical_json_bytes(run_manifest))

    state_path = resolved.step_directory / "state.json"
    scalar = json.loads(state_path.read_bytes())
    scalar["owning_run_identity"] = run_manifest.identity
    state_path.write_bytes(canonical_json_bytes(scalar))
    checkpoint_manifest = replace(
        resolved.checkpoint,
        owning_run_identity=run_manifest.identity,
        scalar_state=_payload_ref(state_path, "state.json"),
    )
    checkpoint_manifest = replace(
        checkpoint_manifest,
        identity=checkpoint_manifest.recompute_identity(),
    )
    (resolved.step_directory / "checkpoint.json").write_bytes(
        canonical_json_bytes(checkpoint_manifest)
    )
    latest = LatestIndex(
        kind="latest-index",
        version=1,
        identity="sha256:" + "0" * 64,
        owning_run_identity=run_manifest.identity,
        step=checkpoint_manifest.step,
        checkpoint_identity=checkpoint_manifest.identity,
    )
    latest = replace(latest, identity=latest.recompute_identity())
    (run / "latest.json").write_bytes(canonical_json_bytes(latest))


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
    for resolved, groups in (
        (left_resolved, left_groups),
        (right_resolved, right_groups),
    ):
        assert isinstance(resolved.run, PretrainingRunManifest)
        state = pretrain.read_scalar_state(resolved)
        expected = expected_next_key(
            seed=int(resolved.run.checkpoint["seed"]),
            microsteps=state.microsteps,
            model=ModelConfig(**dict(resolved.run.model)),
        )
        mx.eval(expected, groups[-1]["next_key"])
        assert bool(mx.array_equal(groups[-1]["next_key"], expected))


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


@pytest.mark.parametrize("hidden_dropout", (0.2, 0.0), ids=("model-only", "disabled"))
def test_interrupted_accumulation_replays_the_complete_window(
    prepared_data: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hidden_dropout: float,
) -> None:
    config = _config(prepared_data, tmp_path / "full")
    config = replace(
        config,
        model=replace(config.model, hidden_dropout=hidden_dropout),
    )
    uninterrupted = pretrain.train(config)
    real_builder = pretrain.build_pretraining_kernels
    monkeypatch.setattr(
        pretrain,
        "build_pretraining_kernels",
        _fault_after_one_successful_microstep(real_builder),
    )
    crashed = tmp_path / "crashed"
    with pytest.raises(InjectedFailure, match="uncommitted"):
        pretrain.train(replace(config, output_run=crashed))
    monkeypatch.setattr(pretrain, "build_pretraining_kernels", real_builder)

    resumed = pretrain.resume(
        crashed,
        data=prepared_data,
        overrides=_overrides(maximum_steps=uninterrupted.step),
    )

    assert resumed.step == uninterrupted.step == 2
    _assert_run_states_equal(resumed.run, uninterrupted.run)


@pytest.mark.parametrize("hidden_dropout", (0.0, 0.2))
def test_legacy_checkpoint_resumes_identically_and_publishes_compact_state(
    prepared_data: Path, tmp_path: Path, hidden_dropout: float
) -> None:
    config = _config(prepared_data, tmp_path / "compact", maximum_steps=1)
    config = replace(config, model=replace(config.model, hidden_dropout=hidden_dropout))
    compact = pretrain.train(config)
    legacy_run = tmp_path / "legacy"
    shutil.copytree(compact.run, legacy_run)
    resolved = resolve_latest_step(
        legacy_run, writable=False, verification=VerificationLevel.FULL
    )
    assert resolved.checkpoint.version == 2
    trainer_path = resolved.step_directory / "trainer.safetensors"
    trainer = dict(mx.load(trainer_path))
    assert set(trainer) == {"accumulation_count", "loss_numerator", "next_key"}
    masters = mx.load(resolved.step_directory / "master.safetensors")
    trainer.update(
        (f"accumulators.{name}", mx.zeros_like(value))
        for name, value in masters.items()
    )
    mx.eval(trainer)
    mx.save_safetensors(trainer_path, trainer)
    legacy = replace(
        resolved.checkpoint,
        version=1,
        trainer=ArrayPayloadRef(
            payload=_payload_ref(trainer_path, "trainer.safetensors"),
            arrays=tuple(
                ArraySpec(name, tuple(value.shape), _dtype_name(value))
                for name, value in sorted(trainer.items())
            ),
        ),
    )
    legacy = replace(legacy, identity=legacy.recompute_identity())
    (resolved.step_directory / "checkpoint.json").write_bytes(
        canonical_json_bytes(legacy)
    )
    latest = read_manifest(
        legacy_run, LatestIndex, VerificationLevel.MANIFEST_TRUSTED
    ).manifest
    latest = replace(latest, checkpoint_identity=legacy.identity)
    latest = replace(latest, identity=latest.recompute_identity())
    (legacy_run / "latest.json").write_bytes(canonical_json_bytes(latest))

    for run in (compact.run, legacy_run):
        assert (
            pretrain.resume(
                run, data=prepared_data, overrides=_overrides(maximum_steps=2)
            ).step
            == 2
        )
        final = resolve_latest_step(
            run, writable=False, verification=VerificationLevel.FULL
        )
        assert final.checkpoint.version == 2
        assert {spec.name for spec in final.checkpoint.trainer.arrays} == {
            "accumulation_count",
            "loss_numerator",
            "next_key",
        }
    _assert_run_states_equal(compact.run, legacy_run)


@pytest.mark.parametrize("interval", (1, 1000))
def test_nonfinite_update_preserves_last_checkpoint_before_cursor_commit(
    prepared_data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interval: int
) -> None:
    config = _config(prepared_data, tmp_path / "failed", maximum_steps=3)
    config = replace(
        config,
        optimizer=replace(config.optimizer, learning_rate=3e38, beta1=0.9, beta2=0.999),
        checkpoint=CheckpointPolicy(interval=interval),
    )
    committed = []
    monkeypatch.setattr(
        data_module.PretrainingBatchStream,
        "commit",
        lambda _self, cursor: committed.append(cursor),
    )
    with pytest.raises(SMLRuntimeError, match="nonfinite"):
        pretrain.train(config)
    assert committed == []
    retained = resolve_latest_step(
        config.output_run, writable=False, verification=VerificationLevel.FULL
    )
    assert retained.step == 0
    assert pretrain.read_scalar_state(retained).rows == 0
    assert [path.name for path in (config.output_run / "checkpoints").iterdir()] == [
        "step-000000000"
    ]


@pytest.mark.parametrize("maximum_steps", (1, 2))
@pytest.mark.parametrize("mutation", ("tokenizer.model", "tokenizer.vocab", "manifest"))
def test_resume_rejects_invalid_tokenizer_before_restore_or_retention(
    mutation: str,
    maximum_steps: int,
    prepared_data: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = tmp_path / "invalid-tokenizer-run"
    _run_with_unpruned_latest(prepared_data, run, monkeypatch)
    tokenizer_path = run / "tokenizer"
    if mutation == "manifest":
        manifest = read_manifest(
            tokenizer_path, TokenizerManifest, VerificationLevel.FULL
        ).manifest
        manifest = replace(manifest, training={**manifest.training, "num_threads": 2})
        manifest = replace(manifest, identity=manifest.recompute_identity())
        (tokenizer_path / "manifest.json").write_bytes(canonical_json_bytes(manifest))
        expected_error = "run tokenizer identity does not match run.json"
    else:
        payload_path = tokenizer_path / mutation
        raw = payload_path.read_bytes()
        payload_path.write_bytes(bytes((raw[0] ^ 1,)) + raw[1:])
        expected_error = "payload identity mismatch"
    before = {
        path.relative_to(run): path.read_bytes()
        for path in run.rglob("*")
        if path.is_file()
    }

    def forbidden_restore(*_args, **_kwargs):
        raise AssertionError("checkpoint restore reached before tokenizer validation")

    monkeypatch.setattr(pretrain, "_restore_checkpoint", forbidden_restore)
    with pytest.raises(SMLArtifactError, match=expected_error):
        pretrain.resume(
            run,
            data=prepared_data,
            overrides=_overrides(maximum_steps=maximum_steps),
        )

    assert {
        path.relative_to(run): path.read_bytes()
        for path in run.rglob("*")
        if path.is_file()
    } == before


def test_resume_rejects_wrong_key_before_runtime_or_retention(
    prepared_data: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trained = pretrain.train(
        _config(prepared_data, tmp_path / "wrong-key-run", maximum_steps=1)
    )
    _replace_latest_trainer_key(trained.run, mx.random.key(999))
    reached: list[str] = []

    def forbidden(name):
        def call(*_args, **_kwargs):
            reached.append(name)
            raise AssertionError(f"{name} reached before RNG validation")

        return call

    monkeypatch.setattr(pretrain, "SMLLanguageModel", forbidden("model"))
    monkeypatch.setattr(pretrain, "PretrainingBatchStream", forbidden("stream"))
    monkeypatch.setattr(pretrain, "prune_to_latest", forbidden("retention"))
    monkeypatch.setattr(
        pretrain,
        "_publish_training_state",
        forbidden("publication"),
    )

    with pytest.raises(
        SMLArtifactError,
        match="checkpoint trainer next RNG key is incorrect",
    ):
        pretrain.resume(
            trained.run,
            data=prepared_data,
            overrides=_overrides(maximum_steps=2),
        )
    assert reached == []


def test_partial_window_progress_supports_full_verify_inference_and_resume(
    tmp_path: Path,
) -> None:
    """An epoch-ending partial accumulation window remains a usable checkpoint."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    payload = b"".join(
        json.dumps(
            {"text": "alpha beta gamma delta epsilon zeta eta theta " * 8}
        ).encode("utf-8")
        + b"\n"
        for _ in range(12)
    )
    (corpus / "tiny-0000.jsonl.zst").write_bytes(
        zstd.ZstdCompressor().compress(payload)
    )
    corpus_config = CorpusConfig(
        input_root=corpus,
        shuffle_files=False,
        min_text_bytes=1,
        max_rows_per_file=None,
    )
    tokenizer = train_tokenizer_bundle(
        TokenizerTrainingConfig(
            corpus=corpus_config,
            vocab_size=300,
            hard_vocab_limit=False,
            num_threads=1,
        ),
        tmp_path / "tokenizer",
    )
    data = prepare_pretraining_bundle(
        PretrainingPreparationConfig(
            corpus=corpus_config,
            tokenizer_bundle=tokenizer.path,
            sequence_length=4,
            shuffle_window_rows=4,
            output_shard_rows=8,
            seed=7,
        ),
        tmp_path / "single-row-data",
    )
    row_count = sum(
        read_manifest(
            data.path,
            PretrainingDataManifest,
            VerificationLevel.FULL,
        ).manifest.shard_row_counts
    )
    data_path = data.path
    config = _config(data_path, tmp_path / "partial-window-run", maximum_steps=1)
    trained = pretrain.train(
        replace(
            config,
            model=replace(config.model, vocab_size=tokenizer.manifest.vocab_size),
            loader=replace(config.loader, microbatch_size=row_count),
            maximum_epochs=1,
        )
    )

    resolved = resolve_latest_step(
        trained.run,
        writable=False,
        verification=VerificationLevel.FULL,
    )
    scalar = pretrain.read_scalar_state(resolved)
    assert scalar.step == 1
    assert scalar.microsteps == 1
    verify_artifact(trained.run, full=True)
    InferenceSession.from_checkpoint(trained.run, full_verify=True)

    resumed = pretrain.resume(
        trained.run,
        data=data_path,
        overrides=_overrides(maximum_steps=2, maximum_epochs=2),
    )

    assert resumed.step == 2


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
    resolved = resolve_latest_step(
        moved_run, writable=False, verification=VerificationLevel.FULL
    )
    assert resolved.step == 2
    saved_model = ModelConfig(**dict(resolved.run.model))
    assert saved_model.rope_scaling_factor == 1.0
    with (
        ArtifactRoot.open(resolved.step_directory, writable=False) as root,
        root.open_payload("model.safetensors") as payload,
    ):
        saved_parameters = mx.load(payload, format="safetensors")
    assert {name: tuple(array.shape) for name, array in saved_parameters.items()} == {
        "embed_tokens.weight": (32, 8),
        "layers.0.input_norm.weight": (8,),
        "layers.0.mlp.down_proj.weight": (8, 16),
        "layers.0.mlp.gate_proj.weight": (16, 8),
        "layers.0.mlp.up_proj.weight": (16, 8),
        "layers.0.post_attn_norm.weight": (8,),
        "layers.0.self_attn.k_proj.weight": (4, 8),
        "layers.0.self_attn.o_proj.weight": (8, 8),
        "layers.0.self_attn.q_proj.weight": (8, 8),
        "layers.0.self_attn.v_proj.weight": (4, 8),
        "norm.weight": (8,),
    }
    assert all(array.dtype == mx.bfloat16 for array in saved_parameters.values())
    model = SMLLanguageModel(saved_model, key=mx.random.key(0))
    logits, cache_state, next_key = model.forward_arrays(
        tree_unflatten(sorted(saved_parameters.items())),
        mx.array([[4, 5, 6, 7]], dtype=mx.int32),
        attention_mask=None,
        positions=None,
        cache_state=None,
        training=False,
        key=None,
    )
    mx.eval(logits)
    assert logits.shape == (1, 4, 32)
    assert logits.dtype == mx.bfloat16
    assert cache_state is None
    assert next_key is None

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


@pytest.mark.parametrize("token_id", (32, 3), ids=("out-of-range", "padding"))
def test_resume_semantic_data_preflight_precedes_restore_prune_and_early_return(
    prepared_data: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    token_id: int,
) -> None:
    """A hash-consistent invalid NPY bundle must fail before checkpoint consumption."""
    completed = pretrain.train(
        _config(prepared_data, tmp_path / "run", maximum_steps=1)
    )
    invalid_data = tmp_path / "token-invalid-data"
    _bind_run_to_token_invalid_data(
        completed.run, prepared_data, invalid_data, token_id=token_id
    )
    reached: list[str] = []

    def forbidden(name):
        def call(*_args, **_kwargs):
            reached.append(name)
            raise AssertionError(f"{name} reached before semantic data preflight")

        return call

    monkeypatch.setattr(pretrain, "_restore_checkpoint", forbidden("restore"))
    monkeypatch.setattr(pretrain, "prune_to_latest", forbidden("retention"))
    monkeypatch.setattr(pretrain, "PretrainingBatchStream", forbidden("stream"))
    monkeypatch.setattr(pretrain, "SMLLanguageModel", forbidden("model"))
    monkeypatch.setattr(
        pretrain, "build_pretraining_kernels", forbidden("compiled kernel")
    )

    with pytest.raises(SMLArtifactError, match="token IDs"):
        pretrain.resume(
            completed.run,
            data=invalid_data,
            overrides=_overrides(maximum_steps=1),
        )

    assert reached == []


def test_resume_reader_rejects_named_step_swap_before_retention(
    prepared_data: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A replaced step name must not redirect consumed state after FULL proof."""
    completed = pretrain.train(
        _config(prepared_data, tmp_path / "run", maximum_steps=1)
    )
    resolved = resolve_latest_step(
        completed.run,
        writable=False,
        verification=VerificationLevel.FULL,
    )
    replacement = tmp_path / "replacement-step"
    shutil.copytree(resolved.step_directory, replacement)
    replacement_state = replacement / "state.json"
    state = json.loads(replacement_state.read_bytes())
    state["rows"] = 999
    replacement_state.write_bytes(canonical_json_bytes(state))

    real_opener = getattr(checkpoint_module, "open_checkpoint_reader", None)
    assert real_opener is not None
    original = tmp_path / "opened-original-step"
    swapped = False

    @contextmanager
    def swap_after_open(*args, **kwargs):
        nonlocal swapped
        with real_opener(*args, **kwargs) as reader:
            resolved.step_directory.rename(original)
            replacement.rename(resolved.step_directory)
            swapped = True
            yield reader

    retentions = 0

    def forbidden_retention(*_args, **_kwargs):
        nonlocal retentions
        retentions += 1
        raise AssertionError("retention reached after a hostile step-name swap")

    monkeypatch.setattr(
        pretrain, "open_checkpoint_reader", swap_after_open, raising=False
    )
    monkeypatch.setattr(pretrain, "prune_to_latest", forbidden_retention)
    with pytest.raises(SMLArtifactError, match="inode|named step|swapped"):
        pretrain.resume(
            completed.run,
            data=prepared_data,
            overrides=_overrides(maximum_steps=1),
        )

    assert swapped is True
    assert retentions == 0


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
    digest = hashlib.sha256(run.name.encode("utf-8")).hexdigest()
    stale = tmp_path / f".sml-tmp-{digest}-{'a' * 32}"
    stale.mkdir()
    sentinel = stale / "must-survive.bin"
    sentinel.write_bytes(b"preflight has not authorized deletion")
    with pytest.raises(SMLArtifactError, match="payload identity"):
        pretrain.train(_config(corrupted_data, run, maximum_steps=1))
    assert allocations == 0
    assert not run.exists()
    assert sentinel.read_bytes() == b"preflight has not authorized deletion"

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
    real_publish = checkpoint_module._publish_checkpoint

    def record_publish(run, build, **kwargs):
        resolved = real_publish(run, build, **kwargs)
        published.append(resolved.step)
        return resolved

    monkeypatch.setattr(checkpoint_module, "_publish_checkpoint", record_publish)

    result = pretrain.train(_config(prepared_data, tmp_path / "run", maximum_steps=3))

    assert result.step == 3
    assert published == [0, 2, 3]
    assert [path.name for path in (result.run / "checkpoints").iterdir()] == [
        "step-000000003"
    ]


def test_checkpoint_retention_rejects_same_step_identity_substitution(
    prepared_data: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    substitute = pretrain.train(
        replace(
            _config(prepared_data, tmp_path / "substitute", maximum_steps=1),
            seed=23,
        )
    )
    substituted = resolve_latest_step(
        substitute.run,
        writable=False,
        verification=VerificationLevel.FULL,
    )
    assert substituted.step == 1

    monkeypatch.setattr(
        checkpoint_module,
        "_prune_to_latest",
        lambda _run, **_kwargs: substituted,
    )
    with pytest.raises(SMLArtifactError, match="identity"):
        pretrain.train(
            replace(
                _config(prepared_data, tmp_path / "target", maximum_steps=1),
                checkpoint=CheckpointPolicy(interval=1),
            )
        )


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
    streams = 0

    def forbidden_allocation(*_args, **_kwargs):
        nonlocal allocations
        allocations += 1
        raise AssertionError("model allocation reached")

    def forbidden_retention(*_args, **_kwargs):
        nonlocal retentions
        retentions += 1
        raise AssertionError("retention reached")

    def forbidden_stream(*_args, **_kwargs):
        nonlocal streams
        streams += 1
        raise AssertionError("stream construction reached")

    monkeypatch.setattr(pretrain, "SMLLanguageModel", forbidden_allocation)
    monkeypatch.setattr(pretrain, "prune_to_latest", forbidden_retention)
    monkeypatch.setattr(pretrain, "PretrainingBatchStream", forbidden_stream)
    with pytest.raises(SMLArtifactError, match="checkpoint|parameter|working|master"):
        pretrain.resume(
            completed.run,
            data=prepared_data,
            overrides=_overrides(maximum_steps=2),
        )
    assert allocations == 0
    assert retentions == 0
    assert streams == 0


@pytest.mark.parametrize(
    "mutation",
    (
        "optimizer-missing-key",
        "optimizer-additional-key",
        "optimizer-wrong-dtype",
        "optimizer-coordinated-shape",
        "trainer-missing-key",
        "trainer-additional-key",
        "trainer-wrong-dtype",
        "trainer-wrong-shape",
        "scalar-state-type",
        "cursor-beyond-order",
        "cursor-noncanonical-boundary",
        "prng-wrong-dtype",
        "prng-wrong-shape",
    ),
)
def test_restored_state_corruption_fails_before_pruning_stream_or_model(
    mutation: str,
    prepared_data: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = tmp_path / "run"
    resolved = _run_with_unpruned_latest(prepared_data, run, monkeypatch)
    masters = mx.load(resolved.step_directory / "master.safetensors")
    parameter_name = next(
        name for name, array in masters.items() if len(array.shape) == 2
    )
    moment_name = f"first_moments.{parameter_name}"
    second_name = f"second_moments.{parameter_name}"

    if mutation.startswith("optimizer-"):
        optimizer = mx.load(resolved.step_directory / "optimizer.safetensors")
        if mutation == "optimizer-missing-key":
            optimizer.pop(moment_name)
        elif mutation == "optimizer-additional-key":
            optimizer["first_moments.unexpected.weight"] = optimizer[moment_name]
        elif mutation == "optimizer-wrong-dtype":
            optimizer[moment_name] = optimizer[moment_name].astype(mx.bfloat16)
        else:
            wrong_shape = (int(np.prod(masters[parameter_name].shape)),)
            optimizer[moment_name] = mx.zeros(wrong_shape, dtype=mx.float32)
            optimizer[second_name] = mx.zeros(wrong_shape, dtype=mx.float32)
        _rewrite_array_group(resolved, "optimizer.safetensors", optimizer)
    elif mutation.startswith(("trainer-", "prng-")):
        trainer = mx.load(resolved.step_directory / "trainer.safetensors")
        if mutation == "trainer-missing-key":
            trainer.pop("accumulation_count")
        elif mutation == "trainer-additional-key":
            trainer["accumulators.unexpected.weight"] = mx.zeros_like(
                masters[parameter_name]
            )
        elif mutation == "trainer-wrong-dtype":
            trainer["loss_numerator"] = trainer["loss_numerator"].astype(mx.bfloat16)
        elif mutation == "trainer-wrong-shape":
            trainer["loss_numerator"] = mx.zeros((1,), dtype=mx.float32)
        elif mutation == "prng-wrong-dtype":
            trainer["next_key"] = trainer["next_key"].astype(mx.int32)
        else:
            trainer["next_key"] = mx.zeros((3,), dtype=mx.uint32)
        _rewrite_array_group(resolved, "trainer.safetensors", trainer)
    elif mutation == "scalar-state-type":
        _rewrite_scalar_state(
            resolved,
            lambda document: document.__setitem__("rows", "one"),
        )
    elif mutation == "cursor-beyond-order":
        _rewrite_scalar_state(
            resolved,
            lambda document: document.__setitem__(
                "cursor",
                {"epoch": 0, "shard_order_position": 3, "row_offset": 0},
            ),
        )
    else:
        _rewrite_scalar_state(
            resolved,
            lambda document: document.__setitem__(
                "cursor",
                {"epoch": 0, "shard_order_position": 0, "row_offset": 3},
            ),
        )

    allocations = 0
    retentions = 0
    streams = 0

    def forbidden_allocation(*_args, **_kwargs):
        nonlocal allocations
        allocations += 1
        raise AssertionError("model allocation reached")

    def forbidden_retention(*_args, **_kwargs):
        nonlocal retentions
        retentions += 1
        raise AssertionError("retention reached")

    def forbidden_stream(*_args, **_kwargs):
        nonlocal streams
        streams += 1
        raise AssertionError("stream construction reached")

    monkeypatch.setattr(pretrain, "SMLLanguageModel", forbidden_allocation)
    monkeypatch.setattr(pretrain, "prune_to_latest", forbidden_retention)
    monkeypatch.setattr(pretrain, "PretrainingBatchStream", forbidden_stream)
    with pytest.raises(SMLArtifactError, match="checkpoint|cursor|optimizer|trainer"):
        pretrain.resume(
            run,
            data=prepared_data,
            overrides=_overrides(maximum_steps=2),
        )
    assert allocations == 0
    assert retentions == 0
    assert streams == 0
    assert sorted(path.name for path in (run / "checkpoints").iterdir()) == [
        "step-000000000",
        "step-000000001",
    ]


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
    real_publish = checkpoint_module._publish_checkpoint

    def record_publish(run, build, **kwargs):
        resolved = real_publish(run, build, **kwargs)
        published.append(resolved.step)
        return resolved

    monkeypatch.setattr(checkpoint_module, "_publish_checkpoint", record_publish)

    result = pretrain.train(config)

    assert result.step == 1
    assert result.rows == 4
    assert result.epoch == 1
    assert published == [0, 1]

    def forbid_completed_runtime(*args, **kwargs):
        raise AssertionError("completed epochs must return before allocating a runtime")

    monkeypatch.setattr(pretrain, "_run_training", forbid_completed_runtime)
    resumed = pretrain.resume(result.run, data=data, overrides=_overrides())
    assert resumed == result
    assert published == [0, 1]


@pytest.mark.parametrize("compile", [False, True])
def test_training_releases_initial_device_arrays_after_update(
    tmp_path, monkeypatch, compile
):
    data = _prepared_bundle(tmp_path / "prepared")
    config = replace(
        _config(data, tmp_path / "run", maximum_steps=2),
        compile=compile,
        checkpoint=CheckpointPolicy(interval=1),
    )
    initial_arrays = []
    real_initial_state = pretrain._initial_state
    real_publish = pretrain._publish_training_state

    def track_initial_state(config):
        model, state = real_initial_state(config)
        trees = (
            model.parameters(),
            state.parameters.master_parameters,
            state.parameters.working_parameters,
            state.optimizer.first_moments,
            state.optimizer.second_moments,
            state.trainer.accumulators,
        )
        initial_arrays.extend(
            weakref.ref(tree["embed_tokens"]["weight"]) for tree in trees
        )
        return model, state

    def check_published_state(run, manifest, state):
        gc.collect()
        assert [
            index
            for index, reference in enumerate(initial_arrays)
            if reference() is not None
        ] == []
        return real_publish(run, manifest, state)

    monkeypatch.setattr(pretrain, "_initial_state", track_initial_state)
    monkeypatch.setattr(pretrain, "_publish_training_state", check_published_state)

    assert pretrain.train(config).step == 2


@pytest.mark.parametrize("compile", [False, True])
def test_resume_releases_restored_device_arrays_after_update(
    tmp_path, monkeypatch, compile
):
    data = _prepared_bundle(tmp_path / "prepared")
    config = replace(
        _config(data, tmp_path / "run", maximum_steps=1),
        compile=compile,
        checkpoint=CheckpointPolicy(interval=1),
    )
    pretrain.train(config)
    restored_arrays = []
    real_restore = pretrain._restore_checkpoint
    real_publish = pretrain._publish_training_state

    def track_restore(reader):
        state = real_restore(reader)
        trees = (
            state.parameters.master_parameters,
            state.parameters.working_parameters,
            state.optimizer.first_moments,
            state.optimizer.second_moments,
            state.trainer.accumulators,
        )
        restored_arrays.extend(
            weakref.ref(array) for tree in trees for _, array in tree_flatten(tree)
        )
        return state

    def check_published_state(run, manifest, state):
        gc.collect()
        assert restored_arrays
        assert all(reference() is None for reference in restored_arrays)
        return real_publish(run, manifest, state)

    monkeypatch.setattr(pretrain, "_restore_checkpoint", track_restore)
    monkeypatch.setattr(pretrain, "_publish_training_state", check_published_state)

    result = pretrain.resume(
        config.output_run, data=data, overrides=_overrides(maximum_steps=2)
    )
    assert result.step == 2


def test_scalar_state_reads_verify_payloads_without_loading_tensors(
    tmp_path, monkeypatch
):
    data = _prepared_bundle(tmp_path / "prepared")
    run = tmp_path / "run"
    result = pretrain.train(_config(data, run, maximum_steps=1))
    resolved = resolve_latest_step(
        run, writable=False, verification=VerificationLevel.FULL
    )

    def forbid_tensor_loading(*args, **kwargs):
        raise AssertionError(
            "scalar progress reads must not deserialize model or optimizer arrays"
        )

    monkeypatch.setattr(
        checkpoint_module, "_load_checkpoint_array_stream", forbid_tensor_loading
    )
    scalar = pretrain.read_scalar_state(resolved)
    assert (scalar.step, scalar.rows, scalar.cursor.epoch) == (
        result.step,
        result.rows,
        result.epoch,
    )

    weights = resolved.step_directory / "model.safetensors"
    payload = bytearray(weights.read_bytes())
    payload[-1] ^= 1
    weights.write_bytes(payload)
    with pytest.raises(SMLArtifactError):
        pretrain.read_scalar_state(resolved)
