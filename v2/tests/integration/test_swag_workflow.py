from __future__ import annotations

import gc
import json
import os
import shutil
import weakref
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import pytest
import sml.inference as inference_module
import zstandard as zstd
from mlx.utils import tree_flatten
from sml.artifacts import checkpoint as checkpoint_module
from sml.artifacts.checkpoint import CheckpointReader, resolve_latest_step
from sml.artifacts.manifest import (
    ArraySpec,
    ArtifactRoot,
    BaseSnapshotManifest,
    ExportManifest,
    LatestIndex,
    LoRACheckpointManifest,
    LoRARunManifest,
    OpenedArtifact,
    PayloadRef,
    TokenizerManifest,
    VerificationLevel,
    canonical_json_bytes,
    file_identity,
    read_manifest,
)
from sml.artifacts.semantics import expected_next_key
from sml.artifacts.verify import verify_artifact
from sml.data.corpus import CorpusConfig
from sml.data.pretraining import (
    PretrainingPreparationConfig,
    prepare_pretraining_bundle,
)
from sml.data.swag import (
    SwagDataBundle,
    SwagPreparationConfig,
    SwagSourceConfig,
    prepare_swag_bundle,
)
from sml.data.tokenizer import TokenizerTrainingConfig, train_tokenizer_bundle
from sml.errors import SMLArtifactError, SMLConfigurationError
from sml.inference import InferenceSession, resolve_model_artifact
from sml.model.config import ModelConfig
from sml.training import swag as swag_module
from sml.training.common import (
    CheckpointPolicy,
    LoaderConfig,
    OptimizerConfig,
    PretrainingConfig,
    ResumeOverrides,
)
from sml.training.lora import (
    LoRAConfig,
    LoRAInitializerConfig,
    lora_config_from_mapping,
)
from sml.training.pretrain import train
from sml.training.swag import (
    SwagTrainingConfig,
    export_merged,
    finetune,
    resume_finetune,
)

VALID_ENDINGS = ("on the mat", "in the car", "by the door", "near a tree")


class FakeSwagProvider:
    def __init__(self, rows: tuple[Mapping[str, object], ...]) -> None:
        self.rows = rows

    def resolve(self, source):
        from sml.data.swag import ResolvedSwagSource

        return ResolvedSwagSource(
            backend=source.backend,
            namespace=source.namespace,
            name=source.name,
            dataset_config=source.dataset_config,
            revision=source.revision,
            split=source.split,
            commit="abc123def456",
            provider_fingerprint="fingerprint-v1",
            provider_package="datasets",
            provider_version="2.0.0",
        )

    def iter_rows(self, resolved) -> Iterator[Mapping[str, object]]:
        yield from self.rows


def _swag_rows(
    count: int, *, prefix: str = "the cat sat"
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "context": f"{prefix} {index}",
            "endings": VALID_ENDINGS,
            "label": index % 4,
        }
        for index in range(count)
    )


def _write_tiny_corpus(path: Path) -> Path:
    path.mkdir()
    lines = [
        json.dumps(
            {
                "text": (
                    "alpha beta gamma delta epsilon zeta eta theta " * 8
                    + f"row {index}"
                )
            }
        ).encode("utf-8")
        for index in range(24)
    ]
    payload = b"\n".join(lines) + b"\n"
    (path / "tiny-0000.jsonl.zst").write_bytes(zstd.ZstdCompressor().compress(payload))
    return path


def _corpus_config(path: Path) -> CorpusConfig:
    return CorpusConfig(
        input_root=path,
        shuffle_files=False,
        min_text_bytes=1,
        max_rows_per_file=None,
    )


def _prepare_tiny_data(root: Path):
    corpus = _write_tiny_corpus(root / "corpus")
    tokenizer = train_tokenizer_bundle(
        TokenizerTrainingConfig(
            corpus=_corpus_config(corpus),
            vocab_size=300,
            hard_vocab_limit=False,
            num_threads=1,
        ),
        root / "tokenizer",
    )
    data = prepare_pretraining_bundle(
        PretrainingPreparationConfig(
            corpus=_corpus_config(corpus),
            tokenizer_bundle=tokenizer.path,
            sequence_length=32,
            shuffle_window_rows=5,
            output_shard_rows=3,
            seed=17,
        ),
        root / "data",
    )
    return tokenizer, data


def _tiny_pretraining_config(data_path: Path, output_run: Path, vocab_size: int):
    return PretrainingConfig(
        data=data_path,
        output_run=output_run,
        model=ModelConfig(
            vocab_size=vocab_size,
            hidden_size=8,
            num_layers=1,
            num_q_heads=2,
            num_kv_heads=1,
            intermediate_size=16,
            original_context_length=32,
            rope_scaling_factor=1.0,
            hidden_dropout=0.0,
        ),
        optimizer=OptimizerConfig(
            learning_rate=0.01,
            beta1=0.5,
            beta2=0.5,
            schedule_steps=4,
            warmup_steps=0,
        ),
        loader=LoaderConfig(
            microbatch_size=1,
            gradient_accumulation_steps=1,
            prefetch_depth=2,
            epoch_seed=13,
        ),
        checkpoint=CheckpointPolicy(interval=1),
        maximum_steps=1,
        maximum_epochs=2,
        log_interval=1,
        seed=19,
        compile=False,
    )


def tiny_swag_training_config(base_run: Path, bundle, output: Path, **overrides):
    values = {
        "base_checkpoint": base_run,
        "data": bundle.path,
        "output_run": output,
        "lora": LoRAConfig(
            rank=2,
            alpha=4.0,
            scaling_mode="lora",
            dropout=0.0,
            target_modules=("q_proj", "v_proj"),
            initializer=LoRAInitializerConfig(lora_a=0.01, lora_b=0.0),
        ),
        "optimizer": OptimizerConfig(
            learning_rate=1e-4,
            schedule_steps=8_192,
        ),
        "loader": LoaderConfig(
            microbatch_size=1,
            gradient_accumulation_steps=1,
            prefetch_depth=1,
            epoch_seed=13,
        ),
        "checkpoint": CheckpointPolicy(interval=1),
        "maximum_steps": 1,
        "maximum_epochs": 5,
        "log_interval": 1,
        "seed": 19,
        "compile": False,
    }
    values.update(overrides)
    return SwagTrainingConfig(**values)


def move_to_unavailable_location(path: Path) -> None:
    gone = path.with_name(path.name + ".gone")
    path.rename(gone)
    shutil.rmtree(gone)


def _load_group(step_directory: Path, name: str) -> dict[str, mx.array]:
    return dict(mx.load(step_directory / name))


def assert_checkpoint_array_dtypes(
    step,
    *,
    adapters: mx.Dtype,
    optimizer_moments: mx.Dtype,
) -> None:
    adapter_arrays = _load_group(step.step_directory, "adapters.safetensors")
    optimizer_arrays = _load_group(step.step_directory, "optimizer.safetensors")
    trainer_arrays = _load_group(step.step_directory, "trainer.safetensors")
    assert adapter_arrays
    assert all(array.dtype == adapters for array in adapter_arrays.values())
    moment_arrays = [array for key, array in optimizer_arrays.items() if key != "step"]
    assert moment_arrays
    assert all(array.dtype == optimizer_moments for array in moment_arrays)
    assert step.checkpoint.version == 2
    assert set(trainer_arrays) == {"accumulation_count", "next_key", "loss_numerator"}
    assert trainer_arrays["accumulation_count"].dtype == mx.int32
    assert trainer_arrays["next_key"].dtype == mx.uint32
    assert trainer_arrays["loss_numerator"].dtype == mx.float32


def assert_base_snapshot_array_dtypes(base_directory: Path, *, model: mx.Dtype) -> None:
    arrays = _load_group(base_directory, "model.safetensors")
    assert arrays
    assert all(array.dtype == model for array in arrays.values())


def _latest_checkpoint_state(run: Path) -> dict[str, object]:
    step = resolve_latest_step(
        run,
        writable=False,
        verification=VerificationLevel.FULL,
    )
    return {
        "step": step.step,
        "adapters": _load_group(step.step_directory, "adapters.safetensors"),
        "optimizer": _load_group(step.step_directory, "optimizer.safetensors"),
        "trainer": _load_group(step.step_directory, "trainer.safetensors"),
        "scalar": json.loads((step.step_directory / "state.json").read_text()),
    }


def _assert_array_maps_equal(
    actual: Mapping[str, mx.array],
    expected: Mapping[str, mx.array],
) -> None:
    assert set(actual) == set(expected)
    mx.eval(*actual.values(), *expected.values())
    for name, array in actual.items():
        assert bool(mx.array_equal(array, expected[name]).item()), name


def _payload_ref(path: Path, logical_path: str) -> PayloadRef:
    with path.open("rb") as payload:
        identity = file_identity(payload)
    return PayloadRef(logical_path, identity, path.stat().st_size)


def _rewrite_bound_base_snapshot(run: Path, **changes: object) -> None:
    snapshot = read_manifest(
        run / "base",
        BaseSnapshotManifest,
        VerificationLevel.MANIFEST_TRUSTED,
    ).manifest
    snapshot = replace(snapshot, **changes)
    snapshot = replace(snapshot, identity=snapshot.recompute_identity())
    (run / "base" / BaseSnapshotManifest.MANIFEST_FILENAME).write_bytes(
        canonical_json_bytes(snapshot)
    )
    run_manifest = read_manifest(
        run,
        LoRARunManifest,
        VerificationLevel.MANIFEST_TRUSTED,
    ).manifest
    run_manifest = replace(run_manifest, base_identity=snapshot.identity)
    run_manifest = replace(run_manifest, identity=run_manifest.recompute_identity())

    _write_bound_lora_run(run, run_manifest)


def _write_bound_lora_run(run: Path, run_manifest: LoRARunManifest) -> None:
    (run / LoRARunManifest.MANIFEST_FILENAME).write_bytes(
        canonical_json_bytes(run_manifest)
    )
    rebound_latest: LoRACheckpointManifest | None = None
    latest = read_manifest(
        run, LatestIndex, VerificationLevel.MANIFEST_TRUSTED
    ).manifest
    for step_directory in (run / "checkpoints").iterdir():
        if not step_directory.is_dir():
            continue
        checkpoint = read_manifest(
            step_directory,
            LoRACheckpointManifest,
            VerificationLevel.MANIFEST_TRUSTED,
        ).manifest
        state_path = step_directory / "state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["owning_run_identity"] = run_manifest.identity
        state_path.write_bytes(canonical_json_bytes(state))
        checkpoint = replace(
            checkpoint,
            owning_run_identity=run_manifest.identity,
            scalar_state=_payload_ref(state_path, "state.json"),
        )
        checkpoint = replace(checkpoint, identity=checkpoint.recompute_identity())
        (step_directory / LoRACheckpointManifest.MANIFEST_FILENAME).write_bytes(
            canonical_json_bytes(checkpoint)
        )
        if checkpoint.step == latest.step:
            rebound_latest = checkpoint
    if rebound_latest is None:
        raise AssertionError("LoRA run has no latest checkpoint to rebind")
    latest = replace(
        latest,
        owning_run_identity=run_manifest.identity,
        checkpoint_identity=rebound_latest.identity,
    )
    latest = replace(latest, identity=latest.recompute_identity())
    (run / LatestIndex.MANIFEST_FILENAME).write_bytes(canonical_json_bytes(latest))


def _replace_latest_trainer_key(run: Path, key: mx.array) -> None:
    resolved = resolve_latest_step(
        run,
        writable=False,
        verification=VerificationLevel.FULL,
    )
    trainer_path = resolved.step_directory / "trainer.safetensors"
    trainer = _load_group(resolved.step_directory, "trainer.safetensors")
    mx.eval(*trainer.values())
    trainer["next_key"] = key
    mx.save_safetensors(trainer_path, trainer)
    assert isinstance(resolved.checkpoint, LoRACheckpointManifest)
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


@pytest.fixture(scope="module")
def _tiny_base_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("swag-workflow-base")
    tokenizer, data = _prepare_tiny_data(root)
    result = train(
        _tiny_pretraining_config(
            data.path,
            root / "run",
            tokenizer.manifest.vocab_size,
        )
    )
    return result.run


@pytest.fixture
def tiny_base_run(tmp_path: Path, _tiny_base_template: Path) -> Path:
    destination = tmp_path / "base-run"
    shutil.copytree(_tiny_base_template, destination)
    return destination


@pytest.fixture
def tiny_swag_bundle(tiny_base_run: Path, tmp_path: Path) -> Iterator[SwagDataBundle]:
    resolved = resolve_model_artifact(tiny_base_run, full_verify=True)
    bundle = prepare_swag_bundle(
        SwagPreparationConfig(
            provider=FakeSwagProvider(_swag_rows(5)),
            source=SwagSourceConfig(revision="deadbeef" * 5),
            maximum_length=32,
            bucket_boundaries=(16, 32),
        ),
        resolved,
        tmp_path / "swag-bundle",
    )
    try:
        yield bundle
    finally:
        bundle.close()


@pytest.fixture
def tiny_lora_run(tiny_base_run: Path, tiny_swag_bundle, tmp_path: Path) -> Path:
    trained = finetune(
        tiny_swag_training_config(
            tiny_base_run,
            tiny_swag_bundle,
            tmp_path / "tiny-lora-run",
            maximum_steps=1,
        )
    )
    return trained.run


def test_lora_run_is_self_contained_after_source_run_removal(
    tiny_base_run, tiny_swag_bundle, tmp_path
):
    trained = finetune(
        tiny_swag_training_config(
            tiny_base_run,
            tiny_swag_bundle,
            tmp_path / "lora-run",
            maximum_steps=1,
        )
    )
    move_to_unavailable_location(tiny_base_run)
    resumed = resume_finetune(
        trained.run,
        data=tiny_swag_bundle.path,
        overrides=ResumeOverrides(maximum_steps=2),
    )
    session = InferenceSession.from_checkpoint(resumed.run, full_verify=True)
    exported = export_merged(resumed.run, tmp_path / "export")
    export_session = InferenceSession.from_checkpoint(exported.path, full_verify=True)
    assert resumed.step == 2
    assert session.model_identity.artifact_kind == "lora-run"
    assert session.resolved_model.model_config.rope_scaling_factor == 1.0
    assert export_session.model_identity.artifact_kind == "export"
    assert export_session.resolved_model.model_config.rope_scaling_factor == 1.0


def test_lora_checkpoint_omits_frozen_base(tiny_lora_run):
    step = resolve_latest_step(
        tiny_lora_run,
        writable=False,
        verification=VerificationLevel.FULL,
    )
    assert {path.name for path in step.step_directory.iterdir()} == {
        "adapters.safetensors",
        "optimizer.safetensors",
        "trainer.safetensors",
        "state.json",
        "checkpoint.json",
    }
    assert (tiny_lora_run / "base/model.safetensors").is_file()
    assert_checkpoint_array_dtypes(
        step,
        adapters=mx.float32,
        optimizer_moments=mx.float32,
    )
    assert_base_snapshot_array_dtypes(tiny_lora_run / "base", model=mx.bfloat16)
    verified = read_manifest(
        tiny_lora_run / "base",
        BaseSnapshotManifest,
        VerificationLevel.FULL,
    )
    assert verified.manifest.precision.get("working_parameter_dtype") == "bfloat16"
    assert verified.manifest.precision.get("master_weights") is True
    assert "adapter_parameter_dtype" not in verified.manifest.precision


def test_legacy_lora_checkpoint_resumes_identically_and_saves_compact_state(
    tiny_lora_run, tiny_swag_bundle, tmp_path
):
    legacy_run = tmp_path / "legacy-lora-run"
    shutil.copytree(tiny_lora_run, legacy_run)
    resolved = resolve_latest_step(
        legacy_run, writable=False, verification=VerificationLevel.FULL
    )
    checkpoint = resolved.checkpoint
    assert isinstance(checkpoint, LoRACheckpointManifest)
    assert checkpoint.version == 2
    trainer_path = resolved.step_directory / "trainer.safetensors"
    trainer = _load_group(resolved.step_directory, "trainer.safetensors")
    specs = {spec.name: spec for spec in checkpoint.trainer.arrays}
    adapters = _load_group(resolved.step_directory, "adapters.safetensors")
    mx.eval(trainer, adapters)
    for name, array in adapters.items():
        accumulator_name = f"accumulators.{name}"
        trainer[accumulator_name] = mx.zeros_like(array)
        specs[accumulator_name] = ArraySpec(accumulator_name, array.shape, "float32")
    mx.save_safetensors(trainer_path, trainer)
    checkpoint = replace(
        checkpoint,
        version=1,
        trainer=replace(
            checkpoint.trainer,
            payload=_payload_ref(trainer_path, "trainer.safetensors"),
            arrays=tuple(specs[name] for name in sorted(specs)),
        ),
    )
    checkpoint = replace(checkpoint, identity=checkpoint.recompute_identity())
    (resolved.step_directory / "checkpoint.json").write_bytes(
        canonical_json_bytes(checkpoint)
    )
    latest = read_manifest(
        legacy_run, LatestIndex, VerificationLevel.MANIFEST_TRUSTED
    ).manifest
    latest = replace(latest, checkpoint_identity=checkpoint.identity)
    latest = replace(latest, identity=latest.recompute_identity())
    (legacy_run / "latest.json").write_bytes(canonical_json_bytes(latest))
    verify_artifact(legacy_run, full=True)

    for run in (tiny_lora_run, legacy_run):
        resumed = resume_finetune(
            run,
            data=tiny_swag_bundle.path,
            overrides=ResumeOverrides(maximum_steps=resolved.step + 1),
        )
        assert resumed.step == resolved.step + 1
        latest = resolve_latest_step(
            run, writable=False, verification=VerificationLevel.FULL
        )
        assert latest.checkpoint.version == 2
        assert {spec.name for spec in latest.checkpoint.trainer.arrays} == {
            "accumulation_count",
            "next_key",
            "loss_numerator",
        }
    compact_state = _latest_checkpoint_state(tiny_lora_run)
    legacy_state = _latest_checkpoint_state(legacy_run)
    assert legacy_state["scalar"] == compact_state["scalar"]
    for group in ("adapters", "optimizer", "trainer"):
        _assert_array_maps_equal(legacy_state[group], compact_state[group])


def test_full_lora_resolve_applies_exact_run_semantics(tiny_lora_run: Path) -> None:
    run = read_manifest(
        tiny_lora_run,
        LoRARunManifest,
        VerificationLevel.MANIFEST_TRUSTED,
    ).manifest
    optimizer = dict(run.optimizer)
    optimizer["beta1"] = 1.0
    rebound = replace(run, optimizer=optimizer)
    rebound = replace(rebound, identity=rebound.recompute_identity())
    _write_bound_lora_run(tiny_lora_run, rebound)

    with pytest.raises(SMLArtifactError, match="invalid run optimizer"):
        resolve_model_artifact(tiny_lora_run, full_verify=True)


@pytest.mark.parametrize("full", (False, True))
def test_recursive_lora_run_verification_reports_recovered_latest(
    tiny_lora_run: Path,
    full: bool,
) -> None:
    """A proved recovered LoRA winner remains a successful public result."""
    (tiny_lora_run / "latest.json").unlink()

    result = verify_artifact(tiny_lora_run, full=full)

    assert result.latest_recovered is True
    assert result.pruning_pending is False


def test_export_rejects_resigned_progress_outside_step_bounds_before_publication(
    tiny_lora_run: Path,
    tmp_path: Path,
) -> None:
    """Merged derivatives require valid source progress, not only valid arrays."""
    resolved = resolve_latest_step(
        tiny_lora_run,
        writable=False,
        verification=VerificationLevel.FULL,
    )
    state_path = resolved.step_directory / resolved.checkpoint.scalar_state.logical_path
    state = json.loads(state_path.read_bytes())
    state["microsteps"] = 0
    state["examples"] = 0
    state_path.write_bytes(canonical_json_bytes(state))
    checkpoint = replace(
        resolved.checkpoint,
        scalar_state=_payload_ref(state_path, "state.json"),
    )
    checkpoint = replace(checkpoint, identity=checkpoint.recompute_identity())
    (resolved.step_directory / "checkpoint.json").write_bytes(
        canonical_json_bytes(checkpoint)
    )
    latest = read_manifest(
        tiny_lora_run,
        LatestIndex,
        VerificationLevel.MANIFEST_TRUSTED,
    ).manifest
    latest = replace(latest, checkpoint_identity=checkpoint.identity)
    latest = replace(latest, identity=latest.recompute_identity())
    (tiny_lora_run / "latest.json").write_bytes(canonical_json_bytes(latest))
    output = tmp_path / "invalid-progress-export"

    with pytest.raises(SMLArtifactError, match="microsteps disagree"):
        export_merged(tiny_lora_run, output)

    assert not output.exists()


def test_export_rejects_resigned_wrong_terminal_key_before_publication(
    tiny_lora_run: Path,
    tmp_path: Path,
) -> None:
    """Merged derivatives require the seed-derived terminal trainer key."""
    _replace_latest_trainer_key(tiny_lora_run, mx.random.key(999))
    output = tmp_path / "wrong-key-export"

    with pytest.raises(SMLArtifactError, match="next RNG key is incorrect"):
        export_merged(tiny_lora_run, output)

    assert not output.exists()


def test_export_materializes_only_adapter_group(
    tiny_lora_run: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FULL export streams training-only state without materializing its tensors."""
    real_load = checkpoint_module._load_checkpoint_array_stream

    def reject_training_only_materialization(stream, reference, **kwargs):
        if reference.payload.logical_path != "adapters.safetensors":
            raise AssertionError(
                f"materialized training-only group: {reference.payload.logical_path}"
            )
        return real_load(stream, reference, **kwargs)

    monkeypatch.setattr(
        checkpoint_module,
        "_load_checkpoint_array_stream",
        reject_training_only_materialization,
    )

    exported = export_merged(tiny_lora_run, tmp_path / "adapter-only-export")

    assert exported.path.is_dir()


@pytest.mark.parametrize("full_verify", (False, True))
def test_export_model_identity_distinguishes_changed_weights(
    tiny_lora_run: Path,
    tmp_path: Path,
    full_verify: bool,
) -> None:
    first = export_merged(tiny_lora_run, tmp_path / "first-export").path
    second = tmp_path / "second-export"
    shutil.copytree(first, second)
    first_manifest = read_manifest(
        first, ExportManifest, VerificationLevel.FULL
    ).manifest
    weights_path = second / "model.safetensors"
    arrays = dict(mx.load(weights_path))
    arrays["norm.weight"] = (arrays["norm.weight"] + 1.0).astype(mx.bfloat16)
    mx.save_safetensors(weights_path, arrays)
    second_manifest = replace(
        first_manifest,
        model_weights=replace(
            first_manifest.model_weights,
            payload=_payload_ref(weights_path, "model.safetensors"),
        ),
    )
    second_manifest = replace(
        second_manifest, identity=second_manifest.recompute_identity()
    )
    (second / "manifest.json").write_bytes(canonical_json_bytes(second_manifest))

    first_identity = resolve_model_artifact(first, full_verify=full_verify).identity()
    second_identity = resolve_model_artifact(second, full_verify=full_verify).identity()

    assert first_identity.tokenizer_identity == second_identity.tokenizer_identity
    assert first_identity.step == second_identity.step
    assert first_identity.artifact_identity == first_manifest.identity
    assert second_identity.artifact_identity == second_manifest.identity
    assert first_identity != second_identity


def test_full_export_resolve_applies_exact_export_semantics(
    tiny_lora_run: Path,
    tmp_path: Path,
) -> None:
    exported = export_merged(tiny_lora_run, tmp_path / "semantic-export")
    manifest = read_manifest(
        exported.path,
        ExportManifest,
        VerificationLevel.MANIFEST_TRUSTED,
    ).manifest
    precision = dict(manifest.precision)
    precision["adapter_parameter_dtype"] = "bfloat16"
    rebound = replace(manifest, precision=precision)
    rebound = replace(rebound, identity=rebound.recompute_identity())
    (exported.path / ExportManifest.MANIFEST_FILENAME).write_bytes(
        canonical_json_bytes(rebound)
    )

    with pytest.raises(SMLArtifactError, match="invalid export precision"):
        resolve_model_artifact(exported.path, full_verify=True)


@pytest.mark.parametrize(
    ("hidden_dropout", "lora_dropout"),
    ((0.0, 0.5), (0.2, 0.5), (0.0, 0.0)),
    ids=("lora-only", "mixed", "disabled"),
)
def test_uninterrupted_and_interrupted_adapter_state_match(
    tmp_path,
    hidden_dropout,
    lora_dropout,
):
    tokenizer, data = _prepare_tiny_data(tmp_path)
    base_config = _tiny_pretraining_config(
        data.path,
        tmp_path / "base-run",
        tokenizer.manifest.vocab_size,
    )
    base_config = replace(
        base_config,
        model=replace(base_config.model, hidden_dropout=hidden_dropout),
    )
    base_run = train(base_config).run
    resolved_base = resolve_model_artifact(base_run, full_verify=True)
    bundle = prepare_swag_bundle(
        SwagPreparationConfig(
            provider=FakeSwagProvider(_swag_rows(5)),
            source=SwagSourceConfig(revision="deadbeef" * 5),
            maximum_length=32,
            bucket_boundaries=(16, 32),
        ),
        resolved_base,
        tmp_path / "swag-bundle",
    )
    lora = LoRAConfig(
        rank=3,
        alpha=1.0,
        scaling_mode="lora",
        dropout=lora_dropout,
        target_modules=("q_proj", "v_proj"),
        initializer=LoRAInitializerConfig(lora_a=0.05, lora_b=0.05),
    )
    try:
        uninterrupted = finetune(
            tiny_swag_training_config(
                base_run,
                bundle,
                tmp_path / "uninterrupted",
                lora=lora,
                maximum_steps=2,
                compile=True,
            )
        )
        first = finetune(
            tiny_swag_training_config(
                base_run,
                bundle,
                tmp_path / "interrupted",
                lora=lora,
                maximum_steps=1,
                compile=True,
            )
        )
        resumed = resume_finetune(
            first.run,
            data=bundle.path,
            overrides=ResumeOverrides(maximum_steps=2),
        )
    finally:
        bundle.close()
    left = _latest_checkpoint_state(uninterrupted.run)
    right = _latest_checkpoint_state(resumed.run)
    assert left["step"] == right["step"] == 2
    assert left["scalar"]["cursor"] == right["scalar"]["cursor"]
    assert left["scalar"]["step"] == right["scalar"]["step"]
    assert "step" in left["optimizer"]
    assert "next_key" in left["trainer"]
    _assert_array_maps_equal(left["adapters"], right["adapters"])
    _assert_array_maps_equal(left["optimizer"], right["optimizer"])
    _assert_array_maps_equal(left["trainer"], right["trainer"])
    for run, state in ((uninterrupted.run, left), (resumed.run, right)):
        manifest = read_manifest(
            run,
            LoRARunManifest,
            VerificationLevel.MANIFEST_TRUSTED,
        ).manifest
        expected = expected_next_key(
            seed=int(manifest.checkpoint["seed"]),
            microsteps=state["scalar"]["microsteps"],
            model=ModelConfig(**dict(manifest.model)),
            lora=lora_config_from_mapping(dict(manifest.lora)),
        )
        mx.eval(expected, state["trainer"]["next_key"])
        assert bool(mx.array_equal(state["trainer"]["next_key"], expected))


@pytest.mark.parametrize("maximum_steps", (1, 2))
@pytest.mark.parametrize("mutation", ("tokenizer.model", "tokenizer.vocab", "manifest"))
def test_resume_rejects_invalid_tokenizer_before_restore_or_retention(
    mutation,
    maximum_steps,
    tiny_base_run,
    tiny_swag_bundle,
    tmp_path,
    monkeypatch,
):
    run = _run_with_unpruned_lora_history(
        tiny_base_run,
        tiny_swag_bundle,
        tmp_path / "invalid-tokenizer-run",
        monkeypatch,
    )
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

    monkeypatch.setattr(swag_module, "_restore_adapter_checkpoint", forbidden_restore)
    with pytest.raises(SMLArtifactError, match=expected_error):
        resume_finetune(
            run,
            data=tiny_swag_bundle.path,
            overrides=ResumeOverrides(maximum_steps=maximum_steps),
        )

    assert {
        path.relative_to(run): path.read_bytes()
        for path in run.rglob("*")
        if path.is_file()
    } == before


def test_resume_rejects_wrong_key_before_runtime_or_retention(
    tiny_base_run,
    tiny_swag_bundle,
    tmp_path,
    monkeypatch,
):
    trained = finetune(
        tiny_swag_training_config(
            tiny_base_run,
            tiny_swag_bundle,
            tmp_path / "wrong-key-run",
            lora=LoRAConfig(
                dropout=0.5,
                target_modules=("q_proj", "v_proj"),
            ),
            maximum_steps=1,
        )
    )
    _replace_latest_trainer_key(trained.run, mx.random.key(999))
    reached: list[str] = []

    def forbidden(name):
        def call(*_args, **_kwargs):
            reached.append(name)
            raise AssertionError(f"{name} reached before RNG validation")

        return call

    monkeypatch.setattr(swag_module, "SMLLanguageModel", forbidden("model"))
    monkeypatch.setattr(swag_module, "SwagBatchStream", forbidden("stream"))
    monkeypatch.setattr(swag_module, "prune_to_latest", forbidden("retention"))
    monkeypatch.setattr(
        swag_module,
        "_publish_training_state",
        forbidden("publication"),
    )

    with pytest.raises(
        SMLArtifactError,
        match="checkpoint trainer next RNG key is incorrect",
    ):
        resume_finetune(
            trained.run,
            data=tiny_swag_bundle.path,
            overrides=ResumeOverrides(maximum_steps=2),
        )
    assert reached == []


def test_lora_microbatch_progress_supports_full_checkpoint_consumers(
    tiny_base_run,
    tiny_swag_bundle,
    tmp_path,
) -> None:
    """Accumulated LoRA microbatches form a complete, verifiable optimizer step."""
    trained = finetune(
        tiny_swag_training_config(
            tiny_base_run,
            tiny_swag_bundle,
            tmp_path / "microbatch-progress-run",
            loader=LoaderConfig(
                microbatch_size=2,
                gradient_accumulation_steps=2,
                prefetch_depth=1,
                epoch_seed=13,
            ),
            maximum_steps=1,
        )
    )

    resolved = resolve_latest_step(
        trained.run,
        writable=False,
        verification=VerificationLevel.FULL,
    )
    state = json.loads((resolved.step_directory / "state.json").read_text())
    assert state["step"] == 1
    assert state["microsteps"] == 2
    assert 2 <= state["examples"] <= 4
    verify_artifact(trained.run, full=True)
    InferenceSession.from_checkpoint(trained.run, full_verify=True)

    resumed = resume_finetune(
        trained.run,
        data=tiny_swag_bundle.path,
        overrides=ResumeOverrides(maximum_steps=2),
    )
    assert resumed.run == trained.run
    assert resumed.step == 2
    resumed_resolved = resolve_latest_step(
        resumed.run,
        writable=False,
        verification=VerificationLevel.FULL,
    )
    resumed_state = json.loads(
        (resumed_resolved.step_directory / "state.json").read_text()
    )
    assert resumed_resolved.step == resumed_state["step"] == 2
    verify_artifact(resumed.run, full=True)


def test_swag_accumulates_microbatches_and_logs_tail_metrics(
    tiny_base_run, tiny_swag_bundle, tmp_path, monkeypatch, capsys
) -> None:
    updates = []
    original_step = swag_module.SwagKernels.optimizer_step

    def record_step(kernels, adapters, optimizer, trainer):
        updates.append(
            (
                int(trainer.valid_count.item()),
                float((trainer.loss_numerator / trainer.valid_count).item()),
                float((trainer.correct_count / trainer.valid_count).item()),
                float(
                    swag_module.learning_rate_at(
                        optimizer.step, kernels.optimizer_config
                    ).item()
                ),
            )
        )
        return original_step(kernels, adapters, optimizer, trainer)

    monkeypatch.setattr(swag_module.SwagKernels, "optimizer_step", record_step)
    capsys.readouterr()
    trained = finetune(
        tiny_swag_training_config(
            tiny_base_run,
            tiny_swag_bundle,
            tmp_path / "accumulation-logs-run",
            loader=LoaderConfig(microbatch_size=2, gradient_accumulation_steps=2),
            maximum_steps=None,
            maximum_epochs=1,
            log_interval=2,
        )
    )
    output = capsys.readouterr()
    assert trained.step == 2
    assert trained.examples == 5
    assert [update[0] for update in updates] == [4, 1]
    lines = output.err.splitlines()
    assert output.out == ""
    assert len(lines) == 1
    assert lines[0].startswith("swag step=2 epoch=1 examples=5 ")
    metrics = dict(field.split("=", 1) for field in lines[0].split()[1:])
    assert float(metrics["loss"]) == pytest.approx(updates[-1][1], abs=1e-6)
    assert float(metrics["accuracy"]) == pytest.approx(updates[-1][2], abs=1e-6)
    assert float(metrics["learning_rate"]) == pytest.approx(updates[-1][3], rel=1e-5)


def test_resume_rejects_mismatches_before_allocation(
    tiny_base_run, tiny_swag_bundle, tmp_path, monkeypatch
):
    trained = finetune(
        tiny_swag_training_config(
            tiny_base_run,
            tiny_swag_bundle,
            tmp_path / "mismatch-run",
            maximum_steps=1,
        )
    )
    other_bundle = prepare_swag_bundle(
        SwagPreparationConfig(
            provider=FakeSwagProvider(_swag_rows(5, prefix="the dog ran")),
            source=SwagSourceConfig(revision="deadbeef" * 5),
            maximum_length=32,
            bucket_boundaries=(16, 32),
        ),
        resolve_model_artifact(tiny_base_run, full_verify=True),
        tmp_path / "other-swag",
    )
    other_bundle_path = other_bundle.path
    other_bundle.close()
    allocated: list[str] = []

    def forbidden(name: str):
        def _forbidden(*_args, **_kwargs):
            allocated.append(name)
            raise AssertionError(f"{name} was constructed")

        return _forbidden

    monkeypatch.setattr(swag_module, "apply_lora", forbidden("apply_lora"))
    monkeypatch.setattr(
        swag_module, "initialize_adam_state", forbidden("initialize_adam_state")
    )
    monkeypatch.setattr(
        swag_module, "build_swag_kernels", forbidden("build_swag_kernels")
    )
    monkeypatch.setattr(swag_module, "SwagBatchStream", forbidden("SwagBatchStream"))
    monkeypatch.setattr(swag_module, "SMLLanguageModel", forbidden("SMLLanguageModel"))

    with pytest.raises((SMLArtifactError, SMLConfigurationError)):
        resume_finetune(
            trained.run,
            data=other_bundle_path,
            overrides=ResumeOverrides(maximum_steps=2),
        )
    assert allocated == []

    model_run = tmp_path / "mismatch-model-run"
    shutil.copytree(trained.run, model_run)
    model = dict(
        read_manifest(
            model_run / "base",
            BaseSnapshotManifest,
            VerificationLevel.MANIFEST_TRUSTED,
        ).manifest.model
    )
    model["hidden_size"] = 16
    _rewrite_bound_base_snapshot(model_run, model=model)
    with pytest.raises(SMLArtifactError, match="model configuration"):
        resume_finetune(
            model_run,
            data=tiny_swag_bundle.path,
            overrides=ResumeOverrides(maximum_steps=2),
        )
    assert allocated == []

    precision_run = tmp_path / "mismatch-precision-run"
    shutil.copytree(trained.run, precision_run)
    precision = dict(
        read_manifest(
            precision_run / "base",
            BaseSnapshotManifest,
            VerificationLevel.MANIFEST_TRUSTED,
        ).manifest.precision
    )
    precision["working_parameter_dtype"] = "float32"
    _rewrite_bound_base_snapshot(precision_run, precision=precision)
    with pytest.raises(SMLArtifactError, match="canonical pretraining precision"):
        resume_finetune(
            precision_run,
            data=tiny_swag_bundle.path,
            overrides=ResumeOverrides(maximum_steps=2),
        )
    assert allocated == []

    tokenizer_run = tmp_path / "mismatch-tokenizer-run"
    shutil.copytree(trained.run, tokenizer_run)
    _rewrite_bound_base_snapshot(
        tokenizer_run,
        tokenizer_identity="sha256:" + "a" * 64,
    )
    with pytest.raises(SMLArtifactError, match="tokenizer identity"):
        resume_finetune(
            tokenizer_run,
            data=tiny_swag_bundle.path,
            overrides=ResumeOverrides(maximum_steps=2),
        )
    assert allocated == []

    step = resolve_latest_step(
        trained.run,
        writable=False,
        verification=VerificationLevel.FULL,
    )
    adapters = _load_group(step.step_directory, "adapters.safetensors")
    mx.save_safetensors(
        str(step.step_directory / "adapters.safetensors"),
        {name: array.astype(mx.bfloat16) for name, array in adapters.items()},
    )
    with pytest.raises((SMLArtifactError, SMLConfigurationError)):
        resume_finetune(
            trained.run,
            data=tiny_swag_bundle.path,
            overrides=ResumeOverrides(maximum_steps=2),
        )
    assert allocated == []


def test_limit_satisfied_resume_returns_before_iterator_or_kernel_construction(
    tiny_base_run, tiny_swag_bundle, tmp_path, monkeypatch
):
    completed = finetune(
        tiny_swag_training_config(
            tiny_base_run,
            tiny_swag_bundle,
            tmp_path / "completed-run",
            maximum_steps=1,
        )
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("runtime construction was reached")

    monkeypatch.setattr(swag_module, "SwagBatchStream", forbidden)
    monkeypatch.setattr(swag_module, "SMLLanguageModel", forbidden)
    monkeypatch.setattr(swag_module, "build_swag_kernels", forbidden)

    result = resume_finetune(
        completed.run,
        data=tiny_swag_bundle.path,
        overrides=ResumeOverrides(maximum_steps=1),
    )
    assert result.step == completed.step == 1
    assert result.run == completed.run


def _run_with_unpruned_lora_history(
    tiny_base_run: Path,
    tiny_swag_bundle: SwagDataBundle,
    run: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    def interrupt_retention(*_args, **_kwargs):
        raise RuntimeError("retention interrupted")

    with (
        monkeypatch.context() as interrupted,
        pytest.raises(RuntimeError, match="retention interrupted"),
    ):
        interrupted.setattr(checkpoint_module, "_prune_to_latest", interrupt_retention)
        finetune(
            tiny_swag_training_config(
                tiny_base_run,
                tiny_swag_bundle,
                run,
                maximum_steps=1,
            )
        )
    assert sorted(path.name for path in (run / "checkpoints").iterdir()) == [
        "step-000000000",
        "step-000000001",
    ]
    return run


@pytest.mark.parametrize("full_verify", (False, True))
def test_lora_resolution_reports_recovery_with_pending_pruning(
    tiny_base_run,
    tiny_swag_bundle,
    tmp_path,
    monkeypatch,
    full_verify,
):
    """Recovered selection and unfinished retention are independent status bits."""
    run = _run_with_unpruned_lora_history(
        tiny_base_run,
        tiny_swag_bundle,
        tmp_path / f"recovered-unpruned-{full_verify}",
        monkeypatch,
    )
    (run / "latest.json").unlink()

    resolved = resolve_model_artifact(run, full_verify=full_verify)

    assert resolved.latest_recovered is True
    assert resolved.pruning_pending is True
    assert resolved.identity().latest_recovered is True
    assert resolved.identity().pruning_pending is True


def test_limit_satisfied_resume_prunes_stale_history_after_full_reader_closes(
    tiny_base_run,
    tiny_swag_bundle,
    tmp_path,
    monkeypatch,
):
    run = _run_with_unpruned_lora_history(
        tiny_base_run,
        tiny_swag_bundle,
        tmp_path / "stale-limit-run",
        monkeypatch,
    )
    real_open = swag_module.open_latest_checkpoint_reader
    real_prune = swag_module.prune_to_latest
    full_readers: list[CheckpointReader] = []
    prune_calls = 0

    @contextmanager
    def record_reader(*args, **kwargs):
        with real_open(*args, **kwargs) as reader:
            if kwargs.get("verification") is VerificationLevel.FULL:
                full_readers.append(reader)
            yield reader

    def prune_after_reader_close(*args, **kwargs):
        nonlocal prune_calls
        prune_calls += 1
        assert len(full_readers) == 1
        assert full_readers[0]._owned_step.descriptor == -1
        return real_prune(*args, **kwargs)

    monkeypatch.setattr(swag_module, "open_latest_checkpoint_reader", record_reader)
    monkeypatch.setattr(swag_module, "prune_to_latest", prune_after_reader_close)

    result = resume_finetune(
        run,
        data=tiny_swag_bundle.path,
        overrides=ResumeOverrides(maximum_steps=1),
    )

    assert result.step == 1
    assert prune_calls == 1
    assert sorted(path.name for path in (run / "checkpoints").iterdir()) == [
        "step-000000001"
    ]


def test_resume_prunes_stale_history_before_continuing_model_failure(
    tiny_base_run,
    tiny_swag_bundle,
    tmp_path,
    monkeypatch,
):
    run = _run_with_unpruned_lora_history(
        tiny_base_run,
        tiny_swag_bundle,
        tmp_path / "stale-continuing-run",
        monkeypatch,
    )
    real_open = swag_module.open_latest_checkpoint_reader
    real_prune = swag_module.prune_to_latest
    full_readers: list[CheckpointReader] = []
    order: list[str] = []

    @contextmanager
    def record_reader(*args, **kwargs):
        with real_open(*args, **kwargs) as reader:
            if kwargs.get("verification") is VerificationLevel.FULL:
                full_readers.append(reader)
            yield reader

    def record_prune(*args, **kwargs):
        assert len(full_readers) == 1
        assert full_readers[0]._owned_step.descriptor == -1
        retained = real_prune(*args, **kwargs)
        order.append("retention")
        return retained

    def fail_model_construction(*_args, **_kwargs):
        order.append("model")
        raise RuntimeError("model construction failed")

    monkeypatch.setattr(swag_module, "open_latest_checkpoint_reader", record_reader)
    monkeypatch.setattr(swag_module, "prune_to_latest", record_prune)
    monkeypatch.setattr(swag_module, "_wrap_copied_base", fail_model_construction)

    with pytest.raises(RuntimeError, match="model construction failed"):
        resume_finetune(
            run,
            data=tiny_swag_bundle.path,
            overrides=ResumeOverrides(maximum_steps=2),
        )

    assert order == ["retention", "model"]
    assert sorted(path.name for path in (run / "checkpoints").iterdir()) == [
        "step-000000001"
    ]


def test_finetune_releases_selected_base_before_training(
    tiny_base_run, tiny_swag_bundle, tmp_path, monkeypatch
):
    real_select = swag_module._select_pretraining_base
    real_run = swag_module._run_training
    selections = []

    class TrackedSelection:
        def __init__(self, selected):
            self.selected = selected

        def __getattr__(self, name):
            return getattr(self.selected, name)

    def track_selection(*args, **kwargs):
        selected = TrackedSelection(real_select(*args, **kwargs))
        selections.append(weakref.ref(selected))
        return selected

    def check_training(*args, **kwargs):
        gc.collect()
        assert len(selections) == 1
        assert selections[0]() is None
        return real_run(*args, **kwargs)

    monkeypatch.setattr(swag_module, "_select_pretraining_base", track_selection)
    monkeypatch.setattr(swag_module, "_run_training", check_training)

    result = finetune(
        tiny_swag_training_config(
            tiny_base_run, tiny_swag_bundle, tmp_path / "released-base-run"
        )
    )

    assert result.step == 1


@pytest.mark.parametrize("compile", (False, True))
@pytest.mark.parametrize("resume", (False, True), ids=("fresh", "resume"))
def test_swag_releases_superseded_device_arrays_after_update(
    tiny_base_run, tiny_swag_bundle, tmp_path, monkeypatch, compile, resume
):
    config = tiny_swag_training_config(
        tiny_base_run,
        tiny_swag_bundle,
        tmp_path / "released-state-run",
        maximum_steps=2,
        compile=compile,
    )
    config = replace(config, lora=replace(config.lora, dropout=0.2))
    if resume:
        finetune(replace(config, maximum_steps=1))

    real_wrap = swag_module._wrap_copied_base
    real_run = swag_module._run_training
    real_publish = swag_module._publish_training_state
    original_arrays = []
    published_steps = []

    def track_arrays(tree):
        original_arrays.extend(weakref.ref(array) for _, array in tree_flatten(tree))

    def track_wrapped_base(*args, **kwargs):
        model, adapters, frozen = real_wrap(*args, **kwargs)
        track_arrays(adapters)
        return model, adapters, frozen

    def track_training(run, manifest, config, model, restored, bundle):
        track_arrays(restored.adapters)
        track_arrays(restored.optimizer.first_moments)
        track_arrays(restored.optimizer.second_moments)
        track_arrays(restored.trainer.accumulators)
        return real_run(run, manifest, config, model, restored, bundle)

    def check_publication(run, manifest, state):
        gc.collect()
        assert original_arrays
        assert all(reference() is None for reference in original_arrays)
        published_steps.append(state.scalar.step)
        return real_publish(run, manifest, state)

    monkeypatch.setattr(swag_module, "_wrap_copied_base", track_wrapped_base)
    monkeypatch.setattr(swag_module, "_run_training", track_training)
    monkeypatch.setattr(swag_module, "_publish_training_state", check_publication)

    if resume:
        result = resume_finetune(
            config.output_run,
            data=tiny_swag_bundle.path,
            overrides=ResumeOverrides(maximum_steps=2),
        )
    else:
        result = finetune(config)

    assert result.step == 2
    assert published_steps == ([2] if resume else [1, 2])


def test_finetune_closes_swag_bundle_exactly_once_on_success(
    tiny_base_run, tiny_swag_bundle, tmp_path, monkeypatch
):
    original_verified = swag_module._verified_swag
    original_close = SwagDataBundle.close
    opened: list[SwagDataBundle] = []
    close_counts: dict[int, int] = {}

    def record_verified(*args, **kwargs):
        bundle = original_verified(*args, **kwargs)
        opened.append(bundle)
        return bundle

    def record_close(bundle):
        close_counts[id(bundle)] = close_counts.get(id(bundle), 0) + 1
        original_close(bundle)

    monkeypatch.setattr(swag_module, "_verified_swag", record_verified)
    monkeypatch.setattr(SwagDataBundle, "close", record_close)
    finetune(
        tiny_swag_training_config(
            tiny_base_run,
            tiny_swag_bundle,
            tmp_path / "close-success-run",
            maximum_steps=1,
        )
    )

    assert len(opened) == 1
    assert opened[0]._closed
    assert close_counts == {id(opened[0]): 1}


@pytest.mark.parametrize("failure_stage", ("setup", "training"))
def test_finetune_closes_swag_bundle_when_setup_or_training_fails(
    tiny_base_run, tiny_swag_bundle, tmp_path, monkeypatch, failure_stage
):
    original_verified = swag_module._verified_swag
    original_close = SwagDataBundle.close
    opened: list[SwagDataBundle] = []
    close_counts: dict[int, int] = {}

    def record_verified(*args, **kwargs):
        bundle = original_verified(*args, **kwargs)
        opened.append(bundle)
        return bundle

    def fail(*_args, **_kwargs):
        raise RuntimeError(f"{failure_stage} failed")

    def record_close(bundle):
        close_counts[id(bundle)] = close_counts.get(id(bundle), 0) + 1
        original_close(bundle)

    monkeypatch.setattr(swag_module, "_verified_swag", record_verified)
    monkeypatch.setattr(SwagDataBundle, "close", record_close)
    monkeypatch.setattr(
        swag_module,
        "publish_run" if failure_stage == "setup" else "_run_training",
        fail,
    )
    with pytest.raises(RuntimeError, match=f"{failure_stage} failed"):
        finetune(
            tiny_swag_training_config(
                tiny_base_run,
                tiny_swag_bundle,
                tmp_path / f"close-{failure_stage}-run",
                maximum_steps=1,
            )
        )

    assert len(opened) == 1
    assert opened[0]._closed
    assert close_counts == {id(opened[0]): 1}


def test_limit_satisfied_resume_closes_swag_bundle(
    tiny_base_run, tiny_swag_bundle, tmp_path, monkeypatch
):
    completed = finetune(
        tiny_swag_training_config(
            tiny_base_run,
            tiny_swag_bundle,
            tmp_path / "close-resume-run",
            maximum_steps=1,
        )
    )
    original_verified = swag_module._verified_swag
    original_close = SwagDataBundle.close
    opened: list[SwagDataBundle] = []
    close_counts: dict[int, int] = {}

    def record_verified(*args, **kwargs):
        bundle = original_verified(*args, **kwargs)
        opened.append(bundle)
        return bundle

    def record_close(bundle):
        close_counts[id(bundle)] = close_counts.get(id(bundle), 0) + 1
        original_close(bundle)

    monkeypatch.setattr(swag_module, "_verified_swag", record_verified)
    monkeypatch.setattr(SwagDataBundle, "close", record_close)
    result = resume_finetune(
        completed.run,
        data=tiny_swag_bundle.path,
        overrides=ResumeOverrides(maximum_steps=1),
    )

    assert result == completed
    assert len(opened) == 1
    assert opened[0]._closed
    assert close_counts == {id(opened[0]): 1}


def test_resume_uses_manifest_data_locator_when_data_is_omitted(
    tiny_base_run, tiny_swag_bundle, tmp_path
):
    completed = finetune(
        tiny_swag_training_config(
            tiny_base_run,
            tiny_swag_bundle,
            tmp_path / "locator-run",
            maximum_steps=1,
        )
    )

    result = resume_finetune(
        completed.run,
        data=None,
        overrides=ResumeOverrides(maximum_steps=1),
    )

    assert result == completed


def test_export_uses_manifest_kind_and_recovered_latest(
    tiny_base_run, tiny_swag_bundle, tmp_path
):
    trained = finetune(
        tiny_swag_training_config(
            tiny_base_run,
            tiny_swag_bundle,
            tmp_path / "step-export-run",
            maximum_steps=2,
        )
    )
    latest = resolve_latest_step(
        trained.run,
        writable=False,
        verification=VerificationLevel.FULL,
    )
    exported = export_merged(trained.run, tmp_path / "export-latest")
    session = InferenceSession.from_checkpoint(exported.path, full_verify=True)
    assert session.model_identity.artifact_kind == "export"
    checkpoint_names = sorted(
        path.name for path in latest.step_directory.parent.iterdir()
    )
    assert checkpoint_names == [latest.step_directory.name]
    with pytest.raises(SMLArtifactError):
        export_merged(latest.step_directory, tmp_path / "export-from-step")
    with pytest.raises(SMLArtifactError):
        InferenceSession.from_checkpoint(latest.step_directory, full_verify=True)


def test_export_consumes_run_descriptor_after_outer_name_replacement(
    tiny_base_run,
    tiny_swag_bundle,
    tmp_path,
    monkeypatch,
):
    """Catches export reopening the run pathname after proving its latest step."""
    trained = finetune(
        tiny_swag_training_config(
            tiny_base_run,
            tiny_swag_bundle,
            tmp_path / "descriptor-export-run",
            maximum_steps=1,
        )
    )
    run = trained.run
    displaced = tmp_path / "displaced-run"
    real_open_latest = swag_module.open_latest_checkpoint_reader
    closed_descriptors: list[int] = []

    @contextmanager
    def replace_outer_name(*args, **kwargs):
        with real_open_latest(*args, **kwargs) as reader:
            run.rename(displaced)
            run.mkdir()
            try:
                yield reader
            finally:
                run.rmdir()
                displaced.rename(run)
        closed_descriptors.append(reader._owned_step.descriptor)

    monkeypatch.setattr(
        swag_module, "open_latest_checkpoint_reader", replace_outer_name
    )

    exported = export_merged(run, tmp_path / "descriptor-export")

    assert exported.path.is_dir()
    assert closed_descriptors == [-1]


def test_merged_inference_uses_open_export_after_outer_name_replacement(
    tiny_base_run, tiny_swag_bundle, tmp_path, monkeypatch
):
    """Merged inference retains the export root while loading its child and payload."""
    trained = finetune(
        tiny_swag_training_config(
            tiny_base_run,
            tiny_swag_bundle,
            tmp_path / "inference-export-source",
            maximum_steps=1,
        )
    )
    exported = export_merged(trained.run, tmp_path / "inference-export")
    displaced = tmp_path / "displaced-export"
    descriptors: list[int] = []
    real_dispatch = inference_module.open_dispatched_artifact

    def replace_outer_name(path, root, verification):
        artifact = real_dispatch(path, root, verification)
        descriptors.append(artifact.root.fileno())
        exported.path.rename(displaced)
        exported.path.mkdir()
        return artifact

    monkeypatch.setattr(
        inference_module,
        "open_dispatched_artifact",
        replace_outer_name,
    )
    try:
        resolved = resolve_model_artifact(exported.path, full_verify=True)
    finally:
        exported.path.rmdir()
        displaced.rename(exported.path)

    assert resolved.artifact_kind == "export"
    assert len(descriptors) == 1
    with pytest.raises(OSError):
        os.fstat(descriptors[0])


@pytest.mark.parametrize("replacement", ["tokenizer", "model.safetensors"])
def test_merged_inference_fails_closed_on_child_or_payload_replacement(
    tiny_base_run, tiny_swag_bundle, tmp_path, monkeypatch, replacement: str
):
    """An export never consumes a new tokenizer child or model payload by pathname."""
    trained = finetune(
        tiny_swag_training_config(
            tiny_base_run,
            tiny_swag_bundle,
            tmp_path / f"inference-{replacement}-source",
            maximum_steps=1,
        )
    )
    exported = export_merged(trained.run, tmp_path / f"inference-{replacement}-export")
    descriptors: list[int] = []
    real_dispatch = inference_module.open_dispatched_artifact
    target = exported.path / replacement
    displaced = tmp_path / f"displaced-{replacement}"

    def replace_verified_name(path, root, verification):
        artifact = real_dispatch(path, root, verification)
        descriptors.append(artifact.root.fileno())
        if target.is_dir():
            target.rename(displaced)
            target.mkdir()
        else:
            target.write_bytes(b"replacement")
        return artifact

    monkeypatch.setattr(
        inference_module,
        "open_dispatched_artifact",
        replace_verified_name,
    )
    try:
        with pytest.raises(SMLArtifactError):
            resolve_model_artifact(exported.path, full_verify=True)
    finally:
        if target.is_dir():
            target.rmdir()
            displaced.rename(target)
    assert len(descriptors) == 1
    with pytest.raises(OSError):
        os.fstat(descriptors[0])


@pytest.mark.parametrize("target", ["tokenizer", "model.safetensors"])
def test_merged_inference_consumes_post_open_export_owner_after_name_replacement(
    tiny_base_run, tiny_swag_bundle, tmp_path, monkeypatch, target: str
):
    """Child/payload replacement after FULL proof cannot redirect export inference."""
    trained = finetune(
        tiny_swag_training_config(
            tiny_base_run,
            tiny_swag_bundle,
            tmp_path / f"post-open-{target}-source",
            maximum_steps=1,
        )
    )
    exported = export_merged(trained.run, tmp_path / f"post-open-{target}-export")
    descriptors: list[int] = []
    moved = tmp_path / f"retained-{target}"
    real_duplicate = ArtifactRoot.duplicate
    real_child = OpenedArtifact.open_child
    real_payload = OpenedArtifact.open_payload

    def capture_outer(root):
        duplicated = real_duplicate(root)
        descriptors.append(duplicated.fileno())
        return duplicated

    def replace_child(artifact, logical_path, manifest_types):
        child = real_child(artifact, logical_path, manifest_types)
        if target == "tokenizer" and artifact.path == exported.path:
            descriptors.append(child.root.fileno())
            source = exported.path / "tokenizer"
            source.rename(moved)
            source.mkdir()
        return child

    def replace_payload(artifact, reference):
        payload = real_payload(artifact, reference)
        if target == "tokenizer" and artifact.path == exported.path / "tokenizer":
            descriptors.append(payload.stream.fileno())
        if target == "model.safetensors" and artifact.path == exported.path:
            descriptors.append(payload.stream.fileno())
            source = exported.path / "model.safetensors"
            source.rename(moved)
            source.write_bytes(b"replacement")
        return payload

    monkeypatch.setattr(ArtifactRoot, "duplicate", capture_outer)
    monkeypatch.setattr(OpenedArtifact, "open_child", replace_child)
    monkeypatch.setattr(OpenedArtifact, "open_payload", replace_payload)
    try:
        if target == "model.safetensors":
            with pytest.raises(SMLArtifactError, match="payload changed during use"):
                resolve_model_artifact(exported.path, full_verify=True)
            resolved = None
        else:
            resolved = resolve_model_artifact(exported.path, full_verify=True)
    finally:
        source = exported.path / target
        if target == "tokenizer":
            source.rmdir()
        else:
            source.unlink()
        moved.rename(source)
    if resolved is not None:
        assert resolved.artifact_kind == "export"
    for descriptor in descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)


@pytest.mark.parametrize("child", ["base", "tokenizer"])
def test_export_fails_closed_when_open_run_child_is_replaced(
    tiny_base_run,
    tiny_swag_bundle,
    tmp_path,
    monkeypatch,
    child: str,
):
    """Export never falls back to a replacement base or tokenizer child pathname."""
    trained = finetune(
        tiny_swag_training_config(
            tiny_base_run,
            tiny_swag_bundle,
            tmp_path / f"replaced-{child}-run",
            maximum_steps=1,
        )
    )
    run = trained.run
    displaced = tmp_path / f"displaced-{child}"
    descriptors: list[int] = []
    real_open_child = CheckpointReader.open_run_child

    def replace_after_open(reader, logical_path, manifest_types):
        artifact = real_open_child(reader, logical_path, manifest_types)
        if logical_path == child:
            descriptors.extend(
                [
                    reader._run_descriptor,
                    reader._checkpoints_descriptor,
                    reader._owned_step.descriptor,
                    artifact.root.fileno(),
                ]
            )
            source = run / child
            source.rename(displaced)
            source.mkdir()
        return artifact

    monkeypatch.setattr(CheckpointReader, "open_run_child", replace_after_open)
    try:
        exported = export_merged(run, tmp_path / f"replaced-{child}-export")
    finally:
        source = run / child
        source.rmdir()
        displaced.rename(source)
    assert exported.path.is_dir()
    for descriptor in descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_fresh_finetune_uses_materialized_base_payload_after_name_replacement(
    tiny_base_run, tiny_swag_bundle, tmp_path, monkeypatch
):
    """Copied-base publication cannot reopen a replacement checkpoint payload."""
    descriptors: list[int] = []
    real_open_latest = swag_module.open_latest_checkpoint_reader
    moved: Path | None = None
    replacement: Path | None = None

    @contextmanager
    def replace_after_materialization(*args, **kwargs):
        nonlocal moved, replacement
        with real_open_latest(*args, **kwargs) as reader:
            descriptors.extend(
                [
                    reader._run_descriptor,
                    reader._checkpoints_descriptor,
                    reader._owned_step.descriptor,
                ]
            )
            payload = reader.resolved.step_directory / "model.safetensors"
            moved = payload.with_name("retained-model.safetensors")
            payload.rename(moved)
            payload.write_bytes(b"replacement")
            replacement = payload
            yield reader

    monkeypatch.setattr(
        swag_module, "open_latest_checkpoint_reader", replace_after_materialization
    )
    try:
        trained = finetune(
            tiny_swag_training_config(
                tiny_base_run,
                tiny_swag_bundle,
                tmp_path / "fresh-replaced-base-run",
                maximum_steps=1,
            )
        )
    finally:
        if replacement is not None and moved is not None:
            replacement.unlink()
            moved.rename(replacement)
    assert trained.run.is_dir()
    for descriptor in descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_export_closes_reader_when_merge_fails(
    tiny_base_run, tiny_swag_bundle, tmp_path, monkeypatch
):
    """A semantic merge failure remains primary and releases the checkpoint owner."""
    trained = finetune(
        tiny_swag_training_config(
            tiny_base_run,
            tiny_swag_bundle,
            tmp_path / "merge-failure-run",
            maximum_steps=1,
        )
    )
    closed_descriptors: list[int] = []
    real_open_latest = swag_module.open_latest_checkpoint_reader

    @contextmanager
    def record_reader(*args, **kwargs):
        reader = None
        try:
            with real_open_latest(*args, **kwargs) as reader:
                yield reader
        finally:
            if reader is not None:
                closed_descriptors.append(reader._owned_step.descriptor)

    monkeypatch.setattr(swag_module, "open_latest_checkpoint_reader", record_reader)
    monkeypatch.setattr(
        swag_module,
        "merged_model_weights",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("merge failed")),
    )

    with pytest.raises(RuntimeError, match="merge failed"):
        export_merged(trained.run, tmp_path / "merge-failure-export")
    assert closed_descriptors == [-1]


def test_export_uses_loaded_adapter_payload_after_name_replacement(
    tiny_base_run, tiny_swag_bundle, tmp_path, monkeypatch
):
    """The adapter tensors come from the verified checkpoint payload, not its path."""
    trained = finetune(
        tiny_swag_training_config(
            tiny_base_run,
            tiny_swag_bundle,
            tmp_path / "adapter-payload-run",
            maximum_steps=1,
        )
    )
    closed_descriptors: list[int] = []
    real_open_latest = swag_module.open_latest_checkpoint_reader

    @contextmanager
    def replace_payload(*args, **kwargs):
        with real_open_latest(*args, **kwargs) as reader:
            (reader.resolved.step_directory / "adapters.safetensors").write_bytes(
                b"replacement"
            )
            yield reader
        closed_descriptors.append(reader._owned_step.descriptor)

    monkeypatch.setattr(swag_module, "open_latest_checkpoint_reader", replace_payload)

    exported = export_merged(trained.run, tmp_path / "adapter-payload-export")

    assert exported.path.is_dir()
    assert closed_descriptors == [-1]


def test_resume_closes_run_and_children_when_checkpoint_restore_fails(
    tiny_base_run, tiny_swag_bundle, tmp_path, monkeypatch
):
    """A LoRA resume restore error releases both its preliminary and full readers."""
    completed = finetune(
        tiny_swag_training_config(
            tiny_base_run,
            tiny_swag_bundle,
            tmp_path / "resume-checkpoint-failure-run",
            maximum_steps=1,
        )
    )
    closed_descriptors: list[int] = []
    real_open_latest = swag_module.open_latest_checkpoint_reader

    @contextmanager
    def record_reader(*args, **kwargs):
        reader = None
        try:
            with real_open_latest(*args, **kwargs) as reader:
                yield reader
        finally:
            if reader is not None:
                closed_descriptors.append(reader._owned_step.descriptor)

    monkeypatch.setattr(swag_module, "open_latest_checkpoint_reader", record_reader)
    monkeypatch.setattr(
        swag_module,
        "_restore_adapter_checkpoint",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("restore failed")),
    )

    with pytest.raises(RuntimeError, match="restore failed"):
        resume_finetune(
            completed.run,
            data=tiny_swag_bundle.path,
            overrides=ResumeOverrides(maximum_steps=2),
        )
    assert closed_descriptors == [-1, -1]


def test_resume_closes_reader_when_copied_base_load_fails(
    tiny_base_run, tiny_swag_bundle, tmp_path, monkeypatch
):
    """A copied-base load error is semantic-primary and releases the full reader."""
    completed = finetune(
        tiny_swag_training_config(
            tiny_base_run,
            tiny_swag_bundle,
            tmp_path / "resume-base-failure-run",
            maximum_steps=1,
        )
    )
    closed_descriptors: list[int] = []
    real_open_latest = swag_module.open_latest_checkpoint_reader

    @contextmanager
    def record_reader(*args, **kwargs):
        reader = None
        try:
            with real_open_latest(*args, **kwargs) as reader:
                yield reader
        finally:
            if reader is not None:
                closed_descriptors.append(reader._owned_step.descriptor)

    monkeypatch.setattr(swag_module, "open_latest_checkpoint_reader", record_reader)
    monkeypatch.setattr(
        swag_module,
        "_load_base_snapshot_arrays",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("base failed")),
    )

    with pytest.raises(RuntimeError, match="base failed"):
        resume_finetune(
            completed.run,
            data=tiny_swag_bundle.path,
            overrides=ResumeOverrides(maximum_steps=2),
        )
    assert closed_descriptors == [-1, -1]


def test_resolve_rejects_export_that_changes_copied_base_rope(
    tiny_base_run, tiny_swag_bundle, tmp_path
):
    trained = finetune(
        tiny_swag_training_config(
            tiny_base_run,
            tiny_swag_bundle,
            tmp_path / "rope-run",
            maximum_steps=1,
        )
    )
    exported = export_merged(trained.run, tmp_path / "rope-export")
    verified = read_manifest(
        exported.path,
        ExportManifest,
        VerificationLevel.MANIFEST_TRUSTED,
    )
    model = dict(verified.manifest.model)
    model["rope_scaling_factor"] = 2.0
    tampered = replace(verified.manifest, model=model)
    tampered = replace(tampered, identity=tampered.recompute_identity())
    (exported.path / "manifest.json").write_bytes(canonical_json_bytes(tampered))
    with pytest.raises(SMLArtifactError):
        InferenceSession.from_checkpoint(exported.path, full_verify=True)


def test_resolve_rejects_resigned_export_tokenizer_payload_mismatch(
    tiny_base_run, tiny_swag_bundle, tmp_path
):
    """A self-consistent outer export must still bind its live tokenizer child."""
    trained = finetune(
        tiny_swag_training_config(
            tiny_base_run,
            tiny_swag_bundle,
            tmp_path / "tokenizer-binding-run",
            maximum_steps=1,
        )
    )
    exported = export_merged(trained.run, tmp_path / "tokenizer-binding-export")
    verified = read_manifest(
        exported.path,
        ExportManifest,
        VerificationLevel.MANIFEST_TRUSTED,
    )
    resigned = replace(
        verified.manifest,
        tokenizer_model=verified.manifest.tokenizer_vocab,
    )
    resigned = replace(resigned, identity=resigned.recompute_identity())
    (exported.path / "manifest.json").write_bytes(canonical_json_bytes(resigned))

    with pytest.raises(SMLArtifactError, match="tokenizer payload references"):
        resolve_model_artifact(exported.path, full_verify=True)
