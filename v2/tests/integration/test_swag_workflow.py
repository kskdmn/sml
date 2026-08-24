from __future__ import annotations

import json
import shutil
from collections.abc import Iterator, Mapping
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import pytest
import zstandard as zstd
from sml.artifacts.checkpoint import resolve_latest_step
from sml.artifacts.manifest import (
    BaseSnapshotManifest,
    ExportManifest,
    LatestIndex,
    LoRACheckpointManifest,
    LoRARunManifest,
    PayloadRef,
    VerificationLevel,
    canonical_json_bytes,
    file_identity,
    read_manifest,
)
from sml.data.corpus import CorpusConfig
from sml.data.pretraining import (
    PretrainingPreparationConfig,
    prepare_pretraining_bundle,
)
from sml.data.swag import SwagPreparationConfig, SwagSourceConfig, prepare_swag_bundle
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
from sml.training.lora import LoRAConfig, LoRAInitializerConfig
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
    trainer_accumulators: mx.Dtype,
) -> None:
    adapter_arrays = _load_group(step.step_directory, "adapters.safetensors")
    optimizer_arrays = _load_group(step.step_directory, "optimizer.safetensors")
    trainer_arrays = _load_group(step.step_directory, "trainer.safetensors")
    assert adapter_arrays
    assert all(array.dtype == adapters for array in adapter_arrays.values())
    moment_arrays = [array for key, array in optimizer_arrays.items() if key != "step"]
    assert moment_arrays
    assert all(array.dtype == optimizer_moments for array in moment_arrays)
    accumulator_arrays = [
        array
        for key, array in trainer_arrays.items()
        if key not in {"accumulation_count", "next_key", "loss_numerator"}
    ]
    assert accumulator_arrays
    assert all(array.dtype == trainer_accumulators for array in accumulator_arrays)


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
def tiny_swag_bundle(tiny_base_run: Path, tmp_path: Path):
    resolved = resolve_model_artifact(tiny_base_run, full_verify=True)
    return prepare_swag_bundle(
        SwagPreparationConfig(
            provider=FakeSwagProvider(_swag_rows(5)),
            source=SwagSourceConfig(revision="deadbeef" * 5),
            maximum_length=32,
            bucket_boundaries=(16, 32),
        ),
        resolved,
        tmp_path / "swag-bundle",
    )


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
        trainer_accumulators=mx.float32,
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


def test_uninterrupted_and_interrupted_adapter_state_match(
    tiny_base_run, tiny_swag_bundle, tmp_path
):
    uninterrupted = finetune(
        tiny_swag_training_config(
            tiny_base_run,
            tiny_swag_bundle,
            tmp_path / "uninterrupted",
            maximum_steps=2,
        )
    )
    first = finetune(
        tiny_swag_training_config(
            tiny_base_run,
            tiny_swag_bundle,
            tmp_path / "interrupted",
            maximum_steps=1,
        )
    )
    resumed = resume_finetune(
        first.run,
        data=tiny_swag_bundle.path,
        overrides=ResumeOverrides(maximum_steps=2),
    )
    left = _latest_checkpoint_state(uninterrupted.run)
    right = _latest_checkpoint_state(resumed.run)
    assert left["step"] == right["step"] == 2
    assert left["scalar"]["cursor"] == right["scalar"]["cursor"]
    assert left["scalar"]["step"] == right["scalar"]["step"]
    _assert_array_maps_equal(left["adapters"], right["adapters"])
    _assert_array_maps_equal(left["optimizer"], right["optimizer"])
    _assert_array_maps_equal(left["trainer"], right["trainer"])


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
            data=other_bundle.path,
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


def test_export_uses_recovered_latest_and_rejects_direct_step_paths(
    tiny_base_run, tiny_swag_bundle, tmp_path
):
    trained = finetune(
        tiny_swag_training_config(
            tiny_base_run,
            tiny_swag_bundle,
            tmp_path / "export-run",
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
