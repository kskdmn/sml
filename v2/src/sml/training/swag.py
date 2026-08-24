"""Compiled mean-normalized SWAG ranking kernels and portable LoRA runs."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten, tree_map, tree_unflatten

from sml.artifacts.checkpoint import (
    Published,
    ResolvedStep,
    open_checkpoint_reader,
    prune_to_latest,
    publish_checkpoint,
    publish_immutable_bundle,
    publish_run,
    recover_latest_index,
    resolve_latest_step,
    run_access_lock,
    run_writer_lock,
)
from sml.artifacts.manifest import (
    ArrayPayloadRef,
    ArraySpec,
    ArtifactRoot,
    BaseSnapshotManifest,
    ExportManifest,
    LoRACheckpointManifest,
    LoRARunManifest,
    PayloadRef,
    PretrainingCheckpointManifest,
    PretrainingRunManifest,
    TokenizerManifest,
    VerificationLevel,
    canonical_json_bytes,
    file_identity,
    read_manifest,
    structured_identity,
)
from sml.data.swag import (
    SwagBatch,
    SwagBatchStream,
    SwagCursor,
    SwagDataBundle,
    load_swag_bundle,
)
from sml.errors import SMLArtifactError, SMLConfigurationError
from sml.model.config import ModelConfig
from sml.model.language_model import SMLLanguageModel
from sml.training.common import (
    AdamState,
    CheckpointPolicy,
    LoaderConfig,
    OptimizerConfig,
    ResumeOverrides,
    WeightDecayPolicy,
    _adamw_fp32_update_tree,
    _require_dtype,
    _require_int32_counter,
    accumulate_fp32,
    build_weight_decay_tree,
    initialize_adam_state,
    normalize_and_clip,
)
from sml.training.lora import (
    LoRAConfig,
    LoRAPrecisionConfig,
    apply_lora,
    load_lora_state_dict,
    lora_config_from_mapping,
    merged_model_weights,
    split_adapter_parameters,
)


def _merge_adapter_parameters(adapters: object, frozen_base: object) -> object:
    if isinstance(frozen_base, dict):
        adapter_map = adapters if isinstance(adapters, dict) else {}
        merged: dict[object, object] = {}
        for key in set(frozen_base) | set(adapter_map):
            if key in frozen_base and key in adapter_map:
                merged[key] = _merge_adapter_parameters(
                    adapter_map[key], frozen_base[key]
                )
            elif key in adapter_map:
                merged[key] = adapter_map[key]
            else:
                merged[key] = frozen_base[key]
        return merged
    if isinstance(frozen_base, list):
        adapter_items = (
            adapters if isinstance(adapters, list) else [{} for _ in frozen_base]
        )
        return [
            _merge_adapter_parameters(adapter, frozen)
            for adapter, frozen in zip(adapter_items, frozen_base, strict=True)
        ]
    if isinstance(frozen_base, tuple):
        adapter_items = (
            adapters if isinstance(adapters, tuple) else tuple({} for _ in frozen_base)
        )
        return tuple(
            _merge_adapter_parameters(adapter, frozen)
            for adapter, frozen in zip(adapter_items, frozen_base, strict=True)
        )
    return frozen_base


def score_candidates(
    logits: mx.array,
    input_ids: mx.array,
    score_mask: mx.array,
) -> mx.array:
    """Return FP32 mean continuation-token log-likelihood, including EOS."""

    shifted_logits = logits[..., :-1, :].astype(mx.float32)
    targets = input_ids[..., 1:]
    mask = score_mask[..., 1:].astype(mx.float32)
    target_logit = mx.take_along_axis(
        shifted_logits, mx.expand_dims(targets, axis=-1), axis=-1
    ).squeeze(-1)
    token_ll = target_logit - mx.logsumexp(shifted_logits, axis=-1)
    counted = mx.maximum(mask.sum(axis=-1), mx.array(1.0, dtype=mx.float32))
    return ((token_ll * mask).sum(axis=-1) / counted).astype(mx.float32)


def default_swag_optimizer_config() -> OptimizerConfig:
    return OptimizerConfig(learning_rate=1e-4, schedule_steps=8_192)


_PLACEHOLDER_IDENTITY = "sha256:" + "0" * 64
_CHECKPOINT_ARRAY_PATHS = (
    "adapters.safetensors",
    "optimizer.safetensors",
    "trainer.safetensors",
)
_SAVED_CHECKPOINT_KEYS = frozenset(
    {
        "interval",
        "maximum_steps",
        "maximum_epochs",
        "log_interval",
        "seed",
        "compile",
    }
)
_TOKENIZER_FILES = ("manifest.json", "tokenizer.model", "tokenizer.vocab")


@dataclass(frozen=True, slots=True)
class SwagTrainingConfig:
    base_checkpoint: Path
    data: Path
    output_run: Path
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    optimizer: OptimizerConfig = field(default_factory=default_swag_optimizer_config)
    loader: LoaderConfig = field(default_factory=LoaderConfig)
    checkpoint: CheckpointPolicy = field(
        default_factory=lambda: CheckpointPolicy(interval=500),
    )
    precision: LoRAPrecisionConfig = field(default_factory=LoRAPrecisionConfig)
    maximum_steps: int | None = 8_192
    maximum_epochs: int | None = 5
    log_interval: int = 10
    seed: int = 42
    compile: bool = True

    def __post_init__(self) -> None:
        for field_name in ("base_checkpoint", "data", "output_run"):
            if not isinstance(getattr(self, field_name), Path):
                raise SMLConfigurationError(f"{field_name} must be a Path")
        for field_name, expected_type in (
            ("lora", LoRAConfig),
            ("optimizer", OptimizerConfig),
            ("loader", LoaderConfig),
            ("checkpoint", CheckpointPolicy),
            ("precision", LoRAPrecisionConfig),
        ):
            if not isinstance(getattr(self, field_name), expected_type):
                raise SMLConfigurationError(
                    f"{field_name} has an invalid configuration"
                )
        if self.maximum_steps is None and self.maximum_epochs is None:
            raise SMLConfigurationError(
                "at least one training termination limit is required"
            )
        if self.maximum_steps is not None:
            _require_int32_counter(self.maximum_steps, "maximum_steps")
        if self.maximum_epochs is not None:
            _require_int32_counter(self.maximum_epochs, "maximum_epochs")
        _require_int32_counter(self.log_interval, "log_interval")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed <= 2**32 - 1
        ):
            raise SMLConfigurationError("seed must be an unsigned 32-bit integer")
        if not isinstance(self.compile, bool):
            raise SMLConfigurationError("compile must be a bool")


@dataclass(frozen=True, slots=True)
class SwagTrainingResult:
    run: Path
    step: int
    epoch: int
    examples: int


@dataclass(frozen=True, slots=True)
class ExportResult:
    path: Path


@dataclass(frozen=True, slots=True)
class ScalarSwagState:
    step: int
    examples: int
    microsteps: int
    cursor: SwagCursor

    def __post_init__(self) -> None:
        for name in ("step", "examples", "microsteps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise SMLArtifactError(f"checkpoint scalar {name} must be nonnegative")
        if not isinstance(self.cursor, SwagCursor):
            raise SMLArtifactError("checkpoint cursor must be a SwagCursor")


@dataclass(frozen=True, slots=True)
class _RestoredSwagState:
    adapters: dict
    frozen_base: dict
    optimizer: AdamState
    trainer: SwagTrainerState
    scalar: ScalarSwagState


@dataclass(frozen=True, slots=True)
class _SelectedPretrainingBase:
    step: ResolvedStep
    working_bytes: bytes
    tokenizer_files: dict[str, bytes]
    model: dict[str, object]
    precision: dict[str, object]
    tokenizer_identity: str
    model_config: ModelConfig
    identity: str


@dataclass(frozen=True, slots=True)
class SwagKernelConfig:
    accumulation_steps: int
    gradient_clip_norm: float
    compile: bool


@dataclass(frozen=True, slots=True)
class SwagTrainerState:
    accumulators: dict
    valid_count: mx.array
    next_key: mx.array
    loss_numerator: mx.array
    correct_count: mx.array

    def to_tree(self) -> tuple[dict, mx.array, mx.array, mx.array, mx.array]:
        return (
            self.accumulators,
            self.valid_count,
            self.next_key,
            self.loss_numerator,
            self.correct_count,
        )

    @classmethod
    def from_compiled_tree(cls, tree: object) -> SwagTrainerState:
        """Wrap a compiled trainer tree without synchronizing counters."""

        (
            accumulators,
            valid_count,
            next_key,
            loss_numerator,
            correct_count,
        ) = tree
        instance = object.__new__(cls)
        object.__setattr__(instance, "accumulators", accumulators)
        object.__setattr__(instance, "valid_count", valid_count)
        object.__setattr__(instance, "next_key", next_key)
        object.__setattr__(instance, "loss_numerator", loss_numerator)
        object.__setattr__(instance, "correct_count", correct_count)
        return instance


def initial_swag_trainer_state(adapters: dict, *, key: mx.array) -> SwagTrainerState:
    zeros = tree_map(
        lambda parameter: mx.zeros_like(parameter).astype(mx.float32), adapters
    )
    return SwagTrainerState(
        accumulators=zeros,
        valid_count=mx.array(0, dtype=mx.int32),
        next_key=key,
        loss_numerator=mx.array(0.0, dtype=mx.float32),
        correct_count=mx.array(0, dtype=mx.int32),
    )


@dataclass(frozen=True, slots=True)
class SwagKernels:
    compiled_ranking_microstep_core: object
    compiled_optimizer_step_core: object
    kernel_config: SwagKernelConfig
    optimizer_config: OptimizerConfig
    weight_decay_tree: dict

    def ranking_microstep(
        self,
        adapters: dict,
        frozen_base: dict,
        trainer: SwagTrainerState,
        batch: SwagBatch,
    ) -> SwagTrainerState:
        next_tree = self.compiled_ranking_microstep_core(
            adapters,
            frozen_base,
            trainer.to_tree(),
            batch.input_ids,
            batch.score_mask,
            batch.labels,
            batch.example_mask,
            batch.valid_token_mask,
        )
        return SwagTrainerState.from_compiled_tree(next_tree)

    def optimizer_step(
        self,
        adapters: dict,
        optimizer: AdamState,
        trainer: SwagTrainerState,
    ) -> tuple[dict, AdamState, SwagTrainerState]:
        next_adapters, next_adam_tree, next_trainer = self.compiled_optimizer_step_core(
            adapters,
            optimizer.to_tree(),
            trainer.to_tree(),
        )
        _require_dtype(next_adapters, "parameters", mx.float32)
        _require_dtype(next_adam_tree[1], "first_moments", mx.float32)
        _require_dtype(next_adam_tree[2], "second_moments", mx.float32)
        return (
            next_adapters,
            AdamState.from_compiled_tree(next_adam_tree),
            SwagTrainerState.from_compiled_tree(next_trainer),
        )


def build_swag_kernels(
    model: SMLLanguageModel,
    config: SwagTrainingConfig,
    weight_decay_tree: dict,
) -> SwagKernels:
    """Build host wrappers around eager or compiled ranking and optimizer cores."""

    kernel_config = SwagKernelConfig(
        accumulation_steps=config.loader.gradient_accumulation_steps,
        gradient_clip_norm=config.optimizer.gradient_clip_norm,
        compile=config.compile,
    )

    def adapter_loss(adapters, frozen_base, batch_arrays, key):
        input_ids, score_mask, labels, example_mask, valid_token_mask = batch_arrays
        parameters = _merge_adapter_parameters(adapters, frozen_base)
        batch_size, candidate_count, length = input_ids.shape
        flat_ids = input_ids.reshape((batch_size * candidate_count, length))
        flat_valid = valid_token_mask.reshape((batch_size * candidate_count, length))
        flat_score = score_mask.reshape((batch_size * candidate_count, length))
        logits, _cache_state, next_key = model.forward_arrays(
            parameters,
            flat_ids,
            attention_mask=flat_valid,
            positions=None,
            cache_state=None,
            training=True,
            key=key,
        )
        if next_key is None:
            raise RuntimeError("training forward did not return a PRNG key")
        scores = score_candidates(logits, flat_ids, flat_score)
        scores = scores.reshape((batch_size, candidate_count))
        label_scores = mx.take_along_axis(
            scores, mx.expand_dims(labels, axis=-1), axis=-1
        ).squeeze(-1)
        per_slot = mx.logsumexp(scores, axis=-1) - label_scores
        weights = example_mask.astype(mx.float32)
        loss_sum = (per_slot * weights).sum().astype(mx.float32)
        predictions = mx.argmax(scores, axis=-1)
        correct = (
            (predictions == labels).astype(mx.int32) * example_mask.astype(mx.int32)
        ).sum()
        valid = example_mask.astype(mx.int32).sum()
        return loss_sum, (next_key, correct.astype(mx.int32), valid.astype(mx.int32))

    loss_and_grad = mx.value_and_grad(adapter_loss, argnums=0)

    def ranking_microstep_core(
        adapters,
        frozen_base,
        trainer_tree,
        input_ids,
        score_mask,
        labels,
        example_mask,
        valid_token_mask,
    ):
        accumulators, valid_count, key, loss_numerator, correct_count = trainer_tree
        batch_arrays = (
            input_ids,
            score_mask,
            labels,
            example_mask,
            valid_token_mask,
        )
        (loss_sum, (next_key, correct, valid)), gradients = loss_and_grad(
            adapters, frozen_base, batch_arrays, key
        )
        next_accumulators = accumulate_fp32(accumulators, gradients)
        next_valid = (valid_count + valid).astype(mx.int32)
        next_loss = (loss_numerator + loss_sum.astype(mx.float32)).astype(mx.float32)
        next_correct = (correct_count + correct).astype(mx.int32)
        return (
            next_accumulators,
            next_valid,
            next_key,
            next_loss,
            next_correct,
        )

    def swag_optimizer_step_core(adapters, adam_tree, trainer_tree):
        accumulators, valid_count, next_key, loss_numerator, correct_count = (
            trainer_tree
        )
        gradients = normalize_and_clip(
            accumulators,
            valid_count,
            gradient_clip_norm=kernel_config.gradient_clip_norm,
        )
        next_adapters, next_adam_tree = _adamw_fp32_update_tree(
            adapters,
            gradients,
            adam_tree,
            config.optimizer,
            weight_decay_tree,
        )
        reset_accumulators = tree_map(
            lambda accumulator: accumulator - accumulator, accumulators
        )
        next_trainer = (
            reset_accumulators,
            (valid_count - valid_count).astype(mx.int32),
            next_key,
            (loss_numerator - loss_numerator).astype(mx.float32),
            (correct_count - correct_count).astype(mx.int32),
        )
        return next_adapters, next_adam_tree, next_trainer

    return SwagKernels(
        compiled_ranking_microstep_core=(
            mx.compile(ranking_microstep_core)
            if config.compile
            else ranking_microstep_core
        ),
        compiled_optimizer_step_core=(
            mx.compile(swag_optimizer_step_core)
            if config.compile
            else swag_optimizer_step_core
        ),
        kernel_config=kernel_config,
        optimizer_config=config.optimizer,
        weight_decay_tree=weight_decay_tree,
    )


def _payload_ref(path: Path, logical_path: str) -> PayloadRef:
    with path.open("rb") as payload:
        identity = file_identity(payload)
    return PayloadRef(logical_path, identity, path.stat().st_size)


def _dtype_name(array: mx.array) -> str:
    names = {
        mx.bfloat16: "bfloat16",
        mx.float32: "float32",
        mx.int32: "int32",
        mx.uint32: "uint32",
    }
    try:
        return names[array.dtype]
    except KeyError as error:
        raise SMLArtifactError(
            f"unsupported checkpoint array dtype: {array.dtype}"
        ) from error


def _array_payload_ref(
    path: Path, logical_path: str, arrays: Mapping[str, mx.array]
) -> ArrayPayloadRef:
    return ArrayPayloadRef(
        payload=_payload_ref(path, logical_path),
        arrays=tuple(
            ArraySpec(name, tuple(array.shape), _dtype_name(array))
            for name, array in sorted(arrays.items())
        ),
    )


def _mapping_dict(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise SMLArtifactError(f"saved {name} configuration must be a mapping")
    return dict(value)


def _optimizer_from_mapping(value: Mapping[str, object]) -> OptimizerConfig:
    values = dict(value)
    weight_decay = WeightDecayPolicy(
        **_mapping_dict(values.pop("weight_decay"), "weight_decay")
    )
    return OptimizerConfig(weight_decay=weight_decay, **values)


def _run_step_model_identity(resolved: ResolvedStep, tokenizer_identity: str) -> str:
    return structured_identity(
        "sml-resolved-model-identity-v1",
        {
            "artifact_kind": resolved.run.kind,
            "run_identity": resolved.run.identity,
            "step": resolved.step,
            "checkpoint_identity": resolved.checkpoint.identity,
            "run_step_identity": resolved.run_step_identity,
            "tokenizer_identity": tokenizer_identity,
        },
    )


def _load_safetensors(root: Path, logical_path: str) -> dict[str, mx.array]:
    with (
        ArtifactRoot.open(root, writable=False) as artifact,
        artifact.open_payload(logical_path) as payload,
    ):
        arrays = mx.load(payload, format="safetensors")
    if not isinstance(arrays, dict):
        raise SMLArtifactError(f"array payload must be a mapping: {logical_path}")
    mx.eval(*arrays.values())
    return dict(arrays)


def _read_tokenizer_files(source: Path) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    with ArtifactRoot.open(source, writable=False) as root:
        for name in _TOKENIZER_FILES:
            with root.open_payload(name) as payload:
                files[name] = payload.read()
    return files


def _write_tokenizer_directory(destination: Path, files: Mapping[str, bytes]) -> None:
    destination.mkdir()
    for name in _TOKENIZER_FILES:
        payload = files.get(name)
        if not isinstance(payload, bytes):
            raise SMLArtifactError(f"tokenizer payload missing: {name}")
        (destination / name).write_bytes(payload)


def _copy_base_snapshot(
    private_run: Path,
    *,
    base_step: ResolvedStep,
    working_bytes: bytes,
    model: Mapping[str, object],
    precision: Mapping[str, object],
    tokenizer_identity: str,
) -> BaseSnapshotManifest:
    if not isinstance(base_step.checkpoint, PretrainingCheckpointManifest):
        raise SMLArtifactError("LoRA base must be a pretraining checkpoint")
    if not isinstance(base_step.run, PretrainingRunManifest):
        raise SMLArtifactError("LoRA base must be a pretraining run")
    destination = private_run / "base"
    destination.mkdir()
    model_path = destination / "model.safetensors"
    model_path.write_bytes(working_bytes)
    copied = _payload_ref(model_path, "model.safetensors")
    source_ref = base_step.checkpoint.model
    if (
        copied.identity != source_ref.payload.identity
        or copied.byte_size != source_ref.payload.byte_size
    ):
        raise SMLArtifactError("copied base working weights changed during copy")
    manifest = BaseSnapshotManifest(
        kind="base-snapshot",
        version=1,
        identity=_PLACEHOLDER_IDENTITY,
        model=dict(model),
        precision=dict(precision),
        tokenizer_identity=tokenizer_identity,
        working_weights=ArrayPayloadRef(copied, source_ref.arrays),
        diagnostic_source_run_identity=base_step.run.identity,
        diagnostic_source_step=base_step.step,
    )
    manifest = replace(manifest, identity=manifest.recompute_identity())
    (destination / BaseSnapshotManifest.MANIFEST_FILENAME).write_bytes(
        canonical_json_bytes(manifest)
    )
    return manifest


def _verified_swag(
    path: Path,
    *,
    expected_identity: str | None,
    tokenizer_identity: str,
    base_identity: str | None,
) -> SwagDataBundle:
    bundle = load_swag_bundle(path, VerificationLevel.FULL)
    if expected_identity is not None and bundle.manifest.identity != expected_identity:
        raise SMLArtifactError("prepared SWAG identity does not match run.json")
    if bundle.manifest.tokenizer_identity != tokenizer_identity:
        raise SMLArtifactError("prepared SWAG tokenizer identity does not match")
    if base_identity is not None and bundle.manifest.base_identity != base_identity:
        raise SMLArtifactError("prepared SWAG base identity does not match")
    return bundle


def _resolved_fresh_config(config: SwagTrainingConfig) -> SwagTrainingConfig:
    if config.optimizer.schedule_steps is None and config.maximum_steps is not None:
        return replace(
            config,
            optimizer=replace(
                config.optimizer,
                schedule_steps=config.maximum_steps,
            ),
        )
    return config


def _run_manifest(
    config: SwagTrainingConfig,
    *,
    model: Mapping[str, object],
    tokenizer_identity: str,
    base_identity: str,
    data_identity: str,
) -> LoRARunManifest:
    checkpoint = {
        "interval": config.checkpoint.interval,
        "maximum_steps": config.maximum_steps,
        "maximum_epochs": config.maximum_epochs,
        "log_interval": config.log_interval,
        "seed": config.seed,
        "compile": config.compile,
    }
    manifest = LoRARunManifest(
        kind="lora-run",
        version=1,
        identity=_PLACEHOLDER_IDENTITY,
        model=dict(model),
        lora=dataclasses.asdict(config.lora),
        precision=dataclasses.asdict(config.precision),
        optimizer=dataclasses.asdict(config.optimizer),
        loader=dataclasses.asdict(config.loader),
        checkpoint=checkpoint,
        tokenizer_identity=tokenizer_identity,
        base_identity=base_identity,
        data_identity=data_identity,
        diagnostic_data_locator=str(config.data),
    )
    return replace(manifest, identity=manifest.recompute_identity())


def _config_from_run(
    run: Path,
    data: Path,
    manifest: LoRARunManifest,
    overrides: ResumeOverrides,
) -> SwagTrainingConfig:
    try:
        model = ModelConfig(**_mapping_dict(manifest.model, "model"))
        if model.rope_scaling_factor != 1.0:
            raise SMLArtifactError("LoRA run rope_scaling_factor must be exactly 1.0")
        lora = lora_config_from_mapping(_mapping_dict(manifest.lora, "lora"))
        precision = LoRAPrecisionConfig(
            **_mapping_dict(manifest.precision, "precision")
        )
        loader = LoaderConfig(**_mapping_dict(manifest.loader, "loader"))
        optimizer = _optimizer_from_mapping(
            _mapping_dict(manifest.optimizer, "optimizer")
        )
        saved = _mapping_dict(manifest.checkpoint, "checkpoint")
        if set(saved) != _SAVED_CHECKPOINT_KEYS:
            raise SMLArtifactError("saved checkpoint configuration has invalid fields")
        maximum_steps = (
            saved["maximum_steps"]
            if overrides.maximum_steps is None
            else overrides.maximum_steps
        )
        maximum_epochs = (
            saved["maximum_epochs"]
            if overrides.maximum_epochs is None
            else overrides.maximum_epochs
        )
        log_interval = (
            saved["log_interval"]
            if overrides.log_interval is None
            else overrides.log_interval
        )
        interval = (
            saved["interval"]
            if overrides.checkpoint_interval is None
            else overrides.checkpoint_interval
        )
        return SwagTrainingConfig(
            base_checkpoint=run,
            data=data,
            output_run=run,
            lora=lora,
            optimizer=optimizer,
            loader=loader,
            checkpoint=CheckpointPolicy(interval=interval),
            precision=precision,
            maximum_steps=maximum_steps,
            maximum_epochs=maximum_epochs,
            log_interval=log_interval,
            seed=saved["seed"],
            compile=saved["compile"],
        )
    except SMLArtifactError:
        raise
    except (KeyError, TypeError, ValueError, SMLConfigurationError) as error:
        raise SMLArtifactError("invalid saved LoRA configuration") from error


def _scalar_document(
    state: ScalarSwagState,
    *,
    owning_run_identity: str,
) -> dict[str, object]:
    return {
        "kind": "lora-state",
        "version": 1,
        "owning_run_identity": owning_run_identity,
        "step": state.step,
        "examples": state.examples,
        "microsteps": state.microsteps,
        "cursor": {
            "epoch": state.cursor.epoch,
            "bucket_order_position": state.cursor.bucket_order_position,
            "row_offset": state.cursor.row_offset,
        },
    }


def _plain_nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SMLArtifactError(f"{name} must be a nonnegative integer")
    return value


def _parse_scalar_document(
    payload: bytes,
    resolved: ResolvedStep,
) -> ScalarSwagState:
    try:
        raw = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=lambda pairs: _json_object_no_duplicates(pairs),
        )
        if not isinstance(raw, dict):
            raise SMLArtifactError("checkpoint scalar state has invalid fields")
        state = ScalarSwagState(
            step=_plain_nonnegative_int(raw["step"], "checkpoint scalar step"),
            examples=_plain_nonnegative_int(
                raw["examples"], "checkpoint scalar examples"
            ),
            microsteps=_plain_nonnegative_int(
                raw["microsteps"], "checkpoint scalar microsteps"
            ),
            cursor=SwagCursor(
                epoch=_plain_nonnegative_int(raw["cursor"]["epoch"], "cursor epoch"),
                bucket_order_position=_plain_nonnegative_int(
                    raw["cursor"]["bucket_order_position"],
                    "cursor bucket position",
                ),
                row_offset=_plain_nonnegative_int(
                    raw["cursor"]["row_offset"], "cursor row offset"
                ),
            ),
        )
        if state.step != resolved.step:
            raise SMLArtifactError("checkpoint scalar step does not match checkpoint")
        if canonical_json_bytes(raw) != payload:
            raise SMLArtifactError("checkpoint scalar state is not canonical JSON")
        return state
    except SMLArtifactError:
        raise
    except (
        json.JSONDecodeError,
        UnicodeError,
        TypeError,
        ValueError,
        KeyError,
    ) as error:
        raise SMLArtifactError("invalid checkpoint scalar state") from error


def _read_scalar_state(resolved: ResolvedStep) -> ScalarSwagState:
    with (
        ArtifactRoot.open(resolved.step_directory, writable=False) as root,
        root.open_payload("state.json") as payload,
    ):
        return _parse_scalar_document(payload.read(), resolved)


def _json_object_no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise SMLArtifactError(f"duplicate checkpoint scalar key: {key}")
        result[key] = value
    return result


def _flatten_checkpoint_groups(
    adapters: dict,
    optimizer: AdamState,
    trainer: SwagTrainerState,
) -> dict[str, dict[str, mx.array]]:
    adapter_arrays = dict(sorted(tree_flatten(adapters)))
    optimizer_arrays = {
        "step": optimizer.step,
        **{
            f"first_moments.{name}": value
            for name, value in tree_flatten(optimizer.first_moments)
        },
        **{
            f"second_moments.{name}": value
            for name, value in tree_flatten(optimizer.second_moments)
        },
    }
    trainer_arrays = {
        "accumulation_count": trainer.valid_count,
        "next_key": trainer.next_key,
        "loss_numerator": trainer.loss_numerator,
        **{
            f"accumulators.{name}": value
            for name, value in tree_flatten(trainer.accumulators)
        },
    }
    return {
        "adapters.safetensors": dict(sorted(adapter_arrays.items())),
        "optimizer.safetensors": dict(sorted(optimizer_arrays.items())),
        "trainer.safetensors": dict(sorted(trainer_arrays.items())),
    }


def _require_empty_trainer_state(trainer: SwagTrainerState) -> None:
    mx.eval(trainer.to_tree())
    if int(trainer.valid_count.item()) != 0:
        raise SMLArtifactError("checkpoint trainer accumulation must be empty")
    if float(trainer.loss_numerator.item()) != 0.0:
        raise SMLArtifactError("checkpoint trainer loss numerator must be empty")
    if int(trainer.correct_count.item()) != 0:
        raise SMLArtifactError("checkpoint trainer correct count must be empty")
    if any(
        bool(mx.any(value != 0)) for _name, value in tree_flatten(trainer.accumulators)
    ):
        raise SMLArtifactError("checkpoint trainer accumulators must be empty")


def _unflatten_prefixed(arrays: Mapping[str, mx.array], prefix: str) -> dict:
    items = [
        (name.removeprefix(prefix), value)
        for name, value in arrays.items()
        if name.startswith(prefix)
    ]
    if not items:
        raise SMLArtifactError(f"checkpoint has no arrays under {prefix}")
    return tree_unflatten(items)


def _checkpoint_builder(
    run_manifest: LoRARunManifest,
    state: _RestoredSwagState,
):
    adapters = state.adapters
    optimizer = AdamState(
        state.optimizer.step,
        state.optimizer.first_moments,
        state.optimizer.second_moments,
    )
    trainer = SwagTrainerState.from_compiled_tree(state.trainer.to_tree())
    _require_empty_trainer_state(trainer)
    if int(optimizer.step.item()) != state.scalar.step:
        raise SMLArtifactError("Adam step must match the checkpoint step")
    groups = _flatten_checkpoint_groups(adapters, optimizer, trainer)
    adapter_names = set(groups["adapters.safetensors"])
    expected_optimizer = {
        "step",
        *(f"first_moments.{name}" for name in adapter_names),
        *(f"second_moments.{name}" for name in adapter_names),
    }
    expected_trainer = {
        "accumulation_count",
        "next_key",
        "loss_numerator",
        *(f"accumulators.{name}" for name in adapter_names),
    }
    if set(groups["optimizer.safetensors"]) != expected_optimizer:
        raise SMLArtifactError("optimizer checkpoint keys must match adapter keys")
    if set(groups["trainer.safetensors"]) != expected_trainer:
        raise SMLArtifactError("trainer checkpoint keys must match adapter keys")

    def build(private_step: Path) -> LoRACheckpointManifest:
        references: list[ArrayPayloadRef] = []
        for logical_path in _CHECKPOINT_ARRAY_PATHS:
            arrays = groups[logical_path]
            path = private_step / logical_path
            mx.save_safetensors(str(path), arrays)
            references.append(_array_payload_ref(path, logical_path, arrays))
        state_path = private_step / "state.json"
        state_path.write_bytes(
            canonical_json_bytes(
                _scalar_document(
                    state.scalar,
                    owning_run_identity=run_manifest.identity,
                )
            )
        )
        references_by_path = {
            reference.payload.logical_path: reference for reference in references
        }
        manifest = LoRACheckpointManifest(
            kind="lora-checkpoint",
            version=1,
            identity=_PLACEHOLDER_IDENTITY,
            owning_run_identity=run_manifest.identity,
            step=state.scalar.step,
            scalar_state=_payload_ref(state_path, "state.json"),
            adapters=references_by_path["adapters.safetensors"],
            optimizer=references_by_path["optimizer.safetensors"],
            trainer=references_by_path["trainer.safetensors"],
        )
        return replace(manifest, identity=manifest.recompute_identity())

    return build


def _restore_adapter_checkpoint(
    resolved: ResolvedStep,
    *,
    hold_lock: bool = True,
) -> tuple[dict, AdamState, SwagTrainerState, ScalarSwagState]:
    with open_checkpoint_reader(
        resolved.step_directory.parent.parent,
        step=resolved.step,
        expected_checkpoint_identity=resolved.checkpoint.identity,
        hold_lock=hold_lock,
    ) as reader:
        verified = reader.resolved
        if not isinstance(verified.checkpoint, LoRACheckpointManifest):
            raise SMLArtifactError("LoRA resume requires a LoRA checkpoint")
        contents = reader.read_contents()
        scalar = _parse_scalar_document(
            canonical_json_bytes(dict(contents.scalar_state)),
            verified,
        )
        adapters = dict(contents.array_groups["adapters.safetensors"])
        optimizer_arrays = dict(contents.array_groups["optimizer.safetensors"])
        trainer_arrays = dict(contents.array_groups["trainer.safetensors"])
        try:
            _require_dtype(adapters, "adapters", mx.float32)
            adapter_tree = tree_unflatten(list(adapters.items()))
            if not isinstance(adapter_tree, dict):
                raise SMLArtifactError("adapter checkpoint must unflatten to a dict")
            expected_optimizer = {
                "step",
                *(f"first_moments.{name}" for name in adapters),
                *(f"second_moments.{name}" for name in adapters),
            }
            if set(optimizer_arrays) != expected_optimizer:
                raise SMLArtifactError(
                    "optimizer checkpoint keys do not match adapters"
                )
            _require_dtype(
                {k: v for k, v in optimizer_arrays.items() if k != "step"},
                "optimizer",
                mx.float32,
            )
            optimizer = AdamState(
                optimizer_arrays["step"],
                _unflatten_prefixed(optimizer_arrays, "first_moments."),
                _unflatten_prefixed(optimizer_arrays, "second_moments."),
            )
            expected_trainer = {
                "accumulation_count",
                "next_key",
                "loss_numerator",
                *(f"accumulators.{name}" for name in adapters),
            }
            if set(trainer_arrays) != expected_trainer:
                raise SMLArtifactError("trainer checkpoint keys do not match adapters")
            trainer = SwagTrainerState(
                accumulators=_unflatten_prefixed(trainer_arrays, "accumulators."),
                valid_count=trainer_arrays["accumulation_count"],
                next_key=trainer_arrays["next_key"],
                loss_numerator=trainer_arrays["loss_numerator"],
                correct_count=mx.array(0, dtype=mx.int32),
            )
            _require_empty_trainer_state(trainer)
            if int(optimizer.step.item()) != scalar.step:
                raise SMLArtifactError(
                    "checkpoint Adam step does not match scalar step"
                )
            return adapter_tree, optimizer, trainer, scalar
        except SMLArtifactError:
            raise
        except (SMLConfigurationError, TypeError, ValueError) as error:
            raise SMLArtifactError("invalid LoRA checkpoint arrays") from error


def _require_retained_publication(
    published: ResolvedStep,
    retained: ResolvedStep,
) -> None:
    if (
        retained.run.identity != published.run.identity
        or retained.checkpoint.identity != published.checkpoint.identity
    ):
        raise SMLArtifactError(
            "retention did not preserve the published run and checkpoint identity"
        )


def _publish_training_state(
    run: Path,
    manifest: LoRARunManifest,
    state: _RestoredSwagState,
) -> ResolvedStep:
    published = publish_checkpoint(run, _checkpoint_builder(manifest, state))
    retained = prune_to_latest(run)
    _require_retained_publication(published, retained)
    if retained.step != published.step:
        raise SMLArtifactError("retention did not preserve the published latest step")
    return retained


def _limit_reached(config: SwagTrainingConfig, state: ScalarSwagState) -> bool:
    return (
        config.maximum_steps is not None and state.step >= config.maximum_steps
    ) or (
        config.maximum_epochs is not None
        and state.cursor.epoch >= config.maximum_epochs
    )


def _training_result(run: Path, state: ScalarSwagState) -> SwagTrainingResult:
    return SwagTrainingResult(run, state.step, state.cursor.epoch, state.examples)


def _wrap_copied_base(
    model_config: ModelConfig,
    base_arrays: Mapping[str, mx.array],
    lora: LoRAConfig,
    key: mx.array,
) -> tuple[SMLLanguageModel, dict, dict]:
    model_key, adapter_key = mx.random.split(key)
    model = SMLLanguageModel(model_config, key=model_key)
    model.update(tree_unflatten(sorted(base_arrays.items())))
    apply_lora(model, lora, key=adapter_key)
    adapters, frozen_base = split_adapter_parameters(model.parameters())
    if not isinstance(adapters, dict) or not isinstance(frozen_base, dict):
        raise SMLArtifactError("adapter parameter split must return dictionaries")
    mx.eval(adapters, frozen_base)
    _require_dtype(adapters, "adapters", mx.float32)
    _require_dtype(frozen_base, "frozen_base", mx.bfloat16)
    return model, adapters, frozen_base


def _load_base_snapshot_arrays(run: Path) -> dict[str, mx.array]:
    verified = read_manifest(run / "base", BaseSnapshotManifest, VerificationLevel.FULL)
    if verified.manifest.model.get("rope_scaling_factor") != 1.0:
        raise SMLArtifactError("copied base rope_scaling_factor must be exactly 1.0")
    arrays = _load_safetensors(
        run / "base",
        verified.manifest.working_weights.payload.logical_path,
    )
    if not arrays:
        raise SMLArtifactError("copied base snapshot has no working weights")
    for array in arrays.values():
        if array.dtype != mx.bfloat16:
            raise SMLArtifactError("copied base working weights must be bfloat16")
    return arrays


def _run_training(
    run: Path,
    manifest: LoRARunManifest,
    config: SwagTrainingConfig,
    model: SMLLanguageModel,
    restored: _RestoredSwagState,
    bundle: SwagDataBundle,
) -> SwagTrainingResult:
    adapters = restored.adapters
    frozen_base = restored.frozen_base
    optimizer = restored.optimizer
    trainer = restored.trainer
    scalar = restored.scalar
    weight_decay_tree = build_weight_decay_tree(
        adapters,
        config.optimizer.weight_decay,
    )
    kernels = build_swag_kernels(model, config, weight_decay_tree)
    last_published_step = scalar.step
    accumulation_steps = config.loader.gradient_accumulation_steps

    while not _limit_reached(config, scalar):
        if (
            config.maximum_epochs is not None
            and scalar.cursor.epoch >= config.maximum_epochs
        ):
            break
        window_microsteps = 0
        window_examples = 0
        pending_cursor: SwagCursor | None = None
        with SwagBatchStream(bundle, config.loader, cursor=scalar.cursor) as stream:
            for envelope in stream:
                if _limit_reached(config, scalar):
                    break
                batch = SwagBatch.from_envelope(envelope)
                trainer = kernels.ranking_microstep(
                    adapters,
                    frozen_base,
                    trainer,
                    batch,
                )
                pending_cursor = batch.cursor_after
                window_microsteps += 1
                window_examples += int(batch.example_mask.astype(mx.int32).sum().item())
                window_full = int(trainer.valid_count.item()) >= accumulation_steps
                if not window_full:
                    continue
                adapters, optimizer, trainer = kernels.optimizer_step(
                    adapters,
                    optimizer,
                    trainer,
                )
                mx.eval(adapters, optimizer.to_tree(), trainer.to_tree())
                stream.commit(pending_cursor)
                scalar = ScalarSwagState(
                    step=scalar.step + 1,
                    examples=scalar.examples + window_examples,
                    microsteps=scalar.microsteps + window_microsteps,
                    cursor=pending_cursor,
                )
                window_microsteps = 0
                window_examples = 0
                pending_cursor = None
                state = _RestoredSwagState(
                    adapters, frozen_base, optimizer, trainer, scalar
                )
                if scalar.step % config.checkpoint.interval == 0:
                    _publish_training_state(run, manifest, state)
                    last_published_step = scalar.step
                if _limit_reached(config, scalar):
                    break
            if window_microsteps and pending_cursor is not None:
                adapters, optimizer, trainer = kernels.optimizer_step(
                    adapters,
                    optimizer,
                    trainer,
                )
                mx.eval(adapters, optimizer.to_tree(), trainer.to_tree())
                stream.commit(pending_cursor)
                scalar = ScalarSwagState(
                    step=scalar.step + 1,
                    examples=scalar.examples + window_examples,
                    microsteps=scalar.microsteps + window_microsteps,
                    cursor=pending_cursor,
                )
                state = _RestoredSwagState(
                    adapters, frozen_base, optimizer, trainer, scalar
                )
                if scalar.step % config.checkpoint.interval == 0:
                    _publish_training_state(run, manifest, state)
                    last_published_step = scalar.step

    final_state = _RestoredSwagState(adapters, frozen_base, optimizer, trainer, scalar)
    if last_published_step != scalar.step:
        _publish_training_state(run, manifest, final_state)
    return _training_result(run, scalar)


def _select_pretraining_base(run: Path) -> _SelectedPretrainingBase:
    with run_access_lock(run, exclusive=False):
        recovered = recover_latest_index(
            run,
            writable=False,
            verification=VerificationLevel.MANIFEST_TRUSTED,
        )
        if not isinstance(recovered.run, PretrainingRunManifest):
            raise SMLArtifactError("LoRA base must be a pretraining run")
        if recovered.run.model.get("rope_scaling_factor") != 1.0:
            raise SMLArtifactError(
                "copied base rope_scaling_factor must be exactly 1.0"
            )
        tokenizer = read_manifest(
            run / "tokenizer",
            TokenizerManifest,
            VerificationLevel.FULL,
        ).manifest
        if tokenizer.identity != recovered.run.tokenizer_identity:
            raise SMLArtifactError("run tokenizer identity does not match run.json")
        tokenizer_files = _read_tokenizer_files(run / "tokenizer")
        with open_checkpoint_reader(
            run,
            step=recovered.step,
            expected_checkpoint_identity=recovered.checkpoint.identity,
            verification=VerificationLevel.FULL,
            load_array_groups=frozenset({"model.safetensors", "master.safetensors"}),
            hold_lock=False,
        ) as reader:
            reader.read_contents()
            working_bytes = reader.read_payload_bytes("model.safetensors")
            base_step = reader.resolved
        if not isinstance(base_step.run, PretrainingRunManifest):
            raise SMLArtifactError("LoRA base must be a pretraining run")
        model_mapping = dict(base_step.run.model)
        precision_mapping = dict(base_step.run.precision)
        model_config = ModelConfig(**model_mapping)
        if model_config.rope_scaling_factor != 1.0:
            raise SMLArtifactError("resolution never substitutes rope_scaling_factor")
        tokenizer_identity = tokenizer.identity
        return _SelectedPretrainingBase(
            step=base_step,
            working_bytes=working_bytes,
            tokenizer_files=tokenizer_files,
            model=model_mapping,
            precision=precision_mapping,
            tokenizer_identity=tokenizer_identity,
            model_config=model_config,
            identity=_run_step_model_identity(base_step, tokenizer_identity),
        )


def finetune(config: SwagTrainingConfig) -> SwagTrainingResult:
    if not isinstance(config, SwagTrainingConfig):
        raise TypeError("config must be a SwagTrainingConfig")
    config = _resolved_fresh_config(config)
    with run_writer_lock(config.output_run):
        if config.output_run.exists() or config.output_run.is_symlink():
            raise SMLArtifactError(
                f"fresh run target already exists: {config.output_run}"
            )
        selected = _select_pretraining_base(config.base_checkpoint)
        bundle = _verified_swag(
            config.data,
            expected_identity=None,
            tokenizer_identity=selected.tokenizer_identity,
            base_identity=selected.identity,
        )
        runtime: tuple[SMLLanguageModel, _RestoredSwagState] | None = None

        def build(private_run: Path) -> LoRARunManifest:
            nonlocal runtime
            _write_tokenizer_directory(
                private_run / "tokenizer", selected.tokenizer_files
            )
            snapshot = _copy_base_snapshot(
                private_run,
                base_step=selected.step,
                working_bytes=selected.working_bytes,
                model=selected.model,
                precision=selected.precision,
                tokenizer_identity=selected.tokenizer_identity,
            )
            copied_tokenizer = read_manifest(
                private_run / "tokenizer",
                TokenizerManifest,
                VerificationLevel.FULL,
            ).manifest
            if copied_tokenizer.identity != selected.tokenizer_identity:
                raise SMLArtifactError("copied tokenizer identity does not match base")
            (private_run / "checkpoints").mkdir()
            manifest = _run_manifest(
                config,
                model=selected.model,
                tokenizer_identity=selected.tokenizer_identity,
                base_identity=snapshot.identity,
                data_identity=bundle.manifest.identity,
            )
            (private_run / "run.json").write_bytes(canonical_json_bytes(manifest))
            model_key, trainer_key = mx.random.split(mx.random.key(config.seed))
            model, adapters, frozen_base = _wrap_copied_base(
                selected.model_config,
                _load_safetensors(private_run / "base", "model.safetensors"),
                config.lora,
                model_key,
            )
            optimizer = initialize_adam_state(adapters)
            trainer = initial_swag_trainer_state(adapters, key=trainer_key)
            mx.eval(adapters, optimizer.to_tree(), trainer.to_tree(), frozen_base)
            restored = _RestoredSwagState(
                adapters,
                frozen_base,
                optimizer,
                trainer,
                ScalarSwagState(0, 0, 0, SwagCursor.initial()),
            )
            publish_checkpoint(private_run, _checkpoint_builder(manifest, restored))
            runtime = (model, restored)
            return manifest

        published: Published[LoRARunManifest] = publish_run(config.output_run, build)
        if runtime is None:
            raise SMLArtifactError("fresh run builder did not return runtime state")
        if published.manifest.identity != published.manifest.recompute_identity():
            raise SMLArtifactError("published run identity changed during creation")
        return _run_training(
            config.output_run,
            published.manifest,
            config,
            runtime[0],
            runtime[1],
            bundle,
        )


def resume_finetune(
    run: Path,
    *,
    data: Path,
    overrides: ResumeOverrides,
) -> SwagTrainingResult:
    if not isinstance(run, Path):
        raise TypeError("run must be a Path")
    if not isinstance(data, Path):
        raise TypeError("data must be a Path")
    if not isinstance(overrides, ResumeOverrides):
        raise TypeError("overrides must be ResumeOverrides")
    with run_writer_lock(run):
        resolved = resolve_latest_step(
            run,
            writable=False,
            verification=VerificationLevel.MANIFEST_TRUSTED,
        )
        if not isinstance(resolved.run, LoRARunManifest):
            raise SMLArtifactError("LoRA resume requires a LoRA run")
        if resolved.run.model.get("rope_scaling_factor") != 1.0:
            raise SMLArtifactError("LoRA run rope_scaling_factor must be exactly 1.0")
        config = _config_from_run(run, data, resolved.run, overrides)
        bundle = _verified_swag(
            data,
            expected_identity=resolved.run.data_identity,
            tokenizer_identity=resolved.run.tokenizer_identity,
            base_identity=None,
        )
        snapshot = read_manifest(
            run / "base",
            BaseSnapshotManifest,
            VerificationLevel.FULL,
        ).manifest
        if snapshot.identity != resolved.run.base_identity:
            raise SMLArtifactError("run base snapshot identity does not match run.json")
        if snapshot.model.get("rope_scaling_factor") != 1.0:
            raise SMLArtifactError(
                "copied base rope_scaling_factor must be exactly 1.0"
            )
        if snapshot.tokenizer_identity != resolved.run.tokenizer_identity:
            raise SMLArtifactError(
                "run base tokenizer identity does not match run.json"
            )
        tokenizer = read_manifest(
            run / "tokenizer",
            TokenizerManifest,
            VerificationLevel.FULL,
        ).manifest
        if tokenizer.identity != resolved.run.tokenizer_identity:
            raise SMLArtifactError("run tokenizer identity does not match run.json")
        retained = prune_to_latest(run)
        if retained.checkpoint.identity != resolved.checkpoint.identity:
            raise SMLArtifactError("latest checkpoint changed during writable recovery")
        resolved = retained
        scalar = _read_scalar_state(resolved)
        if _limit_reached(config, scalar):
            return _training_result(run, scalar)

        adapters, optimizer, trainer, scalar = _restore_adapter_checkpoint(resolved)
        base_arrays = _load_base_snapshot_arrays(run)
        model_config = ModelConfig(**_mapping_dict(resolved.run.model, "model"))
        model_key, _trainer_key = mx.random.split(mx.random.key(config.seed))
        model, _initialized, frozen_base = _wrap_copied_base(
            model_config,
            base_arrays,
            config.lora,
            model_key,
        )
        restored = _RestoredSwagState(
            adapters,
            frozen_base,
            optimizer,
            trainer,
            scalar,
        )
        return _run_training(
            run,
            resolved.run,
            config,
            model,
            restored,
            bundle,
        )


def _reject_direct_step_path(path: Path) -> Path:
    if not isinstance(path, Path):
        raise TypeError("path must be a Path")
    if path.name.startswith("step-") or (path / "checkpoint.json").exists():
        raise SMLArtifactError("direct checkpoint step paths are rejected")
    return path


def export_merged(checkpoint: Path, output: Path) -> ExportResult:
    checkpoint = _reject_direct_step_path(checkpoint)
    if not isinstance(output, Path):
        raise TypeError("output must be a Path")
    with run_access_lock(checkpoint, exclusive=False):
        recovered = recover_latest_index(
            checkpoint,
            writable=False,
            verification=VerificationLevel.MANIFEST_TRUSTED,
        )
        if not isinstance(recovered.run, LoRARunManifest):
            raise SMLArtifactError("merged export requires a LoRA run")
        if recovered.run.model.get("rope_scaling_factor") != 1.0:
            raise SMLArtifactError("LoRA run rope_scaling_factor must be exactly 1.0")
        snapshot = read_manifest(
            checkpoint / "base",
            BaseSnapshotManifest,
            VerificationLevel.FULL,
        ).manifest
        if snapshot.identity != recovered.run.base_identity:
            raise SMLArtifactError("run base snapshot identity does not match run.json")
        if snapshot.model.get("rope_scaling_factor") != 1.0:
            raise SMLArtifactError(
                "copied base rope_scaling_factor must be exactly 1.0"
            )
        tokenizer = read_manifest(
            checkpoint / "tokenizer",
            TokenizerManifest,
            VerificationLevel.FULL,
        ).manifest
        if tokenizer.identity != recovered.run.tokenizer_identity:
            raise SMLArtifactError("export tokenizer identity does not match the run")
        tokenizer_files = _read_tokenizer_files(checkpoint / "tokenizer")
        adapters, _optimizer, _trainer, _scalar = _restore_adapter_checkpoint(
            recovered,
            hold_lock=False,
        )
        base_arrays = _load_base_snapshot_arrays(checkpoint)
        model_config = ModelConfig(**_mapping_dict(recovered.run.model, "model"))
        lora = lora_config_from_mapping(_mapping_dict(recovered.run.lora, "lora"))
        model_key, _ignored = mx.random.split(mx.random.key(0))
        model, _initialized, _frozen = _wrap_copied_base(
            model_config,
            base_arrays,
            lora,
            model_key,
        )
        load_lora_state_dict(model, dict(tree_flatten(adapters)))
        merged = merged_model_weights(model)
        mx.eval(*merged.values())
        run_manifest = recovered.run
        source_step = recovered.step

    def build(private_path: Path) -> ExportManifest:
        _write_tokenizer_directory(private_path / "tokenizer", tokenizer_files)
        copied_tokenizer = read_manifest(
            private_path / "tokenizer",
            TokenizerManifest,
            VerificationLevel.FULL,
        ).manifest
        if copied_tokenizer.identity != run_manifest.tokenizer_identity:
            raise SMLArtifactError("export tokenizer identity does not match the run")
        weights_path = private_path / "model.safetensors"
        mx.save_safetensors(str(weights_path), dict(sorted(merged.items())))
        model_ref = _array_payload_ref(weights_path, "model.safetensors", merged)
        tokenizer_model = _payload_ref(
            private_path / "tokenizer" / "tokenizer.model",
            "tokenizer/tokenizer.model",
        )
        tokenizer_vocab = _payload_ref(
            private_path / "tokenizer" / "tokenizer.vocab",
            "tokenizer/tokenizer.vocab",
        )
        manifest = ExportManifest(
            kind="export",
            version=1,
            identity=_PLACEHOLDER_IDENTITY,
            model=dict(run_manifest.model),
            precision=dict(run_manifest.precision),
            tokenizer_identity=run_manifest.tokenizer_identity,
            model_weights=model_ref,
            tokenizer_model=tokenizer_model,
            tokenizer_vocab=tokenizer_vocab,
            diagnostic_source_run_identity=run_manifest.identity,
            diagnostic_source_step=source_step,
        )
        return replace(manifest, identity=manifest.recompute_identity())

    published = publish_immutable_bundle(output, build)
    return ExportResult(path=published.path)


__all__ = (
    "ExportResult",
    "SwagKernelConfig",
    "SwagKernels",
    "SwagTrainerState",
    "SwagTrainingConfig",
    "SwagTrainingResult",
    "build_swag_kernels",
    "default_swag_optimizer_config",
    "export_merged",
    "finetune",
    "initial_swag_trainer_state",
    "resume_finetune",
    "score_candidates",
)
