"""Explicit MLX kernels and portable pretraining-run orchestration."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten, tree_map, tree_unflatten

from sml.artifacts.checkpoint import (
    Published,
    ResolvedStep,
    open_checkpoint_reader,
    prune_to_latest,
    publish_checkpoint,
    publish_run,
    resolve_latest_step,
    run_writer_lock,
)
from sml.artifacts.manifest import (
    ArrayPayloadRef,
    ArraySpec,
    ArtifactRoot,
    PayloadRef,
    PretrainingCheckpointManifest,
    PretrainingDataManifest,
    PretrainingRunManifest,
    TokenizerManifest,
    VerificationLevel,
    canonical_json_bytes,
    file_identity,
    read_manifest,
)
from sml.data.pretraining import (
    PreparedDataBundle,
    PretrainingBatchStream,
    PretrainingCursor,
    canonicalize_pretraining_cursor,
    preflight_pretraining_bundle,
)
from sml.errors import SMLArtifactError, SMLConfigurationError, SMLDataError
from sml.model.config import ModelConfig
from sml.model.language_model import SMLLanguageModel, causal_lm_loss
from sml.training.common import (
    AdamState,
    BaseParameterState,
    CheckpointPolicy,
    LoaderConfig,
    OptimizerConfig,
    PrecisionConfig,
    PretrainingConfig,
    ResumeOverrides,
    TrainerState,
    WeightDecayPolicy,
    accumulate_fp32,
    adamw_mixed_precision_update_tree,
    build_weight_decay_tree,
    initialize_adam_state,
    initialize_base_parameter_state,
    learning_rate_at,
    normalize_and_clip,
)
from sml.training.random import counter_random_key

_PLACEHOLDER_IDENTITY = "sha256:" + "0" * 64
_CHECKPOINT_ARRAY_PATHS = (
    "model.safetensors",
    "master.safetensors",
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


@dataclass(frozen=True, slots=True)
class MicrostepState:
    parameters: BaseParameterState
    trainer: TrainerState


@dataclass(frozen=True, slots=True)
class OptimizerStepState:
    parameters: BaseParameterState
    optimizer: AdamState
    trainer: TrainerState
    metrics: dict


@dataclass(frozen=True, slots=True)
class PretrainingKernels:
    compiled_microstep_core: object
    compiled_optimizer_step_core: object
    eager_microstep_core: object
    eager_optimizer_step_core: object

    def microstep(
        self,
        parameters: BaseParameterState,
        trainer: TrainerState,
        rows: object,
    ) -> MicrostepState:
        rows_array = mx.array(rows)
        next_working, next_trainer_tree = self.compiled_microstep_core(
            parameters.working_parameters,
            trainer.to_tree(),
            rows_array[:, :-1],
            rows_array[:, 1:],
        )
        return MicrostepState(
            parameters=BaseParameterState.from_compiled_tree(
                (parameters.master_parameters, next_working)
            ),
            trainer=TrainerState.from_compiled_tree(next_trainer_tree),
        )

    def optimizer_step(
        self,
        parameters: BaseParameterState,
        optimizer: AdamState,
        trainer: TrainerState,
    ) -> OptimizerStepState:
        (
            next_masters,
            next_working,
            next_adam_tree,
            next_trainer_tree,
            metrics,
        ) = self.compiled_optimizer_step_core(
            parameters.master_parameters,
            parameters.working_parameters,
            optimizer.to_tree(),
            trainer.to_tree(),
        )
        mx.eval(
            next_masters,
            next_working,
            next_adam_tree,
            next_trainer_tree,
            metrics,
        )
        return OptimizerStepState(
            parameters=BaseParameterState.from_compiled_tree(
                (next_masters, next_working)
            ),
            optimizer=AdamState.from_compiled_tree(next_adam_tree),
            trainer=TrainerState.from_compiled_tree(next_trainer_tree),
            metrics=metrics,
        )


def build_pretraining_kernels(
    model: SMLLanguageModel,
    config: PretrainingConfig,
    weight_decay_tree: dict,
) -> PretrainingKernels:
    """Build eager and compiled kernels over explicit, built-in state trees."""

    def loss_with_key(
        working_parameters: dict,
        input_ids: mx.array,
        labels: mx.array,
        key: mx.array,
    ) -> tuple[mx.array, mx.array]:
        logits, cache_state, next_key = model.forward_arrays(
            working_parameters,
            input_ids,
            attention_mask=None,
            positions=None,
            cache_state=None,
            training=True,
            key=key,
        )
        assert cache_state is None
        if next_key is None:
            raise RuntimeError("training forward did not return a PRNG key")
        valid_mask = labels != model.config.pad_token_id
        return causal_lm_loss(logits, labels, valid_mask), next_key

    loss_and_grad = mx.value_and_grad(loss_with_key)

    def microstep_core(
        working_parameters: dict,
        trainer_tree: tuple[dict, mx.array, mx.array, mx.array],
        input_ids: mx.array,
        labels: mx.array,
    ) -> tuple[dict, tuple[dict, mx.array, mx.array, mx.array]]:
        accumulators, accumulation_count, key, loss_numerator = trainer_tree
        (loss, next_key), gradients = loss_and_grad(
            working_parameters, input_ids, labels, key
        )
        next_accumulators = accumulate_fp32(accumulators, gradients)
        next_count = (accumulation_count + mx.array(1, dtype=mx.int32)).astype(mx.int32)
        next_loss_numerator = (loss_numerator + loss.astype(mx.float32)).astype(
            mx.float32
        )
        return working_parameters, (
            next_accumulators,
            next_count,
            next_key,
            next_loss_numerator,
        )

    def optimizer_step_core(
        master_parameters: dict,
        working_parameters: dict,
        adam_tree: tuple[mx.array, dict, dict],
        trainer_tree: tuple[dict, mx.array, mx.array, mx.array],
    ) -> tuple[
        dict,
        dict,
        tuple[mx.array, dict, dict],
        tuple[dict, mx.array, mx.array, mx.array],
        dict,
    ]:
        del working_parameters
        accumulators, accumulation_count, next_key, loss_numerator = trainer_tree
        gradients = normalize_and_clip(
            accumulators,
            accumulation_count,
            gradient_clip_norm=config.optimizer.gradient_clip_norm,
        )
        next_masters, next_working, next_adam_tree = adamw_mixed_precision_update_tree(
            master_parameters,
            gradients,
            adam_tree,
            config.optimizer,
            weight_decay_tree,
        )
        reset_accumulators = tree_map(
            lambda accumulator: accumulator - accumulator,
            accumulators,
        )
        next_trainer_tree = (
            reset_accumulators,
            (accumulation_count - accumulation_count).astype(mx.int32),
            next_key,
            (loss_numerator - loss_numerator).astype(mx.float32),
        )
        safe_count = mx.maximum(
            accumulation_count.astype(mx.float32), mx.array(1.0, dtype=mx.float32)
        )
        metrics = {
            "learning_rate": learning_rate_at(adam_tree[0], config.optimizer),
            "accumulation_count": accumulation_count,
            "loss_numerator": loss_numerator,
            "loss": (loss_numerator / safe_count).astype(mx.float32),
        }
        return (
            next_masters,
            next_working,
            next_adam_tree,
            next_trainer_tree,
            metrics,
        )

    return PretrainingKernels(
        compiled_microstep_core=(
            mx.compile(microstep_core) if config.compile else microstep_core
        ),
        compiled_optimizer_step_core=(
            mx.compile(optimizer_step_core) if config.compile else optimizer_step_core
        ),
        eager_microstep_core=microstep_core,
        eager_optimizer_step_core=optimizer_step_core,
    )


@dataclass(frozen=True, slots=True)
class ScalarTrainingState:
    step: int
    rows: int
    microsteps: int
    cursor: PretrainingCursor

    def __post_init__(self) -> None:
        for name in ("step", "rows", "microsteps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise SMLArtifactError(f"checkpoint scalar {name} must be nonnegative")
        if not isinstance(self.cursor, PretrainingCursor):
            raise SMLArtifactError("checkpoint cursor must be a PretrainingCursor")


@dataclass(frozen=True, slots=True)
class TrainingResult:
    run: Path
    step: int
    epoch: int
    rows: int


@dataclass(frozen=True, slots=True)
class _RestoredTrainingState:
    parameters: BaseParameterState
    optimizer: AdamState
    trainer: TrainerState
    scalar: ScalarTrainingState


def _plain_nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SMLArtifactError(f"{name} must be a nonnegative integer")
    return value


def _json_object_no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise SMLArtifactError(f"duplicate checkpoint scalar key: {key}")
        result[key] = value
    return result


def _scalar_document(
    state: ScalarTrainingState,
    *,
    owning_run_identity: str,
) -> dict[str, object]:
    return {
        "kind": "pretraining-state",
        "version": 1,
        "owning_run_identity": owning_run_identity,
        "step": state.step,
        "rows": state.rows,
        "microsteps": state.microsteps,
        "cursor": {
            "epoch": state.cursor.epoch,
            "shard_order_position": state.cursor.shard_order_position,
            "row_offset": state.cursor.row_offset,
        },
    }


def _parse_scalar_document(
    payload: bytes,
    resolved: ResolvedStep,
) -> ScalarTrainingState:
    try:
        raw = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_json_object_no_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant: {value}")
            ),
        )
        if not isinstance(raw, dict) or set(raw) != {
            "kind",
            "version",
            "owning_run_identity",
            "step",
            "rows",
            "microsteps",
            "cursor",
        }:
            raise SMLArtifactError("checkpoint scalar state has invalid fields")
        if raw["kind"] != "pretraining-state" or raw["version"] != 1:
            raise SMLArtifactError("checkpoint scalar state has invalid schema")
        if raw["owning_run_identity"] != resolved.run.identity:
            raise SMLArtifactError("checkpoint scalar state belongs to another run")
        step = _plain_nonnegative_int(raw["step"], "checkpoint scalar step")
        if step != resolved.step:
            raise SMLArtifactError("checkpoint scalar step does not match checkpoint")
        cursor = raw["cursor"]
        if not isinstance(cursor, dict) or set(cursor) != {
            "epoch",
            "shard_order_position",
            "row_offset",
        }:
            raise SMLArtifactError("checkpoint scalar cursor has invalid fields")
        state = ScalarTrainingState(
            step=step,
            rows=_plain_nonnegative_int(raw["rows"], "checkpoint scalar rows"),
            microsteps=_plain_nonnegative_int(
                raw["microsteps"], "checkpoint scalar microsteps"
            ),
            cursor=PretrainingCursor(
                epoch=_plain_nonnegative_int(cursor["epoch"], "cursor epoch"),
                shard_order_position=_plain_nonnegative_int(
                    cursor["shard_order_position"], "cursor shard position"
                ),
                row_offset=_plain_nonnegative_int(
                    cursor["row_offset"], "cursor row offset"
                ),
            ),
        )
        if canonical_json_bytes(raw) != payload:
            raise SMLArtifactError("checkpoint scalar state is not canonical JSON")
        return state
    except SMLArtifactError:
        raise
    except (json.JSONDecodeError, UnicodeError, TypeError, ValueError) as error:
        raise SMLArtifactError("invalid checkpoint scalar state") from error


def read_scalar_state(resolved: ResolvedStep) -> ScalarTrainingState:
    if not isinstance(resolved, ResolvedStep):
        raise TypeError("resolved must be a ResolvedStep")
    with open_checkpoint_reader(
        resolved.step_directory.parent.parent,
        step=resolved.step,
        expected_checkpoint_identity=resolved.checkpoint.identity,
    ) as reader:
        contents = reader.read_contents()
        return _parse_scalar_document(
            canonical_json_bytes(dict(contents.scalar_state)),
            reader.resolved,
        )


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


def _array_payload_ref(path: Path, logical_path: str, arrays: Mapping[str, mx.array]):
    with path.open("rb") as payload:
        identity = file_identity(payload)
    return ArrayPayloadRef(
        payload=PayloadRef(logical_path, identity, path.stat().st_size),
        arrays=tuple(
            ArraySpec(name, tuple(array.shape), _dtype_name(array))
            for name, array in sorted(arrays.items())
        ),
    )


def _flatten_checkpoint_groups(
    parameters: BaseParameterState,
    optimizer: AdamState,
    trainer: TrainerState,
) -> dict[str, dict[str, mx.array]]:
    master = dict(sorted(tree_flatten(parameters.master_parameters)))
    model = dict(sorted(tree_flatten(parameters.working_parameters)))
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
        "accumulation_count": trainer.accumulation_count,
        "next_key": trainer.next_key,
        "loss_numerator": trainer.loss_numerator,
        **{
            f"accumulators.{name}": value
            for name, value in tree_flatten(trainer.accumulators)
        },
    }
    return {
        "model.safetensors": dict(sorted(model.items())),
        "master.safetensors": dict(sorted(master.items())),
        "optimizer.safetensors": dict(sorted(optimizer_arrays.items())),
        "trainer.safetensors": dict(sorted(trainer_arrays.items())),
    }


def _require_empty_trainer_state(trainer: TrainerState) -> None:
    mx.eval(trainer.to_tree())
    if int(trainer.accumulation_count.item()) != 0:
        raise SMLArtifactError("checkpoint trainer accumulation must be empty")
    if float(trainer.loss_numerator.item()) != 0.0:
        raise SMLArtifactError("checkpoint trainer loss numerator must be empty")
    if any(
        bool(mx.any(value != 0)) for _name, value in tree_flatten(trainer.accumulators)
    ):
        raise SMLArtifactError("checkpoint trainer accumulators must be empty")


def _checkpoint_builder(
    run_manifest: PretrainingRunManifest,
    state: _RestoredTrainingState,
):
    parameters = BaseParameterState(
        state.parameters.master_parameters,
        state.parameters.working_parameters,
    )
    optimizer = AdamState(
        state.optimizer.step,
        state.optimizer.first_moments,
        state.optimizer.second_moments,
    )
    trainer = TrainerState(
        state.trainer.accumulators,
        state.trainer.accumulation_count,
        state.trainer.next_key,
        state.trainer.loss_numerator,
    )
    _require_empty_trainer_state(trainer)
    if int(optimizer.step.item()) != state.scalar.step:
        raise SMLArtifactError("Adam step must match the checkpoint step")
    groups = _flatten_checkpoint_groups(
        parameters,
        optimizer,
        trainer,
    )
    master_names = set(groups["master.safetensors"])
    expected_optimizer = {
        "step",
        *(f"first_moments.{name}" for name in master_names),
        *(f"second_moments.{name}" for name in master_names),
    }
    expected_trainer = {
        "accumulation_count",
        "next_key",
        "loss_numerator",
        *(f"accumulators.{name}" for name in master_names),
    }
    if set(groups["optimizer.safetensors"]) != expected_optimizer:
        raise SMLArtifactError("optimizer checkpoint keys must match master keys")
    if set(groups["trainer.safetensors"]) != expected_trainer:
        raise SMLArtifactError("trainer checkpoint keys must match master keys")
    master = groups["master.safetensors"]
    for prefix, group_name in (
        ("first_moments.", "optimizer.safetensors"),
        ("second_moments.", "optimizer.safetensors"),
        ("accumulators.", "trainer.safetensors"),
    ):
        group = groups[group_name]
        for name, master_array in master.items():
            if group[f"{prefix}{name}"].shape != master_array.shape:
                raise SMLArtifactError(
                    "optimizer/trainer checkpoint shapes must match master shapes"
                )

    def build(private_step: Path) -> PretrainingCheckpointManifest:
        references: list[ArrayPayloadRef] = []
        for logical_path in _CHECKPOINT_ARRAY_PATHS:
            arrays = groups[logical_path]
            path = private_step / logical_path
            mx.save_safetensors(path, arrays)
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
        with state_path.open("rb") as payload:
            state_identity = file_identity(payload)
        references_by_path = {
            reference.payload.logical_path: reference for reference in references
        }
        manifest = PretrainingCheckpointManifest(
            kind="pretraining-checkpoint",
            version=1,
            identity=_PLACEHOLDER_IDENTITY,
            owning_run_identity=run_manifest.identity,
            step=state.scalar.step,
            scalar_state=PayloadRef(
                "state.json",
                state_identity,
                state_path.stat().st_size,
            ),
            model=references_by_path["model.safetensors"],
            master=references_by_path["master.safetensors"],
            optimizer=references_by_path["optimizer.safetensors"],
            trainer=references_by_path["trainer.safetensors"],
        )
        return replace(manifest, identity=manifest.recompute_identity())

    return build


def _unflatten_prefixed(arrays: Mapping[str, mx.array], prefix: str) -> dict:
    items = [
        (name.removeprefix(prefix), value)
        for name, value in arrays.items()
        if name.startswith(prefix)
    ]
    if not items:
        raise SMLArtifactError(f"checkpoint has no arrays under {prefix}")
    return tree_unflatten(items)


def _restore_checkpoint(resolved: ResolvedStep) -> _RestoredTrainingState:
    with open_checkpoint_reader(
        resolved.step_directory.parent.parent,
        step=resolved.step,
        expected_checkpoint_identity=resolved.checkpoint.identity,
    ) as reader:
        verified = reader.resolved
        if not isinstance(verified.checkpoint, PretrainingCheckpointManifest):
            raise SMLArtifactError("pretraining requires a pretraining checkpoint")
        contents = reader.read_contents()
        scalar = _parse_scalar_document(
            canonical_json_bytes(dict(contents.scalar_state)),
            verified,
        )
        groups = {
            logical_path: dict(contents.array_groups[logical_path])
            for logical_path in _CHECKPOINT_ARRAY_PATHS
        }

        masters = groups["master.safetensors"]
        working = groups["model.safetensors"]
        try:
            parameters = BaseParameterState(
                tree_unflatten(list(masters.items())),
                tree_unflatten(list(working.items())),
            )
            optimizer_arrays = groups["optimizer.safetensors"]
            if set(optimizer_arrays) != {
                "step",
                *(f"first_moments.{name}" for name in masters),
                *(f"second_moments.{name}" for name in masters),
            }:
                raise SMLArtifactError("optimizer checkpoint keys do not match masters")
            for name, master in masters.items():
                for prefix in ("first_moments.", "second_moments."):
                    if optimizer_arrays[f"{prefix}{name}"].shape != master.shape:
                        raise SMLArtifactError(
                            "optimizer checkpoint shapes do not match masters"
                        )
            optimizer = AdamState(
                optimizer_arrays["step"],
                _unflatten_prefixed(optimizer_arrays, "first_moments."),
                _unflatten_prefixed(optimizer_arrays, "second_moments."),
            )
            trainer_arrays = groups["trainer.safetensors"]
            if set(trainer_arrays) != {
                "accumulation_count",
                "next_key",
                "loss_numerator",
                *(f"accumulators.{name}" for name in masters),
            }:
                raise SMLArtifactError("trainer checkpoint keys do not match masters")
            for name, master in masters.items():
                if trainer_arrays[f"accumulators.{name}"].shape != master.shape:
                    raise SMLArtifactError(
                        "trainer checkpoint shapes do not match masters"
                    )
            trainer = TrainerState(
                _unflatten_prefixed(trainer_arrays, "accumulators."),
                trainer_arrays["accumulation_count"],
                trainer_arrays["next_key"],
                trainer_arrays["loss_numerator"],
            )
            _require_empty_trainer_state(trainer)
            if int(optimizer.step.item()) != scalar.step:
                raise SMLArtifactError(
                    "checkpoint Adam step does not match scalar step"
                )
            return _RestoredTrainingState(parameters, optimizer, trainer, scalar)
        except SMLArtifactError:
            raise
        except (SMLConfigurationError, TypeError, ValueError) as error:
            raise SMLArtifactError("invalid pretraining checkpoint arrays") from error


def _copy_run_tokenizer(data: Path, private_run: Path) -> None:
    destination = private_run / "tokenizer"
    destination.mkdir()
    with ArtifactRoot.open(data, writable=False) as source:
        for name in (
            "tokenizer/manifest.json",
            "tokenizer/tokenizer.model",
            "tokenizer/tokenizer.vocab",
        ):
            with source.open_payload(name) as payload:
                destination.joinpath(Path(name).name).write_bytes(payload.read())


def _verified_data(
    path: Path,
    *,
    expected_identity: str | None,
    model: ModelConfig,
    loader: LoaderConfig,
) -> PreparedDataBundle:
    verified = read_manifest(
        path,
        PretrainingDataManifest,
        VerificationLevel.FULL,
    )
    manifest = verified.manifest
    if expected_identity is not None and manifest.identity != expected_identity:
        raise SMLArtifactError("prepared-data identity does not match run.json")
    tokenizer = read_manifest(
        path / "tokenizer",
        TokenizerManifest,
        VerificationLevel.FULL,
    ).manifest
    if tokenizer.identity != manifest.tokenizer_identity:
        raise SMLArtifactError("prepared-data tokenizer identity mismatch")
    if manifest.sequence_length != model.original_context_length:
        raise SMLArtifactError("prepared-data sequence length does not match model")
    if (
        tokenizer.vocab_size,
        tokenizer.bos_token_id,
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
        tokenizer.unk_token_id,
    ) != (
        model.vocab_size,
        model.bos_token_id,
        model.eos_token_id,
        model.pad_token_id,
        model.unk_token_id,
    ):
        raise SMLArtifactError("prepared-data tokenizer metadata does not match model")
    bundle = PreparedDataBundle(path, manifest, VerificationLevel.FULL)
    preflight_pretraining_bundle(bundle, batch_size=loader.microbatch_size)
    return bundle


def _resolved_fresh_config(config: PretrainingConfig) -> PretrainingConfig:
    if config.model.rope_scaling_factor != 1.0:
        raise SMLConfigurationError(
            "pretraining model rope_scaling_factor must be exactly 1.0"
        )
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
    config: PretrainingConfig, data: PreparedDataBundle
) -> PretrainingRunManifest:
    checkpoint = {
        "interval": config.checkpoint.interval,
        "maximum_steps": config.maximum_steps,
        "maximum_epochs": config.maximum_epochs,
        "log_interval": config.log_interval,
        "seed": config.seed,
        "compile": config.compile,
    }
    manifest = PretrainingRunManifest(
        kind="pretraining-run",
        version=1,
        identity=_PLACEHOLDER_IDENTITY,
        model=dataclasses.asdict(config.model),
        precision=dataclasses.asdict(config.precision),
        optimizer=dataclasses.asdict(config.optimizer),
        loader=dataclasses.asdict(config.loader),
        checkpoint=checkpoint,
        tokenizer_identity=data.manifest.tokenizer_identity,
        data_identity=data.manifest.identity,
        diagnostic_data_locator=str(config.data),
    )
    return replace(manifest, identity=manifest.recompute_identity())


def _mapping_dict(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise SMLArtifactError(f"saved {name} configuration must be a mapping")
    return dict(value)


def _config_from_run(
    run: Path,
    data: Path,
    manifest: PretrainingRunManifest,
    overrides: ResumeOverrides,
) -> PretrainingConfig:
    try:
        model = ModelConfig(**_mapping_dict(manifest.model, "model"))
        precision = PrecisionConfig(**_mapping_dict(manifest.precision, "precision"))
        loader = LoaderConfig(**_mapping_dict(manifest.loader, "loader"))
        optimizer_values = _mapping_dict(manifest.optimizer, "optimizer")
        weight_decay = WeightDecayPolicy(
            **_mapping_dict(optimizer_values.pop("weight_decay"), "weight_decay")
        )
        optimizer = OptimizerConfig(weight_decay=weight_decay, **optimizer_values)
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
        return PretrainingConfig(
            data=data,
            output_run=run,
            model=model,
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
        raise SMLArtifactError("invalid saved pretraining configuration") from error


def _initial_state(
    config: PretrainingConfig,
) -> tuple[SMLLanguageModel, _RestoredTrainingState]:
    model_key, _unused_key = mx.random.split(mx.random.key(config.seed))
    model = SMLLanguageModel(config.model, key=model_key)
    working = model.parameters()
    mx.eval(working)
    parameters = initialize_base_parameter_state(working)
    optimizer = initialize_adam_state(parameters.master_parameters)
    trainer = TrainerState(
        accumulators=tree_map(mx.zeros_like, parameters.master_parameters),
        accumulation_count=mx.array(0, dtype=mx.int32),
        next_key=counter_random_key(config.seed, 0),
        loss_numerator=mx.array(0.0, dtype=mx.float32),
    )
    mx.eval(parameters.to_tree(), optimizer.to_tree(), trainer.to_tree())
    return model, _RestoredTrainingState(
        parameters,
        optimizer,
        trainer,
        ScalarTrainingState(0, 0, 0, PretrainingCursor.initial()),
    )


def _limit_reached(config: PretrainingConfig, state: ScalarTrainingState) -> bool:
    return (
        config.maximum_steps is not None and state.step >= config.maximum_steps
    ) or (
        config.maximum_epochs is not None
        and state.cursor.epoch >= config.maximum_epochs
    )


def _publish_training_state(
    run: Path,
    manifest: PretrainingRunManifest,
    state: _RestoredTrainingState,
) -> ResolvedStep:
    published = publish_checkpoint(run, _checkpoint_builder(manifest, state))
    retained = prune_to_latest(run)
    _require_retained_publication(published, retained)
    if retained.step != published.step:
        raise SMLArtifactError("retention did not preserve the published latest step")
    return retained


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


def _training_result(run: Path, state: ScalarTrainingState) -> TrainingResult:
    return TrainingResult(run, state.step, state.cursor.epoch, state.rows)


def _run_training(
    run: Path,
    manifest: PretrainingRunManifest,
    config: PretrainingConfig,
    model: SMLLanguageModel,
    restored: _RestoredTrainingState,
    stream: PretrainingBatchStream,
) -> TrainingResult:
    parameters = restored.parameters
    optimizer = restored.optimizer
    trainer = restored.trainer
    scalar = restored.scalar
    weight_decay_tree = build_weight_decay_tree(
        parameters.working_parameters,
        config.optimizer.weight_decay,
    )
    kernels = build_pretraining_kernels(model, config, weight_decay_tree)
    last_published_step = scalar.step
    window_microsteps = 0
    pending_cursor: PretrainingCursor | None = None
    current_epoch = scalar.cursor.epoch

    def complete_update() -> None:
        nonlocal parameters, optimizer, trainer, scalar
        nonlocal window_microsteps, pending_cursor, last_published_step
        if window_microsteps == 0 or pending_cursor is None:
            return
        updated = kernels.optimizer_step(parameters, optimizer, trainer)
        parameters = updated.parameters
        optimizer = updated.optimizer
        trainer = updated.trainer
        stream.commit(pending_cursor)
        scalar = ScalarTrainingState(
            step=scalar.step + 1,
            rows=scalar.rows + window_microsteps * config.loader.microbatch_size,
            microsteps=scalar.microsteps + window_microsteps,
            cursor=pending_cursor,
        )
        window_microsteps = 0
        pending_cursor = None
        state = _RestoredTrainingState(parameters, optimizer, trainer, scalar)
        if scalar.step % config.checkpoint.interval == 0:
            _publish_training_state(run, manifest, state)
            last_published_step = scalar.step

    while not _limit_reached(config, scalar):
        if config.maximum_epochs is not None and current_epoch >= config.maximum_epochs:
            break
        for envelope in stream.iter_epoch(current_epoch):
            if _limit_reached(config, scalar):
                break
            with envelope:
                microstep = kernels.microstep(parameters, trainer, envelope.rows)
                pending_cursor = envelope.cursor_after
            parameters = microstep.parameters
            trainer_tree = microstep.trainer.to_tree()
            trainer = TrainerState.from_compiled_tree(
                (
                    trainer_tree[0],
                    trainer_tree[1],
                    counter_random_key(
                        config.seed,
                        scalar.microsteps + window_microsteps + 1,
                    ),
                    trainer_tree[3],
                )
            )
            window_microsteps += 1
            if window_microsteps == config.loader.gradient_accumulation_steps:
                complete_update()
                if _limit_reached(config, scalar):
                    break
        if window_microsteps:
            complete_update()
        current_epoch += 1

    final_state = _RestoredTrainingState(parameters, optimizer, trainer, scalar)
    if last_published_step != scalar.step:
        _publish_training_state(run, manifest, final_state)
    return _training_result(run, scalar)


def train(config: PretrainingConfig) -> TrainingResult:
    if not isinstance(config, PretrainingConfig):
        raise TypeError("config must be a PretrainingConfig")
    config = _resolved_fresh_config(config)
    stream: PretrainingBatchStream | None = None
    runtime: tuple[SMLLanguageModel, _RestoredTrainingState] | None = None
    with run_writer_lock(config.output_run):
        try:
            if config.output_run.exists() or config.output_run.is_symlink():
                raise SMLArtifactError(
                    f"fresh run target already exists: {config.output_run}"
                )
            data = _verified_data(
                config.data,
                expected_identity=None,
                model=config.model,
                loader=config.loader,
            )
            stream = PretrainingBatchStream(
                data,
                batch_size=config.loader.microbatch_size,
                seed=config.loader.epoch_seed,
                prefetch_depth=config.loader.prefetch_depth,
                cursor=PretrainingCursor.initial(),
            )
            manifest = _run_manifest(config, data)

            def build(private_run: Path) -> PretrainingRunManifest:
                nonlocal runtime
                _copy_run_tokenizer(config.data, private_run)
                (private_run / "checkpoints").mkdir()
                (private_run / "run.json").write_bytes(canonical_json_bytes(manifest))
                runtime = _initial_state(config)
                publish_checkpoint(
                    private_run,
                    _checkpoint_builder(manifest, runtime[1]),
                )
                return manifest

            published: Published[PretrainingRunManifest] = publish_run(
                config.output_run, build
            )
            if runtime is None:
                raise SMLArtifactError("fresh run builder did not return runtime state")
            if published.manifest.identity != manifest.identity:
                raise SMLArtifactError("published run identity changed during creation")
            return _run_training(
                config.output_run,
                manifest,
                config,
                runtime[0],
                runtime[1],
                stream,
            )
        finally:
            if stream is not None:
                stream.close()


def resume(
    run: Path,
    *,
    data: Path | None,
    overrides: ResumeOverrides,
) -> TrainingResult:
    if not isinstance(run, Path):
        raise TypeError("run must be a Path")
    if data is not None and not isinstance(data, Path):
        raise TypeError("data must be a Path or None")
    if not isinstance(overrides, ResumeOverrides):
        raise TypeError("overrides must be ResumeOverrides")
    stream: PretrainingBatchStream | None = None
    with run_writer_lock(run):
        try:
            resolved = resolve_latest_step(
                run,
                writable=False,
                verification=VerificationLevel.MANIFEST_TRUSTED,
            )
            if not isinstance(resolved.run, PretrainingRunManifest):
                raise SMLArtifactError("pretraining resume requires a pretraining run")
            diagnostic = resolved.run.diagnostic_data_locator
            data_path = (
                data if data is not None else (Path(diagnostic) if diagnostic else None)
            )
            if data_path is None:
                raise SMLArtifactError(
                    "resume requires a prepared-data bundle location"
                )
            config = _config_from_run(run, data_path, resolved.run, overrides)
            prepared = _verified_data(
                data_path,
                expected_identity=resolved.run.data_identity,
                model=config.model,
                loader=config.loader,
            )
            restored = _restore_checkpoint(resolved)
            scalar = restored.scalar
            try:
                canonical_cursor = canonicalize_pretraining_cursor(
                    scalar.cursor,
                    shard_row_counts=prepared.manifest.shard_row_counts,
                    seed=config.loader.epoch_seed,
                )
            except SMLDataError as error:
                raise SMLArtifactError(
                    "checkpoint cursor is invalid for the prepared data"
                ) from error
            if canonical_cursor != scalar.cursor:
                raise SMLArtifactError(
                    "checkpoint cursor is not in canonical prepared-data form"
                )
            retained = prune_to_latest(run)
            if retained.checkpoint.identity != resolved.checkpoint.identity:
                raise SMLArtifactError(
                    "latest checkpoint changed during writable recovery"
                )
            resolved = retained
            if _limit_reached(config, scalar):
                return _training_result(run, scalar)

            stream = PretrainingBatchStream(
                prepared,
                batch_size=config.loader.microbatch_size,
                seed=config.loader.epoch_seed,
                prefetch_depth=config.loader.prefetch_depth,
                cursor=scalar.cursor,
            )
            model = SMLLanguageModel(config.model, key=mx.random.key(config.seed))
            return _run_training(
                run,
                resolved.run,
                config,
                model,
                restored,
                stream,
            )
        finally:
            if stream is not None:
                stream.close()


__all__ = (
    "MicrostepState",
    "OptimizerStepState",
    "PretrainingKernels",
    "ScalarTrainingState",
    "TrainingResult",
    "build_pretraining_kernels",
    "read_scalar_state",
    "resume",
    "train",
)
