"""Shared exact semantic contracts for model-owning artifacts."""

from __future__ import annotations

import dataclasses

import mlx.core as mx

from sml.artifacts.checkpoint import CheckpointReader, verify_checkpoint_current_state
from sml.artifacts.manifest import (
    TRAINING_RNG_SCHEDULE,
    ArrayPayloadRef,
    ArraySpec,
    BaseSnapshotManifest,
    ExportManifest,
    LoRACheckpointManifest,
    LoRARunManifest,
    PretrainingCheckpointManifest,
    PretrainingRunManifest,
    TokenizerManifest,
)
from sml.errors import SMLArtifactError, SMLConfigurationError
from sml.model.config import ModelConfig
from sml.model.language_model import model_parameter_specs
from sml.training.common import (
    CheckpointPolicy,
    LoaderConfig,
    OptimizerConfig,
    PrecisionConfig,
    WeightDecayPolicy,
)
from sml.training.lora import (
    LoRAConfig,
    LoRAPrecisionConfig,
    lora_config_from_mapping,
    lora_parameter_specs,
)
from sml.training.random import counter_random_key


def _exact_configuration(
    configuration: object,
    projection: object,
    *,
    context: str,
) -> None:
    if dataclasses.asdict(configuration) != projection:
        raise SMLArtifactError(f"{context} is not an exact canonical configuration")


def _model_configuration(projection: object, *, context: str) -> ModelConfig:
    if not isinstance(projection, dict) and not hasattr(projection, "items"):
        raise SMLArtifactError(f"{context} model configuration must be a mapping")
    try:
        configuration = ModelConfig(**dict(projection))
        _exact_configuration(
            configuration,
            dict(projection),
            context=f"{context} model configuration",
        )
    except SMLArtifactError:
        raise
    except (TypeError, ValueError) as error:
        raise SMLArtifactError(f"invalid {context} model configuration") from error
    if configuration.rope_scaling_factor != 1.0:
        raise SMLArtifactError(f"{context} rope_scaling_factor must be exactly 1.0")
    return configuration


def _precision_configuration(
    projection: object,
    precision_type: type[PrecisionConfig | LoRAPrecisionConfig],
    *,
    context: str,
) -> None:
    if not isinstance(projection, dict) and not hasattr(projection, "items"):
        raise SMLArtifactError(f"{context} precision must be a mapping")
    try:
        precision = precision_type(**dict(projection))
        _exact_configuration(
            precision,
            dict(projection),
            context=f"{context} precision",
        )
    except SMLArtifactError:
        raise
    except (TypeError, ValueError, SMLConfigurationError) as error:
        raise SMLArtifactError(f"invalid {context} precision") from error


def _optimizer_configuration(projection: object, *, context: str) -> OptimizerConfig:
    if not isinstance(projection, dict) and not hasattr(projection, "items"):
        raise SMLArtifactError(f"{context} optimizer must be a mapping")
    try:
        values = dict(projection)
        weight_decay = values.pop("weight_decay")
        if not isinstance(weight_decay, dict) and not hasattr(weight_decay, "items"):
            raise TypeError("weight_decay must be a mapping")
        configuration = OptimizerConfig(
            weight_decay=WeightDecayPolicy(**dict(weight_decay)),
            **values,
        )
        _exact_configuration(
            configuration,
            dict(projection),
            context=f"{context} optimizer",
        )
        return configuration
    except SMLArtifactError:
        raise
    except (KeyError, TypeError, ValueError, SMLConfigurationError) as error:
        raise SMLArtifactError(f"invalid {context} optimizer") from error


def _loader_configuration(projection: object, *, context: str) -> LoaderConfig:
    if not isinstance(projection, dict) and not hasattr(projection, "items"):
        raise SMLArtifactError(f"{context} loader must be a mapping")
    try:
        configuration = LoaderConfig(**dict(projection))
        _exact_configuration(
            configuration,
            dict(projection),
            context=f"{context} loader",
        )
        return configuration
    except SMLArtifactError:
        raise
    except (TypeError, ValueError, SMLConfigurationError) as error:
        raise SMLArtifactError(f"invalid {context} loader") from error


def checkpoint_configuration(
    projection: object,
    *,
    context: str,
) -> dict[str, object]:
    if not isinstance(projection, dict) and not hasattr(projection, "items"):
        raise SMLArtifactError(f"{context} checkpoint config must be a mapping")
    values = dict(projection)
    expected = {
        "interval",
        "maximum_steps",
        "maximum_epochs",
        "log_interval",
        "seed",
        "compile",
        "rng_schedule",
    }
    if set(values) != expected:
        raise SMLArtifactError(f"{context} checkpoint config has invalid fields")
    try:
        CheckpointPolicy(interval=values["interval"])
        for name in ("maximum_steps", "maximum_epochs"):
            value = values[name]
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 1 <= value <= 2**31 - 1
            ):
                raise ValueError(f"{name} must be a signed-int32 counter or None")
        if values["maximum_steps"] is None and values["maximum_epochs"] is None:
            raise ValueError("one training termination limit is required")
        log_interval = values["log_interval"]
        if (
            isinstance(log_interval, bool)
            or not isinstance(log_interval, int)
            or not 1 <= log_interval <= 2**31 - 1
        ):
            raise ValueError("log_interval must be a signed-int32 counter")
        seed = values["seed"]
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or not 0 <= seed <= 2**32 - 1
        ):
            raise ValueError("seed must be uint32")
        if not isinstance(values["compile"], bool):
            raise TypeError("compile must be bool")
        if values["rng_schedule"] != TRAINING_RNG_SCHEDULE:
            raise ValueError(f"rng_schedule must be {TRAINING_RNG_SCHEDULE!r}")
    except (TypeError, ValueError, SMLConfigurationError) as error:
        raise SMLArtifactError(f"invalid {context} checkpoint config") from error
    return values


def _tokenizer_matches_model(
    tokenizer: TokenizerManifest,
    model: ModelConfig,
    *,
    context: str,
) -> None:
    tokenizer_values = (
        tokenizer.vocab_size,
        tokenizer.bos_token_id,
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
        tokenizer.unk_token_id,
    )
    model_values = (
        model.vocab_size,
        model.bos_token_id,
        model.eos_token_id,
        model.pad_token_id,
        model.unk_token_id,
    )
    if tokenizer_values != model_values:
        raise SMLArtifactError(f"{context} tokenizer metadata does not match model")


def expected_next_key(
    *,
    seed: int,
    microsteps: int,
    model: ModelConfig,
    lora: LoRAConfig | None = None,
) -> mx.array:
    model_sites = model.num_layers if model.hidden_dropout > 0.0 else 0
    lora_sites = (
        model.num_layers * len(lora.target_modules)
        if lora is not None and lora.dropout > 0.0
        else 0
    )
    active_sites = model_sites + lora_sites
    if microsteps == 0 or active_sites == 0:
        return counter_random_key(seed, 0)
    key = counter_random_key(seed, microsteps - 1)
    for _site in range(active_sites):
        key, _unused = mx.random.split(key)
    return key


def _verify_progress(
    scalar: object,
    *,
    checkpoint_step: int,
    loader: LoaderConfig,
    lora: bool,
) -> int:
    if not isinstance(scalar, dict) and not hasattr(scalar, "items"):
        raise SMLArtifactError("checkpoint scalar current state must be a mapping")
    values = dict(scalar)
    step = values.get("step")
    microsteps = values.get("microsteps")
    if (
        step != checkpoint_step
        or isinstance(microsteps, bool)
        or not isinstance(microsteps, int)
    ):
        raise SMLArtifactError("checkpoint scalar progress disagrees with checkpoint")
    maximum_microsteps = step * loader.gradient_accumulation_steps
    if not step <= microsteps <= maximum_microsteps:
        raise SMLArtifactError("checkpoint scalar microsteps disagree with step")
    if lora:
        examples = values.get("examples")
        if (
            isinstance(examples, bool)
            or not isinstance(examples, int)
            or not microsteps <= examples <= microsteps * loader.microbatch_size
        ):
            raise SMLArtifactError("checkpoint scalar examples disagree with loader")
    elif values.get("rows") != microsteps * loader.microbatch_size:
        raise SMLArtifactError("checkpoint scalar rows disagree with loader")
    return microsteps


def _require_model_specs(
    reference: ArrayPayloadRef,
    model: ModelConfig,
    *,
    context: str,
) -> None:
    expected = model_parameter_specs(model)
    if reference.arrays != expected:
        raise SMLArtifactError(f"{context} model parameter specs do not match config")


def validate_base_semantics(manifest: BaseSnapshotManifest) -> ModelConfig:
    """Apply the exact value-independent base-snapshot semantic contract."""
    model = _model_configuration(manifest.model, context="base snapshot")
    _precision_configuration(
        manifest.precision,
        PrecisionConfig,
        context="base snapshot",
    )
    _require_model_specs(
        manifest.working_weights,
        model,
        context="base snapshot",
    )
    return model


def validate_export_semantics(
    manifest: ExportManifest,
    tokenizer: TokenizerManifest,
) -> ModelConfig:
    """Apply the exact value-independent merged-export semantic contract."""
    model = _model_configuration(manifest.model, context="export")
    _tokenizer_matches_model(tokenizer, model, context="export")
    _precision_configuration(
        manifest.precision,
        LoRAPrecisionConfig,
        context="export",
    )
    _require_model_specs(manifest.model_weights, model, context="export")
    return model


def validate_full_run_semantics(
    reader: CheckpointReader,
    tokenizer: TokenizerManifest,
) -> ModelConfig:
    """Validate one retained run/checkpoint owner without reopening its path."""
    run = reader.resolved.run
    checkpoint = reader.resolved.checkpoint
    model = _model_configuration(run.model, context="run")
    _optimizer_configuration(run.optimizer, context="run")
    loader = _loader_configuration(run.loader, context="run")
    checkpoint_config = checkpoint_configuration(run.checkpoint, context="run")
    _tokenizer_matches_model(tokenizer, model, context="run")
    contents = reader.read_contents()
    if isinstance(run, PretrainingRunManifest):
        if not isinstance(checkpoint, PretrainingCheckpointManifest):
            raise SMLArtifactError("pretraining run owns the wrong checkpoint kind")
        _precision_configuration(
            run.precision,
            PrecisionConfig,
            context="pretraining run",
        )
        expected_model = model_parameter_specs(model)
        expected_master = tuple(
            ArraySpec(spec.name, spec.shape, "float32") for spec in expected_model
        )
        if checkpoint.model.arrays != expected_model:
            raise SMLArtifactError(
                "pretraining checkpoint model parameter specs do not match config"
            )
        if checkpoint.master.arrays != expected_master:
            raise SMLArtifactError(
                "pretraining checkpoint master parameter specs do not match config"
            )
        microsteps = _verify_progress(
            contents.scalar_state,
            checkpoint_step=checkpoint.step,
            loader=loader,
            lora=False,
        )
        expected_key = expected_next_key(
            seed=checkpoint_config["seed"],
            microsteps=microsteps,
            model=model,
        )
    else:
        if not isinstance(run, LoRARunManifest) or not isinstance(
            checkpoint,
            LoRACheckpointManifest,
        ):
            raise SMLArtifactError("LoRA run owns the wrong checkpoint kind")
        try:
            lora = lora_config_from_mapping(run.lora)
            _exact_configuration(lora, dict(run.lora), context="LoRA configuration")
        except SMLArtifactError:
            raise
        except (KeyError, TypeError, ValueError, SMLConfigurationError) as error:
            raise SMLArtifactError("invalid LoRA configuration") from error
        _precision_configuration(
            run.precision,
            LoRAPrecisionConfig,
            context="LoRA run",
        )
        expected_adapters = lora_parameter_specs(model, lora)
        if checkpoint.adapters.arrays != expected_adapters:
            raise SMLArtifactError(
                "LoRA checkpoint adapter parameter specs do not match config"
            )
        microsteps = _verify_progress(
            contents.scalar_state,
            checkpoint_step=checkpoint.step,
            loader=loader,
            lora=True,
        )
        expected_key = expected_next_key(
            seed=checkpoint_config["seed"],
            microsteps=microsteps,
            model=model,
            lora=lora,
        )
    verify_checkpoint_current_state(reader, expected_next_key=expected_key)
    return model


__all__ = [
    "checkpoint_configuration",
    "expected_next_key",
    "validate_base_semantics",
    "validate_export_semantics",
    "validate_full_run_semantics",
]
