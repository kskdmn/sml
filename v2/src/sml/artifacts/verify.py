"""Typed recursive verification for public artifact roots."""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path

import mlx.core as mx

from sml.artifacts.arrays import load_safetensors_payload
from sml.artifacts.checkpoint import (
    CheckpointReader,
    open_latest_checkpoint_reader,
    require_lora_base_snapshot,
    verify_checkpoint_current_state,
)
from sml.artifacts.manifest import (
    CHECKPOINT_MANIFEST_TYPES,
    RUN_MANIFEST_TYPES,
    ArraySpec,
    ArtifactRoot,
    BaseSnapshotManifest,
    CheckpointManifest,
    ExportManifest,
    LoRACheckpointManifest,
    LoRARunManifest,
    OpenedArtifact,
    PayloadRef,
    PretrainingCheckpointManifest,
    PretrainingDataManifest,
    PretrainingRunManifest,
    RunManifest,
    SwagDataManifest,
    TokenizerManifest,
    VerificationLevel,
    _json_object_no_duplicates,
    _parse_manifest,
    _reject_json_constant,
    canonical_json_bytes,
)
from sml.data.pretraining import _verify_opened_pretraining_bundle
from sml.data.swag import _load_opened_swag_bundle
from sml.data.tokenizer import _load_opened_tokenizer_bundle
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

type ArtifactManifest = (
    TokenizerManifest
    | PretrainingDataManifest
    | CheckpointManifest
    | RunManifest
    | BaseSnapshotManifest
    | SwagDataManifest
    | ExportManifest
)

_BUNDLE_TYPES: tuple[type[ArtifactManifest], ...] = (
    TokenizerManifest,
    PretrainingDataManifest,
    BaseSnapshotManifest,
    SwagDataManifest,
    ExportManifest,
)
_MANIFEST_CANDIDATES: tuple[tuple[str, tuple[type[ArtifactManifest], ...]], ...] = (
    (PretrainingRunManifest.MANIFEST_FILENAME, RUN_MANIFEST_TYPES),
    (TokenizerManifest.MANIFEST_FILENAME, _BUNDLE_TYPES),
)


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """A verified artifact plus any independently verified child artifacts."""

    path: Path
    manifest: ArtifactManifest
    verification: VerificationLevel
    children: tuple[VerificationResult, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            raise TypeError("verified artifact path must be a Path")
        if not isinstance(
            self.manifest,
            (
                TokenizerManifest,
                PretrainingDataManifest,
                *CHECKPOINT_MANIFEST_TYPES,
                *RUN_MANIFEST_TYPES,
                BaseSnapshotManifest,
                SwagDataManifest,
                ExportManifest,
            ),
        ):
            raise TypeError("manifest must be a supported artifact manifest")
        if not isinstance(self.verification, VerificationLevel):
            raise TypeError("verification must be a VerificationLevel")
        if not isinstance(self.children, tuple) or not all(
            isinstance(child, VerificationResult) for child in self.children
        ):
            raise TypeError("children must be VerificationResult values")


def _missing_payload(error: SMLArtifactError) -> bool:
    cause: BaseException | None = error
    while cause is not None:
        if isinstance(cause, FileNotFoundError):
            return True
        cause = cause.__cause__
    return False


def _read_candidate_bytes(
    root: ArtifactRoot,
    filename: str,
) -> bytes | None:
    try:
        stream, opened_stat = root._open_payload_with_stat(filename)
    except SMLArtifactError as error:
        if _missing_payload(error):
            return None
        raise
    try:
        encoded = stream.read()
        consumed_stat = os.fstat(stream.fileno())
        opened = (
            opened_stat.st_dev,
            opened_stat.st_ino,
            opened_stat.st_size,
            opened_stat.st_mtime_ns,
            opened_stat.st_ctime_ns,
        )
        consumed = (
            consumed_stat.st_dev,
            consumed_stat.st_ino,
            consumed_stat.st_size,
            consumed_stat.st_mtime_ns,
            consumed_stat.st_ctime_ns,
        )
        if opened != consumed:
            raise SMLArtifactError(f"manifest changed during parsing: {filename}")
    except BaseException as error:
        try:
            stream.close()
        except BaseException as cleanup_error:
            raise error from cleanup_error
        raise
    else:
        stream.close()
        return encoded


def _open_artifact_once(
    path: Path,
    verification: VerificationLevel,
) -> OpenedArtifact[ArtifactManifest]:
    root = ArtifactRoot.open(path, writable=False)
    try:
        candidates: list[tuple[str, bytes, tuple[type[ArtifactManifest], ...]]] = []
        for filename, manifest_types in _MANIFEST_CANDIDATES:
            encoded = _read_candidate_bytes(root, filename)
            if encoded is not None:
                candidates.append((filename, encoded, manifest_types))
        if len(candidates) != 1:
            raise SMLArtifactError(
                "artifact root must contain exactly one of run.json or manifest.json"
            )
        filename, encoded, manifest_types = candidates[0]
        try:
            raw = json.loads(
                encoded.decode("utf-8"),
                parse_constant=_reject_json_constant,
                object_pairs_hook=_json_object_no_duplicates,
            )
            if not isinstance(raw, dict) or not isinstance(raw.get("kind"), str):
                raise SMLArtifactError(
                    "artifact manifest must contain a string kind discriminator"
                )
            types_by_kind = {
                manifest_type.EXPECTED_KIND: manifest_type
                for manifest_type in manifest_types
            }
            try:
                manifest_type = types_by_kind[raw["kind"]]
            except KeyError as error:
                raise SMLArtifactError(
                    f"unsupported artifact kind in {filename}: {raw['kind']!r}"
                ) from error
            manifest = _parse_manifest(raw, manifest_type)
        except SMLArtifactError:
            raise
        except (
            KeyError,
            TypeError,
            UnicodeError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            raise SMLArtifactError(
                f"invalid artifact manifest at {path}: {error}"
            ) from error
        if manifest.recompute_identity() != manifest.identity:
            raise SMLArtifactError("artifact manifest identity mismatch")
        if encoded != canonical_json_bytes(manifest):
            raise SMLArtifactError("artifact manifest must use canonical JSON bytes")
        return OpenedArtifact(
            path=path,
            root=root,
            manifest=manifest,
            verification=verification,
        )
    except BaseException as error:
        try:
            root.close()
        except BaseException as cleanup_error:
            raise error from cleanup_error
        raise


def _verify_payload(artifact: OpenedArtifact, reference: PayloadRef) -> None:
    with artifact.open_payload(reference):
        pass


def _verify_structural_payloads(
    artifact: OpenedArtifact,
    references: tuple[PayloadRef, ...],
) -> None:
    for reference in references:
        _verify_payload(artifact, reference)


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
            precision, dict(projection), context=f"{context} precision"
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


def _checkpoint_configuration(projection: object, *, context: str) -> dict[str, object]:
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
    }
    if set(values) != expected:
        raise SMLArtifactError(f"{context} checkpoint config has invalid fields")
    try:
        CheckpointPolicy(interval=values["interval"])
        for name in ("maximum_steps", "maximum_epochs"):
            value = values[name]
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ValueError(f"{name} must be positive or None")
        if values["maximum_steps"] is None and values["maximum_epochs"] is None:
            raise ValueError("one training termination limit is required")
        log_interval = values["log_interval"]
        if (
            isinstance(log_interval, bool)
            or not isinstance(log_interval, int)
            or log_interval < 1
        ):
            raise ValueError("log_interval must be positive")
        seed = values["seed"]
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or not 0 <= seed <= 2**32 - 1
        ):
            raise ValueError("seed must be uint32")
        if not isinstance(values["compile"], bool):
            raise TypeError("compile must be bool")
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


def _expected_next_key(
    *,
    seed: int,
    microsteps: int,
    model: ModelConfig,
    lora: LoRAConfig | None = None,
) -> mx.array:
    splits_per_microstep = model.num_layers if model.hidden_dropout > 0.0 else 0
    if lora is not None and lora.dropout > 0.0:
        splits_per_microstep += model.num_layers * len(lora.target_modules)
    advances = microsteps * splits_per_microstep
    if advances > 1_000_000:
        raise SMLArtifactError("checkpoint RNG verification exceeds its bounded limit")
    _model_key, key = mx.random.split(mx.random.key(seed))
    for _ in range(advances):
        key, _used_key = mx.random.split(key)
    mx.eval(key)
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
    if microsteps != step * loader.gradient_accumulation_steps:
        raise SMLArtifactError("checkpoint scalar microsteps disagree with step")
    if lora:
        examples = values.get("examples")
        if (
            isinstance(examples, bool)
            or not isinstance(examples, int)
            or not microsteps <= examples <= microsteps * loader.microbatch_size
        ):
            raise SMLArtifactError("checkpoint scalar examples disagree with loader")
    else:
        if values.get("rows") != microsteps * loader.microbatch_size:
            raise SMLArtifactError("checkpoint scalar rows disagree with loader")
    return microsteps


def _require_model_specs(reference, model: ModelConfig, *, context: str) -> None:
    expected = model_parameter_specs(model)
    if reference.arrays != expected:
        raise SMLArtifactError(f"{context} model parameter specs do not match config")


def _tokenizer_binding(
    outer: PretrainingDataManifest | ExportManifest,
    tokenizer: TokenizerManifest,
) -> None:
    if tokenizer.identity != outer.tokenizer_identity:
        raise SMLArtifactError("owned tokenizer identity does not match its manifest")
    expected_model = replace(
        tokenizer.model, logical_path=f"tokenizer/{tokenizer.model.logical_path}"
    )
    expected_vocab = replace(
        tokenizer.vocab, logical_path=f"tokenizer/{tokenizer.vocab.logical_path}"
    )
    if (
        tokenizer.model.logical_path != "tokenizer.model"
        or tokenizer.vocab.logical_path != "tokenizer.vocab"
        or outer.tokenizer_model != expected_model
        or outer.tokenizer_vocab != expected_vocab
    ):
        raise SMLArtifactError(
            "owned tokenizer payload references do not match the nested manifest"
        )


def _verify_opened_tokenizer(
    artifact: OpenedArtifact[TokenizerManifest],
) -> VerificationResult:
    if artifact.verification is VerificationLevel.FULL:
        _load_opened_tokenizer_bundle(artifact)
    else:
        _verify_structural_payloads(
            artifact,
            (artifact.manifest.model, artifact.manifest.vocab),
        )
    return VerificationResult(
        artifact.path,
        artifact.manifest,
        artifact.verification,
    )


def _verify_tokenizer_child(
    artifact: OpenedArtifact,
) -> VerificationResult:
    with artifact.open_child("tokenizer", (TokenizerManifest,)) as tokenizer:
        return _verify_opened_tokenizer(tokenizer)


def _verify_opened_pretraining_data(
    artifact: OpenedArtifact[PretrainingDataManifest],
) -> VerificationResult:
    tokenizer = _verify_tokenizer_child(artifact)
    if not isinstance(tokenizer.manifest, TokenizerManifest):
        raise SMLArtifactError("prepared-data tokenizer has the wrong artifact kind")
    _tokenizer_binding(artifact.manifest, tokenizer.manifest)
    if artifact.verification is VerificationLevel.FULL:
        _verify_opened_pretraining_bundle(
            artifact,
            tokenizer.manifest,
            batch_size=1,
        )
    else:
        _verify_structural_payloads(artifact, artifact.manifest.shards)
    return VerificationResult(
        artifact.path,
        artifact.manifest,
        artifact.verification,
        (tokenizer,),
    )


def _verify_opened_base(
    artifact: OpenedArtifact[BaseSnapshotManifest],
) -> VerificationResult:
    manifest = artifact.manifest
    if artifact.verification is VerificationLevel.FULL:
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
        load_safetensors_payload(artifact, manifest.working_weights)
    else:
        _verify_payload(artifact, manifest.working_weights.payload)
    return VerificationResult(
        artifact.path,
        manifest,
        artifact.verification,
    )


def _verify_opened_swag(
    artifact: OpenedArtifact[SwagDataManifest],
) -> VerificationResult:
    if artifact.verification is VerificationLevel.FULL:
        with _load_opened_swag_bundle(
            artifact,
            validate_projections=True,
        ) as bundle:
            return VerificationResult(
                bundle.path,
                bundle.manifest,
                bundle.verification,
            )
    _verify_structural_payloads(
        artifact,
        tuple(bucket.payload for bucket in artifact.manifest.buckets),
    )
    return VerificationResult(artifact.path, artifact.manifest, artifact.verification)


def _verify_opened_export(
    artifact: OpenedArtifact[ExportManifest],
) -> VerificationResult:
    tokenizer = _verify_tokenizer_child(artifact)
    if not isinstance(tokenizer.manifest, TokenizerManifest):
        raise SMLArtifactError("export tokenizer has the wrong artifact kind")
    _tokenizer_binding(artifact.manifest, tokenizer.manifest)
    manifest = artifact.manifest
    if artifact.verification is VerificationLevel.FULL:
        model = _model_configuration(manifest.model, context="export")
        _precision_configuration(
            manifest.precision,
            LoRAPrecisionConfig,
            context="export",
        )
        _require_model_specs(manifest.model_weights, model, context="export")
        load_safetensors_payload(artifact, manifest.model_weights)
    else:
        _verify_payload(artifact, manifest.model_weights.payload)
    return VerificationResult(
        artifact.path,
        manifest,
        artifact.verification,
        (tokenizer,),
    )


def _verify_opened_bundle(artifact: OpenedArtifact) -> VerificationResult:
    manifest = artifact.manifest
    if isinstance(manifest, TokenizerManifest):
        return _verify_opened_tokenizer(artifact)
    if isinstance(manifest, PretrainingDataManifest):
        return _verify_opened_pretraining_data(artifact)
    if isinstance(manifest, BaseSnapshotManifest):
        return _verify_opened_base(artifact)
    if isinstance(manifest, SwagDataManifest):
        return _verify_opened_swag(artifact)
    if isinstance(manifest, ExportManifest):
        return _verify_opened_export(artifact)
    raise SMLArtifactError(f"unsupported portable artifact kind: {manifest.kind!r}")


def _verify_full_run_semantics(
    reader: CheckpointReader,
    tokenizer: TokenizerManifest,
) -> None:
    run = reader.resolved.run
    checkpoint = reader.resolved.checkpoint
    model = _model_configuration(run.model, context="run")
    _optimizer_configuration(run.optimizer, context="run")
    loader = _loader_configuration(run.loader, context="run")
    checkpoint_config = _checkpoint_configuration(run.checkpoint, context="run")
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
        expected_key = _expected_next_key(
            seed=checkpoint_config["seed"],
            microsteps=microsteps,
            model=model,
        )
    else:
        if not isinstance(run, LoRARunManifest) or not isinstance(
            checkpoint, LoRACheckpointManifest
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
        expected_key = _expected_next_key(
            seed=checkpoint_config["seed"],
            microsteps=microsteps,
            model=model,
            lora=lora,
        )
    verify_checkpoint_current_state(reader, expected_next_key=expected_key)


def _verify_opened_run(artifact: OpenedArtifact[RunManifest]) -> VerificationResult:
    level = artifact.verification
    with open_latest_checkpoint_reader(
        artifact.path,
        verification=level,
        run_descriptor=artifact.root.fileno(),
    ) as reader:
        resolved = reader.resolved
        if resolved.run != artifact.manifest:
            raise SMLArtifactError("run manifest changed during recursive verification")
        if resolved.latest_recovered:
            raise SMLArtifactError(
                "run latest index must directly bind the latest checkpoint"
            )
        with reader.open_run_child("tokenizer", (TokenizerManifest,)) as opened:
            tokenizer = _verify_opened_tokenizer(opened)
        if tokenizer.manifest.identity != resolved.run.tokenizer_identity:
            raise SMLArtifactError("run tokenizer identity does not match run.json")
        children: list[VerificationResult] = [tokenizer]
        if isinstance(resolved.run, LoRARunManifest):
            with reader.open_run_child("base", (BaseSnapshotManifest,)) as opened:
                base = _verify_opened_base(opened)
            if base.manifest.identity != resolved.run.base_identity:
                raise SMLArtifactError(
                    "run base snapshot identity does not match run.json"
                )
            if base.manifest.tokenizer_identity != resolved.run.tokenizer_identity:
                raise SMLArtifactError(
                    "run base tokenizer identity does not match run.json"
                )
            if level is VerificationLevel.FULL:
                require_lora_base_snapshot(base.manifest, resolved.run)
            children.append(base)
        if level is VerificationLevel.FULL:
            _verify_full_run_semantics(reader, tokenizer.manifest)
        checkpoint = VerificationResult(
            resolved.step_directory,
            resolved.checkpoint,
            level,
        )
        children.append(checkpoint)
        result = VerificationResult(
            artifact.path,
            resolved.run,
            level,
            tuple(children),
        )
    return result


def verify_artifact(path: Path, full: bool) -> VerificationResult:
    """Verify a supported artifact root and its owned child artifacts."""
    if not isinstance(path, Path):
        raise TypeError("path must be a Path")
    if not isinstance(full, bool):
        raise TypeError("full must be a bool")
    level = VerificationLevel.FULL if full else VerificationLevel.MANIFEST_TRUSTED
    with _open_artifact_once(path, level) as artifact:
        if isinstance(artifact.manifest, (PretrainingRunManifest, LoRARunManifest)):
            return _verify_opened_run(artifact)
        return _verify_opened_bundle(artifact)


__all__ = ["ArtifactManifest", "VerificationResult", "verify_artifact"]
