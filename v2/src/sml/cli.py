from __future__ import annotations

import argparse
import sys
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar

from sml.errors import (
    SMLArtifactError,
    SMLConfigurationError,
    SMLDataError,
    SMLRuntimeError,
)

_DOMAIN_ERRORS = (
    SMLConfigurationError,
    SMLArtifactError,
    SMLDataError,
    SMLRuntimeError,
)
_EXIT_CODES = {
    SMLConfigurationError: 2,
    SMLArtifactError: 3,
    SMLDataError: 4,
    SMLRuntimeError: 5,
}
_COMMANDS = (
    "tokenize",
    "prepare",
    "train",
    "infer",
    "evaluate",
    "finetune",
    "export",
    "verify",
)
_TRAIN_FLAT = {
    "microbatch_size",
    "gradient_accumulation_steps",
    "prefetch_depth",
    "learning_rate",
    "checkpoint_interval",
    "maximum_steps",
    "maximum_epochs",
    "log_interval",
}
_COMMON_TRAIN_FIELDS = {
    "data",
    "output",
    "resume",
    "seed",
    "compile",
    *_TRAIN_FLAT,
}
_SCHEMAS: dict[str, dict[str, object]] = {
    "tokenize": {
        "input": None,
        "output": None,
        "algorithm": None,
        "vocab_size": None,
        "character_coverage": None,
        "byte_fallback": None,
        "normalization_rule_name": None,
        "num_threads": None,
        "input_sentence_size": None,
        "shuffle_input_sentence": None,
        "self_test_sample_size": None,
        "hard_vocab_limit": None,
        "train_extremely_large_corpus": None,
        "maximum_sentence_length": None,
        "conversation_user_symbols": None,
        "unk_id": None,
        "bos_id": None,
        "eos_id": None,
        "pad_id": None,
        "corpus": {
            "filename_pattern": None,
            "shuffle_files": None,
            "file_order_seed": None,
            "text_field": None,
            "min_text_bytes": None,
            "max_text_bytes": None,
            "max_rows_per_file": None,
        },
    },
    "prepare.pretraining": {
        "input": None,
        "tokenizer": None,
        "output": None,
        "sequence_length": None,
        "shuffle_window_rows": None,
        "shuffle_algorithm": None,
        "output_shard_rows": None,
        "seed": None,
        "corpus": {
            "filename_pattern": None,
            "shuffle_files": None,
            "file_order_seed": None,
            "text_field": None,
            "min_text_bytes": None,
            "max_text_bytes": None,
            "max_rows_per_file": None,
        },
    },
    "prepare.swag": {
        "checkpoint": None,
        "revision": None,
        "output": None,
        "preprocessing_schema_version": None,
        "join_policy": None,
        "overlength_policy": None,
        "bos_policy": None,
        "eos_policy": None,
        "maximum_length": None,
        "bucket_boundaries": None,
        "maximum_examples": None,
        "source": {
            "backend": None,
            "namespace": None,
            "name": None,
            "dataset_config": None,
            "split": None,
        },
    },
    "train": {
        **{name: None for name in _COMMON_TRAIN_FIELDS},
        "model": {
            "vocab_size": None,
            "hidden_size": None,
            "num_layers": None,
            "num_q_heads": None,
            "num_kv_heads": None,
            "intermediate_size": None,
            "original_context_length": None,
            "rope_theta": None,
            "rope_scaling_factor": None,
            "yarn_beta_fast": None,
            "yarn_beta_slow": None,
            "yarn_attention_factor": None,
            "yarn_mscale": None,
            "yarn_mscale_all_dim": None,
            "yarn_truncate": None,
            "rms_norm_epsilon": None,
            "hidden_dropout": None,
            "initializer_range": None,
            "initializers": {
                "embed_tokens": None,
                "lm_head": None,
                "q_proj": None,
                "k_proj": None,
                "v_proj": None,
                "o_proj": None,
                "gate_proj": None,
                "up_proj": None,
                "down_proj": None,
                "other": None,
            },
            "pad_token_id": None,
            "bos_token_id": None,
            "eos_token_id": None,
            "unk_token_id": None,
            "tie_word_embeddings": None,
            "use_cache": None,
        },
        "optimizer": {
            "beta1": None,
            "beta2": None,
            "epsilon": None,
            "bias_correction": None,
            "schedule_steps": None,
            "warmup_steps": None,
            "minimum_learning_rate_ratio": None,
            "gradient_clip_norm": None,
            "weight_decay": {
                "embed_tokens": None,
                "lm_head": None,
                "rms_norm": None,
                "q_proj": None,
                "k_proj": None,
                "v_proj": None,
                "o_proj": None,
                "gate_proj": None,
                "up_proj": None,
                "down_proj": None,
                "lora_a": None,
                "lora_b": None,
                "other": None,
            },
        },
        "precision": {
            "master_parameter_dtype": None,
            "working_parameter_dtype": None,
            "gradient_accumulator_dtype": None,
            "optimizer_state_dtype": None,
            "update_dtype": None,
            "master_weights": None,
            "dynamic_loss_scaling": None,
        },
    },
    "infer": {
        "checkpoint": None,
        "prompt": None,
        "max_new_tokens": None,
        "include_prompt": None,
        "full": None,
        "request": {
            "max_new_tokens": None,
            "include_prompt": None,
            "config": {
                "temperature": None,
                "top_p": None,
                "repetition_penalty": None,
                "no_repeat_ngram_size": None,
                "seed": None,
            },
        },
        "runtime": {
            "batch_size_buckets": None,
            "decode_chunk_size": None,
        },
    },
    "evaluate": {
        "checkpoint": None,
        "tasks": None,
        "output": None,
        "full": None,
        "padding": None,
        "limit": None,
        "runtime": {
            "batch_size_buckets": None,
            "decode_chunk_size": None,
        },
    },
    "finetune": {
        **{name: None for name in _COMMON_TRAIN_FIELDS},
        "checkpoint": None,
        "lora": {
            "rank": None,
            "alpha": None,
            "scaling_mode": None,
            "dropout": None,
            "target_modules": None,
            "initializer": {"lora_a": None, "lora_b": None},
        },
        "optimizer": {
            "beta1": None,
            "beta2": None,
            "epsilon": None,
            "bias_correction": None,
            "schedule_steps": None,
            "warmup_steps": None,
            "minimum_learning_rate_ratio": None,
            "gradient_clip_norm": None,
            "weight_decay": {
                "embed_tokens": None,
                "lm_head": None,
                "rms_norm": None,
                "q_proj": None,
                "k_proj": None,
                "v_proj": None,
                "o_proj": None,
                "gate_proj": None,
                "up_proj": None,
                "down_proj": None,
                "lora_a": None,
                "lora_b": None,
                "other": None,
            },
        },
        "precision": {
            "frozen_base_dtype": None,
            "adapter_parameter_dtype": None,
            "gradient_accumulator_dtype": None,
            "optimizer_state_dtype": None,
            "update_dtype": None,
            "dynamic_loss_scaling": None,
        },
    },
    "export": {"checkpoint": None, "output": None},
    "verify": {"path": None, "full": None},
}


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise SMLConfigurationError(message)


def _add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=argparse.SUPPRESS)


def _add_bool(parser: argparse.ArgumentParser, name: str) -> None:
    parser.add_argument(
        f"--{name.replace('_', '-')}",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
    )


def _add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", type=Path, default=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, default=argparse.SUPPRESS)
    parser.add_argument("--resume", type=Path, default=argparse.SUPPRESS)
    for name in (
        "microbatch_size",
        "gradient_accumulation_steps",
        "prefetch_depth",
        "checkpoint_interval",
        "maximum_steps",
        "maximum_epochs",
        "log_interval",
        "seed",
    ):
        parser.add_argument(
            f"--{name.replace('_', '-')}",
            type=int,
            default=argparse.SUPPRESS,
        )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=argparse.SUPPRESS,
    )
    _add_bool(parser, "compile")
    _add_config(parser)


def _build_parser() -> argparse.ArgumentParser:
    parser = _Parser(description="SML workflows")
    subparsers = parser.add_subparsers(dest="_command")

    tokenize = subparsers.add_parser("tokenize")
    tokenize.add_argument("--input", type=Path, default=argparse.SUPPRESS)
    tokenize.add_argument("--output", type=Path, default=argparse.SUPPRESS)
    for name in (
        "vocab_size",
        "num_threads",
        "input_sentence_size",
        "self_test_sample_size",
        "maximum_sentence_length",
    ):
        tokenize.add_argument(
            f"--{name.replace('_', '-')}",
            type=int,
            default=argparse.SUPPRESS,
        )
    tokenize.add_argument(
        "--character-coverage",
        type=float,
        default=argparse.SUPPRESS,
    )
    _add_config(tokenize)

    prepare = subparsers.add_parser("prepare")
    prepare_subparsers = prepare.add_subparsers(dest="_prepare_command")
    pretraining = prepare_subparsers.add_parser("pretraining")
    pretraining.add_argument("--input", type=Path, default=argparse.SUPPRESS)
    pretraining.add_argument("--tokenizer", type=Path, default=argparse.SUPPRESS)
    pretraining.add_argument("--output", type=Path, default=argparse.SUPPRESS)
    for name in (
        "sequence_length",
        "shuffle_window_rows",
        "output_shard_rows",
        "seed",
    ):
        pretraining.add_argument(
            f"--{name.replace('_', '-')}",
            type=int,
            default=argparse.SUPPRESS,
        )
    _add_config(pretraining)

    swag = prepare_subparsers.add_parser("swag")
    swag.add_argument("--checkpoint", type=Path, default=argparse.SUPPRESS)
    swag.add_argument("--revision", default=argparse.SUPPRESS)
    swag.add_argument("--output", type=Path, default=argparse.SUPPRESS)
    swag.add_argument("--maximum-length", type=int, default=argparse.SUPPRESS)
    swag.add_argument("--maximum-examples", type=int, default=argparse.SUPPRESS)
    _add_config(swag)

    train_parser = subparsers.add_parser("train")
    _add_training_arguments(train_parser)

    infer_parser = subparsers.add_parser("infer")
    infer_parser.add_argument("--checkpoint", type=Path, default=argparse.SUPPRESS)
    infer_parser.add_argument("prompt", nargs="?", default=argparse.SUPPRESS)
    infer_parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=argparse.SUPPRESS,
    )
    _add_bool(infer_parser, "include_prompt")
    infer_parser.add_argument(
        "--temperature",
        type=float,
        default=argparse.SUPPRESS,
    )
    infer_parser.add_argument("--top-p", type=float, default=argparse.SUPPRESS)
    infer_parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=argparse.SUPPRESS,
    )
    infer_parser.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=argparse.SUPPRESS,
    )
    infer_parser.add_argument("--seed", type=int, default=argparse.SUPPRESS)
    _add_bool(infer_parser, "full")
    _add_config(infer_parser)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument(
        "--checkpoint",
        type=Path,
        default=argparse.SUPPRESS,
    )
    evaluate_parser.add_argument(
        "--task",
        dest="tasks",
        action="append",
        choices=("hellaswag", "winogrande"),
        default=argparse.SUPPRESS,
    )
    evaluate_parser.add_argument("--output", type=Path, default=argparse.SUPPRESS)
    evaluate_parser.add_argument(
        "--padding",
        choices=("left", "right"),
        default=argparse.SUPPRESS,
    )
    evaluate_parser.add_argument("--limit", type=int, default=argparse.SUPPRESS)
    _add_bool(evaluate_parser, "full")
    _add_config(evaluate_parser)

    finetune_parser = subparsers.add_parser("finetune")
    finetune_parser.add_argument(
        "--checkpoint",
        type=Path,
        default=argparse.SUPPRESS,
    )
    _add_training_arguments(finetune_parser)

    export_parser = subparsers.add_parser("export")
    export_parser.add_argument(
        "--checkpoint",
        type=Path,
        default=argparse.SUPPRESS,
    )
    export_parser.add_argument("--output", type=Path, default=argparse.SUPPRESS)
    _add_config(export_parser)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("path", nargs="?", type=Path, default=argparse.SUPPRESS)
    _add_bool(verify_parser, "full")
    _add_config(verify_parser)
    return parser


def _command_key(values: Mapping[str, object]) -> str:
    command = values.get("_command")
    if not isinstance(command, str):
        raise SMLConfigurationError("a command is required")
    if command != "prepare":
        return command
    prepare_command = values.get("_prepare_command")
    if not isinstance(prepare_command, str):
        raise SMLConfigurationError("a prepare command is required")
    return f"prepare.{prepare_command}"


def _load_command_table(path: Path, command: str) -> dict[str, object]:
    try:
        with path.open("rb") as config_file:
            document = tomllib.load(config_file)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise SMLConfigurationError(f"invalid config file {path}: {error}") from error

    root = command.split(".")
    if set(document) != {root[0]}:
        raise SMLConfigurationError(
            f"config must contain exactly the [{command}] table"
        )
    table: object = document[root[0]]
    if len(root) == 2:
        if not isinstance(table, dict) or set(table) != {root[1]}:
            raise SMLConfigurationError(
                f"config must contain exactly the [{command}] table"
            )
        table = table[root[1]]
    if not isinstance(table, dict):
        raise SMLConfigurationError(f"[{command}] must be a table")
    _validate_table(table, _SCHEMAS[command], command)
    return dict(table)


def _validate_table(
    table: Mapping[str, object],
    schema: Mapping[str, object],
    location: str,
) -> None:
    unknown = set(table) - set(schema)
    if unknown:
        names = ", ".join(sorted(unknown))
        raise SMLConfigurationError(f"unknown fields in [{location}]: {names}")
    for name, value in table.items():
        nested_schema = schema[name]
        if nested_schema is None:
            if isinstance(value, dict):
                raise SMLConfigurationError(f"{location}.{name} must not be a table")
            continue
        if not isinstance(value, dict):
            raise SMLConfigurationError(f"{location}.{name} must be a table")
        _validate_table(value, nested_schema, f"{location}.{name}")


def _normalize_cli_values(
    namespace: argparse.Namespace,
) -> tuple[str, dict[str, object]]:
    values = vars(namespace).copy()
    command = _command_key(values)
    values.pop("_command", None)
    values.pop("_prepare_command", None)
    config = values.pop("config", None)
    configured = (
        _load_command_table(config, command) if isinstance(config, Path) else {}
    )
    configured.update(values)
    path_fields = {
        "tokenize": ("input", "output"),
        "prepare.pretraining": ("input", "tokenizer", "output"),
        "prepare.swag": ("checkpoint", "output"),
        "train": ("data", "output", "resume"),
        "infer": ("checkpoint",),
        "evaluate": ("checkpoint", "output"),
        "finetune": ("checkpoint", "data", "output", "resume"),
        "export": ("checkpoint", "output"),
        "verify": ("path",),
    }[command]
    for name in path_fields:
        if isinstance(configured.get(name), str):
            configured[name] = Path(configured[name])
    return command, configured


def _require(values: Mapping[str, object], *names: str) -> None:
    missing = [name for name in names if name not in values]
    if missing:
        raise SMLConfigurationError(f"{', '.join(missing)} is required")


def _reject_step_path(value: object, field_name: str) -> None:
    if not isinstance(value, Path):
        return
    if any(part.startswith("step-") for part in value.parts):
        raise SMLConfigurationError(
            f"{field_name} must identify an artifact, not a step-* path"
        )


def _reject_existing_output(values: Mapping[str, object]) -> None:
    if "resume" in values:
        return
    output = values.get("output")
    if isinstance(output, Path) and output.exists():
        raise SMLConfigurationError(f"output already exists: {output}")


def _tuple_values(values: dict[str, object], *names: str) -> dict[str, object]:
    converted = values.copy()
    for name in names:
        value = converted.get(name)
        if isinstance(value, list):
            converted[name] = tuple(value)
    return converted


def _replace_dataclass(instance: Any, values: Mapping[str, object]) -> Any:
    if not values:
        return instance
    allowed = {item.name for item in fields(instance)}
    unknown = set(values) - allowed
    if unknown:
        raise SMLConfigurationError(
            f"unknown configuration fields: {', '.join(sorted(unknown))}"
        )
    return replace(instance, **values)


def _build_corpus(input_root: Path, values: object) -> Any:
    from sml.data.corpus import CorpusConfig

    nested = dict(values) if isinstance(values, dict) else {}
    return CorpusConfig(input_root=input_root, **nested)


def _build_optimizer(values: Mapping[str, object], *, finetune: bool) -> Any:
    from sml.training.common import OptimizerConfig, WeightDecayPolicy

    if finetune:
        from sml.training.swag import default_swag_optimizer_config

        optimizer = default_swag_optimizer_config()
    else:
        optimizer = OptimizerConfig()
    nested = dict(values.get("optimizer", {}))
    weight_decay = nested.pop("weight_decay", {})
    if weight_decay:
        optimizer = replace(
            optimizer,
            weight_decay=_replace_dataclass(
                WeightDecayPolicy(),
                weight_decay,
            ),
        )
    if "learning_rate" in values:
        nested["learning_rate"] = values["learning_rate"]
    return _replace_dataclass(optimizer, nested)


def _build_loader(values: Mapping[str, object]) -> Any:
    from sml.training.common import LoaderConfig

    selected = {
        name: values[name]
        for name in (
            "microbatch_size",
            "gradient_accumulation_steps",
            "prefetch_depth",
        )
        if name in values
    }
    return _replace_dataclass(LoaderConfig(), selected)


def _build_checkpoint(values: Mapping[str, object], *, finetune: bool) -> Any:
    from sml.training.common import CheckpointPolicy

    default_interval = 500 if finetune else 1_000
    interval = values.get("checkpoint_interval", default_interval)
    return CheckpointPolicy(interval=interval)


def _resume_overrides(values: Mapping[str, object]) -> Any:
    from sml.training.common import ResumeOverrides

    return ResumeOverrides(
        **{
            name: values[name]
            for name in (
                "maximum_steps",
                "maximum_epochs",
                "log_interval",
                "checkpoint_interval",
            )
            if name in values
        }
    )


@dataclass(frozen=True, slots=True)
class _Command:
    values: Mapping[str, object]
    command: ClassVar[str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))

    def to_domain(self) -> Any:
        raise NotImplementedError

    def dispatch(self) -> object:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class TokenizeCommand(_Command):
    command: ClassVar[str] = "tokenize"

    def to_domain(self) -> Any:
        from sml.data.tokenizer import TokenizerTrainingConfig

        values = dict(self.values)
        input_root = values.pop("input")
        values.pop("output")
        corpus = _build_corpus(input_root, values.pop("corpus", {}))
        values = _tuple_values(values, "conversation_user_symbols")
        return TokenizerTrainingConfig(corpus=corpus, **values)

    def dispatch(self) -> object:
        from sml.data.tokenizer import train_tokenizer_bundle

        return train_tokenizer_bundle(self.to_domain(), self.values["output"])


@dataclass(frozen=True, slots=True)
class PreparePretrainingCommand(_Command):
    command: ClassVar[str] = "prepare.pretraining"

    def to_domain(self) -> Any:
        from sml.data.pretraining import PretrainingPreparationConfig

        values = dict(self.values)
        input_root = values.pop("input")
        tokenizer = values.pop("tokenizer")
        values.pop("output")
        corpus = _build_corpus(input_root, values.pop("corpus", {}))
        return PretrainingPreparationConfig(
            corpus=corpus,
            tokenizer_bundle=tokenizer,
            **values,
        )

    def dispatch(self) -> object:
        from sml.data.pretraining import prepare_pretraining_bundle

        return prepare_pretraining_bundle(self.to_domain(), self.values["output"])


@dataclass(frozen=True, slots=True)
class PrepareSwagCommand(_Command):
    command: ClassVar[str] = "prepare.swag"

    def to_domain(self) -> Any:
        from sml.data.swag import (
            HuggingFaceDatasetsSwagProvider,
            SwagPreparationConfig,
            SwagSourceConfig,
        )

        values = dict(self.values)
        values.pop("checkpoint")
        values.pop("output")
        revision = values.pop("revision")
        source_values = dict(values.pop("source", {}))
        source = SwagSourceConfig(revision=revision, **source_values)
        values = _tuple_values(values, "bucket_boundaries")
        return SwagPreparationConfig(
            provider=HuggingFaceDatasetsSwagProvider(),
            source=source,
            **values,
        )

    def dispatch(self) -> object:
        from sml.data.swag import prepare_swag_bundle
        from sml.inference import resolve_model_artifact

        base = resolve_model_artifact(
            self.values["checkpoint"],
            full_verify=True,
        )
        return prepare_swag_bundle(
            self.to_domain(),
            base,
            self.values["output"],
        )


@dataclass(frozen=True, slots=True)
class TrainCommand(_Command):
    command: ClassVar[str] = "train"

    def to_domain(self) -> Any:
        if "resume" in self.values:
            return _resume_overrides(self.values)
        from sml.model.config import InitializerConfig, ModelConfig
        from sml.training.common import PrecisionConfig, PretrainingConfig

        model_values = dict(self.values.get("model", {}))
        initializer_values = model_values.pop("initializers", {})
        model = _replace_dataclass(ModelConfig(), model_values)
        if initializer_values:
            model = replace(
                model,
                initializers=_replace_dataclass(
                    InitializerConfig.depth_scaled(
                        model.initializer_range,
                        model.num_layers,
                    ),
                    initializer_values,
                ),
            )
        if model.rope_scaling_factor != 1.0:
            raise SMLConfigurationError(
                "pretraining model rope_scaling_factor must be exactly 1.0"
            )
        precision = _replace_dataclass(
            PrecisionConfig(),
            self.values.get("precision", {}),
        )
        scalar_values = {
            name: self.values[name]
            for name in (
                "maximum_steps",
                "maximum_epochs",
                "log_interval",
                "seed",
                "compile",
            )
            if name in self.values
        }
        return PretrainingConfig(
            data=self.values["data"],
            output_run=self.values["output"],
            model=model,
            optimizer=_build_optimizer(self.values, finetune=False),
            loader=_build_loader(self.values),
            checkpoint=_build_checkpoint(self.values, finetune=False),
            precision=precision,
            **scalar_values,
        )

    def dispatch(self) -> object:
        if "resume" in self.values:
            from sml.training.pretrain import resume

            return resume(
                self.values["resume"],
                data=self.values.get("data"),
                overrides=self.to_domain(),
            )
        from sml.training.pretrain import train

        return train(self.to_domain())


@dataclass(frozen=True, slots=True)
class InferCommand(_Command):
    command: ClassVar[str] = "infer"

    def to_domain(self) -> Any:
        from sml.inference import (
            GenerationRequest,
            InferenceConfig,
            InferenceRuntimeConfig,
        )
        from sml.model.config import GenerationConfig

        request_values = dict(self.values.get("request", {}))
        generation = _replace_dataclass(
            GenerationConfig(),
            request_values.pop("config", {}),
        )
        generation_cli = {
            name: self.values[name]
            for name in (
                "temperature",
                "top_p",
                "repetition_penalty",
                "no_repeat_ngram_size",
                "seed",
            )
            if name in self.values
        }
        generation = _replace_dataclass(generation, generation_cli)
        runtime_values = _tuple_values(
            dict(self.values.get("runtime", {})),
            "batch_size_buckets",
        )
        runtime = _replace_dataclass(InferenceRuntimeConfig(), runtime_values)
        request = GenerationRequest(
            max_new_tokens=self.values.get(
                "max_new_tokens",
                request_values.get("max_new_tokens", 128),
            ),
            config=generation,
            include_prompt=self.values.get(
                "include_prompt",
                request_values.get("include_prompt", False),
            ),
        )
        return InferenceConfig(
            checkpoint=self.values["checkpoint"],
            prompt=self.values["prompt"],
            request=request,
            full_verify=self.values.get("full", False),
            runtime=runtime,
        )

    def dispatch(self) -> object:
        from sml.inference import infer

        return infer(self.to_domain())


@dataclass(frozen=True, slots=True)
class EvaluateCommand(_Command):
    command: ClassVar[str] = "evaluate"

    def to_domain(self) -> Any:
        from sml.evaluation import EvaluationConfig
        from sml.inference import InferenceRuntimeConfig

        runtime_values = _tuple_values(
            dict(self.values.get("runtime", {})),
            "batch_size_buckets",
        )
        runtime = _replace_dataclass(InferenceRuntimeConfig(), runtime_values)
        values = {
            name: self.values[name]
            for name in ("padding", "limit")
            if name in self.values
        }
        tasks = self.values["tasks"]
        return EvaluationConfig(
            checkpoint=self.values["checkpoint"],
            tasks=tuple(tasks),
            output=self.values["output"],
            full_verify=self.values.get("full", False),
            runtime=runtime,
            **values,
        )

    def dispatch(self) -> object:
        from sml.evaluation import evaluate

        return evaluate(self.to_domain())


@dataclass(frozen=True, slots=True)
class FinetuneCommand(_Command):
    command: ClassVar[str] = "finetune"

    def to_domain(self) -> Any:
        if "resume" in self.values:
            return _resume_overrides(self.values)
        from sml.training.lora import (
            LoRAConfig,
            LoRAInitializerConfig,
            LoRAPrecisionConfig,
        )
        from sml.training.swag import SwagTrainingConfig

        lora_values = dict(self.values.get("lora", {}))
        initializer_values = lora_values.pop("initializer", {})
        if initializer_values:
            lora_values["initializer"] = _replace_dataclass(
                LoRAInitializerConfig(),
                initializer_values,
            )
        lora_values = _tuple_values(lora_values, "target_modules")
        lora = _replace_dataclass(LoRAConfig(), lora_values)
        precision = _replace_dataclass(
            LoRAPrecisionConfig(),
            self.values.get("precision", {}),
        )
        scalar_values = {
            name: self.values[name]
            for name in (
                "maximum_steps",
                "maximum_epochs",
                "log_interval",
                "seed",
                "compile",
            )
            if name in self.values
        }
        return SwagTrainingConfig(
            base_checkpoint=self.values["checkpoint"],
            data=self.values["data"],
            output_run=self.values["output"],
            lora=lora,
            optimizer=_build_optimizer(self.values, finetune=True),
            loader=_build_loader(self.values),
            checkpoint=_build_checkpoint(self.values, finetune=True),
            precision=precision,
            **scalar_values,
        )

    def dispatch(self) -> object:
        if "resume" in self.values:
            from sml.training.swag import resume_finetune

            return resume_finetune(
                self.values["resume"],
                data=self.values["data"],
                overrides=self.to_domain(),
            )
        from sml.training.swag import finetune

        return finetune(self.to_domain())


@dataclass(frozen=True, slots=True)
class ExportCommand(_Command):
    command: ClassVar[str] = "export"

    def to_domain(self) -> tuple[Path, Path]:
        return self.values["checkpoint"], self.values["output"]

    def dispatch(self) -> object:
        from sml.training.swag import export_merged

        return export_merged(*self.to_domain())


@dataclass(frozen=True, slots=True)
class VerifyCommand(_Command):
    command: ClassVar[str] = "verify"

    def to_domain(self) -> tuple[Path, bool]:
        return self.values["path"], self.values.get("full", False)

    def dispatch(self) -> object:
        from sml.artifacts.verify import verify_artifact

        path, full = self.to_domain()
        return verify_artifact(path, full=full)


_DTO_TYPES: dict[str, type[_Command]] = {
    command_type.command: command_type
    for command_type in (
        TokenizeCommand,
        PreparePretrainingCommand,
        PrepareSwagCommand,
        TrainCommand,
        InferCommand,
        EvaluateCommand,
        FinetuneCommand,
        ExportCommand,
        VerifyCommand,
    )
}


def _validate_command(command: str, values: dict[str, object]) -> None:
    required = {
        "tokenize": ("input", "output"),
        "prepare.pretraining": ("input", "tokenizer", "output"),
        "prepare.swag": ("checkpoint", "revision", "output"),
        "infer": ("checkpoint", "prompt"),
        "evaluate": ("checkpoint", "tasks", "output"),
        "export": ("checkpoint", "output"),
        "verify": ("path",),
    }
    if command in required:
        _require(values, *required[command])
    if command == "train":
        if "resume" in values:
            forbidden = set(values) & {
                "output",
                "model",
                "optimizer",
                "precision",
                "microbatch_size",
                "gradient_accumulation_steps",
                "prefetch_depth",
                "learning_rate",
                "seed",
                "compile",
            }
            if forbidden:
                raise SMLConfigurationError(
                    "resume rejects fresh-run configuration: "
                    + ", ".join(sorted(forbidden))
                )
        else:
            _require(values, "data", "output")
    if command == "finetune":
        if "resume" in values:
            forbidden = set(values) & {
                "checkpoint",
                "output",
                "lora",
                "optimizer",
                "precision",
                "microbatch_size",
                "gradient_accumulation_steps",
                "prefetch_depth",
                "learning_rate",
                "seed",
                "compile",
            }
            if forbidden:
                raise SMLConfigurationError(
                    "resume rejects fresh-run configuration: "
                    + ", ".join(sorted(forbidden))
                )
            _require(values, "data")
        else:
            _require(values, "checkpoint", "data", "output")
    for field_name in ("checkpoint", "resume"):
        if field_name in values:
            _reject_step_path(values[field_name], field_name)
    if command in {
        "tokenize",
        "prepare.pretraining",
        "prepare.swag",
        "train",
        "finetune",
        "export",
    }:
        _reject_existing_output(values)
    if command == "evaluate":
        tasks = values.get("tasks")
        if not isinstance(tasks, (list, tuple)) or not tasks:
            raise SMLConfigurationError("at least one evaluation task is required")
        if any(task not in {"hellaswag", "winogrande"} for task in tasks):
            raise SMLConfigurationError(
                "evaluation task must be hellaswag or winogrande"
            )


def parse_command(argv: Sequence[str]) -> _Command:
    arguments = list(argv)
    if any(
        argument == "--step" or argument.startswith("--step=") for argument in arguments
    ):
        raise SMLConfigurationError("historical --step selection is not supported")
    parser = _build_parser()
    namespace = parser.parse_args(arguments)
    command, values = _normalize_cli_values(namespace)
    _validate_command(command, values)
    dto = _DTO_TYPES[command](values)
    try:
        dto.to_domain()
    except _DOMAIN_ERRORS:
        raise
    except (TypeError, ValueError) as error:
        raise SMLConfigurationError(str(error)) from error
    return dto


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()
    if not arguments:
        raise SMLConfigurationError("a command is required")
    if arguments in (["--help"], ["-h"]):
        parser.print_help()
        return 0
    try:
        command = parse_command(arguments)
        result = command.dispatch()
    except _DOMAIN_ERRORS as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return _EXIT_CODES[type(error)]
    print(result)
    return 0
