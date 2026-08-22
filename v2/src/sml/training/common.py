from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import mlx.core as mx
from mlx.utils import tree_map, tree_map_with_path

from sml.errors import SMLConfigurationError
from sml.model.config import ModelConfig

_INT32_MAX = 2**31 - 1


def _require_finite(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SMLConfigurationError(f"{field_name} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise SMLConfigurationError(f"{field_name} must be finite")
    return normalized


def _require_positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SMLConfigurationError(f"{field_name} must be a positive integer")
    return value


def _require_int32_counter(
    value: object, field_name: str, *, allow_zero: bool = False
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SMLConfigurationError(f"{field_name} must be an int32 counter")
    if value < 0 or (value == 0 and not allow_zero) or value > _INT32_MAX:
        raise SMLConfigurationError(f"{field_name} must be an int32 counter")
    return value


def _require_probability(value: object, field_name: str) -> float:
    normalized = _require_finite(value, field_name)
    if not 0.0 <= normalized < 1.0:
        raise SMLConfigurationError(f"{field_name} must be in [0, 1)")
    return normalized


def _materialized_truth(condition: mx.array) -> bool | None:
    try:
        return bool(condition)
    except ValueError:
        return None


def _require_top_level_dict(tree: object, name: str) -> dict:
    if not isinstance(tree, dict):
        raise SMLConfigurationError(f"{name} must be a top-level dict array tree")
    return tree


def _array_leaves(tree: object, name: str) -> list[mx.array]:
    leaves: list[mx.array] = []

    def visit(node: object, path: tuple[object, ...]) -> None:
        location = ".".join(str(part) for part in path)
        if isinstance(node, dict):
            for key, value in node.items():
                visit(value, (*path, key))
            return
        if isinstance(node, (list, tuple)):
            for index, value in enumerate(node):
                visit(value, (*path, index))
            return
        if not isinstance(node, mx.array):
            raise SMLConfigurationError(f"{name}.{location} must be an MLX array")
        leaves.append(node)

    visit(tree, ())
    if not leaves:
        raise SMLConfigurationError(f"{name} must not be empty")
    return leaves


def _require_same_structure(
    left: object,
    right: object,
    *,
    left_name: str,
    right_name: str,
) -> tuple[dict[str, mx.array], dict[str, mx.array]]:
    _array_leaves(left, left_name)
    _array_leaves(right, right_name)

    def compare(left_node: object, right_node: object) -> None:
        if isinstance(left_node, mx.array):
            if not isinstance(right_node, mx.array):
                raise SMLConfigurationError(
                    f"{left_name} and {right_name} must have exact matching structure"
                )
            if left_node.shape != right_node.shape:
                raise SMLConfigurationError(
                    f"{left_name} and {right_name} must have exact matching shapes"
                )
            return
        if type(left_node) is not type(right_node):
            raise SMLConfigurationError(
                f"{left_name} and {right_name} must have exact matching structure"
            )
        if isinstance(left_node, dict):
            if set(left_node) != set(right_node):
                raise SMLConfigurationError(
                    f"{left_name} and {right_name} must have exact matching keys"
                )
            for key in left_node:
                compare(left_node[key], right_node[key])
            return
        if isinstance(left_node, (list, tuple)):
            if len(left_node) != len(right_node):
                raise SMLConfigurationError(
                    f"{left_name} and {right_name} must have exact matching structure"
                )
            for left_value, right_value in zip(left_node, right_node, strict=True):
                compare(left_value, right_value)
            return
        raise SMLConfigurationError(
            f"{left_name} and {right_name} must have exact matching structure"
        )

    compare(left, right)
    return {}, {}


def _require_matching_tree_keys(
    left: object, right: object, *, left_name: str, right_name: str
) -> None:
    def compare(left_node: object, right_node: object) -> None:
        if isinstance(left_node, mx.array):
            if isinstance(right_node, (dict, list, tuple)):
                raise SMLConfigurationError(
                    f"{left_name} and {right_name} must have exact matching structure"
                )
            return
        if type(left_node) is not type(right_node):
            raise SMLConfigurationError(
                f"{left_name} and {right_name} must have exact matching structure"
            )
        if isinstance(left_node, dict):
            if set(left_node) != set(right_node):
                raise SMLConfigurationError(
                    f"{left_name} and {right_name} must have exact matching keys"
                )
            for key in left_node:
                compare(left_node[key], right_node[key])
            return
        if isinstance(left_node, (list, tuple)):
            if len(left_node) != len(right_node):
                raise SMLConfigurationError(
                    f"{left_name} and {right_name} must have exact matching structure"
                )
            for left_value, right_value in zip(left_node, right_node, strict=True):
                compare(left_value, right_value)
            return
        raise SMLConfigurationError(
            f"{left_name} and {right_name} must have exact matching structure"
        )

    _array_leaves(left, left_name)
    compare(left, right)


def _require_dtype(tree: object, name: str, dtype: mx.Dtype) -> dict[str, mx.array]:
    leaves = _array_leaves(tree, name)
    if any(leaf.dtype != dtype for leaf in leaves):
        raise SMLConfigurationError(f"{name} leaves must have dtype {dtype}")
    return {}


def _path_category(path: str) -> str:
    if path.startswith("embed_tokens."):
        return "embed_tokens"
    if path.startswith("lm_head."):
        return "lm_head"
    leaf_name = path.rsplit(".", maxsplit=1)[-1].lower()
    if leaf_name == "lora_a":
        return "lora_a"
    if leaf_name == "lora_b":
        return "lora_b"
    if (
        ".input_norm." in path
        or ".post_attn_norm." in path
        or ".rms_norm." in path
        or path.startswith("norm.")
    ):
        return "rms_norm"
    for category in (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ):
        if f".{category}." in path:
            return category
    return "other"


@dataclass(frozen=True, slots=True)
class WeightDecayPolicy:
    embed_tokens: float = 0.0
    lm_head: float = 0.0
    rms_norm: float = 0.0
    q_proj: float = 0.1
    k_proj: float = 0.1
    v_proj: float = 0.1
    o_proj: float = 0.1
    gate_proj: float = 0.1
    up_proj: float = 0.1
    down_proj: float = 0.1
    lora_a: float = 0.0
    lora_b: float = 0.0
    other: float = 0.1

    def __post_init__(self) -> None:
        for field_name in (
            "embed_tokens",
            "lm_head",
            "rms_norm",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lora_a",
            "lora_b",
            "other",
        ):
            if _require_finite(getattr(self, field_name), field_name) < 0.0:
                raise SMLConfigurationError(f"{field_name} must be non-negative")


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    learning_rate: float = 3e-4
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    bias_correction: bool = False
    schedule_steps: int | None = 268_000
    warmup_steps: int | None = None
    minimum_learning_rate_ratio: float = 0.1
    gradient_clip_norm: float = 1.0
    weight_decay: WeightDecayPolicy = field(default_factory=WeightDecayPolicy)

    def __post_init__(self) -> None:
        if _require_finite(self.learning_rate, "learning_rate") <= 0.0:
            raise SMLConfigurationError("learning_rate must be positive")
        _require_probability(self.beta1, "beta1")
        _require_probability(self.beta2, "beta2")
        if _require_finite(self.epsilon, "epsilon") <= 0.0:
            raise SMLConfigurationError("epsilon must be positive")
        if not isinstance(self.bias_correction, bool):
            raise SMLConfigurationError("bias_correction must be a bool")
        if self.schedule_steps is not None:
            _require_int32_counter(self.schedule_steps, "schedule_steps")
        if self.warmup_steps is not None:
            _require_int32_counter(self.warmup_steps, "warmup_steps", allow_zero=True)
        if self.schedule_steps is not None and not (
            0 <= resolved_warmup_steps(self) < self.schedule_steps
        ):
            raise SMLConfigurationError(
                "warmup_steps must satisfy 0 <= warmup_steps < schedule_steps"
            )
        minimum_ratio = _require_finite(
            self.minimum_learning_rate_ratio, "minimum_learning_rate_ratio"
        )
        if not 0.0 <= minimum_ratio <= 1.0:
            raise SMLConfigurationError("minimum_learning_rate_ratio must be in [0, 1]")
        if _require_finite(self.gradient_clip_norm, "gradient_clip_norm") <= 0.0:
            raise SMLConfigurationError("gradient_clip_norm must be positive")
        if not isinstance(self.weight_decay, WeightDecayPolicy):
            raise SMLConfigurationError("weight_decay must be a WeightDecayPolicy")


@dataclass(frozen=True, slots=True)
class LoaderConfig:
    microbatch_size: int = 1
    gradient_accumulation_steps: int = 8
    prefetch_depth: int = 2
    epoch_seed: int = 42

    def __post_init__(self) -> None:
        for field_name in (
            "microbatch_size",
            "gradient_accumulation_steps",
            "prefetch_depth",
        ):
            _require_int32_counter(getattr(self, field_name), field_name)
        if (
            isinstance(self.epoch_seed, bool)
            or not isinstance(self.epoch_seed, int)
            or not 0 <= self.epoch_seed <= 2**32 - 1
        ):
            raise SMLConfigurationError("epoch_seed must be an unsigned 32-bit integer")


@dataclass(frozen=True, slots=True)
class CheckpointPolicy:
    interval: int = 1_000
    keep_last: int | None = None

    def __post_init__(self) -> None:
        _require_int32_counter(self.interval, "interval")
        if self.keep_last is not None:
            _require_positive_int(self.keep_last, "keep_last")


@dataclass(frozen=True, slots=True)
class PrecisionConfig:
    master_parameter_dtype: Literal["float32"] = "float32"
    working_parameter_dtype: Literal["bfloat16"] = "bfloat16"
    gradient_accumulator_dtype: Literal["float32"] = "float32"
    optimizer_state_dtype: Literal["float32"] = "float32"
    update_dtype: Literal["float32"] = "float32"
    master_weights: Literal[True] = True
    dynamic_loss_scaling: Literal[False] = False

    def __post_init__(self) -> None:
        expected = {
            "master_parameter_dtype": "float32",
            "working_parameter_dtype": "bfloat16",
            "gradient_accumulator_dtype": "float32",
            "optimizer_state_dtype": "float32",
            "update_dtype": "float32",
            "master_weights": True,
            "dynamic_loss_scaling": False,
        }
        for field_name, value in expected.items():
            actual = getattr(self, field_name)
            if actual != value or type(actual) is not type(value):
                raise SMLConfigurationError(f"{field_name} must be {value!r}")


@dataclass(frozen=True, slots=True)
class PretrainingConfig:
    data: Path
    output_run: Path
    model: ModelConfig
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    loader: LoaderConfig = field(default_factory=LoaderConfig)
    checkpoint: CheckpointPolicy = field(default_factory=CheckpointPolicy)
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)
    maximum_steps: int | None = None
    maximum_epochs: int | None = 1
    log_interval: int = 10
    seed: int = 42
    compile: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.data, Path) or not isinstance(self.output_run, Path):
            raise SMLConfigurationError("data and output_run must be Paths")
        if not isinstance(self.model, ModelConfig):
            raise SMLConfigurationError("model must be a ModelConfig")
        if self.model.rope_scaling_factor != 1.0:
            raise SMLConfigurationError(
                "pretraining model rope_scaling_factor must be exactly 1.0"
            )
        for field_name, expected_type in (
            ("optimizer", OptimizerConfig),
            ("loader", LoaderConfig),
            ("checkpoint", CheckpointPolicy),
            ("precision", PrecisionConfig),
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


def resolved_warmup_steps(config: OptimizerConfig) -> int:
    if config.warmup_steps is not None:
        return config.warmup_steps
    return int(
        0.01 * (10_000 if config.schedule_steps is None else config.schedule_steps)
    )


def learning_rate_at(step: mx.array, config: OptimizerConfig) -> mx.array:
    step_float = step.astype(mx.float32)
    learning_rate = mx.array(config.learning_rate, dtype=mx.float32)
    warmup_steps = resolved_warmup_steps(config)
    warmup = (step_float + 1.0) / max(1, warmup_steps)
    if config.schedule_steps is None:
        return mx.where(
            step_float < warmup_steps, learning_rate * warmup, learning_rate
        )
    decay_steps = config.schedule_steps - warmup_steps
    progress = mx.clip(
        (step_float - warmup_steps) / decay_steps,
        a_min=0.0,
        a_max=1.0,
    )
    cosine_ratio = config.minimum_learning_rate_ratio + (
        1.0 - config.minimum_learning_rate_ratio
    ) * (0.5 * (1.0 + mx.cos(math.pi * progress)))
    decayed = learning_rate * cosine_ratio
    return mx.where(step_float < warmup_steps, learning_rate * warmup, decayed)


@dataclass(frozen=True, slots=True)
class BaseParameterState:
    master_parameters: dict
    working_parameters: dict

    def __post_init__(self) -> None:
        _require_top_level_dict(self.master_parameters, "master_parameters")
        _require_top_level_dict(self.working_parameters, "working_parameters")
        _require_same_structure(
            self.master_parameters,
            self.working_parameters,
            left_name="master_parameters",
            right_name="working_parameters",
        )
        if any(
            leaf.dtype != mx.float32
            for leaf in _array_leaves(self.master_parameters, "master_parameters")
        ):
            raise SMLConfigurationError(
                "master_parameters leaves must have dtype float32"
            )
        if any(
            leaf.dtype != mx.bfloat16
            for leaf in _array_leaves(self.working_parameters, "working_parameters")
        ):
            raise SMLConfigurationError(
                "working_parameters leaves must have dtype bfloat16"
            )
        for master, working in zip(
            _array_leaves(self.master_parameters, "master_parameters"),
            _array_leaves(self.working_parameters, "working_parameters"),
            strict=True,
        ):
            if not bool(mx.array_equal(working, master.astype(mx.bfloat16))):
                raise SMLConfigurationError(
                    "working_parameters leaves must be exact bfloat16 casts of masters"
                )

    def to_tree(self) -> tuple[dict, dict]:
        return self.master_parameters, self.working_parameters

    @classmethod
    def from_tree(cls, tree: object) -> BaseParameterState:
        if not isinstance(tree, tuple) or len(tree) != 2:
            raise SMLConfigurationError(
                "base parameter state tree must be a two-item tuple"
            )
        masters, working = tree
        if not isinstance(masters, dict) or not isinstance(working, dict):
            raise SMLConfigurationError("base parameter state tree must contain dicts")
        return cls(masters, working)

    @classmethod
    def from_compiled_tree(cls, tree: object) -> BaseParameterState:
        """Wrap a compiled result without materializing its array values on the host."""
        if not isinstance(tree, tuple) or len(tree) != 2:
            raise SMLConfigurationError(
                "base parameter state tree must be a two-item tuple"
            )
        masters, working = tree
        if not isinstance(masters, dict) or not isinstance(working, dict):
            raise SMLConfigurationError("base parameter state tree must contain dicts")
        _require_top_level_dict(masters, "master_parameters")
        _require_top_level_dict(working, "working_parameters")
        _require_same_structure(
            masters,
            working,
            left_name="master_parameters",
            right_name="working_parameters",
        )
        _require_dtype(masters, "master_parameters", mx.float32)
        _require_dtype(working, "working_parameters", mx.bfloat16)
        instance = object.__new__(cls)
        object.__setattr__(instance, "master_parameters", masters)
        object.__setattr__(instance, "working_parameters", working)
        return instance


@dataclass(frozen=True, slots=True)
class AdamState:
    step: mx.array
    first_moments: dict
    second_moments: dict

    def __post_init__(self) -> None:
        if (
            not isinstance(self.step, mx.array)
            or self.step.dtype != mx.int32
            or self.step.shape != ()
        ):
            raise SMLConfigurationError("Adam step must be an int32 scalar")
        if _materialized_truth((self.step >= 0) & (self.step <= _INT32_MAX)) is False:
            raise SMLConfigurationError("Adam step must be an int32 counter")
        _require_top_level_dict(self.first_moments, "first_moments")
        _require_top_level_dict(self.second_moments, "second_moments")
        _require_same_structure(
            self.first_moments,
            self.second_moments,
            left_name="first_moments",
            right_name="second_moments",
        )
        if any(
            leaf.dtype != mx.float32
            for leaf in _array_leaves(self.first_moments, "first_moments")
        ):
            raise SMLConfigurationError("first_moments leaves must have dtype float32")
        if any(
            leaf.dtype != mx.float32
            for leaf in _array_leaves(self.second_moments, "second_moments")
        ):
            raise SMLConfigurationError("second_moments leaves must have dtype float32")

    def to_tree(self) -> tuple[mx.array, dict, dict]:
        return self.step, self.first_moments, self.second_moments

    @classmethod
    def from_tree(cls, tree: object) -> AdamState:
        if not isinstance(tree, tuple) or len(tree) != 3:
            raise SMLConfigurationError("Adam state tree must be a three-item tuple")
        step, first_moments, second_moments = tree
        if not isinstance(first_moments, dict) or not isinstance(second_moments, dict):
            raise SMLConfigurationError("Adam state tree must contain moment dicts")
        return cls(step, first_moments, second_moments)

    @classmethod
    def from_compiled_tree(cls, tree: object) -> AdamState:
        """Wrap a compiled result without synchronizing its device counter."""
        if not isinstance(tree, tuple) or len(tree) != 3:
            raise SMLConfigurationError("Adam state tree must be a three-item tuple")
        step, first_moments, second_moments = tree
        if not isinstance(first_moments, dict) or not isinstance(second_moments, dict):
            raise SMLConfigurationError("Adam state tree must contain moment dicts")
        if not isinstance(step, mx.array) or step.dtype != mx.int32 or step.shape != ():
            raise SMLConfigurationError("Adam step must be an int32 scalar")
        _require_top_level_dict(first_moments, "first_moments")
        _require_top_level_dict(second_moments, "second_moments")
        _require_same_structure(
            first_moments,
            second_moments,
            left_name="first_moments",
            right_name="second_moments",
        )
        _require_dtype(first_moments, "first_moments", mx.float32)
        _require_dtype(second_moments, "second_moments", mx.float32)
        instance = object.__new__(cls)
        object.__setattr__(instance, "step", step)
        object.__setattr__(instance, "first_moments", first_moments)
        object.__setattr__(instance, "second_moments", second_moments)
        return instance


@dataclass(frozen=True, slots=True)
class TrainerState:
    accumulators: dict
    accumulation_count: mx.array
    next_key: mx.array

    def __post_init__(self) -> None:
        _require_top_level_dict(self.accumulators, "accumulators")
        _require_dtype(self.accumulators, "accumulators", mx.float32)
        if (
            not isinstance(self.accumulation_count, mx.array)
            or self.accumulation_count.dtype != mx.int32
            or self.accumulation_count.shape != ()
        ):
            raise SMLConfigurationError("accumulation_count must be an int32 scalar")
        if (
            _materialized_truth(
                (self.accumulation_count >= 0) & (self.accumulation_count <= _INT32_MAX)
            )
            is False
        ):
            raise SMLConfigurationError("accumulation_count must be an int32 counter")
        if (
            not isinstance(self.next_key, mx.array)
            or self.next_key.dtype != mx.uint32
            or self.next_key.shape != (2,)
        ):
            raise SMLConfigurationError("next_key must be a uint32 MLX random key")

    def to_tree(self) -> tuple[dict, mx.array, mx.array]:
        return self.accumulators, self.accumulation_count, self.next_key

    @classmethod
    def from_tree(cls, tree: object) -> TrainerState:
        if not isinstance(tree, tuple) or len(tree) != 3:
            raise SMLConfigurationError("trainer state tree must be a three-item tuple")
        accumulators, accumulation_count, next_key = tree
        if not isinstance(accumulators, dict):
            raise SMLConfigurationError(
                "trainer state tree must contain accumulator dicts"
            )
        return cls(accumulators, accumulation_count, next_key)

    @classmethod
    def from_compiled_tree(cls, tree: object) -> TrainerState:
        """Wrap a compiled result without synchronizing its device counter."""
        if not isinstance(tree, tuple) or len(tree) != 3:
            raise SMLConfigurationError("trainer state tree must be a three-item tuple")
        accumulators, accumulation_count, next_key = tree
        if not isinstance(accumulators, dict):
            raise SMLConfigurationError(
                "trainer state tree must contain accumulator dicts"
            )
        _require_top_level_dict(accumulators, "accumulators")
        _require_dtype(accumulators, "accumulators", mx.float32)
        if (
            not isinstance(accumulation_count, mx.array)
            or accumulation_count.dtype != mx.int32
            or accumulation_count.shape != ()
        ):
            raise SMLConfigurationError("accumulation_count must be an int32 scalar")
        if (
            not isinstance(next_key, mx.array)
            or next_key.dtype != mx.uint32
            or next_key.shape != (2,)
        ):
            raise SMLConfigurationError("next_key must be a uint32 MLX random key")
        instance = object.__new__(cls)
        object.__setattr__(instance, "accumulators", accumulators)
        object.__setattr__(instance, "accumulation_count", accumulation_count)
        object.__setattr__(instance, "next_key", next_key)
        return instance


def initialize_base_parameter_state(working_parameters: dict) -> BaseParameterState:
    _require_top_level_dict(working_parameters, "working_parameters")
    _require_dtype(working_parameters, "working_parameters", mx.bfloat16)
    master_parameters = tree_map(
        lambda parameter: parameter.astype(mx.float32), working_parameters
    )
    return BaseParameterState(master_parameters, working_parameters)


def initialize_adam_state(master_parameters: dict) -> AdamState:
    _require_top_level_dict(master_parameters, "master_parameters")
    _require_dtype(master_parameters, "master_parameters", mx.float32)
    return AdamState(
        step=mx.array(0, dtype=mx.int32),
        first_moments=tree_map(mx.zeros_like, master_parameters),
        second_moments=tree_map(mx.zeros_like, master_parameters),
    )


def build_weight_decay_tree(named_parameters: dict, policy: WeightDecayPolicy) -> dict:
    _require_top_level_dict(named_parameters, "named_parameters")
    _array_leaves(named_parameters, "named_parameters")
    if not isinstance(policy, WeightDecayPolicy):
        raise SMLConfigurationError("policy must be a WeightDecayPolicy")
    return tree_map_with_path(
        lambda path, _parameter: getattr(policy, _path_category(path)), named_parameters
    )


def accumulate_fp32(accumulators: dict, gradients: dict) -> dict:
    _require_top_level_dict(accumulators, "accumulators")
    _require_top_level_dict(gradients, "gradients")
    _require_dtype(accumulators, "accumulators", mx.float32)
    _require_same_structure(
        accumulators,
        gradients,
        left_name="accumulators",
        right_name="gradients",
    )
    return tree_map(
        lambda accumulator, gradient: (
            accumulator.astype(mx.float32) + gradient.astype(mx.float32)
        ).astype(mx.float32),
        accumulators,
        gradients,
    )


def normalize_and_clip(
    accumulated_gradients: dict,
    normalization_count: mx.array,
    *,
    gradient_clip_norm: float,
) -> dict:
    _require_top_level_dict(accumulated_gradients, "accumulated_gradients")
    _require_dtype(accumulated_gradients, "accumulated_gradients", mx.float32)
    if (
        not isinstance(normalization_count, mx.array)
        or normalization_count.dtype != mx.int32
        or normalization_count.shape != ()
    ):
        raise SMLConfigurationError("normalization_count must be an int32 scalar")
    if (
        _materialized_truth(
            (normalization_count > 0) & (normalization_count <= _INT32_MAX)
        )
        is False
    ):
        raise SMLConfigurationError(
            "normalization_count must be a positive int32 counter"
        )
    if _require_finite(gradient_clip_norm, "gradient_clip_norm") <= 0.0:
        raise SMLConfigurationError("gradient_clip_norm must be positive")
    valid_count = (normalization_count > 0) & (normalization_count <= _INT32_MAX)
    safe_count = mx.where(
        valid_count,
        normalization_count.astype(mx.float32),
        mx.array(1.0, dtype=mx.float32),
    )
    normalized = tree_map(
        lambda gradient: gradient / safe_count,
        accumulated_gradients,
    )
    squared_norm = sum(
        (
            mx.sum(mx.square(gradient.astype(mx.float32)))
            for gradient in _array_leaves(normalized, "normalized_gradients")
        ),
        mx.array(0.0, dtype=mx.float32),
    )
    global_norm = mx.sqrt(squared_norm)
    scale = mx.minimum(
        mx.array(1.0, dtype=mx.float32),
        mx.array(gradient_clip_norm, dtype=mx.float32)
        / mx.maximum(global_norm, mx.array(1e-12, dtype=mx.float32)),
    )
    return tree_map(
        lambda gradient: mx.where(
            valid_count,
            (gradient * scale).astype(mx.float32),
            mx.zeros_like(gradient),
        ),
        normalized,
    )


def _decay_coefficient(value: object, config: OptimizerConfig) -> float:
    if isinstance(value, bool):
        return config.weight_decay.other if value else 0.0
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        coefficient = float(value)
        if math.isfinite(coefficient) and coefficient >= 0.0:
            return coefficient
    raise SMLConfigurationError(
        "weight_decay_tree leaves must be bool or non-negative floats"
    )


def adamw_mixed_precision_update(
    master_parameters: dict,
    gradients: dict,
    state: AdamState,
    config: OptimizerConfig,
    weight_decay_tree: dict,
) -> tuple[dict, dict, AdamState]:
    if not isinstance(state, AdamState):
        raise SMLConfigurationError("state must be an AdamState")
    if not isinstance(config, OptimizerConfig):
        raise SMLConfigurationError("config must be an OptimizerConfig")
    _require_top_level_dict(master_parameters, "master_parameters")
    _require_top_level_dict(gradients, "gradients")
    _require_top_level_dict(weight_decay_tree, "weight_decay_tree")
    _require_same_structure(
        master_parameters,
        gradients,
        left_name="master_parameters",
        right_name="gradients",
    )
    _require_dtype(master_parameters, "master_parameters", mx.float32)
    gradient_leaves = _array_leaves(gradients, "gradients")
    if any(leaf.dtype not in (mx.bfloat16, mx.float32) for leaf in gradient_leaves):
        raise SMLConfigurationError(
            "gradients leaves must have dtype bfloat16 or float32"
        )
    _require_same_structure(
        master_parameters,
        state.first_moments,
        left_name="master_parameters",
        right_name="first_moments",
    )
    _require_same_structure(
        master_parameters,
        state.second_moments,
        left_name="master_parameters",
        right_name="second_moments",
    )
    _require_matching_tree_keys(
        master_parameters,
        weight_decay_tree,
        left_name="master_parameters",
        right_name="weight_decay_tree",
    )
    fp32_gradients = tree_map(lambda gradient: gradient.astype(mx.float32), gradients)
    can_increment = state.step < _INT32_MAX
    if _materialized_truth(can_increment) is False:
        raise SMLConfigurationError(
            "Adam step cannot be incremented past int32 maximum"
        )
    first_moments = tree_map(
        lambda first, gradient: (
            config.beta1 * first.astype(mx.float32) + (1.0 - config.beta1) * gradient
        ).astype(mx.float32),
        state.first_moments,
        fp32_gradients,
    )
    second_moments = tree_map(
        lambda second, gradient: (
            config.beta2 * second.astype(mx.float32)
            + (1.0 - config.beta2) * mx.square(gradient)
        ).astype(mx.float32),
        state.second_moments,
        fp32_gradients,
    )
    if _materialized_truth(can_increment) is None:
        first_moments = tree_map(
            lambda previous, updated: mx.where(can_increment, updated, previous),
            state.first_moments,
            first_moments,
        )
        second_moments = tree_map(
            lambda previous, updated: mx.where(can_increment, updated, previous),
            state.second_moments,
            second_moments,
        )
    completed_updates = state.step.astype(mx.float32) + 1.0
    if config.bias_correction:
        first_denominator = 1.0 - mx.power(config.beta1, completed_updates)
        second_denominator = 1.0 - mx.power(config.beta2, completed_updates)
        first_for_update = tree_map(
            lambda first: first / first_denominator, first_moments
        )
        second_for_update = tree_map(
            lambda second: second / second_denominator, second_moments
        )
    else:
        first_for_update = first_moments
        second_for_update = second_moments
    learning_rate = learning_rate_at(state.step, config)
    updated_masters = tree_map(
        lambda master, first, second, decay: (
            master.astype(mx.float32)
            - learning_rate
            * (
                first.astype(mx.float32)
                / (mx.sqrt(second.astype(mx.float32)) + config.epsilon)
                + _decay_coefficient(decay, config) * master.astype(mx.float32)
            )
        ).astype(mx.float32),
        master_parameters,
        first_for_update,
        second_for_update,
        weight_decay_tree,
    )
    if _materialized_truth(can_increment) is None:
        updated_masters = tree_map(
            lambda master, updated: mx.where(can_increment, updated, master),
            master_parameters,
            updated_masters,
        )
    working_parameters = tree_map(
        lambda master: master.astype(mx.bfloat16), updated_masters
    )
    return (
        updated_masters,
        working_parameters,
        AdamState(
            step=mx.where(
                can_increment,
                state.step + mx.array(1, dtype=mx.int32),
                state.step,
            ).astype(mx.int32),
            first_moments=first_moments,
            second_moments=second_moments,
        ),
    )


__all__ = (
    "AdamState",
    "BaseParameterState",
    "CheckpointPolicy",
    "LoaderConfig",
    "OptimizerConfig",
    "PrecisionConfig",
    "PretrainingConfig",
    "TrainerState",
    "WeightDecayPolicy",
    "accumulate_fp32",
    "adamw_mixed_precision_update",
    "build_weight_decay_tree",
    "initialize_adam_state",
    "initialize_base_parameter_state",
    "learning_rate_at",
    "normalize_and_clip",
    "resolved_warmup_steps",
)
