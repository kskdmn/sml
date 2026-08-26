from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten

from sml.errors import SMLConfigurationError
from sml.model.layers import LoRAAdapterSpec, LoRAForwardPolicy, _Linear, _linear

_ALLOWED_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def _require_finite(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SMLConfigurationError(f"{field_name} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise SMLConfigurationError(f"{field_name} must be finite")
    return normalized


@dataclass(frozen=True, slots=True)
class LoRAInitializerConfig:
    lora_a: float = 0.01
    lora_b: float = 0.0

    def __post_init__(self) -> None:
        for field_name in ("lora_a", "lora_b"):
            value = _require_finite(getattr(self, field_name), field_name)
            if value < 0.0:
                raise SMLConfigurationError(
                    f"{field_name} must be non-negative and finite"
                )


@dataclass(frozen=True, slots=True)
class LoRAConfig:
    rank: int = 16
    alpha: float = 32.0
    scaling_mode: Literal["lora", "rslora"] = "rslora"
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    initializer: LoRAInitializerConfig = field(default_factory=LoRAInitializerConfig)

    def __post_init__(self) -> None:
        if (
            isinstance(self.rank, bool)
            or not isinstance(self.rank, int)
            or self.rank <= 0
        ):
            raise SMLConfigurationError("rank must be positive")
        if _require_finite(self.alpha, "alpha") <= 0.0:
            raise SMLConfigurationError("alpha must be positive")
        if self.scaling_mode not in ("lora", "rslora"):
            raise SMLConfigurationError("scaling_mode must be one of: lora, rslora")
        dropout = _require_finite(self.dropout, "dropout")
        if not 0.0 <= dropout < 1.0:
            raise SMLConfigurationError("dropout must be in [0, 1)")
        if not isinstance(self.initializer, LoRAInitializerConfig):
            raise SMLConfigurationError("initializer must be a LoRAInitializerConfig")

        targets = tuple(self.target_modules)
        if (
            not targets
            or any(not isinstance(name, str) or not name for name in targets)
            or len(set(targets)) != len(targets)
            or any(name not in _ALLOWED_TARGET_MODULES for name in targets)
        ):
            raise SMLConfigurationError("target_modules must contain module names")
        object.__setattr__(self, "target_modules", targets)


@dataclass(frozen=True, slots=True)
class LoRAPrecisionConfig:
    frozen_base_dtype: Literal["bfloat16"] = "bfloat16"
    adapter_parameter_dtype: Literal["float32"] = "float32"
    gradient_accumulator_dtype: Literal["float32"] = "float32"
    optimizer_state_dtype: Literal["float32"] = "float32"
    update_dtype: Literal["float32"] = "float32"
    dynamic_loss_scaling: Literal[False] = False

    def __post_init__(self) -> None:
        expected = {
            "frozen_base_dtype": "bfloat16",
            "adapter_parameter_dtype": "float32",
            "gradient_accumulator_dtype": "float32",
            "optimizer_state_dtype": "float32",
            "update_dtype": "float32",
            "dynamic_loss_scaling": False,
        }
        for field_name, value in expected.items():
            actual = getattr(self, field_name)
            if actual != value or type(actual) is not type(value):
                raise SMLConfigurationError(f"{field_name} must be {value!r}")


def lora_config_from_mapping(value: Mapping[str, object]) -> LoRAConfig:
    if not isinstance(value, Mapping):
        raise SMLConfigurationError("lora must be a mapping")
    values = dict(value)
    initializer_values = values.pop("initializer")
    if not isinstance(initializer_values, Mapping):
        raise SMLConfigurationError("initializer must be a mapping")
    target_modules = values.get("target_modules")
    if isinstance(target_modules, list):
        values["target_modules"] = tuple(target_modules)
    return LoRAConfig(
        initializer=LoRAInitializerConfig(**dict(initializer_values)),
        **values,
    )


def split_adapter_parameters(parameters: object) -> tuple[object, object]:
    """Split LoRA adapter leaves from the frozen BF16 base parameter tree."""

    if isinstance(parameters, dict):
        adapters: dict = {}
        frozen: dict = {}
        for key, value in parameters.items():
            if key in {"lora_a", "lora_b"}:
                adapters[key] = value
            elif isinstance(value, (dict, list, tuple)):
                nested_adapters, nested_frozen = split_adapter_parameters(value)
                if nested_adapters:
                    adapters[key] = nested_adapters
                if nested_frozen:
                    frozen[key] = nested_frozen
            else:
                frozen[key] = value
        return adapters, frozen
    if isinstance(parameters, list):
        adapter_items = []
        frozen_items = []
        has_adapters = False
        has_frozen = False
        for value in parameters:
            nested_adapters, nested_frozen = split_adapter_parameters(value)
            adapter_items.append(nested_adapters)
            frozen_items.append(nested_frozen)
            has_adapters = has_adapters or bool(nested_adapters)
            has_frozen = has_frozen or bool(nested_frozen)
        return (
            adapter_items if has_adapters else {},
            frozen_items if has_frozen else {},
        )
    return {}, parameters


def _lora_scale(config: LoRAConfig) -> float:
    alpha = float(config.alpha)
    rank = float(config.rank)
    if config.scaling_mode == "lora":
        return alpha / rank
    return alpha / math.sqrt(rank)


class LoRALinear(nn.Module):
    def __init__(
        self,
        linear: _Linear,
        config: LoRAConfig,
        *,
        module_path: str,
        spec: LoRAAdapterSpec,
        key: mx.array,
    ) -> None:
        super().__init__()
        if not isinstance(linear, _Linear):
            raise SMLConfigurationError("LoRA wraps package linear modules")
        if spec.module_path != module_path:
            raise SMLConfigurationError("LoRA adapter spec path must match module path")

        input_dims = int(linear.weight.shape[1])
        output_dims = int(linear.weight.shape[0])
        self.base = linear
        self.base.freeze()
        self.module_path = module_path
        self.spec = spec

        key, adapter_a_key = mx.random.split(key)
        _, adapter_b_key = mx.random.split(key)
        self.lora_a = mx.random.normal(
            shape=(config.rank, input_dims),
            scale=config.initializer.lora_a,
            key=adapter_a_key,
        ).astype(mx.float32)
        if config.initializer.lora_b == 0.0:
            self.lora_b = mx.zeros((output_dims, config.rank), dtype=mx.float32)
        else:
            self.lora_b = mx.random.normal(
                shape=(output_dims, config.rank),
                scale=config.initializer.lora_b,
                key=adapter_b_key,
            ).astype(mx.float32)

    def __call__(
        self,
        x: mx.array,
        *,
        key: mx.array | None = None,
        training: bool = False,
    ) -> tuple[mx.array, mx.array | None]:
        return _linear(
            x,
            self.parameters(),
            module_path=self.module_path,
            lora_policy=LoRAForwardPolicy((self.spec,)),
            training=training,
            key=key,
        )


def _get_submodule(module, path: str):
    current = module
    if not path:
        return current
    for part in path.split("."):
        if isinstance(current, (list, tuple)):
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current


def _split_parent(model, name: str):
    if "." not in name:
        return model, name
    parent_name, child_name = name.rsplit(".", 1)
    return _get_submodule(model, parent_name), child_name


def apply_lora(model, config: LoRAConfig, *, key: mx.array):
    targets: list[tuple[str, _Linear]] = []
    for name, module in model.named_modules():
        if not isinstance(module, _Linear):
            continue
        short_name = name.rsplit(".", 1)[-1]
        if short_name not in config.target_modules:
            continue
        targets.append((name, module))

    scale = _lora_scale(config)
    policy = LoRAForwardPolicy(
        tuple(
            LoRAAdapterSpec(
                module_path=name,
                scale=scale,
                dropout=config.dropout,
            )
            for name, _module in targets
        )
    )
    model.lora_forward_policy = policy
    model.freeze()
    for name, module in targets:
        key, module_key = mx.random.split(key)
        parent, child_name = _split_parent(model, name)
        spec = policy.for_module(name)
        if spec is None:
            raise RuntimeError(f"missing LoRA policy spec for {name}")
        wrapper = LoRALinear(
            module,
            config,
            module_path=name,
            spec=spec,
            key=module_key,
        )
        setattr(parent, child_name, wrapper)
        wrapper.unfreeze(keys=["lora_a", "lora_b"], recurse=False)
    return model


def lora_state_dict(model) -> dict[str, mx.array]:
    state: dict[str, mx.array] = {}
    for name, module in model.named_modules():
        if not isinstance(module, LoRALinear):
            continue
        prefix = f"{name}." if name else ""
        state[f"{prefix}lora_a"] = module.lora_a
        state[f"{prefix}lora_b"] = module.lora_b
    return dict(sorted(state.items()))


def load_lora_state_dict(model, state: dict[str, mx.array]) -> None:
    expected = lora_state_dict(model)
    missing = sorted(set(expected) - set(state))
    additional = sorted(set(state) - set(expected))
    if missing:
        raise SMLConfigurationError(f"missing adapter keys: {missing}")
    if additional:
        raise SMLConfigurationError(f"additional adapter keys: {additional}")
    for name in sorted(expected):
        array = state[name]
        current = expected[name]
        if list(array.shape) != list(current.shape):
            raise SMLConfigurationError(f"adapter shape mismatch for {name}")
        if array.dtype != current.dtype:
            raise SMLConfigurationError(f"adapter dtype mismatch for {name}")
    model.load_weights(list(state.items()), strict=False)
    mx.eval(model.parameters())


def merged_model_weights(model) -> dict[str, mx.array]:
    lora_modules = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, LoRALinear)
    }
    merged: dict[str, mx.array] = {}
    for name, value in tree_flatten(model.parameters()):
        if name.endswith((".lora_a", ".lora_b", ".scale")):
            continue
        if name.endswith(".base.weight"):
            prefix = name[: -len(".base.weight")]
            module = lora_modules[prefix]
            merged[f"{prefix}.weight"] = (
                module.base.weight.astype(mx.float32)
                + module.scale.astype(mx.float32) * (module.lora_b @ module.lora_a)
            ).astype(mx.bfloat16)
            continue
        merged[name] = value
    return merged
