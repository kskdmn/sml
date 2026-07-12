"""
Low-rank adaptation (LoRA) helpers for MLX SML fine-tuning.

Wraps selected ``nn.Linear`` layers with trainable low-rank deltas while keeping
base weights frozen. ``merge_lora`` folds adapters into base weights so merged
checkpoints load through the standard inference path.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn

if TYPE_CHECKING:
    from sml import SMLLanguageModel

LORA_A_SUFFIX = "lora_A"
LORA_B_SUFFIX = "lora_B"
ATTENTION_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
)
MLP_TARGET_MODULES = (
    "gate_proj",
    "up_proj",
    "down_proj",
)
LORA_SCALING_MODES = ("lora", "rslora")


@dataclass(slots=True)
class LoRAConfig:
    """
    Low-rank adapter settings for SWAG fine-tuning.

    Adapters attach to attention and MLP projections by module name.
    """

    rank: int = 16
    alpha: float = 32.0
    scaling_mode: str = "rslora"
    dropout: float = 0.05
    target_modules: tuple[str, ...] | None = None
    target_q_proj: bool = True
    target_k_proj: bool = True
    target_v_proj: bool = True
    target_o_proj: bool = True
    target_gate_proj: bool = False
    target_up_proj: bool = False
    target_down_proj: bool = False

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if not math.isfinite(self.alpha) or self.alpha <= 0.0:
            raise ValueError("LoRA alpha must be positive")
        if self.scaling_mode not in LORA_SCALING_MODES:
            valid_modes = ", ".join(LORA_SCALING_MODES)
            raise ValueError(f"LoRA scaling_mode must be one of: {valid_modes}")
        if not math.isfinite(self.dropout) or self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")
        if self.target_modules is None:
            requested_targets = (
                (self.target_q_proj, "q_proj"),
                (self.target_k_proj, "k_proj"),
                (self.target_v_proj, "v_proj"),
                (self.target_o_proj, "o_proj"),
                (self.target_gate_proj, "gate_proj"),
                (self.target_up_proj, "up_proj"),
                (self.target_down_proj, "down_proj"),
            )
            self.target_modules = tuple(
                module_name
                for should_target, module_name in requested_targets
                if should_target
            )
        else:
            self.target_modules = tuple(self.target_modules)
        if not self.target_modules or any(
            not isinstance(module_name, str) or not module_name
            for module_name in self.target_modules
        ):
            raise ValueError("LoRA target_modules must contain module names")


class LoRALinear(nn.Module):
    def __init__(
        self,
        linear,
        rank: int,
        alpha: float,
        scaling_mode: str = "rslora",
        dropout: float = 0.0,
        lora_a_initializer_range: float = 0.01,
        lora_b_initializer_range: float = 0.0,
    ) -> None:
        """
        Compute ``linear(x) + scaling * (x @ A.T @ B.T)`` with base weights frozen.
        """
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if alpha <= 0.0:
            raise ValueError("LoRA alpha must be positive")
        if scaling_mode not in LORA_SCALING_MODES:
            valid_modes = ", ".join(LORA_SCALING_MODES)
            raise ValueError(f"LoRA scaling_mode must be one of: {valid_modes}")
        for field_name, value in (
            ("lora_a_initializer_range", lora_a_initializer_range),
            ("lora_b_initializer_range", lora_b_initializer_range),
        ):
            if value is None or not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{field_name} must be non-negative and finite")

        self.linear = linear
        self.rank = rank
        self.scaling_mode = scaling_mode
        self.scaling = (
            alpha / rank if scaling_mode == "lora" else alpha / math.sqrt(rank)
        )
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        input_dims = linear.weight.shape[1]
        output_dims = linear.weight.shape[0]
        self.lora_A = mx.random.normal(
            shape=(rank, input_dims),
            scale=lora_a_initializer_range,
        )
        self.lora_B = (
            mx.zeros((output_dims, rank), dtype=linear.weight.dtype)
            if lora_b_initializer_range == 0.0
            else mx.random.normal(
                shape=(output_dims, rank),
                scale=lora_b_initializer_range,
            )
        )
        self.linear.freeze()

    def __call__(self, x: mx.array) -> mx.array:
        output = self.linear(x)
        lora_x = self.lora_dropout(x)
        lora_output = lora_x @ self.lora_A.T @ self.lora_B.T
        if lora_output.dtype != output.dtype:
            lora_output = lora_output.astype(output.dtype)
        return output + self.scaling * lora_output

    def merge(self):
        """
        Fold the low-rank delta into the wrapped linear layer and return it.
        """
        delta = self.scaling * (self.lora_B @ self.lora_A)
        if delta.dtype != self.linear.weight.dtype:
            delta = delta.astype(self.linear.weight.dtype)
        self.linear.weight = self.linear.weight + delta
        return self.linear


def resolve_lora_initializer_range(
    parameter_initializer_range,
    field_name: str,
    default: float,
) -> float:
    if parameter_initializer_range is None:
        return default
    if isinstance(parameter_initializer_range, Mapping):
        return parameter_initializer_range.get(field_name, default)
    return getattr(parameter_initializer_range, field_name, default)


def apply_lora(
    model: "SMLLanguageModel",
    config: LoRAConfig,
    parameter_initializer_range=None,
) -> "SMLLanguageModel":
    """
    Replace matching linear projections with ``LoRALinear`` wrappers in place.
    """
    lora_a_initializer_range = resolve_lora_initializer_range(
        parameter_initializer_range,
        "lora_a",
        0.01,
    )
    lora_b_initializer_range = resolve_lora_initializer_range(
        parameter_initializer_range,
        "lora_b",
        0.0,
    )
    target_modules: list[tuple[str, object]] = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue

        short_name = name.rsplit(".", 1)[-1]
        if short_name not in config.target_modules:
            continue

        target_modules.append((name, module))

    if not target_modules:
        return model

    model.freeze()
    for name, module in target_modules:
        parent, child_name = _split_parent(model, name)
        wrapper = LoRALinear(
            module,
            rank=config.rank,
            alpha=config.alpha,
            scaling_mode=config.scaling_mode,
            dropout=config.dropout,
            lora_a_initializer_range=lora_a_initializer_range,
            lora_b_initializer_range=lora_b_initializer_range,
        )
        wrapper.unfreeze(keys=[LORA_A_SUFFIX, LORA_B_SUFFIX], recurse=False)
        setattr(parent, child_name, wrapper)

    return model


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


def iter_lora_modules(model) -> Iterator[LoRALinear]:
    for module in model.modules():
        if isinstance(module, LoRALinear):
            yield module


def lora_parameters(model) -> list[mx.array]:
    """
    Return only LoRA adapter arrays for optimizer construction.
    """
    return [
        parameter
        for module in iter_lora_modules(model)
        for parameter in (module.lora_A, module.lora_B)
    ]


def lora_state_dict(model) -> dict[str, mx.array]:
    """
    Collect adapter arrays keyed by their location in the full model state.
    """
    state: dict[str, mx.array] = {}
    for name, module in model.named_modules():
        if not isinstance(module, LoRALinear):
            continue
        state[f"{name}.{LORA_A_SUFFIX}"] = module.lora_A
        state[f"{name}.{LORA_B_SUFFIX}"] = module.lora_B
    return state


def load_lora_state_dict(model, state: dict[str, mx.array]) -> None:
    """
    Restore adapter arrays produced by ``lora_state_dict``.
    """
    for name, module in model.named_modules():
        if not isinstance(module, LoRALinear):
            continue

        key_a = f"{name}.{LORA_A_SUFFIX}"
        key_b = f"{name}.{LORA_B_SUFFIX}"
        if key_a not in state or key_b not in state:
            raise ValueError(f"LoRA checkpoint is missing adapter weights for {name}")
        if tuple(state[key_a].shape) != tuple(module.lora_A.shape):
            raise ValueError(f"LoRA checkpoint has wrong A shape for {name}")
        if tuple(state[key_b].shape) != tuple(module.lora_B.shape):
            raise ValueError(f"LoRA checkpoint has wrong B shape for {name}")

        module.lora_A = state[key_a].astype(module.lora_A.dtype)
        module.lora_B = state[key_b].astype(module.lora_B.dtype)


def merge_lora(model) -> None:
    """
    Replace every ``LoRALinear`` wrapper with its merged base linear layer.
    """
    for name, module in list(model.named_modules()):
        if not isinstance(module, LoRALinear):
            continue

        parent, child_name = _split_parent(model, name)
        setattr(parent, child_name, module.merge())


def count_lora_modules(model) -> int:
    return sum(1 for _ in iter_lora_modules(model))


def require_lora_modules(model, target_modules: Iterable[str]) -> None:
    """
    Fail fast when no projection matched the configured target module names.
    """
    if count_lora_modules(model) == 0:
        targets = ", ".join(sorted(set(target_modules)))
        raise RuntimeError(
            f"No LoRA adapters were applied to target modules: {targets}"
        )
