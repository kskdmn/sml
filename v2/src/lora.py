"""
Low-rank adaptation (LoRA) helpers for MLX SML fine-tuning.

Wraps selected ``nn.Linear`` layers with trainable low-rank deltas while keeping
base weights frozen. ``merge_lora`` folds adapters into base weights so merged
checkpoints load through the standard inference path.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mlx.core as mx

try:
    import mlx.nn as nn
except RuntimeError as exc:  # pragma: no cover - depends on host MLX access
    nn = None
    _MLX_IMPORT_ERROR = exc
    _ModuleBase = object
else:
    _MLX_IMPORT_ERROR = None
    _ModuleBase = nn.Module

if TYPE_CHECKING:
    from sml import SMLLanguageModel

LORA_A_SUFFIX = "lora_A"
LORA_B_SUFFIX = "lora_B"


@dataclass(slots=True)
class LoRAConfig:
    """
    Low-rank adapter settings for SWAG fine-tuning.

    Adapters attach to attention and MLP projections by module name.
    """

    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        # "gate_proj",
        # "up_proj",
        # "down_proj",
    )


def _require_mlx_nn():
    if nn is None:
        raise RuntimeError(f"mlx.nn is not available: {_MLX_IMPORT_ERROR}")
    return nn


class LoRALinear(_ModuleBase):
    def __init__(
        self,
        linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        """
        Compute ``linear(x) + scaling * (x @ A.T @ B.T)`` with base weights frozen.
        """
        mlx_nn = _require_mlx_nn()
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if alpha <= 0.0:
            raise ValueError("LoRA alpha must be positive")

        self.linear = linear
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_dropout = mlx_nn.Dropout(dropout) if dropout > 0.0 else mlx_nn.Identity()
        input_dims = linear.weight.shape[1]
        output_dims = linear.weight.shape[0]
        self.lora_A = mx.random.normal(shape=(rank, input_dims)) * 0.01
        self.lora_B = mx.zeros((output_dims, rank), dtype=linear.weight.dtype)
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


def apply_lora(model: "SMLLanguageModel", config: LoRAConfig) -> "SMLLanguageModel":
    """
    Replace matching linear projections with ``LoRALinear`` wrappers in place.
    """
    mlx_nn = _require_mlx_nn()
    target_modules: list[tuple[str, object]] = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, mlx_nn.Linear):
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
            dropout=config.dropout,
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
        raise RuntimeError(f"No LoRA adapters were applied to target modules: {targets}")
