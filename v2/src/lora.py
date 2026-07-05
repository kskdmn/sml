"""
Low-rank adaptation (LoRA) helpers for SML fine-tuning.

Wraps selected ``nn.Linear`` layers with trainable low-rank deltas while keeping
base weights frozen. ``merge_lora`` folds adapters into the base weights so
merged checkpoints load through the standard inference path.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

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


class LoRALinear(nn.Module):
    def __init__(
        self,
        linear: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        """
        Compute ``linear(x) + scaling * (x @ A.T @ B.T)`` with base weights frozen.
        """
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if alpha <= 0.0:
            raise ValueError("LoRA alpha must be positive")

        self.linear = linear
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        device = linear.weight.device
        self.lora_A = nn.Parameter(torch.empty(rank, linear.in_features, device=device))
        self.lora_B = nn.Parameter(torch.empty(linear.out_features, rank, device=device))
        self.reset_lora_parameters()

        for parameter in self.linear.parameters():
            parameter.requires_grad = False

    def reset_lora_parameters(self) -> None:
        """
        Initialize A with Kaiming uniform noise and B with zeros (standard LoRA init).
        """
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.linear(x)
        # Outer autocast can keep matmul operands in bf16 even after .float().
        with torch.autocast(device_type=x.device.type, enabled=False):
            lora_x = self.lora_dropout(x).to(dtype=self.lora_A.dtype)
            lora_output = lora_x @ self.lora_A.T @ self.lora_B.T
        return output + lora_output.to(dtype=output.dtype) * self.scaling

    def merge(self) -> nn.Linear:
        """
        Fold the low-rank delta into the wrapped linear layer and return it.
        """
        delta = self.lora_B @ self.lora_A
        self.linear.weight.data.add_(self.scaling * delta.to(dtype=self.linear.weight.dtype))
        return self.linear


def apply_lora(model: SMLLanguageModel, config: LoRAConfig) -> SMLLanguageModel:
    """
    Replace matching linear projections with ``LoRALinear`` wrappers in place.
    """
    target_modules: list[tuple[str, nn.Linear]] = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue

        short_name = name.rsplit(".", 1)[-1]
        if short_name not in config.target_modules:
            continue

        target_modules.append((name, module))

    if not target_modules:
        return model

    for parameter in model.parameters():
        parameter.requires_grad = False

    for name, module in target_modules:
        parent_name, child_name = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        setattr(
            parent,
            child_name,
            LoRALinear(
                module,
                rank=config.rank,
                alpha=config.alpha,
                dropout=config.dropout,
            ),
        )

    return model


def iter_lora_modules(model: nn.Module) -> Iterator[LoRALinear]:
    for module in model.modules():
        if isinstance(module, LoRALinear):
            yield module


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    """
    Return only LoRA adapter parameters for optimizer construction.
    """
    return [
        parameter
        for module in iter_lora_modules(model)
        for parameter in (module.lora_A, module.lora_B)
    ]


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """
    Collect adapter tensors keyed by their location in the full model state dict.
    """
    state: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if not isinstance(module, LoRALinear):
            continue
        state[f"{name}.{LORA_A_SUFFIX}"] = module.lora_A.detach().cpu()
        state[f"{name}.{LORA_B_SUFFIX}"] = module.lora_B.detach().cpu()
    return state


def load_lora_state_dict(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    """
    Restore adapter tensors produced by ``lora_state_dict``.
    """
    for name, module in model.named_modules():
        if not isinstance(module, LoRALinear):
            continue

        key_a = f"{name}.{LORA_A_SUFFIX}"
        key_b = f"{name}.{LORA_B_SUFFIX}"
        if key_a not in state or key_b not in state:
            raise ValueError(f"LoRA checkpoint is missing adapter weights for {name}")

        module.lora_A.data.copy_(state[key_a])
        module.lora_B.data.copy_(state[key_b])


def merge_lora(model: nn.Module) -> None:
    """
    Replace every ``LoRALinear`` wrapper with its merged base linear layer.
    """
    for name, module in list(model.named_modules()):
        if not isinstance(module, LoRALinear):
            continue

        parent_name, child_name = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        setattr(parent, child_name, module.merge())


def count_lora_modules(model: nn.Module) -> int:
    return sum(1 for _ in iter_lora_modules(model))


def require_lora_modules(model: nn.Module, target_modules: Iterable[str]) -> None:
    """
    Fail fast when no projection matched the configured target module names.
    """
    if count_lora_modules(model) == 0:
        targets = ", ".join(sorted(set(target_modules)))
        raise RuntimeError(f"No LoRA adapters were applied to target modules: {targets}")
