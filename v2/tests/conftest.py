from __future__ import annotations

import json
import sys
from pathlib import Path

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


EQUIVALENCE_FIXTURE_DIR = PROJECT_DIR / "tests" / "equivalence" / "fixtures"


@pytest.fixture(scope="session")
def legacy_control() -> dict[str, object]:
    control_path = EQUIVALENCE_FIXTURE_DIR / "legacy-control.json"
    return json.loads(control_path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def legacy_arrays():
    loaded = mx.load(str(EQUIVALENCE_FIXTURE_DIR / "legacy-arrays.safetensors"))
    return dict(sorted(loaded.items()))


def _validate_and_load_legacy_state(
    model,
    legacy_arrays,
    mappings: list[dict[str, object]],
    *,
    namespace: str,
) -> None:
    expected = {str(mapping["destination"]): mapping for mapping in mappings}
    actual = dict(tree_flatten(model.parameters()))
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        raise ValueError(
            f"legacy state destination mismatch: missing={missing}, "
            f"unexpected={unexpected}"
        )

    weights = []
    for name in sorted(expected):
        array = legacy_arrays[f"{namespace}.{name}"]
        record = expected[name]
        if list(actual[name].shape) != record["shape"]:
            raise ValueError(f"legacy state shape mismatch for {name}")
        if str(actual[name].dtype) != record["dtype"]:
            raise ValueError(f"legacy state dtype mismatch for {name}")
        if list(array.shape) != record["shape"] or str(array.dtype) != record["dtype"]:
            raise ValueError(f"captured legacy payload metadata mismatch for {name}")
        weights.append((name, array))

    model.load_weights(weights, strict=True)
    mx.eval(model.parameters())


def load_legacy_model_state(model, legacy_arrays, legacy_control) -> None:
    _validate_and_load_legacy_state(
        model,
        legacy_arrays,
        legacy_control["parameter_state"]["mapping"],
        namespace="model_state",
    )


@pytest.fixture(name="load_legacy_model_state")
def _load_legacy_model_state_fixture():
    return load_legacy_model_state


def _package_lora_destination(name: str) -> str:
    if name.endswith(".linear.weight"):
        return f"{name[: -len('.linear.weight')]}.base.weight"
    if name.endswith(".lora_A"):
        return f"{name[: -len('.lora_A')]}.lora_a"
    if name.endswith(".lora_B"):
        return f"{name[: -len('.lora_B')]}.lora_b"
    return name


def load_legacy_lora_state(model, legacy_arrays, legacy_control) -> None:
    lora_state = legacy_control["lora_parameter_state"]
    mappings = [*lora_state["base_mapping"], *lora_state["adapter_mapping"]]
    adapter_sources = {
        str(mapping["destination"]) for mapping in lora_state["adapter_mapping"]
    }
    expected = {}
    for mapping in mappings:
        source_destination = str(mapping["destination"])
        packaged = _package_lora_destination(source_destination)
        expected[packaged] = (source_destination, mapping)
    actual = dict(tree_flatten(model.parameters()))
    scale_leaves = {
        name: value for name, value in actual.items() if name.endswith(".scale")
    }
    comparable = {
        name: value for name, value in actual.items() if name not in scale_leaves
    }
    if set(comparable) != set(expected):
        missing = sorted(set(expected) - set(comparable))
        unexpected = sorted(set(comparable) - set(expected))
        raise ValueError(
            f"legacy LoRA destination mismatch: missing={missing}, "
            f"unexpected={unexpected}"
        )

    weights = []
    for name in sorted(expected):
        source_destination, record = expected[name]
        namespace = (
            "lora_state" if source_destination in adapter_sources else "lora_base_state"
        )
        array = legacy_arrays[f"{namespace}.{source_destination}"]
        if list(actual[name].shape) != record["shape"]:
            raise ValueError(f"legacy LoRA shape mismatch for {name}")
        if str(actual[name].dtype) != record["dtype"]:
            raise ValueError(f"legacy LoRA dtype mismatch for {name}")
        if list(array.shape) != record["shape"] or str(array.dtype) != record["dtype"]:
            raise ValueError(f"captured legacy LoRA metadata mismatch for {name}")
        weights.append((name, array))
    weights.extend(sorted(scale_leaves.items()))
    model.load_weights(weights, strict=True)
    mx.eval(model.parameters())


@pytest.fixture(name="load_legacy_lora_state")
def _load_legacy_lora_state_fixture():
    return load_legacy_lora_state
