from __future__ import annotations

import importlib
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from v2.benchmarks.schema import CanonicalWorkload, JsonValue, MetricName
from v2.benchmarks.workload import (
    REPLACEMENT_PRECISION_POLICY,
    canonical_execution_order_identity,
    canonical_input_identity,
    canonical_metric_projection,
)

METRIC_OWNER_IMPORTS: dict[MetricName, str] = {
    "prepared-data": "sml.data.pretraining",
    "pretraining-compute": "sml.model.language_model",
    "pretraining-end-to-end": "sml.training.pretrain",
    "swag-end-to-end": "sml.training.swag",
    "inference-prefill": "sml.inference",
    "inference-decode": "sml.inference",
    "checkpoint-pause": "sml.artifacts.checkpoint",
    "compile-cold-start": "sml.model.language_model",
    "peak-metal-memory": "sml.training.pretrain",
}

METRIC_FACTORY_NAMES: dict[MetricName, str] = {
    metric: "build_benchmark_workload" for metric in METRIC_OWNER_IMPORTS
}


@dataclass(frozen=True, slots=True)
class UnavailableNativeWorkload:
    metric: MetricName
    owner_import: str
    source_root: Path
    reason: str


@dataclass(frozen=True, slots=True)
class ReplacementNativeWorkload:
    metric: MetricName
    owner_import: str
    source_root: Path
    canonical_workload: CanonicalWorkload
    native_configuration: dict[str, JsonValue]
    native_representation_identity: str
    canonical_row_identity: str
    canonical_input_identity: str
    canonical_projection: dict[str, JsonValue]
    execution_order_identity: str
    initial_parameter_identity: str
    startup_verification_seconds: float
    runtime: object


def _owner_path(source_root: Path, owner_import: str) -> Path | None:
    relative = Path(*owner_import.split("."))
    module_path = source_root / "v2" / "src" / relative.with_suffix(".py")
    package_path = source_root / "v2" / "src" / relative / "__init__.py"
    if module_path.is_file():
        return module_path
    if package_path.is_file():
        return package_path
    return None


def _import_owner(source_root: Path, owner_import: str):
    source_directory = source_root / "v2" / "src"
    source_text = str(source_directory)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    importlib.invalidate_caches()
    module = importlib.import_module(owner_import)
    module_file = Path(module.__file__).resolve()
    if not module_file.is_relative_to(source_directory.resolve()):
        raise RuntimeError(
            f"replacement owner {owner_import!r} resolved outside source checkout"
        )
    return module


def resolve_native_workload(
    metric: MetricName,
    canonical_workload: CanonicalWorkload,
    source_root: Path,
) -> ReplacementNativeWorkload | UnavailableNativeWorkload:
    owner_import = METRIC_OWNER_IMPORTS[metric]
    resolved_root = source_root.resolve()
    if _owner_path(resolved_root, owner_import) is None:
        return UnavailableNativeWorkload(
            metric=metric,
            owner_import=owner_import,
            source_root=resolved_root,
            reason="planned owner module is not present",
        )
    owner = _import_owner(resolved_root, owner_import)
    factory_name = METRIC_FACTORY_NAMES[metric]
    factory = getattr(owner, factory_name, None)
    if factory is None:
        return UnavailableNativeWorkload(
            metric=metric,
            owner_import=owner_import,
            source_root=resolved_root,
            reason=f"planned owner has not enabled {factory_name}",
        )
    verification_start = time.perf_counter()
    runtime = factory(metric, canonical_workload)
    verification_seconds = time.perf_counter() - verification_start
    required_attributes = (
        "native_configuration",
        "native_representation_identity",
        "canonical_row_identity",
        "canonical_input_identity",
        "canonical_projection",
        "execution_order_identity",
        "initial_parameter_identity",
        "verification_level",
        "run",
    )
    missing = tuple(name for name in required_attributes if not hasattr(runtime, name))
    if missing:
        raise RuntimeError(
            f"replacement benchmark runtime is missing attributes: {', '.join(missing)}"
        )
    expected_row_identity = canonical_workload.semantic_identities[
        "canonical_training_rows"
    ]
    expected_input_identity = canonical_input_identity(metric, canonical_workload)
    expected_projection = canonical_metric_projection(metric, canonical_workload)
    if runtime.verification_level != "full":
        raise RuntimeError("replacement benchmark input verification must be full")
    if runtime.canonical_row_identity != expected_row_identity:
        raise RuntimeError("replacement runtime resolved different canonical rows")
    if runtime.canonical_input_identity != expected_input_identity:
        raise RuntimeError("replacement runtime resolved different canonical inputs")
    if dict(runtime.canonical_projection) != expected_projection:
        raise RuntimeError("replacement runtime failed canonical workload round trip")
    expected_execution_order = canonical_execution_order_identity(
        metric, canonical_workload
    )
    if runtime.execution_order_identity != expected_execution_order:
        raise RuntimeError(
            "replacement runtime resolved a different logical work order"
        )
    native_configuration = dict(runtime.native_configuration)
    if (
        native_configuration.get("parameter_precision_policy")
        != REPLACEMENT_PRECISION_POLICY
    ):
        raise RuntimeError("replacement runtime has the wrong precision-policy proof")
    for name in ("native_representation_identity", "initial_parameter_identity"):
        if re.fullmatch(r"sha256:[0-9a-f]{64}", getattr(runtime, name)) is None:
            raise RuntimeError(f"replacement runtime {name} is not a SHA-256 identity")
    return ReplacementNativeWorkload(
        metric=metric,
        owner_import=owner_import,
        source_root=resolved_root,
        canonical_workload=canonical_workload,
        native_configuration=native_configuration,
        native_representation_identity=runtime.native_representation_identity,
        canonical_row_identity=runtime.canonical_row_identity,
        canonical_input_identity=runtime.canonical_input_identity,
        canonical_projection=expected_projection,
        execution_order_identity=expected_execution_order,
        initial_parameter_identity=runtime.initial_parameter_identity,
        startup_verification_seconds=verification_seconds,
        runtime=runtime,
    )


def run_warmup(
    metric: MetricName,
    native_workload: ReplacementNativeWorkload,
    units: int,
) -> None:
    if metric != native_workload.metric:
        raise ValueError("metric does not match native workload")
    native_workload.runtime.run(units)
    reset = getattr(native_workload.runtime, "reset_after_warmup", None)
    if reset is not None:
        reset()


def run_measured(
    metric: MetricName,
    native_workload: ReplacementNativeWorkload,
    units: int,
) -> float:
    if metric != native_workload.metric:
        raise ValueError("metric does not match native workload")
    reset = getattr(native_workload.runtime, "reset_measured_order", None)
    if reset is not None:
        reset()
    return float(native_workload.runtime.run(units))
