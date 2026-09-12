"""Bind the current implementation from a verified source checkout."""

from __future__ import annotations

import importlib
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from v2.benchmarks.schema import CanonicalWorkload, JsonValue, MetricName
from v2.benchmarks.workload import (
    PRECISION_POLICY,
    canonical_execution_order,
    canonical_input_identity,
    canonical_metric_projection,
    execution_order_identity,
)


@dataclass(frozen=True, slots=True)
class NativeWorkload:
    metric: MetricName
    source_root: Path
    canonical_workload: CanonicalWorkload
    native_configuration: dict[str, JsonValue]
    native_representation_identity: str
    canonical_row_identity: str
    canonical_input_identity: str
    canonical_projection: dict[str, JsonValue]
    initial_parameter_identity: str
    startup_verification_seconds: float
    runtime: object

    @property
    def execution_order_identity(self) -> str:
        identity = self.runtime.execution_order_identity
        if identity is None:
            raise RuntimeError(
                "benchmark execution order has not been verified after measurement"
            )
        return identity


def resolve_native_workload(metric, canonical_workload, source_root):
    source_root = source_root.resolve()
    source_directory = source_root / "v2" / "src"
    sys.path.insert(0, str(source_directory))
    try:
        importlib.invalidate_caches()
        package = importlib.import_module("sml")
        if not Path(package.__file__).resolve().is_relative_to(source_directory):
            raise RuntimeError("benchmark runtime resolved outside source checkout")
        from v2.benchmarks.adapters.native import build_benchmark_workload

        started = time.perf_counter()
        runtime = build_benchmark_workload(metric, canonical_workload)
        elapsed = time.perf_counter() - started
    finally:
        sys.path.remove(str(source_directory))
    expected = {
        "canonical_row_identity": canonical_workload.semantic_identities[
            "canonical_training_rows"
        ],
        "canonical_input_identity": canonical_input_identity(
            metric, canonical_workload
        ),
        "canonical_projection": canonical_metric_projection(metric, canonical_workload),
        "verification_level": "full",
    }
    try:
        if runtime.execution_order_identity is not None:
            raise RuntimeError(
                "benchmark runtime claims execution order before measurement"
            )
        for name, value in expected.items():
            if getattr(runtime, name) != value:
                raise RuntimeError(f"benchmark runtime has invalid {name}")
        if (
            runtime.native_configuration.get("parameter_precision_policy")
            != PRECISION_POLICY
        ):
            raise RuntimeError("benchmark runtime has the wrong precision policy")
        for name in ("native_representation_identity", "initial_parameter_identity"):
            if re.fullmatch(r"sha256:[0-9a-f]{64}", getattr(runtime, name)) is None:
                raise RuntimeError(f"benchmark runtime has invalid {name}")
        return NativeWorkload(
            metric=metric,
            source_root=source_root,
            canonical_workload=canonical_workload,
            native_configuration=runtime.native_configuration,
            native_representation_identity=runtime.native_representation_identity,
            canonical_row_identity=runtime.canonical_row_identity,
            canonical_input_identity=runtime.canonical_input_identity,
            canonical_projection=runtime.canonical_projection,
            initial_parameter_identity=runtime.initial_parameter_identity,
            startup_verification_seconds=elapsed,
            runtime=runtime,
        )
    except BaseException:
        runtime.close()
        raise


def run_warmup(metric, native_workload, units):
    if metric != native_workload.metric:
        raise ValueError("metric does not match native workload")
    native_workload.runtime.run(units)
    reset = getattr(native_workload.runtime, "reset_after_warmup", None)
    if reset is not None:
        reset()


def run_measured(metric, native_workload, units):
    if metric != native_workload.metric:
        raise ValueError("metric does not match native workload")
    return float(native_workload.runtime.run(units))


def begin_measured_order(metric, native_workload):
    if metric != native_workload.metric:
        raise ValueError("metric does not match native workload")
    native_workload.runtime.execution_order_identity = None
    native_workload.runtime.reset_measured_order()


def verify_measured_order(metric, native_workload):
    if metric != native_workload.metric:
        raise ValueError("metric does not match native workload")
    native_workload.runtime.execution_order_identity = None
    observed = native_workload.runtime.observed_execution_order()
    expected = canonical_execution_order(metric, native_workload.canonical_workload)
    if observed != expected:
        raise RuntimeError(
            "benchmark observed execution order differs from canonical work"
        )
    native_workload.runtime.execution_order_identity = execution_order_identity(
        metric, native_workload.canonical_workload, observed
    )
