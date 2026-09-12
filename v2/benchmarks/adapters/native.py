"""Harness-owned bindings for current production operations."""

from __future__ import annotations

import atexit
from pathlib import Path
from tempfile import TemporaryDirectory

from v2.benchmarks.workload import (
    PRECISION_POLICY,
    canonical_input_identity,
    canonical_metric_projection,
    structured_identity,
)


class NativeRuntime:
    def __init__(self, metric, workload):
        expected_precision = {
            "compute_dtype": "bfloat16",
            "master_parameter_dtype": "float32",
            "working_parameter_dtype": "bfloat16",
            "moment_dtype": "float32",
        }
        if dict(workload.precision) != expected_precision:
            raise ValueError(
                "native benchmark requires the canonical precision contract"
            )
        if workload.optimizer["name"] != "adamw":
            raise ValueError("native benchmark requires AdamW")
        self._directory = TemporaryDirectory(prefix=f"sml-native-{metric}-")
        self._runtime = None
        self._closed = False
        try:
            if metric in {"inference-prefill", "inference-decode"}:
                from v2.benchmarks.adapters.native_inference import make_runtime
            elif metric == "swag-end-to-end":
                from v2.benchmarks.adapters.native_swag import make_runtime
            else:
                from v2.benchmarks.adapters.native_training import make_runtime
            self._runtime = make_runtime(metric, workload, Path(self._directory.name))
            self.canonical_projection = canonical_metric_projection(metric, workload)
            self.native_configuration = {
                "metric": metric,
                "parameter_precision_policy": PRECISION_POLICY,
                "canonical_projection": self.canonical_projection,
                "canonical_projection_identity": structured_identity(
                    "sml-benchmark-metric-projection-v1", self.canonical_projection
                ),
                "rope_scaling_factor": float(workload.model["rope_scaling_factor"]),
                "implementation": "production-explicit-state-kernels-v1",
            }
            self.native_representation_identity = (
                self._runtime.native_representation_identity
            )
            self.initial_parameter_identity = self._runtime.initial_parameter_identity
            self.canonical_row_identity = workload.semantic_identities[
                "canonical_training_rows"
            ]
            self.canonical_input_identity = canonical_input_identity(metric, workload)
            if self._runtime.canonical_input_identity != self.canonical_input_identity:
                raise ValueError("native benchmark input verification failed")
            self.execution_order_identity = None
            self.verification_level = "full"
        except BaseException:
            self.close()
            raise
        atexit.register(self.close)

    def __getattr__(self, name):
        if self._runtime is None:
            raise AttributeError(name)
        return getattr(self._runtime, name)

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self._runtime is not None:
                close = getattr(self._runtime, "close", None)
                if close is not None:
                    close()
        finally:
            self._directory.cleanup()
            atexit.unregister(self.close)


def build_benchmark_workload(metric, workload):
    if metric == "prepared-data":
        from v2.benchmarks.adapters.prepared_data import (
            build_prepared_data_benchmark_workload,
        )

        return build_prepared_data_benchmark_workload(metric, workload)
    return NativeRuntime(metric, workload)
