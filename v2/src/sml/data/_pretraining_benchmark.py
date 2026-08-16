"""Pinned benchmark bridge for the production prepared-data loader."""

from __future__ import annotations

import atexit
import importlib
import math
import threading
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from sml.artifacts.checkpoint import publish_immutable_bundle
from sml.artifacts.manifest import (
    PayloadRef,
    PretrainingDataManifest,
    TokenizerManifest,
    VerificationLevel,
    canonical_json_bytes,
    read_manifest,
    row_content_identity,
)
from sml.data.pretraining import (
    PreparedDataBundle,
    PretrainingBatchStream,
    PretrainingCursor,
    _payload_ref,
    _write_shard,
)
from sml.errors import SMLArtifactError
from v2.benchmarks.schema import CanonicalWorkload
from v2.benchmarks.workload import (
    REPLACEMENT_PRECISION_POLICY,
    canonical_execution_order_identity,
    canonical_input_identity,
    canonical_metric_projection,
    fixed_canonical_rows,
    semantic_row_content_identity,
    structured_identity,
)

_BENCHMARK_METRIC = "prepared-data"
_BENCHMARK_SEED = 0
_PREFETCH_DEPTH = 2
_PLACEHOLDER_IDENTITY = "sha256:" + "0" * 64
_MODEL_BYTES = b"sml-benchmark-tokenizer-model-v1\n"
_VOCAB_BYTES = b"sml-benchmark-tokenizer-vocab-v1\n"
_INT32 = np.dtype("<i4")

_PROJECTION_FIELDS = {
    "metric",
    "model",
    "optimizer",
    "precision",
    "loader",
    "compilation",
    "generation",
    "canonical_input_identity",
    "canonical_execution_order_identity",
    "initial_parameter_specification_identity",
    "work_unit",
}


def _require_plain_positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _require_plain_nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _prepared_projection(
    canonical_workload: object,
) -> tuple[CanonicalWorkload, dict[str, object]]:
    if not isinstance(canonical_workload, CanonicalWorkload):
        raise TypeError("canonical_workload must be a CanonicalWorkload")
    try:
        projection = canonical_metric_projection(_BENCHMARK_METRIC, canonical_workload)
    except (KeyError, StopIteration, TypeError, ValueError) as error:
        raise ValueError(
            "canonical_workload must contain the exact prepared-data projection"
        ) from error
    if set(projection) != _PROJECTION_FIELDS:
        raise ValueError(
            "canonical_workload must contain the exact prepared-data projection"
        )
    if projection["metric"] != _BENCHMARK_METRIC:
        raise ValueError("canonical workload projection must be prepared-data")
    for name in (
        "model",
        "optimizer",
        "precision",
        "loader",
        "compilation",
        "generation",
        "work_unit",
    ):
        if not isinstance(projection[name], dict):
            raise TypeError(f"prepared-data projection {name} must be an object")
    return canonical_workload, projection


def _projection_integer(
    projection: dict[str, object],
    section: str,
    name: str,
    *,
    positive: bool,
) -> int:
    values = projection[section]
    if not isinstance(values, dict) or name not in values:
        raise ValueError(f"prepared-data projection is missing {section}.{name}")
    if positive:
        return _require_plain_positive_int(values[name], name)
    return _require_plain_nonnegative_int(values[name], name)


def _tokenizer_payload_reference(reference: PayloadRef) -> PayloadRef:
    return replace(reference, logical_path=f"tokenizer/{reference.logical_path}")


def _write_benchmark_tokenizer(
    private_path: Path,
    *,
    canonical_workload: CanonicalWorkload,
    projection: dict[str, object],
    vocab_size: int,
    special_ids: dict[str, int],
) -> tuple[TokenizerManifest, PayloadRef, PayloadRef]:
    tokenizer_path = private_path / "tokenizer"
    tokenizer_path.mkdir()
    model_path = tokenizer_path / "tokenizer.model"
    vocab_path = tokenizer_path / "tokenizer.vocab"
    model_path.write_bytes(_MODEL_BYTES)
    vocab_path.write_bytes(_VOCAB_BYTES)
    model = _payload_ref(model_path, "tokenizer.model")
    vocab = _payload_ref(vocab_path, "tokenizer.vocab")
    manifest = TokenizerManifest(
        kind="tokenizer",
        version=1,
        identity=_PLACEHOLDER_IDENTITY,
        algorithm="benchmark-fixed-canonical-tokenizer-v1",
        training={
            "kind": "pinned-canonical-benchmark-tokenizer",
            "semantic_identity": canonical_workload.semantic_identities[
                "benchmark_tokenizer"
            ],
            "canonical_projection_identity": structured_identity(
                "sml-benchmark-tokenizer-projection-v1",
                {
                    "vocab_size": vocab_size,
                    **special_ids,
                    "canonical_input_identity": projection["canonical_input_identity"],
                },
            ),
        },
        vocab_size=vocab_size,
        bos_token_id=special_ids["bos_token_id"],
        eos_token_id=special_ids["eos_token_id"],
        pad_token_id=special_ids["pad_token_id"],
        unk_token_id=special_ids["unk_token_id"],
        model=model,
        vocab=vocab,
        diagnostic_source_locator=None,
    )
    manifest = replace(manifest, identity=manifest.recompute_identity())
    tokenizer_path.joinpath("manifest.json").write_bytes(canonical_json_bytes(manifest))
    return (
        manifest,
        _tokenizer_payload_reference(model),
        _tokenizer_payload_reference(vocab),
    )


def _materialize_bundle(
    canonical_workload: object,
    output: Path,
) -> PreparedDataBundle:
    workload, projection = _prepared_projection(canonical_workload)
    if not isinstance(output, Path):
        raise TypeError("output must be a Path")

    row_count = _projection_integer(projection, "loader", "row_count", positive=True)
    sequence_length = _projection_integer(
        projection, "loader", "sequence_length", positive=True
    )
    row_width = sequence_length + 1
    vocab_size = _projection_integer(projection, "model", "vocab_size", positive=True)
    special_ids = {
        name: _projection_integer(projection, "model", name, positive=False)
        for name in (
            "bos_token_id",
            "eos_token_id",
            "pad_token_id",
            "unk_token_id",
        )
    }
    if len(set(special_ids.values())) != len(special_ids):
        raise ValueError("special token IDs must be unique")
    if any(token_id >= vocab_size for token_id in special_ids.values()):
        raise ValueError("special token IDs must be smaller than vocab_size")

    loader = projection["loader"]
    if not isinstance(loader, dict):
        raise TypeError("prepared-data projection loader must be an object")
    if loader.get("canonical_dtype") != "int32":
        raise ValueError("prepared-data canonical_dtype must be int32")
    if loader.get("replacement_dtype") != "int32":
        raise ValueError("prepared-data replacement_dtype must be int32")
    if loader.get("row_order") != "fixed-canonical-order-v1":
        raise ValueError("prepared-data row_order must be fixed-canonical-order-v1")

    rows = fixed_canonical_rows(
        row_count=row_count,
        row_width=row_width,
        vocab_size=vocab_size,
    )
    rows = np.ascontiguousarray(rows, dtype=_INT32)
    if rows.shape != (row_count, row_width):
        raise ValueError("canonical prepared-data rows have the wrong shape")
    canonical_row_identity = semantic_row_content_identity(rows)
    expected_identity = workload.semantic_identities.get("canonical_training_rows")
    if canonical_row_identity != expected_identity:
        raise ValueError("canonical prepared-data row identity mismatch")
    product_row_identity = row_content_identity(rows, row_count, row_width)

    def build(private_path: Path) -> PretrainingDataManifest:
        tokenizer, tokenizer_model, tokenizer_vocab = _write_benchmark_tokenizer(
            private_path,
            canonical_workload=workload,
            projection=projection,
            vocab_size=vocab_size,
            special_ids=special_ids,
        )
        shard_directory = private_path / "shards"
        shard_directory.mkdir()
        shard_path = shard_directory / "train-000000.npy"
        _write_shard(shard_path, rows)
        shard = _payload_ref(shard_path, "shards/train-000000.npy")
        manifest = PretrainingDataManifest(
            kind="pretraining-data",
            version=1,
            identity=_PLACEHOLDER_IDENTITY,
            sequence_length=sequence_length,
            row_width=row_width,
            dtype="int32",
            shard_row_counts=(row_count,),
            shards=(shard,),
            preparation_seed=_BENCHMARK_SEED,
            row_order_policy={
                "algorithm": "benchmark-fixed-canonical-order-v1",
                "output_shard_rows": row_count,
            },
            tokenizer_identity=tokenizer.identity,
            tokenizer_model=tokenizer_model,
            tokenizer_vocab=tokenizer_vocab,
            source_summary={
                "kind": "pinned-canonical-benchmark-rows",
                "row_count": row_count,
            },
            diagnostic_source_locator=None,
            row_content_identity=product_row_identity,
        )
        return replace(manifest, identity=manifest.recompute_identity())

    published = publish_immutable_bundle(output, build)
    verified = read_manifest(
        output,
        PretrainingDataManifest,
        VerificationLevel.FULL,
    )
    if verified.manifest != published.manifest:
        raise SMLArtifactError(
            "verified benchmark bundle manifest does not match publication"
        )
    if verified.manifest.row_content_identity != row_content_identity(
        rows, row_count, row_width
    ):
        raise SMLArtifactError("verified benchmark bundle row identity mismatch")
    return PreparedDataBundle(
        path=output,
        manifest=verified.manifest,
        verification=verified.verification,
    )


class _PreparedDataBenchmarkRuntime:
    def __init__(
        self,
        canonical_workload: CanonicalWorkload,
        *,
        batch_size: int,
    ) -> None:
        self._batch_size = _require_plain_positive_int(batch_size, "batch_size")
        self._operation_lock = threading.RLock()
        self._state_lock = threading.Lock()
        self._closed = False
        self._prepared_stream: PretrainingBatchStream | None = None
        self._bundle: PreparedDataBundle | None = None
        self._temporary_directory = TemporaryDirectory(
            prefix="sml-prepared-data-benchmark-"
        )
        self._temporary_root = Path(self._temporary_directory.name)
        self._atexit_callback = self.close
        atexit.register(self._atexit_callback)
        try:
            self._bundle = _materialize_bundle(
                canonical_workload,
                self._temporary_root / "bundle",
            )
            self._mx = importlib.import_module("mlx.core")
            self._prepared_stream = self._new_initial_stream()
        except BaseException:
            self.close()
            raise

    @property
    def bundle(self) -> PreparedDataBundle:
        bundle = self._bundle
        if bundle is None:
            raise RuntimeError("prepared-data benchmark bundle is unavailable")
        return bundle

    @property
    def temporary_root(self) -> Path:
        return self._temporary_root

    def _new_initial_stream(self) -> PretrainingBatchStream:
        return PretrainingBatchStream(
            self.bundle,
            batch_size=self._batch_size,
            seed=_BENCHMARK_SEED,
            prefetch_depth=_PREFETCH_DEPTH,
            cursor=PretrainingCursor.initial(),
        )

    def _take_prepared_stream(self) -> PretrainingBatchStream:
        with self._state_lock:
            if self._closed:
                raise RuntimeError("prepared-data benchmark runtime is closed")
            stream = self._prepared_stream
            if stream is None:
                raise RuntimeError("prepared-data benchmark stream is not prepared")
            self._prepared_stream = None
            return stream

    def run(self, units: int) -> float:
        units = _require_plain_positive_int(units, "units")
        with self._operation_lock:
            stream = self._take_prepared_stream()
            try:
                for _ in range(units):
                    try:
                        envelope = next(stream)
                    except StopIteration as error:
                        raise RuntimeError(
                            "prepared-data stream exhausted before requested units"
                        ) from error
                    try:
                        device_rows = self._mx.array(envelope.rows)
                    finally:
                        envelope.release()
                    input_ids = device_rows[:, :-1]
                    labels = device_rows[:, 1:]
                    self._mx.eval(input_ids, labels)
                return float(units)
            finally:
                stream.close()

    def reset_after_warmup(self) -> None:
        with self._operation_lock:
            with self._state_lock:
                if self._closed:
                    raise RuntimeError("prepared-data benchmark runtime is closed")
                stream = self._prepared_stream
                self._prepared_stream = None
            if stream is not None:
                stream.close()
            prepared = self._new_initial_stream()
            with self._state_lock:
                if self._closed:
                    prepared.close()
                    raise RuntimeError("prepared-data benchmark runtime is closed")
                self._prepared_stream = prepared

    def reset_measured_order(self) -> None:
        with self._operation_lock, self._state_lock:
            if self._closed:
                raise RuntimeError("prepared-data benchmark runtime is closed")
            stream = self._prepared_stream
            if stream is None or stream.committed_cursor != PretrainingCursor.initial():
                raise RuntimeError(
                    "prepared-data measured stream is not ready at canonical order"
                )

    def close(self) -> None:
        with self._operation_lock:
            with self._state_lock:
                if self._closed:
                    return
                self._closed = True
                stream = self._prepared_stream
                self._prepared_stream = None
            try:
                if stream is not None:
                    stream.close()
            finally:
                atexit.unregister(self._atexit_callback)
                self._temporary_directory.cleanup()


def build_prepared_data_benchmark_workload(
    metric: str,
    canonical_workload: object,
) -> object:
    if metric != _BENCHMARK_METRIC:
        raise ValueError(f"benchmark metric must be {_BENCHMARK_METRIC!r}")
    workload, projection = _prepared_projection(canonical_workload)
    sequence_length = _projection_integer(
        projection, "loader", "sequence_length", positive=True
    )
    batch_size = _projection_integer(
        projection, "loader", "microbatch_size", positive=True
    )
    gradient_accumulation_steps = _projection_integer(
        projection,
        "optimizer",
        "gradient_accumulation_steps",
        positive=True,
    )
    model = projection["model"]
    if not isinstance(model, dict):
        raise TypeError("prepared-data projection model must be an object")
    rope_scaling_factor = model.get("rope_scaling_factor")
    if (
        isinstance(rope_scaling_factor, bool)
        or not isinstance(rope_scaling_factor, (int, float))
        or not math.isfinite(float(rope_scaling_factor))
        or float(rope_scaling_factor) <= 0.0
    ):
        raise ValueError("rope_scaling_factor must be a positive finite number")

    runtime = _PreparedDataBenchmarkRuntime(workload, batch_size=batch_size)
    try:
        canonical_row_identity = workload.semantic_identities["canonical_training_rows"]
        input_identity = canonical_input_identity(_BENCHMARK_METRIC, workload)
        execution_identity = canonical_execution_order_identity(
            _BENCHMARK_METRIC, workload
        )
        runtime.native_configuration = {
            "metric": "prepared-data",
            "native_input_format": "prepared-data-bundle-npy-int32",
            "parameter_dtype": "bfloat16",
            "moment_dtype": "float32",
            "master_parameters": True,
            "rope_scaling_factor": float(
                canonical_workload.model["rope_scaling_factor"]
            ),
            "sequence_length": sequence_length,
            "microbatch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "canonical_row_identity": canonical_row_identity,
            "canonical_input_identity": input_identity,
            "canonical_projection_identity": structured_identity(
                "sml-benchmark-metric-projection-v1",
                canonical_metric_projection("prepared-data", canonical_workload),
            ),
            "canonical_execution_order_identity": execution_identity,
            "parameter_precision_policy": REPLACEMENT_PRECISION_POLICY,
        }
        runtime.native_representation_identity = runtime.bundle.manifest.identity
        runtime.canonical_row_identity = canonical_row_identity
        runtime.canonical_input_identity = input_identity
        runtime.canonical_projection = canonical_metric_projection(
            _BENCHMARK_METRIC, workload
        )
        runtime.execution_order_identity = execution_identity
        runtime.initial_parameter_identity = workload.semantic_identities[
            "initial_bf16_parameters"
        ]
        runtime.verification_level = "full"
        return runtime
    except BaseException:
        runtime.close()
        raise


__all__ = ["build_prepared_data_benchmark_workload"]
