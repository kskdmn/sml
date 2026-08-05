from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np

from v2.benchmarks.schema import (
    CanonicalWorkload,
    JsonValue,
    WorkUnitDefinition,
)

HARNESS_COMPONENTS = (
    Path("v2/benchmarks/schema.py"),
    Path("v2/benchmarks/workload.py"),
    Path("v2/benchmarks/runner.py"),
    Path("v2/benchmarks/journal.py"),
    Path("v2/benchmarks/analysis.py"),
    Path("v2/benchmarks/adapters/legacy.py"),
    Path("v2/benchmarks/adapters/replacement.py"),
    Path("v2/tests/unit/test_benchmark_analysis.py"),
)

BENCHMARK_CORPUS = (
    "A quiet observatory watches the winter sky while its clock records each star.",
    "Engineers compare two careful implementations with the same ordered inputs.",
    "The river crosses granite shelves, turns through reeds, and reaches the harbor.",
    "A librarian repairs an old index so every volume can be found without guessing.",
    "Small repeatable experiments reveal which change improved the complete system.",
    "Morning trains carry bread, letters, tools, and travelers between neighboring towns.",
    "The workshop keeps measurements, materials, and instructions beside every prototype.",
    "Clouds moved east after sunset; by midnight the telescope saw a clear horizon.",
    "Each candidate ending must preserve its context, label, token mask, and exact order.",
    "A generation request owns its prompt, decoding policy, random seed, and token budget.",
    "Durable records name the code, machine, inputs, and environment that produced them.",
    "When a result is noisy, the machine cools before the complete experiment repeats.",
)
BENCHMARK_SWAG_ENDINGS = (
    "the careful procedure finishes and the result is recorded.",
    "an unrelated parade suddenly appears inside the locked workshop.",
    "the observer discards every instrument before taking a measurement.",
    "the sequence reverses itself without any cause or intervention.",
)
BENCHMARK_TOKENIZER = {
    "algorithm": "unicode-nfkc-sha256-wordpiece-v1",
    "normalization": "NFKC",
    "piece_pattern": r"\w+|[^\w\s]",
    "pad_token_id": 0,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "unk_token_id": 3,
}
LEGACY_PRECISION_POLICY = (
    "legacy BF16 persistent parameters and BF16 Adam moments without authoritative "
    "master parameters"
)
REPLACEMENT_PRECISION_POLICY = (
    "replacement FP32 authoritative master parameters and FP32 Adam moments with "
    "derived BF16 working parameters"
)


@dataclass(frozen=True, slots=True)
class FixedSwagExamples:
    example_ids: tuple[int, ...]
    input_ids: np.ndarray
    labels: np.ndarray
    candidate_labels: np.ndarray
    identity: str


@dataclass(frozen=True, slots=True)
class FixedInferenceRequests:
    request_ids: tuple[int, ...]
    prompt_ids: np.ndarray
    decode_tokens: int
    identity: str


def canonical_json_bytes(value: JsonValue) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def structured_identity(domain: str, value: JsonValue) -> str:
    digest = hashlib.sha256()
    digest.update(domain.encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical_json_bytes(value))
    return f"sha256:{digest.hexdigest()}"


def semantic_row_content_identity(rows: np.ndarray) -> str:
    array = np.asarray(rows)
    if array.ndim != 2:
        raise ValueError("canonical token rows must be a two-dimensional matrix")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError("canonical token rows must contain integers")
    canonical = np.ascontiguousarray(array, dtype=np.dtype("<i4"))
    digest = hashlib.sha256()
    digest.update(b"sml-pretraining-rows-v1\0")
    digest.update(int(canonical.shape[0]).to_bytes(8, "little", signed=False))
    digest.update(int(canonical.shape[1]).to_bytes(8, "little", signed=False))
    digest.update(canonical.tobytes(order="C"))
    return f"sha256:{digest.hexdigest()}"


def semantic_array_identity(domain: str, arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    digest.update(domain.encode("ascii"))
    digest.update(b"\0")
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        encoded_name = name.encode("utf-8")
        encoded_dtype = array.dtype.str.encode("ascii")
        digest.update(len(encoded_name).to_bytes(4, "little"))
        digest.update(encoded_name)
        digest.update(len(encoded_dtype).to_bytes(4, "little"))
        digest.update(encoded_dtype)
        digest.update(len(array.shape).to_bytes(4, "little"))
        for dimension in array.shape:
            digest.update(int(dimension).to_bytes(8, "little", signed=False))
        digest.update(array.tobytes(order="C"))
    return f"sha256:{digest.hexdigest()}"


def benchmark_tokenizer_identity(vocab_size: int) -> str:
    return structured_identity(
        "sml-benchmark-tokenizer-v1",
        {**BENCHMARK_TOKENIZER, "vocab_size": vocab_size},
    )


def source_corpus_identity() -> str:
    return structured_identity("sml-benchmark-source-corpus-v1", list(BENCHMARK_CORPUS))


def _benchmark_token_stream(vocab_size: int) -> np.ndarray:
    if vocab_size <= 4:
        raise ValueError("vocab_size must leave room for ordinary tokens")
    tokens: list[int] = []
    for document in BENCHMARK_CORPUS:
        normalized = unicodedata.normalize("NFKC", document)
        tokens.append(int(BENCHMARK_TOKENIZER["bos_token_id"]))
        tokens.extend(_benchmark_encode_text(normalized, vocab_size))
        tokens.append(int(BENCHMARK_TOKENIZER["eos_token_id"]))
    return np.asarray(tokens, dtype=np.int32)


def _benchmark_encode_text(text: str, vocab_size: int) -> list[int]:
    normalized = unicodedata.normalize("NFKC", text)
    tokens = []
    for piece in re.findall(str(BENCHMARK_TOKENIZER["piece_pattern"]), normalized):
        digest = hashlib.sha256(piece.casefold().encode("utf-8")).digest()
        tokens.append(4 + int.from_bytes(digest[:8], "little") % (vocab_size - 4))
    return tokens


def _merge_overrides(
    target: dict[str, JsonValue], overrides: dict[str, JsonValue] | None
) -> None:
    for key, value in (overrides or {}).items():
        current = target.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            _merge_overrides(current, value)
        else:
            target[key] = value


def fixed_canonical_rows(
    *,
    row_count: int = 968,
    row_width: int = 1_025,
    vocab_size: int = 28_672,
) -> np.ndarray:
    if row_count <= 0 or row_width <= 0:
        raise ValueError("canonical row dimensions must be positive")
    stream = _benchmark_token_stream(vocab_size)
    required = row_count * row_width
    repeats = (required + len(stream) - 1) // len(stream)
    return np.tile(stream, repeats)[:required].reshape((row_count, row_width)).copy()


def fixed_swag_examples(
    workload: CanonicalWorkload,
    _rows: np.ndarray,
    *,
    order: tuple[int, ...] | None = None,
) -> FixedSwagExamples:
    swag_loader = workload.loader["swag"]
    if not isinstance(swag_loader, dict):
        raise ValueError("SWAG loader configuration must be an object")
    example_count = int(swag_loader["example_count"])
    sequence_length = int(swag_loader["sequence_length"])
    example_ids = tuple(range(example_count)) if order is None else order
    if sorted(example_ids) != list(range(example_count)):
        raise ValueError("SWAG example order must be a permutation of every example")
    pad_token_id = int(workload.model["pad_token_id"])
    bos_token_id = int(workload.model["bos_token_id"])
    eos_token_id = int(workload.model["eos_token_id"])
    vocab_size = int(workload.model["vocab_size"])
    candidates = np.full(
        (example_count, 4, sequence_length), pad_token_id, dtype=np.int32
    )
    labels = np.full_like(candidates, pad_token_id)
    candidate_labels = np.empty((example_count,), dtype=np.int32)
    for output_index, example_id in enumerate(example_ids):
        context = (
            f"Benchmark scenario {example_id}. "
            f"{BENCHMARK_CORPUS[example_id % len(BENCHMARK_CORPUS)]}"
        )
        context_tokens = _benchmark_encode_text(context, vocab_size)
        for candidate_index in range(4):
            ending_tokens = _benchmark_encode_text(
                BENCHMARK_SWAG_ENDINGS[candidate_index], vocab_size
            )
            ending_tokens = ending_tokens[: max(1, sequence_length - 2)]
            context_budget = max(0, sequence_length - len(ending_tokens) - 2)
            input_tokens = [
                bos_token_id,
                *context_tokens[:context_budget],
                *ending_tokens,
                eos_token_id,
            ]
            candidates[output_index, candidate_index, : len(input_tokens)] = (
                input_tokens
            )
            ending_start = 1 + min(len(context_tokens), context_budget)
            labels[
                output_index,
                candidate_index,
                ending_start : len(input_tokens),
            ] = input_tokens[ending_start:]
        candidate_labels[output_index] = example_id % 4
    identity = semantic_array_identity(
        "sml-benchmark-swag-examples-v1",
        {
            "example_ids": np.asarray(example_ids, dtype=np.dtype("<i4")),
            "input_ids": candidates.astype(np.dtype("<i4"), copy=False),
            "labels": labels.astype(np.dtype("<i4"), copy=False),
            "candidate_labels": candidate_labels.astype(np.dtype("<i4"), copy=False),
        },
    )
    return FixedSwagExamples(
        example_ids=example_ids,
        input_ids=candidates,
        labels=labels,
        candidate_labels=candidate_labels,
        identity=identity,
    )


def fixed_inference_requests(
    workload: CanonicalWorkload,
    rows: np.ndarray,
    *,
    order: tuple[int, ...] | None = None,
) -> FixedInferenceRequests:
    request_count = int(workload.generation["request_count"])
    prompt_tokens = int(workload.generation["prompt_tokens"])
    request_ids = tuple(range(request_count)) if order is None else order
    if sorted(request_ids) != list(range(request_count)):
        raise ValueError(
            "inference request order must contain every request exactly once"
        )
    canonical = np.asarray(rows, dtype=np.int32)
    prompts = np.stack(
        [
            canonical[request_id % len(canonical), :prompt_tokens]
            for request_id in request_ids
        ]
    ).astype(np.int32, copy=False)
    policy = np.frombuffer(
        canonical_json_bytes(
            {
                key: value
                for key, value in workload.generation.items()
                if key != "request_count"
            }
        ),
        dtype=np.uint8,
    )
    identity = semantic_array_identity(
        "sml-benchmark-inference-requests-v1",
        {
            "request_ids": np.asarray(request_ids, dtype=np.dtype("<i4")),
            "prompt_ids": prompts.astype(np.dtype("<i4"), copy=False),
            "generation_policy_json": policy,
        },
    )
    return FixedInferenceRequests(
        request_ids=request_ids,
        prompt_ids=prompts,
        decode_tokens=int(workload.generation["decode_chunk_size"]),
        identity=identity,
    )


def _work_units(request_count: int) -> tuple[WorkUnitDefinition, ...]:
    definitions = (
        (
            "prepared-data",
            "higher-is-better",
            "batches",
            "one on-device microbatch",
            "before requesting the next native prepared-data batch",
            "after the batch transfer is synchronized",
        ),
        (
            "pretraining-compute",
            "higher-is-better",
            "tokens",
            "one optimizer step",
            "before the first compiled microstep",
            "after the optimizer update is synchronized",
        ),
        (
            "pretraining-end-to-end",
            "higher-is-better",
            "tokens",
            "one accumulation window and optimizer update",
            "before requesting the first microbatch",
            "after the optimizer update is synchronized",
        ),
        (
            "swag-end-to-end",
            "higher-is-better",
            "examples",
            "one fixed-shape SWAG batch",
            "before requesting the encoded batch",
            "after the adapter update is synchronized",
        ),
        (
            "inference-prefill",
            "higher-is-better",
            "tokens",
            "one fixed request batch prefill",
            "before the prefill call",
            "after prefill logits and cache state are synchronized",
        ),
        (
            "inference-decode",
            "higher-is-better",
            "tokens",
            "one fixed decode chunk",
            "before the decode chunk call",
            "after decoded tokens and cache state are synchronized",
        ),
        (
            "checkpoint-pause",
            "lower-is-better",
            "seconds",
            "one durable checkpoint replacement",
            "before checkpoint serialization",
            "after directory fsync and obsolete-step pruning",
        ),
        (
            "compile-cold-start",
            "lower-is-better",
            "seconds",
            "one cold compiled-kernel invocation",
            "before the first compiled invocation in a fresh process",
            "after its outputs are synchronized",
        ),
        (
            "peak-metal-memory",
            "lower-is-better",
            "bytes",
            "one end-to-end pretraining optimizer step",
            "after resetting the Metal peak-memory counter",
            "after the optimizer update is synchronized",
        ),
    )
    measured_units = {
        "inference-prefill": request_count,
        "inference-decode": request_count,
        "compile-cold-start": 1,
    }
    return tuple(
        WorkUnitDefinition(
            metric=metric,
            direction=direction,
            numerator=numerator,
            work_unit=work_unit,
            start_boundary=start_boundary,
            end_boundary=end_boundary,
            measured_units=measured_units.get(metric, 100),
        )
        for metric, direction, numerator, work_unit, start_boundary, end_boundary in definitions
    )


def build_canonical_workload(
    *,
    model_overrides: dict[str, JsonValue] | None = None,
    optimizer_overrides: dict[str, JsonValue] | None = None,
    loader_overrides: dict[str, JsonValue] | None = None,
    generation_overrides: dict[str, JsonValue] | None = None,
    row_count: int = 968,
) -> CanonicalWorkload:
    model: dict[str, JsonValue] = {
        "vocab_size": 28_672,
        "hidden_size": 768,
        "num_layers": 12,
        "num_q_heads": 12,
        "num_kv_heads": 3,
        "intermediate_size": 2_176,
        "original_max_position_embeddings": 1_024,
        "rope_theta": 10_000.0,
        "rope_scaling_factor": 1.0,
        "yarn_beta_fast": 32.0,
        "yarn_beta_slow": 1.0,
        "yarn_attention_factor": None,
        "yarn_mscale": None,
        "yarn_mscale_all_dim": None,
        "yarn_truncate": True,
        "rms_norm_eps": 1e-6,
        "hidden_dropout": 0.01,
        "initializer_range": 0.02,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "unk_token_id": 3,
        "tie_word_embeddings": True,
        "use_cache": True,
    }
    optimizer: dict[str, JsonValue] = {
        "name": "adamw",
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
        "bias_correction": False,
        "optimizer_weight_decay": 0.0,
        "learning_rate": 3e-4,
        "gradient_accumulation_steps": 8,
        "max_grad_norm": 1.0,
        "warmup_steps": 2_680,
        "total_steps": 268_000,
        "minimum_learning_rate_ratio": 0.1,
        "seed": 42,
        "parameter_weight_decay": {
            "embed_tokens": 0.0,
            "lm_head": 0.0,
            "rms_norm": 0.0,
            "q_proj": 0.1,
            "k_proj": 0.1,
            "v_proj": 0.1,
            "o_proj": 0.1,
            "gate_proj": 0.1,
            "up_proj": 0.1,
            "down_proj": 0.1,
            "other": 0.1,
        },
        "swag": {
            "betas": [0.9, 0.999],
            "epsilon": 1e-8,
            "bias_correction": False,
            "optimizer_weight_decay": 0.0,
            "learning_rate": 1e-4,
            "gradient_accumulation_steps": 8,
            "max_grad_norm": 1.0,
            "warmup_steps": 81,
            "total_steps": 8_192,
            "minimum_learning_rate_ratio": 0.1,
            "seed": 42,
            "lora": {
                "rank": 16,
                "alpha": 32.0,
                "scaling_mode": "rslora",
                "dropout": 0.05,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            },
            "parameter_weight_decay": {
                "embed_tokens": 0.0,
                "lm_head": 0.0,
                "rms_norm": 0.0,
                "q_proj": 0.1,
                "k_proj": 0.1,
                "v_proj": 0.1,
                "o_proj": 0.1,
                "gate_proj": 0.1,
                "up_proj": 0.1,
                "down_proj": 0.1,
                "other": 0.1,
                "lora_a": 0.0,
                "lora_b": 0.0,
            },
        },
    }
    loader: dict[str, JsonValue] = {
        "sequence_length": 1_024,
        "microbatch_size": 1,
        "row_count": row_count,
        "canonical_dtype": "int32",
        "legacy_dtype": "uint16",
        "replacement_dtype": "int32",
        "row_order": "fixed-canonical-order-v1",
        "swag": {
            "dataset_name": "allenai/swag",
            "dataset_config": "regular",
            "dataset_split": "train",
            "dataset_revision": "benchmark-fixed-examples-v1",
            "cache_format": "npz-int32-v1",
            "example_count": 128,
            "sequence_length": 256,
            "batch_size": 1,
            "shuffle_examples": False,
        },
    }
    generation: dict[str, JsonValue] = {
        "request_count": 32,
        "prompt_tokens": 128,
        "decode_chunk_size": 8,
        "temperature": 0.0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        "no_repeat_ngram_size": 0,
        "seed": 42,
        "request_order": "fixed-canonical-order-v1",
    }
    _merge_overrides(model, model_overrides)
    _merge_overrides(optimizer, optimizer_overrides)
    _merge_overrides(loader, loader_overrides)
    _merge_overrides(generation, generation_overrides)
    if "parameter_initializer_range" not in model:
        initializer_range = float(model["initializer_range"])
        residual_range = initializer_range / (2 * int(model["num_layers"])) ** 0.5
        model["parameter_initializer_range"] = {
            "embed_tokens": initializer_range,
            "lm_head": initializer_range,
            "q_proj": initializer_range,
            "k_proj": initializer_range,
            "v_proj": initializer_range,
            "o_proj": residual_range,
            "gate_proj": initializer_range,
            "up_proj": initializer_range,
            "down_proj": residual_range,
            "other": initializer_range,
        }
    vocab_size = int(model["vocab_size"])
    sequence_length = int(loader["sequence_length"])
    rows = fixed_canonical_rows(
        row_count=row_count,
        row_width=sequence_length + 1,
        vocab_size=vocab_size,
    )
    row_identity = semantic_row_content_identity(rows)
    workload = CanonicalWorkload(
        schema_version=1,
        model=model,
        optimizer=optimizer,
        precision={
            "compute_dtype": "bfloat16",
            "legacy_parameter_dtype": "bfloat16",
            "legacy_moment_dtype": "bfloat16",
            "replacement_master_parameter_dtype": "float32",
            "replacement_working_parameter_dtype": "bfloat16",
            "replacement_moment_dtype": "float32",
        },
        loader=loader,
        compilation={
            "compilation_passes": 1,
            "warmup_units": 20,
            "measured_units": 100,
            "fresh_processes": True,
            "state_reset_policy": "fresh-native-workload-per-process",
        },
        generation=generation,
        semantic_identities={
            "benchmark_tokenizer": benchmark_tokenizer_identity(vocab_size),
            "source_corpus_sample": source_corpus_identity(),
            "canonical_training_rows": row_identity,
            "initial_bf16_parameters": structured_identity(
                "sml-benchmark-initial-parameters-v1",
                {"model": model, "seed": optimizer["seed"], "dtype": "bfloat16"},
            ),
        },
        native_representation_identities={
            "legacy-prepared-data-schema": structured_identity(
                "sml-benchmark-native-representation-v1",
                {"format": "npz", "dtype": "uint16", "row_identity": row_identity},
            ),
            "replacement-prepared-data-schema": structured_identity(
                "sml-benchmark-native-representation-v1",
                {"format": "npy", "dtype": "int32", "row_identity": row_identity},
            ),
            "legacy-swag-cache-schema": structured_identity(
                "sml-benchmark-native-representation-v1",
                {
                    "format": "npz",
                    "dtype": "int32",
                    "layout": "example,candidate,token",
                },
            ),
            "replacement-swag-cache-schema": structured_identity(
                "sml-benchmark-native-representation-v1",
                {
                    "format": "npy-buckets",
                    "dtype": "int32",
                    "layout": "example,candidate,token",
                },
            ),
            "legacy-inference-request-schema": structured_identity(
                "sml-benchmark-native-representation-v1",
                {"format": "mlx-arrays", "dtype": "int32", "layout": "request,token"},
            ),
            "replacement-inference-request-schema": structured_identity(
                "sml-benchmark-native-representation-v1",
                {
                    "format": "request-batches",
                    "dtype": "int32",
                    "layout": "request,token",
                },
            ),
        },
        work_units=_work_units(int(generation["request_count"])),
        synchronization_boundaries=(
            "mlx.core.synchronize immediately before every timed region",
            "mlx.core.synchronize immediately after every timed region",
        ),
        required_environment={
            "chip": "Apple M5",
            "cpu_cores": 10,
            "gpu_cores": 10,
            "unified_memory_bytes": 24 * 1024**3,
            "power_connected": True,
            "power_mode": "automatic",
            "low_power_mode": False,
            "thermal_state": "nominal",
            "memory_pressure": "normal",
            "competing_gpu_workload": False,
        },
        software_requirements={
            "python": "3.12.13",
            "mlx": ">=0.32.0,<1.0.0",
            "numpy": ">=2.4.6,<3.0.0",
            "sentencepiece": ">=0.2.1,<1.0.0",
        },
    )
    swag = fixed_swag_examples(workload, rows)
    requests = fixed_inference_requests(workload, rows)
    return replace(
        workload,
        semantic_identities={
            **workload.semantic_identities,
            "canonical_swag_examples": swag.identity,
            "canonical_inference_requests": requests.identity,
        },
    )


def canonical_input_identity(metric: str, workload: CanonicalWorkload) -> str:
    if metric == "swag-end-to-end":
        return workload.semantic_identities["canonical_swag_examples"]
    if metric in ("inference-prefill", "inference-decode"):
        return workload.semantic_identities["canonical_inference_requests"]
    return workload.semantic_identities["canonical_training_rows"]


def canonical_execution_order(
    metric: str, workload: CanonicalWorkload
) -> tuple[int, ...]:
    measured_units = next(
        unit.measured_units for unit in workload.work_units if unit.metric == metric
    )
    if metric == "prepared-data":
        work_count = measured_units * int(workload.loader["microbatch_size"])
        modulus = int(workload.loader["row_count"])
    elif metric in (
        "pretraining-compute",
        "pretraining-end-to-end",
        "peak-metal-memory",
    ):
        work_count = (
            measured_units
            * int(workload.optimizer["gradient_accumulation_steps"])
            * int(workload.loader["microbatch_size"])
        )
        modulus = int(workload.loader["row_count"])
    elif metric == "swag-end-to-end":
        optimizer = workload.optimizer["swag"]
        loader = workload.loader["swag"]
        if not isinstance(optimizer, dict) or not isinstance(loader, dict):
            raise ValueError("SWAG configuration must be objects")
        work_count = (
            measured_units
            * int(optimizer["gradient_accumulation_steps"])
            * int(loader["batch_size"])
        )
        modulus = int(loader["example_count"])
    elif metric in ("inference-prefill", "inference-decode"):
        work_count = measured_units
        modulus = int(workload.generation["request_count"])
    else:
        work_count = measured_units
        modulus = measured_units
    return tuple(index % modulus for index in range(work_count))


def canonical_execution_order_identity(metric: str, workload: CanonicalWorkload) -> str:
    return structured_identity(
        "sml-benchmark-execution-order-v1",
        {
            "metric": metric,
            "canonical_input_identity": canonical_input_identity(metric, workload),
            "ordered_work_ids": list(canonical_execution_order(metric, workload)),
        },
    )


def canonical_metric_projection(
    metric: str, workload: CanonicalWorkload
) -> dict[str, JsonValue]:
    work_unit = next(unit for unit in workload.work_units if unit.metric == metric)
    return {
        "metric": metric,
        "model": workload.model,
        "optimizer": workload.optimizer,
        "precision": workload.precision,
        "loader": workload.loader,
        "compilation": workload.compilation,
        "generation": workload.generation,
        "canonical_input_identity": canonical_input_identity(metric, workload),
        "canonical_execution_order_identity": canonical_execution_order_identity(
            metric, workload
        ),
        "initial_parameter_specification_identity": workload.semantic_identities[
            "initial_bf16_parameters"
        ],
        "work_unit": {
            "numerator": work_unit.numerator,
            "work_unit": work_unit.work_unit,
            "start_boundary": work_unit.start_boundary,
            "end_boundary": work_unit.end_boundary,
            "measured_units": work_unit.measured_units,
        },
    }


def canonical_workload_identity(workload: CanonicalWorkload) -> str:
    return structured_identity(
        "sml-canonical-benchmark-workload-v1", workload.to_dict()
    )


def harness_content_identity(root: Path) -> str:
    digest = hashlib.sha256()
    for relative_path in HARNESS_COMPONENTS:
        path = root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"missing harness identity component: {path}")
        digest.update(path.read_bytes())
    return f"sha256:{digest.hexdigest()}"


def file_identity(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def write_paired_pretraining_representations(
    rows: np.ndarray,
    output: Path,
) -> dict[str, JsonValue]:
    canonical = np.ascontiguousarray(rows, dtype=np.int32)
    if canonical.ndim != 2:
        raise ValueError("benchmark pretraining rows must be two-dimensional")
    if canonical.size and (int(canonical.min()) < 0 or int(canonical.max()) > 65_535):
        raise ValueError("legacy uint16 representation cannot encode token rows")
    output.mkdir(parents=True, exist_ok=True)
    legacy_path = output / "legacy-pretraining.npz"
    replacement_path = output / "replacement-pretraining.npy"
    np.savez(legacy_path, tokens=canonical.astype(np.uint16))
    np.save(replacement_path, canonical, allow_pickle=False)
    row_identity = semantic_row_content_identity(canonical)
    with np.load(legacy_path) as archive:
        if semantic_row_content_identity(archive["tokens"]) != row_identity:
            raise RuntimeError("legacy representation changed canonical rows")
    if semantic_row_content_identity(np.load(replacement_path)) != row_identity:
        raise RuntimeError("replacement representation changed canonical rows")
    return {
        "canonical_row_identity": row_identity,
        "row_count": int(canonical.shape[0]),
        "row_width": int(canonical.shape[1]),
        "legacy_format": "npz",
        "legacy_dtype": "uint16",
        "legacy_file_identity": file_identity(legacy_path),
        "legacy_byte_size": legacy_path.stat().st_size,
        "replacement_format": "npy",
        "replacement_dtype": "int32",
        "replacement_file_identity": file_identity(replacement_path),
        "replacement_byte_size": replacement_path.stat().st_size,
    }


def ordered_file_identity(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return f"sha256:{digest.hexdigest()}"
