from __future__ import annotations

# Canonical artifact validation reports malformed content uniformly as ValueError.
# ruff: noqa: TRY004
import hashlib
import importlib
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from v2.benchmarks.schema import CanonicalWorkload, JsonValue, MetricName
from v2.benchmarks.workload import (
    LEGACY_PRECISION_POLICY,
    canonical_execution_order_identity,
    canonical_input_identity,
    canonical_metric_projection,
    file_identity,
    fixed_canonical_rows,
    fixed_inference_requests,
    fixed_swag_examples,
    semantic_array_identity,
    semantic_row_content_identity,
    structured_identity,
)


class _NativeRuntime:
    def run(self, units: int) -> float:
        raise NotImplementedError

    def reset_after_warmup(self) -> None:
        return None

    def reset_measured_order(self) -> None:
        return None

    @property
    def initial_parameter_identity(self) -> str | None:
        return None


@dataclass(frozen=True, slots=True)
class LegacyNativeWorkload:
    metric: MetricName
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
    runtime: _NativeRuntime
    temporary_directory: tempfile.TemporaryDirectory[str]


def _legacy_import(source_root: Path, module_name: str):
    source_directory = source_root / "v2" / "src"
    source_text = str(source_directory)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    importlib.invalidate_caches()
    module = importlib.import_module(module_name)
    module_file = Path(module.__file__).resolve()
    if not module_file.is_relative_to(source_directory.resolve()):
        raise RuntimeError(
            f"legacy module {module_name!r} resolved outside source checkout: "
            f"{module_file}"
        )
    return module


def _write_and_verify_legacy_rows(
    workload: CanonicalWorkload,
    directory: Path,
) -> tuple[np.ndarray, Path, str, str, float]:
    sequence_length = int(workload.loader["sequence_length"])
    vocab_size = int(workload.model["vocab_size"])
    row_count = int(workload.loader["row_count"])
    rows = fixed_canonical_rows(
        row_count=row_count,
        row_width=sequence_length + 1,
        vocab_size=vocab_size,
    )
    expected_identity = workload.semantic_identities["canonical_training_rows"]
    row_identity = semantic_row_content_identity(rows)
    if row_identity != expected_identity:
        raise ValueError("canonical training-row identity does not match workload")

    path = directory / "legacy-pretraining.npz"
    np.savez(path, tokens=rows.astype(np.uint16))
    representation_identity = file_identity(path)
    verification_start = time.perf_counter()
    with np.load(path) as archive:
        verified = np.asarray(archive["tokens"], dtype=np.int32)
        verified_identity = semantic_row_content_identity(verified)
    verification_seconds = time.perf_counter() - verification_start
    if verified_identity != row_identity:
        raise ValueError("legacy prepared-data verification changed canonical rows")
    return rows, path, representation_identity, row_identity, verification_seconds


def _write_and_verify_swag_examples(
    workload: CanonicalWorkload,
    rows: np.ndarray,
    directory: Path,
) -> tuple[Path, str, float]:
    examples = fixed_swag_examples(workload, rows)
    if examples.identity != workload.semantic_identities["canonical_swag_examples"]:
        raise ValueError("canonical SWAG identity does not match workload")
    path = directory / "legacy-swag-cache.npz"
    np.savez(
        path,
        example_ids=np.asarray(examples.example_ids, dtype=np.int32),
        input_ids=examples.input_ids,
        labels=examples.labels,
        candidate_labels=examples.candidate_labels,
    )
    verification_start = time.perf_counter()
    with np.load(path) as archive:
        identity = semantic_array_identity(
            "sml-benchmark-swag-examples-v1",
            {
                "example_ids": archive["example_ids"],
                "input_ids": archive["input_ids"],
                "labels": archive["labels"],
                "candidate_labels": archive["candidate_labels"],
            },
        )
    verification_seconds = time.perf_counter() - verification_start
    if identity != examples.identity:
        raise ValueError("legacy SWAG cache changed canonical examples")
    return path, file_identity(path), verification_seconds


def _write_and_verify_inference_requests(
    workload: CanonicalWorkload,
    rows: np.ndarray,
    directory: Path,
) -> tuple[np.ndarray, str, float]:
    requests = fixed_inference_requests(workload, rows)
    if (
        requests.identity
        != workload.semantic_identities["canonical_inference_requests"]
    ):
        raise ValueError("canonical inference-request identity does not match workload")
    path = directory / "legacy-inference-requests.npz"
    np.savez(
        path,
        request_ids=np.asarray(requests.request_ids, dtype=np.int32),
        prompt_ids=requests.prompt_ids,
    )
    verification_start = time.perf_counter()
    with np.load(path) as archive:
        request_ids = tuple(int(value) for value in archive["request_ids"])
        prompt_ids = np.asarray(archive["prompt_ids"], dtype=np.int32)
    loaded = fixed_inference_requests(workload, rows, order=request_ids)
    if not np.array_equal(prompt_ids, loaded.prompt_ids):
        raise ValueError("legacy inference bundle changed canonical prompts")
    verification_seconds = time.perf_counter() - verification_start
    return prompt_ids, file_identity(path), verification_seconds


def _model_config(workload: CanonicalWorkload, sml_module):
    return sml_module.SMLConfig(**workload.model)


def _prepare_model(workload: CanonicalWorkload, source_root: Path, *, training: bool):
    mx = importlib.import_module("mlx.core")
    sml_module = _legacy_import(source_root, "sml")
    train_module = _legacy_import(source_root, "train_sml")
    utilities = _legacy_import(source_root, "utils")
    utilities.set_seed(int(workload.optimizer["seed"]))
    model = sml_module.SMLLanguageModel(_model_config(workload, sml_module))
    train_module.apply_model_dtype(model, str(workload.precision["compute_dtype"]))
    model.train() if training else model.eval()
    mx.eval(model.parameters())
    return mx, sml_module, train_module, utilities, model


def _parameter_identity(model) -> str:
    mx = importlib.import_module("mlx.core")
    tree_flatten = importlib.import_module("mlx.utils").tree_flatten
    digest = hashlib.sha256()
    digest.update(b"sml-benchmark-bf16-parameters-v1\0")
    for name, value in sorted(
        tree_flatten(model.parameters()), key=lambda item: item[0]
    ):
        array = np.asarray(value.astype(mx.float32))
        encoded_name = name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(4, "little"))
        digest.update(encoded_name)
        digest.update(str(value.dtype).encode("ascii"))
        for dimension in array.shape:
            digest.update(int(dimension).to_bytes(8, "little"))
        digest.update(np.ascontiguousarray(array).tobytes())
    return f"sha256:{digest.hexdigest()}"


class _PreparedDataRuntime(_NativeRuntime):
    def __init__(self, source_root: Path, path: Path, sequence_length: int) -> None:
        self.mx = importlib.import_module("mlx.core")
        self.train = _legacy_import(source_root, "train_sml")
        self.path = path
        self.sequence_length = sequence_length
        self.iterator = self._new_iterator()

    def _new_iterator(self):
        blocks = self.train.iter_prepared_token_blocks(
            [self.path], self.sequence_length
        )
        return self.train.iter_mlx_batches(blocks, batch_size=1)

    def run(self, units: int) -> float:
        for _ in range(units):
            try:
                batch = next(self.iterator)
            except StopIteration:
                self.iterator = self._new_iterator()
                batch = next(self.iterator)
            self.mx.eval(batch["input_ids"], batch["labels"])
        return float(units)

    def reset_measured_order(self) -> None:
        self.iterator = self._new_iterator()


class _PretrainingRuntime(_NativeRuntime):
    def __init__(
        self,
        workload: CanonicalWorkload,
        source_root: Path,
        rows: np.ndarray,
        prepared_path: Path,
        *,
        include_loader: bool,
    ) -> None:
        (
            self.mx,
            self.sml,
            self.train,
            self.utilities,
            self.model,
        ) = _prepare_model(workload, source_root, training=True)
        self.optimizers = importlib.import_module("mlx.optimizers")
        self.nn = importlib.import_module("mlx.nn")
        self.workload = workload
        self.rows = rows
        self.prepared_path = prepared_path
        self.include_loader = include_loader
        self.sequence_length = int(workload.loader["sequence_length"])
        self.accumulation_steps = int(workload.optimizer["gradient_accumulation_steps"])
        self.batch_size = int(workload.loader["microbatch_size"])
        self.row_index = 0
        self.loader = self._new_loader() if include_loader else None
        self.schedule = self.utilities.build_lr_schedule(
            learning_rate=float(workload.optimizer["learning_rate"]),
            total_steps=int(workload.optimizer["total_steps"]),
            warmup_steps=int(workload.optimizer["warmup_steps"]),
            min_lr_ratio=float(workload.optimizer["minimum_learning_rate_ratio"]),
        )
        self.optimizer = self.optimizers.AdamW(
            learning_rate=self.schedule,
            betas=list(workload.optimizer["betas"]),
            eps=float(workload.optimizer["epsilon"]),
            weight_decay=float(workload.optimizer["optimizer_weight_decay"]),
            bias_correction=bool(workload.optimizer["bias_correction"]),
        )
        self.optimizer.init(self.model.trainable_parameters())
        training_config = self.train.TrainingConfig(
            sequence_length=self.sequence_length,
            batch_size=self.batch_size,
            gradient_accumulation_steps=self.accumulation_steps,
            learning_rate=float(workload.optimizer["learning_rate"]),
            max_grad_norm=float(workload.optimizer["max_grad_norm"]),
            warmup_steps=int(workload.optimizer["warmup_steps"]),
            lr_total_steps=int(workload.optimizer["total_steps"]),
            min_lr_ratio=float(workload.optimizer["minimum_learning_rate_ratio"]),
            seed=int(workload.optimizer["seed"]),
            autocast_dtype=str(workload.precision["compute_dtype"]),
            parameter_weight_decay=self.train.ParameterWeightDecayConfig(
                **workload.optimizer["parameter_weight_decay"]
            ),
        )
        self.training_config = training_config
        self.weight_decay_tree = self.train.build_parameter_weight_decay_tree(
            self.model.trainable_parameters(),
            parameter_weight_decay=training_config.parameter_weight_decay,
        )
        self.loss_and_grad = self.nn.value_and_grad(
            self.model,
            self.utilities.build_loss_fn(self.model),
        )
        self._initial_parameter_identity = _parameter_identity(self.model)

    def _new_loader(self):
        blocks = self.train.iter_prepared_token_blocks(
            [self.prepared_path], self.sequence_length
        )
        return self.train.iter_mlx_batches(blocks, batch_size=self.batch_size)

    def _next_batch(self):
        if self.include_loader:
            try:
                return next(self.loader)
            except StopIteration:
                self.loader = self._new_loader()
                return next(self.loader)
        batch_rows = []
        for _ in range(self.batch_size):
            batch_rows.append(self.rows[self.row_index % len(self.rows)])
            self.row_index += 1
        batch_array = np.asarray(batch_rows, dtype=np.int32)
        return {
            "input_ids": self.mx.array(batch_array[:, :-1], dtype=self.mx.int32),
            "labels": self.mx.array(batch_array[:, 1:], dtype=self.mx.int32),
        }

    def run(self, units: int) -> float:
        for _ in range(units):
            accumulation = self.train.GradientAccumulationWindow()
            for _ in range(self.accumulation_steps):
                batch = self._next_batch()
                loss, grads = self.loss_and_grad(batch["input_ids"], batch["labels"])
                self.mx.eval(loss, grads)
                self.train.accumulate_gradients(
                    accumulation,
                    grads,
                    float(loss.item()),
                    self.accumulation_steps,
                )
            grads, _loss, _micro_batches = self.train.consume_accumulated_grads(
                accumulation, self.accumulation_steps
            )
            clipped, _norm = self.train.clip_gradients_by_global_norm(
                grads, float(self.workload.optimizer["max_grad_norm"])
            )
            self.train.apply_decoupled_weight_decay(
                self.model,
                self.weight_decay_tree,
                self.schedule(self.optimizer.step),
            )
            self.train._retie_embeddings_if_needed(self.model)
            self.optimizer.update(self.model, clipped)
            self.train._retie_embeddings_if_needed(self.model)
            self.mx.eval(self.model.parameters(), self.optimizer.state)
        tokens = (
            units * self.accumulation_steps * self.batch_size * self.sequence_length
        )
        return float(tokens)

    def reset_measured_order(self) -> None:
        self.row_index = 0
        if self.include_loader:
            self.loader = self._new_loader()

    @property
    def initial_parameter_identity(self) -> str:
        return self._initial_parameter_identity


class _InferencePrefillRuntime(_NativeRuntime):
    def __init__(
        self, workload: CanonicalWorkload, source_root: Path, prompts: np.ndarray
    ) -> None:
        self.mx, _sml, _train, _utilities, self.model = _prepare_model(
            workload, source_root, training=False
        )
        self.prompt_tokens = int(workload.generation["prompt_tokens"])
        self.prompts = np.asarray(prompts, dtype=np.int32)
        self.request_index = 0
        self.measured_work_ids: list[int] = []
        self._initial_parameter_identity = _parameter_identity(self.model)

    def run(self, units: int) -> float:
        for _ in range(units):
            prompt = self.prompts[self.request_index % len(self.prompts)]
            self.measured_work_ids.append(self.request_index % len(self.prompts))
            self.request_index += 1
            output = self.model(self.mx.array(prompt[None, :], dtype=self.mx.int32))
            self.mx.eval(output.logits)
        return float(units * self.prompt_tokens)

    def reset_measured_order(self) -> None:
        self.request_index = 0
        self.measured_work_ids = []

    @property
    def initial_parameter_identity(self) -> str:
        return self._initial_parameter_identity


class _InferenceDecodeRuntime(_NativeRuntime):
    def __init__(
        self, workload: CanonicalWorkload, source_root: Path, prompts: np.ndarray
    ) -> None:
        self.mx, self.sml, _train, _utilities, self.model = _prepare_model(
            workload, source_root, training=False
        )
        self.prompt_tokens = int(workload.generation["prompt_tokens"])
        self.chunk_size = int(workload.generation["decode_chunk_size"])
        self.capacity = int(workload.model["original_max_position_embeddings"])
        self.prompts = np.asarray(prompts, dtype=np.int32)
        self.generation_config = self.sml.GenerationConfig(
            temperature=float(workload.generation["temperature"]),
            top_p=float(workload.generation["top_p"]),
            repetition_penalty=float(workload.generation["repetition_penalty"]),
            no_repeat_ngram_size=int(workload.generation["no_repeat_ngram_size"]),
            seed=int(workload.generation["seed"]),
        )
        self._initial_parameter_identity = _parameter_identity(self.model)
        self.measured_states = [self._prepare_state(prompt) for prompt in self.prompts]
        self.request_index = 0
        self.measured_work_ids: list[int] = []

    def _prepare_state(self, prompt: np.ndarray):
        prompt_array = self.mx.array(prompt[None, :], dtype=self.mx.int32)
        cache = self.sml.KVCache(max_seq_len=self.capacity)
        output = self.model(prompt_array, kv_cache=cache)
        logits = output.logits[:, -1, :]
        logits = self.sml.apply_repetition_penalty(
            logits, prompt_array, self.generation_config.repetition_penalty
        )
        logits = self.sml.apply_no_repeat_ngram(
            logits, prompt_array, self.generation_config.no_repeat_ngram_size
        )
        next_token = self.sml.select_next_token(
            logits, self.generation_config, key=None
        ).astype(self.mx.int32)
        generated = self.mx.concatenate([prompt_array, next_token], axis=1)
        self.mx.eval(next_token, generated)
        return cache, next_token, generated

    def _decode_state(self, state) -> None:
        cache, next_token, generated = state
        for _ in range(self.chunk_size):
            output = self.model(next_token, kv_cache=cache)
            logits = output.logits[:, -1, :]
            logits = self.sml.apply_repetition_penalty(
                logits, generated, self.generation_config.repetition_penalty
            )
            logits = self.sml.apply_no_repeat_ngram(
                logits, generated, self.generation_config.no_repeat_ngram_size
            )
            next_token = self.sml.select_next_token(
                logits, self.generation_config, key=None
            ).astype(self.mx.int32)
            generated = self.mx.concatenate([generated, next_token], axis=1)
            self.mx.eval(next_token)

    def run(self, units: int) -> float:
        for index in range(units):
            self._decode_state(
                self._prepare_state(self.prompts[index % len(self.prompts)])
            )
        return float(units * self.chunk_size)

    def run_measured(self, units: int) -> float:
        if units > len(self.measured_states):
            raise ValueError("decode measurement exceeds fixed request set")
        for request_id, state in enumerate(self.measured_states[:units]):
            self.measured_work_ids.append(request_id)
            self._decode_state(state)
        return float(units * self.chunk_size)

    def reset_after_warmup(self) -> None:
        return None

    def reset_measured_order(self) -> None:
        self.request_index = 0
        self.measured_work_ids = []

    @property
    def initial_parameter_identity(self) -> str:
        return self._initial_parameter_identity


class _SwagRuntime(_NativeRuntime):
    def __init__(
        self, workload: CanonicalWorkload, source_root: Path, cache_path: Path
    ) -> None:
        (
            self.mx,
            _sml,
            self.train,
            self.utilities,
            self.model,
        ) = _prepare_model(workload, source_root, training=True)
        self.nn = importlib.import_module("mlx.nn")
        self.optimizers = importlib.import_module("mlx.optimizers")
        self.swag = _legacy_import(source_root, "ft_swag")
        optimizer_config = workload.optimizer["swag"]
        loader_config = workload.loader["swag"]
        if not isinstance(optimizer_config, dict) or not isinstance(
            loader_config, dict
        ):
            raise ValueError("SWAG optimizer and loader configuration must be objects")
        self.accumulation_steps = int(optimizer_config["gradient_accumulation_steps"])
        self.sequence_length = min(
            int(loader_config["sequence_length"]),
            int(workload.model["original_max_position_embeddings"]),
        )
        self.batch_size = int(loader_config["batch_size"])
        lora_config = optimizer_config["lora"]
        if not isinstance(lora_config, dict):
            raise ValueError("SWAG LoRA configuration must be an object")
        fine_config = self.swag.SwagFineTuneConfig(
            dataset_name=str(loader_config["dataset_name"]),
            dataset_config=str(loader_config["dataset_config"]),
            dataset_split=str(loader_config["dataset_split"]),
            sequence_length=self.sequence_length,
            batch_size=self.batch_size,
            gradient_accumulation_steps=self.accumulation_steps,
            learning_rate=float(optimizer_config["learning_rate"]),
            max_grad_norm=float(optimizer_config["max_grad_norm"]),
            warmup_steps=int(optimizer_config["warmup_steps"]),
            lr_total_steps=int(optimizer_config["total_steps"]),
            seed=int(optimizer_config["seed"]),
            min_lr_ratio=float(optimizer_config["minimum_learning_rate_ratio"]),
            lora=self.swag.LoRAConfig(**lora_config),
            parameter_weight_decay=self.swag.SwagParameterWeightDecayConfig(
                **optimizer_config["parameter_weight_decay"]
            ),
            autocast_dtype=str(workload.precision["compute_dtype"]),
        )
        self.fine_config = fine_config
        self.swag.prepare_lora_model(self.model, fine_config)
        self.model.train()
        self.schedule = self.utilities.build_lr_schedule(
            learning_rate=float(optimizer_config["learning_rate"]),
            total_steps=int(optimizer_config["total_steps"]),
            warmup_steps=int(optimizer_config["warmup_steps"]),
            min_lr_ratio=float(optimizer_config["minimum_learning_rate_ratio"]),
        )
        self.optimizer = self.optimizers.AdamW(
            learning_rate=self.schedule,
            betas=list(optimizer_config["betas"]),
            eps=float(optimizer_config["epsilon"]),
            weight_decay=float(optimizer_config["optimizer_weight_decay"]),
            bias_correction=bool(optimizer_config["bias_correction"]),
        )
        self.optimizer.init(self.model.trainable_parameters())
        self.weight_decay_tree = self.train.build_parameter_weight_decay_tree(
            self.model.trainable_parameters(),
            parameter_weight_decay=fine_config.parameter_weight_decay,
        )
        loss_fn = self.swag.build_swag_ranking_loss_fn(
            self.model, pad_token_id=self.model.config.pad_token_id
        )
        self.loss_and_grad = self.nn.value_and_grad(self.model, loss_fn)
        self.cache_path = cache_path
        self.example_index = 0
        self.measured_work_ids: list[int] = []
        self.max_grad_norm = float(optimizer_config["max_grad_norm"])
        self._initial_parameter_identity = _parameter_identity(self.model)

    def _next_batch(self):
        with np.load(self.cache_path) as archive:
            count = int(archive["input_ids"].shape[0])
            indices = [
                (self.example_index + offset) % count
                for offset in range(self.batch_size)
            ]
            example_ids = np.asarray(archive["example_ids"][indices], dtype=np.int32)
            self.measured_work_ids.extend(int(value) for value in example_ids)
            self.example_index += self.batch_size
            input_ids = np.asarray(archive["input_ids"][indices], dtype=np.int32)
            labels = np.asarray(archive["labels"][indices], dtype=np.int32)
            candidate_labels = np.asarray(
                archive["candidate_labels"][indices], dtype=np.int32
            )
        return (
            self.mx.array(input_ids, dtype=self.mx.int32),
            self.mx.array(labels, dtype=self.mx.int32),
            self.mx.array(candidate_labels, dtype=self.mx.int32),
        )

    def run(self, units: int) -> float:
        for _ in range(units):
            accumulation = self.train.GradientAccumulationWindow()
            for _ in range(self.accumulation_steps):
                input_ids, labels, candidate_labels = self._next_batch()
                loss, grads = self.loss_and_grad(input_ids, labels, candidate_labels)
                self.mx.eval(loss, grads)
                self.train.accumulate_gradients(
                    accumulation,
                    grads,
                    float(loss.item()),
                    self.accumulation_steps,
                )
            grads, _loss, _micro_batches = self.train.consume_accumulated_grads(
                accumulation, self.accumulation_steps
            )
            clipped, _norm = self.train.clip_gradients_by_global_norm(
                grads, self.max_grad_norm
            )
            self.train.apply_decoupled_weight_decay(
                self.model,
                self.weight_decay_tree,
                self.schedule(self.optimizer.step),
            )
            self.optimizer.update(self.model, clipped)
            self.mx.eval(self.model.parameters(), self.optimizer.state)
        return float(units * self.accumulation_steps * self.batch_size)

    def reset_measured_order(self) -> None:
        self.example_index = 0
        self.measured_work_ids = []

    @property
    def initial_parameter_identity(self) -> str:
        return self._initial_parameter_identity


class _CheckpointRuntime(_NativeRuntime):
    def __init__(
        self,
        workload: CanonicalWorkload,
        source_root: Path,
        rows: np.ndarray,
        prepared_path: Path,
        output_directory: Path,
    ) -> None:
        self.training = _PretrainingRuntime(
            workload,
            source_root,
            rows,
            prepared_path,
            include_loader=False,
        )
        self.train = self.training.train
        self.output_directory = output_directory / "checkpoints"
        self.output_directory.mkdir()
        self.current_path: Path | None = None
        self.model_config = self.training.model.config
        self.training_config = self.training.training_config
        self.step = 0

    def run(self, units: int) -> float:
        for _ in range(units):
            self.step += 1
            staging = self.output_directory / f".step-{self.step}.staging"
            published = self.output_directory / f"step-{self.step}"
            self.train.save_checkpoint(
                staging,
                self.training.model,
                self.training.optimizer,
                self.model_config,
                self.training_config,
                self.step,
            )
            _fsync_tree(staging)
            os.replace(staging, published)
            _fsync_directory(self.output_directory)
            previous = self.current_path
            self.current_path = published
            if previous is not None:
                shutil.rmtree(previous)
                _fsync_directory(self.output_directory)
        return float(units)

    @property
    def initial_parameter_identity(self) -> str:
        return self.training.initial_parameter_identity


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(path: Path) -> None:
    for child in sorted(path.rglob("*")):
        if child.is_file():
            descriptor = os.open(child, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    _fsync_directory(path)


def resolve_native_workload(
    metric: MetricName,
    canonical_workload: CanonicalWorkload,
    source_root: Path,
) -> LegacyNativeWorkload:
    resolved_root = source_root.resolve()
    temporary_directory = tempfile.TemporaryDirectory(prefix=f"sml-v2-{metric}-")
    temporary_path = Path(temporary_directory.name)
    (
        rows,
        prepared_path,
        representation_identity,
        row_identity,
        verification_seconds,
    ) = _write_and_verify_legacy_rows(canonical_workload, temporary_path)
    semantic_identity = canonical_input_identity(metric, canonical_workload)
    native_representation_identity = representation_identity
    extra_verification_seconds = 0.0

    if metric == "prepared-data":
        runtime: _NativeRuntime = _PreparedDataRuntime(
            resolved_root,
            prepared_path,
            int(canonical_workload.loader["sequence_length"]),
        )
    elif metric in (
        "pretraining-compute",
        "compile-cold-start",
        "peak-metal-memory",
    ):
        runtime = _PretrainingRuntime(
            canonical_workload,
            resolved_root,
            rows,
            prepared_path,
            include_loader=False,
        )
    elif metric == "pretraining-end-to-end":
        runtime = _PretrainingRuntime(
            canonical_workload,
            resolved_root,
            rows,
            prepared_path,
            include_loader=True,
        )
    elif metric == "swag-end-to-end":
        swag_path, native_representation_identity, extra_verification_seconds = (
            _write_and_verify_swag_examples(canonical_workload, rows, temporary_path)
        )
        runtime = _SwagRuntime(canonical_workload, resolved_root, swag_path)
    elif metric == "inference-prefill":
        prompts, native_representation_identity, extra_verification_seconds = (
            _write_and_verify_inference_requests(
                canonical_workload, rows, temporary_path
            )
        )
        runtime = _InferencePrefillRuntime(canonical_workload, resolved_root, prompts)
    elif metric == "inference-decode":
        prompts, native_representation_identity, extra_verification_seconds = (
            _write_and_verify_inference_requests(
                canonical_workload, rows, temporary_path
            )
        )
        runtime = _InferenceDecodeRuntime(canonical_workload, resolved_root, prompts)
    elif metric == "checkpoint-pause":
        runtime = _CheckpointRuntime(
            canonical_workload,
            resolved_root,
            rows,
            prepared_path,
            temporary_path,
        )
    else:
        raise ValueError(f"unsupported legacy metric: {metric}")

    swag_loader = canonical_workload.loader["swag"]
    swag_optimizer = canonical_workload.optimizer["swag"]
    if not isinstance(swag_loader, dict) or not isinstance(swag_optimizer, dict):
        raise ValueError("SWAG canonical configuration must be objects")
    projection = canonical_metric_projection(metric, canonical_workload)
    native_format = (
        "swag-npz-int32"
        if metric == "swag-end-to-end"
        else (
            "inference-request-npz-int32"
            if metric in ("inference-prefill", "inference-decode")
            else "prepared-data-npz-uint16"
        )
    )
    native_configuration: dict[str, JsonValue] = {
        "metric": metric,
        "native_input_format": native_format,
        "parameter_dtype": "bfloat16",
        "moment_dtype": "bfloat16",
        "master_parameters": False,
        "rope_scaling_factor": canonical_workload.model["rope_scaling_factor"],
        "sequence_length": (
            swag_loader["sequence_length"]
            if metric == "swag-end-to-end"
            else canonical_workload.loader["sequence_length"]
        ),
        "microbatch_size": (
            swag_loader["batch_size"]
            if metric == "swag-end-to-end"
            else canonical_workload.loader["microbatch_size"]
        ),
        "gradient_accumulation_steps": (
            swag_optimizer["gradient_accumulation_steps"]
            if metric == "swag-end-to-end"
            else canonical_workload.optimizer["gradient_accumulation_steps"]
        ),
        "canonical_row_identity": row_identity,
        "canonical_input_identity": semantic_identity,
        "canonical_projection_identity": structured_identity(
            "sml-benchmark-metric-projection-v1", projection
        ),
        "canonical_execution_order_identity": canonical_execution_order_identity(
            metric, canonical_workload
        ),
        "parameter_precision_policy": LEGACY_PRECISION_POLICY,
    }
    initial_identity = (
        runtime.initial_parameter_identity
        or canonical_workload.semantic_identities["initial_bf16_parameters"]
    )
    return LegacyNativeWorkload(
        metric=metric,
        source_root=resolved_root,
        canonical_workload=canonical_workload,
        native_configuration=native_configuration,
        native_representation_identity=native_representation_identity,
        canonical_row_identity=row_identity,
        canonical_input_identity=semantic_identity,
        canonical_projection=projection,
        execution_order_identity=canonical_execution_order_identity(
            metric, canonical_workload
        ),
        initial_parameter_identity=initial_identity,
        startup_verification_seconds=verification_seconds + extra_verification_seconds,
        runtime=runtime,
        temporary_directory=temporary_directory,
    )


def run_warmup(
    metric: MetricName,
    native_workload: LegacyNativeWorkload,
    units: int,
) -> None:
    if metric != native_workload.metric:
        raise ValueError("metric does not match native workload")
    native_workload.runtime.run(units)
    native_workload.runtime.reset_after_warmup()


def run_measured(
    metric: MetricName,
    native_workload: LegacyNativeWorkload,
    units: int,
) -> float:
    if metric != native_workload.metric:
        raise ValueError("metric does not match native workload")
    native_workload.runtime.reset_measured_order()
    measured = getattr(native_workload.runtime, "run_measured", None)
    if measured is not None:
        return float(measured(units))
    return native_workload.runtime.run(units)
