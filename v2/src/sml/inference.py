"""Persistent non-reentrant latest-only inference session loading."""

from __future__ import annotations

import secrets
import threading
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

import mlx.core as mx
from mlx.utils import tree_unflatten

from sml.artifacts.checkpoint import (
    ResolvedStep,
    open_checkpoint_reader,
    recover_latest_index,
    run_access_lock,
)
from sml.artifacts.manifest import (
    BaseSnapshotManifest,
    ExportManifest,
    LoRARunManifest,
    PretrainingRunManifest,
    VerificationLevel,
    read_manifest,
)
from sml.data.tokenizer import LoadedTokenizer, load_tokenizer_bundle
from sml.errors import SMLArtifactError, SMLRuntimeError
from sml.model.cache import allocate_kv_state, reset_kv_state
from sml.model.config import GenerationConfig, ModelConfig
from sml.model.generation import (
    apply_no_repeat_ngram,
    apply_repetition_penalty,
    select_next_token_arrays,
)
from sml.model.language_model import SMLLanguageModel
from sml.training.lora import (
    apply_lora,
    load_lora_state_dict,
    lora_config_from_mapping,
    merged_model_weights,
)

_MODEL_GROUP = "model.safetensors"
_MASTER_GROUP = "master.safetensors"
_ADAPTER_GROUP = "adapters.safetensors"


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    artifact_kind: str
    run_identity: str | None
    step: int | None
    checkpoint_identity: str | None
    run_step_identity: str | None
    tokenizer_identity: str
    verification: VerificationLevel


@dataclass(frozen=True, slots=True)
class GenerationResult:
    text: str
    token_ids: tuple[int, ...]
    seed: int | None
    model: ModelIdentity


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    max_new_tokens: int
    config: GenerationConfig = field(default_factory=GenerationConfig)
    include_prompt: bool = False

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_new_tokens, bool)
            or not isinstance(self.max_new_tokens, int)
            or self.max_new_tokens < 0
        ):
            raise ValueError("max_new_tokens must be non-negative")
        if not isinstance(self.config, GenerationConfig):
            raise TypeError("config must be a GenerationConfig")
        if not isinstance(self.include_prompt, bool):
            raise TypeError("include_prompt must be a bool")


@dataclass(frozen=True, slots=True)
class InferenceRuntimeConfig:
    batch_size_buckets: tuple[int, ...] = (1, 2, 4, 8, 16)
    decode_chunk_size: int = 8

    def __post_init__(self) -> None:
        buckets = self.batch_size_buckets
        if not isinstance(buckets, tuple) or not buckets:
            raise ValueError("batch_size_buckets must be a nonempty tuple")
        previous = 0
        for bucket in buckets:
            if isinstance(bucket, bool) or not isinstance(bucket, int) or bucket <= 0:
                raise ValueError("batch size buckets must be positive")
            if bucket <= previous:
                raise ValueError("batch size buckets must be strictly increasing")
            previous = bucket
        if (
            isinstance(self.decode_chunk_size, bool)
            or not isinstance(self.decode_chunk_size, int)
            or self.decode_chunk_size <= 0
        ):
            raise ValueError("decode_chunk_size must be positive")


_DEFAULT_RUNTIME = InferenceRuntimeConfig()


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    checkpoint: Path
    prompt: str
    request: GenerationRequest
    full_verify: bool = False
    runtime: InferenceRuntimeConfig = _DEFAULT_RUNTIME

    def __post_init__(self) -> None:
        if not isinstance(self.checkpoint, Path):
            raise TypeError("checkpoint must be a Path")
        if not isinstance(self.prompt, str):
            raise TypeError("prompt must be a string")
        if not isinstance(self.request, GenerationRequest):
            raise TypeError("request must be a GenerationRequest")
        if not isinstance(self.full_verify, bool):
            raise TypeError("full_verify must be a bool")
        if not isinstance(self.runtime, InferenceRuntimeConfig):
            raise TypeError("runtime must be an InferenceRuntimeConfig")


@dataclass(frozen=True, slots=True)
class GenerationKernelKey:
    temperature: float
    top_p: float
    repetition_penalty: float
    no_repeat_ngram_size: int

    @classmethod
    def from_config(cls, config: GenerationConfig) -> GenerationKernelKey:
        return cls(
            temperature=config.temperature,
            top_p=config.top_p,
            repetition_penalty=config.repetition_penalty,
            no_repeat_ngram_size=config.no_repeat_ngram_size,
        )


@dataclass(frozen=True, slots=True)
class ScoringKernelKey:
    length_bucket: int
    batch_size_bucket: int
    padding: str


@dataclass(frozen=True, slots=True)
class _PreparedRequest:
    caller_index: int
    prompt_ids: tuple[int, ...]
    length_bucket: int
    kernel_key: GenerationKernelKey
    seed: int
    key: mx.array
    max_new_tokens: int
    include_prompt: bool


@dataclass(frozen=True, slots=True)
class GenerationBucket:
    length_bucket: int
    batch_size_bucket: int
    kernel_key: GenerationKernelKey
    keys: mx.array
    request_mask: mx.array
    prompt_ids: tuple[tuple[int, ...], ...]
    prompt_lengths: mx.array
    max_new_tokens: mx.array
    seeds: tuple[int, ...]
    caller_indices: tuple[int, ...]
    include_prompt: tuple[bool, ...]
    host_max_new: int


def allocate_generation_seed() -> int:
    return secrets.randbits(32)


def vmapped_select_one_token(
    logits: mx.array,
    keys: mx.array,
    request_mask: mx.array,
    kernel_key: GenerationKernelKey,
) -> tuple[mx.array, mx.array]:
    def select_one_token(logits_row, key):
        return select_next_token_arrays(
            logits_row,
            key,
            temperature=kernel_key.temperature,
            top_p=kernel_key.top_p,
        )

    selected, next_keys = mx.vmap(select_one_token, in_axes=(0, 0))(logits, keys)
    selected = mx.where(request_mask, selected, mx.zeros_like(selected))
    next_keys = mx.where(request_mask[:, None], next_keys, keys)
    return selected, next_keys


def infer(config: InferenceConfig) -> GenerationResult:
    if not isinstance(config, InferenceConfig):
        raise TypeError("config must be an InferenceConfig")
    session = InferenceSession.from_checkpoint(
        config.checkpoint,
        full_verify=config.full_verify,
        runtime=config.runtime,
    )
    return session.generate(config.prompt, config.request)


@dataclass(frozen=True, slots=True)
class ResolvedModel:
    artifact_kind: str
    run_identity: str | None
    step: int | None
    checkpoint_identity: str | None
    run_step_identity: str | None
    verification: VerificationLevel
    model_config: ModelConfig
    tokenizer: LoadedTokenizer
    model_arrays: Mapping[str, mx.array]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_arrays",
            MappingProxyType(dict(self.model_arrays)),
        )

    @property
    def tokenizer_identity(self) -> str:
        return self.tokenizer.manifest.identity

    def identity(self) -> ModelIdentity:
        return ModelIdentity(
            artifact_kind=self.artifact_kind,
            run_identity=self.run_identity,
            step=self.step,
            checkpoint_identity=self.checkpoint_identity,
            run_step_identity=self.run_step_identity,
            tokenizer_identity=self.tokenizer_identity,
            verification=self.verification,
        )


class _CallGuard:
    def __init__(self) -> None:
        self._lock = threading.Lock()

    @contextmanager
    def acquire(self):
        if not self._lock.acquire(blocking=False):
            raise SMLRuntimeError("session is non-reentrant")
        try:
            yield
        finally:
            self._lock.release()


class _Lease:
    def __init__(
        self,
        pool: BufferPool,
        token_storage: mx.array,
        cache_state: object,
        pool_key: tuple[int, ...],
    ) -> None:
        self.token_storage = token_storage
        self.cache_state = cache_state
        self._pool = pool
        self._pool_key = pool_key
        self._released = False

    def discard(self) -> None:
        if self._released:
            return
        self._pool._return(self)
        self._released = True
        self.token_storage = None
        self.cache_state = None


class BufferPool:
    def __init__(self) -> None:
        self._active = 0
        self._free: dict[tuple[int, ...], list[tuple[mx.array, object]]] = {}

    @property
    def active_leases(self) -> int:
        return self._active

    def lease(
        self,
        *,
        batch_size: int,
        capacity: int,
        config: ModelConfig,
    ) -> _Lease:
        key = (
            batch_size,
            capacity,
            config.num_layers,
            config.num_kv_heads,
            config.head_dim,
        )
        available = self._free.get(key)
        if available:
            token_storage, cache_state = available.pop()
        else:
            token_storage = mx.zeros((batch_size, capacity), dtype=mx.int32)
            cache_state = allocate_kv_state(config, batch_size, capacity, mx.bfloat16)
            mx.eval(token_storage, cache_state)
        self._active += 1
        return _Lease(self, token_storage, cache_state, key)

    def release(self, lease: _Lease) -> None:
        lease.discard()

    def _reset_cache(self, cache_state: object) -> object:
        return reset_kv_state(cache_state)

    def _return(self, lease: _Lease) -> None:
        self._active = max(0, self._active - 1)
        self._free.setdefault(lease._pool_key, []).append(
            (lease.token_storage, self._reset_cache(lease.cache_state))
        )


def _length_buckets(effective_context_length: int) -> tuple[int, ...]:
    buckets: list[int] = []
    size = 1
    while size < effective_context_length:
        buckets.append(size)
        size *= 2
    buckets.append(effective_context_length)
    return tuple(buckets)


def _pad_scoring_row(
    token_ids: tuple[int, ...],
    continuation_start: int,
    capacity: int,
    padding: str,
    pad_id: int,
) -> tuple[list[int], list[bool], list[bool]]:
    length = len(token_ids)
    pad_count = capacity - length
    offset = 0 if padding == "right" else pad_count
    padded_ids = [pad_id] * capacity
    attention = [False] * capacity
    target_mask = [False] * capacity
    for index, token_id in enumerate(token_ids):
        position = offset + index
        padded_ids[position] = token_id
        attention[position] = True
        if index >= continuation_start:
            target_mask[position] = True
    return padded_ids, attention, target_mask


def _require_run_path(path: Path) -> Path:
    if not isinstance(path, Path):
        raise TypeError("path must be a Path")
    if path.name.startswith("step-"):
        raise SMLArtifactError("direct step-* paths are rejected")
    if (path / "checkpoint.json").exists():
        raise SMLArtifactError("direct checkpoint step paths are rejected")
    return path


def load_owned_model_arrays(
    run: Path,
    *,
    full_verify: bool,
) -> tuple[ResolvedStep, Mapping[str, mx.array]]:
    if not isinstance(full_verify, bool):
        raise TypeError("full_verify must be a bool")
    verification = (
        VerificationLevel.FULL if full_verify else VerificationLevel.MANIFEST_TRUSTED
    )
    load_groups = frozenset(
        {_MODEL_GROUP, _MASTER_GROUP} if full_verify else {_MODEL_GROUP}
    )
    with run_access_lock(run, exclusive=False):
        recovered = recover_latest_index(
            run,
            writable=False,
            verification=VerificationLevel.MANIFEST_TRUSTED,
        )
        with open_checkpoint_reader(
            run,
            step=recovered.step,
            expected_checkpoint_identity=recovered.checkpoint.identity,
            verification=verification,
            load_array_groups=load_groups,
            hold_lock=False,
        ) as reader:
            contents = reader.read_contents()
            model_group = contents.array_groups[_MODEL_GROUP]
            owned = {name: model_group[name] for name in model_group}
            mx.eval(*owned.values())
            return reader.resolved, MappingProxyType(owned)


def _require_unit_rope(model: Mapping[str, object], *, context: str) -> ModelConfig:
    rope_factor = model.get("rope_scaling_factor")
    if rope_factor != 1.0:
        raise SMLArtifactError(f"{context} rope_scaling_factor must be exactly 1.0")
    model_config = ModelConfig(**dict(model))
    if model_config.rope_scaling_factor != 1.0:
        raise SMLArtifactError("resolution never substitutes rope_scaling_factor")
    return model_config


def _resolve_pretraining_run(path: Path, *, full_verify: bool) -> ResolvedModel:
    resolved_step, owned = load_owned_model_arrays(path, full_verify=full_verify)
    if not isinstance(resolved_step.run, PretrainingRunManifest):
        raise SMLArtifactError("pretraining resolution requires a pretraining run")
    model_config = _require_unit_rope(resolved_step.run.model, context="pretraining")
    verification = resolved_step.verification
    tokenizer = load_tokenizer_bundle(path / "tokenizer", verification)
    if tokenizer.manifest.identity != resolved_step.run.tokenizer_identity:
        raise SMLArtifactError("run tokenizer identity does not match run.json")
    return ResolvedModel(
        artifact_kind=resolved_step.run.kind,
        run_identity=resolved_step.run.identity,
        step=resolved_step.step,
        checkpoint_identity=resolved_step.checkpoint.identity,
        run_step_identity=resolved_step.run_step_identity,
        verification=verification,
        model_config=model_config,
        tokenizer=tokenizer,
        model_arrays=owned,
    )


def _resolve_lora_run(path: Path, *, full_verify: bool) -> ResolvedModel:
    verification = (
        VerificationLevel.FULL if full_verify else VerificationLevel.MANIFEST_TRUSTED
    )
    load_groups = (
        frozenset({_ADAPTER_GROUP, "optimizer.safetensors", "trainer.safetensors"})
        if full_verify
        else frozenset({_ADAPTER_GROUP})
    )
    with run_access_lock(path, exclusive=False):
        recovered = recover_latest_index(
            path,
            writable=False,
            verification=VerificationLevel.MANIFEST_TRUSTED,
        )
        if not isinstance(recovered.run, LoRARunManifest):
            raise SMLArtifactError("LoRA resolution requires a LoRA run")
        model_config = _require_unit_rope(recovered.run.model, context="LoRA run")
        base = read_manifest(path / "base", BaseSnapshotManifest, verification)
        if base.manifest.identity != recovered.run.base_identity:
            raise SMLArtifactError("run base snapshot identity does not match run.json")
        if base.manifest.model.get("rope_scaling_factor") != 1.0:
            raise SMLArtifactError(
                "copied base rope_scaling_factor must be exactly 1.0"
            )
        base_arrays = dict(
            mx.load(
                str(path / "base" / base.manifest.working_weights.payload.logical_path)
            )
        )
        mx.eval(*base_arrays.values())
        with open_checkpoint_reader(
            path,
            step=recovered.step,
            expected_checkpoint_identity=recovered.checkpoint.identity,
            verification=verification,
            load_array_groups=load_groups,
            hold_lock=False,
        ) as reader:
            contents = reader.read_contents()
            adapters = dict(contents.array_groups[_ADAPTER_GROUP])
            mx.eval(*adapters.values())
            resolved_step = reader.resolved
        tokenizer = load_tokenizer_bundle(path / "tokenizer", verification)
        if tokenizer.manifest.identity != recovered.run.tokenizer_identity:
            raise SMLArtifactError("run tokenizer identity does not match run.json")
        model = SMLLanguageModel(model_config, key=mx.random.key(0))
        model.update(tree_unflatten(sorted(base_arrays.items())))
        apply_lora(
            model,
            lora_config_from_mapping(dict(recovered.run.lora)),
            key=mx.random.key(1),
        )
        load_lora_state_dict(model, adapters)
        merged = merged_model_weights(model)
        mx.eval(*merged.values())
        return ResolvedModel(
            artifact_kind=recovered.run.kind,
            run_identity=recovered.run.identity,
            step=resolved_step.step,
            checkpoint_identity=resolved_step.checkpoint.identity,
            run_step_identity=resolved_step.run_step_identity,
            verification=verification,
            model_config=model_config,
            tokenizer=tokenizer,
            model_arrays=merged,
        )


def _resolve_export(path: Path, *, full_verify: bool) -> ResolvedModel:
    verification = (
        VerificationLevel.FULL if full_verify else VerificationLevel.MANIFEST_TRUSTED
    )
    verified = read_manifest(path, ExportManifest, verification)
    model_config = _require_unit_rope(verified.manifest.model, context="export")
    tokenizer = load_tokenizer_bundle(path / "tokenizer", verification)
    if tokenizer.manifest.identity != verified.manifest.tokenizer_identity:
        raise SMLArtifactError("export tokenizer identity does not match manifest")
    arrays = dict(
        mx.load(str(path / verified.manifest.model_weights.payload.logical_path))
    )
    mx.eval(*arrays.values())
    return ResolvedModel(
        artifact_kind=verified.manifest.kind,
        run_identity=None,
        step=verified.manifest.diagnostic_source_step,
        checkpoint_identity=None,
        run_step_identity=None,
        verification=verification,
        model_config=model_config,
        tokenizer=tokenizer,
        model_arrays=arrays,
    )


def resolve_model_artifact(path: Path, *, full_verify: bool) -> ResolvedModel:
    path = _require_run_path(path)
    if not isinstance(full_verify, bool):
        raise TypeError("full_verify must be a bool")
    if (path / PretrainingRunManifest.MANIFEST_FILENAME).exists():
        recovered = recover_latest_index(
            path,
            writable=False,
            verification=VerificationLevel.MANIFEST_TRUSTED,
        )
        if isinstance(recovered.run, PretrainingRunManifest):
            return _resolve_pretraining_run(path, full_verify=full_verify)
        if isinstance(recovered.run, LoRARunManifest):
            return _resolve_lora_run(path, full_verify=full_verify)
        raise SMLArtifactError("unsupported run kind for model resolution")
    return _resolve_export(path, full_verify=full_verify)


class InferenceSession:
    def __init__(
        self,
        resolved: ResolvedModel,
        runtime: InferenceRuntimeConfig,
    ) -> None:
        self._resolved = resolved
        self._runtime = runtime
        self._call_guard = _CallGuard()
        self.buffer_pool = BufferPool()
        self._model = SMLLanguageModel(resolved.model_config, key=mx.random.key(0))
        self._parameters = tree_unflatten(sorted(resolved.model_arrays.items()))
        self._compiled: dict[tuple[object, ...], object] = {}
        self._scoring_compiled: dict[ScoringKernelKey, object] = {}

    @classmethod
    def from_checkpoint(
        cls,
        path: Path,
        *,
        full_verify: bool = False,
        runtime: InferenceRuntimeConfig = _DEFAULT_RUNTIME,
    ) -> InferenceSession:
        if not isinstance(runtime, InferenceRuntimeConfig):
            raise TypeError("runtime must be an InferenceRuntimeConfig")
        resolved = resolve_model_artifact(path, full_verify=full_verify)
        return cls(resolved, runtime)

    @property
    def resolved_model(self) -> ResolvedModel:
        return self._resolved

    @property
    def model_identity(self) -> ModelIdentity:
        return self._resolved.identity()

    @property
    def length_buckets(self) -> tuple[int, ...]:
        return _length_buckets(self._resolved.model_config.effective_context_length)

    def generate(self, text: str, request: GenerationRequest) -> GenerationResult:
        self._require_generate_args(text, request)
        with self._call_guard.acquire():
            return self._generate_batch(((text, request),))[0]

    def generate_batch(
        self,
        items: Sequence[tuple[str, GenerationRequest]],
    ) -> tuple[GenerationResult, ...]:
        if not items:
            return ()
        for text, request in items:
            self._require_generate_args(text, request)
        with self._call_guard.acquire():
            return self._generate_batch(items)

    def score_encoded_loglikelihoods(
        self,
        items: Sequence[tuple[tuple[int, ...], int]],
        *,
        padding: str,
    ) -> tuple[tuple[float, bool], ...]:
        if padding not in ("left", "right"):
            raise ValueError("padding must be 'left' or 'right'")
        if not items:
            return ()
        with self._call_guard.acquire():
            return self._score_encoded_loglikelihoods(items, padding=padding)

    def _require_generate_args(self, text: object, request: object) -> None:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not isinstance(request, GenerationRequest):
            raise TypeError("request must be a GenerationRequest")

    def _generate_batch(
        self,
        items: Sequence[tuple[str, GenerationRequest]],
    ) -> tuple[GenerationResult, ...]:
        results: list[GenerationResult | None] = [None] * len(items)
        for bucket in self._bucketize(items):
            lease = None
            try:
                lease = self.buffer_pool.lease(
                    batch_size=bucket.batch_size_bucket,
                    capacity=bucket.length_bucket,
                    config=self._resolved.model_config,
                )
                bucket_results = self._decode_chunk(bucket, lease)
                for caller_index, result in bucket_results:
                    results[caller_index] = result
            finally:
                if lease is not None:
                    self.buffer_pool.release(lease)
        return tuple(results)

    def _bucketize(
        self,
        items: Sequence[tuple[str, GenerationRequest]],
    ) -> tuple[GenerationBucket, ...]:
        prepared: list[_PreparedRequest] = []
        for caller_index, (text, request) in enumerate(items):
            prompt_ids = self._encode_prompt(text)
            length_bucket = self._select_length_bucket(
                len(prompt_ids) + request.max_new_tokens
            )
            seed = request.config.seed
            if seed is None:
                seed = allocate_generation_seed()
            prepared.append(
                _PreparedRequest(
                    caller_index=caller_index,
                    prompt_ids=prompt_ids,
                    length_bucket=length_bucket,
                    kernel_key=GenerationKernelKey.from_config(request.config),
                    seed=seed,
                    key=mx.random.key(seed),
                    max_new_tokens=request.max_new_tokens,
                    include_prompt=request.include_prompt,
                )
            )

        groups: dict[tuple[int, GenerationKernelKey], list[_PreparedRequest]] = {}
        group_order: list[tuple[int, GenerationKernelKey]] = []
        for item in prepared:
            group_key = (item.length_bucket, item.kernel_key)
            if group_key not in groups:
                groups[group_key] = []
                group_order.append(group_key)
            groups[group_key].append(item)

        max_batch = self._runtime.batch_size_buckets[-1]
        buckets: list[GenerationBucket] = []
        for group_key in group_order:
            members = groups[group_key]
            for start in range(0, len(members), max_batch):
                chunk = members[start : start + max_batch]
                batch_size_bucket = self._select_batch_size_bucket(len(chunk))
                buckets.append(
                    self._pad_bucket(
                        chunk, group_key[0], batch_size_bucket, group_key[1]
                    )
                )
        return tuple(buckets)

    def _select_batch_size_bucket(self, group_size: int) -> int:
        for bucket in self._runtime.batch_size_buckets:
            if group_size <= bucket:
                return bucket
        return self._runtime.batch_size_buckets[-1]

    def _pad_bucket(
        self,
        members: Sequence[_PreparedRequest],
        length_bucket: int,
        batch_size_bucket: int,
        kernel_key: GenerationKernelKey,
    ) -> GenerationBucket:
        pad_id = self._resolved.model_config.pad_token_id
        prompt_ids: list[tuple[int, ...]] = []
        prompt_lengths: list[int] = []
        max_new: list[int] = []
        seeds: list[int] = []
        caller_indices: list[int] = []
        include_prompt: list[bool] = []
        keys: list[mx.array] = []
        real_mask: list[bool] = []
        for item in members:
            prompt_ids.append(item.prompt_ids)
            prompt_lengths.append(len(item.prompt_ids))
            max_new.append(item.max_new_tokens)
            seeds.append(item.seed)
            caller_indices.append(item.caller_index)
            include_prompt.append(item.include_prompt)
            keys.append(item.key)
            real_mask.append(True)
        while len(prompt_ids) < batch_size_bucket:
            prompt_ids.append((pad_id,))
            prompt_lengths.append(1)
            max_new.append(0)
            seeds.append(0)
            caller_indices.append(-1)
            include_prompt.append(False)
            keys.append(mx.random.key(0))
            real_mask.append(False)
        return GenerationBucket(
            length_bucket=length_bucket,
            batch_size_bucket=batch_size_bucket,
            kernel_key=kernel_key,
            keys=mx.stack(keys),
            request_mask=mx.array(real_mask, dtype=mx.bool_),
            prompt_ids=tuple(prompt_ids),
            prompt_lengths=mx.array(prompt_lengths, dtype=mx.int32),
            max_new_tokens=mx.array(max_new, dtype=mx.int32),
            seeds=tuple(seeds),
            caller_indices=tuple(caller_indices),
            include_prompt=tuple(include_prompt),
            host_max_new=max(
                (item.max_new_tokens for item in members),
                default=0,
            ),
        )

    def _encode_prompt(self, text: str) -> tuple[int, ...]:
        processor = self._resolved.tokenizer.processor
        bos_id = int(processor.bos_id())
        encoded = [int(token) for token in processor.encode(text)]
        if bos_id >= 0 and (not encoded or encoded[0] != bos_id):
            encoded = [bos_id, *encoded]
        if not encoded:
            raise SMLRuntimeError(
                "prompt encodes to no tokens without a usable BOS token"
            )
        return tuple(encoded)

    def _select_length_bucket(self, required: int) -> int:
        for bucket in self.length_buckets:
            if required <= bucket:
                return bucket
        raise SMLRuntimeError(
            "prompt overflow: required capacity exceeds effective context length"
        )

    def _score_encoded_loglikelihoods(
        self,
        items: Sequence[tuple[tuple[int, ...], int]],
        *,
        padding: str,
    ) -> tuple[tuple[float, bool], ...]:
        prepared: list[tuple[int, tuple[int, ...], int, int]] = []
        for caller_index, (token_ids, continuation_start) in enumerate(items):
            if continuation_start < 0 or continuation_start > len(token_ids):
                raise SMLRuntimeError("invalid continuation start")
            length_bucket = self._select_length_bucket(len(token_ids))
            prepared.append(
                (caller_index, token_ids, continuation_start, length_bucket)
            )

        groups: dict[int, list[tuple[int, tuple[int, ...], int, int]]] = {}
        group_order: list[int] = []
        for item in prepared:
            length_bucket = item[3]
            if length_bucket not in groups:
                groups[length_bucket] = []
                group_order.append(length_bucket)
            groups[length_bucket].append(item)

        results: list[tuple[float, bool] | None] = [None] * len(items)
        max_batch = self._runtime.batch_size_buckets[-1]
        for length_bucket in group_order:
            members = groups[length_bucket]
            for start in range(0, len(members), max_batch):
                chunk = members[start : start + max_batch]
                batch_size_bucket = self._select_batch_size_bucket(len(chunk))
                scored = self._score_chunk(
                    chunk,
                    length_bucket,
                    batch_size_bucket,
                    padding,
                )
                for (caller_index, _token_ids, _start, _bucket), result in zip(
                    chunk, scored, strict=True
                ):
                    results[caller_index] = result
        return tuple(results)

    def _score_chunk(
        self,
        members: Sequence[tuple[int, tuple[int, ...], int, int]],
        length_bucket: int,
        batch_size_bucket: int,
        padding: str,
    ) -> tuple[tuple[float, bool], ...]:
        compiled = self._compiled_scoring_kernel(
            length_bucket, batch_size_bucket, padding
        )
        pad_id = self._resolved.model_config.pad_token_id
        rows: list[list[int]] = []
        attention_rows: list[list[bool]] = []
        target_rows: list[list[bool]] = []
        request_rows: list[bool] = []
        for _caller_index, token_ids, continuation_start, _length_bucket in members:
            padded_ids, attention, target_mask = _pad_scoring_row(
                token_ids,
                continuation_start,
                length_bucket,
                padding,
                pad_id,
            )
            rows.append(padded_ids)
            attention_rows.append(attention)
            target_rows.append(target_mask)
            request_rows.append(True)
        while len(rows) < batch_size_bucket:
            synthetic_ids = [pad_id] * length_bucket
            synthetic_attention = [False] * length_bucket
            synthetic_attention[0] = True
            rows.append(synthetic_ids)
            attention_rows.append(synthetic_attention)
            target_rows.append([False] * length_bucket)
            request_rows.append(False)

        input_ids = mx.array(rows, dtype=mx.int32)
        attention_mask = mx.array(attention_rows, dtype=mx.bool_)
        target_mask = mx.array(target_rows, dtype=mx.bool_)
        request_mask = mx.array(request_rows, dtype=mx.bool_)
        positions = mx.cumsum(attention_mask.astype(mx.int32), axis=1) - 1
        positions = mx.where(
            attention_mask,
            positions,
            mx.zeros(positions.shape, dtype=mx.int32),
        )
        lease = None
        try:
            lease = self.buffer_pool.lease(
                batch_size=batch_size_bucket,
                capacity=length_bucket,
                config=self._resolved.model_config,
            )
            lease.token_storage[:, :] = input_ids
            log_likelihood, greedy_match = compiled(
                self._parameters,
                lease.token_storage,
                attention_mask,
                positions,
                target_mask,
                request_mask,
            )
            mx.eval(log_likelihood, greedy_match)
        finally:
            if lease is not None:
                self.buffer_pool.release(lease)

        likelihoods = log_likelihood.tolist()
        matches = greedy_match.tolist()
        return tuple(
            (float(likelihoods[index]), bool(matches[index]))
            for index in range(len(members))
        )

    def _compiled_scoring_kernel(
        self,
        length_bucket: int,
        batch_size_bucket: int,
        padding: str,
    ):
        key = ScoringKernelKey(length_bucket, batch_size_bucket, padding)
        compiled = self._scoring_compiled.get(key)
        if compiled is not None:
            return compiled

        model = self._model

        def _score(
            parameters,
            input_ids,
            attention_mask,
            positions,
            target_mask,
            request_mask,
        ):
            logits, _cache_state, _next_key = model.forward_arrays(
                parameters,
                input_ids,
                attention_mask=attention_mask,
                positions=positions,
                cache_state=None,
                training=False,
                key=None,
            )
            predictor = logits[:, :-1, :].astype(mx.float32)
            targets = input_ids[:, 1:]
            log_z = mx.logsumexp(predictor, axis=-1, keepdims=True)
            log_probs = predictor - log_z
            gathered = mx.take_along_axis(log_probs, targets[..., None], axis=-1)[
                ..., 0
            ]
            continuation_mask = target_mask[:, 1:]
            log_likelihood = mx.sum(
                gathered * continuation_mask.astype(mx.float32),
                axis=-1,
            ) * request_mask.astype(mx.float32)
            greedy = mx.argmax(predictor, axis=-1)
            matches = (greedy == targets) | (~continuation_mask)
            greedy_match = mx.all(matches, axis=-1) | (~request_mask)
            return log_likelihood, greedy_match

        compiled = mx.compile(_score)
        self._scoring_compiled[key] = compiled
        return compiled

    def _compiled_kernels(
        self,
        length_bucket: int,
        batch_size_bucket: int,
        kernel_key: GenerationKernelKey,
    ):
        key = (length_bucket, batch_size_bucket, kernel_key)
        compiled = self._compiled.get(key)
        if compiled is not None:
            return compiled

        model = self._model
        chunk_size = self._runtime.decode_chunk_size
        eos_id = self._resolved.model_config.eos_token_id
        temperature = kernel_key.temperature
        top_p = kernel_key.top_p
        repetition_penalty = kernel_key.repetition_penalty
        ngram_size = kernel_key.no_repeat_ngram_size

        def _forward(parameters, input_ids, attention_mask, positions, cache_state):
            logits, cache_state, _next_key = model.forward_arrays(
                parameters,
                input_ids,
                attention_mask=attention_mask,
                positions=positions,
                cache_state=cache_state,
                training=False,
                key=None,
            )
            return logits, cache_state

        def select_one_token(logits_row, key):
            return select_next_token_arrays(
                logits_row,
                key,
                temperature=temperature,
                top_p=top_p,
            )

        select_batch = mx.vmap(select_one_token, in_axes=(0, 0))

        def _decode_chunk_core(
            parameters,
            tokens,
            cache_state,
            logits,
            lengths,
            generated,
            finished,
            max_new,
            keys,
            request_mask,
        ):
            for _ in range(chunk_size):
                active = request_mask & (~finished) & (generated < max_new)
                scored = logits.astype(mx.float32)
                scored = apply_repetition_penalty(
                    scored, tokens, lengths, repetition_penalty
                )
                scored = apply_no_repeat_ngram(scored, tokens, lengths, ngram_size)
                selected, next_keys = select_batch(scored, keys)
                selected = mx.where(request_mask, selected, mx.zeros_like(selected))
                next_keys = mx.where(request_mask[:, None], next_keys, keys)
                selected_i32 = selected.astype(mx.int32)
                capacity = tokens.shape[1]
                positions = mx.clip(lengths, 0, capacity - 1)
                slot = (
                    mx.arange(capacity, dtype=mx.int32)[None, :] == positions[:, None]
                )
                write = slot & active[:, None]
                tokens = mx.where(
                    write,
                    mx.broadcast_to(selected_i32[:, None], tokens.shape),
                    tokens,
                )
                wrote = active
                new_lengths = mx.where(wrote, lengths + 1, lengths)
                new_generated = mx.where(wrote, generated + 1, generated)
                hit_eos = wrote & (selected_i32 == eos_id)
                finished = (
                    finished | ~request_mask | hit_eos | (new_generated >= max_new)
                )
                step_ids = selected_i32[:, None]
                step_mask = wrote[:, None]
                step_positions = lengths[:, None]
                logits, cache_state = _forward(
                    parameters,
                    step_ids,
                    step_mask,
                    step_positions,
                    cache_state,
                )
                logits = logits[:, 0]
                lengths = new_lengths
                generated = new_generated
                keys = next_keys
            return (
                tokens,
                cache_state,
                logits,
                lengths,
                generated,
                finished,
                keys,
            )

        compiled = (mx.compile(_forward), mx.compile(_decode_chunk_core))
        self._compiled[key] = compiled
        return compiled

    def _decode_chunk(self, bucket: GenerationBucket, lease: _Lease):
        prefill, decode_chunk = self._compiled_kernels(
            bucket.length_bucket,
            bucket.batch_size_bucket,
            bucket.kernel_key,
        )
        pad_id = self._resolved.model_config.pad_token_id
        capacity = bucket.length_bucket
        batch_size = bucket.batch_size_bucket
        rows = []
        for prompt in bucket.prompt_ids:
            row = [pad_id] * capacity
            row[: len(prompt)] = list(prompt)
            rows.append(row)
        lease.token_storage[:, :] = mx.array(rows, dtype=mx.int32)
        token_range = mx.arange(capacity, dtype=mx.int32)[None, :]
        real_mask = (
            token_range < bucket.prompt_lengths[:, None]
        ) & bucket.request_mask[:, None]
        synthetic_mask = (~bucket.request_mask)[:, None] & (token_range == 0)
        attention_mask = real_mask | synthetic_mask
        positions = mx.where(
            attention_mask,
            mx.broadcast_to(token_range, (batch_size, capacity)),
            mx.zeros((batch_size, capacity), dtype=mx.int32),
        )
        logits, cache_state = prefill(
            self._parameters,
            lease.token_storage,
            attention_mask,
            positions,
            lease.cache_state,
        )
        last_index = mx.clip(bucket.prompt_lengths - 1, 0, capacity - 1)
        gather_index = mx.broadcast_to(
            last_index[:, None, None],
            (batch_size, 1, logits.shape[-1]),
        )
        next_logits = mx.take_along_axis(logits, gather_index, axis=1)[:, 0, :]
        lengths = bucket.prompt_lengths.astype(mx.int32)
        generated = mx.zeros((batch_size,), dtype=mx.int32)
        finished = ~bucket.request_mask
        keys = bucket.keys
        max_new = bucket.max_new_tokens
        mx.eval(
            lease.token_storage,
            cache_state,
            next_logits,
            lengths,
            generated,
            finished,
            keys,
        )

        max_steps = bucket.host_max_new
        chunk_size = self._runtime.decode_chunk_size
        n_chunks = (max_steps + chunk_size - 1) // chunk_size if max_steps else 0
        tokens = lease.token_storage
        for _chunk in range(n_chunks):
            (
                tokens,
                cache_state,
                next_logits,
                lengths,
                generated,
                finished,
                keys,
            ) = decode_chunk(
                self._parameters,
                tokens,
                cache_state,
                next_logits,
                lengths,
                generated,
                finished,
                max_new,
                keys,
                bucket.request_mask,
            )
            mx.eval(
                tokens,
                cache_state,
                next_logits,
                lengths,
                generated,
                finished,
                keys,
            )
        lease.token_storage = tokens
        lease.cache_state = cache_state
        return self._host_results(bucket, tokens, generated)

    def _host_results(
        self,
        bucket: GenerationBucket,
        tokens: mx.array,
        generated: mx.array,
    ) -> tuple[tuple[int, GenerationResult], ...]:
        token_rows = tokens.tolist()
        generated_counts = generated.tolist()
        results: list[tuple[int, GenerationResult]] = []
        identity = self.model_identity
        processor = self._resolved.tokenizer.processor
        for index, caller_index in enumerate(bucket.caller_indices):
            if caller_index < 0:
                continue
            prompt = bucket.prompt_ids[index]
            start = len(prompt)
            count = int(generated_counts[index])
            continuation = tuple(
                int(token) for token in token_rows[index][start : start + count]
            )
            token_ids = (
                prompt + continuation if bucket.include_prompt[index] else continuation
            )
            results.append(
                (
                    caller_index,
                    GenerationResult(
                        text=processor.decode(list(token_ids)),
                        token_ids=token_ids,
                        seed=bucket.seeds[index],
                        model=identity,
                    ),
                )
            )
        return tuple(results)


__all__ = (
    "BufferPool",
    "GenerationBucket",
    "GenerationKernelKey",
    "GenerationRequest",
    "GenerationResult",
    "InferenceConfig",
    "InferenceRuntimeConfig",
    "InferenceSession",
    "ModelIdentity",
    "ResolvedModel",
    "ScoringKernelKey",
    "allocate_generation_seed",
    "infer",
    "load_owned_model_arrays",
    "resolve_model_artifact",
    "vmapped_select_one_token",
)
