from __future__ import annotations

import inspect
import json
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import pytest
import zstandard as zstd
from sml import inference
from sml.artifacts import checkpoint
from sml.artifacts.checkpoint import VerifiedCheckpointContents
from sml.data.corpus import CorpusConfig
from sml.data.pretraining import (
    PretrainingPreparationConfig,
    prepare_pretraining_bundle,
)
from sml.data.tokenizer import TokenizerTrainingConfig, train_tokenizer_bundle
from sml.errors import SMLArtifactError, SMLRuntimeError
from sml.inference import (
    GenerationKernelKey,
    GenerationRequest,
    GenerationResult,
    InferenceRuntimeConfig,
    InferenceSession,
    resolve_model_artifact,
)
from sml.model.config import GenerationConfig, ModelConfig
from sml.training.common import (
    CheckpointPolicy,
    LoaderConfig,
    OptimizerConfig,
    PretrainingConfig,
    ResumeOverrides,
)
from sml.training.pretrain import resume, train

EXPECTED_SECOND_TOKENS = (7, 11)


def raise_after_one_token(*_args, **_kwargs):
    raise SMLRuntimeError("decode failed after one token")


def expected_second_tokens() -> tuple[int, ...]:
    return EXPECTED_SECOND_TOKENS


def _write_tiny_corpus(path: Path) -> Path:
    path.mkdir()
    lines = [
        json.dumps(
            {
                "text": (
                    "alpha beta gamma delta epsilon zeta eta theta " * 8
                    + f"row {index}"
                )
            }
        ).encode("utf-8")
        for index in range(24)
    ]
    payload = b"\n".join(lines) + b"\n"
    (path / "tiny-0000.jsonl.zst").write_bytes(zstd.ZstdCompressor().compress(payload))
    return path


def _corpus_config(path: Path) -> CorpusConfig:
    return CorpusConfig(
        input_root=path,
        shuffle_files=False,
        min_text_bytes=1,
        max_rows_per_file=None,
    )


def _prepare_tiny_data(root: Path):
    corpus = _write_tiny_corpus(root / "corpus")
    tokenizer = train_tokenizer_bundle(
        TokenizerTrainingConfig(
            corpus=_corpus_config(corpus),
            vocab_size=300,
            hard_vocab_limit=False,
            num_threads=1,
        ),
        root / "tokenizer",
    )
    data = prepare_pretraining_bundle(
        PretrainingPreparationConfig(
            corpus=_corpus_config(corpus),
            tokenizer_bundle=tokenizer.path,
            sequence_length=32,
            shuffle_window_rows=5,
            output_shard_rows=3,
            seed=17,
        ),
        root / "data",
    )
    return tokenizer, data


def _tiny_pretraining_config(data_path: Path, output_run: Path, vocab_size: int):
    return PretrainingConfig(
        data=data_path,
        output_run=output_run,
        model=ModelConfig(
            vocab_size=vocab_size,
            hidden_size=8,
            num_layers=1,
            num_q_heads=2,
            num_kv_heads=1,
            intermediate_size=16,
            original_context_length=32,
            rope_scaling_factor=1.0,
            hidden_dropout=0.0,
        ),
        optimizer=OptimizerConfig(
            learning_rate=0.01,
            beta1=0.5,
            beta2=0.5,
            schedule_steps=4,
            warmup_steps=0,
        ),
        loader=LoaderConfig(
            microbatch_size=1,
            gradient_accumulation_steps=1,
            prefetch_depth=2,
            epoch_seed=13,
        ),
        checkpoint=CheckpointPolicy(interval=1),
        maximum_steps=2,
        maximum_epochs=2,
        log_interval=1,
        seed=19,
        compile=False,
    )


@pytest.fixture(scope="module")
def _tiny_run_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("inference-unit")
    tokenizer, data = _prepare_tiny_data(root)
    result = train(
        _tiny_pretraining_config(
            data.path,
            root / "run",
            tokenizer.manifest.vocab_size,
        )
    )
    return result.run


@pytest.fixture
def tiny_pretraining_run(tmp_path: Path, _tiny_run_template: Path) -> Path:
    dest = tmp_path / "run"
    shutil.copytree(_tiny_run_template, dest)
    return dest


@pytest.fixture
def tiny_session(tiny_pretraining_run: Path) -> InferenceSession:
    return InferenceSession.from_checkpoint(tiny_pretraining_run)


def publish_new_valid_step(run: Path, step: int) -> None:
    resume(run, data=None, overrides=ResumeOverrides(maximum_steps=step))


def test_session_runtime_config_rejects_non_increasing_batch_buckets() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        InferenceRuntimeConfig(batch_size_buckets=(1, 2, 2, 8))


def test_session_runtime_config_rejects_non_positive_batch_buckets() -> None:
    with pytest.raises(ValueError, match="positive"):
        InferenceRuntimeConfig(batch_size_buckets=(1, 0, 4))


def test_resolve_rejects_direct_step_path(tmp_path: Path) -> None:
    step_path = tmp_path / "checkpoints" / "step-000000001"
    step_path.mkdir(parents=True)
    with pytest.raises(SMLArtifactError, match="step"):
        resolve_model_artifact(step_path, full_verify=False)


def test_resolve_rejects_historical_selector_on_step_directory(
    tiny_pretraining_run: Path,
) -> None:
    step_path = next((tiny_pretraining_run / "checkpoints").glob("step-*"))
    with pytest.raises(SMLArtifactError, match="step"):
        resolve_model_artifact(step_path, full_verify=False)


def test_verified_checkpoint_contents_mappings_are_immutable_for_session_resolve() -> (
    None
):
    scalar = {
        "kind": "pretraining-state",
        "cursor": {"epoch": 0, "shard_order_position": 0, "row_offset": 0},
    }
    inner = {"weight": mx.array([1.0], dtype=mx.float32)}
    groups = {"model.safetensors": inner}
    contents = VerifiedCheckpointContents(scalar, groups)

    with pytest.raises(TypeError):
        contents.scalar_state["kind"] = "mutated"
    with pytest.raises(TypeError):
        contents.scalar_state["cursor"]["epoch"] = 1
    with pytest.raises(TypeError):
        contents.array_groups["trainer.safetensors"] = {}
    with pytest.raises(TypeError):
        contents.array_groups["model.safetensors"]["weight"] = mx.array(
            [2.0], dtype=mx.float32
        )


def test_session_loads_latest_once_and_pins_identity(
    tiny_pretraining_run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0
    real_loader = inference.load_owned_model_arrays

    def counted_loader(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_loader(*args, **kwargs)

    monkeypatch.setattr(inference, "load_owned_model_arrays", counted_loader)
    session = InferenceSession.from_checkpoint(tiny_pretraining_run)
    first_identity = session.model_identity
    assert session.resolved_model.model_config.rope_scaling_factor == 1.0
    publish_new_valid_step(tiny_pretraining_run, step=4)
    assert session.model_identity == first_identity
    assert calls == 1


def test_failed_call_cannot_contaminate_next_call(
    tiny_session: InferenceSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tiny_session, "_generate_batch", raise_after_one_token)
    with pytest.raises(SMLRuntimeError, match="decode"):
        tiny_session.generate("first", GenerationRequest(max_new_tokens=2))

    def deterministic_batch(items):
        return tuple(
            GenerationResult(
                text="",
                token_ids=EXPECTED_SECOND_TOKENS,
                seed=request.config.seed,
                model=tiny_session.model_identity,
            )
            for _text, request in items
        )

    monkeypatch.setattr(tiny_session, "_generate_batch", deterministic_batch)
    assert (
        tiny_session.generate("second", GenerationRequest(max_new_tokens=2)).token_ids
        == expected_second_tokens()
    )


def test_overlapping_session_call_fails_before_state_mutation(
    tiny_session: InferenceSession,
) -> None:
    with (
        tiny_session._call_guard.acquire(),
        pytest.raises(SMLRuntimeError, match="non-reentrant"),
    ):
        tiny_session.generate("blocked", GenerationRequest(max_new_tokens=1))
    assert tiny_session.buffer_pool.active_leases == 0


def test_session_generate_returns_tokens_and_releases_lease(
    tiny_session: InferenceSession,
) -> None:
    result = tiny_session.generate("alpha", GenerationRequest(max_new_tokens=1))
    assert len(result.token_ids) == 1
    assert result.model == tiny_session.model_identity
    assert tiny_session.buffer_pool.active_leases == 0


def test_short_prompt_and_long_generation_select_independent_buckets(
    tiny_session: InferenceSession,
) -> None:
    prompt = "alpha"
    request = GenerationRequest(max_new_tokens=16)
    prompt_ids = tiny_session._encode_prompt(prompt)

    bucket = tiny_session._bucketize(((prompt, request),))[0]

    assert bucket.prefill_length_bucket == tiny_session._select_length_bucket(
        len(prompt_ids)
    )
    assert bucket.cache_capacity_bucket == tiny_session._select_length_bucket(
        len(prompt_ids) + request.max_new_tokens
    )
    assert bucket.prefill_length_bucket < bucket.cache_capacity_bucket


def test_generation_compile_cache_keys_both_length_domains(
    tiny_session: InferenceSession,
) -> None:
    prompt = "alpha"
    request = GenerationRequest(max_new_tokens=16)
    bucket = tiny_session._bucketize(((prompt, request),))[0]

    tiny_session.generate(prompt, request)

    assert (
        bucket.prefill_length_bucket,
        bucket.cache_capacity_bucket,
        bucket.batch_size_bucket,
        bucket.kernel_key,
    ) in tiny_session._compiled


def test_prompt_shape_change_with_same_capacity_uses_distinct_compile_keys(
    tiny_session: InferenceSession,
) -> None:
    short_prompt = "alpha"
    long_prompt = "alpha beta gamma delta"
    short_prompt_ids = tiny_session._encode_prompt(short_prompt)
    long_prompt_ids = tiny_session._encode_prompt(long_prompt)
    shared_required_length = len(long_prompt_ids) + 8
    short_request = GenerationRequest(
        max_new_tokens=shared_required_length - len(short_prompt_ids)
    )
    long_request = GenerationRequest(
        max_new_tokens=shared_required_length - len(long_prompt_ids)
    )
    short_bucket = tiny_session._bucketize(((short_prompt, short_request),))[0]
    long_bucket = tiny_session._bucketize(((long_prompt, long_request),))[0]
    assert short_bucket.prefill_length_bucket != long_bucket.prefill_length_bucket
    assert short_bucket.cache_capacity_bucket == long_bucket.cache_capacity_bucket

    tiny_session.generate(short_prompt, short_request)
    tiny_session.generate(long_prompt, long_request)

    assert len(tiny_session._compiled) == 2
    assert {key[0] for key in tiny_session._compiled} == {
        short_bucket.prefill_length_bucket,
        long_bucket.prefill_length_bucket,
    }


def test_capacity_change_with_same_prompt_shape_uses_distinct_compile_keys(
    tiny_session: InferenceSession,
) -> None:
    prompt = "alpha"
    small = GenerationRequest(max_new_tokens=1)
    large = GenerationRequest(max_new_tokens=8)
    small_bucket = tiny_session._bucketize(((prompt, small),))[0]
    large_bucket = tiny_session._bucketize(((prompt, large),))[0]
    assert small_bucket.prefill_length_bucket == large_bucket.prefill_length_bucket
    assert small_bucket.cache_capacity_bucket != large_bucket.cache_capacity_bucket

    tiny_session.generate(prompt, small)
    tiny_session.generate(prompt, large)

    assert len(tiny_session._compiled) == 2
    assert {key[1] for key in tiny_session._compiled} == {
        small_bucket.cache_capacity_bucket,
        large_bucket.cache_capacity_bucket,
    }


def test_session_compile_cache_reuses_shape_and_policy_key(
    tiny_session: InferenceSession,
) -> None:
    prompt = "alpha"
    prompt_ids = tiny_session._encode_prompt(prompt)
    small = GenerationRequest(max_new_tokens=1)
    large = GenerationRequest(max_new_tokens=8)
    sampled = GenerationRequest(
        max_new_tokens=1,
        config=GenerationConfig(temperature=0.8, seed=3),
    )
    small_bucket = tiny_session._select_length_bucket(len(prompt_ids) + 1)
    large_bucket = tiny_session._select_length_bucket(len(prompt_ids) + 8)
    assert small_bucket != large_bucket

    tiny_session.generate(prompt, small)
    tiny_session.generate(prompt, small)
    assert len(tiny_session._compiled) == 1
    first_key = next(iter(tiny_session._compiled))
    assert first_key == (
        tiny_session._select_length_bucket(len(prompt_ids)),
        small_bucket,
        1,
        GenerationKernelKey(
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.0,
            no_repeat_ngram_size=0,
        ),
    )

    tiny_session.generate(prompt, large)
    tiny_session.generate(prompt, sampled)
    assert len(tiny_session._compiled) == 3
    assert (
        tiny_session._select_length_bucket(len(prompt_ids)),
        large_bucket,
        1,
        GenerationKernelKey(
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.0,
            no_repeat_ngram_size=0,
        ),
    ) in tiny_session._compiled
    assert (
        tiny_session._select_length_bucket(len(prompt_ids)),
        small_bucket,
        1,
        GenerationKernelKey(
            temperature=0.8,
            top_p=1.0,
            repetition_penalty=1.0,
            no_repeat_ngram_size=0,
        ),
    ) in tiny_session._compiled


def test_decode_chunk_does_not_call_item_and_returns_continuation(
    tiny_pretraining_run: Path,
) -> None:
    source = inspect.getsource(InferenceSession._decode_chunk)
    assert ".item(" not in source
    session = InferenceSession.from_checkpoint(
        tiny_pretraining_run,
        runtime=InferenceRuntimeConfig(decode_chunk_size=2),
    )
    result = session.generate("alpha", GenerationRequest(max_new_tokens=5))
    assert result.token_ids
    assert len(result.token_ids) <= 5


def test_session_reuses_pooled_token_storage_for_same_bucket(
    tiny_session: InferenceSession,
) -> None:
    leased: list[object] = []
    released: list[object] = []
    pool = tiny_session.buffer_pool
    real_lease = pool.lease
    real_release = pool.release

    def wrapped_lease(**kwargs):
        lease = real_lease(**kwargs)
        leased.append(lease.token_storage)
        return lease

    def wrapped_release(lease):
        released.append(lease.token_storage)
        return real_release(lease)

    pool.lease = wrapped_lease
    pool.release = wrapped_release
    request = GenerationRequest(max_new_tokens=1)
    tiny_session.generate("alpha", request)
    tiny_session.generate("alpha", request)
    assert pool.active_leases == 0
    assert len(leased) == 2
    assert len(released) == 2
    assert leased[1] is released[0]


def _kv_arrays(cache_state: object) -> tuple[object, ...]:
    keys, values, lengths = cache_state
    return (*keys, *values, lengths)


def test_session_reuses_pooled_kv_arrays_for_same_bucket(
    tiny_session: InferenceSession,
) -> None:
    leased: list[tuple[object, ...]] = []
    released: list[tuple[object, ...]] = []
    pool = tiny_session.buffer_pool
    real_lease = pool.lease
    real_release = pool.release

    def wrapped_lease(**kwargs):
        lease = real_lease(**kwargs)
        leased.append(_kv_arrays(lease.cache_state))
        return lease

    def wrapped_release(lease):
        released.append(_kv_arrays(lease.cache_state))
        return real_release(lease)

    pool.lease = wrapped_lease
    pool.release = wrapped_release
    request = GenerationRequest(max_new_tokens=1)
    tiny_session.generate("alpha", request)
    tiny_session.generate("alpha", request)
    assert pool.active_leases == 0
    assert len(leased) == 2
    assert len(released) == 2
    assert len(leased[1]) == len(released[0])
    for reused, previous in zip(leased[1], released[0], strict=True):
        assert reused is previous


def test_failed_lease_does_not_increment_active_leases(
    tiny_session: InferenceSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = tiny_session.buffer_pool
    assert pool.active_leases == 0

    def boom(*_args, **_kwargs):
        raise RuntimeError("alloc failed")

    monkeypatch.setattr(inference.mx, "zeros", boom)
    with pytest.raises(RuntimeError, match="alloc failed"):
        pool.lease(
            batch_size=1,
            capacity=8,
            config=tiny_session.resolved_model.model_config,
        )
    assert pool.active_leases == 0


def test_load_owned_model_arrays_does_not_nest_run_access_lock(
    tiny_pretraining_run: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    depth = 0
    max_depth = 0
    real_lock = checkpoint._protected_lock

    @contextmanager
    def tracking(protected, *, category, exclusive, wait=False):
        nonlocal depth, max_depth
        if category == "run-access":
            depth += 1
            max_depth = max(max_depth, depth)
        try:
            with real_lock(
                protected,
                category=category,
                exclusive=exclusive,
                wait=wait,
            ):
                yield
        finally:
            if category == "run-access":
                depth -= 1

    monkeypatch.setattr(checkpoint, "_protected_lock", tracking)
    inference.load_owned_model_arrays(tiny_pretraining_run, full_verify=False)
    assert max_depth == 1


inference_module = inference


@dataclass(frozen=True, slots=True)
class _CompileSpyKey:
    prefill_length_bucket: int
    cache_capacity_bucket: int
    batch_size_bucket: int
    kernel_key: GenerationKernelKey


class _CompileSpy:
    def __init__(self, session: InferenceSession) -> None:
        self._session = session

    @property
    def keys(self) -> set[_CompileSpyKey]:
        compiled = self._session._compiled
        keys: set[_CompileSpyKey] = set()
        for key in compiled:
            if isinstance(key, tuple) and len(key) == 4:
                keys.add(_CompileSpyKey(key[0], key[1], key[2], key[3]))
            else:
                keys.add(
                    _CompileSpyKey(
                        key.prefill_length_bucket,
                        key.cache_capacity_bucket,
                        key.batch_size_bucket,
                        key.kernel_key,
                    )
                )
        return keys


@pytest.fixture
def compile_spy(tiny_session: InferenceSession) -> _CompileSpy:
    return _CompileSpy(tiny_session)


def source_contains(module: object, snippet: str) -> bool:
    return snippet in inspect.getsource(module)


def source_has_none_of(module: object, forbidden: list[str]) -> bool:
    source = inspect.getsource(module)
    return all(token not in source for token in forbidden)


def seed_requests(count: int) -> list[tuple[str, GenerationRequest]]:
    return [
        (
            "alpha",
            GenerationRequest(
                max_new_tokens=2,
                config=GenerationConfig(temperature=0.8, top_p=0.9, seed=21 + index),
            ),
        )
        for index in range(count)
    ]


def fixed_bucket_logits(*, batch_size: int) -> mx.array:
    peak = mx.array([8.0, 0.0, -8.0, -8.0], dtype=mx.float32)
    return mx.broadcast_to(peak[None, :], (batch_size, peak.shape[0]))


def vmapped_select_one_token(logits, keys, request_mask, kernel_key):
    return inference.vmapped_select_one_token(
        logits,
        keys,
        request_mask,
        kernel_key,
    )


def test_empty_batch_returns_before_taking_the_call_guard(
    tiny_session: InferenceSession,
) -> None:
    with tiny_session._call_guard.acquire():
        assert tiny_session.generate_batch([]) == ()
    assert tiny_session.buffer_pool.active_leases == 0


def test_infer_constructs_one_session_and_delegates_once_to_generate(
    tiny_pretraining_run: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sml.inference import InferenceConfig, infer

    calls: list[tuple[str, GenerationRequest]] = []
    real_generate = InferenceSession.generate

    def tracking(self, text, request):
        calls.append((text, request))
        return real_generate(self, text, request)

    monkeypatch.setattr(InferenceSession, "generate", tracking)
    request = GenerationRequest(max_new_tokens=1)
    result = infer(
        InferenceConfig(
            checkpoint=tiny_pretraining_run,
            prompt="alpha",
            request=request,
        )
    )
    assert len(calls) == 1
    assert calls[0] == ("alpha", request)
    assert result.token_ids
    assert result.model.run_identity is not None


def test_generate_delegates_to_the_same_batch_engine(
    tiny_session: InferenceSession,
) -> None:
    request = GenerationRequest(
        max_new_tokens=2,
        config=GenerationConfig(temperature=0.8, top_p=0.9, seed=41),
    )
    single = tiny_session.generate("alpha", request)
    batched = tiny_session.generate_batch([("alpha", request)])[0]
    assert single.token_ids == batched.token_ids
    assert single.seed == batched.seed == 41


def test_batch_cardinality_reuses_fixed_compiled_bucket(
    tiny_session: InferenceSession, compile_spy: _CompileSpy
) -> None:
    tiny_session.generate_batch(seed_requests(3))
    compiled_after_three = set(compile_spy.keys)
    tiny_session.generate_batch(seed_requests(4))
    assert set(compile_spy.keys) == compiled_after_three
    assert len(compiled_after_three) == 1
    assert next(iter(compiled_after_three)).batch_size_bucket == 4


def test_sampling_vmaps_scalar_keys_and_ignores_synthetic_slots(
    tiny_session: InferenceSession,
) -> None:
    bucket = tiny_session._bucketize(seed_requests(3))[0]
    assert bucket.keys.shape == (4, 2)
    assert bucket.request_mask.tolist() == [True, True, True, False]
    selected, next_keys = vmapped_select_one_token(
        fixed_bucket_logits(batch_size=4),
        bucket.keys,
        bucket.request_mask,
        bucket.kernel_key,
    )
    mx.eval(selected, next_keys)
    assert selected.shape == (4,)
    assert next_keys.shape == (4, 2)
    assert source_contains(inference_module, "mx.vmap(select_one_token")
    assert source_has_none_of(
        inference_module, ["mx.random.categorical(logits, key=keys)"]
    )
