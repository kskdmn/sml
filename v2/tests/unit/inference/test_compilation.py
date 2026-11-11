from __future__ import annotations

import weakref

import mlx.core as mx
import pytest
from sml import inference
from sml.inference import (
    GenerationKernelKey,
    GenerationRequest,
    InferenceSession,
)
from sml.model.config import GenerationConfig


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


def test_prefill_uses_prompt_bucket_while_lease_uses_capacity_bucket(
    tiny_session: InferenceSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt = "alpha"
    request = GenerationRequest(max_new_tokens=16)
    bucket = tiny_session._bucketize(((prompt, request),))[0]
    assert bucket.prefill_length_bucket < bucket.cache_capacity_bucket

    captured: dict[str, int] = {}
    leased_storage: list[mx.array] = []
    prefill_rows: list[list[int]] = []
    storage_rows: list[list[int]] = []
    real_lease = tiny_session.buffer_pool.lease
    real_compiled = tiny_session._compiled_kernels

    def lease_spy(*, batch_size, capacity, config):
        captured["lease_batch"] = batch_size
        captured["leased_capacity"] = capacity
        lease = real_lease(batch_size=batch_size, capacity=capacity, config=config)
        captured["token_storage_capacity"] = int(lease.token_storage.shape[1])
        leased_storage.append(lease.token_storage)
        return lease

    def compiled_spy(
        prefill_length_bucket,
        cache_capacity_bucket,
        batch_size_bucket,
        kernel_key,
    ):
        captured["compiled_prefill_length"] = prefill_length_bucket
        captured["compiled_cache_capacity"] = cache_capacity_bucket
        prefill, decode = real_compiled(
            prefill_length_bucket,
            cache_capacity_bucket,
            batch_size_bucket,
            kernel_key,
        )

        def prefill_shape_spy(
            parameters,
            input_ids,
            attention_mask,
            positions,
            cache_state,
        ):
            captured["prefill_tokens"] = int(input_ids.shape[1])
            captured["prefill_mask"] = int(attention_mask.shape[1])
            captured["prefill_positions"] = int(positions.shape[1])
            captured["cache_capacity"] = int(cache_state[0][0].shape[2])
            prefill_rows.extend(input_ids.tolist())
            storage_rows.extend(
                leased_storage[0][:, : bucket.prefill_length_bucket].tolist()
            )
            logits, returned_cache = prefill(
                parameters,
                input_ids,
                attention_mask,
                positions,
                cache_state,
            )
            captured["prefill_logit_rows"] = int(logits.shape[1])
            return logits, returned_cache

        return prefill_shape_spy, decode

    monkeypatch.setattr(tiny_session.buffer_pool, "lease", lease_spy)
    monkeypatch.setattr(tiny_session, "_compiled_kernels", compiled_spy)
    tiny_session.generate(prompt, request)

    assert captured == {
        "lease_batch": bucket.batch_size_bucket,
        "leased_capacity": bucket.cache_capacity_bucket,
        "token_storage_capacity": bucket.cache_capacity_bucket,
        "compiled_prefill_length": bucket.prefill_length_bucket,
        "compiled_cache_capacity": bucket.cache_capacity_bucket,
        "prefill_tokens": bucket.prefill_length_bucket,
        "prefill_mask": bucket.prefill_length_bucket,
        "prefill_positions": bucket.prefill_length_bucket,
        "prefill_logit_rows": 1,
        "cache_capacity": bucket.cache_capacity_bucket,
    }
    expected_rows = [
        [
            *prompt_ids,
            *(
                [tiny_session.resolved_model.model_config.pad_token_id]
                * (bucket.prefill_length_bucket - len(prompt_ids))
            ),
        ]
        for prompt_ids in bucket.prompt_ids
    ]
    assert prefill_rows == expected_rows
    assert storage_rows == expected_rows


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


def test_generation_policy_cache_evicts_old_families_and_shares_prefill(
    tiny_session: InferenceSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(inference, "_COMPILATION_CACHE_LIMIT", 2)
    policies = [
        GenerationKernelKey.from_config(GenerationConfig(temperature=temperature))
        for temperature in (0.5, 0.6, 0.7)
    ]
    shape = (4, 16, 1)
    prefill, first_decode = tiny_session._compiled_kernels(*shape, policies[0])
    second_prefill, second_decode = tiny_session._compiled_kernels(*shape, policies[1])
    second_reference = weakref.ref(second_decode)
    del second_decode

    assert tiny_session._compiled_kernels(*shape, policies[0]) == (
        prefill,
        first_decode,
    )
    third_prefill, _third_decode = tiny_session._compiled_kernels(*shape, policies[2])

    assert prefill is second_prefill is third_prefill
    assert len(tiny_session._prefill_compiled) == 1
    assert len(tiny_session._compiled) == 2
    assert (*shape, policies[0]) in tiny_session._compiled
    assert (*shape, policies[1]) not in tiny_session._compiled
    assert second_reference() is None


@pytest.mark.parametrize("operation", ["prefill", "scoring"])
def test_shape_compilation_caches_release_least_recently_used_functions(
    tiny_session: InferenceSession,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    monkeypatch.setattr(inference, "_COMPILATION_CACHE_LIMIT", 2)
    real_compile = mx.compile

    def tracked_compile(function):
        compiled = real_compile(function)

        def call(*args, **kwargs):
            return compiled(*args, **kwargs)

        return call

    monkeypatch.setattr(mx, "compile", tracked_compile)
    policy = GenerationKernelKey.from_config(GenerationConfig())

    def get_kernel(capacity):
        if operation == "prefill":
            return tiny_session._compiled_kernels(2, capacity, 1, policy)[0]
        return tiny_session._compiled_scoring_kernel(capacity, 1, "right", 1)

    first = get_kernel(4)
    second = get_kernel(8)
    second_reference = weakref.ref(second)
    del second
    assert get_kernel(4) is first
    third = get_kernel(16)

    assert second_reference() is None
    assert get_kernel(4) is first
    assert get_kernel(16) is third


def test_compilation_eviction_preserves_seeded_generation(
    tiny_session: InferenceSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(inference, "_COMPILATION_CACHE_LIMIT", 2)
    requests = [
        GenerationRequest(
            max_new_tokens=3,
            config=GenerationConfig(temperature=temperature, top_p=0.9, seed=43),
        )
        for temperature in (0.8, 0.9, 1.0)
    ]

    expected = tiny_session.generate("alpha", requests[0])
    tiny_session.generate("alpha", requests[1])
    tiny_session.generate("alpha", requests[2])
    actual = tiny_session.generate("alpha", requests[0])

    assert actual == expected
    assert len(tiny_session._compiled) == 2
    assert len(tiny_session._prefill_compiled) == 1


def test_decode_variant_eviction_releases_compilations_and_replays_tokens(
    tiny_session: InferenceSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(inference, "_DECODE_VARIANT_CACHE_LIMIT", 1)
    real_compile = mx.compile
    compiled_references = []

    def tracked_compile(function):
        compiled = real_compile(function)

        def call(*args, **kwargs):
            return compiled(*args, **kwargs)

        compiled_references.append(weakref.ref(call))
        return call

    config = GenerationConfig(temperature=0.8, top_p=0.9, seed=43)
    policy = GenerationKernelKey.from_config(config)
    # Reserve the same shapes so only the requested chunk length changes.
    prefill, decode = tiny_session._compiled_kernels(4, 16, 1, policy)
    monkeypatch.setattr(mx, "compile", tracked_compile)
    cache = inference.allocate_kv_state(
        tiny_session.resolved_model.model_config, 1, 16, mx.bfloat16
    )
    input_ids = mx.array([[1, 4, 5, 6]], dtype=mx.int32)
    positions = mx.array([[0, 1, 2, 3]], dtype=mx.int32)
    attention = mx.ones((1, 4), dtype=mx.bool_)
    logits, cache = prefill(
        tiny_session._parameters, input_ids, attention, positions, cache
    )
    state = (
        tiny_session._parameters,
        mx.pad(input_ids, ((0, 0), (0, 12))),
        cache,
        logits[:, 0],
        mx.array([4], dtype=mx.int32),
        mx.array([0], dtype=mx.int32),
        mx.array([False]),
        mx.array([3], dtype=mx.int32),
        mx.random.key(43)[None, :],
        mx.array([True]),
    )
    expected = decode(1, True, *state)
    mx.eval(expected)
    other = decode(2, True, *state)
    mx.eval(other)
    assert compiled_references[0]() is None

    actual = decode(1, True, *state)
    mx.eval(actual)

    assert len(compiled_references) == 3
    assert compiled_references[1]() is None
    assert bool(mx.array_equal(actual[0], expected[0]))
    assert bool(mx.array_equal(actual[4], expected[4]))


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


def test_batch_cardinality_reuses_fixed_compiled_bucket(
    tiny_session: InferenceSession,
) -> None:
    tiny_session.generate_batch(seed_requests(3))
    compiled_after_three = set(tiny_session._compiled)
    tiny_session.generate_batch(seed_requests(4))
    assert set(tiny_session._compiled) == compiled_after_three
    assert len(compiled_after_three) == 1
    assert next(iter(compiled_after_three))[2] == 4
