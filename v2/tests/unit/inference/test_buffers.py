from __future__ import annotations

import mlx.core as mx
import pytest
from sml import inference
from sml.inference import (
    GenerationRequest,
    InferenceSession,
)
from sml.model.config import GenerationConfig


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


def test_reused_token_storage_tail_does_not_affect_generation(
    tiny_session: InferenceSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt = "alpha"
    request = GenerationRequest(
        max_new_tokens=16,
        config=GenerationConfig(
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.2,
            no_repeat_ngram_size=2,
            seed=71,
        ),
    )
    bucket = tiny_session._bucketize(((prompt, request),))[0]
    assert bucket.prefill_length_bucket < bucket.cache_capacity_bucket
    expected = tiny_session.generate(prompt, request)

    pool = tiny_session.buffer_pool
    primed = pool.lease(
        batch_size=bucket.batch_size_bucket,
        capacity=bucket.cache_capacity_bucket,
        config=tiny_session.resolved_model.model_config,
    )
    sentinel = tiny_session._encode_prompt(prompt)[-1]
    primed.token_storage[:, :] = mx.full(
        primed.token_storage.shape,
        sentinel,
        dtype=mx.int32,
    )
    mx.eval(primed.token_storage)
    primed_storage = primed.token_storage
    pool.release(primed)

    reused: list[mx.array] = []
    reused_tails: list[list[list[int]]] = []
    real_lease = pool.lease

    def lease_spy(**kwargs):
        lease = real_lease(**kwargs)
        reused.append(lease.token_storage)
        reused_tails.append(
            lease.token_storage[:, bucket.prefill_length_bucket :].tolist()
        )
        return lease

    monkeypatch.setattr(pool, "lease", lease_spy)
    actual = tiny_session.generate(prompt, request)

    assert len(reused) == 1
    assert reused[0] is primed_storage
    assert reused_tails == [
        [[sentinel] * (bucket.cache_capacity_bucket - bucket.prefill_length_bucket)]
    ]
    assert actual.token_ids == expected.token_ids


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
