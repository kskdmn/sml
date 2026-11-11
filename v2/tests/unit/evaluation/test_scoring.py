from __future__ import annotations

import io
import math
from types import SimpleNamespace

import mlx.core as mx
import pytest
import sentencepiece as spm
from sml import inference
from sml.errors import SMLRuntimeError
from sml.evaluation import (
    LoglikelihoodRequest,
    _encode_loglikelihood_request,
    score_loglikelihood_batch,
)
from sml.inference import InferenceSession
from sml_test_support.evaluation import (
    assert_loglikelihood_results_close,
    heterogeneous_scoring_requests,
    independent_scoring_oracle,
    scoring_requests,
)


def test_score_cardinality_reuses_fixed_compiled_bucket(
    tiny_session: InferenceSession,
) -> None:
    score_loglikelihood_batch(tiny_session, scoring_requests(3), padding="right")
    compiled_after_three = set(tiny_session._scoring_compiled)
    score_loglikelihood_batch(tiny_session, scoring_requests(4), padding="right")
    assert set(tiny_session._scoring_compiled) == compiled_after_three
    assert len(compiled_after_three) == 1
    assert next(iter(compiled_after_three)).batch_size_bucket == 4


def test_target_log_probabilities_gather_before_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictor = mx.array(
        [[[1.0, 2.0, 3.0], [3.0, 1.0, -1.0]]],
        dtype=mx.float32,
    )
    targets = mx.array([[2, 0]], dtype=mx.int32)
    calls: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = []
    original = inference.mx.take_along_axis

    def spy(array: mx.array, indices: mx.array, *, axis: int) -> mx.array:
        result = original(array, indices, axis=axis)
        calls.append(
            (
                tuple(array.shape),
                tuple(indices.shape),
                tuple(result.shape),
            )
        )
        return result

    monkeypatch.setattr(inference.mx, "take_along_axis", spy)
    actual = inference._target_log_probabilities(predictor, targets)
    expected = mx.array(
        [[-0.4076059644443804, -0.1429316284998996]],
        dtype=mx.float32,
    )
    mx.eval(actual, expected)

    assert tuple(actual.shape) == (1, 2)
    assert actual.dtype == mx.float32
    assert calls == [((1, 2, 3), (1, 2, 1), (1, 2, 1))]
    assert bool(mx.allclose(actual, expected, atol=1e-7, rtol=1e-7).item())


def test_compiled_scoring_uses_token_shaped_target_log_probabilities(
    tiny_session: InferenceSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = []
    original = inference._target_log_probabilities

    def spy(
        predictor_logits_fp32: mx.array,
        target_token_ids: mx.array,
    ) -> mx.array:
        result = original(predictor_logits_fp32, target_token_ids)
        calls.append(
            (
                tuple(predictor_logits_fp32.shape),
                tuple(target_token_ids.shape),
                tuple(result.shape),
            )
        )
        return result

    monkeypatch.setattr(inference, "_target_log_probabilities", spy)
    results = score_loglikelihood_batch(
        tiny_session,
        scoring_requests(3),
        padding="right",
    )

    assert len(results) == 3
    assert len(calls) == 1
    predictor_shape, target_shape, result_shape = calls[0]
    assert predictor_shape[:-1] == target_shape == result_shape
    assert predictor_shape[-1] == tiny_session.resolved_model.model_config.vocab_size


@pytest.mark.parametrize("padding", ["left", "right"])
def test_heterogeneous_batch_scoring_matches_oracle_without_generation_buffers(
    tiny_session: InferenceSession,
    monkeypatch: pytest.MonkeyPatch,
    padding: str,
) -> None:
    requests = heterogeneous_scoring_requests()
    expected = independent_scoring_oracle(tiny_session, requests)

    def forbid_kv_allocation(*_args, **_kwargs):
        pytest.fail("likelihood scoring must not allocate a generation KV cache")

    monkeypatch.setattr(inference, "allocate_kv_state", forbid_kv_allocation)
    actual = score_loglikelihood_batch(tiny_session, requests, padding=padding)

    assert_loglikelihood_results_close(
        actual,
        expected,
        atol=1e-5,
        rtol=1e-5,
    )
    assert score_loglikelihood_batch(tiny_session, requests, padding=padding) == actual
    assert tiny_session.buffer_pool.active_leases == 0
    assert not tiny_session.buffer_pool._free


def test_continuation_only_scoring_is_finite(tiny_session: InferenceSession) -> None:
    result = score_loglikelihood_batch(
        tiny_session,
        [LoglikelihoodRequest(context="", continuation="alpha")],
        padding="right",
    )[0]
    assert math.isfinite(result.log_likelihood)
    assert result.greedy_match in (True, False)


@pytest.fixture(scope="module")
def boundary_processor():
    model = io.BytesIO()
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(
            [
                "alpha beta gamma delta epsilon",
                "the quick brown fox jumps over the lazy dog",
                "there is a person walking and running",
            ]
            * 100
        ),
        model_writer=model,
        vocab_size=400,
        hard_vocab_limit=False,
        minloglevel=2,
        model_type="bpe",
        pad_id=3,
        byte_fallback=True,
        num_threads=1,
    )
    return spm.SentencePieceProcessor(model_proto=model.getvalue())


@pytest.mark.parametrize(
    ("context", "continuation", "start", "scored_pieces"),
    [
        ("the quick brow", "n fox", 2, ("▁brown", "▁fox")),
        ("猫 the quick brow", "n fox", 6, ("▁brown", "▁fox")),
        ("the quick ｂｒｏｗ", "ｎ fox", 2, ("▁brown", "▁fox")),
        ("alph", "a", 0, ("▁alpha",)),
        ("the quick brown", "🦊", 3, ("<0xF0>", "<0x9F>", "<0xA6>", "<0x8A>")),
        ("a", "\u0301", 1, ("<0xC3>", "<0xA1>")),
        ("the quick a", "\u0301", 3, ("<0xC3>", "<0xA1>")),
        ("猫 a", "\u0301", 5, ("<0xC3>", "<0xA1>")),
    ],
)
def test_continuation_boundary_uses_source_text(
    boundary_processor, context, continuation, start, scored_pieces
):
    processor = boundary_processor
    session = SimpleNamespace(
        resolved_model=SimpleNamespace(
            tokenizer=SimpleNamespace(processor=processor),
            model_config=SimpleNamespace(bos_token_id=processor.bos_id()),
        )
    )
    expected_ids = tuple(processor.encode(context + continuation))
    if start == 0:
        expected_ids = (processor.bos_id(), *expected_ids)
        start = 1

    token_ids, continuation_start = _encode_loglikelihood_request(
        session, LoglikelihoodRequest(context, continuation)
    )

    assert token_ids == expected_ids
    assert continuation_start == start
    assert tuple(processor.id_to_piece(token) for token in token_ids[start:]) == (
        scored_pieces
    )


def test_boundary_resegmentation_does_not_score_changed_context_tokens():
    model = io.BytesIO()
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(
            ["abcd"] * 10 + ["cd"] * 100 + ["bc"] * 50 + ["ab"] * 25
        ),
        model_writer=model,
        vocab_size=271,
        hard_vocab_limit=False,
        minloglevel=2,
        model_type="bpe",
        pad_id=3,
        byte_fallback=True,
        num_threads=1,
    )
    processor = spm.SentencePieceProcessor(model_proto=model.getvalue())
    assert processor.encode("abc", out_type=str) == ["▁", "a", "bc"]
    assert processor.encode("abcd", out_type=str) == ["▁ab", "cd"]
    session = SimpleNamespace(
        resolved_model=SimpleNamespace(
            tokenizer=SimpleNamespace(processor=processor),
            model_config=SimpleNamespace(bos_token_id=processor.bos_id()),
        )
    )

    token_ids, continuation_start = _encode_loglikelihood_request(
        session, LoglikelihoodRequest("abc", "d")
    )

    assert token_ids == tuple(processor.encode("abcd"))
    assert continuation_start == 1
    assert processor.id_to_piece(token_ids[continuation_start]) == "cd"


def test_empty_context_without_prefix_fails_before_bucketing(
    tiny_session: InferenceSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tiny_session.resolved_model.tokenizer.processor,
        "bos_id",
        lambda: -1,
    )
    with pytest.raises(SMLRuntimeError, match="prefix"):
        score_loglikelihood_batch(
            tiny_session,
            [LoglikelihoodRequest(context="", continuation="alpha")],
            padding="right",
        )
    assert not tiny_session._scoring_compiled


def test_nonempty_in_vocab_context_does_not_prepend_prefix(
    tiny_session: InferenceSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[tuple[tuple[int, ...], int]] = []
    original = tiny_session.score_encoded_loglikelihoods

    def spy(items, *, padding: str):
        captured.extend(items)
        return original(items, padding=padding)

    monkeypatch.setattr(tiny_session, "score_encoded_loglikelihoods", spy)
    score_loglikelihood_batch(
        tiny_session,
        [LoglikelihoodRequest(context="alpha", continuation=" beta")],
        padding="right",
    )
    assert captured
    token_ids, continuation_start = captured[0]
    processor = tiny_session.resolved_model.tokenizer.processor
    context_ids = tuple(int(token) for token in processor.encode("alpha"))
    assert token_ids[: len(context_ids)] == context_ids
    assert continuation_start == len(context_ids)


def test_compiled_scoring_kernel_receives_request_mask(
    tiny_session: InferenceSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[tuple[tuple[object, ...], dict[str, object]]] = []
    original = tiny_session._compiled_scoring_kernel

    def wrapping(
        length_bucket: int,
        batch_size_bucket: int,
        padding: str,
        continuation_length_bucket: int,
    ):
        compiled = original(
            length_bucket, batch_size_bucket, padding, continuation_length_bucket
        )

        def spy(*args, **kwargs):
            captured.append((args, kwargs))
            return compiled(*args, **kwargs)

        return spy

    monkeypatch.setattr(tiny_session, "_compiled_scoring_kernel", wrapping)
    score_loglikelihood_batch(tiny_session, scoring_requests(3), padding="right")
    assert captured
    args, kwargs = captured[0]
    values = list(args) + list(kwargs.values())
    found = False
    for value in values:
        if getattr(value, "ndim", None) != 1:
            continue
        converted = value.tolist()
        if converted == [True, True, True, False]:
            found = True
            break
    assert found, "compiled scoring kernel did not receive a boolean request mask"
