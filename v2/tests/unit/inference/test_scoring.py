from __future__ import annotations

import mlx.core as mx
import pytest
from sml import inference
from sml.errors import SMLRuntimeError
from sml.inference import (
    InferenceSession,
)


@pytest.mark.parametrize("token_ids", [(), (4,), (4, 5), (4, 5, 6)])
def test_zero_context_scoring_uses_bos_independently_of_padding(
    tiny_session: InferenceSession, token_ids: tuple[int, ...]
) -> None:
    bos_id = tiny_session.resolved_model.model_config.bos_token_id
    expected = tiny_session.score_encoded_loglikelihoods(
        (((bos_id, *token_ids), 1),), padding="right"
    )[0]

    for padding in ("left", "right"):
        actual = tiny_session.score_encoded_loglikelihoods(
            ((token_ids, 0),), padding=padding
        )[0]
        assert actual[0] == pytest.approx(expected[0], abs=1e-6)
        assert actual[1] == expected[1]
    if not token_ids:
        assert expected == (0.0, True)


def test_zero_context_scoring_counts_bos_toward_capacity(
    tiny_session: InferenceSession,
) -> None:
    capacity = tiny_session.resolved_model.model_config.effective_context_length
    with pytest.raises(SMLRuntimeError, match="context length"):
        tiny_session.score_encoded_loglikelihoods(
            (((4,) * capacity, 0),), padding="right"
        )


@pytest.mark.parametrize("padding", ["left", "right"])
def test_empty_continuations_preserve_mixed_batch_order(
    tiny_session: InferenceSession, padding: str
) -> None:
    nonempty = (((4, 5), 1), ((4, 5, 6), 1))
    expected = tiny_session.score_encoded_loglikelihoods(nonempty, padding=padding)
    actual = tiny_session.score_encoded_loglikelihoods(
        (nonempty[0], ((4,), 1), nonempty[1], ((), 0)), padding=padding
    )
    assert actual == (expected[0], (0.0, True), expected[1], (0.0, True))


@pytest.mark.parametrize("continuation_start", [-1, 4, True, 0.5])
def test_encoded_scoring_rejects_invalid_continuation_indices(
    tiny_session: InferenceSession, continuation_start: object
) -> None:
    with pytest.raises(SMLRuntimeError, match="invalid continuation start"):
        tiny_session.score_encoded_loglikelihoods(
            (((4, 5, 6), continuation_start),), padding="right"
        )


@pytest.mark.parametrize("padding", ["left", "right"])
def test_scoring_projects_only_bucketed_continuation_predictors(
    tiny_session: InferenceSession,
    monkeypatch: pytest.MonkeyPatch,
    padding: str,
) -> None:
    items = (
        ((1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), 12),
        ((1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), 9),
        ((1, 6, 7, 8, 9, 10, 11, 12, 13, 14), 7),
    )
    expected = []
    forward = tiny_session._model.forward_arrays
    for token_ids, start in items:
        input_ids = mx.array([token_ids], dtype=mx.int32)
        logits, _, _ = forward(
            tiny_session._parameters,
            input_ids,
            attention_mask=None,
            positions=None,
            cache_state=None,
            training=False,
            key=None,
        )
        predictor = logits[:, start - 1 : -1, :].astype(mx.float32)
        targets = input_ids[:, start:]
        likelihood = mx.sum(inference._target_log_probabilities(predictor, targets))
        greedy_match = mx.all(mx.argmax(predictor, axis=-1) == targets)
        mx.eval(likelihood, greedy_match)
        expected.append((float(likelihood.item()), bool(greedy_match.item())))

    projected_shapes: list[tuple[int, ...]] = []

    def selected_forward(parameters, input_ids, **kwargs):
        projected_shapes.append(tuple(kwargs["logits_positions"].shape))
        return forward(parameters, input_ids, **kwargs)

    monkeypatch.setattr(tiny_session._model, "forward_arrays", selected_forward)
    actual = tiny_session.score_encoded_loglikelihoods(items, padding=padding)

    assert projected_shapes == [(4, 4)]
    for result, reference in zip(actual, expected, strict=True):
        assert result[0] == pytest.approx(reference[0], abs=1e-5, rel=1e-5)
        assert result[1] == reference[1]
    assert {
        key.continuation_length_bucket for key in tiny_session._scoring_compiled
    } == {4}


def test_scoring_reuses_continuation_shape_buckets(tiny_session: InferenceSession):
    token_ids = tuple(range(4, 17))
    tiny_session.score_encoded_loglikelihoods(((token_ids, 10),), padding="right")
    initial = dict(tiny_session._scoring_compiled)
    tiny_session.score_encoded_loglikelihoods(((token_ids, 9),), padding="right")
    assert tiny_session._scoring_compiled == initial

    tiny_session.score_encoded_loglikelihoods(((token_ids, 8),), padding="right")
    assert len(tiny_session._scoring_compiled) == 2
    assert {
        key.continuation_length_bucket for key in tiny_session._scoring_compiled
    } == {
        4,
        8,
    }
