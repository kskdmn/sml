from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
from sml.errors import SMLRuntimeError
from sml.evaluation import (
    SMLEvalLM,
)
from sml.inference import GenerationResult, InferenceSession


def test_sml_eval_lm_records_both_methods_in_dispatch_order(
    tiny_session: InferenceSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sml import evaluation

    recorder = evaluation._EvaluationRequestRecorder()
    lm = SMLEvalLM(tiny_session, recorder=recorder)
    monkeypatch.setattr(
        tiny_session,
        "generate_batch",
        lambda items: tuple(
            GenerationResult("ok", (), None, tiny_session.model_identity) for _ in items
        ),
    )
    generated = SimpleNamespace(
        task_name="hellaswag",
        doc_id=3,
        repeats=1,
        args=("alpha", {"max_gen_toks": 1, "until": []}),
    )
    scored = SimpleNamespace(
        task_name="hellaswag", doc_id=2, repeats=1, args=("alpha", " beta")
    )
    lm.generate_until([generated])
    lm.loglikelihood([scored])
    expected = evaluation._EvaluationRequestRecorder()
    expected.record("generate_until", (generated,))
    expected.record("loglikelihood", (scored,))
    assert recorder.identity_for("hellaswag") == expected.identity_for("hellaswag")


def test_generate_until_truncates_at_stop_strings(
    tiny_session: InferenceSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = tiny_session.model_identity

    def fake_generate_batch(items):
        return tuple(
            GenerationResult(
                text="hello world STOP more",
                token_ids=(1, 2, 3),
                seed=None,
                model=identity,
            )
            for _ in items
        )

    monkeypatch.setattr(tiny_session, "generate_batch", fake_generate_batch)
    lm = SMLEvalLM(tiny_session)
    completions = lm.generate_until(
        [
            SimpleNamespace(
                args=(
                    "alpha",
                    {"until": ["STOP"], "max_gen_toks": 8, "do_sample": False},
                )
            )
        ]
    )
    assert completions == ["hello world "]


def test_generate_until_batches_requests(
    tiny_session: InferenceSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    real_generate_batch = tiny_session.generate_batch

    def recording_generate_batch(items):
        calls.append(items)
        return real_generate_batch(items)

    monkeypatch.setattr(tiny_session, "generate_batch", recording_generate_batch)
    lm = SMLEvalLM(tiny_session)
    requests = [
        SimpleNamespace(args=("alpha", {"max_gen_toks": 1, "until": []})),
        SimpleNamespace(args=("beta", {"max_gen_toks": 1, "until": []})),
    ]
    completions = lm.generate_until(requests)
    assert len(calls) == 1
    assert len(calls[0]) == 2
    assert len(completions) == 2
    assert all(isinstance(completion, str) for completion in completions)


def test_generate_until_rejects_sampling(tiny_session: InferenceSession) -> None:
    lm = SMLEvalLM(tiny_session)
    with pytest.raises(SMLRuntimeError, match="greedy|sample"):
        lm.generate_until(
            [SimpleNamespace(args=("alpha", {"do_sample": True, "max_gen_toks": 1}))]
        )


def test_unsupported_eval_methods_raise(tiny_session: InferenceSession) -> None:
    lm = SMLEvalLM(tiny_session)
    with pytest.raises(SMLRuntimeError, match="unsupported"):
        lm.loglikelihood_rolling([])


def test_sml_eval_lm_loglikelihood_uses_batch_scorer(
    tiny_session: InferenceSession,
) -> None:
    lm = SMLEvalLM(tiny_session)
    request = SimpleNamespace(args=("alpha", " beta"))
    scored = lm.loglikelihood([request])
    assert len(scored) == 1
    log_likelihood, greedy_match = scored[0]
    assert math.isfinite(log_likelihood)
    assert greedy_match in (True, False)
