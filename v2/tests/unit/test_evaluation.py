# ruff: noqa: F811
from __future__ import annotations

import math
import socket
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from sml.errors import SMLRuntimeError
from sml.evaluation import (
    EvaluationConfig,
    LoglikelihoodRequest,
    SMLEvalLM,
    evaluate,
    score_loglikelihood_batch,
)
from sml.inference import GenerationResult, InferenceSession
from test_inference import (  # noqa: F401
    _tiny_run_template,
    publish_new_valid_step,
    tiny_pretraining_run,
    tiny_session,
)

IN_VOCAB_CONTEXTS = ("alpha", "alpha beta", "alpha beta gamma")
IN_VOCAB_CONTINUATIONS = (" beta", " gamma", " delta")


@dataclass(frozen=True, slots=True)
class _ScoringCompileKey:
    length_bucket: int
    batch_size_bucket: int
    padding: str


class _ScoringCompileSpy:
    def __init__(self, session: InferenceSession) -> None:
        self._session = session

    @property
    def scoring_keys(self) -> set[_ScoringCompileKey]:
        compiled = getattr(self._session, "_scoring_compiled", {})
        keys: set[_ScoringCompileKey] = set()
        for key in compiled:
            if isinstance(key, tuple) and len(key) == 3:
                keys.add(_ScoringCompileKey(key[0], key[1], key[2]))
            else:
                keys.add(
                    _ScoringCompileKey(
                        key.length_bucket,
                        key.batch_size_bucket,
                        key.padding,
                    )
                )
        return keys


@pytest.fixture
def compile_spy(tiny_session: InferenceSession) -> _ScoringCompileSpy:
    return _ScoringCompileSpy(tiny_session)


def scoring_requests(count: int) -> tuple[LoglikelihoodRequest, ...]:
    return tuple(
        LoglikelihoodRequest(context="alpha", continuation=" beta")
        for _ in range(count)
    )


def heterogeneous_scoring_requests() -> tuple[LoglikelihoodRequest, ...]:
    return tuple(
        LoglikelihoodRequest(context=context, continuation=continuation)
        for context, continuation in zip(
            IN_VOCAB_CONTEXTS,
            IN_VOCAB_CONTINUATIONS,
            strict=True,
        )
    )


def assert_loglikelihood_results_close(
    left: tuple,
    right: tuple,
    *,
    atol: float,
    rtol: float,
) -> None:
    assert len(left) == len(right)
    for actual, expected in zip(left, right, strict=True):
        assert actual.greedy_match is expected.greedy_match
        assert actual.log_likelihood == pytest.approx(
            expected.log_likelihood,
            abs=atol,
            rel=rtol,
        )


def tiny_evaluation_config(
    run: Path,
    tmp_path: Path,
    tasks: tuple[str, ...] = ("hellaswag",),
) -> EvaluationConfig:
    return EvaluationConfig(
        checkpoint=run,
        tasks=tasks,
        output=tmp_path / "evaluation.json",
    )


@pytest.fixture
def fake_lm_eval(monkeypatch: pytest.MonkeyPatch):
    from sml import evaluation

    calls: list[dict[str, object]] = []

    def simple_evaluate(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        tasks = tuple(kwargs.get("tasks") or ())
        return {"results": {task: {"acc,none": 0.0} for task in tasks}}

    module = SimpleNamespace(simple_evaluate=simple_evaluate, calls=calls)
    monkeypatch.setattr(evaluation, "_import_lm_eval", lambda: module)
    return module


def test_score_cardinality_reuses_fixed_compiled_bucket(
    tiny_session: InferenceSession,
    compile_spy: _ScoringCompileSpy,
) -> None:
    score_loglikelihood_batch(tiny_session, scoring_requests(3), padding="right")
    compiled_after_three = set(compile_spy.scoring_keys)
    score_loglikelihood_batch(tiny_session, scoring_requests(4), padding="right")
    assert set(compile_spy.scoring_keys) == compiled_after_three
    assert len(compiled_after_three) == 1
    assert next(iter(compiled_after_three)).batch_size_bucket == 4


def test_continuation_only_scoring_is_finite(tiny_session: InferenceSession) -> None:
    result = score_loglikelihood_batch(
        tiny_session,
        [LoglikelihoodRequest(context="", continuation="alpha")],
        padding="right",
    )[0]
    assert math.isfinite(result.log_likelihood)
    assert result.greedy_match in (True, False)


def test_greedy_match_is_boolean_and_stable(tiny_session: InferenceSession) -> None:
    request = LoglikelihoodRequest(context="alpha", continuation=" beta")
    first = score_loglikelihood_batch(tiny_session, [request], padding="right")[0]
    second = score_loglikelihood_batch(tiny_session, [request], padding="right")[0]
    assert first.greedy_match in (True, False)
    assert first.greedy_match is second.greedy_match
    assert first.log_likelihood == pytest.approx(
        second.log_likelihood, abs=2e-2, rel=2e-2
    )


def test_empty_context_inserts_prefix_token(tiny_session: InferenceSession) -> None:
    processor = tiny_session.resolved_model.tokenizer.processor
    assert int(processor.bos_id()) >= 0
    result = score_loglikelihood_batch(
        tiny_session,
        [LoglikelihoodRequest(context="", continuation=" beta")],
        padding="left",
    )[0]
    assert math.isfinite(result.log_likelihood)


def test_empty_context_without_prefix_fails_before_bucketing(
    tiny_session: InferenceSession,
    compile_spy: _ScoringCompileSpy,
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
    assert compile_spy.scoring_keys == set()


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


def test_evaluation_config_requires_tasks(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="task"):
        EvaluationConfig(
            checkpoint=tmp_path,
            tasks=(),
            output=tmp_path / "out.json",
        )


def test_evaluation_config_requires_positive_limit(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="limit"):
        EvaluationConfig(
            checkpoint=tmp_path,
            tasks=("hellaswag",),
            output=tmp_path / "out.json",
            limit=0,
        )


def test_evaluation_config_allows_repeated_supported_tasks(tmp_path: Path) -> None:
    config = EvaluationConfig(
        checkpoint=tmp_path,
        tasks=("hellaswag", "winogrande", "hellaswag"),
        output=tmp_path / "out.json",
    )
    assert config.tasks == ("hellaswag", "winogrande", "hellaswag")


def test_evaluation_config_rejects_unknown_tasks(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="hellaswag|winogrande"):
        EvaluationConfig(
            checkpoint=tmp_path,
            tasks=("mmlu",),
            output=tmp_path / "out.json",
        )


def test_importing_evaluation_does_not_import_lm_eval() -> None:
    from sml import evaluation

    assert "lm_eval" not in evaluation.__dict__
    source = Path(evaluation.__file__).read_text(encoding="utf-8")
    tree_import_lines = [
        line for line in source.splitlines() if line.startswith(("import ", "from "))
    ]
    assert all("lm_eval" not in line for line in tree_import_lines)


def test_evaluate_does_not_use_the_network(
    tiny_pretraining_run: Path,
    fake_lm_eval,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fail_network(*_args, **_kwargs):
        raise AssertionError("network")

    monkeypatch.setattr(socket, "create_connection", fail_network)
    result = evaluate(tiny_evaluation_config(tiny_pretraining_run, tmp_path))
    assert result.output.exists()
    assert fake_lm_eval.calls


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
