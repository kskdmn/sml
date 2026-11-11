from __future__ import annotations

import importlib
import math
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest
from sml.evaluation import (
    EvaluationConfig,
    LoglikelihoodRequest,
)
from sml.inference import InferenceSession

IN_VOCAB_CONTEXTS = ("alpha", "alpha beta", "alpha beta gamma")
IN_VOCAB_CONTINUATIONS = (" beta", " gamma", " delta")
_PROVIDER_STUBS = Path(__file__).resolve().parents[1] / "fixtures" / "provider_stubs"


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


def independent_scoring_oracle(
    session: InferenceSession,
    requests: tuple[LoglikelihoodRequest, ...],
) -> tuple[SimpleNamespace, ...]:
    processor = session.resolved_model.tokenizer.processor
    results: list[SimpleNamespace] = []
    for request in requests:
        context_ids = tuple(int(token) for token in processor.encode(request.context))
        token_ids = tuple(
            int(token)
            for token in processor.encode(request.context + request.continuation)
        )
        assert token_ids[: len(context_ids)] == context_ids
        assert context_ids
        assert len(token_ids) > len(context_ids)

        input_ids = mx.array([token_ids], dtype=mx.int32)
        attention_mask = mx.ones(input_ids.shape, dtype=mx.bool_)
        positions = mx.arange(len(token_ids), dtype=mx.int32)[None, :]
        logits, _cache_state, _next_key = session._model.forward_arrays(
            session._parameters,
            input_ids,
            attention_mask=attention_mask,
            positions=positions,
            cache_state=None,
            training=False,
            key=None,
        )
        predictor_rows = logits[0, :-1, :].astype(mx.float32)
        mx.eval(predictor_rows)

        log_likelihood = 0.0
        greedy_match = True
        host_predictors = predictor_rows.tolist()
        for target_position in range(len(context_ids), len(token_ids)):
            row = host_predictors[target_position - 1]
            maximum = max(row)
            log_normalizer = maximum + math.log(
                math.fsum(math.exp(value - maximum) for value in row)
            )
            target_id = token_ids[target_position]
            log_likelihood += row[target_id] - log_normalizer
            greedy_id = max(range(len(row)), key=row.__getitem__)
            greedy_match = greedy_match and greedy_id == target_id
        results.append(
            SimpleNamespace(
                log_likelihood=log_likelihood,
                greedy_match=greedy_match,
            )
        )
    return tuple(results)


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
def fake_lm_eval(fake_provider, monkeypatch: pytest.MonkeyPatch):
    from sml import evaluation

    calls: list[dict[str, object]] = []

    def simple_evaluate(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        raw_result = fake_provider.simple_evaluate(**kwargs)
        if module.result is not None:
            return module.result
        raw_result = dict(raw_result)
        raw_result["results"] = {
            task_name: {"acc,none": 0.5, "acc_stderr,none": 0.01}
            for task_name in raw_result["results"]
        }
        module.result = raw_result
        return raw_result

    module = SimpleNamespace(result=None, calls=calls)
    monkeypatch.setattr(
        evaluation,
        "_import_lm_eval",
        lambda: replace(fake_provider, simple_evaluate=simple_evaluate),
    )
    return module


@pytest.fixture
def fake_provider(monkeypatch: pytest.MonkeyPatch):
    """Load the checked-in 0.4.12-shaped provider without using the network."""
    from sml import evaluation

    original = {
        name: module
        for name, module in sys.modules.items()
        if name == "lm_eval" or name.startswith("lm_eval.")
    }
    for name in tuple(original):
        del sys.modules[name]
    monkeypatch.syspath_prepend(str(_PROVIDER_STUBS))
    importlib.invalidate_caches()
    module = importlib.import_module("lm_eval")
    task_module = importlib.import_module("lm_eval.tasks")
    loader_module = importlib.import_module("lm_eval.tasks._yaml_loader")
    try:
        yield evaluation._LMEvalProvider(
            simple_evaluate=module.simple_evaluate,
            task_manager_type=task_module.TaskManager,
            load_yaml=loader_module.load_yaml,
            package_root=Path(module.__file__).resolve().parent,
            package_version=module.__version__,
        )
    finally:
        for name in tuple(sys.modules):
            if name == "lm_eval" or name.startswith("lm_eval."):
                del sys.modules[name]
        sys.modules.update(original)
