# ruff: noqa: F811
from __future__ import annotations

import math
import sys
from dataclasses import replace
from pathlib import Path

_UNIT_DIR = Path(__file__).resolve().parents[1] / "unit"
if str(_UNIT_DIR) not in sys.path:
    sys.path.insert(0, str(_UNIT_DIR))

import pytest
from sml.errors import SMLRuntimeError
from sml.evaluation import (
    LoglikelihoodRequest,
    evaluate,
    read_evaluation_result,
    score_loglikelihood_batch,
)
from sml.evaluation_result import evaluation_result_identity
from sml.inference import InferenceSession
from test_evaluation import (  # noqa: F401
    fake_lm_eval,
    fake_provider,
    tiny_evaluation_config,
)
from test_inference import (  # noqa: F401
    _tiny_run_template,
    publish_new_valid_step,
    tiny_pretraining_run,
    tiny_session,
)


def test_evaluation_result_pins_resolved_identity(
    tiny_pretraining_run: Path,
    fake_lm_eval,
    tmp_path: Path,
) -> None:
    """Breaks if the current evaluation workflow keeps writing the legacy v1 schema."""
    config = tiny_evaluation_config(
        tiny_pretraining_run, tmp_path, tasks=("hellaswag",)
    )
    result = evaluate(config)
    pinned = result.model
    publish_new_valid_step(tiny_pretraining_run, step=pinned.step + 1)
    persisted = read_evaluation_result(config.output)
    assert result.version == persisted.version == 2
    assert persisted.model == pinned
    assert persisted.tasks[0].task_name == "hellaswag"
    assert persisted.tasks[0].metric_payload["acc,none"] == 0.5
    assert persisted.provider_result == result.provider_result
    assert persisted.identity == evaluation_result_identity(persisted)


def test_full_evaluation_persists_recovered_model_operational_state(
    tiny_pretraining_run: Path,
    fake_lm_eval,
    tmp_path: Path,
) -> None:
    """Evaluation persistence retains recovered-index and retention status."""
    (tiny_pretraining_run / "latest.json").write_bytes(b"not-json")
    config = replace(
        tiny_evaluation_config(tiny_pretraining_run, tmp_path),
        full_verify=True,
    )

    result = evaluate(config)
    persisted = read_evaluation_result(config.output)

    assert result.model.latest_recovered is True
    assert result.model.pruning_pending is False
    assert persisted.model.latest_recovered is True
    assert persisted.model.pruning_pending is False


def test_evaluate_is_idempotent_for_identical_output(
    tiny_pretraining_run: Path,
    fake_lm_eval,
    tmp_path: Path,
) -> None:
    config = tiny_evaluation_config(tiny_pretraining_run, tmp_path)
    first = evaluate(config)
    second = evaluate(config)
    assert first == second
    assert read_evaluation_result(config.output) == first


def test_evaluate_rejects_conflicting_existing_output(
    tiny_pretraining_run: Path,
    fake_lm_eval,
    tmp_path: Path,
) -> None:
    config = tiny_evaluation_config(tiny_pretraining_run, tmp_path)
    evaluate(config)
    config.output.write_text('{"different": true}\n', encoding="utf-8")
    with pytest.raises(SMLRuntimeError, match="collision|exists"):
        evaluate(config)


def test_evaluate_rejects_repeated_tasks_before_provider_execution(
    tiny_pretraining_run: Path,
    fake_lm_eval,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="duplicate"):
        tiny_evaluation_config(
            tiny_pretraining_run,
            tmp_path,
            tasks=("hellaswag", "winogrande", "hellaswag"),
        )
    assert not fake_lm_eval.calls


@pytest.mark.parametrize("padding", ["left", "right"])
def test_score_loglikelihood_batch_returns_finite_boolean_scores_for_both_padding_layouts(
    tiny_session: InferenceSession,
    padding: str,
) -> None:
    requests = (
        LoglikelihoodRequest(context="alpha", continuation=" beta"),
        LoglikelihoodRequest(context="alpha beta", continuation=" gamma"),
        LoglikelihoodRequest(context="alpha beta gamma", continuation=" delta"),
    )
    results = score_loglikelihood_batch(tiny_session, requests, padding=padding)
    assert len(results) == len(requests)
    for result in results:
        assert math.isfinite(result.log_likelihood)
        assert isinstance(result.greedy_match, bool)
