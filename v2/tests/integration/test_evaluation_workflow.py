# ruff: noqa: F811
from __future__ import annotations

import sys
from pathlib import Path

_UNIT_DIR = Path(__file__).resolve().parents[1] / "unit"
if str(_UNIT_DIR) not in sys.path:
    sys.path.insert(0, str(_UNIT_DIR))

import pytest
from sml.errors import SMLRuntimeError
from sml.evaluation import evaluate, read_evaluation_result
from test_evaluation import fake_lm_eval, tiny_evaluation_config  # noqa: F401
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
    result = evaluate(
        tiny_evaluation_config(tiny_pretraining_run, tmp_path, tasks=("hellaswag",))
    )
    pinned = result.model
    publish_new_valid_step(tiny_pretraining_run, step=pinned.step + 1)
    persisted = read_evaluation_result(result.output)
    assert persisted.model == pinned
    assert persisted.tasks == ("hellaswag",)
    assert persisted.output == result.output
    assert persisted.provider_versions == result.provider_versions
    assert persisted.provider_versions == tuple(
        sorted(persisted.provider_versions, key=lambda item: item[0])
    )


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


def test_evaluate_supports_repeated_tasks(
    tiny_pretraining_run: Path,
    fake_lm_eval,
    tmp_path: Path,
) -> None:
    result = evaluate(
        tiny_evaluation_config(
            tiny_pretraining_run,
            tmp_path,
            tasks=("hellaswag", "winogrande", "hellaswag"),
        )
    )
    assert result.tasks == ("hellaswag", "winogrande", "hellaswag")
    assert fake_lm_eval.calls
    assert fake_lm_eval.calls[0]["tasks"] == [
        "hellaswag",
        "winogrande",
        "hellaswag",
    ]
