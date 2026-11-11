from __future__ import annotations

import pytest
from sml.evaluation import (
    EvaluationConfig,
)


@pytest.mark.parametrize(
    "overrides,message",
    [
        pytest.param({"tasks": ()}, "task", id="empty-tasks"),
        pytest.param({"limit": 0}, "limit", id="nonpositive-limit"),
        pytest.param(
            {"tasks": ("hellaswag", "winogrande", "hellaswag")},
            "duplicate",
            id="duplicate-task",
        ),
        pytest.param(
            {"tasks": ("mmlu",)}, "hellaswag|winogrande", id="unsupported-task"
        ),
    ],
)
def test_evaluation_config_rejects_invalid_controls(tmp_path, overrides, message):
    values = {
        "checkpoint": tmp_path,
        "tasks": ("hellaswag",),
        "output": tmp_path / "out.json",
    }
    with pytest.raises(ValueError, match=message):
        EvaluationConfig(**(values | overrides))
