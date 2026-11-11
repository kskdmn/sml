from __future__ import annotations

import tracemalloc
from types import SimpleNamespace

import pytest
from sml.errors import SMLRuntimeError


def test_request_recorder_hashes_actual_order_per_task() -> None:
    from sml import evaluation

    first = SimpleNamespace(
        task_name="hellaswag", doc_id=2, repeats=1, args=("a", " b")
    )
    second = SimpleNamespace(
        task_name="hellaswag", doc_id=1, repeats=1, args=("c", " d")
    )
    recorder = evaluation._EvaluationRequestRecorder()
    recorder.record("loglikelihood", (first, second))
    reversed_recorder = evaluation._EvaluationRequestRecorder()
    reversed_recorder.record("loglikelihood", (second, first))
    assert recorder.identity_for("hellaswag") != reversed_recorder.identity_for(
        "hellaswag"
    )


def test_request_recorder_requires_unambiguous_task_ownership() -> None:
    from sml import evaluation

    recorder = evaluation._EvaluationRequestRecorder()
    with pytest.raises(SMLRuntimeError, match="task_name|ownership"):
        recorder.record(
            "loglikelihood",
            (SimpleNamespace(doc_id=1, repeats=1, args=("a", " b")),),
        )


def test_request_recorder_preserves_canonical_identity_across_batches() -> None:
    from sml import evaluation
    from sml.artifacts.manifest import structured_identity

    records = (
        {
            "request_type": "loglikelihood",
            "task_name": "hellaswag",
            "doc_id": 2,
            "repeats": 1,
            "args": ("Context: 日本語\n", " continuation"),
        },
        {
            "request_type": "generate_until",
            "task_name": "hellaswag",
            "doc_id": 1,
            "repeats": 2,
            "args": ("next", {"until": ["\n", '"'], "temperature": -0.0}),
        },
    )
    recorder = evaluation._EvaluationRequestRecorder()
    for index, record in enumerate(records):
        recorder.record(record["request_type"], (SimpleNamespace(**record),))
        expected = structured_identity(
            "sml-evaluation-requests-v1", records[: index + 1]
        )
        assert recorder.identity_for("hellaswag") == expected
        assert recorder.identity_for("hellaswag") == expected
        recorder.record(
            "loglikelihood",
            (
                SimpleNamespace(
                    task_name="winogrande", doc_id=index, repeats=1, args=("a", " b")
                ),
            ),
        )
        assert recorder.identity_for("hellaswag") == expected
    assert recorder.task_names == frozenset({"hellaswag", "winogrande"})


def test_request_recorder_memory_does_not_grow_with_prompt_corpus() -> None:
    from sml import evaluation

    recorder = evaluation._EvaluationRequestRecorder()
    tracemalloc.start()
    try:
        for index in range(1000):
            recorder.record(
                "loglikelihood",
                (
                    SimpleNamespace(
                        task_name="hellaswag",
                        doc_id=index,
                        repeats=1,
                        args=(f"{index}:" + "x" * 8192, " continuation"),
                    ),
                ),
            )
        assert recorder.identity_for("hellaswag").startswith("sha256:")
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 1024 * 1024
