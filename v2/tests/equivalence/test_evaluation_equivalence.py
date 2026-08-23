# ruff: noqa: F811
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_UNIT_DIR = Path(__file__).resolve().parents[1] / "unit"
if str(_UNIT_DIR) not in sys.path:
    sys.path.insert(0, str(_UNIT_DIR))

from sml.evaluation import score_loglikelihood_batch
from sml.inference import InferenceSession
from test_evaluation import (
    assert_loglikelihood_results_close,
    heterogeneous_scoring_requests,
)
from test_inference import (  # noqa: F401
    _tiny_run_template,
    tiny_pretraining_run,
    tiny_session,
)


@pytest.mark.parametrize("padding", ["left", "right"])
def test_padded_loglikelihood_matches_serial(
    tiny_session: InferenceSession,
    padding: str,
) -> None:
    requests = heterogeneous_scoring_requests()
    serial = tuple(
        score_loglikelihood_batch(tiny_session, [request], padding=padding)[0]
        for request in requests
    )
    batched = score_loglikelihood_batch(tiny_session, requests, padding=padding)
    assert_loglikelihood_results_close(serial, batched, atol=2e-2, rtol=2e-2)
