from __future__ import annotations

import pytest
from sml.evaluation import score_loglikelihood_batch
from sml.inference import InferenceSession
from sml_test_support.evaluation import (
    assert_loglikelihood_results_close,
    heterogeneous_scoring_requests,
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
