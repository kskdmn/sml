from __future__ import annotations

import pytest

from v2.benchmarks.quality import (
    PretrainingQualityWorkload,
    build_pretraining_quality_workload,
)

from . import ROOT


@pytest.fixture(scope="package")
def canonical_workload() -> PretrainingQualityWorkload:
    return build_pretraining_quality_workload(ROOT)
