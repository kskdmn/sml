from __future__ import annotations

import pytest

from v2.benchmarks.swag_quality import (
    SwagQualityWorkload,
    build_swag_quality_workload,
)

from . import ROOT


@pytest.fixture(scope="package")
def canonical_workload() -> SwagQualityWorkload:
    return build_swag_quality_workload(ROOT)
