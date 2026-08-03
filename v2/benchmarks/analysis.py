from __future__ import annotations

from dataclasses import dataclass, replace
from statistics import median
from typing import Literal, Sequence

import numpy as np

Direction = Literal["higher-is-better", "lower-is-better"]
Decision = Literal["pass", "fail", "inconclusive", "too-noisy"]


@dataclass(frozen=True, slots=True)
class MetricReport:
    reference_median: float
    reference_mad: float
    candidate_median: float
    candidate_mad: float
    paired_ratios: tuple[float, ...]
    median_ratio: float
    ratio_mad: float
    lower_confidence_bound: float
    decision: Decision


def _median_absolute_deviation(values: Sequence[float], center: float) -> float:
    return float(median(abs(value - center) for value in values))


def decide(
    report: MetricReport,
    minimum_ratio: float,
    maximum_dispersion: float,
    require_lower_bound: bool,
) -> Decision:
    dispersions = (
        report.reference_mad / report.reference_median,
        report.candidate_mad / report.candidate_median,
        report.ratio_mad / report.median_ratio,
    )
    if any(value > maximum_dispersion for value in dispersions):
        return "too-noisy"
    if report.median_ratio < minimum_ratio:
        return "fail"
    if require_lower_bound and report.lower_confidence_bound < minimum_ratio:
        return "inconclusive"
    return "pass"


def analyze_pairs(
    reference: Sequence[float],
    candidate: Sequence[float],
    *,
    direction: Direction,
    bootstrap_seed: int,
    resamples: int,
    minimum_ratio: float,
    maximum_dispersion: float,
    require_lower_bound: bool,
) -> MetricReport:
    reference_values = tuple(float(value) for value in reference)
    candidate_values = tuple(float(value) for value in candidate)
    if not reference_values or len(reference_values) != len(candidate_values):
        raise ValueError("reference and candidate must contain equal non-zero pairs")
    if direction not in ("higher-is-better", "lower-is-better"):
        raise ValueError(f"unsupported metric direction: {direction!r}")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    if minimum_ratio <= 0:
        raise ValueError("minimum_ratio must be positive")
    if maximum_dispersion < 0:
        raise ValueError("maximum_dispersion must be non-negative")
    all_values = (*reference_values, *candidate_values)
    if any(not np.isfinite(value) or value <= 0 for value in all_values):
        raise ValueError("paired measurements must be finite and positive")

    if direction == "higher-is-better":
        ratios = tuple(
            candidate_value / reference_value
            for reference_value, candidate_value in zip(
                reference_values, candidate_values, strict=True
            )
        )
    else:
        ratios = tuple(
            reference_value / candidate_value
            for reference_value, candidate_value in zip(
                reference_values, candidate_values, strict=True
            )
        )

    reference_median = float(median(reference_values))
    candidate_median = float(median(candidate_values))
    median_ratio = float(median(ratios))

    random = np.random.default_rng(bootstrap_seed)
    sampled_indices = random.integers(
        0,
        len(ratios),
        size=(resamples, len(ratios)),
    )
    sampled_ratios = np.asarray(ratios, dtype=np.float64)[sampled_indices]
    bootstrap_medians = np.median(sampled_ratios, axis=1)
    lower_confidence_bound = float(np.percentile(bootstrap_medians, 5.0))

    report = MetricReport(
        reference_median=reference_median,
        reference_mad=_median_absolute_deviation(reference_values, reference_median),
        candidate_median=candidate_median,
        candidate_mad=_median_absolute_deviation(candidate_values, candidate_median),
        paired_ratios=ratios,
        median_ratio=median_ratio,
        ratio_mad=_median_absolute_deviation(ratios, median_ratio),
        lower_confidence_bound=lower_confidence_bound,
        decision="fail",
    )
    return replace(
        report,
        decision=decide(
            report,
            minimum_ratio=minimum_ratio,
            maximum_dispersion=maximum_dispersion,
            require_lower_bound=require_lower_bound,
        ),
    )
