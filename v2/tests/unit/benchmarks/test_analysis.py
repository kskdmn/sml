import pytest

from v2.benchmarks.analysis import analyze_pairs


@pytest.mark.parametrize(
    "direction,reference,candidate,expected_ratio",
    [
        ("higher-is-better", 100.0, 103.0, 1.03),
        ("lower-is-better", 10.0, 8.0, 1.25),
    ],
)
def test_paired_ratios_and_bounds_follow_metric_direction(
    direction, reference, candidate, expected_ratio
):
    report = analyze_pairs(
        reference=[reference] * 5,
        candidate=[candidate] * 5,
        direction=direction,
        bootstrap_seed=1729,
        resamples=10_000,
        minimum_ratio=1.03,
        maximum_dispersion=0.02,
        require_lower_bound=True,
    )
    assert report.paired_ratios == (expected_ratio,) * 5
    assert report.median_ratio == expected_ratio
    assert report.lower_confidence_bound == expected_ratio
    assert report.decision == "pass"


def test_noise_and_confidence_fail_closed():
    noisy = analyze_pairs(
        reference=[100.0, 100.0, 100.0, 100.0, 100.0],
        candidate=[95.0, 100.0, 103.0, 106.0, 110.0],
        direction="higher-is-better",
        bootstrap_seed=1729,
        resamples=10_000,
        minimum_ratio=0.97,
        maximum_dispersion=0.02,
        require_lower_bound=True,
    )
    assert noisy.decision == "too-noisy"


def test_bootstrap_is_reproducible_and_point_only_pass_is_inconclusive():
    arguments = {
        "reference": [100.0, 100.0, 100.0, 100.0, 100.0],
        "candidate": [96.0, 98.0, 100.0, 103.0, 108.0],
        "direction": "higher-is-better",
        "bootstrap_seed": 1729,
        "resamples": 10_000,
        "minimum_ratio": 0.99,
        "maximum_dispersion": 0.10,
        "require_lower_bound": True,
    }
    first = analyze_pairs(**arguments)
    second = analyze_pairs(**arguments)
    assert first.lower_confidence_bound == second.lower_confidence_bound
    assert first.median_ratio >= 0.99
    assert first.lower_confidence_bound < 0.99
    assert first.decision == "inconclusive"


@pytest.mark.parametrize(
    ("reference", "candidate"),
    [([], []), ([1.0], []), ([0.0], [1.0]), ([float("nan")], [1.0])],
)
def test_analysis_rejects_invalid_pairs(reference, candidate):
    with pytest.raises(ValueError):
        analyze_pairs(
            reference,
            candidate,
            direction="higher-is-better",
            bootstrap_seed=1729,
            resamples=10_000,
            minimum_ratio=0.97,
            maximum_dispersion=0.02,
            require_lower_bound=False,
        )
