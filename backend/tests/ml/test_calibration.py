"""Tests for the calibration fit, thresholds and sampler (`app/ml/calibration.py`).

Synthetic data with a known true curve, so the fit can be checked
against the truth rather than against itself.
"""

import numpy as np
import pytest

from app.ml.calibration import (
    draw_sample,
    fit_calibration,
    has_enough_answers,
    rule_of_three_bound,
    score_to_logit,
)

TRUE_INTERCEPT = -1.0
TRUE_SLOPE = 1.5


def _synthetic(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Scores spread evenly over [0.05, 0.999], answers drawn from the true curve."""
    rng = np.random.default_rng(seed)
    scores = np.round(rng.uniform(0.05, 0.999, n), 3)
    p = 1 / (1 + np.exp(-(TRUE_INTERCEPT + TRUE_SLOPE * score_to_logit(scores))))
    right = (rng.uniform(size=n) < p).astype(int)
    return scores, right


def test_fit_recovers_the_true_curve() -> None:
    scores, right = _synthetic(4000, seed=1)
    fit = fit_calibration(scores, right)
    assert fit.intercept == pytest.approx(TRUE_INTERCEPT, abs=0.15)
    assert fit.slope == pytest.approx(TRUE_SLOPE, abs=0.15)


def test_band_covers_the_truth_and_stays_in_0_1() -> None:
    scores, right = _synthetic(200, seed=2)
    fit = fit_calibration(scores, right)
    grid = np.array([0.0, 0.2, 0.5, 0.8, 0.999, 1.0])
    p, lower, upper = fit.predict(grid)
    truth = 1 / (1 + np.exp(-(TRUE_INTERCEPT + TRUE_SLOPE * score_to_logit(grid))))
    assert np.all((lower >= 0) & (upper <= 1))
    assert np.all((lower <= p) & (p <= upper))
    assert np.all((lower <= truth) & (truth <= upper))


def test_band_narrows_with_more_answers() -> None:
    small = fit_calibration(*_synthetic(40, seed=3))
    large = fit_calibration(*_synthetic(2000, seed=3))
    _, lo_s, hi_s = small.predict(np.array([0.7]))
    _, lo_l, hi_l = large.predict(np.array([0.7]))
    assert (hi_l - lo_l)[0] < (hi_s - lo_s)[0]


def test_perfect_separation_stays_finite() -> None:
    """Every wrong answer below every right one: an unpenalised fit diverges."""
    scores = np.array([0.1, 0.2, 0.3, 0.35, 0.4, 0.42, 0.45, 0.47, 0.48, 0.49,
                       0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.999, 1.0])
    right = np.array([0] * 10 + [1] * 10)
    fit = fit_calibration(scores, right)
    assert np.isfinite(fit.intercept) and np.isfinite(fit.slope)
    assert fit.slope > 0
    assert np.all(np.isfinite(fit.covariance))
    threshold = fit.threshold(0.95)
    assert threshold is not None
    assert 0.49 < threshold < 1.0


def test_tightly_clustered_scores_are_not_flattened_by_the_penalty() -> None:
    """All wrong at 0.998, all right at 0.999: the curve must still reach 95% at 0.999."""
    scores = np.array([0.998] * 10 + [0.999] * 10)
    right = np.array([0] * 10 + [1] * 10)
    fit = fit_calibration(scores, right)
    p, _, _ = fit.predict(np.array([0.999]))
    assert p[0] > 0.95
    assert fit.threshold(0.95) is not None


def test_threshold_lands_where_the_curve_reaches_the_target() -> None:
    scores, right = _synthetic(4000, seed=4)
    fit = fit_calibration(scores, right)
    threshold = fit.threshold(0.9)
    assert threshold is not None
    p, _, _ = fit.predict(np.array([threshold]))
    assert p[0] == pytest.approx(0.9, abs=0.01)
    # Labels above the threshold are right more often than the target.
    p_above, _, _ = fit.predict(np.array([threshold + 0.01]))
    assert p_above[0] > 0.9


def test_threshold_beyond_the_answered_range_is_not_reached() -> None:
    """Never extrapolate: a target the curve only reaches past the top score is None."""
    rng = np.random.default_rng(5)
    scores = np.round(rng.uniform(0.3, 0.8, 300), 3)
    right = (rng.uniform(size=300) < 0.6).astype(int)
    fit = fit_calibration(scores, right)
    assert fit.threshold(0.99) is None


def test_threshold_already_met_returns_the_lowest_answered_score() -> None:
    rng = np.random.default_rng(6)
    scores = np.round(rng.uniform(0.6, 0.99, 400), 3)
    p = 1 / (1 + np.exp(-(4.0 + 1.0 * score_to_logit(scores))))
    right = (rng.uniform(size=400) < p).astype(int)
    right[:2] = 0  # keep the fit defined
    fit = fit_calibration(scores, right)
    assert fit.threshold(0.9) == pytest.approx(scores.min(), abs=1e-3)


def test_a_falling_curve_gives_no_threshold() -> None:
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.95])
    right = np.array([1, 1, 1, 0, 0, 1])
    fit = fit_calibration(scores, right)
    assert fit.slope < 0
    assert fit.threshold(0.9) is None


def test_fit_refuses_one_sided_answers() -> None:
    with pytest.raises(ValueError):
        fit_calibration(np.array([0.5, 0.9]), np.array([1, 1]))


def test_minimum_needs_ten_of_each() -> None:
    assert has_enough_answers(10, 10)
    assert not has_enough_answers(50, 9)
    assert not has_enough_answers(9, 50)


def test_rule_of_three() -> None:
    assert rule_of_three_bound(60) == pytest.approx(0.95)
    assert rule_of_three_bound(3) is None


def test_sample_spreads_over_the_score_range() -> None:
    """A pool piled up near 0.999 still yields draws from every bin."""
    rng = np.random.default_rng(7)
    high = [(f"h{i}", 0.999) for i in range(500)]
    low = [(f"l{i}", s) for i, s in enumerate([0.1, 0.3, 0.5, 0.7])]
    picked = draw_sample(high + low, batch_size=20, n_bins=5, rng=rng)
    assert len(picked) == 20
    assert {f"l{i}" for i in range(4)} <= set(picked)


def test_sample_moves_an_empty_bins_share_to_the_others() -> None:
    rng = np.random.default_rng(8)
    pool = [(f"a{i}", 0.2) for i in range(2)] + [(f"b{i}", 0.9) for i in range(50)]
    picked = draw_sample(pool, batch_size=20, n_bins=5, rng=rng)
    assert len(picked) == 20
    assert len(set(picked)) == 20


def test_sample_returns_what_is_left_when_the_pool_is_small() -> None:
    rng = np.random.default_rng(9)
    pool = [("a", 0.5), ("b", 0.5), ("c", 0.9)]
    assert sorted(draw_sample(pool, batch_size=20, n_bins=5, rng=rng)) == ["a", "b", "c"]
    assert draw_sample([], batch_size=20, n_bins=5, rng=rng) == []
