"""Mean, SD and 95% CI of a set of site rates.

Hand-checked against the formulas in the module docstring. The lower
bound stops at zero: with a few skewed sites the t-interval reaches
below it, and a negative rate does not exist.
"""

import pytest

from app.core.rate_statistics import summarise_rates, t_critical_95


def test_four_sites_give_the_hand_calculated_interval():
    stats = summarise_rates([10.0, 12.0, 14.0, 16.0])

    assert stats.n == 4
    assert stats.mean == 13.0
    # sqrt((9 + 1 + 1 + 9) / 3)
    assert stats.sd == pytest.approx(2.581989, abs=1e-6)
    # 3.182 * 2.581989 / sqrt(4) = 4.107944
    assert stats.ci_low == pytest.approx(8.892056, abs=1e-6)
    assert stats.ci_high == pytest.approx(17.107944, abs=1e-6)


def test_the_lower_bound_stops_at_zero():
    stats = summarise_rates([2.0, 4.0, 6.0])

    assert (stats.mean, stats.sd) == (4.0, 2.0)
    # 4.303 * 2 / sqrt(3) = 4.968676 is more than the mean.
    assert stats.ci_low == 0.0
    assert stats.ci_high == pytest.approx(8.968676, abs=1e-6)


def test_sites_that_agree_have_no_spread():
    stats = summarise_rates([5.0, 5.0, 5.0])

    assert (stats.mean, stats.sd, stats.ci_low, stats.ci_high) == (5.0, 0.0, 5.0, 5.0)


def test_one_site_has_a_mean_and_no_interval():
    stats = summarise_rates([7.5])

    assert (stats.n, stats.mean) == (1, 7.5)
    assert (stats.sd, stats.ci_low, stats.ci_high) == (None, None, None)


def test_no_site_has_nothing():
    stats = summarise_rates([])

    assert stats.n == 0
    assert (stats.mean, stats.sd, stats.ci_low, stats.ci_high) == (None,) * 4


def test_two_sites_use_the_widest_t():
    stats = summarise_rates([0.0, 10.0])

    # sd = 7.071068, t(1) = 12.706, margin = 12.706 * 7.071068 / sqrt(2)
    assert stats.ci_high == pytest.approx(5.0 + 63.53, abs=1e-6)
    assert stats.ci_low == 0.0


@pytest.mark.parametrize(
    ("degrees_of_freedom", "expected"),
    [
        (1, 12.706),
        (2, 4.303),
        (30, 2.042),
        # Between rows: the lower row, never a narrower interval.
        (31, 2.042),
        (45, 2.021),
        (119, 2.000),
        (120, 1.980),
        (121, 1.960),
        (1000, 1.960),
    ],
)
def test_t_by_degrees_of_freedom(degrees_of_freedom, expected):
    assert t_critical_95(degrees_of_freedom) == expected


def test_t_never_grows_with_more_sites():
    values = [t_critical_95(df) for df in range(1, 200)]

    assert values == sorted(values, reverse=True)


def test_t_needs_a_degree_of_freedom():
    with pytest.raises(ValueError):
        t_critical_95(0)
