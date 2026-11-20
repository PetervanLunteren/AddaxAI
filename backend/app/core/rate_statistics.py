"""Mean, spread and confidence interval of a set of rates.

The Trend page takes the site as the sampling unit: every site gets its
own observation rate, and what is reported is the mean of those rates
with a 95% confidence interval from Student's t,

    SE = SD / sqrt(n)        95% CI = mean ± t(n - 1) × SE

the usual way a camera trap rate is reported with its uncertainty (it is
also what New Zealand's DOC prescribes for its camera trap catch index).

Pure arithmetic: no database and no app imports, so it is tested on its
own.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Two-sided 95% critical values of Student's t (the 0.975 quantile) by
# degrees of freedom. A table and not scipy: the bundled backend ships
# without scipy (backend.spec), and these 33 numbers never change.
_T_975: dict[int, float] = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
    40: 2.021, 60: 2.000, 120: 1.980,
}
_T_975_NORMAL = 1.960


def t_critical_95(degrees_of_freedom: int) -> float:
    """t for a two-sided 95% interval.

    Between two table rows it takes the lower row, whose t is larger, so
    an interval is never narrower than the exact one. Above 120 degrees
    of freedom t is the normal value to three decimals.
    """
    if degrees_of_freedom < 1:
        raise ValueError("t needs at least one degree of freedom")
    if degrees_of_freedom > 120:
        return _T_975_NORMAL
    return _T_975[max(df for df in _T_975 if df <= degrees_of_freedom)]


@dataclass(frozen=True)
class RateStats:
    """What a set of rates says together. `n` is how many went in.

    `mean` is None without any rate. `sd` and the interval are None with
    fewer than two: one site has no spread to measure.
    """

    n: int
    mean: float | None
    sd: float | None
    ci_low: float | None
    ci_high: float | None


def summarise_rates(rates: list[float]) -> RateStats:
    n = len(rates)
    if n == 0:
        return RateStats(0, None, None, None, None)
    mean = math.fsum(rates) / n
    if n == 1:
        return RateStats(1, mean, None, None, None)
    sd = math.sqrt(math.fsum((r - mean) ** 2 for r in rates) / (n - 1))
    margin = t_critical_95(n - 1) * sd / math.sqrt(n)
    # A rate cannot be negative. A t-interval on a few skewed site rates
    # can reach below zero, so the lower bound stops there.
    return RateStats(n, mean, sd, max(0.0, mean - margin), mean + margin)
