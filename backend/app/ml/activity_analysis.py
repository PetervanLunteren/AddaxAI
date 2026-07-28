"""
Activity-pattern analysis: circular KDE, overlap coefficient, bootstrap CI,
diel classification.

Backs the Plots → Activity overlap page. Implements the conventions used
by the R `overlap` and `activity` packages so results are comparable to
the published camera-trap literature:

- Per-species smoothing: **von Mises kernel density** evaluated on a 240-
  point grid over [0, 24) hours. Reference: Ridout & Linkie 2009 (J Agric
  Biol Environ Stat 14:322-337); `overlap` R package vignette by
  Meredith & Ridout 2014.
- Pairwise overlap: **Δ coefficient** = ∫ min(f_a, f_b) dt over [0, 24].
  Same definition Ridout & Linkie use. The "Δ4 vs Δ1" choice in the
  literature is about which density estimator is plugged in; we use a
  single von Mises KDE for both species and report the conventional label
  based on the smaller sample size (Δ4 if min-N ≥ 50, Δ1 otherwise).
- Bootstrap CI for Δ: 1000 percentile-bootstrap reps. Canonical is
  10 000; 1000 is sufficient for an interactive UI and runs in well under
  a second for typical n.
- Diel classification: Bennie et al. 2014 ≥ 0.70 density-in-phase rule
  (diurnal / nocturnal / crepuscular / cathemeral). Phase boundaries come
  from the project's sun bands (dawn/sunrise/sunset/dusk); when sun bands
  aren't available (polar latitudes, missing tz), falls back to a fixed
  06:00-18:00 day window and surfaces that fact via the diel classification
  itself never being marked crepuscular in the fallback path.

The math is pure numpy/scipy — no DB, no FastAPI. Tested in
`backend/tests/api/test_activity_overlap.py`.
"""

from typing import Literal

import numpy as np

from app.api.schemas.statistics import SunBands

DielClass = Literal["diurnal", "nocturnal", "crepuscular", "cathemeral"]
DeltaEstimator = Literal["delta1", "delta4"]

# Number of grid points used to evaluate the KDE on [0, 24).
KDE_GRID_SAMPLES: int = 240
# Default von Mises concentration parameter. ~5 corresponds to a roughly
# 1.7-hour-equivalent kernel bandwidth, a sensible compromise for typical
# camera-trap n in the dozens to low hundreds. Larger kappa = sharper
# peaks, smaller kappa = smoother. Tunable per call.
DEFAULT_KAPPA: float = 5.0
# Bootstrap reps for the Δ confidence interval. Canonical is 10 000; we
# use 1000 to keep the interactive endpoint snappy.
BOOTSTRAP_REPS: int = 1000
# Bennie et al. 2014 density-in-phase threshold for diel classification.
DIEL_THRESHOLD: float = 0.70


def fit_circular_kde(
    times_hours: np.ndarray,
    *,
    samples: int = KDE_GRID_SAMPLES,
    kappa: float = DEFAULT_KAPPA,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit a circular von Mises KDE to detection times in [0, 24).

    Returns (grid_hours, density) where:
    - grid_hours is `samples` evenly-spaced points in [0, 24)
    - density is the KDE evaluated at each grid point, in units of
      probability density per hour. Integrating density * (24/samples)
      over the full grid yields ~1.0.

    Algorithm: convert hours to radians, evaluate the unnormalized von
    Mises kernel sum at each grid angle, then post-normalize so the
    density integrates to exactly 1.0 over [0, 24). The closed-form
    normalization for von Mises is 1/(2π·I₀(κ)) but we skip computing
    the modified Bessel function I₀ — post-normalization via the grid
    sum is mathematically equivalent and avoids a scipy dependency.

    Edge case: empty input returns a flat grid of zeros (caller is
    responsible for warning the user about insufficient data).
    """
    grid_hours = np.linspace(0.0, 24.0, samples, endpoint=False)
    if len(times_hours) == 0:
        return grid_hours, np.zeros(samples)

    # Convert to radians on [0, 2π)
    angles = np.asarray(times_hours, dtype=np.float64) * (2.0 * np.pi / 24.0)
    grid_angles = grid_hours * (2.0 * np.pi / 24.0)

    # Unnormalized von Mises kernel sum:
    #   raw(θ) = mean_i exp(κ · cos(θ - θ_i))
    # cos_diff is shape (n_data, n_grid).
    cos_diff = np.cos(grid_angles[None, :] - angles[:, None])
    raw = np.exp(kappa * cos_diff).mean(axis=0)

    # Post-normalize so the density integrates to 1.0 over [0, 24)
    # using a left-Riemann sum with step dx = 24/samples.
    dx = 24.0 / samples
    integral = raw.sum() * dx
    if integral == 0.0:
        return grid_hours, np.zeros(samples)
    density_per_hour = raw / integral
    return grid_hours, density_per_hour


def overlap_coefficient(
    density_a: np.ndarray,
    density_b: np.ndarray,
    *,
    samples: int = KDE_GRID_SAMPLES,
) -> float:
    """
    Compute the overlap coefficient Δ = ∫ min(f_a, f_b) dt over [0, 24].

    Both inputs must be evaluated on the same uniform grid (use the
    output of `fit_circular_kde` with matching `samples`). When both
    densities each integrate to 1.0, Δ ∈ [0, 1] where 1.0 = identical
    distributions and 0.0 = disjoint distributions.
    """
    if density_a.shape != density_b.shape:
        raise ValueError(
            f"density shapes must match, got {density_a.shape} vs {density_b.shape}"
        )
    dx = 24.0 / samples
    return float(np.minimum(density_a, density_b).sum() * dx)


def bootstrap_overlap_ci(
    times_a: np.ndarray,
    times_b: np.ndarray,
    *,
    reps: int = BOOTSTRAP_REPS,
    samples: int = KDE_GRID_SAMPLES,
    kappa: float = DEFAULT_KAPPA,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Compute Δ with a percentile bootstrap 95% CI.

    Resamples each species independently with replacement, refits the
    KDE on each resample, recomputes Δ, and returns the 2.5/97.5
    percentiles of the bootstrap distribution alongside the point
    estimate from the original data. Fixed seed = deterministic results
    across requests (good for testing and avoids confusing UI flicker
    on identical inputs).

    Returns (delta_point, ci_low, ci_high).
    """
    times_a = np.asarray(times_a, dtype=np.float64)
    times_b = np.asarray(times_b, dtype=np.float64)
    n_a = len(times_a)
    n_b = len(times_b)
    if n_a == 0 or n_b == 0:
        return 0.0, 0.0, 0.0

    _, density_a = fit_circular_kde(times_a, samples=samples, kappa=kappa)
    _, density_b = fit_circular_kde(times_b, samples=samples, kappa=kappa)
    delta_point = overlap_coefficient(density_a, density_b, samples=samples)

    rng = np.random.default_rng(seed)
    deltas = np.empty(reps, dtype=np.float64)
    for i in range(reps):
        sample_a = rng.choice(times_a, size=n_a, replace=True)
        sample_b = rng.choice(times_b, size=n_b, replace=True)
        _, da = fit_circular_kde(sample_a, samples=samples, kappa=kappa)
        _, db = fit_circular_kde(sample_b, samples=samples, kappa=kappa)
        deltas[i] = overlap_coefficient(da, db, samples=samples)

    ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])
    return delta_point, float(ci_low), float(ci_high)


def estimator_label(min_n: int) -> DeltaEstimator:
    """
    Return the conventional Δ-estimator label for a given smaller sample
    size, following Ridout & Linkie 2009 / `overlap` R package guidance.

    The Δ4 estimator (continuous KDE) is recommended when the smaller
    sample is ≥ 50 because it has lower bias for moderate-to-large n.
    Δ1 (binned histogram) is recommended below 50 where Δ4's bandwidth
    becomes unstable. We use a single von Mises KDE method internally
    regardless; the label here tells the user which conventional name
    to cite when reporting the value.
    """
    return "delta1" if min_n < 50 else "delta4"


def classify_diel(
    grid_hours: np.ndarray,
    density: np.ndarray,
    sun_bands: SunBands | None,
    *,
    threshold: float = DIEL_THRESHOLD,
) -> tuple[DielClass, dict[str, float]]:
    """
    Classify a species as diurnal / nocturnal / crepuscular / cathemeral.

    Bennie et al. 2014 rule: a species is classified as the dominant
    phase if ≥ `threshold` of its activity density falls in that phase;
    otherwise cathemeral. Phases come from the project's sun bands:
    - day: [sunrise, sunset)
    - twilight: [dawn, sunrise) ∪ [sunset, dusk)
    - night: everything else

    When sun_bands is None (polar latitudes, unknown timezone), falls
    back to a fixed 06:00 / 18:00 day window with no twilight phase, and
    can therefore only return diurnal / nocturnal / cathemeral (never
    crepuscular).

    Returns (label, density_by_phase) where density_by_phase is a dict
    {"day": float, "night": float, "twilight": float} that sums to ~1.0
    and is suitable to display in a UI tooltip alongside the threshold
    rule.
    """
    if len(grid_hours) < 2:
        return "cathemeral", {"day": 0.0, "night": 0.0, "twilight": 0.0}

    dx = float(grid_hours[1] - grid_hours[0])

    if sun_bands is None:
        day_mask = (grid_hours >= 6.0) & (grid_hours < 18.0)
        day = float(density[day_mask].sum() * dx)
        night = max(0.0, 1.0 - day)
        density_by_phase = {"day": day, "night": night, "twilight": 0.0}
    else:
        day_mask = (grid_hours >= sun_bands.sunrise) & (grid_hours < sun_bands.sunset)
        twilight_mask = (
            ((grid_hours >= sun_bands.dawn) & (grid_hours < sun_bands.sunrise))
            | ((grid_hours >= sun_bands.sunset) & (grid_hours < sun_bands.dusk))
        )
        day = float(density[day_mask].sum() * dx)
        twilight = float(density[twilight_mask].sum() * dx)
        night = max(0.0, 1.0 - day - twilight)
        density_by_phase = {"day": day, "night": night, "twilight": twilight}

    max_phase = max(density_by_phase, key=density_by_phase.__getitem__)
    if density_by_phase[max_phase] >= threshold:
        if max_phase == "day":
            return "diurnal", density_by_phase
        if max_phase == "night":
            return "nocturnal", density_by_phase
        return "crepuscular", density_by_phase
    return "cathemeral", density_by_phase
