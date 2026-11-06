"""
Confidence calibration: how often an AI label is right at a given score.

Backs the Insights → Calibration page. The user answers "Is this a
wolf?" on a sample the app draws across the label's score range, and
this module turns those answers into a curve: classifier score in,
chance that the label is right out. The thresholds read off that curve
("from 0.67 up, at least 95 of 100 wolf labels are right") are advice
only; nothing in counts or exports uses them.

Why a sample the app draws, rather than the verified detections: the
Labels page is used to find and fix mistakes, so the verified set leans
towards wrong labels and the curve would come out too pessimistic.
Choosing detections by score alone leaves P(right | score) unbiased,
whatever the mix of scores, which is what lets the sampler spread its
draws evenly over the score range.

The model is the one BirdNET-Analyzer uses in its review tab: a
two-parameter logistic regression on logit(score). Fitted by Newton
steps with a small ridge penalty on the slope only. Without it the fit
never settles when every wrong answer scores below every right one
(perfect separation), which is likely with 10 + 10 answers. The
intercept is left free so the penalty does not pull the curve towards
50%. The fit runs on standardised logits and is mapped back, so the
penalty is equally mild whether the answered scores span 0.01 to 1.0
or sit together between 0.998 and 0.999.

The band is the delta method on the fitted coefficients, computed on
the logit scale and transformed back, so it always stays in [0, 1].

Pure numpy, no DB, no FastAPI. Tested in
`backend/tests/ml/test_calibration.py`.
"""

from dataclasses import dataclass

import numpy as np

# Scores are stored rounded to three decimals, so 0.000 and 1.000 occur
# and would map to an infinite logit. Clip to half a rounding step.
SCORE_CLIP: float = 0.0005
# Ridge penalty on the slope. Small next to the information in 20+
# answers, large enough to keep a separated sample finite.
SLOPE_PENALTY: float = 0.1
# A curve shows only from this many right and this many wrong answers.
MIN_RIGHT: int = 10
MIN_WRONG: int = 10
# Fixed targets shown on the page.
TARGETS: tuple[float, ...] = (0.90, 0.95, 0.99)
# Two-sided 95% band.
Z_95: float = 1.959963984540054
# Newton iterations; the penalised problem converges in well under this.
_MAX_ITER: int = 100
_TOL: float = 1e-10


def score_to_logit(scores: np.ndarray) -> np.ndarray:
    """logit(score), clipped so a rounded 0.000 or 1.000 stays finite."""
    s = np.clip(np.asarray(scores, dtype=float), SCORE_CLIP, 1.0 - SCORE_CLIP)
    return np.log(s / (1.0 - s))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass(frozen=True)
class CalibrationFit:
    """P(right | score) = sigmoid(intercept + slope * logit(score))."""

    intercept: float
    slope: float
    # 2x2 covariance of (intercept, slope): inverse of the penalised
    # Hessian at the optimum.
    covariance: np.ndarray
    # Logit range of the answered scores. The curve is not extrapolated
    # beyond it.
    logit_min: float
    logit_max: float
    n: int

    def predict(self, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Curve and 95% band at `scores`: (p, lower, upper)."""
        x = score_to_logit(scores)
        eta = self.intercept + self.slope * x
        design = np.stack([np.ones_like(x), x], axis=-1)
        var = np.einsum("...i,ij,...j->...", design, self.covariance, design)
        half = Z_95 * np.sqrt(np.maximum(var, 0.0))
        return _sigmoid(eta), _sigmoid(eta - half), _sigmoid(eta + half)

    def threshold(self, target: float) -> float | None:
        """Lowest score from which labels are right at least `target` of the time.

        This is the score where the curve reaches the target. With a
        rising curve every label above it is right even more often, so
        the advice stays on the cautious side. None ("not reached") when
        the curve only gets there beyond the highest answered score, or
        does not rise at all. When the target is already met at the
        lowest answered score, that score is returned: the answers say
        nothing about what happens below it.
        """
        if self.slope <= 0:
            return None
        x_star = (np.log(target / (1.0 - target)) - self.intercept) / self.slope
        if x_star > self.logit_max:
            return None
        x_star = max(x_star, self.logit_min)
        return float(_sigmoid(np.asarray(x_star)))


def fit_calibration(scores: np.ndarray, right: np.ndarray) -> CalibrationFit:
    """Fit the curve on answered scores. `right` is 1 for right, 0 for wrong.

    Callers check `has_enough_answers` first; this only refuses input it
    cannot fit at all.
    """
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(right, dtype=float)
    if scores.shape != y.shape or scores.ndim != 1:
        raise ValueError("scores and right must be 1-D arrays of the same length")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("right must hold only 0 and 1")
    if y.sum() == 0 or y.sum() == len(y):
        raise ValueError("need at least one right and one wrong answer")

    x = score_to_logit(scores)
    centre = float(x.mean())
    spread = float(x.std()) or 1.0
    z = (x - centre) / spread
    design = np.stack([np.ones_like(z), z], axis=1)
    penalty = np.diag([0.0, SLOPE_PENALTY])
    beta = np.zeros(2)
    hessian = penalty
    for _ in range(_MAX_ITER):
        p = _sigmoid(design @ beta)
        gradient = design.T @ (y - p) - penalty @ beta
        hessian = (design.T * (p * (1.0 - p))) @ design + penalty
        step = np.linalg.solve(hessian, gradient)
        beta = beta + step
        if np.max(np.abs(step)) < _TOL:
            break
    else:
        raise RuntimeError("calibration fit did not converge")

    p = _sigmoid(design @ beta)
    hessian = (design.T * (p * (1.0 - p))) @ design + penalty
    # Back from the standardised scale: a + b*z = (a - b*c/s) + (b/s)*x.
    to_x = np.array([[1.0, -centre / spread], [0.0, 1.0 / spread]])
    return CalibrationFit(
        intercept=float(beta[0] - beta[1] * centre / spread),
        slope=float(beta[1] / spread),
        covariance=to_x @ np.linalg.inv(hessian) @ to_x.T,
        logit_min=float(x.min()),
        logit_max=float(x.max()),
        n=len(y),
    )


def has_enough_answers(n_right: int, n_wrong: int) -> bool:
    return n_right >= MIN_RIGHT and n_wrong >= MIN_WRONG


def rule_of_three_bound(n_right: int) -> float | None:
    """Lower 95% bound on the share right when none of `n_right` answers was wrong.

    "No errors in 60 answers: at least 95% right" (1 - 3/n). The sample
    leans towards low scores, so for the label as a whole this is on the
    cautious side. None below 3 answers, where the bound is not above 0.
    """
    if n_right <= 3:
        return None
    return 1.0 - 3.0 / n_right


def draw_sample(
    candidates: list[tuple[str, float]],
    *,
    batch_size: int,
    n_bins: int,
    rng: np.random.Generator,
    score_range: tuple[float, float] | None = None,
) -> list[str]:
    """Pick up to `batch_size` ids spread evenly over the score range.

    `candidates` are (id, score) pairs not answered yet. The score range
    (`score_range`, or else the one they span) is cut in `n_bins` equal
    bins and the draws go round the bins one at a time, so a bin that
    runs out hands its share to the others. With most scores near 0.999
    the low bins are often thin, and this keeps the batch full instead
    of short.
    """
    if not candidates:
        return []
    scores = np.array([s for _, s in candidates], dtype=float)
    lo, hi = score_range if score_range else (float(scores.min()), float(scores.max()))
    if hi > lo:
        bins = np.minimum(((scores - lo) / (hi - lo) * n_bins).astype(int), n_bins - 1)
    else:
        bins = np.zeros(len(scores), dtype=int)
    queues: list[list[str]] = []
    for b in range(n_bins):
        members = [candidates[i][0] for i in np.flatnonzero(bins == b)]
        rng.shuffle(members)
        if members:
            queues.append(members)

    picked: list[str] = []
    while queues and len(picked) < batch_size:
        for queue in list(queues):
            if len(picked) == batch_size:
                break
            picked.append(queue.pop())
            if not queue:
                queues.remove(queue)
    return picked
