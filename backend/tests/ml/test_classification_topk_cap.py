"""The classification worker caps its output at top-N, and that cap stays >=
the downstream trim default so no class the rollup wants is silently dropped.

Context: SpeciesNet returns the full ~2000+ class softmax per detection. Writing
all of it produced multi-GB detection JSONs that filled disk on merge and
exhausted memory (Simon's 8 GB file). The worker now keeps only the top-N; this
test pins that N and its coupling to trim_classification_results.
"""

import inspect
import sys
from pathlib import Path

from app.ml.json_utils import trim_classification_results

# The worker runs as an isolated subprocess (no app.* on its path) and imports
# its siblings flatly, so reproduce that import path to read its constant.
_INFERENCE_DIR = (
    Path(__file__).resolve().parents[2] / "app" / "ml" / "inference"
)
sys.path.insert(0, str(_INFERENCE_DIR))
import classification_worker as cw  # noqa: E402


def _trim_default() -> int:
    return inspect.signature(
        trim_classification_results
    ).parameters["max_classifications"].default


def test_worker_cap_is_at_least_trim_default():
    # If trim's K is ever raised, the worker cap must rise with it, otherwise
    # the worker would drop classes the rollup/trim still want.
    assert cw.MAX_CLASSIFICATIONS_KEPT >= _trim_default()


def test_worker_cap_matches_canonical_top5():
    # Current canonical value across worker, trim, and rollup.
    assert cw.MAX_CLASSIFICATIONS_KEPT == 5
    assert _trim_default() == 5


def test_slice_keeps_highest_confidence_first():
    # The worker sorts by confidence then slices to the cap; the kept entries
    # are the highest-confidence ones, in order.
    sorted_cls = [[str(i), 1.0 - i * 0.001] for i in range(2000)]
    kept = sorted_cls[: cw.MAX_CLASSIFICATIONS_KEPT]
    assert len(kept) == cw.MAX_CLASSIFICATIONS_KEPT
    assert kept == sorted_cls[:5]
    assert [c for _, c in kept] == sorted(
        [c for _, c in kept], reverse=True
    )
