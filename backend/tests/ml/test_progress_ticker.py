"""The throttle that keeps a million-file ingest from flooding the UI.

Every emission crosses a thread boundary, allocates a Future and wakes the
event loop, and the websocket manager has no rate limiting of its own. So
the loops report per file and this decides how much of that reaches the
UI. Two properties matter and neither is obvious from reading the loops:
the total has to appear immediately rather than after the first interval,
and the last item has to be emitted or the bar stops just short of the end
and looks stuck at 99%.
"""

from app.ml.progress import PROGRESS_INTERVAL_S, ProgressTicker


def _recorder():
    calls: list[tuple[int, int]] = []
    return calls, lambda done, total: calls.append((done, total))


def test_emits_the_total_up_front():
    """Otherwise the count is blank until the first interval elapses."""
    calls, cb = _recorder()
    ProgressTicker(cb, 500)
    assert calls == [(0, 500)]


def test_throttles_the_middle():
    """A tight loop must not produce one call per item."""
    calls, cb = _recorder()
    ticker = ProgressTicker(cb, 10_000)
    for done in range(1, 10_000):
        ticker.tick(done)
    # Only the up-front (0, total). Nothing else can fire, because the
    # loop finishes well inside one interval and the last item is 10_000,
    # which this loop never reaches.
    assert calls == [(0, 10_000)]


def test_always_emits_the_last_item():
    """The bar has to reach the end even when the throttle is closed."""
    calls, cb = _recorder()
    ticker = ProgressTicker(cb, 3)
    for done in (1, 2, 3):
        ticker.tick(done)
    assert calls[0] == (0, 3)
    assert calls[-1] == (3, 3)


def test_emits_once_the_interval_has_passed(monkeypatch):
    """Time-based, not count-based, so slow files still report."""
    clock = {"t": 1000.0}
    monkeypatch.setattr(
        "app.ml.progress.time.monotonic", lambda: clock["t"]
    )
    calls, cb = _recorder()
    ticker = ProgressTicker(cb, 100)
    ticker.tick(1)
    assert calls == [(0, 100)]  # throttled, no time has passed
    clock["t"] += PROGRESS_INTERVAL_S
    ticker.tick(2)
    assert calls[-1] == (2, 100)


def test_no_callback_is_a_no_op():
    """Callers should never have to guard the call site."""
    ticker = ProgressTicker(None, 10)
    ticker.tick(1)
    ticker.tick(10)


def test_zero_total_still_reports_and_does_not_divide():
    """An empty deployment must not crash the loop that reports it."""
    calls, cb = _recorder()
    ticker = ProgressTicker(cb, 0)
    ticker.tick(0)
    assert calls[0] == (0, 0)
