"""
Throttling for per-item progress callbacks.

The long ingest loops report once per file so the UI can show a real
count, but a million files must not become a million websocket frames and
a million event-loop wakeups. Every emission crosses a thread boundary
(`asyncio.run_coroutine_threadsafe`), allocates a Future and wakes the
loop, and `ws_manager.send_progress` has no rate limiting of its own.

Throttling happens here, at the source, rather than in the worker's
callback: it stops the work before it is done instead of after, and every
caller gets the same behaviour without repeating the dance.

The frontend's own 16 ms coalescing does not help with any of this. It
caps how often React re-renders, long after the frames have been built,
sent and parsed.
"""

from __future__ import annotations

import time
from collections.abc import Callable

# Slow enough that a huge run costs a handful of frames per second,
# fast enough that the count and the ETA still look live.
PROGRESS_INTERVAL_S = 0.5


class ProgressTicker:
    """
    Call `callback(done, total)` at most every `PROGRESS_INTERVAL_S`.

    Emits `(0, total)` up front so the total appears immediately rather
    than after the first interval, and always emits the final item so the
    bar reaches the end instead of stopping just short of it.

    A `None` callback makes every method a no-op, so callers never have to
    guard the call site.
    """

    def __init__(
        self,
        callback: Callable[[int, int], None] | None,
        total: int,
    ) -> None:
        self._callback = callback
        self._total = total
        # Start the clock at the up-front emission, not at zero. Zero is
        # an eternity ago on a monotonic clock, so the first tick would
        # always fire and every run would send two frames back to back.
        self._last_emit = time.monotonic()
        if self._callback:
            self._callback(0, total)

    def tick(self, done: int) -> None:
        """Report `done` of the total, subject to the throttle."""
        if not self._callback:
            return
        now = time.monotonic()
        if now - self._last_emit >= PROGRESS_INTERVAL_S or done >= self._total:
            self._last_emit = now
            self._callback(done, self._total)
