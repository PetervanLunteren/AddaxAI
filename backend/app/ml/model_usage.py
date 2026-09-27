"""In-process guards for model packs used by active inference workers."""

from __future__ import annotations

import threading
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager

_lock = threading.RLock()
_active: Counter[str] = Counter()


@contextmanager
def model_usage_guard() -> Iterator[None]:
    """Serialize model deletion against a worker starting to use that model."""
    with _lock:
        yield


def acquire_model_usage(model_ids: Iterable[str | None]) -> Callable[[], None]:
    """Mark models active until the returned idempotent release is called."""
    ids = tuple({model_id for model_id in model_ids if model_id})
    with _lock:
        _active.update(ids)

    released = False

    def release() -> None:
        nonlocal released
        with _lock:
            if released:
                return
            released = True
            for model_id in ids:
                _active[model_id] -= 1
                if _active[model_id] <= 0:
                    del _active[model_id]

    return release


def is_model_active(model_id: str) -> bool:
    with _lock:
        return _active.get(model_id, 0) > 0
