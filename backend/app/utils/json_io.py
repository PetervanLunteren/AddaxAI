"""Verified JSON writes for files that are expensive to recreate.

A detection or classification results JSON can represent days of GPU
work, and it often lives on an external drive. A write the OS reports
as successful can still land corrupt there (delayed-write failures on
removable drives, a full or failing disk). Seen in the field on
2026-10-01: a 45-hour MegaDetector run wrote its results JSON, the
write returned cleanly, and nine seconds later the file was unparseable
16 MB in. `json.dump` cannot produce invalid JSON, so bytes on disk
that differ from what Python wrote are a storage problem; the job here
is to catch that while the data still exists in memory, when a retry
is still possible.
"""

import json
import os
from pathlib import Path
from typing import Any

from app.core.logging_config import get_logger

logger = get_logger(__name__)

WRITE_ATTEMPTS = 3


def write_json_verified(path: Path, obj: Any, *, indent: int | None = 2) -> None:
    """Write `obj` as JSON to `path`, verified, atomically.

    Writes to a `.partial` sibling, fsyncs, parses the written file
    back, and only then renames it over `path`, so a crash or a failed
    attempt never leaves a half file under the real name. Retries up to
    `WRITE_ATTEMPTS` times, then raises OSError naming the path.

    The verify read can be served from the OS cache, so a drive that
    lies past fsync can still slip through. This catches truncation,
    disk-full and delayed-write failures, which are the observed cases.
    """
    partial = path.with_name(path.name + ".partial")
    last_error: Exception | None = None
    try:
        for attempt in range(1, WRITE_ATTEMPTS + 1):
            try:
                with open(partial, "w") as f:
                    json.dump(obj, f, indent=indent)
                    f.flush()
                    os.fsync(f.fileno())
                # Validate without building the object: a full json.load
                # would hold a second copy of a result set that already
                # sits in the caller's memory, roughly 4x the file size
                # extra at the end of exactly the giant runs this guards.
                # The discard hook keeps it to the slurped file text.
                with open(partial) as f:
                    json.load(f, object_pairs_hook=lambda pairs: None)
                os.replace(partial, path)
                return
            except (OSError, ValueError) as e:
                last_error = e
                logger.warning(
                    f"Write attempt {attempt}/{WRITE_ATTEMPTS} for {path.name} "
                    f"produced an unreadable file: {e}"
                )
        raise OSError(
            f"Could not write a readable results file to {path} after "
            f"{WRITE_ATTEMPTS} attempts. The drive may be full or failing. "
            f"Last error: {last_error}"
        )
    finally:
        # Covers every exit: a failed attempt, the final raise, and
        # errors the retry loop does not own (a non-serializable object
        # raises TypeError on the first attempt). After a successful
        # rename the partial no longer exists.
        partial.unlink(missing_ok=True)
