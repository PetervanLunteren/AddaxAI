"""
Subprocess runner with tail capture.

The platform's biggest diagnostic blind spot has been: long-running
external processes (micromamba, ffmpeg, model inference workers) emit
verbose progress on stdout/stderr that gets logged at debug level, and
production deployments run at INFO so the per-line output is dropped.
When such a process eventually fails, only the final summary line
survives in the RuntimeError, and the actual error (pip stack-trace,
ffmpeg codec mismatch, classification worker traceback) is lost.

This module provides a small helper that:
- Streams merged stdout/stderr line-by-line with a callback for live
  progress reporting.
- Keeps a fixed-size ring buffer of the most recent lines, so when the
  subprocess exits non-zero the caller can dump the actual error tail
  at ERROR level without polluting backend.log on the success path.

Adopt this for any subprocess where the failure mode is "exit with a
non-zero code after some stderr/stdout we'd want to see."
"""

import subprocess
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class StreamedResult:
    """Outcome of a streamed subprocess run.

    `output_tail` is oldest-first and capped at the helper's `max_tail`.
    `last_line` is the most recent non-empty line, useful for short
    user-facing error messages.
    """

    returncode: int
    last_line: str
    output_tail: list[str]


def stream_with_tail(
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
    on_line: Callable[[str], None] | None = None,
    max_tail: int = 300,
    poll_interval: float = 0.01,
    popen_factory: Callable[..., "subprocess.Popen[Any]"] = subprocess.Popen,
    job_id: str | None = None,
) -> StreamedResult:
    """
    Spawn `cmd`, stream merged stdout/stderr line-by-line, and return a
    StreamedResult once the process exits.

    Each non-empty stripped line is forwarded to `on_line` (for live
    progress / parsing) and appended to a ring buffer of the last
    `max_tail` lines. Output is decoded as UTF-8 with replacement for
    malformed bytes so Windows' locale encoding cannot abort a run.
    Blocks until the subprocess terminates.

    `popen_factory` lets callers swap in a process-group launcher
    (e.g. `app.core.subprocess_group.popen_group`) so cancellation can
    kill the whole tree. Defaults to plain `subprocess.Popen` for
    callers that don't need cancellation.

    `job_id` opts the run into cooperative cancellation: the subprocess
    is launched in its own process group and registered via
    `track_subprocess`, so a concurrent `request_cancel(job_id)` kills
    the whole tree. The caller is responsible for noticing the resulting
    non-zero exit and raising `JobCancelledError`. No-op when None, which
    keeps non-cancellable callers (most of them) on plain Popen.
    """
    # Cancellable runs need a killable process group. Only override the
    # default factory; an explicit popen_factory from the caller wins.
    if job_id is not None and popen_factory is subprocess.Popen:
        from app.core.subprocess_group import popen_group

        popen_factory = popen_group

    process = popen_factory(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
        cwd=str(cwd) if cwd else None,
    )

    last_line = ""
    output_tail: deque[str] = deque(maxlen=max_tail)

    from app.core.job_cancellation import track_subprocess

    # track_subprocess is a no-op when job_id is None, so the read loop
    # is identical for cancellable and non-cancellable callers.
    with track_subprocess(job_id, process):
        while True:
            if process.poll() is not None:
                # Drain anything still in the pipe before breaking out.
                remaining = process.stdout.read() if process.stdout else ""
                if remaining:
                    for raw in remaining.splitlines():
                        line = raw.strip()
                        if not line:
                            continue
                        last_line = line
                        output_tail.append(line)
                        if on_line:
                            on_line(line)
                break

            raw = process.stdout.readline() if process.stdout else ""
            if raw:
                line = raw.strip()
                if line:
                    last_line = line
                    output_tail.append(line)
                    if on_line:
                        on_line(line)
            else:
                time.sleep(poll_interval)

        process.wait()

    return StreamedResult(
        returncode=process.returncode,
        last_line=last_line,
        output_tail=list(output_tail),
    )


def log_subprocess_failure(
    label: str, cmd: list[str], result: StreamedResult
) -> None:
    """
    Emit a single ERROR log entry containing the captured tail of a
    failed subprocess. Call this immediately before raising so
    backend.log carries the real diagnostic content, not just the
    summary line that surfaces in the user-facing exception.
    """
    if result.output_tail:
        tail_block = "\n".join(result.output_tail)
        tail_summary = (
            f"Last {len(result.output_tail)} lines of subprocess output:\n"
            f"{tail_block}"
        )
    else:
        tail_summary = "(no output captured)"
    logger.error(
        f"{label} failed (returncode={result.returncode}). "
        f"Command: {' '.join(cmd)}\n{tail_summary}"
    )
