"""Tests for locale-independent streamed subprocess output."""

from __future__ import annotations

import io
import subprocess
import sys
from typing import Any, cast

from app.utils.subprocess_runner import stream_with_tail


class _CompletedProcess:
    """Small Popen stand-in for inspecting how the runner opens pipes."""

    def __init__(self, output: str = "done\n", returncode: int = 0) -> None:
        self.stdout = io.StringIO(output)
        self.returncode = returncode

    def poll(self) -> int:
        return self.returncode

    def wait(self) -> int:
        return self.returncode


def test_popen_uses_utf8_with_replacement() -> None:
    observed: dict[str, Any] = {}

    def factory(_cmd: list[str], **kwargs: Any) -> subprocess.Popen[Any]:
        observed.update(kwargs)
        return cast(subprocess.Popen[Any], _CompletedProcess())

    result = stream_with_tail(["example"], popen_factory=factory)

    assert result.output_tail == ["done"]
    assert observed["text"] is True
    assert observed["encoding"] == "utf-8"
    assert observed["errors"] == "replace"


def test_stream_decodes_utf8_and_replaces_malformed_bytes() -> None:
    expected_line = "日本語 🦌"
    payload = expected_line.encode("utf-8") + b"\nbroken:\x81\n"
    script = (
        "import sys; "
        f"sys.stdout.buffer.write(bytes.fromhex('{payload.hex()}')); "
        "sys.stdout.buffer.flush()"
    )
    seen: list[str] = []

    result = stream_with_tail([sys.executable, "-c", script], on_line=seen.append)

    assert result.returncode == 0
    assert result.output_tail == [expected_line, "broken:\ufffd"]
    assert result.last_line == "broken:\ufffd"
    assert seen == result.output_tail


def test_nonzero_exit_keeps_output_tail() -> None:
    script = "import sys; print('failure detail', flush=True); sys.exit(7)"

    result = stream_with_tail([sys.executable, "-c", script])

    assert result.returncode == 7
    assert result.last_line == "failure detail"
    assert result.output_tail == ["failure detail"]
