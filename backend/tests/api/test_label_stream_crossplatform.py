"""The async label stream reads subprocess stdout via a background thread +
queue (not select, which only works on sockets on Windows -> WinError 10038).

This drives stream_labels_subprocess_async against a fake subprocess and asserts
the lines are streamed through, exercising the cross-platform read path.
"""

import asyncio
import json
import threading

from app.services import label_service


class _FakeStdout:
    def __init__(self, lines: list[str]):
        self._it = iter(lines)

    def __iter__(self):
        return self._it


class _FakeStderr:
    def read(self) -> str:
        return ""


class _FakeProc:
    def __init__(self, lines: list[str]):
        self.stdout = _FakeStdout(lines)
        self.stderr = _FakeStderr()
        self.returncode = 0

    def poll(self):
        return 0  # already exited; stdout EOF drives the loop

    def kill(self):
        pass

    def wait(self, timeout=None):
        return 0


class _FakeRequest:
    async def is_disconnected(self) -> bool:
        return False


def test_async_stream_yields_subprocess_lines(monkeypatch):
    lines = [
        '{"type": "progress", "done": 1}\n',
        '{"type": "result", "order": ["a", "b"]}\n',
    ]
    monkeypatch.setattr(
        label_service.subprocess, "Popen", lambda *a, **k: _FakeProc(lines)
    )
    monkeypatch.setattr(label_service, "_get_env_python", lambda: "python")
    monkeypatch.setattr(label_service, "_get_db_path", lambda: ":memory:")

    async def run() -> list[bytes]:
        out: list[bytes] = []
        async for chunk in label_service.stream_labels_subprocess_async(
            _FakeRequest(), "sort", "p1", {}
        ):
            out.append(chunk)
        return out

    chunks = asyncio.run(run())
    text = b"".join(chunks).decode()
    assert '"type": "progress"' in text
    assert '"type": "result"' in text
    # Exactly the two input lines, each newline-terminated, no error appended.
    assert text.count("\n") == 2


class _HungProc:
    """A subprocess that never writes and never exits until killed."""

    def __init__(self):
        self.killed = threading.Event()
        self.stdout = self._block()
        self.stderr = _FakeStderr()
        self.returncode = None

    def _block(self):
        self.killed.wait(timeout=5)
        return
        yield  # makes this a generator that ends once killed

    def poll(self):
        return None if not self.killed.is_set() else -9

    def kill(self):
        self.killed.set()
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


def test_async_stream_times_out_on_the_shared_ceiling(monkeypatch):
    """A hung subprocess ends in one error line and is killed, and the
    ceiling it hits is LABELS_SUBPROCESS_TIMEOUT_S, whatever the params."""
    proc = _HungProc()
    monkeypatch.setattr(label_service.subprocess, "Popen", lambda *a, **k: proc)
    monkeypatch.setattr(label_service, "_get_env_python", lambda: "python")
    monkeypatch.setattr(label_service, "_get_db_path", lambda: ":memory:")
    monkeypatch.setattr(label_service, "LABELS_SUBPROCESS_TIMEOUT_S", -1)

    async def run() -> list[bytes]:
        out: list[bytes] = []
        async for chunk in label_service.stream_labels_subprocess_async(
            _FakeRequest(), "cohorts", "p1", {"max_detections": 999_999}
        ):
            out.append(chunk)
        return out

    lines = [json.loads(c) for c in asyncio.run(run())]

    # The first line is the timeout. (A second "failed (exit -9)" line
    # follows today, because the timeout path does not count as terminal.)
    assert lines[0] == {
        "type": "error",
        "message": "Labels subprocess timed out after -1s",
    }
    assert proc.killed.is_set()
