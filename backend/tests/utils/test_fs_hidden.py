"""
A folder AddaxAI may not write to fails with a sentence, not an errno.

The read-only drive cannot be made on CI, so the three errnos are raised
through a monkeypatched `Path.mkdir`; the real refusal is pinned with a
`chmod 555` parent, skipped where it is not enforced (running as root).
"""

import errno
import os
from pathlib import Path

import pytest

from app.utils.fs_hidden import FOLDER_NOT_WRITABLE_MESSAGE, mkdir_hidden_addaxai


def _raise(code: int):
    def mkdir(self, *args, **kwargs):
        raise OSError(code, os.strerror(code), str(self))

    return mkdir


@pytest.mark.parametrize("code", [errno.EROFS, errno.EACCES, errno.EPERM])
def test_a_folder_that_cannot_be_written_names_the_cause(monkeypatch, tmp_path, code):
    target = tmp_path / ".addaxai" / "projects" / "p"
    monkeypatch.setattr(Path, "mkdir", _raise(code))
    with pytest.raises(OSError) as exc:
        mkdir_hidden_addaxai(target)
    assert str(exc.value).startswith(FOLDER_NOT_WRITABLE_MESSAGE)
    assert str(target) in str(exc.value)
    assert exc.value.__cause__.errno == code


def test_any_other_error_passes_through_untouched(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "mkdir", _raise(errno.ENOSPC))
    with pytest.raises(OSError) as exc:
        mkdir_hidden_addaxai(tmp_path / ".addaxai")
    assert exc.value.errno == errno.ENOSPC
    assert FOLDER_NOT_WRITABLE_MESSAGE not in str(exc.value)


def test_a_real_unwritable_folder_gets_the_message(tmp_path):
    tmp_path.chmod(0o555)
    try:
        if os.access(tmp_path, os.W_OK):
            pytest.skip("permissions not enforced (running as root)")
        with pytest.raises(OSError, match="AddaxAI cannot write to this folder"):
            mkdir_hidden_addaxai(tmp_path / ".addaxai" / "projects" / "p")
    finally:
        tmp_path.chmod(0o755)
