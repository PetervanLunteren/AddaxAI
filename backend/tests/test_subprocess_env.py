"""Tests for the hardened subprocess environment builder."""

from app.utils.subprocess_env import clean_python_env


def test_clean_python_env_strips_leaks(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/home/user/anaconda3/lib/site-packages")
    monkeypatch.setenv("PYTHONHOME", "/home/user/anaconda3")
    monkeypatch.setenv("PATH", "/usr/bin")

    env = clean_python_env()

    assert env["PYTHONNOUSERSITE"] == "1"
    assert "PYTHONPATH" not in env
    assert "PYTHONHOME" not in env
    # Unrelated variables pass through untouched.
    assert env["PATH"] == "/usr/bin"


def test_clean_python_env_makes_a_native_crash_leave_a_trace(monkeypatch):
    # A detector that dies with an access violation used to leave only
    # the exit code behind (3221225477, nothing else, 2026-10-11). The
    # fault handler prints the Python stack, and unbuffered output keeps
    # the lines printed just before from being lost with the process.
    monkeypatch.delenv("PYTHONFAULTHANDLER", raising=False)
    monkeypatch.delenv("PYTHONUNBUFFERED", raising=False)
    env = clean_python_env()
    assert env["PYTHONFAULTHANDLER"] == "1"
    assert env["PYTHONUNBUFFERED"] == "1"


def test_clean_python_env_overrides_win(monkeypatch):
    monkeypatch.delenv("PYTHONPATH", raising=False)
    env = clean_python_env(CUDA_VISIBLE_DEVICES="-1", PIP_RETRIES="5")
    assert env["CUDA_VISIBLE_DEVICES"] == "-1"
    assert env["PIP_RETRIES"] == "5"
    assert env["PYTHONNOUSERSITE"] == "1"
