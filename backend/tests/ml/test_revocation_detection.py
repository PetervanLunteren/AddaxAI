"""Tests for the certificate-revocation failure and its opt-out.

Windows schannel refuses a certificate whose revocation status it cannot
establish. On a network that inspects TLS this kills every environment
build before a single package is read, and the raw micromamba output
names nothing the user can act on. These tests pin the one failure we
translate into an actionable error, and the marker file that is the only
way to skip the check.

The failure itself is Windows-only and cannot be produced on a developer
machine, so the detection is driven with the exact text a real run
emitted (reported 2026-08-14) rather than by provoking it.
"""

from pathlib import Path
from typing import Any

import pytest

from app.ml import environment_manager
from app.ml.environment_manager import (
    EnvironmentManager,
    TlsRevocationCheckError,
    allow_revocation_skip,
    is_revocation_failure,
    revocation_marker_path,
    revocation_skip_allowed,
)
from app.utils.subprocess_runner import StreamedResult

YAML = """name: env-probe
channels:
  - conda-forge
dependencies:
  - python=3.11
"""

# Verbatim from the failing Windows install, including the trailing
# "Subdir ... not loaded!" lines that are all the user's error box had
# room for. The schannel line sits above them, which is why the detection
# reads the whole captured output and not the five lines we display.
REAL_FAILURE = [
    "schannel: next InitializeSecurityContext failed: "
    "CRYPT_E_NO_REVOCATION_CHECK (0x80092012) - The revocation function "
    "was unable to check revocation for the certificate.",
    "Subdir pkgs/main/noarch not loaded!",
    "Subdir pkgs/r/noarch not loaded!",
    "Subdir pkgs/msys2/noarch not loaded!",
    "If you run into this error repeatedly, your package cache may be "
    "corrupted.",
    "critical libmamba Download error",
]

# The sibling verdict: a real revocation server that could not be
# reached, rather than a certificate carrying no revocation data.
OFFLINE_FAILURE = [
    "schannel: next InitializeSecurityContext failed: "
    "CRYPT_E_REVOCATION_OFFLINE (0x80092013) - The revocation function "
    "was unable to check revocation because the revocation server was "
    "offline.",
]

UNRELATED_FAILURE = [
    "ERROR: Could not find a version that satisfies the requirement foo",
    "critical libmamba pip failed to install packages",
]


def _build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    output_tail: list[str],
) -> None:
    """Run a build whose micromamba fails with the given output."""
    yaml_path = tmp_path / "environment.yml"
    yaml_path.write_text(YAML)
    micromamba = tmp_path / "micromamba"
    micromamba.write_text("")

    def fake_stream(cmd: list[str], **kwargs: Any) -> StreamedResult:
        return StreamedResult(
            returncode=1, last_line=output_tail[-1], output_tail=output_tail
        )

    monkeypatch.setattr(environment_manager, "stream_with_tail", fake_stream)

    mgr = EnvironmentManager(
        envs_dir=tmp_path / "envs", micromamba_path=micromamba
    )
    mgr._create_env("probe", tmp_path / "envs" / "env-probe", yaml_path)


@pytest.mark.parametrize(
    "lines", [REAL_FAILURE, OFFLINE_FAILURE], ids=["no-check", "offline"]
)
def test_both_revocation_verdicts_are_recognised(lines: list[str]) -> None:
    """One rule covers both. Same cause for the user, same fix."""
    assert is_revocation_failure(lines) is True


def test_an_ordinary_build_failure_is_not_mistaken_for_one() -> None:
    assert is_revocation_failure(UNRELATED_FAILURE) is False


def test_revocation_failure_raises_the_actionable_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The user gets wording they can act on, not micromamba's output.

    The type has to survive `_create_env`'s outer handler, which wraps
    every other exception into a plain RuntimeError. If it were wrapped,
    the API could not tell this failure from any other and would never
    offer the opt-out."""
    monkeypatch.setenv("ADDAXAI_USER_DATA_DIR", str(tmp_path))

    with pytest.raises(TlsRevocationCheckError) as exc_info:
        _build(tmp_path, monkeypatch, REAL_FAILURE)

    message = str(exc_info.value)
    assert "revoked" in message
    # The raw output belongs in backend.log, not in a box with a button
    # under it.
    assert "CRYPT_E_NO_REVOCATION_CHECK" not in message
    assert "Subdir" not in message


def test_other_failures_still_raise_the_generic_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ADDAXAI_USER_DATA_DIR", str(tmp_path))

    with pytest.raises(RuntimeError) as exc_info:
        _build(tmp_path, monkeypatch, UNRELATED_FAILURE)

    assert not isinstance(exc_info.value, TlsRevocationCheckError)
    assert "pip failed to install packages" in str(exc_info.value)


def test_no_second_offer_once_the_check_is_already_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Skipping the check did not cure it, so do not offer that again.

    Without this the user would be handed a button that reruns the same
    build with the same setting and fails the same way."""
    monkeypatch.setenv("ADDAXAI_USER_DATA_DIR", str(tmp_path))
    allow_revocation_skip()

    with pytest.raises(RuntimeError) as exc_info:
        _build(tmp_path, monkeypatch, REAL_FAILURE)

    assert not isinstance(exc_info.value, TlsRevocationCheckError)


def test_marker_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Off until written, on afterwards, and it explains itself.

    The file outlives the click that made it and is the only record that
    this machine builds without the check, so it has to say what it does
    and how to undo it."""
    monkeypatch.setenv("ADDAXAI_USER_DATA_DIR", str(tmp_path))
    assert revocation_skip_allowed() is False

    marker = allow_revocation_skip()

    assert marker == revocation_marker_path()
    assert revocation_skip_allowed() is True
    note = marker.read_text(encoding="utf-8")
    assert "Delete this file" in note
    assert "still verified" in note


def test_writing_the_marker_twice_is_harmless(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The endpoint behind it is a plain POST a user can double-click."""
    monkeypatch.setenv("ADDAXAI_USER_DATA_DIR", str(tmp_path))

    first = allow_revocation_skip()
    second = allow_revocation_skip()

    assert first == second
    assert revocation_skip_allowed() is True
