"""Detection and removal of a legacy AddaxAI (v5 / v6) install.

The real install roots are absolute and platform-specific, so every test
here points the module's path helpers at a tmp_path tree instead.

Windows directory junctions cannot be created on the CI runners, so the
junction branch (`_is_junction` / `os.rmdir` on a reparse point) is not
covered here. It is verified by hand on Windows; see DEVELOPERS.md.
"""

import sys

import pytest

from app.services import legacy_install


def _make_legacy(root, version="6.37"):
    """A minimal tree that looks like a legacy install."""
    gui = root / "AddaxAI"
    gui.mkdir(parents=True)
    (gui / "AddaxAI_GUI.py").write_text("# legacy gui")
    (gui / "version.txt").write_text(f"{version}\n")
    (root / "envs").mkdir()
    (root / "envs" / "env-base").mkdir()
    (root / "envs" / "env-base" / "python").write_text("binary")
    (root / "models").mkdir()
    (root / "models" / "md_v5a.pt").write_text("weights")
    (root / "launch_count.json").write_text("{}")
    return root


@pytest.fixture
def at_root(tmp_path, monkeypatch):
    """Point the scanner at tmp_path and away from the real machine."""
    root = tmp_path / "AddaxAI_files"
    monkeypatch.setattr(legacy_install, "_user_root", lambda: root)
    monkeypatch.setattr(legacy_install, "_junction", lambda: None)
    monkeypatch.setattr(legacy_install, "_manual_root", lambda: None)
    monkeypatch.setattr(legacy_install, "_desktop_leftovers", lambda: [])
    return root


# ---------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------


def test_nothing_found_when_root_absent(at_root):
    found = legacy_install.scan()
    assert not found.found
    assert found.root is None
    assert found.removable == []


def test_found_by_gui_marker(at_root):
    _make_legacy(at_root)
    found = legacy_install.scan()
    assert found.found
    assert found.root == at_root
    assert found.version == "6.37"


def test_shim_only_folder_is_not_a_legacy_install(at_root):
    """On Windows our own installer creates AddaxAI_files just to hold the
    Timelapse shim. That must never be reported as a legacy install."""
    (at_root / "AddaxAI").mkdir(parents=True)
    (at_root / "AddaxAI" / "open.bat").write_text("@echo off")

    found = legacy_install.scan()
    assert not found.found
    assert found.root is None


def test_missing_version_file_is_tolerated(at_root):
    _make_legacy(at_root)
    (at_root / "AddaxAI" / "version.txt").unlink()

    found = legacy_install.scan()
    assert found.found
    assert found.version is None


def test_manual_root_is_reported_but_not_removable(at_root, tmp_path, monkeypatch):
    manual = _make_legacy(tmp_path / "ProgramFiles" / "AddaxAI_files")
    monkeypatch.setattr(legacy_install, "_manual_root", lambda: manual)

    found = legacy_install.scan()
    assert found.found
    assert found.manual == (manual,)
    assert found.removable == []


def test_junction_only_offered_alongside_an_install(at_root, tmp_path, monkeypatch):
    junction = tmp_path / "EcoAssist_files"
    junction.mkdir()
    monkeypatch.setattr(legacy_install, "_junction", lambda: junction)

    # No legacy install, so an orphaned link must not trigger a prompt.
    assert legacy_install.scan().junction is None

    _make_legacy(at_root)
    assert legacy_install.scan().junction == junction


# ---------------------------------------------------------------------
# Removal
# ---------------------------------------------------------------------


def test_removes_the_whole_tree_off_windows(at_root, monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    _make_legacy(at_root)

    assert legacy_install.remove() == []
    assert not at_root.exists()


def test_windows_purge_keeps_the_timelapse_shim(at_root, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    _make_legacy(at_root)
    shim = at_root / "AddaxAI" / "open.bat"
    shim.write_text("@echo off")

    assert legacy_install.remove() == []

    # The shim survives, everything legacy around it does not.
    assert shim.is_file()
    assert not (at_root / "AddaxAI" / "AddaxAI_GUI.py").exists()
    assert not (at_root / "envs").exists()
    assert not (at_root / "models").exists()
    assert not (at_root / "launch_count.json").exists()

    # And the folder no longer counts as a legacy install.
    assert not legacy_install.scan().found


def test_removal_leaves_everything_outside_the_root_alone(at_root, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    _make_legacy(at_root)
    bystander = tmp_path / "my-camera-trap-photos"
    bystander.mkdir()
    (bystander / "IMG_0001.jpg").write_text("not ours")

    legacy_install.remove()

    assert (bystander / "IMG_0001.jpg").is_file()


def test_removes_desktop_leftovers_only_during_purge(at_root, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    shortcut = tmp_path / "Desktop" / "AddaxAI.app"
    shortcut.parent.mkdir()
    shortcut.symlink_to(at_root / "AddaxAI.app")
    monkeypatch.setattr(legacy_install, "_desktop_leftovers", lambda: [shortcut])
    _make_legacy(at_root)

    # scan() must not touch the desktop: on macOS that triggers a
    # permission prompt, and it runs on every launch.
    legacy_install.scan()
    assert shortcut.is_symlink()

    legacy_install.remove()
    # Dangling by now (its target went with the root), so is_symlink,
    # not exists, is what proves it is gone.
    assert not shortcut.is_symlink()


def test_survivors_reported_when_the_marker_stays(at_root, monkeypatch):
    """A locked file leaves the install detectable, and remove() must say
    so rather than reporting success."""
    monkeypatch.setattr(sys, "platform", "darwin")
    _make_legacy(at_root)
    monkeypatch.setattr(legacy_install, "_remove", lambda path: None)

    assert legacy_install.remove() == [at_root]


def test_removal_is_a_no_op_when_nothing_is_installed(at_root):
    assert legacy_install.remove() == []
