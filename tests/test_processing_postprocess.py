"""Tests for addaxai.processing.postprocess — file separation and postprocessing."""

import math
import os
import pytest

from addaxai.processing.postprocess import (
    CONF_DIRS,
    format_size,
    move_files,
)


# --- CONF_DIRS ---

def test_conf_dirs_has_all_buckets():
    assert len(CONF_DIRS) == 11  # 0.0 through 1.0
    assert CONF_DIRS[0.0] == "conf_0.0"
    assert CONF_DIRS[1.0] == "conf_0.9-1.0"


def test_conf_dirs_keys_are_tenths():
    for key in CONF_DIRS:
        assert round(key * 10) == key * 10  # all are clean tenths


# --- format_size ---

def test_format_size_bytes():
    assert format_size(500) == "500 B"


def test_format_size_kb():
    result = format_size(2048)
    assert "KB" in result


def test_format_size_mb():
    result = format_size(5 * 1024 * 1024)
    assert "MB" in result


# --- move_files ---

def test_move_files_moves_to_detection_dir(tmp_path):
    src = tmp_path / "source"
    dst = tmp_path / "dest"
    src.mkdir()
    (src / "photo.jpg").write_text("img")

    result = move_files(
        file="photo.jpg",
        detection_type="animal",
        file_placement=1,  # move
        max_detection_conf=0.9,
        sep_conf=False,
        dst_root=str(dst),
        src_dir=str(src),
        manually_checked=False,
    )

    assert os.path.exists(os.path.join(str(dst), "animal", "photo.jpg"))
    assert not os.path.exists(os.path.join(str(src), "photo.jpg"))
    assert result == os.path.join("animal", "photo.jpg")


def test_move_files_copies_when_placement_is_2(tmp_path):
    src = tmp_path / "source"
    dst = tmp_path / "dest"
    src.mkdir()
    (src / "photo.jpg").write_text("img")

    move_files(
        file="photo.jpg",
        detection_type="person",
        file_placement=2,  # copy
        max_detection_conf=0.8,
        sep_conf=False,
        dst_root=str(dst),
        src_dir=str(src),
        manually_checked=False,
    )

    # Both source and dest should exist
    assert os.path.exists(os.path.join(str(src), "photo.jpg"))
    assert os.path.exists(os.path.join(str(dst), "person", "photo.jpg"))


def test_move_files_with_confidence_sorting(tmp_path):
    src = tmp_path / "source"
    dst = tmp_path / "dest"
    src.mkdir()
    (src / "photo.jpg").write_text("img")

    result = move_files(
        file="photo.jpg",
        detection_type="animal",
        file_placement=2,
        max_detection_conf=0.85,
        sep_conf=True,
        dst_root=str(dst),
        src_dir=str(src),
        manually_checked=False,
    )

    # Should be in animal/conf_0.8-0.9/photo.jpg
    assert "conf_0.8-0.9" in result
    assert os.path.exists(os.path.join(str(dst), result))


def test_move_files_verified_gets_verified_dir(tmp_path):
    src = tmp_path / "source"
    dst = tmp_path / "dest"
    src.mkdir()
    (src / "photo.jpg").write_text("img")

    result = move_files(
        file="photo.jpg",
        detection_type="animal",
        file_placement=2,
        max_detection_conf=0.9,
        sep_conf=True,
        dst_root=str(dst),
        src_dir=str(src),
        manually_checked=True,
    )

    assert "verified" in result


def test_move_files_empty_skips_confidence(tmp_path):
    src = tmp_path / "source"
    dst = tmp_path / "dest"
    src.mkdir()
    (src / "photo.jpg").write_text("img")

    result = move_files(
        file="photo.jpg",
        detection_type="empty",
        file_placement=2,
        max_detection_conf=0.0,
        sep_conf=True,  # conf sorting enabled but type is "empty"
        dst_root=str(dst),
        src_dir=str(src),
        manually_checked=False,
    )

    # Empty files don't get confidence subdirs
    assert "conf_" not in result
    assert result == os.path.join("empty", "photo.jpg")
