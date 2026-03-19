"""Tests for addaxai.utils.files — pure file/string utility functions."""

import os
import pytest

from addaxai.utils.files import (
    contains_special_characters,
    get_size,
    is_valid_float,
    natural_sort_key,
    remove_ansi_escape_sequences,
    shorten_path,
    sort_checkpoint_files,
)


# --- is_valid_float ---

def test_is_valid_float_with_int():
    assert is_valid_float("42") is True


def test_is_valid_float_with_float():
    assert is_valid_float("3.14") is True


def test_is_valid_float_with_text():
    assert is_valid_float("abc") is False


def test_is_valid_float_with_empty():
    assert is_valid_float("") is False


# --- get_size ---

def test_get_size_bytes(tmp_path):
    f = tmp_path / "tiny.txt"
    f.write_text("hi")
    result = get_size(str(f))
    assert "bytes" in result


def test_get_size_kb(tmp_path):
    f = tmp_path / "medium.txt"
    f.write_bytes(b"x" * 2048)
    result = get_size(str(f))
    assert "KB" in result


# --- shorten_path ---

def test_shorten_path_short_unchanged():
    assert shorten_path("/a/b", 20) == "/a/b"


def test_shorten_path_long_truncated():
    result = shorten_path("/very/long/path/to/some/deep/folder", 15)
    assert result.startswith("...")
    assert len(result) == 15


# --- natural_sort_key ---

def test_natural_sort_key_ordering():
    items = ["IMG_10.jpg", "IMG_2.jpg", "IMG_1.jpg"]
    sorted_items = sorted(items, key=natural_sort_key)
    assert sorted_items == ["IMG_1.jpg", "IMG_2.jpg", "IMG_10.jpg"]


def test_natural_sort_key_case_insensitive():
    items = ["Banana", "apple", "Cherry"]
    sorted_items = sorted(items, key=natural_sort_key)
    assert sorted_items == ["apple", "Banana", "Cherry"]


# --- contains_special_characters ---

def test_contains_special_characters_clean_path():
    result = contains_special_characters("/home/user/photos")
    assert result[0] is False


def test_contains_special_characters_with_special():
    result = contains_special_characters("/home/user/phötös")
    assert result[0] is True
    assert result[1] == "ö"


# --- remove_ansi_escape_sequences ---

def test_remove_ansi_plain_text():
    assert remove_ansi_escape_sequences("hello world") == "hello world"


def test_remove_ansi_with_codes():
    text = "\x1B[31mred text\x1B[0m"
    assert remove_ansi_escape_sequences(text) == "red text"


# --- sort_checkpoint_files ---

def test_sort_checkpoint_files_ordering():
    files = [
        "md_checkpoint_20240101120000.json",
        "md_checkpoint_20240315080000.json",
        "md_checkpoint_20240201150000.json",
    ]
    sorted_files = sort_checkpoint_files(files)
    assert sorted_files[0] == "md_checkpoint_20240315080000.json"  # most recent first
    assert sorted_files[-1] == "md_checkpoint_20240101120000.json"
