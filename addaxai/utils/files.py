"""Pure file and string utility functions for AddaxAI.

No UI or heavy dependencies — only stdlib.
"""

import datetime
import os
import re
from typing import Any, List, Optional


def is_valid_float(value: str) -> bool:
    """Check if a string can be converted to float."""
    try:
        float(value)
        return True
    except ValueError:
        return False


def get_size(path: str) -> Optional[str]:
    """Return human-readable file size string."""
    size = os.path.getsize(path)
    if size < 1024:
        return f"{size} bytes"
    elif size < pow(1024, 2):
        return f"{round(size / 1024, 2)} KB"
    elif size < pow(1024, 3):
        return f"{round(size / pow(1024, 2), 2)} MB"
    elif size < pow(1024, 4):
        return f"{round(size / pow(1024, 3), 2)} GB"
    return None


def shorten_path(path: str, length: int) -> str:
    """Truncate a path string with leading '...' if too long."""
    if len(path) > length:
        path = "..." + path[0 - length + 3:]
    return path


def natural_sort_key(s: str) -> List[Any]:
    """Split string into text/number chunks for natural sort ordering."""
    s = s.strip()
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]


def contains_special_characters(path: str) -> List[Any]:
    """Check if path contains characters outside the allowed set.

    Returns:
        [True, char] if a special character is found, [False, ""] otherwise.
    """
    allowed_characters = set(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-./ +\\:'()"
    )
    for char in path:
        if char not in allowed_characters:
            return [True, char]
    return [False, ""]


def remove_ansi_escape_sequences(text: str) -> str:
    """Strip ANSI escape codes from a string."""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)


def sort_checkpoint_files(files: List[str]) -> List[str]:
    """Sort checkpoint filenames by embedded timestamp, most recent first."""
    def get_timestamp(file):
        timestamp_str = file.split('_')[2].split('.')[0]
        return datetime.datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")
    return sorted(files, key=get_timestamp, reverse=True)
