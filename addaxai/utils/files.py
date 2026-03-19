"""Pure file and string utility functions for AddaxAI.

No UI or heavy dependencies — only stdlib.
"""

import datetime
import os
import re


def is_valid_float(value):
    """Check if a string can be converted to float."""
    try:
        float(value)
        return True
    except ValueError:
        return False


def get_size(path):
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


def shorten_path(path, length):
    """Truncate a path string with leading '...' if too long."""
    if len(path) > length:
        path = "..." + path[0 - length + 3:]
    return path


def natural_sort_key(s):
    """Split string into text/number chunks for natural sort ordering."""
    s = s.strip()
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]


def contains_special_characters(path):
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


def remove_ansi_escape_sequences(text):
    """Strip ANSI escape codes from a string."""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)


def sort_checkpoint_files(files):
    """Sort checkpoint filenames by embedded timestamp, most recent first."""
    def get_timestamp(file):
        timestamp_str = file.split('_')[2].split('.')[0]
        return datetime.datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")
    return sorted(files, key=get_timestamp, reverse=True)
