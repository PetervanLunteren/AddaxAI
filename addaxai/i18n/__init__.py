"""Internationalization (i18n) support for AddaxAI.

Usage:
    from addaxai.i18n import init, t, set_language, lang_idx

    # Call once at startup, passing the saved lang_idx (0=EN, 1=ES, 2=FR)
    init(lang_idx=0)

    # Look up a translated string
    label_text = t("browse")   # returns "Browse" / "Examinar" / "Parcourir"

    # Switch language (e.g. when user clicks the language button)
    set_language(1)  # switch to Spanish

    # Get current language index (for backward compatibility)
    idx = lang_idx()  # returns 0, 1, or 2
"""

import json
import os
from typing import Any, Dict

_strings: Dict[str, Dict[str, Any]] = {}   # {"en": {...}, "es": {...}, "fr": {...}}
_current: str = "en"
_LANG_CODES = ["en", "es", "fr"]


def init(lang_idx: int = 0) -> None:
    """Load all language JSON files. Call once at startup.

    Args:
        lang_idx: Index of the starting language (0=EN, 1=ES, 2=FR).
    """
    global _strings, _current
    base = os.path.dirname(__file__)
    for code in _LANG_CODES:
        path = os.path.join(base, f"{code}.json")
        with open(path, encoding="utf-8") as f:
            _strings[code] = json.load(f)
    _current = _LANG_CODES[lang_idx]


def set_language(lang_idx: int) -> None:
    """Switch the current language.

    Args:
        lang_idx: Index of the target language (0=EN, 1=ES, 2=FR).
    """
    global _current
    _current = _LANG_CODES[lang_idx]


def t(key: str) -> Any:
    """Look up a translation by key in the current language.

    Args:
        key: The translation key (e.g. "browse", "lbl_model").

    Returns:
        The translated string, or a list of strings for dpd_* keys.

    Raises:
        KeyError: If the key is not found in the current language's JSON.
    """
    return _strings[_current][key]


def lang_idx() -> int:
    """Return the current language index (0=EN, 1=ES, 2=FR).

    For backward compatibility with code that still uses lang_idx as an integer.
    """
    return _LANG_CODES.index(_current)
