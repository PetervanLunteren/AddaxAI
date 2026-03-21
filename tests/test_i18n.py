"""Tests for the addaxai.i18n module (Phase 2.1).

Tests that all 3 language JSON files load correctly, have matching keys,
and that t() / set_language() / lang_idx() behave correctly.
"""

import json
import os

import pytest

from addaxai.i18n import init, t, set_language, lang_idx

# Load all 3 JSON files for structural checks
_I18N_DIR = os.path.join(os.path.dirname(__file__), "..", "addaxai", "i18n")
_LANG_CODES = ["en", "es", "fr"]


def _load_json(code):
    path = os.path.join(_I18N_DIR, f"{code}.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(autouse=True)
def reset_to_english():
    """Re-initialise to English before every test so tests are independent."""
    init(0)
    yield
    init(0)


# ── structural checks ────────────────────────────────────────────────────────

def test_all_json_files_load():
    for code in _LANG_CODES:
        data = _load_json(code)
        assert isinstance(data, dict), f"{code}.json did not load as a dict"


def test_all_json_files_have_identical_key_sets():
    dicts = {code: _load_json(code) for code in _LANG_CODES}
    keys_en = set(dicts["en"].keys())
    keys_es = set(dicts["es"].keys())
    keys_fr = set(dicts["fr"].keys())
    assert keys_en == keys_es, f"en vs es key mismatch:\n  only in en: {keys_en - keys_es}\n  only in es: {keys_es - keys_en}"
    assert keys_en == keys_fr, f"en vs fr key mismatch:\n  only in en: {keys_en - keys_fr}\n  only in fr: {keys_fr - keys_en}"


def test_no_empty_string_values_in_english():
    data = _load_json("en")
    for key, value in data.items():
        if isinstance(value, str):
            assert value != "", f"en.json: key '{key}' has an empty string value"
        elif isinstance(value, list):
            for item in value:
                assert item != "", f"en.json: key '{key}' has an empty string in its list"


def test_dpd_keys_return_lists():
    init(0)
    dpd_keys = [k for k in _load_json("en").keys() if k.startswith("dpd_")]
    assert dpd_keys, "No dpd_* keys found in en.json"
    for key in dpd_keys:
        result = t(key)
        assert isinstance(result, list), f"t('{key}') should return a list, got {type(result)}"


# ── t() function behaviour ───────────────────────────────────────────────────

def test_t_returns_english_by_default():
    init(0)
    assert t("browse") == "Browse"


def test_t_returns_spanish():
    init(1)
    assert t("browse") == "Examinar"


def test_t_returns_french():
    init(2)
    assert t("browse") == "Parcourir"


def test_t_returns_correct_dpd_vis_size_english():
    init(0)
    result = t("dpd_vis_size")
    assert isinstance(result, list)
    assert result[0] == "Extra small"
    assert len(result) == 5


def test_t_returns_correct_dpd_vis_size_spanish():
    init(1)
    result = t("dpd_vis_size")
    assert result[0] == "Extra pequeño"


def test_t_returns_correct_dpd_vis_size_french():
    init(2)
    result = t("dpd_vis_size")
    assert result[0] == "Extra petit"


def test_t_raises_on_missing_key():
    init(0)
    with pytest.raises(KeyError):
        t("this_key_does_not_exist_xyzzy")


# ── set_language() and lang_idx() ────────────────────────────────────────────

def test_set_language_switches_translation():
    init(0)
    assert t("error") == "Error"
    set_language(1)
    assert t("error") == "Error"  # same in EN and ES
    set_language(2)
    assert t("error") == "Erreur"


def test_set_language_then_lang_idx():
    init(0)
    assert lang_idx() == 0
    set_language(1)
    assert lang_idx() == 1
    set_language(2)
    assert lang_idx() == 2
    set_language(0)
    assert lang_idx() == 0


def test_init_with_lang_idx_1():
    init(1)
    assert lang_idx() == 1
    assert t("cancel") == "Cancelar"


def test_init_with_lang_idx_2():
    init(2)
    assert lang_idx() == 2
    assert t("cancel") == "Annuler"


# ── specific key spot-checks ─────────────────────────────────────────────────

def test_browse_all_languages():
    expected = {"en": "Browse", "es": "Examinar", "fr": "Parcourir"}
    for i, code in enumerate(_LANG_CODES):
        init(i)
        assert t("browse") == expected[code], f"browse mismatch for {code}"


def test_warning_all_languages():
    expected = {"en": "Warning", "es": "Advertencia", "fr": "Avertissement"}
    for i, code in enumerate(_LANG_CODES):
        init(i)
        assert t("warning") == expected[code]


def test_fst_step_all_languages():
    expected = {
        "en": "Step 1: Select folder",
        "es": "Paso 1: Seleccione carpeta",
        "fr": "Étape 1: Sélectionner le dossier",
    }
    for i, code in enumerate(_LANG_CODES):
        init(i)
        assert t("fst_step") == expected[code]


def test_dpd_exp_format_same_all_languages():
    """Export format options are identical in all 3 languages."""
    for i in range(3):
        init(i)
        result = t("dpd_exp_format")
        assert result == ["XLSX", "CSV", "COCO", "Sensing Clues (TSV)"]
