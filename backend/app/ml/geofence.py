"""
Geofence module for geographic label filtering.

Reads geofence data from classification model directories to determine
which labels are valid for a given country/state. Used to auto-populate
excluded_classes when a project has geographic location configured.

The geofence JSON maps taxonomy keys to allowed countries:
    {
        "mammalia;carnivora;felidae;panthera;pardus": {
            "allow": {"KEN": [], "USA": ["CA", "FL"], ...}
        }
    }

The labels file maps taxonomy keys to common names:
    UUID;class;order;family;genus;species;common_name

Matching is exact: each label's taxonomy key is checked directly
against the geofence. No parent traversal is needed because the
geofence already contains entries at every taxonomy level that the
model can produce.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from app.core.logging_config import get_logger

logger = get_logger(__name__)

# Known labels file pattern for SpeciesNet models
LABELS_EXTENSION = ".labels.txt"


def find_geofence_file(model_dir: Path) -> Path | None:
    """
    Find a geofence JSON file in a model directory.

    Looks for files matching 'geofence_release.*.json'.

    Args:
        model_dir: Path to model directory

    Returns:
        Path to geofence file, or None if not found
    """
    matches = sorted(model_dir.glob("geofence_release.*.json"))
    if matches:
        return matches[-1]
    return None


def find_labels_file(model_dir: Path) -> Path | None:
    """
    Find a labels file in a model directory.

    Looks for files matching '*.labels.txt'.

    Args:
        model_dir: Path to model directory

    Returns:
        Path to labels file, or None if not found
    """
    matches = sorted(model_dir.glob(f"*{LABELS_EXTENSION}"))
    if matches:
        return matches[0]
    return None


@lru_cache(maxsize=4)
def _load_geofence_cached(geofence_path: str) -> dict:
    """Load and cache geofence JSON (keyed by string path for lru_cache)."""
    with open(geofence_path) as f:
        return json.load(f)


def load_geofence(model_dir: Path) -> dict | None:
    """
    Load geofence data from a model directory.

    Results are cached in memory after first load.

    Args:
        model_dir: Path to model directory

    Returns:
        Parsed geofence dict, or None if no geofence file exists
    """
    geofence_path = find_geofence_file(model_dir)
    if geofence_path is None:
        return None
    return _load_geofence_cached(str(geofence_path))


@lru_cache(maxsize=4)
def _parse_labels_cached(labels_path: str) -> tuple[tuple[str, str], ...]:
    """Parse and cache labels file (keyed by string path for lru_cache)."""
    labels = []
    with open(labels_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(";")
            if len(parts) < 7:
                continue
            taxonomy_key = ";".join(parts[1:6])
            common_name = parts[6]
            labels.append((common_name, taxonomy_key))
    return tuple(labels)


def parse_labels_file(labels_path: Path) -> list[dict]:
    """
    Parse a SpeciesNet-format labels file.

    Each line has format: UUID;class;order;family;genus;species;common_name
    Results are cached after first load.

    Args:
        labels_path: Path to .labels.txt file

    Returns:
        List of dicts with 'common_name' and 'taxonomy_key' fields
    """
    return [
        {"common_name": name, "taxonomy_key": key}
        for name, key in _parse_labels_cached(str(labels_path))
    ]


@lru_cache(maxsize=64)
def _get_allowed_labels_cached(
    geofence_path: str,
    labels_path: str,
    country_code: str,
    state_code: str | None,
) -> tuple[str, ...]:
    """Cached core of get_allowed_labels (string args for lru_cache)."""
    geofence = _load_geofence_cached(geofence_path)
    labels = _parse_labels_cached(labels_path)
    country_upper = country_code.upper()
    allowed = []

    for common_name, taxonomy_key in labels:
        if taxonomy_key == ";;;;":
            allowed.append(common_name)
            continue

        geofence_entry = geofence.get(taxonomy_key)
        if geofence_entry is None:
            allowed.append(common_name)
            continue

        allow_dict = geofence_entry.get("allow", {})
        if country_upper not in allow_dict:
            continue

        if (
            country_upper == "USA"
            and state_code
            and state_code.upper() not in ("NONE", "")
        ):
            state_list = allow_dict[country_upper]
            if state_list and state_code.upper() not in state_list:
                continue

        allowed.append(common_name)

    return tuple(allowed)


def get_allowed_labels(
    model_dir: Path,
    country_code: str,
    state_code: str | None = None,
) -> list[str]:
    """
    Get labels allowed for a given country/state based on geofence data.

    For each label in the model's labels file, checks whether the label's
    taxonomy key exists in the geofence and whether the country (and
    optionally state) is in the allow list. Results are cached.

    Args:
        model_dir: Path to model directory
        country_code: ISO country code (e.g., 'KEN', 'USA')
        state_code: Optional US state code (e.g., 'CA', 'TX')

    Returns:
        List of allowed common_name strings

    Raises:
        FileNotFoundError: If geofence or labels file is missing
    """
    geofence_path = find_geofence_file(model_dir)
    if geofence_path is None:
        raise FileNotFoundError(
            f"No geofence file found in {model_dir}"
        )

    labels_path = find_labels_file(model_dir)
    if labels_path is None:
        raise FileNotFoundError(
            f"No labels file found in {model_dir}"
        )

    return list(_get_allowed_labels_cached(
        str(geofence_path), str(labels_path), country_code, state_code,
    ))


def get_all_labels(model_dir: Path) -> list[str]:
    """
    Get all label common names from a model's labels file.

    Args:
        model_dir: Path to model directory

    Returns:
        List of all common_name strings

    Raises:
        FileNotFoundError: If labels file is missing
    """
    labels_path = find_labels_file(model_dir)
    if labels_path is None:
        raise FileNotFoundError(
            f"No labels file found in {model_dir}"
        )
    labels = parse_labels_file(labels_path)
    return [entry["common_name"] for entry in labels]


def compute_excluded_classes(
    model_dir: Path,
    country_code: str,
    state_code: str | None = None,
) -> list[str]:
    """
    Compute excluded_classes list for a country/state combination.

    Returns all labels NOT allowed for the given country/state.

    Args:
        model_dir: Path to model directory
        country_code: ISO country code
        state_code: Optional US state code

    Returns:
        List of excluded common_name strings
    """
    allowed = set(
        get_allowed_labels(model_dir, country_code, state_code)
    )
    all_labels = get_all_labels(model_dir)
    return [label for label in all_labels if label not in allowed]


def get_available_countries(model_dir: Path) -> list[str]:
    """
    Get all country codes that appear in any geofence entry.

    Args:
        model_dir: Path to model directory

    Returns:
        Sorted list of ISO country codes
    """
    geofence = load_geofence(model_dir)
    if geofence is None:
        return []

    countries = set()
    for entry in geofence.values():
        countries.update(entry.get("allow", {}).keys())

    return sorted(countries)
