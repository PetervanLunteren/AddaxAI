"""Tests for app.ml.geofence.

Validates that AddaxAI's geofence decisions match the official SpeciesNet
API's should_geofence_animal_classification(). Tests use the real
SpeciesNet model data (geofence JSON + labels file).

Three test categories (per SpeciesNet developer recommendation):
1. Known-allowed and known-blocked pairs (ecologically obvious)
2. Random taxa/location pairs compared against the official API
3. Block-rule entries (TODO: add after SpeciesNet update)
"""

import json
import random
import subprocess
import textwrap
from pathlib import Path

import pytest

from app.ml.geofence import (
    find_geofence_file,
    find_labels_file,
    get_allowed_labels,
    load_geofence,
    parse_labels_file,
)

MODEL_DIR = Path.home() / "AddaxAI/models/cls/SPECIESNET-v4-0-1-A-v1"
ENV_PYTHON = Path.home() / "AddaxAI/envs/env-addaxai-base/bin/python"

requires_model = pytest.mark.skipif(
    not MODEL_DIR.exists(), reason="SpeciesNet model not installed"
)
requires_env = pytest.mark.skipif(
    not ENV_PYTHON.exists(), reason="env-addaxai-base not installed"
)


# --- Known-allowed pairs (25) ---
# Species that should be allowed in their native range.

KNOWN_ALLOWED = [
    # African wildlife in Kenya
    ("african elephant", "KEN", None),
    ("lion", "KEN", None),
    ("giraffe", "KEN", None),
    ("common wildebeest", "KEN", None),
    ("impala", "KEN", None),
    ("plains zebra", "KEN", None),
    ("cheetah", "KEN", None),
    ("spotted hyaena", "KEN", None),
    ("olive baboon", "KEN", None),
    ("hippopotamus", "KEN", None),
    # European wildlife in Netherlands
    ("european roe deer", "NLD", None),
    ("beech marten", "NLD", None),
    ("european rabbit", "NLD", None),
    ("western european hedgehog", "NLD", None),
    ("red fox", "NLD", None),
    # North American wildlife in USA
    ("white-tailed deer", "USA", None),
    ("northern raccoon", "USA", None),
    ("coyote", "USA", None),
    ("wild turkey", "USA", None),
    ("virginia opossum", "USA", None),
    # South American wildlife in Brazil
    ("giant anteater", "BRA", None),
    ("nine-banded armadillo", "BRA", None),
    ("colombian red howler monkey", "BRA", None),
    # Australian wildlife in Australia
    ("eastern grey kangaroo", "AUS", None),
    ("common wombat", "AUS", None),
]


# --- Known-blocked pairs (25) ---
# Species that should not appear in these countries.

KNOWN_BLOCKED = [
    # African wildlife blocked in Netherlands
    ("african elephant", "NLD", None),
    ("lion", "NLD", None),
    ("giraffe", "NLD", None),
    ("impala", "NLD", None),
    ("olive baboon", "NLD", None),
    ("spotted hyaena", "NLD", None),
    ("hippopotamus", "NLD", None),
    ("cheetah", "NLD", None),
    # South American wildlife blocked in Kenya
    ("giant anteater", "KEN", None),
    ("nine-banded armadillo", "KEN", None),
    ("mantled howler monkey", "KEN", None),
    # Australian wildlife blocked in Kenya
    ("eastern grey kangaroo", "KEN", None),
    ("common wombat", "KEN", None),
    # North American wildlife blocked in Netherlands
    ("eastern gray squirrel", "NLD", None),
    ("pronghorn", "NLD", None),
    ("american black bear", "NLD", None),
    ("virginia opossum", "NLD", None),
    # African wildlife blocked in Australia
    ("lion", "AUS", None),
    ("giraffe", "AUS", None),
    ("impala", "AUS", None),
    # European wildlife blocked in Brazil
    ("european roe deer", "BRA", None),
    ("western european hedgehog", "BRA", None),
    # South American wildlife blocked in Netherlands
    ("giant anteater", "NLD", None),
    # Australian wildlife blocked in Netherlands
    ("eastern grey kangaroo", "NLD", None),
    ("common wombat", "NLD", None),
]


# --- Block-rule pairs ---
# European roe deer: allowed in USA via allow list, but blocked via block rule.
# Dingo: allowed everywhere via allow, blocked everywhere via block, except AUS.

BLOCK_RULE_PAIRS = [
    # Roe deer: allowed in Europe, blocked in USA despite being in allow list
    ("european roe deer", "NLD", None, True),   # allowed (in allow, no block)
    ("european roe deer", "DEU", None, True),   # allowed (in allow, no block)
    ("european roe deer", "USA", None, False),  # blocked (in allow AND block)
    # Dingo: allowed AND blocked everywhere, only truly allowed in AUS
    # (AUS is in the allow list but NOT in the block list)
    ("dingo", "AUS", None, True),               # allowed (in allow, not in block)
    ("dingo", "NLD", None, False),              # blocked (in allow AND block)
    ("dingo", "KEN", None, False),              # blocked (in allow AND block)
]


@requires_model
class TestKnownAllowedSpecies:
    """Species that should be allowed in their native range."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.allowed_labels = {}

    def _get_allowed(self, country, state=None):
        cache_key = (country, state)
        if cache_key not in self.allowed_labels:
            self.allowed_labels[cache_key] = get_allowed_labels(
                MODEL_DIR, country, state
            )
        return self.allowed_labels[cache_key]

    @pytest.mark.parametrize(
        "species,country,state",
        KNOWN_ALLOWED,
        ids=[f"{s}-{c}" for s, c, _ in KNOWN_ALLOWED],
    )
    def test_allowed(self, species, country, state):
        allowed = self._get_allowed(country, state)
        assert species in allowed, (
            f"{species} should be allowed in {country}"
            f" but is not in the allowed list"
        )


@requires_model
class TestKnownBlockedSpecies:
    """Species that should not appear outside their native range."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.allowed_labels = {}

    def _get_allowed(self, country, state=None):
        cache_key = (country, state)
        if cache_key not in self.allowed_labels:
            self.allowed_labels[cache_key] = get_allowed_labels(
                MODEL_DIR, country, state
            )
        return self.allowed_labels[cache_key]

    @pytest.mark.parametrize(
        "species,country,state",
        KNOWN_BLOCKED,
        ids=[f"{s}-{c}" for s, c, _ in KNOWN_BLOCKED],
    )
    def test_blocked(self, species, country, state):
        allowed = self._get_allowed(country, state)
        assert species not in allowed, (
            f"{species} should be blocked in {country}"
            f" but is in the allowed list"
        )


@requires_model
class TestBlockRules:
    """Species with both allow and block rules (edge cases)."""

    @pytest.mark.parametrize(
        "species,country,state,expected_allowed",
        BLOCK_RULE_PAIRS,
        ids=[
            f"{s}-{c}-{'allowed' if a else 'blocked'}"
            for s, c, _, a in BLOCK_RULE_PAIRS
        ],
    )
    @pytest.mark.xfail(
        reason="AddaxAI does not implement geofence block rules yet",
        strict=False,
    )
    def test_block_rule(self, species, country, state, expected_allowed):
        allowed = get_allowed_labels(MODEL_DIR, country, state)
        if expected_allowed:
            assert species in allowed, (
                f"{species} should be allowed in {country}"
            )
        else:
            assert species not in allowed, (
                f"{species} should be blocked in {country}"
            )


# --- Random taxa vs official API ---

@requires_model
@requires_env
@pytest.mark.slow
class TestRandomTaxaMatchOfficialAPI:
    """Compare AddaxAI's geofence decisions against the official SpeciesNet API.

    Tests 500 random taxa x 10 random countries (+ 10 US states for USA).
    Agreement must be 100% (no floating-point math involved).
    """

    def test_random_taxa_match(self):
        geofence_path = find_geofence_file(MODEL_DIR)
        labels_path = find_labels_file(MODEL_DIR)
        assert geofence_path and labels_path

        geofence = load_geofence(MODEL_DIR)

        # Read raw lines to get full 7-token labels for the official API
        raw_labels = []
        with open(labels_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(";")
                if len(parts) >= 7:
                    raw_labels.append({
                        "full_label": line,
                        "common_name": parts[6],
                    })

        # Pick 500 random taxa
        rng = random.Random(42)
        sample_labels = rng.sample(raw_labels, min(500, len(raw_labels)))

        # Pick 10 random countries from geofence data
        all_countries: set[str] = set()
        for rules in geofence.values():
            all_countries.update(rules.get("allow", {}).keys())
        sample_countries = rng.sample(sorted(all_countries), 10)

        # Pick 10 US states
        us_states = [
            "CA", "FL", "TX", "NY", "WA", "CO", "MT", "AK", "HI", "ME",
        ]

        # Build query list: (full_7token_label, country, state)
        queries = []
        for label_info in sample_labels:
            for country in sample_countries:
                state = None
                if country == "USA":
                    state = rng.choice(us_states)
                queries.append((
                    label_info["full_label"],
                    country,
                    state,
                ))

        # Run official API via subprocess
        script = textwrap.dedent("""\
            import json, sys
            from speciesnet.geofence_utils import (
                should_geofence_animal_classification,
            )
            from speciesnet.taxonomy_utils import get_full_class_string

            geofence_map = json.load(open(sys.argv[1]))
            queries = json.load(sys.stdin)
            results = []
            for label, country, state in queries:
                result = should_geofence_animal_classification(
                    label, country, state, geofence_map,
                    enable_geofence=True,
                )
                results.append(result)
            json.dump(results, sys.stdout)
        """)

        proc = subprocess.run(
            [str(ENV_PYTHON), "-c", script, str(geofence_path)],
            input=json.dumps(queries),
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert proc.returncode == 0, (
            f"Official API subprocess failed: {proc.stderr}"
        )
        official_results = json.loads(proc.stdout)
        assert len(official_results) == len(queries)

        # Compare against AddaxAI
        allowed_cache: dict[tuple, list[str]] = {}
        mismatches = []

        for i, (full_label, country, state) in enumerate(queries):
            cache_key = (country, state)
            if cache_key not in allowed_cache:
                allowed_cache[cache_key] = get_allowed_labels(
                    MODEL_DIR, country, state
                )

            parts = full_label.split(";")
            common_name = parts[6] if len(parts) >= 7 else full_label
            addaxai_allowed = common_name in allowed_cache[cache_key]
            official_blocked = official_results[i]

            # official True = blocked, AddaxAI in list = allowed
            if addaxai_allowed == official_blocked:
                taxonomy_key = ";".join(full_label.split(";")[1:6])
                mismatches.append(
                    f"{common_name} in {country}"
                    f"{'/' + state if state else ''}: "
                    f"official={'blocked' if official_blocked else 'allowed'}"
                    f" addaxai={'allowed' if addaxai_allowed else 'blocked'}"
                    f" key={taxonomy_key}"
                )

        assert not mismatches, (
            f"{len(mismatches)} geofence mismatches out of"
            f" {len(queries)} queries:\n"
            + "\n".join(mismatches[:20])
        )


# TODO: after updating to the new SpeciesNet version, add tests for
# the geofence changes file (block rules added on top of allow rules).
# The current geofence JSON has 2 entries with block rules:
# - european roe deer: allowed in Europe, blocked in USA
# - dingo: allowed everywhere, blocked everywhere except AUS
# These are partially covered by TestBlockRules above, but the
# developer mentioned a separate changes file with more entries.
