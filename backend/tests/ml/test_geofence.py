"""Tests for app.ml.geofence.

Validates that AddaxAI's geofence decisions match the official SpeciesNet
API's should_geofence_animal_classification(). Tests use the real
SpeciesNet v4.0.2a model data (geofence JSON + labels file).

Three test categories (per SpeciesNet developer recommendation):
1. Known-allowed and known-blocked pairs (ecologically obvious)
2. Block-rule entries (all 56 entries with block rules in v4.0.2a)
3. Exhaustive taxa/location comparison against the official API
"""

import json
import subprocess
import textwrap
from pathlib import Path

import pytest

from app.ml.geofence import (
    _get_allowed_labels_cached,
    _load_geofence_cached,
    _parse_labels_cached,
    find_geofence_file,
    find_labels_file,
    get_allowed_labels,
    load_geofence,
)

MODEL_DIR = Path.home() / "AddaxAI/models/cls/SPECIESNET-v4-0-2-A"
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
    ("african buffalo", "KEN", None),
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
    ("common wombat", "AUS", None),
    ("koala", "AUS", None),
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
    ("cheetah", "NLD", None),
    ("african buffalo", "NLD", None),
    # South American wildlife blocked in Kenya
    ("giant anteater", "KEN", None),
    ("nine-banded armadillo", "KEN", None),
    ("mantled howler monkey", "KEN", None),
    # Australian wildlife blocked in Kenya
    ("common wombat", "KEN", None),
    ("koala", "KEN", None),
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
    ("common wombat", "NLD", None),
    ("koala", "NLD", None),
]


# --- Block-rule pairs ---
# All species with block rules in v4.0.2a geofence. For each, test a
# blocked country and (when available) an allowed-but-not-blocked country.
# v4.0.2a has 56 block-rule entries covering pittidae, macropodidae,
# hippopotamidae, potoroidae, and individual species.

BLOCK_RULE_PAIRS = [
    # Pittidae (birds, blocked in 91 countries)
    ("blue pitta", "ABW", None, False),
    ("noisy pitta", "ABW", None, False),
    ("blue-winged pitta", "ABW", None, False),
    # Roe deer: allowed in Europe, blocked in USA
    ("european roe deer", "NLD", None, True),
    ("european roe deer", "DEU", None, True),
    ("european roe deer", "USA", None, False),
    # Sika deer: allowed in China; in USA, only DE and NJ are truly
    # allowed (in allow but not in block). TX/CA/etc. are in both
    # allow and block, so block wins.
    ("sika deer", "CHN", None, True),
    ("sika deer", "USA", "DE", True),   # allowed (allow, no block)
    ("sika deer", "USA", "NJ", True),   # allowed (allow, no block)
    ("sika deer", "USA", "TX", False),  # blocked (in both allow and block)
    ("sika deer", "USA", "CA", False),  # blocked (block only)
    # Hippopotamus: blocked in 210 countries (not in allow for most)
    ("hippopotamus", "ABW", None, False),
    # Dingo: blocked everywhere except no allowed-only country exists
    ("dingo", "ABW", None, False),
    ("dingo", "NLD", None, False),
    # Badgers
    ("eurasian badger", "AFG", None, True),
    ("eurasian badger", "USA", None, False),
    ("asian badger", "USA", None, False),
    # Macropodidae (kangaroos/wallabies, blocked in 245 countries)
    ("eastern grey kangaroo", "ABW", None, False),
    ("western gray kangaroo", "ABW", None, False),
    ("red kangaroo", "ABW", None, False),
    ("agile wallaby", "ABW", None, False),
    ("swamp wallaby", "ABW", None, False),
    ("quokka", "ABW", None, False),
    ("red-necked wallaby", "ABW", None, False),
    ("common wallaroo", "ABW", None, False),
    # Potoroidae (potoroos/bettongs, blocked in 248 countries)
    ("rufous bettong", "ABW", None, False),
    ("long-nosed potoroo", "ABW", None, False),
    ("long-footed potoroo", "ABW", None, False),
    # Eurasian red squirrel: allowed in Europe, blocked in USA
    ("eurasian red squirrel", "ALA", None, True),
    ("eurasian red squirrel", "USA", None, False),
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
    """Species with both allow and block rules (edge cases).

    v4.0.2a has 56 entries with block rules. These test representative
    species from each group with both a blocked and (when available)
    an allowed country.
    """

    @pytest.mark.parametrize(
        "species,country,state,expected_allowed",
        BLOCK_RULE_PAIRS,
        ids=[
            f"{s}-{c}-{'allowed' if a else 'blocked'}"
            for s, c, _, a in BLOCK_RULE_PAIRS
        ],
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


# --- geofence_fixes.csv entries ---
# Per SpeciesNet developer recommendation: include every (taxon,
# country, state) combination from geofence_fixes.csv (the file used
# to generate the release geofence from the base geofence). These
# are the "complicated" cases where bugs are most likely.

FIXES_CSV = Path(__file__).parent / "fixtures" / "geofence_fixes.csv"


def _load_fix_cases() -> list[tuple]:
    """Parse geofence_fixes.csv and expand each row to concrete test cases.

    Each row has: species (taxonomy key), rule (allow/block), country,
    state. We find all species in the labels file whose taxonomy key
    matches the row's species (exact or descendant match) and create
    one test case per matching species.
    """
    import csv
    from collections import defaultdict

    if not FIXES_CSV.exists() or not MODEL_DIR.exists():
        return []

    labels_path = find_labels_file(MODEL_DIR)
    if not labels_path:
        return []

    # Build index: taxonomy key prefix -> list of (full_label, taxonomy_key)
    species_by_prefix: dict[str, list[tuple[str, str]]] = defaultdict(list)
    with open(labels_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(";")
            if len(parts) < 7:
                continue
            taxonomy_key = ";".join(parts[1:6])
            full_label = line
            # Index at every prefix level
            for i in range(1, 6):
                prefix = ";".join(parts[1:1 + i]) + ";" * (5 - i)
                species_by_prefix[prefix].append((full_label, taxonomy_key))

    cases = []
    with open(FIXES_CSV) as f:
        reader = csv.DictReader(
            line for line in f if not line.strip().startswith("#")
        )
        for row in reader:
            species_key = row["species"]
            rule = row["rule"]
            country = row["country_code"]
            state = row["admin1_region_code"] or None
            matches = species_by_prefix.get(species_key, [])
            for full_label, tk in matches:
                # Exact or descendant match
                if tk == species_key or tk.startswith(
                    species_key.rstrip(";") + ";"
                ):
                    cases.append((full_label, rule, country, state))

    return cases


@requires_model
@requires_env
@pytest.mark.slow
class TestGeofenceFixesMatchOfficialAPI:
    """Verify AddaxAI matches the official API for every fix case.

    Per SpeciesNet developer recommendation: test every (taxon,
    country, state) combination in geofence_fixes.csv. If my code
    matches the official should_geofence_animal_classification() on
    every fix case, I can be confident about the corner cases.
    """

    def test_fixes_match_official(self):
        _parse_labels_cached.cache_clear()
        _get_allowed_labels_cached.cache_clear()
        _load_geofence_cached.cache_clear()

        cases = _load_fix_cases()
        assert cases, "No fix cases loaded"

        geofence_path = find_geofence_file(MODEL_DIR)
        labels_path = find_labels_file(MODEL_DIR)
        assert geofence_path and labels_path

        # Run official API via subprocess on all cases
        queries = [(c[0], c[2], c[3]) for c in cases]
        script = textwrap.dedent("""\
            import json, sys
            from speciesnet.geofence_utils import (
                should_geofence_animal_classification,
            )

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
            timeout=120,
        )
        assert proc.returncode == 0, (
            f"Official API subprocess failed: {proc.stderr}"
        )
        official_results = json.loads(proc.stdout)

        # Compare against AddaxAI for each case
        cache: dict[tuple, list[str]] = {}
        mismatches = []

        # Build name-dedup mapping same way the test does
        seen_names: set[str] = set()
        label_to_name: dict[str, str] = {}
        with open(labels_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(";")
                if len(parts) < 7:
                    continue
                common_name = parts[6]
                if not common_name or common_name in seen_names:
                    taxonomy = [p for p in parts[1:6] if p]
                    if taxonomy:
                        common_name = taxonomy[-1]
                if common_name in seen_names:
                    common_name = f"{common_name} ({parts[0][:8]})"
                seen_names.add(common_name)
                label_to_name[line] = common_name

        for i, (full_label, rule, country, state) in enumerate(cases):
            key = (country, state)
            if key not in cache:
                cache[key] = get_allowed_labels(MODEL_DIR, country, state)
            common_name = label_to_name.get(full_label)
            addaxai_allowed = common_name in cache[key]
            official_blocked = official_results[i]
            addaxai_blocked = not addaxai_allowed

            if addaxai_blocked != official_blocked:
                mismatches.append(
                    f"{common_name} in {country}/{state or '-'}: "
                    f"official={'block' if official_blocked else 'allow'} "
                    f"addaxai={'block' if addaxai_blocked else 'allow'}"
                )

        assert not mismatches, (
            f"{len(mismatches)} mismatches on {len(cases)} fix cases:\n"
            + "\n".join(mismatches[:20])
        )


# --- Exhaustive comparison vs official API ---

@requires_model
@requires_env
@pytest.mark.slow
class TestExhaustiveMatchOfficialAPI:
    """Compare AddaxAI's geofence decisions against the official SpeciesNet API.

    Tests ALL taxa x ALL countries from the geofence data. Agreement
    must be 100% (no floating-point math involved).
    """

    def test_exhaustive_match(self):
        # Clear caches to ensure fresh data with current parse logic
        _parse_labels_cached.cache_clear()
        _get_allowed_labels_cached.cache_clear()
        _load_geofence_cached.cache_clear()

        geofence_path = find_geofence_file(MODEL_DIR)
        labels_path = find_labels_file(MODEL_DIR)
        assert geofence_path and labels_path

        geofence = load_geofence(MODEL_DIR)

        # Read raw lines to get full 7-token labels for the official API.
        # Use the same name-dedup logic as inference.py for empty names.
        raw_labels = []
        seen_names: set[str] = set()
        with open(labels_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(";")
                if len(parts) >= 7:
                    common_name = parts[6]
                    if not common_name or common_name in seen_names:
                        taxonomy = [p for p in parts[1:6] if p]
                        if taxonomy:
                            common_name = taxonomy[-1]
                    if common_name in seen_names:
                        common_name = f"{common_name} ({parts[0][:8]})"
                    seen_names.add(common_name)
                    raw_labels.append({
                        "full_label": line,
                        "common_name": common_name,
                    })

        # Get ALL countries from geofence data
        all_countries: set[str] = set()
        for rules in geofence.values():
            all_countries.update(rules.get("allow", {}).keys())
            all_countries.update(rules.get("block", {}).keys())
        sorted_countries = sorted(all_countries)

        # US states for state-level testing
        us_states = [
            "CA", "FL", "TX", "NY", "WA", "CO", "MT", "AK", "HI", "ME",
        ]

        # Build exhaustive query list: all taxa x all countries
        # Each entry: (full_label, common_name, country, state)
        queries = []
        for label_info in raw_labels:
            for country in sorted_countries:
                queries.append((
                    label_info["full_label"],
                    label_info["common_name"],
                    country,
                    None,
                ))
            # Also test US states
            for state in us_states:
                queries.append((
                    label_info["full_label"],
                    label_info["common_name"],
                    "USA",
                    state,
                ))

        # Run official API via subprocess (batch all queries at once)
        script = textwrap.dedent("""\
            import json, sys
            from speciesnet.geofence_utils import (
                should_geofence_animal_classification,
            )

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

        # Send only (full_label, country, state) to subprocess
        subprocess_queries = [
            (q[0], q[2], q[3]) for q in queries
        ]

        proc = subprocess.run(
            [str(ENV_PYTHON), "-c", script, str(geofence_path)],
            input=json.dumps(subprocess_queries),
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert proc.returncode == 0, (
            f"Official API subprocess failed: {proc.stderr}"
        )
        official_results = json.loads(proc.stdout)
        assert len(official_results) == len(queries)

        # Compare against AddaxAI
        allowed_cache: dict[tuple, list[str]] = {}
        mismatches = []

        for i, (full_label, common_name, country, state) in enumerate(queries):
            cache_key = (country, state)
            if cache_key not in allowed_cache:
                allowed_cache[cache_key] = get_allowed_labels(
                    MODEL_DIR, country, state
                )

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

        total = len(queries)
        assert not mismatches, (
            f"{len(mismatches)} geofence mismatches out of"
            f" {total} queries ({len(mismatches)/total*100:.2f}%):\n"
            + "\n".join(mismatches[:30])
        )
