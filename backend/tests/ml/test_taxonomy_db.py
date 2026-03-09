"""Tests for app.ml.taxonomy_db."""

import csv
import json

import pytest

from app.ml.taxonomy_db import (
    add_rollup_taxonomy_entry,
    populate_taxonomy_from_csv,
    populate_taxonomy_from_json,
)
from app.models.species_taxonomy import SpeciesTaxonomy

SAMPLE_TAXONOMY_ROWS = [
    {
        "model_class": "leopard", "class": "mammalia",
        "order": "carnivora", "family": "felidae",
        "genus": "panthera", "species": "pardus",
    },
    {
        "model_class": "lion", "class": "mammalia",
        "order": "carnivora", "family": "felidae",
        "genus": "panthera", "species": "leo",
    },
    {
        "model_class": "buffalo", "class": "mammalia",
        "order": "artiodactyla", "family": "bovidae",
        "genus": "syncerus", "species": "caffer",
    },
    {
        "model_class": "bird", "class": "aves",
        "order": "", "family": "", "genus": "", "species": "",
    },
]

MODEL_ID = "EUR-DF-v1-3"


@pytest.fixture
def taxonomy_csv(tmp_path):
    """Write sample taxonomy CSV and return its path."""
    csv_path = tmp_path / "taxonomy.csv"
    fieldnames = ["model_class", "class", "order", "family", "genus", "species"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(SAMPLE_TAXONOMY_ROWS)
    return csv_path


@pytest.fixture
def taxonomy_lookup():
    """Simple lookup dict matching the CSV data."""
    return {
        "leopard": {
            "class": "mammalia", "order": "carnivora",
            "family": "felidae", "genus": "panthera",
            "species": "pardus",
        },
        "lion": {
            "class": "mammalia", "order": "carnivora",
            "family": "felidae", "genus": "panthera",
            "species": "leo",
        },
        "buffalo": {
            "class": "mammalia", "order": "artiodactyla",
            "family": "bovidae", "genus": "syncerus",
            "species": "caffer",
        },
        "bird": {"class": "aves"},
    }


def test_populate_from_csv(db, taxonomy_csv):
    """Inserts correct rows with levels and taxonomy columns."""
    count = populate_taxonomy_from_csv(MODEL_ID, taxonomy_csv, db)
    assert count == 4

    rows = db.query(SpeciesTaxonomy).filter(
        SpeciesTaxonomy.classification_model_id == MODEL_ID
    ).all()
    assert len(rows) == 4

    by_name = {r.name: r for r in rows}

    # Leopard: full taxonomy → level=species
    leopard = by_name["leopard"]
    assert leopard.level == "species"
    assert leopard.taxon_class == "mammalia"
    assert leopard.taxon_family == "felidae"
    assert leopard.taxon_species == "pardus"
    assert leopard.is_custom is False

    # Bird: only class → level=class
    bird = by_name["bird"]
    assert bird.level == "class"
    assert bird.taxon_class == "aves"
    assert bird.taxon_order is None


def test_populate_idempotent(db, taxonomy_csv):
    """Calling twice doesn't duplicate rows."""
    count1 = populate_taxonomy_from_csv(MODEL_ID, taxonomy_csv, db)
    count2 = populate_taxonomy_from_csv(MODEL_ID, taxonomy_csv, db)
    assert count1 == 4
    assert count2 == 0

    total = db.query(SpeciesTaxonomy).filter(
        SpeciesTaxonomy.classification_model_id == MODEL_ID
    ).count()
    assert total == 4


def test_populate_missing_csv(db, tmp_path):
    """Returns 0 for non-existent CSV."""
    count = populate_taxonomy_from_csv(MODEL_ID, tmp_path / "nope.csv", db)
    assert count == 0


def test_add_rollup_entry(db, taxonomy_lookup):
    """Creates rolled-up entry with correct ancestors."""
    result = add_rollup_taxonomy_entry(
        MODEL_ID, "felidae", "family", taxonomy_lookup, db
    )
    assert result is True

    row = db.query(SpeciesTaxonomy).filter(
        SpeciesTaxonomy.classification_model_id == MODEL_ID,
        SpeciesTaxonomy.name == "felidae",
    ).first()
    assert row is not None
    assert row.level == "family"
    assert row.taxon_class == "mammalia"
    assert row.taxon_order == "carnivora"
    assert row.taxon_family == "felidae"
    assert row.taxon_genus is None  # family-level rollup has no genus
    assert row.taxon_species is None
    assert row.is_custom is False


def test_add_rollup_idempotent(db, taxonomy_lookup):
    """Skip if entry already exists."""
    r1 = add_rollup_taxonomy_entry(MODEL_ID, "felidae", "family", taxonomy_lookup, db)
    r2 = add_rollup_taxonomy_entry(MODEL_ID, "felidae", "family", taxonomy_lookup, db)
    assert r1 is True
    assert r2 is False

    total = db.query(SpeciesTaxonomy).filter(
        SpeciesTaxonomy.name == "felidae"
    ).count()
    assert total == 1


def test_different_models_no_collision(db, taxonomy_csv):
    """Same species name in different models creates separate rows."""
    count1 = populate_taxonomy_from_csv("model-A", taxonomy_csv, db)
    count2 = populate_taxonomy_from_csv("model-B", taxonomy_csv, db)
    assert count1 == 4
    assert count2 == 4

    total = db.query(SpeciesTaxonomy).count()
    assert total == 8


# ---------- populate_taxonomy_from_json tests ----------

SAMPLE_JSON_DATA = {
    "classification_categories": {
        "0": "domestic cattle",
        "1": "bovidae",
        "2": "mammalia",
        "3": "blank",
    },
    "classification_category_descriptions": {
        "0": "uuid-0;mammalia;cetartiodactyla;bovidae;bos;taurus;domestic cattle",
        "1": "uuid-1;mammalia;cetartiodactyla;bovidae;;;bovidae",
        "2": "uuid-2;mammalia;;;;;mammalia",
        "3": "uuid-3;;;;;;blank",
    },
}


@pytest.fixture
def results_json(tmp_path):
    """Write sample SpeciesNet results JSON and return its path."""
    json_path = tmp_path / "results.json"
    with open(json_path, "w") as f:
        json.dump(SAMPLE_JSON_DATA, f)
    return json_path


def test_populate_from_json(db, results_json):
    """Inserts correct rows from SpeciesNet JSON, skipping blank."""
    count = populate_taxonomy_from_json(MODEL_ID, results_json, db)
    # 3 entries: domestic cattle, bovidae, mammalia — blank is skipped
    assert count == 3

    rows = db.query(SpeciesTaxonomy).filter(
        SpeciesTaxonomy.classification_model_id == MODEL_ID
    ).all()
    assert len(rows) == 3

    by_name = {r.name: r for r in rows}

    # Full species-level entry
    cattle = by_name["domestic cattle"]
    assert cattle.level == "species"
    assert cattle.taxon_class == "mammalia"
    assert cattle.taxon_order == "cetartiodactyla"
    assert cattle.taxon_family == "bovidae"
    assert cattle.taxon_genus == "bos"
    assert cattle.taxon_species == "taurus"
    assert cattle.is_custom is False

    # Family-level entry (genus and species empty)
    bovidae = by_name["bovidae"]
    assert bovidae.level == "family"
    assert bovidae.taxon_class == "mammalia"
    assert bovidae.taxon_order == "cetartiodactyla"
    assert bovidae.taxon_family == "bovidae"
    assert bovidae.taxon_genus is None
    assert bovidae.taxon_species is None

    # Class-level entry (only class populated)
    mammalia = by_name["mammalia"]
    assert mammalia.level == "class"
    assert mammalia.taxon_class == "mammalia"
    assert mammalia.taxon_order is None


def test_populate_from_json_idempotent(db, results_json):
    """Calling twice doesn't duplicate rows."""
    count1 = populate_taxonomy_from_json(MODEL_ID, results_json, db)
    count2 = populate_taxonomy_from_json(MODEL_ID, results_json, db)
    assert count1 == 3
    assert count2 == 0

    total = db.query(SpeciesTaxonomy).filter(
        SpeciesTaxonomy.classification_model_id == MODEL_ID
    ).count()
    assert total == 3


def test_populate_from_json_missing_file(db, tmp_path):
    """Returns 0 for non-existent JSON."""
    count = populate_taxonomy_from_json(MODEL_ID, tmp_path / "nope.json", db)
    assert count == 0


def test_populate_from_json_no_descriptions(db, tmp_path):
    """Returns 0 when JSON has no classification_category_descriptions."""
    json_path = tmp_path / "results.json"
    with open(json_path, "w") as f:
        json.dump({"images": []}, f)
    count = populate_taxonomy_from_json(MODEL_ID, json_path, db)
    assert count == 0
