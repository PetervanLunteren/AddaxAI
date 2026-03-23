"""Tests for app.ml.taxonomy_db."""

import csv

import pytest

from app.ml.taxonomy_db import (
    add_rollup_taxonomy_entry,
    populate_taxonomy_from_csv,
)
from app.models.label_taxonomy import LabelTaxonomy

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

    rows = db.query(LabelTaxonomy).filter(
        LabelTaxonomy.classification_model_id == MODEL_ID
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

    total = db.query(LabelTaxonomy).filter(
        LabelTaxonomy.classification_model_id == MODEL_ID
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

    row = db.query(LabelTaxonomy).filter(
        LabelTaxonomy.classification_model_id == MODEL_ID,
        LabelTaxonomy.name == "felidae",
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

    total = db.query(LabelTaxonomy).filter(
        LabelTaxonomy.name == "felidae"
    ).count()
    assert total == 1


def test_different_models_no_collision(db, taxonomy_csv):
    """Same label name in different models creates separate rows."""
    count1 = populate_taxonomy_from_csv("model-A", taxonomy_csv, db)
    count2 = populate_taxonomy_from_csv("model-B", taxonomy_csv, db)
    assert count1 == 4
    assert count2 == 4

    total = db.query(LabelTaxonomy).count()
    assert total == 8


