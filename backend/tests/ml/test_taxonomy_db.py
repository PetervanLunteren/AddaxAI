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
def leopard_entry():
    """Representative taxonomy entry, as the rollup hands it over."""
    return {
        "class": "mammalia", "order": "carnivora",
        "family": "felidae", "genus": "panthera",
        "species": "pardus",
    }


VARIANT_TAXONOMY_ROWS = [
    {
        "model_class": "red fox adult", "class": "mammalia",
        "order": "carnivora", "family": "canidae",
        "genus": "vulpes", "species": "vulpes", "variant": "adult",
    },
    {
        "model_class": "red fox juvenile", "class": "mammalia",
        "order": "carnivora", "family": "canidae",
        "genus": "vulpes", "species": "vulpes", "variant": "juvenile",
    },
    {
        "model_class": "wolf", "class": "mammalia",
        "order": "carnivora", "family": "canidae",
        "genus": "canis", "species": "lupus", "variant": "",
    },
]


@pytest.fixture
def variant_taxonomy_csv(tmp_path):
    """Taxonomy CSV carrying the optional variant column."""
    csv_path = tmp_path / "taxonomy.csv"
    fieldnames = [
        "model_class", "class", "order", "family",
        "genus", "species", "variant",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(VARIANT_TAXONOMY_ROWS)
    return csv_path


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


def test_populate_sets_both_names(db, taxonomy_csv):
    """Each row gets a common_name and a scientific_name."""
    populate_taxonomy_from_csv(MODEL_ID, taxonomy_csv, db)
    rows = {
        r.name: r
        for r in db.query(LabelTaxonomy)
        .filter(LabelTaxonomy.classification_model_id == MODEL_ID)
        .all()
    }
    # Species: common = cleaned label, scientific = abbreviated binomial.
    leopard = rows["leopard"]
    assert leopard.common_name == "Leopard"
    assert leopard.scientific_name == "P. pardus"


def test_rollup_genus_common_equals_scientific(db, leopard_entry):
    """A genus rollup has no common name, so both names are the Latin taxon."""
    add_rollup_taxonomy_entry(MODEL_ID, "panthera", "genus", leopard_entry, db)
    row = (
        db.query(LabelTaxonomy)
        .filter(
            LabelTaxonomy.classification_model_id == MODEL_ID,
            LabelTaxonomy.name == "panthera",
        )
        .one()
    )
    assert row.common_name == "Panthera"
    assert row.scientific_name == "Panthera"


def test_builtin_labels_set_both_names(db):
    """Builtin labels carry both names (identical, capitalised category)."""
    from app.ml.taxonomy_db import ensure_builtin_labels

    ensure_builtin_labels(db)
    animal = (
        db.query(LabelTaxonomy)
        .filter(LabelTaxonomy.name == "animal")
        .one()
    )
    assert animal.common_name == "Animal"
    assert animal.scientific_name == "Animal"


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


def test_add_rollup_entry(db, leopard_entry):
    """Creates rolled-up entry with correct ancestors."""
    result = add_rollup_taxonomy_entry(
        MODEL_ID, "felidae", "family", leopard_entry, db
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


def test_add_rollup_class_level_only_sets_class(db, leopard_entry):
    """A class-level rollup must NOT inherit lower ranks from the matched
    source row. The lookup pairs class=mammalia with concrete order/family/
    genus/species; copying those would falsely claim the rolled-up taxon
    belongs to a specific descendant chain."""
    result = add_rollup_taxonomy_entry(
        MODEL_ID, "mammalia", "class", leopard_entry, db
    )
    assert result is True

    row = db.query(LabelTaxonomy).filter(
        LabelTaxonomy.classification_model_id == MODEL_ID,
        LabelTaxonomy.name == "mammalia",
    ).first()
    assert row is not None
    assert row.level == "class"
    assert row.taxon_class == "mammalia"
    assert row.taxon_order is None
    assert row.taxon_family is None
    assert row.taxon_genus is None
    assert row.taxon_species is None


def test_add_rollup_order_level_only_sets_class_and_order(db, leopard_entry):
    """An order-level rollup must inherit class and order, but not lower."""
    result = add_rollup_taxonomy_entry(
        MODEL_ID, "carnivora", "order", leopard_entry, db
    )
    assert result is True

    row = db.query(LabelTaxonomy).filter(
        LabelTaxonomy.classification_model_id == MODEL_ID,
        LabelTaxonomy.name == "carnivora",
    ).first()
    assert row is not None
    assert row.level == "order"
    assert row.taxon_class == "mammalia"
    assert row.taxon_order == "carnivora"
    assert row.taxon_family is None
    assert row.taxon_genus is None
    assert row.taxon_species is None


def test_add_rollup_genus_level_inherits_through_genus(db, leopard_entry):
    """A genus-level rollup keeps class/order/family/genus, not species."""
    result = add_rollup_taxonomy_entry(
        MODEL_ID, "panthera", "genus", leopard_entry, db
    )
    assert result is True

    row = db.query(LabelTaxonomy).filter(
        LabelTaxonomy.classification_model_id == MODEL_ID,
        LabelTaxonomy.name == "panthera",
    ).first()
    assert row is not None
    assert row.level == "genus"
    assert row.taxon_class == "mammalia"
    assert row.taxon_order == "carnivora"
    assert row.taxon_family == "felidae"
    assert row.taxon_genus == "panthera"
    assert row.taxon_species is None


def test_add_rollup_idempotent(db, leopard_entry):
    """Skip if entry already exists."""
    r1 = add_rollup_taxonomy_entry(MODEL_ID, "felidae", "family", leopard_entry, db)
    r2 = add_rollup_taxonomy_entry(MODEL_ID, "felidae", "family", leopard_entry, db)
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


# --- variant column ---


def test_populate_variant_rows(db, variant_taxonomy_csv):
    """A non-empty variant cell fills the column, the level and both names."""
    count = populate_taxonomy_from_csv(MODEL_ID, variant_taxonomy_csv, db)
    assert count == 3

    by_name = {
        r.name: r
        for r in db.query(LabelTaxonomy)
        .filter(LabelTaxonomy.classification_model_id == MODEL_ID)
        .all()
    }

    adult = by_name["red fox adult"]
    assert adult.level == "variant"
    assert adult.taxon_variant == "adult"
    assert adult.taxon_species == "vulpes"
    assert adult.common_name == "Red fox adult"
    # The variant makes the scientific name unique per class.
    assert adult.scientific_name == "V. vulpes (adult)"

    juvenile = by_name["red fox juvenile"]
    assert juvenile.scientific_name == "V. vulpes (juvenile)"
    assert juvenile.scientific_name != adult.scientific_name


def test_populate_variant_empty_cell_is_plain_species_row(
    db, variant_taxonomy_csv
):
    """An empty variant cell behaves exactly like a CSV without the column."""
    populate_taxonomy_from_csv(MODEL_ID, variant_taxonomy_csv, db)
    wolf = (
        db.query(LabelTaxonomy)
        .filter(
            LabelTaxonomy.classification_model_id == MODEL_ID,
            LabelTaxonomy.name == "wolf",
        )
        .one()
    )
    assert wolf.level == "species"
    assert wolf.taxon_variant is None
    assert wolf.scientific_name == "C. lupus"


def test_populate_variant_idempotent(db, variant_taxonomy_csv):
    """Re-population of a variant CSV inserts nothing the second time."""
    count1 = populate_taxonomy_from_csv(MODEL_ID, variant_taxonomy_csv, db)
    count2 = populate_taxonomy_from_csv(MODEL_ID, variant_taxonomy_csv, db)
    assert count1 == 3
    assert count2 == 0




def test_category_descriptions_read_taxonomy_csv_as_utf8(tmp_path):
    """A taxonomy.csv with an accented common name must read the same on
    every OS. Windows opens text as cp1252 unless told otherwise, so the
    description carried mojibake into label_taxonomy. Reproduced by
    forcing an ASCII locale in a child interpreter, like the manifest test."""
    import os
    import subprocess
    import sys

    csv_path = tmp_path / "taxonomy.csv"
    csv_path.write_bytes(
        "model_class,class,order,family,genus,species\n"
        "señuelo,mammalia,carnivora,canidae,vulpes,vulpes\n".encode()
    )
    code = (
        "import sys; from pathlib import Path; "
        "from app.ml.json_utils import build_classification_category_descriptions as b; "
        # The name is spelled as an escape so argv stays ASCII: under the C
        # locale Linux decodes the command line as ASCII and refuses a raw ñ
        # before running anything. The child still builds the string señuelo.
        f"d = b({{'0': 'se\\xf1uelo'}}, Path({str(csv_path)!r})); "
        "sys.stdout.buffer.write(d['0'].encode('utf-8'))"
    )
    env = {**os.environ, "LC_ALL": "C", "LANG": "C", "PYTHONUTF8": "0",
           "PYTHONCOERCECLOCALE": "0"}
    result = subprocess.run(
        [sys.executable, "-X", "utf8=0", "-c", code], env=env, capture_output=True,
        cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    description = result.stdout.decode("utf-8")
    assert description.startswith("señuelo;mammalia;carnivora;canidae;vulpes;vulpes")
