"""Species colours: related species get the most contrasting colours.

The rule under test (`crud/label_colors.py`): species present in the
project are sorted by taxonomy so siblings sit next to each other, then
walk a palette ordered farthest-first. So two rodents never share or
neighbour a colour, which is what a person scanning a block of rodent
crops needs to notice the odd one out.
"""

from app.api.crud.event import get_filter_options
from app.api.crud.label_colors import (
    CATEGORY_COLORS,
    REJECTED_LABEL_COLOR,
    SPECIES_PALETTE,
    assign_label_colors,
    fallback_color,
)
from app.ml.taxonomy_db import BUILTIN_MODEL_ID
from app.models.label_taxonomy import LabelTaxonomy
from tests.conftest import make_deployment, make_detection, make_file, make_project

MODEL_ID = "EUR-DF-v1-3"


def _taxon(db, name, **ranks) -> LabelTaxonomy:
    row = LabelTaxonomy(
        classification_model_id=MODEL_ID,
        name=name,
        level="species" if ranks.get("taxon_species") else "family",
        **ranks,
    )
    db.add(row)
    db.flush()
    return row


def _project_with(db, *labels, threshold=0.2):
    """A project holding one detection per (taxonomy row, confidence)."""
    project = make_project(
        db, classification_model_id=MODEL_ID, counting_threshold=threshold
    )
    deployment = make_deployment(db, project_id=project.id)
    for row, confidence, verified in labels:
        f = make_file(db, deployment_id=deployment.id)
        make_detection(
            db,
            file_id=f.id,
            label=row.name,
            label_taxonomy_id=row.id,
            confidence=confidence,
            verified=verified,
        )
    return project


def _rodents_and_others(db):
    rat = _taxon(
        db, "black rat", taxon_class="mammalia", taxon_order="rodentia",
        taxon_family="muridae", taxon_genus="rattus", taxon_species="rattus",
    )
    brown_rat = _taxon(
        db, "brown rat", taxon_class="mammalia", taxon_order="rodentia",
        taxon_family="muridae", taxon_genus="rattus", taxon_species="norvegicus",
    )
    mouse = _taxon(
        db, "house mouse", taxon_class="mammalia", taxon_order="rodentia",
        taxon_family="muridae", taxon_genus="mus", taxon_species="musculus",
    )
    fox = _taxon(
        db, "red fox", taxon_class="mammalia", taxon_order="carnivora",
        taxon_family="canidae", taxon_genus="vulpes", taxon_species="vulpes",
    )
    blackbird = _taxon(
        db, "blackbird", taxon_class="aves", taxon_order="passeriformes",
        taxon_family="turdidae", taxon_genus="turdus", taxon_species="merula",
    )
    return rat, brown_rat, mouse, fox, blackbird


def test_siblings_take_consecutive_palette_entries(db):
    """The two rats are taxonomic neighbours, so they get consecutive
    entries of the farthest-first palette: the most contrast available."""
    rat, brown_rat, mouse, fox, blackbird = _rodents_and_others(db)
    project = _project_with(
        db, *((row, 0.9, False) for row in (rat, brown_rat, mouse, fox, blackbird))
    )

    colors = assign_label_colors(db, project.id)

    idx = {name: SPECIES_PALETTE.index(colors[name]) for name in colors}
    # Sorted by taxonomy: aves first, then mammalia > carnivora > rodentia,
    # and inside muridae: mus before rattus, then the two rattus species.
    assert idx["blackbird"] == 0
    assert idx["red fox"] == 1
    assert idx["house mouse"] == 2
    assert idx["brown rat"] == 3  # norvegicus
    assert idx["black rat"] == 4  # rattus
    assert abs(idx["brown rat"] - idx["black rat"]) == 1


def test_no_two_present_species_share_a_colour_up_to_the_palette_size(db):
    rows = [
        _taxon(
            db, f"species {i}", taxon_class="mammalia", taxon_order="o",
            taxon_family="f", taxon_genus=f"g{i}", taxon_species=f"s{i}",
        )
        for i in range(len(SPECIES_PALETTE))
    ]
    project = _project_with(db, *((row, 0.9, False) for row in rows))

    colors = assign_label_colors(db, project.id)

    by_id = [colors[row.id] for row in rows]
    assert len(set(by_id)) == len(SPECIES_PALETTE)


def test_the_thirteenth_species_wraps_onto_the_first_colour(db):
    rows = [
        _taxon(
            db, f"species {i:02d}", taxon_class="mammalia", taxon_order="o",
            taxon_family="f", taxon_genus=f"g{i:02d}", taxon_species=f"s{i:02d}",
        )
        for i in range(len(SPECIES_PALETTE) + 1)
    ]
    project = _project_with(db, *((row, 0.9, False) for row in rows))

    colors = assign_label_colors(db, project.id)

    assert colors[rows[0].id] == colors[rows[-1].id] == SPECIES_PALETTE[0]


def test_map_is_keyed_by_taxonomy_id_and_lowercased_name(db):
    """The grid colours by label_taxonomy_id where it has one and by
    label name elsewhere; both must answer the same colour."""
    fox = _taxon(
        db, "Red Fox", taxon_class="mammalia", taxon_order="carnivora",
        taxon_family="canidae", taxon_genus="vulpes", taxon_species="vulpes",
    )
    project = _project_with(db, (fox, 0.9, False))

    colors = assign_label_colors(db, project.id)

    assert colors[fox.id] == colors["red fox"]
    assert "Red Fox" not in colors


def test_present_means_threshold_or_verified(db):
    """A species only seen below the counting threshold is not in the
    map, unless a person verified it. Same rule as the label filter."""
    rat, brown_rat, mouse, fox, blackbird = _rodents_and_others(db)
    project = _project_with(
        db,
        (rat, 0.9, False),
        (brown_rat, 0.05, False),  # below threshold, unverified
        (mouse, 0.05, True),  # below threshold, verified
        threshold=0.2,
    )

    colors = assign_label_colors(db, project.id)

    assert rat.id in colors
    assert brown_rat.id not in colors
    assert mouse.id in colors


def test_the_map_covers_exactly_the_labels_the_filter_offers(db):
    """One definition of "present": the label filter and the colour map
    read the same query, so a species can never be filterable without a
    colour or coloured without being filterable."""
    rat, brown_rat, mouse, fox, blackbird = _rodents_and_others(db)
    project = _project_with(
        db, (rat, 0.9, False), (fox, 0.3, False), (blackbird, 0.01, False)
    )

    colors = assign_label_colors(db, project.id)
    offered = get_filter_options(db, project.id)["labels"]

    assert {k for k in colors if k in {rat.id, fox.id, blackbird.id}} == set(offered)


def test_an_empty_project_has_an_empty_map(db):
    project = make_project(db, classification_model_id=MODEL_ID)
    assert assign_label_colors(db, project.id) == {}


def test_fallback_is_deterministic_and_from_the_palette():
    assert fallback_color("Aardvark") == fallback_color(" aardvark ")
    assert fallback_color("aardvark") in SPECIES_PALETTE


def test_palette_avoids_the_category_colours():
    """A species must never look like an unlabelled animal, person or
    vehicle box (#0f6064, #ff8945, #71b7ba)."""
    assert {"#0f6064", "#ff8945", "#71b7ba"}.isdisjoint(SPECIES_PALETTE)
    assert len(set(SPECIES_PALETTE)) == len(SPECIES_PALETTE) == 12


def test_endpoint_returns_the_map(client, db):
    rat, brown_rat, mouse, fox, blackbird = _rodents_and_others(db)
    project = _project_with(db, (rat, 0.9, False), (fox, 0.9, False))

    resp = client.get(f"/api/projects/{project.id}/label-colors")

    assert resp.status_code == 200
    assert resp.json() == assign_label_colors(db, project.id)


def test_endpoint_404_for_unknown_project(client):
    resp = client.get("/api/projects/nope/label-colors")
    assert resp.status_code == 404


def test_builtin_category_rows_keep_the_category_colour_and_take_no_slot(db):
    """An unclassified person or vehicle box carries a __builtin__
    taxonomy row. It is not a species: it keeps the category colour the
    export uses, and the first palette entry still goes to a real species."""
    person = LabelTaxonomy(
        classification_model_id=BUILTIN_MODEL_ID, name="person", level="none"
    )
    db.add(person)
    db.flush()
    rat, brown_rat, mouse, fox, blackbird = _rodents_and_others(db)
    project = _project_with(db, (person, 0.9, False), (blackbird, 0.9, False))

    colors = assign_label_colors(db, project.id)

    assert colors[person.id] == colors["person"] == CATEGORY_COLORS["person"]
    assert colors[blackbird.id] == SPECIES_PALETTE[0]


def test_rejected_labels_keep_a_neutral_colour_and_take_no_slot(db):
    """A box a person rejected above the threshold is still "present" (the
    grid keeps showing the verdict), but it is not a species. It used to
    take the first palette slot and shift every real species by one colour
    the moment someone pressed X or relabelled to a model's "non-animal"."""
    rejected = LabelTaxonomy(
        classification_model_id=MODEL_ID, name="non-animal", level="unknown"
    )
    db.add(rejected)
    db.flush()
    rat, brown_rat, mouse, fox, blackbird = _rodents_and_others(db)
    project = _project_with(
        db, (rejected, 0.9, True), (blackbird, 0.9, False), (fox, 0.9, False)
    )

    colors = assign_label_colors(db, project.id)

    assert colors[rejected.id] == colors["non-animal"] == REJECTED_LABEL_COLOR
    assert REJECTED_LABEL_COLOR not in SPECIES_PALETTE
    # The two real species still start at the top of the palette, in
    # taxonomic order: aves before mammalia.
    assert colors[blackbird.id] == SPECIES_PALETTE[0]
    assert colors[fox.id] == SPECIES_PALETTE[1]
