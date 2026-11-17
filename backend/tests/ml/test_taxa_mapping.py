"""Reading and checking a custom SpeciesNet taxa list (AddaxAI#136)."""

import hashlib
import json

import pytest

from app.ml.postprocessing import compute_postprocessing_settings_hash
from app.ml.taxa_mapping import (
    INPUT_LATIN,
    USER_TAXON_NAME,
    TaxaListError,
    load_speciesnet_taxa,
    mapped_labels,
    parse_taxa_list,
    shared_taxa,
    write_md_taxa_list,
)

# The shape of SpeciesNet's taxonomy_release: GUID;class;order;family;
# genus;species;common name. `strigiformes` has no line of its own here,
# only a child, the way some parents are missing in the real file.
TAXONOMY = """\
g1;;;;;;animal
g2;;;;;;blank
g3;mammalia;artiodactyla;bovidae;aepyceros;melampus;impala
g4;mammalia;artiodactyla;bovidae;aepyceros;;aepyceros species
g5;mammalia;artiodactyla;bovidae;;;bovid family
g6;aves;accipitriformes;accipitridae;;;hawk family
g7;aves;strigiformes;strigidae;bubo;bubo;eurasian eagle-owl
g8;mammalia;carnivora;canidae;canis;lupus familiaris;domestic dog
"""


@pytest.fixture
def taxa(tmp_path):
    path = tmp_path / "taxonomy_release.1.txt"
    path.write_text(TAXONOMY, encoding="utf-8")
    return load_speciesnet_taxa(str(path))


def _csv(*lines: str) -> str:
    return "\n".join(lines) + "\n"


def test_taxa_are_keyed_the_way_megadetector_keys_them(taxa):
    assert taxa["aepyceros melampus"]["species"] == "melampus"
    assert taxa["aepyceros"] == {
        "class": "mammalia",
        "order": "artiodactyla",
        "family": "bovidae",
        "genus": "aepyceros",
    }
    # A trinomial species keeps all three tokens.
    assert "canis lupus familiaris" in taxa
    # A parent with no line of its own is still a taxon.
    assert taxa["strigiformes"] == {"class": "aves", "order": "strigiformes"}
    assert "animal" not in taxa and "blank" not in taxa


def test_a_good_file_loads_lowercased_with_the_note_columns_dropped(taxa):
    text = _csv(
        "﻿input_latin,user_taxon_name,input_common,user_alt_name",
        "Aepyceros, Impala ,impala species,aepyceros melampus",
        "",
        "accipitridae,bird of prey,,",
        "strigiformes,Bird of prey,,",
    )
    rows = parse_taxa_list(text, taxa)
    assert rows == [
        {INPUT_LATIN: "aepyceros", USER_TAXON_NAME: "impala"},
        {INPUT_LATIN: "accipitridae", USER_TAXON_NAME: "bird of prey"},
        {INPUT_LATIN: "strigiformes", USER_TAXON_NAME: "bird of prey"},
    ]
    assert mapped_labels(rows) == ["impala", "bird of prey"]


def test_a_file_without_the_columns_is_refused(taxa):
    # The pre-rename names, which the pinned megadetector reads.
    with pytest.raises(TaxaListError) as e:
        parse_taxa_list(_csv("latin,common", "aepyceros,impala"), taxa)
    assert e.value.problems == [
        f"Missing column: {INPUT_LATIN}",
        f"Missing column: {USER_TAXON_NAME}",
    ]


def test_every_bad_row_is_named_and_the_file_refused(taxa):
    text = _csv(
        "input_latin,user_taxon_name",
        "aepyceros,impala",
        "panthera leo,lion",
        "aepyceros,antelope",
        "bovidae,",
    )
    with pytest.raises(TaxaListError) as e:
        parse_taxa_list(text, taxa)
    assert e.value.problems == [
        "Row 3 (panthera leo): not a SpeciesNet taxon",
        "Row 4 (aepyceros): this taxon is listed twice",
        "Row 5 (bovidae): input_latin and user_taxon_name are both required",
    ]


def test_a_label_longer_than_a_custom_label_is_refused(taxa):
    with pytest.raises(TaxaListError) as e:
        parse_taxa_list(_csv("input_latin,user_taxon_name", "aepyceros," + "a" * 201), taxa)
    assert e.value.problems == [
        "Row 2 (aepyceros): user_taxon_name is longer than 200 characters"
    ]


def test_an_empty_file_is_refused(taxa):
    with pytest.raises(TaxaListError):
        parse_taxa_list(_csv("input_latin,user_taxon_name"), taxa)


def test_each_label_gets_the_lowest_taxon_its_rows_share(taxa):
    rows = [
        {INPUT_LATIN: "aepyceros", USER_TAXON_NAME: "impala"},
        {INPUT_LATIN: "accipitridae", USER_TAXON_NAME: "bird of prey"},
        {INPUT_LATIN: "strigiformes", USER_TAXON_NAME: "bird of prey"},
        {INPUT_LATIN: "bovidae", USER_TAXON_NAME: "other"},
        {INPUT_LATIN: "strigiformes", USER_TAXON_NAME: "other"},
    ]
    shared = shared_taxa(rows, taxa)
    assert shared["impala"].level == "genus"
    assert shared["impala"].ranks["genus"] == "aepyceros"
    assert shared["bird of prey"].level == "class"
    assert shared["bird of prey"].ranks == {"class": "aves"}
    # A mammal and a bird share no class.
    assert shared["other"].level == "kingdom"
    assert shared["other"].ranks == {}


def test_megadetector_gets_the_column_names_it_reads(tmp_path):
    path = tmp_path / "list.csv"
    write_md_taxa_list(
        [{INPUT_LATIN: "aepyceros", USER_TAXON_NAME: "impala"}], path
    )
    assert path.read_text(encoding="utf-8").splitlines() == [
        "latin,common",
        "aepyceros,impala",
    ]


def _project(**overrides):
    from types import SimpleNamespace

    fields = dict(
        event_smoothing=True,
        smoothing_strength="normal",
        taxonomic_rollup=True,
        independence_interval=1800,
        excluded_classes=[],
        country_code=None,
        state_code=None,
        taxa_mapping=None,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_a_project_without_a_list_keeps_its_settings_hash():
    """Upgrading must not flag every existing project for reprocessing."""
    before = hashlib.sha256(
        json.dumps(
            {
                "event_smoothing": True,
                "smoothing_strength": "normal",
                "taxonomic_rollup": True,
                "independence_interval": 1800,
                "excluded_classes": [],
                "country_code": None,
                "state_code": None,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    assert compute_postprocessing_settings_hash(_project()) == before


def test_a_new_list_changes_the_settings_hash():
    rows = [{INPUT_LATIN: "aepyceros", USER_TAXON_NAME: "impala"}]
    other = [{INPUT_LATIN: "aepyceros", USER_TAXON_NAME: "antelope"}]
    with_list = compute_postprocessing_settings_hash(
        _project(taxa_mapping={"file_name": "a.csv", "rows": rows})
    )
    assert with_list != compute_postprocessing_settings_hash(_project())
    assert with_list != compute_postprocessing_settings_hash(
        _project(taxa_mapping={"file_name": "a.csv", "rows": other})
    )
    # The file name is not a setting.
    assert with_list == compute_postprocessing_settings_hash(
        _project(taxa_mapping={"file_name": "b.csv", "rows": rows})
    )


def test_labels_outside_the_list_get_a_row(db):
    """What megadetector 10.0.28 wrote for a list with impala and birds of
    prey: a jabiru stops at aves ("bird"), which SpeciesNet's label file
    has no class for, so without a row it would miss the label filter."""
    from types import SimpleNamespace

    from app.ml.postprocessing import add_taxa_list_ancestor_rows
    from app.models.label_taxonomy import LabelTaxonomy

    md = {
        "classification_categories": {
            "0": "impala", "2": "human", "3": "blank",
            "5": "bird of prey", "6": "bird", "8": "animal",
        },
        "classification_category_descriptions": {
            "0": "guid;mammalia;artiodactyla;bovidae;aepyceros;;aepyceros species",
            "2": "human;mammalia;primates;hominidae;homo;sapiens;human",
            "3": "blank;;;;;;blank",
            "5": "guid;aves;accipitriformes;accipitridae;;;accipitridae family"
            "|guid;aves;strigiformes;;;;owl",
            "6": "guid;aves;;;;;bird",
            "8": "animal;;;;;;animal",
        },
    }
    project = SimpleNamespace(
        classification_model_id="SN",
        taxa_mapping={"rows": [
            {INPUT_LATIN: "aepyceros", USER_TAXON_NAME: "impala"},
            {INPUT_LATIN: "accipitridae", USER_TAXON_NAME: "bird of prey"},
        ]},
    )
    add_taxa_list_ancestor_rows(md, project, db)

    rows = db.query(LabelTaxonomy).filter(LabelTaxonomy.classification_model_id == "SN").all()
    assert [(r.name, r.level, r.taxon_class, r.project_id) for r in rows] == [
        ("bird", "class", "aves", None)
    ]
