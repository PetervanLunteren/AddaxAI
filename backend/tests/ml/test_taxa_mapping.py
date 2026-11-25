"""Reading and checking a custom SpeciesNet taxa list (AddaxAI#136)."""

import hashlib
import json

import pytest

from app.ml.postprocessing import compute_postprocessing_settings_hash
from app.ml.taxa_mapping import (
    LABEL,
    LATIN,
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
    # Starts with a byte order mark, as Excel's "CSV UTF-8" writes it.
    text = _csv(
        "\ufeffinput_latin,output_name,input_common,output_alt_name",
        "Aepyceros, Impala ,impala species,aepyceros melampus",
        "",
        "accipitridae,bird of prey,,",
        "strigiformes,Bird of prey,,",
    )
    rows = parse_taxa_list(text, taxa)
    assert rows == [
        {LATIN: "aepyceros", LABEL: "impala"},
        {LATIN: "accipitridae", LABEL: "bird of prey"},
        {LATIN: "strigiformes", LABEL: "bird of prey"},
    ]
    assert mapped_labels(rows) == ["impala", "bird of prey"]


def test_the_older_columns_load_as_megadetector_reads_them(taxa):
    # The label is original_common when it is filled, common otherwise.
    text = _csv(
        "latin,common,original_latin,original_common,notes",
        "aepyceros,impala species,,Impala,",
        "accipitridae,bird of prey,,,",
    )
    assert parse_taxa_list(text, taxa) == [
        {LATIN: "aepyceros", LABEL: "impala"},
        {LATIN: "accipitridae", LABEL: "bird of prey"},
    ]


def test_dans_example_file_loads_as_megadetector_reads_it():
    """ppp-botha_speciesnet.csv from Dan Morris's taxonomy mapping skill
    (agentmorris/agentmorrispublic, skills/speciesnet-taxonomy-mapping,
    commit 23d4015e), unchanged. Here so the rules cannot drift from
    MegaDetector's without a failing test. Its taxa are taken as known:
    the check against SpeciesNet has tests of its own above."""
    from pathlib import Path

    text = (Path(__file__).parent / "fixtures" / "ppp-botha_speciesnet.csv").read_text(
        encoding="utf-8"
    )
    known = {line.split(",")[0]: {} for line in text.splitlines()[1:]}
    rows = parse_taxa_list(text, known)

    # Human, Vehicle, Blank and the species SpeciesNet lacks have no
    # Latin name: skipped, not refused.
    assert len(rows) == 75
    assert rows[0] == {LATIN: "chiroptera", LABEL: "bat_unspecified"}
    # Two species onto one label, by original_common.
    assert {LATIN: "equus quagga", LABEL: "zebra"} in rows
    assert {LATIN: "equus zebra", LABEL: "zebra"} in rows


def test_a_file_without_the_columns_is_refused_quoting_its_first_line(taxa):
    with pytest.raises(TaxaListError) as e:
        parse_taxa_list(_csv("taxon,name", "aepyceros,impala"), taxa)
    assert e.value.problems == [
        "Missing column: input_latin (or latin in the older format). "
        "The first line reads: taxon,name"
    ]
    with pytest.raises(TaxaListError) as e:
        parse_taxa_list(_csv("input_latin,input_common", "aepyceros,impala"), taxa)
    assert e.value.problems == [
        "Missing column: output_name. The first line reads: input_latin,input_common"
    ]
    with pytest.raises(TaxaListError) as e:
        parse_taxa_list(_csv("latin,original_common", "aepyceros,impala"), taxa)
    assert e.value.problems == [
        "Missing column: common. The first line reads: latin,original_common"
    ]
    # Tab separated, upper case: the quoted line shows what is wrong.
    with pytest.raises(TaxaListError) as e:
        parse_taxa_list(_csv("Input_Latin\tOutput_Name", "aepyceros\timpala"), taxa)
    assert e.value.problems[0].endswith("The first line reads: Input_Latin\tOutput_Name")
    with pytest.raises(TaxaListError) as e:
        parse_taxa_list("", taxa)
    assert e.value.problems[0].endswith("The first line reads: (empty)")


def test_a_semicolon_file_from_a_european_excel_loads(taxa):
    """Excel in most of Europe saves "CSV UTF-8" with ";". The header
    decides: a comma file keeps a ";" inside a label."""
    rows = [{LATIN: "aepyceros", LABEL: "impala"}, {LATIN: "bovidae", LABEL: "bovid, large"}]
    assert parse_taxa_list(
        _csv("input_latin;output_name", "aepyceros;impala", 'bovidae;"bovid, large"'), taxa
    ) == rows
    assert parse_taxa_list(
        _csv("input_latin,output_name", "aepyceros,impala; adult"), taxa
    ) == [{LATIN: "aepyceros", LABEL: "impala; adult"}]


def test_a_file_that_is_not_utf8_is_refused(taxa):
    """A browser reads Excel's plain "CSV" (cp1252) with every accented
    letter turned into U+FFFD. Loading it would store broken labels."""
    text = "input_latin,output_name\naepyceros,gazelle de Granté\n".encode("cp1252")
    with pytest.raises(TaxaListError) as e:
        parse_taxa_list(text.decode("utf-8", errors="replace"), taxa)
    assert e.value.problems == ['The file is not UTF-8. Save it from Excel as "CSV UTF-8".']


def test_every_bad_row_is_named_by_its_line_and_the_file_refused(taxa):
    text = _csv(
        "input_latin,output_name",
        "aepyceros,impala",
        ",human",
        "",
        "panthera leo,lion",
        "aepyceros,antelope",
        "bovidae,",
    )
    with pytest.raises(TaxaListError) as e:
        parse_taxa_list(text, taxa)
    # The row without a Latin name and the blank line are skipped, and
    # still counted in the line numbers.
    assert e.value.problems == [
        "Row 5 (panthera leo): not a SpeciesNet taxon",
        "Row 6 (aepyceros): this taxon is listed twice",
        "Row 7 (bovidae): needs both a Latin name and a label",
    ]


def test_a_label_longer_than_a_custom_label_is_refused(taxa):
    with pytest.raises(TaxaListError) as e:
        parse_taxa_list(_csv("input_latin,output_name", "aepyceros," + "a" * 201), taxa)
    assert e.value.problems == [
        "Row 2 (aepyceros): the label is longer than 200 characters"
    ]


def test_a_huge_bad_file_is_refused_with_a_bounded_list(taxa):
    rows = [f"{'x' * 1000}{i},label" for i in range(150)]
    with pytest.raises(TaxaListError) as e:
        parse_taxa_list(_csv("input_latin,output_name", *rows), taxa)
    assert len(e.value.problems) == 101
    assert e.value.problems[0] == f"Row 2 ({'x' * 80}): not a SpeciesNet taxon"
    assert e.value.problems[-1] == "and 50 more"


def test_a_file_with_no_latin_names_is_refused(taxa):
    with pytest.raises(TaxaListError) as e:
        parse_taxa_list(_csv("input_latin,output_name", ",human"), taxa)
    assert e.value.problems == ["The file has no rows with a Latin name"]


def test_each_label_gets_the_lowest_taxon_its_rows_share(taxa):
    rows = [
        {LATIN: "aepyceros", LABEL: "impala"},
        {LATIN: "accipitridae", LABEL: "bird of prey"},
        {LATIN: "strigiformes", LABEL: "bird of prey"},
        {LATIN: "bovidae", LABEL: "other"},
        {LATIN: "strigiformes", LABEL: "other"},
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
        [{LATIN: "aepyceros", LABEL: "impala"}], path
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
    rows = [{LATIN: "aepyceros", LABEL: "impala"}]
    other = [{LATIN: "aepyceros", LABEL: "antelope"}]
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
            {LATIN: "aepyceros", LABEL: "impala"},
            {LATIN: "accipitridae", LABEL: "bird of prey"},
        ]},
    )
    add_taxa_list_ancestor_rows(md, project, db)

    rows = db.query(LabelTaxonomy).filter(LabelTaxonomy.classification_model_id == "SN").all()
    assert [(r.name, r.level, r.taxon_class, r.project_id) for r in rows] == [
        ("bird", "class", "aves", None)
    ]


def test_a_run_gives_every_list_label_its_row(db, tmp_path, monkeypatch):
    """A project package carries only the rows something uses, so an
    imported project can hold a list whose labels have no row. The run
    writes them, or new boxes on those labels miss the filter."""
    from app.ml import postprocessing
    from app.models.label_taxonomy import LabelTaxonomy
    from tests.conftest import make_project

    (tmp_path / "taxonomy_release.1.txt").write_text(TAXONOMY, encoding="utf-8")
    rows = [
        {LATIN: "aepyceros", LABEL: "impala"},
        {LATIN: "accipitridae", LABEL: "bird of prey"},
        {LATIN: "strigiformes", LABEL: "bird of prey"},
    ]
    project = make_project(
        db, classification_model_id="SN", taxa_mapping={"file_name": "a.csv", "rows": rows}
    )
    monkeypatch.setattr(postprocessing, "_run_ml_script", lambda _s, md, _o, _j: md)

    postprocessing.apply_taxa_mapping({}, project, tmp_path, db, None)

    own = db.query(LabelTaxonomy).filter(LabelTaxonomy.project_id == project.id).all()
    assert sorted((r.name, r.level) for r in own) == [
        ("bird of prey", "class"),
        ("impala", "genus"),
    ]


def test_a_run_refuses_a_list_the_model_no_longer_knows(db, tmp_path, monkeypatch):
    """A model update can drop a taxon: the run fails naming the row."""
    from app.ml import postprocessing
    from tests.conftest import make_project

    (tmp_path / "taxonomy_release.1.txt").write_text(TAXONOMY, encoding="utf-8")
    rows = [{LATIN: "panthera leo", LABEL: "lion"}]
    project = make_project(
        db, classification_model_id="SN", taxa_mapping={"file_name": "a.csv", "rows": rows}
    )
    monkeypatch.setattr(postprocessing, "_run_ml_script", lambda _s, md, _o, _j: md)

    with pytest.raises(TaxaListError) as e:
        postprocessing.apply_taxa_mapping({}, project, tmp_path, db, None)
    assert e.value.problems == ["Row 2 (panthera leo): not a SpeciesNet taxon"]
