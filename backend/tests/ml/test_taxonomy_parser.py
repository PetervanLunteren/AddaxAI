"""Tests for app.ml.taxonomy_parser."""

import pytest

from app.ml.taxonomy_parser import get_all_leaf_classes, parse_taxonomy_csv


def _write_csv(tmp_path, rows):
    """Write taxonomy CSV with header + rows."""
    csv_path = tmp_path / "taxonomy.csv"
    header = "model_class,class,order,family,genus,species\n"
    csv_path.write_text(header + "\n".join(rows))
    return csv_path


def test_parse_single_species(tmp_path):
    csv_path = _write_csv(tmp_path, [
        "leopard,mammalia,carnivora,felidae,panthera,pardus",
    ])
    tree = parse_taxonomy_csv(csv_path)
    assert len(tree) >= 1
    leaves = get_all_leaf_classes(tree)
    assert "leopard" in leaves


def test_parse_multiple_species(tmp_path):
    csv_path = _write_csv(tmp_path, [
        "leopard,mammalia,carnivora,felidae,panthera,pardus",
        "elephant,mammalia,proboscidea,elephantidae,loxodonta,africana",
    ])
    tree = parse_taxonomy_csv(csv_path)
    leaves = get_all_leaf_classes(tree)
    assert "leopard" in leaves
    assert "elephant" in leaves


def test_parse_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        parse_taxonomy_csv(tmp_path / "missing.csv")


def test_get_all_leaf_classes(tmp_path):
    csv_path = _write_csv(tmp_path, [
        "leopard,mammalia,carnivora,felidae,panthera,pardus",
        "lion,mammalia,carnivora,felidae,panthera,leo",
    ])
    tree = parse_taxonomy_csv(csv_path)
    leaves = get_all_leaf_classes(tree)
    assert len(leaves) == 2
    assert set(leaves) == {"leopard", "lion"}


def test_get_leaf_classes_nested(tmp_path):
    csv_path = _write_csv(tmp_path, [
        "leopard,mammalia,carnivora,felidae,panthera,pardus",
        "elephant,mammalia,proboscidea,elephantidae,loxodonta,africana",
        "eagle,aves,accipitriformes,accipitridae,,",
    ])
    tree = parse_taxonomy_csv(csv_path)
    leaves = get_all_leaf_classes(tree)
    assert len(leaves) == 3


def test_leaf_count_annotation(tmp_path):
    csv_path = _write_csv(tmp_path, [
        "leopard,mammalia,carnivora,felidae,panthera,pardus",
        "lion,mammalia,carnivora,felidae,panthera,leo",
    ])
    tree = parse_taxonomy_csv(csv_path)
    # Parent nodes should have child_count field, not markup in name
    parent_nodes = [n for n in tree if n["children"]]
    if parent_nodes:
        assert "child_count" in parent_nodes[0]
        assert parent_nodes[0]["child_count"] == 2
        # Name should be clean (no markup)
        assert "categories" not in parent_nodes[0]["name"]


def test_species_leaf_has_annotation(tmp_path):
    csv_path = _write_csv(tmp_path, [
        "leopard,mammalia,carnivora,felidae,panthera,pardus",
    ])
    tree = parse_taxonomy_csv(csv_path)

    def find_leaf(nodes, target_id):
        for n in nodes:
            if n["id"] == target_id:
                return n
            found = find_leaf(n.get("children", []), target_id)
            if found:
                return found
        return None

    leaf = find_leaf(tree, "leopard")
    assert leaf is not None
    # Name should be clean scientific text, annotation should be display name
    assert leaf["name"] == "P. pardus"
    assert leaf["annotation"] == "leopard"


def _find_leaf(nodes, target_id):
    for n in nodes:
        if n["id"] == target_id:
            return n
        found = _find_leaf(n.get("children", []), target_id)
        if found:
            return found
    return None


@pytest.mark.parametrize(
    "row,model_class,name,annotation",
    [
        # Leaf is named for the taxon; the model's own label annotates it.
        ("leopard,mammalia,carnivora,felidae,panthera,pardus",
         "leopard", "P. pardus", "leopard"),
        ("eagle,aves,accipitriformes,accipitridae,,",
         "eagle", "Accipitridae", "eagle"),
        ("baboon,mammalia,primates,cercopithecidae,papio,",
         "baboon", "Papio", "baboon"),
        ("rodent,mammalia,rodentia,,,", "rodent", "Rodentia", "rodent"),
        # No second name to give, so the annotation names the rank instead.
        ("gorilla,mammalia,primates,hominidae,gorilla,",
         "gorilla", "Gorilla", "genus"),
        ("felidae,mammalia,carnivora,felidae,,",
         "felidae", "Felidae", "family"),
        ("aves,aves,,,,", "aves", "Aves", "class"),
    ],
)
def test_leaf_naming_rule(tmp_path, row, model_class, name, annotation):
    """One rule at every rank: the leaf is named for the taxon and annotated
    with the model's own label, or with the rank when the label *is* the
    taxon name. Shared with the label filter tree, see
    tests/api/test_label_tree.py.
    """
    tree = parse_taxonomy_csv(_write_csv(tmp_path, [row]))

    leaf = _find_leaf(tree, model_class)
    assert leaf is not None
    assert leaf["name"] == name
    assert leaf["annotation"] == annotation
    # Name should be clean (no markup)
    assert "_" not in leaf["name"]


def test_no_leaf_is_annotated_unspecified(tmp_path):
    """The literal "unspecified" is gone. It said nothing a user could act
    on and, for a class named after its own taxon, wrongly implied a rollup.
    """
    tree = parse_taxonomy_csv(_write_csv(tmp_path, [
        "leopard,mammalia,carnivora,felidae,panthera,pardus",
        "baboon,mammalia,primates,cercopithecidae,papio,",
        "felidae,mammalia,carnivora,felidae,,",
        "aves,aves,,,,",
    ]))

    seen = []

    def walk(nodes):
        for n in nodes:
            if n.get("annotation"):
                seen.append(n["annotation"])
            walk(n["children"])

    walk(tree)
    assert seen  # guard against the walk silently finding nothing
    assert "unspecified" not in seen


def test_rank_annotation_does_not_change_the_selectable_classes(tmp_path):
    """The annotation is display only. Every model class stays a leaf, which
    is what get_all_leaf_classes collects and what excluded_classes stores.
    """
    tree = parse_taxonomy_csv(_write_csv(tmp_path, [
        "leopard,mammalia,carnivora,felidae,panthera,pardus",
        "baboon,mammalia,primates,cercopithecidae,papio,",
        "rodent,mammalia,rodentia,,,",
    ]))

    assert sorted(get_all_leaf_classes(tree)) == ["baboon", "leopard", "rodent"]


def test_leaves_sorted(tmp_path):
    csv_path = _write_csv(tmp_path, [
        "zebra,mammalia,perissodactyla,equidae,equus,quagga",
        "antelope,mammalia,artiodactyla,bovidae,antilope,cervicapra",
    ])
    tree = parse_taxonomy_csv(csv_path)
    leaves = get_all_leaf_classes(tree)
    assert leaves == sorted(leaves)


# --- variant column ---


def _write_variant_csv(tmp_path, rows):
    """Write taxonomy CSV including the optional variant column."""
    csv_path = tmp_path / "taxonomy.csv"
    header = "model_class,class,order,family,genus,species,variant\n"
    csv_path.write_text(header + "\n".join(rows))
    return csv_path


def _find(nodes, name):
    for node in nodes:
        if node["name"] == name:
            return node
    raise AssertionError(
        f"no node named {name!r} in {[n['name'] for n in nodes]}"
    )


def test_variant_rows_nest_under_a_species_node(tmp_path):
    """Variants of one species share a species parent node showing the
    binomial; the leaves show only the variant with the model class as
    annotation. A plain species row keeps its leaf under the genus."""
    csv_path = _write_variant_csv(tmp_path, [
        "red fox adult,mammalia,carnivora,canidae,vulpes,vulpes,adult",
        "red fox juvenile,mammalia,carnivora,canidae,vulpes,vulpes,juvenile",
        "wolf,mammalia,carnivora,canidae,canis,lupus,",
    ])
    tree = parse_taxonomy_csv(csv_path)

    canidae = _find(
        _find(_find(tree, "Mammalia")["children"], "Carnivora")["children"],
        "Canidae",
    )
    # Plain species: unchanged leaf under genus.
    canis = _find(canidae["children"], "Canis")
    wolf_leaf = _find(canis["children"], "C. lupus")
    assert wolf_leaf["annotation"] == "wolf"
    assert not wolf_leaf["children"]

    # Variants: species parent with suffix-only leaves.
    vulpes = _find(canidae["children"], "Vulpes")
    species_node = _find(vulpes["children"], "V. vulpes")
    adult_leaf = _find(species_node["children"], "Adult")
    assert adult_leaf["id"] == "red fox adult"
    assert adult_leaf["annotation"] == "red fox adult"
    juvenile_leaf = _find(species_node["children"], "Juvenile")
    assert juvenile_leaf["id"] == "red fox juvenile"

    # Selectable classes are still exactly the model classes.
    assert set(get_all_leaf_classes(tree)) == {
        "red fox adult", "red fox juvenile", "wolf",
    }


def test_variant_column_absent_changes_nothing(tmp_path):
    """A CSV without the variant column parses exactly as before."""
    csv_path = _write_csv(tmp_path, [
        "leopard,mammalia,carnivora,felidae,panthera,pardus",
    ])
    tree = parse_taxonomy_csv(csv_path)
    felidae = _find(
        _find(_find(tree, "Mammalia")["children"], "Carnivora")["children"],
        "Felidae",
    )
    panthera = _find(felidae["children"], "Panthera")
    leaf = _find(panthera["children"], "P. pardus")
    assert not leaf["children"]
