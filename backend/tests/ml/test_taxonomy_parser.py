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


def test_unspecified_leaf_has_annotation(tmp_path):
    csv_path = _write_csv(tmp_path, [
        "eagle,aves,accipitriformes,accipitridae,,",
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

    leaf = find_leaf(tree, "eagle")
    assert leaf is not None
    assert leaf["annotation"] == "unspecified"
    # Name should be clean (no markup)
    assert "_" not in leaf["name"]


def test_leaves_sorted(tmp_path):
    csv_path = _write_csv(tmp_path, [
        "zebra,mammalia,perissodactyla,equidae,equus,quagga",
        "antelope,mammalia,artiodactyla,bovidae,antilope,cervicapra",
    ])
    tree = parse_taxonomy_csv(csv_path)
    leaves = get_all_leaf_classes(tree)
    assert leaves == sorted(leaves)
