"""
Taxonomy CSV parser.

Parses taxonomy.csv files from HuggingFace into hierarchical tree structure.

Following DEVELOPERS.md principles:
- Type hints everywhere
- Explicit error handling
- Crash early if data is invalid
"""

import csv
from pathlib import Path
from typing import TypedDict


class TaxonomyNode(TypedDict):
    """Node in taxonomy tree."""

    id: str  # e.g., "mammalia", "carnivora", "felidae", "leopard"
    name: str  # Display name (titlecased)
    level: int  # 1-6 (class, order, family, genus, species, model_class)
    children: list["TaxonomyNode"]
    selected: bool  # Default selection state


def parse_taxonomy_csv(csv_path: Path) -> list[TaxonomyNode]:
    """
    Parse taxonomy.csv into hierarchical tree structure.

    CSV format (6 columns):
    - model_class: Common name (user-facing, e.g., "leopard")
    - class: Taxonomic class (e.g., "mammalia")
    - order: Taxonomic order (e.g., "carnivora")
    - family: Taxonomic family (e.g., "felidae")
    - genus: Taxonomic genus (e.g., "panthera", may be empty)
    - species: Taxonomic species (e.g., "pardus", may be empty)

    Returns:
        List of root-level taxonomy nodes (classes)

    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If CSV format is invalid
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Taxonomy CSV not found: {csv_path}")

    # Read CSV
    rows = []
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception as e:
        raise ValueError(f"Failed to read taxonomy CSV: {e}") from e

    if not rows:
        raise ValueError("Taxonomy CSV is empty")

    # Build hierarchical tree as nested dict
    tree: dict = {}

    for row in rows:
        model_class = row.get("model_class", "").strip()
        class_ = row.get("class", "").strip()
        order = row.get("order", "").strip()
        family = row.get("family", "").strip()
        genus = row.get("genus", "").strip()
        species = row.get("species", "").strip()

        if not model_class:
            continue  # Skip rows without model_class

        # Build path through tree
        current = tree

        # Level 1: Class
        if class_:
            if class_ not in current:
                current[class_] = {"_children": {}, "_leaf": False}
            current = current[class_]["_children"]

        # Level 2: Order
        if order:
            if order not in current:
                current[order] = {"_children": {}, "_leaf": False}
            current = current[order]["_children"]

        # Level 3: Family
        if family:
            if family not in current:
                current[family] = {"_children": {}, "_leaf": False}
            current = current[family]["_children"]

        # Level 4: Genus (optional)
        if genus:
            if genus not in current:
                current[genus] = {"_children": {}, "_leaf": False}
            current = current[genus]["_children"]

        # Level 5: Species (optional)
        if species:
            if species not in current:
                current[species] = {"_children": {}, "_leaf": False}
            current = current[species]["_children"]

        # Level 6: Model class (leaf node)
        current[model_class] = {
            "_children": {},
            "_leaf": True,
            "_common_name": model_class,
        }

    # Convert nested dict to TaxonomyNode list
    def dict_to_nodes(d: dict, level: int) -> list[TaxonomyNode]:
        nodes = []
        for key, value in d.items():
            if key.startswith("_"):  # Skip metadata
                continue

            node: TaxonomyNode = {
                "id": key,
                "name": value.get("_common_name", key).title(),
                "level": level,
                "children": dict_to_nodes(value["_children"], level + 1),
                "selected": True,  # Default: all selected
            }
            nodes.append(node)

        return sorted(nodes, key=lambda n: n["name"])

    return dict_to_nodes(tree, 1)


def get_all_leaf_classes(tree: list[TaxonomyNode]) -> list[str]:
    """
    Extract all leaf node (model_class) IDs from tree.

    Returns list of all selectable class names (e.g., ["leopard", "elephant", ...]).
    """
    leaves = []

    def collect_leaves(nodes: list[TaxonomyNode]):
        for node in nodes:
            if not node["children"]:  # Leaf node
                leaves.append(node["id"])
            else:
                collect_leaves(node["children"])

    collect_leaves(tree)
    return leaves
