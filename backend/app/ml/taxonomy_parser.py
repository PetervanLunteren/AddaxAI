"""
Taxonomy CSV parser with advanced tree building.

Implements the same logic as streamlit-AddaxAI:
- Handles missing/partial taxonomy gracefully
- Merges redundant single-child chains
- Sorts leaves before parents (both alphabetically)
- Adds descendant counts to parent nodes
- Groups unknown taxonomy under "other"

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
    name: str  # Display name with formatting
    level: int  # 1-6 (class, order, family, genus, species, model_class)
    children: list["TaxonomyNode"]
    selected: bool  # Default selection state


def parse_taxonomy_csv(csv_path: Path) -> list[TaxonomyNode]:
    """
    Parse taxonomy.csv into hierarchical tree structure with advanced features.

    CSV format (6 columns):
    - model_class: Common name (user-facing, e.g., "leopard")
    - class: Taxonomic class (e.g., "mammalia")
    - order: Taxonomic order (e.g., "carnivora")
    - family: Taxonomic family (e.g., "felidae")
    - genus: Taxonomic genus (e.g., "panthera", may be empty)
    - species: Taxonomic species (e.g., "pardus", may be empty)

    Returns:
        List of root-level taxonomy nodes

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

    # Build tree using Streamlit logic
    root: dict = {}
    other_key = "__other__"
    other_label = "other"

    def ensure_other_group():
        """Create 'other' group for items with no taxonomy."""
        if other_key not in root:
            root[other_key] = {
                "_label": other_label,
                "_value": other_label,
                "_children": {},
                "_level": "other",
            }
        return root[other_key]["_children"]

    def format_prefix(level_name: str, taxon_name: str) -> str:
        """Format taxonomy level prefix (e.g., 'class Mammalia')."""
        display_name = taxon_name if level_name == "species" else taxon_name.title()
        return f"{level_name} {display_name}"

    # Process each row
    for row in rows:
        model_class = row.get("model_class", "").strip()
        if not model_class:
            continue

        class_name = row.get("class", "").strip()
        order_name = row.get("order", "").strip()
        family_name = row.get("family", "").strip()
        genus_name = row.get("genus", "").strip()
        species_name = row.get("species", "").strip()

        # No taxonomy at all -> group under "other"
        if not any([class_name, order_name, family_name, genus_name, species_name]):
            other_children = ensure_other_group()
            if model_class not in other_children:
                other_children[model_class] = {
                    "_label": f"**{model_class}** (_unknown taxonomy_)",
                    "_value": model_class,
                    "_children": {},
                    "_level": "other",
                }
            continue

        # No class but has other info -> place at root with unknown taxonomy
        if not class_name:
            taxonomic_value = species_name or model_class
            label = f"{taxonomic_value} (**{model_class}**, _unknown taxonomy_)"
            if model_class not in root:
                root[model_class] = {
                    "_label": label,
                    "_value": model_class,
                    "_children": {},
                    "_level": "unknown",
                }
            continue

        # Build path through hierarchy
        levels = [
            ("class", class_name),
            ("order", order_name),
            ("family", family_name),
            ("genus", genus_name),
            ("species", species_name),
        ]

        current_level = root
        path_components = []
        species_available = bool(species_name)

        for idx, (level_name, taxon_name) in enumerate(levels):
            if not taxon_name:
                continue

            # Check if this is an "unspecified branch" (all remaining levels have same value)
            remaining_names = [name for _, name in levels[idx:] if name]
            unspecified_branch = (
                len(set(remaining_names)) == 1 if remaining_names else False
            )

            label_with_prefix = format_prefix(level_name, taxon_name)

            # Handle unspecified branch
            if unspecified_branch and level_name != "species" and not species_available:
                path_components.append(f"{level_name}:{taxon_name}")
                node_value = "|".join(path_components)

                if node_value not in current_level:
                    current_level[node_value] = {
                        "_label": label_with_prefix,
                        "_value": node_value,
                        "_children": {},
                        "_level": level_name,
                    }

                current_level = current_level[node_value]["_children"]

                # Add model_class as leaf with "unspecified" marker
                if model_class not in current_level:
                    current_level[model_class] = {
                        "_label": f"**{model_class}** (_unspecified_)",
                        "_value": model_class,
                        "_children": {},
                        "_level": "unspecified",
                    }
                break

            # Check if this is the last level with data
            is_last_level = idx == len(levels) - 1 or not any(
                levels[j][1] for j in range(idx + 1, len(levels))
            )

            if is_last_level:
                # Leaf node
                if level_name == "species":
                    label = f"{label_with_prefix} (**{model_class}**)"
                else:
                    label = f"**{model_class}** (_unspecified_)"

                if model_class not in current_level:
                    current_level[model_class] = {
                        "_label": label,
                        "_value": model_class,
                        "_children": {},
                        "_level": level_name,
                    }
            else:
                # Parent node - continue building path
                path_components.append(f"{level_name}:{taxon_name}")
                node_value = "|".join(path_components)

                if node_value not in current_level:
                    current_level[node_value] = {
                        "_label": label_with_prefix,
                        "_value": node_value,
                        "_children": {},
                        "_level": level_name,
                    }

                current_level = current_level[node_value]["_children"]

    # Convert dict to list
    def dict_to_list(d: dict) -> list[dict]:
        result = []
        for node_val in d.values():
            children_list = (
                dict_to_list(node_val["_children"]) if node_val["_children"] else []
            )
            node = {"_label": node_val["_label"], "_value": node_val["_value"]}
            if children_list:
                node["_children"] = children_list
            result.append(node)
        return result

    # Merge single-child redundant nodes
    def merge_single_redundant_nodes(nodes: list[dict]) -> list[dict]:
        merged = []
        for node in nodes:
            if "_children" in node and len(node["_children"]) == 1:
                child = node["_children"][0]

                # Check if parent and child have same prefix
                parent_prefix = node["_label"].split(" ")[0]
                child_prefix = child["_label"].split(" ")[0]

                if parent_prefix == child_prefix:
                    # Merge: replace parent with child
                    node["_label"] = child["_label"]
                    node["_value"] = child["_value"]

                    # Adopt grandchildren if any
                    grandkids = child.get("_children", [])
                    if grandkids:
                        node["_children"] = merge_single_redundant_nodes(grandkids)
                    else:
                        node.pop("_children", None)  # Make it a leaf

            # Recurse into children if still has them
            if "_children" in node and node["_children"]:
                node["_children"] = merge_single_redundant_nodes(node["_children"])

            merged.append(node)
        return merged

    # Sort leaves first, then parents (both alphabetically)
    def sort_leaf_first(nodes: list[dict]) -> list[dict]:
        leaves = []
        parents = []

        for node in nodes:
            if "_children" in node and node["_children"]:
                # Recurse first
                node["_children"] = sort_leaf_first(node["_children"])
                parents.append(node)
            else:
                leaves.append(node)

        # Sort both groups alphabetically (case-insensitive)
        leaves.sort(key=lambda x: x["_label"].lower())
        parents.sort(key=lambda x: x["_label"].lower())

        return leaves + parents

    # Add descendant counts to parent node labels with teal color
    def annotate_counts(nodes: list[dict]) -> int:
        total = 0
        for node in nodes:
            if "_children" in node and node["_children"]:
                child_total = annotate_counts(node["_children"])
                # Format count with code styling and teal color (#086164)
                node["_label"] = f"{node['_label']} `({child_total})`"
                total += child_total
            else:
                total += 1
        return total

    # Convert to final TaxonomyNode format
    def to_taxonomy_nodes(nodes: list[dict], level: int) -> list[TaxonomyNode]:
        result = []
        for node in nodes:
            taxonomy_node: TaxonomyNode = {
                "id": node["_value"],
                "name": node["_label"],
                "level": level,
                "children": (
                    to_taxonomy_nodes(node["_children"], level + 1)
                    if "_children" in node
                    else []
                ),
                "selected": True,
            }
            result.append(taxonomy_node)
        return result

    # Apply all transformations
    raw_tree = dict_to_list(root)
    merged_tree = merge_single_redundant_nodes(raw_tree)
    sorted_tree = sort_leaf_first(merged_tree)
    annotate_counts(sorted_tree)

    return to_taxonomy_nodes(sorted_tree, 1)


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
