#!/usr/bin/env python3
"""
Generate taxonomy.csv from a SpeciesNet labels file.

Converts the 7-token labels file (UUID;class;order;family;genus;species;common_name)
into AddaxAI's taxonomy.csv format (model_class,class,order,family,genus,species).

Usage:
    cd backend
    source venv/bin/activate
    python scripts/generate_taxonomy_csv.py /path/to/model_dir

The script finds the *.labels*.txt file in the model directory,
parses it, and writes taxonomy.csv in the same directory.
"""

import csv
import sys
from pathlib import Path


def generate_taxonomy_csv(model_dir: Path) -> Path:
    """Generate taxonomy.csv from the labels file in model_dir."""
    # Find labels file (supports both *.labels.txt and *.labels.DATE.txt)
    matches = sorted(model_dir.glob("*.labels*.txt"))
    if not matches:
        print(f"Error: no labels file found in {model_dir}", file=sys.stderr)
        sys.exit(1)

    labels_path = matches[-1]
    output_path = model_dir / "taxonomy.csv"

    rows = []
    seen_names: set[str] = set()

    with open(labels_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(";")
            if len(parts) < 7:
                continue

            common_name = parts[6]

            # Handle empty or duplicate names (same logic as inference.py)
            if not common_name or common_name in seen_names:
                taxonomy = [p for p in parts[1:6] if p]
                if taxonomy:
                    common_name = taxonomy[-1]

            if common_name in seen_names:
                common_name = f"{common_name} ({parts[0][:8]})"

            seen_names.add(common_name)

            rows.append({
                "model_class": common_name,
                "class": parts[1],
                "order": parts[2],
                "family": parts[3],
                "genus": parts[4],
                "species": parts[5],
            })

    fieldnames = ["model_class", "class", "order", "family", "genus", "species"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Labels file: {labels_path}")
    print(f"Output: {output_path}")
    print(f"Rows: {len(rows)}")
    return output_path


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <model_dir>", file=sys.stderr)
        sys.exit(1)

    model_dir = Path(sys.argv[1])
    if not model_dir.is_dir():
        print(f"Error: {model_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    generate_taxonomy_csv(model_dir)
