#!/usr/bin/env python3
"""
Compare official SpeciesNet output with AddaxAI DB detections.

Compares the labels produced by the official SpeciesNet API
(run_md_and_speciesnet) with the labels stored in AddaxAI's database
after a full analysis run. Reports exact matches, label format
differences (Latin vs common name), and real label disagreements.

Usage:
    cd backend
    source venv/bin/activate
    python scripts/compare_speciesnet.py \
        --gt /path/to/SPPNET_ground_truth.json \
        --project-id <uuid>

Options:
    --gt          Path to the official SpeciesNet ground truth JSON
    --project-id  AddaxAI project UUID (default: most recently created)
    --verbose     Show all individual differences
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

# Add backend to path so we can import app modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings

# Taxonomy levels from broadest to most specific
_TAXONOMY_LEVELS = ["class", "order", "family", "genus", "species"]


def _build_gt_label_map(gt: dict) -> dict[str, str]:
    """
    Build a mapping from GT common name to AddaxAI label convention.

    For species-level entries (genus + species present), keeps the common
    name since both systems use it (e.g. "domestic cattle").

    For higher-level rollup labels (family, order, class), extracts the
    Latin taxon value to match AddaxAI's convention. For example,
    "bovidae family" becomes "bovidae", "mammal" becomes "mammalia".

    Uses classification_category_descriptions from the GT JSON
    (7-token format: UUID;class;order;family;genus;species;common_name).
    """
    descs = gt.get("classification_category_descriptions", {})
    cats = gt.get("classification_categories", {})
    name_map: dict[str, str] = {}

    for cat_id, common_name in cats.items():
        desc = descs.get(cat_id, "")
        parts = desc.split(";")
        if len(parts) < 7:
            continue

        # parts: [UUID, class, order, family, genus, species, common_name]
        taxon_class = parts[1]
        taxon_order = parts[2]
        taxon_family = parts[3]
        taxon_genus = parts[4]
        taxon_species = parts[5]

        if taxon_species:
            # Species-level: keep common name (both systems match)
            name_map[common_name] = common_name
        elif taxon_genus:
            name_map[common_name] = taxon_genus
        elif taxon_family:
            name_map[common_name] = taxon_family
        elif taxon_order:
            name_map[common_name] = taxon_order
        elif taxon_class:
            name_map[common_name] = taxon_class

    return name_map


def load_gt(gt_path: Path) -> dict[tuple[str, tuple], tuple[str, float]]:
    """
    Load ground truth JSON, convert labels to Latin names, return lookup.

    Uses classification_category_descriptions to extract Latin taxonomy
    names so they can be compared directly with AddaxAI's Latin labels.
    """
    with open(gt_path) as f:
        gt = json.load(f)

    gt_cats = gt["classification_categories"]
    latin_map = _build_gt_label_map(gt)
    lookup: dict[tuple[str, tuple], tuple[str, float]] = {}

    for img in gt["images"]:
        fname = Path(img["file"]).name
        for det in img.get("detections", []):
            bbox = tuple(round(x, 4) for x in det["bbox"])
            cls = det.get("classifications", [])
            if cls:
                top = sorted(cls, key=lambda x: -x[1])[0]
                common_name = gt_cats.get(str(top[0]), f"?{top[0]}")
                label = latin_map.get(common_name, common_name)
                lookup[(fname, bbox)] = (label, top[1])

    return lookup


def load_db_detections(
    project_id: str, db_path: Path
) -> dict[tuple[str, tuple], tuple[str, float]]:
    """Load AddaxAI DB detections and return lookup by (filename, bbox)."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    dets = conn.execute(
        """
        SELECT f.file_path, d.label, d.label_confidence,
               d.bbox_x, d.bbox_y, d.bbox_width, d.bbox_height,
               lt.level, lt.taxon_class, lt.taxon_order,
               lt.taxon_family, lt.taxon_genus, lt.taxon_species
        FROM detections d
        JOIN files f ON d.file_id = f.id
        JOIN deployments dep ON f.deployment_id = dep.id
        JOIN sites s ON dep.site_id = s.id
        LEFT JOIN label_taxonomy lt ON d.label_taxonomy_id = lt.id
        WHERE s.project_id = ?
        """,
        (project_id,),
    ).fetchall()

    conn.close()

    lookup: dict[tuple[str, tuple], tuple[str, float]] = {}
    for d in dets:
        fname = Path(d["file_path"]).name
        bbox = (
            round(d["bbox_x"], 4),
            round(d["bbox_y"], 4),
            round(d["bbox_width"], 4),
            round(d["bbox_height"], 4),
        )
        # Resolve to Latin taxon name (matching GT convention).
        # Species-level: use label (common name, both systems match).
        # Higher levels: use the taxon column matching the label's level.
        label = d["label"] or "(unlabeled)"
        level = d["level"]
        level_to_col = {
            "class": "taxon_class",
            "order": "taxon_order",
            "family": "taxon_family",
            "genus": "taxon_genus",
        }
        col = level_to_col.get(level)
        if col and d[col]:
            label = d[col]
        conf = d["label_confidence"] or 0.0
        lookup[(fname, bbox)] = (label, conf)

    return lookup


def compare(
    gt_lookup: dict[tuple[str, tuple], tuple[str, float]],
    db_lookup: dict[tuple[str, tuple], tuple[str, float]],
    verbose: bool = False,
) -> dict:
    """Compare GT (Latin-converted) vs DB detections and return summary."""
    exact = 0
    conf_diff: list[tuple] = []
    real_diff: list[tuple] = []
    gt_only = 0
    db_only = 0

    for key, (gt_label, gt_conf) in gt_lookup.items():
        db_entry = db_lookup.get(key)
        if db_entry is None:
            gt_only += 1
            continue

        db_label, db_conf = db_entry

        if gt_label == db_label:
            if abs(gt_conf - db_conf) < 0.01:
                exact += 1
            else:
                conf_diff.append((key, gt_label, gt_conf, db_conf))
        else:
            real_diff.append(
                (key, gt_label, gt_conf, db_label, db_conf)
            )

    # DB detections not in GT (skip videos / files not in GT)
    gt_filenames = {fname for fname, _ in gt_lookup}
    for key in db_lookup:
        if key not in gt_lookup:
            fname, _ = key
            if fname in gt_filenames:
                db_only += 1

    total_compared = exact + len(conf_diff) + len(real_diff)

    return {
        "exact": exact,
        "conf_diff": conf_diff,
        "real_diff": real_diff,
        "gt_only": gt_only,
        "db_only": db_only,
        "total_compared": total_compared,
    }


def print_results(results: dict, verbose: bool = False) -> None:
    """Print comparison results."""
    total = results["total_compared"]
    exact = results["exact"]

    print()
    print("=" * 60)
    print("  SpeciesNet comparison: official API vs AddaxAI DB")
    print("=" * 60)
    print()
    print(f"  Exact match:              {exact}")
    print(f"  Confidence-only diff:     {len(results['conf_diff'])}")
    print(f"  Label differences:        {len(results['real_diff'])}")
    print(f"  In GT only (not in DB):   {results['gt_only']}")
    print(f"  In DB only (not in GT):   {results['db_only']}")

    if total > 0:
        match_pct = exact / total * 100
        print(f"\n  Match rate: {match_pct:.1f}%")

    if results["real_diff"]:
        print(f"\n--- Real label differences ({len(results['real_diff'])}) ---")

        from collections import Counter

        diff_types = Counter()
        for _, gt_l, _, db_l, _ in results["real_diff"]:
            diff_types[(gt_l, db_l)] += 1

        print("\nBy type (official -> AddaxAI):")
        for (gt_l, db_l), count in diff_types.most_common(20):
            print(f"  {gt_l} -> {db_l}: {count}x")

        if verbose:
            print("\nAll differences:")
            for key, gt_l, gt_c, db_l, db_c in results["real_diff"]:
                fname, _ = key
                print(
                    f"  {fname}: "
                    f"official={gt_l} ({gt_c}) "
                    f"addaxai={db_l} ({db_c:.4f})"
                )

    if results["conf_diff"]:
        diffs = [abs(gt_c - db_c) for _, _, gt_c, db_c in results["conf_diff"]]
        print(
            f"\n--- Confidence differences ({len(diffs)}) ---"
            f"\n  min={min(diffs):.4f}  max={max(diffs):.4f}"
            f"  avg={sum(diffs) / len(diffs):.4f}"
        )
        if verbose:
            for key, label, gt_c, db_c in results["conf_diff"][:10]:
                fname, _ = key
                print(
                    f"  {fname}: {label} "
                    f"official={gt_c} addaxai={db_c:.4f} "
                    f"diff={abs(gt_c - db_c):.4f}"
                )
            remaining = len(results["conf_diff"]) - 10
            if remaining > 0:
                print(f"  ... and {remaining} more")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare official SpeciesNet output with AddaxAI DB."
    )
    parser.add_argument(
        "--gt",
        required=True,
        type=Path,
        help="Path to SPPNET_ground_truth.json",
    )
    parser.add_argument(
        "--project-id",
        required=False,
        default=None,
        help="AddaxAI project UUID (default: most recently created)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show all individual differences",
    )
    args = parser.parse_args()

    if not args.gt.exists():
        print(f"Error: GT file not found: {args.gt}", file=sys.stderr)
        sys.exit(1)

    settings = get_settings()
    db_path = settings.user_data_dir / "addaxai.db"
    if not db_path.exists():
        print(f"Error: database not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    # Resolve project ID
    conn = sqlite3.connect(str(db_path))
    if args.project_id:
        row = conn.execute(
            "SELECT id, country_code FROM projects WHERE id = ?",
            (args.project_id,),
        ).fetchone()
        if not row:
            print(
                f"Error: project {args.project_id} not found",
                file=sys.stderr,
            )
            conn.close()
            sys.exit(1)
    else:
        row = conn.execute(
            "SELECT id, country_code FROM projects"
            " ORDER BY created_at DESC LIMIT 1",
        ).fetchone()
        if not row:
            print("Error: no projects in database", file=sys.stderr)
            conn.close()
            sys.exit(1)
        args.project_id = row[0]
    conn.close()

    print(f"GT file: {args.gt}")
    print(f"Project: {args.project_id}")
    print(f"Country: {row[1] or 'not set'}")

    gt_lookup = load_gt(args.gt)
    db_lookup = load_db_detections(args.project_id, db_path)

    print(f"GT detections (with classifications): {len(gt_lookup)}")
    print(f"DB detections (with labels): {len(db_lookup)}")

    results = compare(gt_lookup, db_lookup, args.verbose)
    print_results(results, args.verbose)

    # Exit code: 0 if no real differences, 1 if there are
    sys.exit(1 if results["real_diff"] else 0)


if __name__ == "__main__":
    main()
