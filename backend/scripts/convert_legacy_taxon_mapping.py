#!/usr/bin/env python3
"""
Convert a legacy AddaxAI taxon-mapping.csv into the WebUI's taxonomy.csv.

Legacy format (as published in HuggingFace model repos under
Addax-Data-Science/<model>/taxon-mapping.csv):

    gbif_usage_key,model_class,n_images,
    level_class,level_order,level_family,level_genus,level_species,
    only_above_1000,only_above_10000,only_above_100000

Each `level_<rank>` cell holds a string of the form `<rank> <name>`,
e.g. "order Carnivora", "family Felidae", "species Felis catus". When
the actual rank for that row is unknown, the cell falls back to a
higher rank (e.g. "class Aves" appearing in `level_species` for a row
where species-level info is unavailable).

WebUI format (consumed by app.ml.taxonomic_rollup.load_taxonomy_lookup):

    model_class,class,order,family,genus,species

All values lowercase, empty when unknown, species column holds the
epithet only (so "species Felis catus" -> "catus").

Caveat
------
The legacy CSV is taxonomically broken for non-mammals: it stuffs
orders into the class column (e.g. "class Testudines" for tortoises,
where Testudines is actually an order in class Reptilia). This script
translates the legacy data literally; the output will need manual
review and correction for any row where the resulting `class` value
isn't one of the well-known taxonomic classes. Suspect rows are
printed to stderr at the end of each run.

Usage
-----
    cd backend && source venv/bin/activate

    # Local file
    python scripts/convert_legacy_taxon_mapping.py path/to/taxon-mapping.csv

    # HuggingFace URL (downloaded to a temp file, then converted)
    python scripts/convert_legacy_taxon_mapping.py \\
        https://huggingface.co/Addax-Data-Science/HWI-ADS-v1/resolve/main/taxon-mapping.csv \\
        ~/AddaxAI/models/cls/HWI-ADS-v1/taxonomy.csv

If the output path is omitted, writes `taxonomy.csv` next to the input
(local files only). For URLs the output path is required.
"""

from __future__ import annotations

import argparse
import csv
import sys
import tempfile
import urllib.request
from pathlib import Path

WEBUI_HEADER = ["model_class", "class", "order", "family", "genus", "species"]
RANKS = ["class", "order", "family", "genus", "species"]

# Well-known taxonomic classes seen in camera-trap models. Used purely
# to flag suspect rows in the converted output: when the `class` cell
# holds something not on this list, the legacy CSV likely had an
# order-rank value mislabelled as a class (a known bug for reptiles,
# amphibians, fish, etc. in the source repo's taxon-mapping.csv).
KNOWN_CLASSES = {
    "mammalia",
    "aves",
    "reptilia",
    "amphibia",
    "actinopterygii",
    "chondrichthyes",
    "insecta",
    "arachnida",
}


def _strip_rank_prefix(value: str, rank: str) -> str | None:
    """Return the name component when `value` matches the expected rank.

    Legacy cells look like "order Carnivora". Returns "carnivora" when
    `rank == "order"`. Returns None when the cell's prefix doesn't
    match the requested rank — that happens when the source row has no
    info at this resolution and falls back to a coarser rank.
    """
    value = value.strip()
    if not value:
        return None
    prefix = f"{rank} "
    if not value.lower().startswith(prefix):
        return None
    name = value[len(prefix):].strip()
    if rank == "species":
        # "species Felis catus" -> "catus" (binomial epithet only).
        # Edge case "species axis axis" -> "axis" via last-token rule.
        parts = name.split()
        if len(parts) >= 2:
            name = parts[-1]
    return name.lower() or None


def convert_row(row: dict[str, str]) -> dict[str, str]:
    """Convert one legacy row to a WebUI row."""
    out = {col: "" for col in WEBUI_HEADER}
    out["model_class"] = (row.get("model_class") or "").strip().lower()
    for rank in RANKS:
        src_value = row.get(f"level_{rank}", "") or ""
        name = _strip_rank_prefix(src_value, rank)
        if name:
            out[rank] = name
    return out


def convert_csv(
    input_path: Path, output_path: Path
) -> tuple[int, int, list[tuple[str, str]]]:
    """Convert `input_path` (legacy CSV) to `output_path` (WebUI CSV).

    Returns (rows_written, rows_skipped, suspect_rows). `suspect_rows`
    is a list of (model_class, class_value) for rows where the
    converted `class` cell is non-empty and not on the well-known
    taxonomic-classes list, so the caller can flag them for manual
    review.
    """
    written = 0
    skipped = 0
    suspect: list[tuple[str, str]] = []
    with open(input_path, newline="", encoding="utf-8") as src:
        reader = csv.DictReader(src)
        if "model_class" not in (reader.fieldnames or []):
            raise SystemExit(
                f"Input {input_path} is missing the required `model_class` column. "
                f"Headers seen: {reader.fieldnames}"
            )
        with open(output_path, "w", newline="", encoding="utf-8") as dst:
            writer = csv.DictWriter(dst, fieldnames=WEBUI_HEADER)
            writer.writeheader()
            for row in reader:
                converted = convert_row(row)
                if not converted["model_class"]:
                    skipped += 1
                    continue
                writer.writerow(converted)
                written += 1
                cls = converted["class"]
                if cls and cls not in KNOWN_CLASSES:
                    suspect.append((converted["model_class"], cls))
    return written, skipped, suspect


def _resolve_input(arg: str) -> tuple[Path, bool]:
    """Resolve the input arg to a local path. Download if it's a URL.

    Returns (path, is_temp). Caller must remove the file when is_temp.
    """
    if arg.startswith(("http://", "https://")):
        tmp = tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False, prefix="legacy_taxon_"
        )
        tmp.close()
        urllib.request.urlretrieve(arg, tmp.name)
        return Path(tmp.name), True
    path = Path(arg)
    if not path.exists():
        raise SystemExit(f"Input not found: {path}")
    return path, False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert a legacy AddaxAI taxon-mapping.csv to WebUI taxonomy.csv format."
    )
    parser.add_argument(
        "input",
        help="Legacy taxon-mapping.csv: local path or http(s):// URL.",
    )
    parser.add_argument(
        "output",
        nargs="?",
        help=(
            "Output taxonomy.csv path. Defaults to `taxonomy.csv` next to a "
            "local input file. Required when input is a URL."
        ),
    )
    args = parser.parse_args(argv)

    input_path, is_temp = _resolve_input(args.input)
    try:
        if args.output:
            output_path = Path(args.output)
        elif is_temp:
            raise SystemExit(
                "Output path is required when input is a URL "
                "(no sensible default location)."
            )
        else:
            output_path = input_path.with_name("taxonomy.csv")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        written, skipped, suspect = convert_csv(input_path, output_path)
    finally:
        if is_temp:
            input_path.unlink(missing_ok=True)

    print(f"Wrote {output_path} ({written} rows, {skipped} skipped)")
    if suspect:
        print(
            f"\nWARNING: {len(suspect)} row(s) have a `class` value that "
            f"isn't a well-known taxonomic class. The legacy CSV often puts "
            f"orders into the class column for non-mammals (e.g. 'Testudines' "
            f"is actually an order in Reptilia, not a class). Review these "
            f"rows and manually correct the class/order columns:",
            file=sys.stderr,
        )
        for model_class, cls in suspect:
            print(f"  {model_class:40s}  class={cls}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
