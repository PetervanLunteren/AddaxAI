#!/usr/bin/env python3
"""
Patch the 27 broken rows in the taxonomies that shipped before the port.

Targeted on purpose. Regenerating these files wholesale would rewrite
thousands of rows that are already correct just to fix a couple, and most
of them were hand-authored with no legacy source to regenerate from. So
each correction is written out and every other row is left untouched.

Two defect classes, both found by auditing all 15 shipped taxonomies:

  genus/species  The species column holds a full binomial whose genus
                 contradicts the genus column, so
                 format_scientific_name_from_taxonomy_row renders
                 "H. parahyaena brunnea". 21 rows.
  rank swap      class holds a GBIF order (squamata) with `reptilia`
                 below it in order. Wrong in both GBIF's backbone and
                 Linnaean terms. 5 rows.

Not every odd-looking row is a defect:

  - `equus` + `zebra hartmannae` is a valid subspecies and renders
    "E. zebra hartmannae". Left alone.
  - `capra` + `aegagrus hircus` is the valid trinomial for the domestic
    goat, which GBIF happens not to carry. Left alone.
  - SPECIESNET-v4-0-2-A is not touched at all. Its `grizzly bear` row
    reads `ursus` + `u. arctos` and renders "U. u. arctos", which looks
    exactly like a mangled abbreviation, but it is what SpeciesNet itself
    publishes in taxonomy_release and in the labels file the model ships.
    Our taxonomy.csv is generated from that labels file by
    generate_taxonomy_csv.py, so "correcting" it here would diverge from
    upstream and be silently reverted the next time anyone regenerates.
    It is an upstream question, not ours.
"""

import csv
import sys
from pathlib import Path

# model -> model_class -> corrected rank columns.
# Every value here was checked against GBIF and the current literature.
FIXES: dict[str, dict[str, dict[str, str]]] = {
    "SAH-DRY-ADS-v1": {
        # All 14 come from the resolver rerun against the legacy
        # taxon-mapping.csv, whose GBIF keys reproduced 312 of the 326
        # shipped rows exactly. These are the 14 that disagreed, and in
        # every one the shipped file had paired an accepted genus with a
        # synonym's full binomial.
        "striped ground squirrel": {"genus": "euxerus", "species": "erythropus"},
        "brown hyena": {"genus": "parahyaena", "species": "brunnea"},
        "slender mongoose": {"genus": "herpestes", "species": "sanguineus"},
        "small cape grey mongoose": {"genus": "herpestes", "species": "pulverulentus"},
        "eland": {"genus": "tragelaphus", "species": "oryx"},
        "suni": {"genus": "nesotragus", "species": "moschatus"},
        "white-eyed slaty flycatcher": {"genus": "melaenornis", "species": "fischeri"},
        "gray-headed oliveback": {"genus": "delacourella", "species": "capistrata"},
        "red-crested bustard": {"genus": "eupodotis", "species": "ruficrista"},
        "buff-crested bustard": {"genus": "eupodotis", "species": "gindiana"},
        "white-quilled bustard": {"genus": "eupodotis", "species": "afraoides"},
        "scaly ground-roller": {"genus": "brachypteracias", "species": "squamiger"},
        "laughing dove": {"genus": "streptopelia", "species": "senegalensis"},
        "madagascar turtle-dove": {"genus": "streptopelia", "species": "picturata"},
    },
    "KIR-HEX-v1": {
        # Uncia was folded back into Panthera; the row has the two swapped.
        "panthera_uncia": {"genus": "panthera", "species": "uncia"},
    },
    "TAS-BB-v1": {
        "sooty_shearwater": {"genus": "ardenna", "species": "grisea"},
        "yellow_throated_honeyeater": {"genus": "nesoptilotis", "species": "flavicollis"},
        "tasmanian_nativehen": {"genus": "tribonyx", "species": "mortierii"},
        "brown_quail": {"genus": "synoicus", "species": "ypsilophora"},
        # GBIF files Squamata at class rank; AddaxAI shows a Linnaean tree.
        "blotched_blue_tongue": {"class": "reptilia", "order": "squamata"},
        "skink": {"class": "reptilia", "order": "squamata"},
        "snake": {"class": "reptilia", "order": "squamata"},
    },
    "PAM-SDZWA-v1": {
        "highland coati": {"genus": "nasuella", "species": "olivacea"},
        "jaguarundi": {"genus": "herpailurus", "species": "yagouaroundi"},
        # A generic "reptile" could be a turtle, so it stops at the class.
        # Matches what PAN-SDZWA-v1 ships from GBIF key 12170551.
        "reptile": {"class": "reptilia", "order": ""},
    },
    "SWUSA-SDZWA-v3": {
        "reptile": {"class": "reptilia", "order": ""},
    },
}

RANKS = ["class", "order", "family", "genus", "species"]
HEADER = ["model_class", *RANKS]


def render(row: dict) -> str:
    """Lineage plus the name the UI actually prints."""
    g, s = row["genus"], row["species"]
    if g and s:
        name = f"{g[0].upper()}. {s}"
    else:
        name = next(
            (row[r].capitalize() for r in ("genus", "family", "order", "class") if row[r]),
            "(none)",
        )
    lineage = " > ".join(row[r] for r in RANKS if row[r]) or "(no taxonomy)"
    return f"{lineage}   [{name}]"


def main() -> int:
    src_dir = Path(sys.argv[1])   # dir holding <model>/taxonomy.csv as shipped
    out_dir = Path(sys.argv[2])
    total = 0
    for model, fixes in FIXES.items():
        src = src_dir / model / "taxonomy.csv"
        if not src.exists():
            print(f"{model}: no shipped taxonomy.csv at {src}", file=sys.stderr)
            return 1
        rows = list(csv.DictReader(open(src)))
        applied = 0
        print(f"\n=== {model} ===")
        for row in rows:
            fix = fixes.get(row["model_class"])
            if not fix:
                continue
            before = render(row)
            row.update(fix)
            print(f"  {row['model_class']}\n      was {before}\n      now {render(row)}")
            applied += 1
        missing = set(fixes) - {r["model_class"] for r in rows}
        if missing:
            print(f"  !! not found in the file: {sorted(missing)}", file=sys.stderr)
            return 1
        dest = out_dir / model
        dest.mkdir(parents=True, exist_ok=True)
        with open(dest / "taxonomy.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=HEADER)
            w.writeheader()
            w.writerows(rows)
        print(f"  {applied} row(s) fixed, {len(rows) - applied} untouched")
        total += applied
    print(f"\n{total} rows fixed across {len(FIXES)} models")
    return 0


if __name__ == "__main__":
    sys.exit(main())
