#!/usr/bin/env python3
"""
Build a WebUI taxonomy.csv by resolving model class names against GBIF.

This is the single producer of taxonomy.csv for ported models. It runs at
staging time on a developer machine, never at runtime, so the GBIF calls
are not an app dependency.

Input: a CSV with a `model_class` column, local or an http(s) URL. Every
legacy taxon-mapping.csv qualifies as-is, in all three of the header
shapes the legacy repos publish.

Output (consumed by app.ml.taxonomic_rollup.load_taxonomy_lookup and
app.ml.taxonomy_db.populate_taxonomy_from_csv):

    model_class,class,order,family,genus,species

All values lowercase, empty when unknown. The species column holds the
name minus its leading genus, so "Panthera pardus" -> "pardus" and the
subspecies "Panthera pardus saxicolor" -> "pardus saxicolor". That
matches what is already shipped (see SAH-DRY-ADS-v1 and TKM-ADS-v1).

What gets asked, in order
-------------------------
Each row is resolved from the best evidence it carries. One rule, three
sources, first hit wins:

  1. `GBIF_usageKey` / `gbif_usage_key`  -> resolve the key directly.
     Authoritative, and what the legacy CSVs were built from. A negative
     key is a sentinel, not a key: cls-training-pipeline uses -1 for
     "not applicable or a mixed taxon" and -10/-20/-30 for wolf/dog/
     dingo, whose lineage is hardcoded into the level_* columns instead.
     Those fall through to 2, where the hardcoded answer already is.
  2. `scientific_name`, or the finest non-empty legacy `level_*` cell
     (any "<rank> " prefix stripped)   -> match that name against GBIF.
     Handles all three legacy header shapes uniformly, including the
     DeepForestVision file whose cells carry no rank prefixes.
  3. Nothing usable -> empty row, reported for review.

`model_class` itself is NEVER matched against GBIF. Model classes are
English common names and GBIF's matcher answers them badly: "aardvark"
comes back as kingdom Animalia, and "serval" matches a beetle genus
(Mordellidae) with an EXACT, full-confidence hit. Silently mislabelling a
serval as an insect is worse than leaving the row empty, so for a model
with no taxonomy at all, author a `scientific_name` column and feed that
in rather than relying on the class names.

Why GBIF and not the legacy level_* columns verbatim
----------------------------------------------------
When a key is present, GBIF is a strictly better source: the legacy cells
are a stale snapshot in three mutually incompatible shapes. Resolving the
326 keys of the Sub-Saharan model reproduces its hand-made, shipped
taxonomy.csv exactly, row for row, so the key path is known-good.

GBIF helps, the literature decides
----------------------------------
GBIF is a lookup, not the standard. Where its backbone disagrees with the
current literature, the literature wins and this script corrects it. Two
mechanisms, both explicit:

**ORDER_AS_CLASS.** GBIF's backbone has no Reptilia class: it places
Squamata, Testudines and Crocodylia at *class* rank with an empty order,
so a tortoise comes back as `class=Testudines`. The same happens for
amphibian orders on some keys. The legacy CSVs copied that faithfully; it
was never an AddaxAI bug. AddaxAI shows a Linnaean tree, so these fold
back under their real class, reproducing what SAH-DRY-ADS-v1 ships:

    leopard tortoise,reptilia,testudines,testudinidae,stigmochelys,pardalis

The table mirrors the one in cls-training-pipeline/taxon-mapping, which
is the producer side of this same problem. Keep the two in step.

**Per-row overrides.** Supply any of the five rank columns in the input
CSV and that row is written from those columns alone: GBIF is not
consulted, and the columns left blank stay blank. One rule, so the input
CSV always shows exactly what the row will become.

Use it wherever GBIF lags a published reclassification (the 2021 split
moved the American mink to *Neogale*, which GBIF still resolves as a
genus-rank synonym of *Mustela* with no species at all) and for mixed
groups that have no single taxon (`snake sp` is `reptilia,squamata` and
nothing below, since GBIF files the suborder Serpentes under *family*).

A record's own rank is trusted over its rank fields
---------------------------------------------------
At rank R, the value for column R is taken from canonicalName rather than
from GBIF's `class`/`order`/... fields, because the two disagree. Key
12170551 ("Reptilia", the key this project uses for unknown reptiles)
comes back as rank=CLASS, canonicalName=Reptilia, and class=Squamata.
Trusting the field would claim every unidentified reptile is a squamate
rather than possibly a turtle.

Review the output
-----------------
Rows that could not be resolved, matched only fuzzily, or matched at a
coarser rank than requested are printed to stderr. Non-label classes
(bait, blank, empty, ...) are expected to be empty and are not reported.
GBIF also silently resolves synonyms, which changes the genus:
"Cephalophus monticola" comes back as "Philantomba monticola". Read the
summary before uploading anything.

Usage
-----
    cd backend
    source venv/bin/activate

    # From a legacy taxon-mapping.csv in a HuggingFace repo
    python scripts/resolve_taxonomy_gbif.py \\
        https://huggingface.co/Addax-Data-Science/IND-ADS-v1/resolve/main/taxon-mapping.csv \\
        out/IND-ADS-v1/taxonomy.csv

    # For a model with no legacy taxonomy, hand-author the scientific
    # names first, then resolve them:
    #     model_class,scientific_name
    #     eastern chipmunk,Tamias striatus
    python scripts/resolve_taxonomy_gbif.py names.csv out/taxonomy.csv

Options:
    --no-key      Ignore GBIF keys and resolve via the name path instead.
                  Use to check whether a legacy key still agrees with the
                  name it claims to be.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

# Reuse the app's definition rather than restating it: these are the
# classes the pipeline throws away, so they get an empty taxonomy by
# design and must never be sent to GBIF.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.ml.label_exclusion import NON_LABEL_CLASSES  # noqa: E402

WEBUI_HEADER = ["model_class", "class", "order", "family", "genus", "species"]
RANKS = ["class", "order", "family", "genus", "species"]

GBIF_API = "https://api.gbif.org/v1"

# Column names the legacy repos use for the GBIF key. Both spellings are
# in the wild across models.
_KEY_COLUMNS = ("GBIF_usageKey", "gbif_usage_key")

# GBIF ranks at or below species; only these fill the species column.
_SPECIES_RANKS = frozenset({"SPECIES", "SUBSPECIES", "VARIETY", "FORM"})

# Taxa GBIF's backbone returns at class rank that are really orders. Maps
# the bad class -> (real class, real order). Mirrors ORDER_AS_CLASS in
# cls-training-pipeline/taxon-mapping; keep the two in step. Deliberately
# an explicit, closed list rather than a rule inferred from the data.
ORDER_AS_CLASS: dict[str, tuple[str, str]] = {
    "squamata": ("reptilia", "squamata"),
    "testudines": ("reptilia", "testudines"),
    "crocodylia": ("reptilia", "crocodylia"),
    "rhynchocephalia": ("reptilia", "rhynchocephalia"),
    "anura": ("amphibia", "anura"),
    "caudata": ("amphibia", "caudata"),
    "urodela": ("amphibia", "urodela"),
    "gymnophiona": ("amphibia", "gymnophiona"),
}

# GBIF ranks, coarse to fine, and the column each one owns.
_RANK_COLUMN = {
    "CLASS": "class",
    "ORDER": "order",
    "FAMILY": "family",
    "GENUS": "genus",
}

# Below this GBIF match confidence the row is reported for review.
_MIN_MATCH_CONFIDENCE = 90


@dataclass(frozen=True)
class Entry:
    """One model class plus whatever evidence we have to resolve it."""

    model_class: str
    gbif_key: str | None
    scientific_name: str | None
    # Rank columns supplied by hand in the input CSV. These win over
    # whatever GBIF says, for the columns they cover. Use where GBIF's
    # backbone lags the literature.
    overrides: dict[str, str] = field(default_factory=dict)


def _gbif_get(path: str, **params: str) -> dict | None:
    """GET a GBIF endpoint. Returns None on any failure (reported later)."""
    url = f"{GBIF_API}/{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return json.load(response)
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
        print(f"  GBIF request failed ({url}): {e}", file=sys.stderr)
        return None


def resolve_by_key(key: str) -> dict | None:
    """Resolve a GBIF usage key to its record."""
    record = _gbif_get(f"species/{key}")
    if record and record.get("key"):
        return record
    return None


def resolve_by_name(name: str) -> dict | None:
    """
    Match a name against the GBIF backbone, restricted to animals.

    GBIF's matcher is case-sensitive in practice: "Aves" matches the
    class exactly, while "aves" degrades to a HIGHERRANK hit on kingdom
    Animalia. The legacy CSVs are all-lowercase, so capitalise before
    asking. Scientific names capitalise the leading term only ("Panthera
    pardus"), which is exactly what this does.
    """
    if not name:
        return None
    capitalised = name[0].upper() + name[1:]
    record = _gbif_get(
        "species/match", name=capitalised, kingdom="Animalia", strict="false"
    )
    if record and record.get("matchType") not in (None, "NONE"):
        return record
    return None


def _split_binomial(record: dict) -> tuple[str, str] | None:
    """
    Split a species-rank record's canonical name into (genus, epithet).

    Returns None above species rank. A scientific name is always "Genus
    epithet", so both columns are taken from that one string and can
    never contradict each other:

        "Panthera pardus"             -> ("panthera", "pardus")
        "Equus zebra hartmannae"      -> ("equus", "zebra hartmannae")

    Deriving genus from the binomial rather than from GBIF's `genus`
    field matters because the two disagree on reclassified taxa. GBIF
    returns canonicalName "Parahyaena brunnea" alongside genus "Hyaena",
    and pairing those produces `genus=hyaena, species="parahyaena
    brunnea"` — which the UI renders as "H. parahyaena brunnea". That
    exact defect is live in the shipped SAH-DRY-ADS-v1 taxonomy for
    roughly 18 species.

    The name a key resolves to is kept as-is even when GBIF calls it a
    synonym: the African wild cat's key is Felis lybica, which GBIF still
    folds into Felis silvestris, and collapsing the two would be a
    regression. Synonyms are reported for review instead.
    """
    rank = (record.get("rank") or "").upper()
    if rank not in _SPECIES_RANKS:
        return None

    canonical = (record.get("canonicalName") or record.get("species") or "").strip()
    parts = canonical.split()
    if len(parts) < 2:
        return None
    return parts[0].lower(), " ".join(parts[1:]).lower()


def record_to_ranks(record: dict) -> dict[str, str]:
    """Map a GBIF record onto the WebUI's five rank columns."""
    ranks = {
        "class": (record.get("class") or "").strip().lower(),
        "order": (record.get("order") or "").strip().lower(),
        "family": (record.get("family") or "").strip().lower(),
        "genus": (record.get("genus") or "").strip().lower(),
        "species": "",
    }

    rank = (record.get("rank") or "").upper()
    canonical = (record.get("canonicalName") or "").strip().lower()

    # A record is the authority on its own rank; GBIF's rank fields can
    # disagree with it. Key 12170551 is rank=CLASS canonicalName=Reptilia
    # with class=Squamata, and only canonicalName is right.
    if rank in _RANK_COLUMN and canonical:
        ranks[_RANK_COLUMN[rank]] = canonical

    binomial = _split_binomial(record)
    if binomial:
        ranks["genus"], ranks["species"] = binomial

    # Fold GBIF's order-at-class-rank taxa back under their real class.
    # Guarded on an empty order so a future backbone that starts filling
    # it in cannot silently lose the value.
    if ranks["class"] in ORDER_AS_CLASS and not ranks["order"]:
        ranks["class"], ranks["order"] = ORDER_AS_CLASS[ranks["class"]]

    return ranks


def _summary(row: dict[str, str]) -> str:
    """Compact lineage for the review log, e.g. "mammalia > carnivora > neogale vison"."""
    parts = [row[r] for r in RANKS if row.get(r)]
    if row.get("genus") and row.get("species"):
        parts = parts[:-2] + [f"{row['genus']} {row['species']}"]
    return " > ".join(parts) or "no taxonomy"


def resolve_entry(entry: Entry) -> tuple[dict[str, str], str | None]:
    """
    Resolve one model class to a taxonomy row.

    Returns (row, warning). `warning` is None when the row is trustworthy
    and a human-readable string when it needs review.
    """
    row = {col: "" for col in WEBUI_HEADER}
    row["model_class"] = entry.model_class

    # Non-label classes carry no taxonomy by design.
    if entry.model_class in NON_LABEL_CLASSES:
        return row, None

    # Any hand-written column means the whole row is hand-written. GBIF
    # is a convenience, not the authority: where it disagrees with the
    # literature, or has no answer at all, this is how a human wins.
    # Blank columns stay blank rather than being filled in behind the
    # author's back, so what the input CSV says is what gets written.
    if entry.overrides:
        row.update(entry.overrides)
        return row, f"hand-written, GBIF not consulted ({_summary(row)})"

    if entry.gbif_key:
        record, matched_by = resolve_by_key(entry.gbif_key), "key"
        if record is None:
            return row, f"GBIF key {entry.gbif_key} did not resolve"
    elif entry.scientific_name:
        record, matched_by = resolve_by_name(entry.scientific_name), "name"
        if record is None:
            return row, f"no GBIF match for {entry.scientific_name!r}"
    else:
        # Deliberately not falling back to model_class: see the module
        # docstring on why "serval" resolves to a beetle.
        return row, "no GBIF key and no scientific name to resolve"

    row.update(record_to_ranks(record))

    canonical = record.get("canonicalName") or "?"
    if not row["class"]:
        return row, f"matched {canonical!r} but GBIF gave no class"

    warning = None
    if matched_by == "name":
        match_type = record.get("matchType")
        confidence = record.get("confidence")
        if match_type != "EXACT":
            warning = (
                f"{match_type} match: {entry.scientific_name!r} -> {canonical!r}"
            )
        elif confidence is not None and confidence < _MIN_MATCH_CONFIDENCE:
            warning = f"low confidence ({confidence}) -> {canonical!r}"

    # Synonyms are kept as written rather than silently swapped for the
    # accepted name: the African wild cat's key is Felis lybica, which
    # GBIF still folds into Felis silvestris, and collapsing the two
    # would lose a real distinction. Report both so the call is a human's.
    status = (record.get("taxonomicStatus") or record.get("status") or "").upper()
    if status == "SYNONYM":
        # `species` carries the accepted binomial at species rank; above
        # it (a synonymised genus like Neogale) only the rank fields do.
        accepted = (
            record.get("accepted")
            or record.get("species")
            or record.get("genus")
            or record.get("family")
            or "an unnamed taxon"
        )
        warning = f"wrote {canonical!r}; GBIF calls it a synonym of {accepted!r}"

    return row, warning


def _usable_gbif_key(raw: str | None) -> str | None:
    """
    Return the key only when it is a real GBIF usage key.

    cls-training-pipeline uses negative numbers as sentinels rather than
    keys: -1 means "not applicable, or a mixed taxon like caprid/raptor",
    and -10/-20/-30 stand for wolf/dog/dingo, whose lineage that pipeline
    hardcodes straight into the level_* columns. Treating a sentinel as
    absent lets those rows fall through to the name path, where the
    hardcoded answer already sits, e.g. dog carries
    `level_species = "species Canis lupus familiaris"`.
    """
    value = (raw or "").strip()
    if not value:
        return None
    try:
        if int(value) <= 0:
            return None
    except ValueError:
        return None
    return value


def _strip_rank_prefix(value: str) -> str | None:
    """Return the name in a `"<rank> Name"` cell, or None if unprefixed."""
    for rank in RANKS:
        if value.lower().startswith(f"{rank} "):
            return value[len(rank) + 1:].strip() or None
    return None


def _finest_legacy_name(row: dict[str, str]) -> str | None:
    """
    Return the most specific real taxon the legacy `level_*` columns carry.

    The rank prefix is the signal. In the prefixed variants a cell either
    reads `"<rank> Name"` and is a real taxon, or it carries a plain
    descriptive string and is deliberate filler:

        small mammal | class Mammalia | Small mammal | Small mammal | ...
        raptor       | class Aves     | Raptor       | Raptor       | ...

    That is the taxon-mapping README's rule for mixed groups: go as deep
    as real ranks allow, then repeat a label. So walk species -> class and
    take the finest *prefixed* cell, which is Mammalia and Aves here, not
    "Small mammal" and "Raptor".

    Only when no cell in the row is prefixed at all is the file using the
    unprefixed shape (the DeepForestVision one, `aardvark | mammalia |
    tubulidentata | ...`), and then the finest non-empty cell is the
    answer.
    """
    cells = [
        ((row.get(f"level_{rank}") or "").strip(), rank)
        for rank in reversed(RANKS)
    ]

    for value, _rank in cells:
        if value and (name := _strip_rank_prefix(value)):
            return name

    # No prefixes anywhere: unprefixed file, take the finest cell as-is.
    for value, _rank in cells:
        if value:
            return value
    return None


def _read_entries(path: Path, use_keys: bool) -> list[Entry]:
    """Read entries from a CSV carrying a `model_class` column."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        if "model_class" not in fieldnames:
            raise SystemExit(
                f"{path} has no `model_class` column. Headers seen: {fieldnames}"
            )

        key_column = next((c for c in _KEY_COLUMNS if c in fieldnames), None)
        if key_column and use_keys:
            print(f"Using GBIF keys from `{key_column}`, falling back to names")
        else:
            print("Resolving by scientific name")

        entries: list[Entry] = []
        seen: set[str] = set()
        for row in reader:
            model_class = (row.get("model_class") or "").strip().lower()
            if not model_class or model_class in seen:
                continue
            seen.add(model_class)

            key = None
            if key_column and use_keys:
                key = _usable_gbif_key(row.get(key_column))

            name = (row.get("scientific_name") or "").strip() or None
            if name is None:
                name = _finest_legacy_name(row)

            # Any rank column present in the input is a hand-written
            # correction and outranks GBIF for that column.
            overrides = {
                rank: (row.get(rank) or "").strip().lower()
                for rank in RANKS
                if (row.get(rank) or "").strip()
            }

            entries.append(Entry(model_class, key, name, overrides))
    return entries


def _resolve_input(arg: str) -> tuple[Path, bool]:
    """Resolve the input arg to a local path, downloading if it's a URL.

    Returns (path, is_temp). Caller removes the file when is_temp.
    """
    if arg.startswith(("http://", "https://")):
        import tempfile

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
        description="Build a WebUI taxonomy.csv by resolving classes against GBIF."
    )
    parser.add_argument(
        "input",
        help="CSV with a `model_class` column (any legacy taxon-mapping.csv "
        "qualifies). Local path or http(s):// URL.",
    )
    parser.add_argument("output", help="Output taxonomy.csv path.")
    parser.add_argument(
        "--no-key",
        action="store_true",
        help="Ignore GBIF keys and resolve via the name path instead.",
    )
    args = parser.parse_args(argv)

    input_path, is_temp = _resolve_input(args.input)
    try:
        entries = _read_entries(input_path, use_keys=not args.no_key)
    finally:
        if is_temp:
            input_path.unlink(missing_ok=True)

    if not entries:
        raise SystemExit(f"No model classes found in {args.input}")

    rows: list[dict[str, str]] = []
    warnings: list[tuple[str, str]] = []
    for i, entry in enumerate(entries, 1):
        print(f"  [{i}/{len(entries)}] {entry.model_class}", file=sys.stderr)
        row, warning = resolve_entry(entry)
        rows.append(row)
        if warning:
            warnings.append((entry.model_class, warning))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=WEBUI_HEADER)
        writer.writeheader()
        writer.writerows(rows)

    resolved = sum(1 for r in rows if r["class"])
    blank_by_design = sum(
        1 for r in rows if r["model_class"] in NON_LABEL_CLASSES
    )
    print(
        f"\nWrote {output_path} ({len(rows)} rows, {resolved} with a class, "
        f"{blank_by_design} non-label)"
    )

    if warnings:
        print(
            f"\n{len(warnings)} row(s) need review before upload:",
            file=sys.stderr,
        )
        for model_class, warning in warnings:
            print(f"  {model_class:40s}  {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
