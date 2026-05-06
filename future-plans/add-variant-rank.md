# Add variant rank to taxonomy

Plan for a future change. Not scheduled. Captured here so we can pick it up without re-deriving the design.

## Why

Some classification models predict below species level (e.g. adult vs juvenile fox). Today `label_taxonomy` stops at species, so these sub-species classes have nowhere clean to live. Without a fix, the model author has to encode the variant inside the species name ("Vulpes vulpes adult"), which:

- splits dashboard and chart counts by variant,
- requires the user to click both "adult" and "juvenile" entries in every filter to see all foxes,
- puts a lie in the formal taxonomy column (species is no longer just the Latin binomial).

## Decision

Add one optional rank below species, called `variant`. Free-form string. The model author writes whatever fits the model: `"juvenile"`, `"male"`, `"melanistic"`, `"adult male"`. A single generic rank covers age, sex, coat colour, morph, and any future sub-species attribute. We do not add separate `lifeStage` / `sex` / `coat` columns to taxonomy.

### Why one column, not three

`lifeStage`, `sex`, `coat` are facets, not ranks. They have no parent / child relation to each other. Putting them on the taxonomy table breaks every consumer that walks the hierarchy in order:

- the filter tree builder needs an arbitrary nesting order, which is wrong by construction,
- charts need a "group by" selector for each facet,
- `Detection.label` is a single string, so a multi-facet model either explodes the taxonomy table combinatorially or needs new columns on `Detection` itself,
- confusion matrix and per-class performance get sparse and noisy with multi-facet classes.

Human-recorded observation attributes (lifeStage, sex, behaviour at verification time) are a separate concern and belong on `Detection`, not on taxonomy. See "Out of scope" below.

## Scope

### Schema

- New nullable column `taxon_variant TEXT` on `label_taxonomy`.
- New value `"variant"` in the `level` enum.
- Alembic migration that adds the column with `NULL` default. Existing rows untouched.

### CSV format

Add an optional `variant` column to `taxonomy.csv`:

```csv
model_class,class,order,family,genus,species,variant
vulpes_vulpes_adult,mammalia,carnivora,canidae,vulpes,vulpes vulpes,adult
vulpes_vulpes_juvenile,mammalia,carnivora,canidae,vulpes,vulpes vulpes,juvenile
sus_scrofa,mammalia,artiodactyla,suidae,sus,sus scrofa,
```

Rules:
- column is optional. Existing CSVs without it keep working.
- when the column is present and the value is non-empty, the row is stored with `level = "variant"`.
- when the value is empty, the row is treated like any other species row.
- empty `variant` plus partial higher ranks still resolves `level` to the deepest non-empty taxonomic rank, same as today.

### Code changes

| File | Change |
|------|--------|
| `backend/alembic/versions/<new>.py` | Add `taxon_variant` column. Update `level` enum if it is checked at the DB layer (it is currently a string in SQLite, no constraint, so likely a no-op). |
| `backend/app/models/label_taxonomy.py` | Add `taxon_variant: Mapped[str \| None]`. |
| `backend/app/ml/taxonomy_db.py` | `populate_taxonomy_from_csv` reads the optional column. Promote `level` to `"variant"` when set. `add_rollup_taxonomy_entry` accepts variant rows. |
| `backend/app/ml/taxonomic_rollup.py` | One extra rung at the bottom: variant > species > genus > family > order > class. Sums confidences from variant siblings up to species when the variant tier is below threshold. |
| `backend/app/api/crud/label_tree.py` | `build_label_filter_tree` adds one nesting level under species. The variant node displays the **suffix only** (e.g. "juvenile"), not the full label. The species node continues to show the binomial. |
| `backend/app/ml/taxonomy_parser.py` | Parser recognises the optional column. |
| `backend/tests/ml/test_taxonomy_db.py` | New cases: CSV with variant column, CSV without, mixed rows, idempotent re-population. |
| `backend/tests/api/test_label_tree.py` | New cases: tree depth gains a level, variant node displays suffix only, count-by-event and count-by-detection both correct, taxonomy with no variants is unchanged. |
| `frontend/src/components/taxonomy/*` (whichever renders the tree) | Render one more level of indentation. No structural change because the tree is already recursive. |

Code that does **not** change because it groups by raw `Detection.label` string and is rank-agnostic:
- dashboard top species, activity pattern, detection trend, alert counters, verification progress
- smoothing, postprocessing
- verification picker, label picker (already shows raw `model_class`)
- camtrap-dp export (default behaviour)
- confusion matrix, per-class performance, activity overlap

These all continue to work at the variant level out of the box. That is the intended UX for v1: dashboards and stats show "Vulpes vulpes juvenile" and "Vulpes vulpes adult" as distinct entries. The user gets species-level totals by clicking the species node in the taxonomy filter, which is the existing pattern for genus > species rollup.

## Behavioural choices, locked in

These were settled in the design discussion. Listed here so the implementer does not re-open them.

| Question | Answer |
|----------|--------|
| Multiple sub-species attributes on one detection (age + sex + coat at once)? | No. One per model. If a model needs both, it concatenates into the variant string ("adult male"). |
| Group adult / juvenile to species in dashboards by default? | No. Show separately. Add a "group to species level" toggle later only if users ask. |
| Confusion matrix / per-class performance evaluated at variant level? | Yes. That is what the model predicts. |
| Verification picker shows variant entries? | Yes when the project's model has variants. Otherwise the picker is unchanged. |
| `variant` column required when present in CSV? | Optional always. Empty cell means "no variant on this row". |
| Rollup walks variant tier? | Yes. Variant > species > genus > family > order > class. |
| Mid-project model swap migrates labels across models? | No. Labels stay tied to the model that produced them. Same as today. |
| Filter tree label for variant nodes? | Suffix only ("juvenile"), not the full binomial. |
| Rank name? | `variant`. Not `form`, `subspecies`, or `morph`. |

## Out of scope

These belong to a separate future feature, not this plan:

### Verification-time observation attributes

If verification grows to let humans record `lifeStage`, `sex`, `behaviour`, `count`, or `individual_id`, those are properties of the observation, not of the class. They belong as nullable columns on `Detection` (or on a sibling `DetectionAttributes` table), with their own facet-style filter UI and their own camtrap-dp export mapping.

Sketch for when that work starts:
- new columns on `Detection`: `life_stage`, `sex`, `behaviour`, `count`, `individual_id`, all nullable.
- verification UI gets small dropdowns next to the label picker.
- filter bar gets independent multiselects for each facet (not a tree).
- camtrap-dp export reads these columns into `lifeStage` / `sex` / `behavior`.
- optional 5-line helper: when the JSON loader sees `taxon_variant` matching a known life-stage vocabulary (`adult`, `subadult`, `juvenile`), prefill `Detection.life_stage` from it. Models that predict age give free verification metadata. The variant string itself stays on `Detection.label` so taxonomy still works.

This is a separate plan, not a step of this one. Keeping the two concerns split is the whole point.

### Group-to-species toggle on dashboards

If users complain about adult / juvenile splitting their per-species charts, add a project-level "group at species rank" toggle. Implementation: charts that today group by `Detection.label` join `label_taxonomy` and group by `COALESCE(taxon_species, Detection.label)`. Defer until asked for.

## Open questions to revisit at implementation time

- Should the `level` enum in `label_taxonomy` be enforced at the DB layer (CHECK constraint) or stay a free string? Currently free. Adding a CHECK is a separate cleanup, not part of this plan.
- Camtrap-dp export: when a row has a variant, does it go into the `scientificName` field or into `taxonRank` / a custom field? Decide at export-feature time; default for now is to use the full `model_class` string verbatim, same as today.
- Frontend tree icon for the variant level: whatever feels visually distinct from species, no firm requirement.

## Estimate

Roughly one focused day of work plus tests, assuming the filter tree component is genuinely recursive (it should be). The blast radius is narrow because most of the codebase is rank-agnostic. The risky bit is the tree builder; everything else is mechanical.
