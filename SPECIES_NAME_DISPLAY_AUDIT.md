# Species name display: common vs scientific — audit + handoff

Status: **investigation complete, no code written.** This documents the
verified findings for the "common name vs scientific name" display
feature so a fresh session can pick up without re-auditing.

Addresses two TODO items (same feature):
- TODO:12 — "display name option toggle for common name or scientific...
  show common if present, else scientific? ... simple or major? where do
  we store the toggle? or just default to common-if-present?"
- TODO:71 — "Global toggle to show labels as common name vs scientific
  name. Labels already carry both via label_taxonomy.display_name; just
  need a per-user or per-project UI switch. Requested by Simon."

---

## TL;DR

- **Feasible with no schema change, no migration, no backfill.** Both
  names already exist on every detection in the API today.
- **Common name = `Detection.label` / `LabelTaxonomy.name`.**
  **Scientific name = `Detection.display_name` / `LabelTaxonomy.display_name`.**
- A sub-audit during investigation wrongly concluded "common names are
  lost / not feasible." That is **wrong** — see "Correction" below.
- Refactor size: **medium — broad but shallow** (no data work; the cost
  is sweeping ~20 frontend render sites + ~5-6 backend name maps).
- Recommended storage: **global, per-user, localStorage** (pure display
  preference; instant toggle, no refetch). Not per-project, not DB.
- Current behavior is **inconsistent**: Edit/Verify shows scientific,
  dashboards already show common. A single preference would unify it.
- Three product decisions are still open (see "Open decisions"). They
  were NOT answered — ask the user before designing.

---

## Correction to a bad sub-audit conclusion

One investigation pass claimed the SpeciesNet common name is discarded
during taxonomy generation and therefore a toggle is "not feasible
without external data." Verified false:

- `backend/scripts/generate_taxonomy_csv.py:45` — `common_name = parts[6]`
  (token 6 of the SpeciesNet `uuid;class;order;family;genus;species;common_name`).
- `:48-54` — if the common name is empty or a duplicate, it falls back to
  the last non-empty taxon level (a Latin name), optionally UUID-disambiguated.
- `:59` — `"model_class": common_name` — so the common name **becomes**
  `model_class`.
- `backend/app/ml/taxonomy_db.py` `populate_taxonomy_from_csv` — `name =
  row["model_class"].lower()`, i.e. `LabelTaxonomy.name` **is** the common
  name (lowercased). `Detection.label` carries the same value.

So the common name is preserved as `label`/`name`. It is NOT in
`display_name` (that's the scientific name). TODO:71's "labels already
carry both via label_taxonomy.display_name" is right that both exist, but
mislabels where: common = `name`/`label`, scientific = `display_name`.

---

## What each field holds (verified)

| concept | field(s) | species example | rollup → genus example |
|---|---|---|---|
| **Common** | `Detection.label`, `LabelTaxonomy.name` | `leopard` | `panthera` (Latin — no common name) |
| **Scientific** | `Detection.display_name`, `LabelTaxonomy.display_name` | `P. pardus` (abbreviated binomial) | `Panthera` |
| Taxon ranks | `LabelTaxonomy.taxon_class/order/family/genus/species` | mammalia / carnivora / felidae / panthera / pardus | … (deeper ranks null on rollup) |

Key code:
- `backend/app/models/detection.py:61-63` — `label`, `label_confidence`, `display_name`.
- `backend/app/api/schemas/detection.py:49-50` — `display_name` documented as "Latin taxonomy display name".
- `backend/app/models/label_taxonomy.py:25-78` — columns: `name`, `display_name`, `taxon_*`, `level`, `is_custom`, `project_id`. **No `common_name` column** (none needed — `name` is it).
- `backend/app/ml/taxonomic_rollup.py:30-54` — `format_display_name_from_taxonomy_row()` builds `display_name`: genus+species → `"P. pardus"` (abbreviated genus); genus-only → `"Panthera"`; family-only → `"Felidae"`; etc. **This is the scientific-name formatter, and it abbreviates the genus.**

### How `display_name` is set (every write path)
- `backend/app/api/crud/detection.py` — `create_human_detection`, `create_observation`, `update_detection`: `display_name = tax.display_name if tax else capitalize(label)`. Category-only edits use the builtin taxonomy's display_name.
- `backend/app/ml/json_pipeline.py:378` — set from the resolved `(taxonomy_id, display_name)` during analysis load; unclassified → `category.capitalize()`.
- `backend/app/ml/postprocessing.py` — re-set on smoothing relabels; cleared to `None` on the exclusion sweep.

---

## The honest catch: common-name coverage is imperfect

- Species **with** a SpeciesNet common name → `label` is a real common
  name (`leopard`, `red fox`).
- **Rollups** (genus/family/…) and SpeciesNet entries whose common name
  was empty/duplicated → `label` is **already** a Latin taxon
  (`panthera`, `felidae`). There is no common name to show.

So "common if present, else scientific" is the only sensible behavior,
and it's close to what just showing `label` already gives — `label`
degrades to Latin exactly where no common name exists. The one rough edge:
a dedup-fallback species can have `label` = a bare epithet (`pardus`)
where scientific would show the binomial (`P. pardus`).

No reliable boolean "has common name" flag exists. Heuristic if needed:
common-name-present ≈ `label` is not equal to any of its `taxon_*` rank
values. Simpler alternative: just render Title-cased `label` in common
mode and accept the graceful Latin degradation.

---

## Current behavior is already inconsistent

- **Scientific (display_name) is shown** on the Edit/Verify page, crop
  cards, detail modals, filter chips. `getDetectionDisplayName()` prefers
  `display_name` (see frontend resolver below).
- **Common (label) is already shown** on dashboard charts via
  `normalizeLabel(species)` (species = the label).

So the app mixes the two today. A single global preference would unify
this — a real side-benefit, and a reason this is worth doing.

---

## Frontend: where names render (≈20 sites, semi-centralized)

Chokepoint helpers (most sites route through these):
- `frontend/src/lib/detection-utils.ts:64-72` — `getDetectionDisplayName(d)`:
  `display_name → label → category`. Used by CropCard, DetectionDetailModal,
  FileVerificationPanel.
- `frontend/src/utils/labels.ts:5-7` — `normalizeLabel()`: underscores→spaces,
  capitalize. Used by dashboard charts, site/deployment info sheets, species
  picker, save-results modal.
- `frontend/src/hooks/useLabelOptions.ts:32-38` — `getDisplayName()` builds
  `LabelOption.displayName` from `display_name` (fallback capitalized label).
- `frontend/src/components/verify/LabelPicker.tsx` — `formatLabel()` (:37-40),
  `TaxonomyCaption` (:42-58); renders `displayName` (scientific) primary,
  `label` (common) as the caption (:268, :334-464).

Backend-resolved maps the frontend just renders:
- `display_labels: Record<taxonomy_id, string>` on FileSummary/EventSummary
  → used by FileCard, EventCard, VerifyFilterBar, MapFilterBar. Built in
  `backend/app/api/crud/file.py:554-572` as `d.display_name or d.label or
  d.category` (i.e. **scientific today**).
- Per-detection: `display_name`, `label`, `label_taxonomy_id`,
  `neighbor_top_display_name` (DetectionSummary), `current_display_name` /
  `suggested_display_name` (CohortItem).
- Stats: `LabelProgressRow.display_name`, `ClassMetrics.display_name`,
  confusion-matrix `class_display_names`, top-species `display_name`/`label`.

Full per-component list (from the frontend audit) — the HIGH-traffic ones:
CropCard, EventCard (VerifyView.tsx:605), FileCard (:157),
DetectionDetailModal (:343,582,606,717), LabelPicker (:268,334-464),
ConfusionMatrix (:140), PerClassPerformanceTable (:112). MEDIUM/LOW: filter
bars, dashboard charts, site/deployment info sheets, map popups
(SitePopup.tsx:81 shows `sp.label` only), SpeciesPicker, SaveResultsModal.

Existing display-settings precedent to copy: `ObservationsSettings.tsx`
persists view settings to localStorage (key pattern
`addaxai:observations-settings-{projectId}`). The naming preference should
be **global**, not per-project.

---

## Backend: serialization points that pre-resolve a name

These currently emit `display_name` (scientific) and would need to also
carry the common name (or both) so the client can switch without refetch:
- `backend/app/api/crud/file.py:554-572` — `display_labels` (file + event summaries).
- Top-species (site/deployment) — `display_name` + `label`.
- Statistics: verification-by-label, per-class metrics, confusion-matrix labels.
- Map species counts (`SpeciesObservationCount`) already send `label`; popups render it.

Note: per-detection responses already include BOTH `label` and
`display_name`, so detection-level sites need no backend change — only the
aggregated maps do.

---

## Refactor size: medium (broad, shallow)

No schema/migration/backfill; both values already exist. Cost is breadth:
1. **Frontend** — add a global preference (context/hook + localStorage) and
   a single `resolveSpeciesName({label, display_name, category}, mode)`
   resolver. Route the ~20 sites through it (most already pass through the 3
   helpers above, so realistically ~5-8 real edit points + the helpers).
2. **Backend** — extend the ~5-6 aggregated name maps to carry the common
   name too (or both names), so toggling doesn't need a refetch.
3. **Risk** — missing a render site → inconsistent naming. Needs a careful
   grep sweep, not deep logic. Add a small test/checklist.

Not a one-liner, not major. Call it broad-but-shallow.

---

## Recommended design (KISS)

- One global per-user preference: **"Species names: Common / Scientific"**
  (familiar iNaturalist/Merlin/eBird mental model).
- "Common" = Title-cased `label`, auto-degrading to scientific where no
  common name exists. "Scientific" = `display_name`.
- Store in **localStorage, global** (not per-project, not DB). Instant
  toggle, no server round-trip.
- One frontend resolver + context; backend maps carry both names.
- Settling the inconsistency (dashboards vs Edit page) falls out for free.

---

## Open decisions — NOT yet answered, ask the user first

1. **Toggle vs single default?**
   (a) Global toggle [matches TODO:71]; (b) no toggle, always
   common-if-present [simplest]; (c) no toggle, keep scientific [do nothing].
2. **If toggle, default mode?** Common (more readable; flips today's
   scientific default on the Edit page) vs Scientific (keeps current).
3. **Scientific format?** Keep abbreviated (`P. pardus`, current) vs full
   binomial (`Panthera pardus`; both parts are stored, slightly larger sweep).

Other clarifications worth confirming:
- Scope: app-wide vs specific pages (TODO says "global").
- Do exports follow the preference, or only the UI? (Exports have their
  own name columns — CSV uses taxon ranks + `classification_label`;
  recognition.json/CamTrap are fixed formats. A *display* toggle is a UI
  concern; recommend exports stay independent unless asked.)

---

## Key files (quick index)

- `backend/scripts/generate_taxonomy_csv.py` — common name → model_class.
- `backend/app/ml/taxonomy_db.py` — CSV → LabelTaxonomy rows.
- `backend/app/ml/taxonomic_rollup.py:30-54` — scientific-name formatter (abbreviated).
- `backend/app/models/label_taxonomy.py`, `backend/app/models/detection.py` — fields.
- `backend/app/api/schemas/detection.py:49-50` — display_name = Latin.
- `backend/app/api/crud/file.py:554-572` — `display_labels` builder.
- `frontend/src/lib/detection-utils.ts:64-72` — `getDetectionDisplayName`.
- `frontend/src/utils/labels.ts:5-7` — `normalizeLabel`.
- `frontend/src/hooks/useLabelOptions.ts`, `frontend/src/components/verify/LabelPicker.tsx` — option formatting.
- `frontend/src/components/verify/ObservationsSettings.tsx` — localStorage settings precedent.
