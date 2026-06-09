# Plan: Labels + Observations verification rebuild

## Context

AddaxAI's verification is one page with a "View as: observations / media / events"
dropdown. All three views write the same truth (`Detection.verified`), but they
conflate two genuinely different jobs, which confuses users:

1. Label cleanup: fix the AI's per-detection species labels (crop grid, similarity
   sort, cohorts). Per detection. Feeds the confusion matrix.
2. Ecological record: record what was actually there as species + count per event,
   including individuals no single frame shows (the "5 elephants" / hidden-in-video
   case). Per event. This is what GBIF / Camtrap-DP / every other platform calls an
   "observation".

Today the count of individuals is never stored as a number; it is derived by
counting detection rows (MaxN), and a "hidden individual" is faked as a box-less
`Detection` row anchored to the video's best frame (`frame_number =
best_frame_number`), which is a lie that pollutes per-frame data and is dropped by
the recognition.json export. The word "observation" is also used backwards: in the
UI it means a per-detection box, but in the data (`EventObservation`) and the
standards it means the event-level count.

This rebuild splits verification into two pages, fixes the terminology to match the
standard, and replaces the fake-frame hack with an explicit human-authoritative
count. The count is shaped to Darwin Core (`organismQuantity`) so it also serves
marine MaxN later with no redesign. Outcome: a clear two-job mental model, honest
data, a conformant Camtrap-DP export, and a future-proof count.

A separate read-only audit of the current behavior lives in
`OBSERVATION_COUNT_AUDIT.md`.

## Decisions locked (from discussion)

- Two pages: **Labels** (today's observations/crop view) and **Observations**
  (today's events view, with a count editor; media view dropped).
- Reclaim "observation" = event-level species:count (Camtrap-DP / DwC sense). The
  per-detection crop machinery renames to "labels".
- Count model: `EventObservation.max_n` stays (AI/box-derived MaxN). Add
  `human_count` (nullable). Effective count = `human_count ?? max_n`, human
  authoritative for stats and exports. Shaped like DwC `organismQuantity`; no
  quantity-type enum yet (YAGNI).
- Retire the box-less fake-frame flow entirely (no backward compat, no users).
- Explicit `Event.verified` (human signed off the counts) distinct from
  `Detection.verified` (label confirmed).
- Count applies to animal species, person, and vehicle; blanks have none.
- Confusion matrix / per-class performance unchanged (stays on `Detection.verified`
  + `original_label`, fed by the Labels page).
- Exports: add `count` to flat CSV; fix Camtrap-DP to emit one conformant
  event-level row per species with the effective count; recognition.json stays
  geometry-only.
- Verification progress headline becomes event-level; file/detection kept secondary.
- Full rename incl. DB: honest finding from the inventory is that no existing DB
  column is misnamed, so there are no column renames; the only schema changes are
  the two new columns. The rename is entirely code / API / UI / docstrings.
- Execute in four phased checkpoints.
- Reuse the existing events verification view as the Observations page, do not
  rebuild: keep the filter bar, the `EventCard` gallery grid, and the
  `EventDetailModal` (gallery + tools on the left, panel on the right). The only
  net-new frontend is turning the right-hand panel's read-only grouped "x N" species
  summary into an editable, event-scoped count list plus a "Verify event" action.

## Final terminology scheme

| Concept | Names | Action |
|---|---|---|
| Per-detection box + its AI label (crop work) | `Detection`, frontend "labels" page | KEEP `Detection`; RENAME frontend "observations" view to "labels" |
| Event-level species:count (the standard "observation") | `EventObservation`, `event_observations`, `observations.csv/xlsx`, `/api/.../observations` export, `File.observation_type` | KEEP (already correct) |
| The crop-sort API + service + schemas currently called "observations" | `routers/observations.py`, `services/observation_service.py`, `schemas/observation.py`, `api/observations.ts`, `observationsApi`, `/api/projects/{id}/observations/{sort,search,cohorts,stats}`, `obs_*` params | RENAME to `labels` |
| Box-less fake-frame mechanism | `create_observation`, `DetectionCreateObservation`, `POST /api/detections/observation`, binoculars/`N` UI, `detectionsApi.createObservation` | RETIRE |

Note the trap: `routers/observations.py` (crop sort, RENAME) is different from the
`/api/.../observations` CSV/XLSX export endpoint and `observations_csv.py` /
`observations_xlsx.py` (event-level, KEEP). Do not rename the export side.

## Phase 1: rename + two-page split scaffold

Goal: "observations" crop machinery becomes "labels"; one Edit page becomes two
routes; nothing changes behaviorally yet (count model and box-less retirement come
later).

Rename (exhaustive, from the inventory):
- Frontend files: `ObservationsTab.tsx`→`LabelsTab.tsx`,
  `ObservationsWelcomePopover`/`ObservationsKeyboardPopover`/`ObservationsSettings`/
  `observationsViewOptions.ts` → `Labels*`; `api/observations.ts`→`api/labels.ts`
  (`observationsApi`→`labelsApi`, `ObservationsProgress*`→`LabelsProgress*`).
- Frontend identifiers/strings: `VerifyViewMode` value `"observations"`→`"labels"`,
  the view-selector option label, `obs_*` URL params → `lbl_*`, query keys
  (`observations-stats`→`labels-stats`), localStorage keys
  (`addaxai:observationsSettings`/`…WelcomeDismissed`→`labels…`), the ~35 user-facing
  strings that say "observation(s)" meaning a crop/label (toasts, headings like
  `FileVerificationPanel` "Observations"→"Labels", tooltips, keyboard-help rows,
  `CropGrid` copy), and the imports of `observationsApi` in `SuggestionsToolbarPill`,
  `DetectionDetailModal`, `LabelsTab`.
- Backend: `routers/observations.py`→`routers/labels.py` (router prefix paths
  `/observations/{sort,search,cohorts,stats}`→`/labels/...`, tag `observations`→
  `labels`); `services/observation_service.py`→`services/label_service.py`;
  `schemas/observation.py`→`schemas/label.py` (`ObservationSort`→`LabelSort`,
  `ObservationFilters`→`LabelFilters`); update `main.py` router include and all
  imports.
- Do NOT touch: `EventObservation`, `event_observations`, `File.observation_type`,
  `DetectionCreateObservation` (retired in Phase 2, not renamed), the
  `observations.csv/xlsx` export router/files, `test_max_n.py` event-obs names.

Two-page split:
- Routes (`App.tsx`): project `…/edit` → `…/labels` + `…/observations`; folder-run
  `…/edit` step → `…/labels` + `…/observations` steps. Add a project index redirect
  so `/projects/:id/edit` (or the bare project) lands on `…/labels`.
- New page wrappers, both thin (they mount existing components, no rebuild):
  `pages/LabelsPage.tsx` mounts `LabelsTab` (today's observations/crop view);
  `pages/ObservationsPage.tsx` mounts the existing events view verbatim (filter bar +
  `EventCard` grid + `EventDetailModal`). Folder-run `FolderRunLabelsStep.tsx` +
  `FolderRunObservationsStep.tsx` wrap the same. In Phase 1 the Observations page is a
  pure move of the current events view to `/observations`; the count editor is added in
  Phase 3.
- Drop the **media** view: the standalone `FilesTab` file grid goes away. A single
  image is an event of one, reviewed by opening its event. `FileDetailModal`,
  `FileVerificationPanel`, `AnnotationCanvas`, `LabelPicker` are retained and reused
  inside the event flow (already shared today).
- Remove the "View as" selector from `VerifyFilterBar`; the filter bar otherwise
  stays on both pages.
- Nav / breadcrumbs / steps: `Sidebar` single "Edit" → "Labels" + "Observations";
  `lib/breadcrumbs.ts` `PROJECT_PAGE_LABELS` and `FOLDER_RUN_STEP_LABELS`; folder-run
  `StepProgress` STEPS + `stepFromPath`.
- DRY: extract the three things `VerifyView` currently holds that both pages need:
  `useDebouncedValue` (also duplicated in `FilesTab`) → `hooks/useDebouncedValue.ts`;
  filter parse/serialize (`filtersFromSearchParams`/`filtersToSearchParams`) →
  `lib/verify-filters.ts`; the shared project/count/verification-stats queries →
  `hooks/useVerifyProjectData.ts`. `VerifyView.tsx` is then retired or reduced to
  shared chrome.

Checkpoint: app builds, both pages render the existing behavior under new names,
no "observations"-as-crop strings remain (`grep -ri observation frontend/src` only
hits event-level/Camtrap usages). `tsc` and `ruff` clean.

## Phase 2: count model + migration + retire box-less (backend)

Schema (`models/`):
- `EventObservation`: add `human_count: int | None` (nullable).
- `Event`: add `verified: bool` (not null, server_default false).

Migration (one Alembic file, follow the repo style; use plain `op.add_column`, not
batch, per DEVELOPERS.md; guard any drop with a presence check):
1. Add the two columns.
2. Backfill `events.verified` from the current derived rule (all MaxN-frame files
   verified; for blank events, any file verified) via one `UPDATE ... CASE`.
3. Backfill `event_observations.human_count = max_n` for every event+species that
   currently has box-less verified detections (bbox NULL + verified), so the counts
   the user already entered survive.
4. Delete the box-less detection rows (bbox_x/y/width/height all NULL). This is a
   sanctioned code path (migration), not an ad-hoc script. MaxN for affected events
   is unaffected because their count was copied to `human_count` in step 3.

Effective count (single source of truth, DRY):
- Add `effective_count(obs) = obs.human_count if obs.human_count is not None else
  obs.max_n` in `crud/event_observation.py`; use it everywhere a count is surfaced.
- Add `set_human_count(db, event_observation_id, count | None)` and a way to add a
  human-only species row (max_n=0, no `max_n_file_id`) so the user can record a
  species the AI missed entirely (this replaces the box-less "add individual" use
  case). Both clear `Event.verified` for the affected event (sign-off is now stale).
- `Event.verified` reset rule: clear it when the event's effective observation set
  changes, i.e. (a) `set_human_count` edits, and (b) inside
  `calculate_max_n_for_event` when the recomputed species/`max_n` set differs from
  the stored one (so a Labels-page relabel/add/delete that changes MaxN un-signs the
  counts, but a pure detection-verify that changes nothing does not). This is the one
  genuinely fiddly bit; implement the diff in the existing delete-and-recreate path.

Event verification becomes stored:
- Replace the derived `is_verified` (in `crud/event.py`) with reads of
  `Event.verified`; update the verified/unverified event filter to `Event.verified`.
- New endpoint `PATCH /api/events/{id}/verify` ({verified: bool}) sets the flag.

Retire box-less (backend):
- Delete `create_observation` (`crud/detection.py`), `DetectionCreateObservation`
  (`schemas/detection.py`), the `POST /api/detections/observation` route
  (`routers/detections.py`) and its import. No deprecation shim.

Checkpoint: migration up/down on a copy of a real DB; `ruff` + `pytest` green
(tests updated in Phase 4, but smoke the migration here).

## Phase 3: Observations page count UI + remove binoculars (frontend)

Additive to the reused events view, not a rebuild. The layout (filter bar, `EventCard`
gallery grid, `EventDetailModal` with gallery + tools left, panel right) is already in
place from Phase 1.

- Right-hand panel of `EventDetailModal`: replace the read-only grouped "x N" species
  summary with an editable, **event-scoped** count list. Per species: a stepper seeded
  from the effective count, with the AI MaxN shown as context ("AI saw 2"); an
  add-missed-species control (creates a human-only `EventObservation`). Reuse the
  existing species-row styling from `FileVerificationPanel`; the difference is it
  reads/writes the event's counts (across all files), not the selected file's
  detections. The left side (gallery, `AnnotationCanvas`, optional box overlay) is
  untouched; heavy per-detection editing stays on the Labels page.
- A "Verify event" action calls the new `PATCH /api/events/{id}/verify`.
- Remove the binoculars button + `N` shortcut + `observeMutation` from
  `EventDetailModal` and `FileDetailModal`; remove `detectionsApi.createObservation`.
- `EventCard` / filmstrip badges show the effective count; `MaxNFrame` carries
  `effective_count`, `EventSummary` carries `verified`.
- Observations-page progress rides on `Event.verified`; Labels-page progress stays on
  detection/file verified.

Checkpoint: open an event, set "3 deer", verify the event; reopen and it persists;
relabelling a detection that changes the species set clears the event sign-off.

## Phase 4: exports + metrics + tests

- Flat CSV (`crud/export.py`): add a `count` column carrying the effective count for
  the detection's event+label.
- Camtrap-DP (`crud/export.py`): emit one `observationLevel=event` row per species per
  event with `count = effective_count` (replaces today's `count=1`-per-detection,
  which is non-conformant). Keep media-level box rows.
- recognition.json: no change (geometry-only; documented limitation).
- Dashboard (`crud/statistics.py`): observation total →
  `sum(coalesce(human_count, max_n))`.
- Verification progress: primary = events verified / total events; keep file/detection
  counts as secondary.
- Performance / confusion matrix (`crud/performance.py`): no change.
- Tests: add `human_count` / effective-count tests (`test_max_n.py`); `Event.verified`
  endpoint + filter tests (`test_events.py`); CSV `count` column and Camtrap-DP
  event-row tests (`test_observations_csv.py` and the camtrap export test);
  dashboard effective-count test; update any test asserting the old derived
  `is_verified` or the retired box-less endpoint.

Checkpoint: full `pytest` + `ruff` green; `tsc` + `vite build` clean; manual export
of CSV and Camtrap-DP shows the count.

## Key design decisions and the one open risk

- Two verbs, two flags: `Detection.verified` (label confirmed, Labels page) and
  `Event.verified` (counts signed off, Observations page) are independent.
- Replacing box-less rows with a count keeps AI boxes intact, so crops, embeddings,
  recognition.json, depth, and Wildbook are unaffected (they read `bbox_*` and never
  check `verified`).
- The fiddly bit is the `Event.verified` auto-reset (above). Recommended rule stated;
  if it proves noisy in practice we can fall back to "only `set_human_count` clears
  it" and let relabels not touch the sign-off. Flagging now, will confirm at the
  Phase 2 checkpoint.

## Future-proofing (marine / drone) — the YAGNI line

Build camera-trap only. Two free, non-precluding choices baked in now: the count is
generic and DwC-shaped (`effective_count` maps to Camtrap-DP `count` / DwC
`organismQuantity`; `max_n` is literally the marine MaxN field already), and "event"
is treated as an abstract grouping unit, not hardcoded as a camera burst. Deferred,
with where each slots in: a `quantity_type` enum (individuals / MaxN / MeanCount) on
the count; per-individual length/biomass as optional observation attributes (marine);
area/density + a moving-sensor deployment + spatial dedup (drone). Known divergence to
note and stop at: a drone "deployment" is a flight over an area, not a fixed lat/lon
site, which conflicts with the current required-coordinates site model
(`SITE_LOCATION_CLEANUP.md`); do not generalize the site model now.

## Representative files

- Frontend rename + split: `App.tsx`, `components/verify/*` (Labels* renames,
  `EventDetailModal`, `FileDetailModal`, `VerifyFilterBar`, `FileVerificationPanel`,
  `CropGrid`), `api/labels.ts`, `lib/breadcrumbs.ts`, `components/layout/Sidebar.tsx`,
  `components/folder-run/StepProgress.tsx`, new `pages/LabelsPage.tsx` /
  `pages/ObservationsPage.tsx` / folder-run steps, extracted `hooks/` + `lib/`.
- Backend rename: `routers/labels.py`, `services/label_service.py`,
  `schemas/label.py`, `main.py`.
- Count model: `models/event_observation.py`, `models/event.py`, one
  `alembic/versions/*` migration, `crud/event_observation.py`, `crud/event.py`,
  `schemas/event.py`, `routers/events.py`, `crud/detection.py`,
  `schemas/detection.py`, `routers/detections.py`.
- Exports / metrics: `crud/export.py`, `crud/statistics.py`.
- Tests under `backend/tests/`.

## Verification

- Per phase: `cd frontend && npx tsc -b && npx vite build`; `cd backend && ruff check
  app tests && pytest`.
- Migration: apply on a copy of a populated `~/AddaxAI/addaxai.db`, confirm
  `event_observations.human_count` and `events.verified` exist, box-less rows gone,
  counts preserved; test `downgrade`.
- Manual (run the app): Labels page does crop relabel / similarity sort as before;
  Observations page lets you set "3 deer", add a missed species, and verify the event;
  dashboard total and a CSV + Camtrap-DP export reflect the human count; recognition.json
  still exports boxes.
- First execution step (after approval): copy this plan to the repo root as
  `OBSERVATION_REBUILD_PLAN.md` for reproduction, per request.
