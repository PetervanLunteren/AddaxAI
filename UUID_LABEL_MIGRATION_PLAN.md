# UUID-based label matching migration plan

## Why

Labels are matched by string name everywhere: `Detection.label`, `EventObservation.label`, `project.excluded_classes`, rollup, exclusion, statistics, filtering. This causes bugs from name normalization (apostrophes, casing), rollup creating ancestor labels not in the exclusion list, and fragile string matching across 100+ locations. The `label_taxonomy` table already has UUIDs, and `Detection` already has a `label_taxonomy_id` FK. The infrastructure is half there.

## Architecture

```
JSON (model output)          DB (all matching via UUID)
─────────────────           ──────────────────────────
classification_categories    label_taxonomy table
  {"1": "grevys zebra"}       id (UUID), name, display_name, taxonomy fields
                              
Phase 6 maps names ────────► Detection.label_taxonomy_id (authoritative FK)
to taxonomy UUIDs             Detection.label (denormalized string for display)
                              Detection.display_name (denormalized for display)

excluded_classes              list[str] of taxonomy UUIDs (not name strings)

EventObservation              label_taxonomy_id FK (not label string)
                              label string kept as denormalized display

CRUD queries                  JOIN/filter on label_taxonomy_id, never on label string
```

**Core principle**: JSON processing stays string-based (model output, we don't control it). The UUID boundary is Phase 6 (DB load). After that point, all matching uses `label_taxonomy_id`.

## What changes and what stays

### Stays the same
- JSON format (`classification_categories`, `classifications`, `classification_category_descriptions`)
- `apply_label_exclusion_to_results()` in `label_exclusion.py` (operates on JSON strings)
- `apply_taxonomic_rollup_to_results()` / `rollup_single_detection()` in `taxonomic_rollup.py` (operates on JSON strings)
- `smoothing_script.py` (subprocess, JSON in/out)
- `trim_classification_results()` in `json_utils.py`
- `Detection.label` column (kept as denormalized display string)
- `Detection.display_name` column (kept as denormalized display string)

### Changes

| Current (string-based) | New (UUID-based) |
|------------------------|------------------|
| `Detection.label_taxonomy_id` nullable, set post-hoc by `link_detections_to_taxonomy()` | Set inline during Phase 6 DB load, authoritative for all matching |
| `EventObservation.label` string, unique constraint `(event_id, label)` | New `label_taxonomy_id` FK, unique constraint `(event_id, label_taxonomy_id)`, `label` kept nullable for display |
| `project.excluded_classes` = `["lion", "zebra"]` (name strings) | `["uuid-1", "uuid-2"]` (taxonomy UUIDs) |
| CRUD queries: `COALESCE(Detection.label, Detection.category)` | `Detection.label_taxonomy_id` (all detections linked, including unclassified via builtin "animal"/"person"/"vehicle") |
| Statistics: `LabelTaxonomy.name == EventObservation.label` (string join) | `LabelTaxonomy.id == EventObservation.label_taxonomy_id` (FK join) |
| Label tree: FK-first with string-match fallback | FK only, no fallback |
| Event filtering: `effective_label.in_(label_name_strings)` | `Detection.label_taxonomy_id.in_(taxonomy_uuids)` |
| Rollup exclusion: `excluded_names = frozenset(name strings)` | `excluded_classes` UUIDs resolved to names at boundary for JSON processing |
| Taxonomy populated after Phase 6 | Taxonomy populated before Phase 6 |

## Detailed changes by file

### Alembic migration (new file)

`backend/alembic/versions/YYYYMMDD_uuid_label_matching.py`

- `event_observations` table:
  - Add `label_taxonomy_id` column: `String(36)`, FK to `label_taxonomy.id`, `ondelete="SET NULL"`, nullable
  - Drop unique constraint `uq_event_obs_event_label` on `(event_id, label)`
  - Add unique constraint `uq_event_obs_event_taxonomy` on `(event_id, label_taxonomy_id)`
  - Add index `idx_event_obs_label_taxonomy` on `label_taxonomy_id`
  - Make `label` column nullable (was `NOT NULL`)
- Data migration (best effort, can be empty since no users):
  - Resolve existing `project.excluded_classes` name strings to taxonomy UUIDs
  - Backfill `event_observations.label_taxonomy_id` from name match

### Models

**`backend/app/models/event_observation.py`**
- Add `label_taxonomy_id` FK column to `LabelTaxonomy`
- Add `label_taxonomy` relationship
- Change `label` to nullable
- Update unique constraint: `(event_id, label_taxonomy_id)`
- Update indexes
- Add `LabelTaxonomy` to `TYPE_CHECKING` imports

**`backend/app/models/label_taxonomy.py`**
- Add `observations` back-reference relationship
- Add `EventObservation` to `TYPE_CHECKING` imports

### Pipeline ordering

**`backend/app/workers/detection_worker.py`**

Move taxonomy population from after Phase 6 to between Phase 5 and Phase 6:
```
Current:  Phase 5 (merge) → Phase 6 (DB load) → populate_taxonomy → link_detections
New:      Phase 5 (merge) → populate_taxonomy + ensure_builtins → Phase 6 (DB load with inline linking)
```

The `link_detections_to_taxonomy()` call can stay as a defensive fallback but should be a no-op if Phase 6 does its job.

### taxonomy_db.py

**`backend/app/ml/taxonomy_db.py`**

New function:
```python
def batch_resolve_taxonomy_ids(
    label_names: list[str],
    model_id: str | None,
    project_id: str,
    db: Session,
) -> dict[str, str]:
    """Return {lowercase_name: taxonomy_id} for a batch of label names.
    
    Priority: model-level → custom → builtin.
    Single query with OR filter instead of N+1.
    """
```

Update `ensure_builtin_labels()`:
```python
def ensure_builtin_labels(db: Session) -> dict[str, str]:
    """Return {"animal": uuid, "person": uuid, "vehicle": uuid}."""
```

### Phase 6: json_pipeline.py

**`backend/app/ml/json_pipeline.py`**

`load_json_to_database()` changes:
- New parameters: `taxonomy_ids: dict[str, str]` (pre-resolved name→UUID mapping), `builtin_taxonomy_ids: dict[str, str]` (category→UUID)
- After extracting `classification_categories` from JSON, build the full name→UUID mapping via `batch_resolve_taxonomy_ids()`
- When creating each Detection:
  - If classified (has label): set `label_taxonomy_id = taxonomy_ids[label_name]`
  - If unclassified: set `label_taxonomy_id = builtin_taxonomy_ids[category]`
  - Set `display_name` from taxonomy lookup at the same time
- This eliminates the need for post-hoc `link_detections_to_taxonomy()`

### Phase 7: postprocessing.py

**`backend/app/ml/postprocessing.py`**

`run_postprocessing_for_deployment()`:
- `project.excluded_classes` now contains UUIDs. Resolve to name strings for JSON processing:
  ```python
  excluded_rows = db.query(LabelTaxonomy.name).filter(LabelTaxonomy.id.in_(project.excluded_classes)).all()
  excluded_names = frozenset(row[0].lower() for row in excluded_rows)
  ```
- Rest of rollup/exclusion stays string-based (JSON processing)

`update_database_from_smoothed_results()`:
- Build `taxonomy_name_to_id` mapping at function start via `batch_resolve_taxonomy_ids()`
- When updating a detection, set `label_taxonomy_id` alongside `label`:
  ```python
  db_det.label_taxonomy_id = taxonomy_name_to_id.get(new_label.lower())
  ```
- Final exclusion sweep: check `label_taxonomy_id in excluded_taxonomy_ids` instead of `label.lower() in excluded_names`
- New parameter: `excluded_taxonomy_ids: set[str] | None` (replaces `excluded_classes: list[str]`)

`reload_raw_classifications_from_json()`:
- Same pattern: resolve excluded UUIDs to names for JSON processing, use UUIDs for DB operations

### CRUD modules

**`backend/app/api/crud/event_observation.py`**

`calculate_max_n_for_event()`:
- Group by `Detection.label_taxonomy_id` instead of `COALESCE(Detection.label, Detection.category)`
- When creating EventObservation rows, set `label_taxonomy_id` (from the detection's FK)
- Set `label` string from a pre-fetched `{taxonomy_id: name}` mapping (denormalized)
- Update `get_max_n_frames()` to include `label_taxonomy_id` in returned dict

**`backend/app/api/crud/event.py`**

`_apply_event_filters()`:
- `labels` parameter becomes list of taxonomy UUIDs
- Filter: `Detection.label_taxonomy_id.in_(taxonomy_uuids)` instead of `COALESCE(label, category).in_(name_strings)`

`get_events_by_project()`:
- Collect `label_taxonomy_id` values instead of label strings
- `EventSummary.labels` becomes list of taxonomy UUIDs
- `display_labels` maps taxonomy UUIDs to display names

`get_filter_options()`:
- Replace `COALESCE(Detection.label, Detection.category)` with `Detection.label_taxonomy_id`
- Return taxonomy UUIDs with display name mapping

**`backend/app/api/crud/label_tree.py`**

`build_label_filter_tree()`:
- Replace `COALESCE(Detection.label, Detection.category)` with `Detection.label_taxonomy_id`
- Remove string-match fallback path (lines ~109-141)
- Leaf node IDs become taxonomy UUIDs
- `all_leaf_ids` becomes list of taxonomy UUIDs

**`backend/app/api/crud/statistics.py`**

All stats functions:
- Replace `LabelTaxonomy.name == EventObservation.label` joins with `LabelTaxonomy.id == EventObservation.label_taxonomy_id`
- Replace `COALESCE(Detection.label, Detection.category)` patterns with `Detection.label_taxonomy_id`

**`backend/app/ml/inference/similarity_script.py`**

- Replace `COALESCE(d.label, d.category) IN (...)` with `d.label_taxonomy_id IN (...)`

### API / schemas

**`backend/app/api/schemas/event.py`**
- `MaxNFrame`: add `label_taxonomy_id: str | None`
- `EventSummary.labels`: semantically becomes taxonomy UUIDs (type stays `list[str]`)
- `EventFilterOptions`: add `display_labels: dict[str, str]` for UUID→display mapping
- `EventFilterOptions.label_event_counts`: keys become taxonomy UUIDs

**`backend/app/api/routers/projects.py`**
- Project creation: call `populate_taxonomy_from_csv()` when classification model is selected, then resolve geofence exclusions to UUIDs
- Geofence endpoint: keeps returning name strings (for display). Frontend resolves to UUIDs before saving to `excluded_classes`.

**`backend/app/api/routers/events.py`**
- Pass taxonomy UUIDs to CRUD (from frontend filter params)

### geofence.py

**`backend/app/ml/geofence.py`**
- Add `compute_excluded_class_ids(model_dir, country, state, db) -> list[str]`:
  1. Call existing `compute_excluded_classes()` for name strings
  2. Resolve names to taxonomy UUIDs via DB query
  3. Return UUID list

### Frontend

**`frontend/src/api/types.ts`**
- `MaxNFrame`: add `label_taxonomy_id?: string`
- `EventSummary.labels`: semantically becomes taxonomy UUIDs
- `EventFilterOptions`: add `display_labels?: Record<string, string>`
- `EventFilterOptions.label_event_counts`: keys become UUIDs

**`frontend/src/pages/SettingsPage.tsx`**
- `excluded_classes` sends/receives UUIDs
- Display names resolved from taxonomy data

**`frontend/src/pages/VerifyPage.tsx`**
- Event labels are UUIDs, display from `display_labels` mapping

**`frontend/src/components/verify/FilterPanel.tsx`** / **`LabelFilterModal.tsx`**
- Filter selection uses taxonomy UUIDs

**`frontend/src/components/verify/LabelPicker.tsx`**
- Options include taxonomy IDs, selection returns UUID

**`frontend/src/components/taxonomy/SpeciesSelectionModal.tsx`**
- Excluded species identified by taxonomy UUID

**`frontend/src/components/projects/CreateProjectDialog.tsx`**
- `excluded_classes` as UUIDs

### Tests to update

- `backend/tests/api/test_events.py` - label filters use UUIDs
- `backend/tests/api/test_label_tree.py` - leaf IDs are UUIDs
- `backend/tests/api/test_label_taxonomy_fk.py` - EventObservation FK
- `backend/tests/integration/test_postprocessing_pipeline.py` - UUID-based exclusion
- `backend/tests/integration/test_images_only_pipeline.py` - Phase 6 inline linking
- `backend/tests/integration/test_event_generation.py` - MaxN by taxonomy_id
- `backend/tests/ml/test_postprocessing.py` - UUID sweep
- `backend/tests/ml/test_taxonomic_rollup.py` - no changes (JSON-level)
- `backend/tests/unit/test_label_exclusion.py` - no changes (JSON-level)
- `backend/tests/test_max_n.py` - MaxN groups by taxonomy_id
- New: `backend/tests/integration/test_uuid_label_matching.py` - end-to-end

## Implementation sequence

1. Alembic migration + model changes (`event_observation.py`, `label_taxonomy.py`)
2. `taxonomy_db.py` (`batch_resolve_taxonomy_ids`, updated `ensure_builtin_labels`)
3. Worker reordering: taxonomy before Phase 6 (`detection_worker.py`)
4. Phase 6 inline taxonomy resolution (`json_pipeline.py`)
5. Phase 7 UUID-based updates + exclusion sweep (`postprocessing.py`, `postprocessing_worker.py`)
6. `geofence.py` (`compute_excluded_class_ids`)
7. CRUD: `event_observation.py`, `event.py`, `label_tree.py`, `statistics.py`
8. `similarity_script.py`
9. API schemas + routers (`schemas/event.py`, `routers/projects.py`, `routers/events.py`, `routers/detections.py`)
10. Frontend (`types.ts`, `SettingsPage`, `VerifyPage`, `FilterPanel`, `LabelFilterModal`, `LabelPicker`, `SpeciesSelectionModal`, `CreateProjectDialog`)
11. Tests

## Important notes

- **No backward compatibility needed** (CONVENTIONS.md rule 6: no users yet)
- **JSON files on disk are untouched** (they are model output, the ground truth)
- **Verified detections are never overwritten** (DEVELOPERS.md rule)
- **Delete old DB and re-run `alembic upgrade head`** after migration (simplest path, no users)
- The `EventObservation.label` model change already started in this session and was reverted. Start fresh.

## Session plan document

The working plan from this session is at:
`/Users/peter/.claude/plans/polymorphic-sauteeing-pebble.md`

## Files touched in this session (uncommitted)

Before starting the UUID migration, commit or review these pending changes:
- Rollup ancestor exclusion check (`taxonomic_rollup.py`, `postprocessing.py`)
- Final exclusion sweep (`postprocessing.py`)
- Species selection modal fix (`SpeciesSelectionModal.tsx`)
- Geofence overwrite removal (`routers/projects.py`)
- Display name fixes (`taxonomy_db.py`, `detection-utils.ts`, several verify components)
- Debug logging in projects router (can be removed)
- DEVELOPERS.md rule addition
