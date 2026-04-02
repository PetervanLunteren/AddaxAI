# Developer Documentation

## After cloning

Activate the commit-msg hook that strips auto-generated co-author lines:

```bash
git config core.hooksPath .githooks
```

This only needs to be run once per clone.

## Logging & Debugging

**Log files:** All logs (backend + frontend) are written to `~/AddaxAI/logs/backend.log`

**Watch logs in real-time:**
```bash
tail -f ~/AddaxAI/logs/backend.log
```

**Add logging in code:**
```python
# Backend (Python)
from app.core.logging_config import get_logger
logger = get_logger(__name__)
logger.info("Operation completed")
logger.error("Something failed", exc_info=True)  # Include stack trace
```

```typescript
// Frontend (TypeScript)
import { logger } from "@/lib/logger";
logger.info("User clicked button", { buttonId: "create-project" });
logger.error("API call failed", { endpoint: "/api/projects", error: err.message });
```

**Log retention:** Automatic rotation at 33MB per file, keeps 3 backups (100MB total, ~7 days).

## Linting (CI enforcement)

GitHub Actions runs **ruff** on every push and PR (`ruff check app tests`). The build fails if there are any errors, so check locally before pushing:

```bash
cd backend
ruff check app tests          # check only
ruff check app tests --fix    # auto-fix import sorting (I001) and unused imports (F401)
```

**Common pitfalls that CI catches:**

| Rule | What it means | How to fix |
|------|---------------|------------|
| **E501** | Line exceeds 100 characters | Break the line — wrap args, use intermediate variables, etc. |
| **I001** | Imports not sorted | Run `ruff check --fix` (auto-fixable) |
| **F401** | Unused import | Remove it, or run `ruff check --fix` |
| **F841** | Variable assigned but never used | Remove the assignment |
| **B904** | `raise` inside `except` without `from` | Use `raise ... from err` or `raise ... from None` |

The max line length is **100 characters** (configured in `pyproject.toml`). This is the #1 source of CI failures — keep lines short.

## Testing

Backend tests use **pytest** with an in-memory SQLite database. Each test gets a fresh DB session that rolls back after the test, so tests are fully isolated.

```bash
cd backend
pytest                        # run all tests
pytest tests/api/             # run only API tests
pytest tests/ml/              # run only ML/taxonomy tests
pytest tests/integration/     # run integration tests
pytest -x                     # stop on first failure
pytest -k "test_label_tree"   # run tests matching a name pattern
```

Coverage is collected automatically (`--cov=app` in `pyproject.toml`).

**Test structure:**

| Directory | What it tests |
|-----------|---------------|
| `tests/api/` | API endpoints via FastAPI `TestClient` |
| `tests/ml/` | ML utilities (taxonomy parsing, rollup, postprocessing) |
| `tests/integration/` | Multi-step pipelines (event generation, detection pipeline) |
| `tests/models/` | SQLAlchemy model constraints and relationships |
| `tests/` (root) | Standalone unit tests (scoring, websocket, etc.) |

**Writing tests:** Use the factory helpers in `tests/conftest.py` (`make_project`, `make_site`, `make_deployment`, `make_file`, `make_detection`, `make_event_with_files`) to build test data. Use the `client` fixture for API tests and the `db` fixture for direct DB tests.

## Detection threshold and verified override

Every project has a `detection_threshold` (e.g. 0.5). Detections below this confidence are hidden from the UI. However, verified detections always pass, regardless of confidence. A human verification is a stronger signal than a model score.

**The rule:** anywhere you query detections and the result is user-facing, apply:

```python
or_(Detection.confidence >= threshold, Detection.verified == True)
```

This must be applied consistently across every module that counts, lists, filters, or displays detections. The places where this is currently enforced:

| Module | What it covers |
|--------|---------------|
| `crud/statistics.py` | Dashboard stats (overview, species, activity, trend, categories) |
| `crud/label_tree.py` | Label filter tree counts (detection and event modes) |
| `crud/event.py` | Event label filter, standalone confidence filter, verification stats, filter options |
| `crud/project.py` | Project card detection counts (single and bulk) |
| `routers/projects.py` | Detection count, label stats, category stats, independent event stats |
| `ml/inference/similarity_script.py` | Similarity sort/search (raw SQL) |

**When adding a new query that touches detections**, check whether the result is user-facing. If yes, apply the threshold with the verified override. If you skip this, detection counts and filter options will be inconsistent with what the user sees in the verification grid.

**Two exceptions where `OR verified` does not apply:**
1. **User-driven confidence range filters** (e.g. a max_confidence ceiling). When a user explicitly sets a confidence range, respect it literally. The override only applies to the project's threshold floor, not to user-specified ceilings.
2. **Per-file detection lists** (`crud/detection.py`). These serve the file detail view where the caller controls what to show. Not tied to the project threshold.

**Common mistake:** writing `Detection.confidence >= threshold` without `OR Detection.verified == True`. This silently drops verified low-confidence detections from counts, filters, and charts. The result is that users see different numbers on different pages.

## Non-label detection skip

MegaDetector sometimes produces false positive bounding boxes. When a classification model (SpeciesNet or custom) classifies a detection as one of the non-label classes, the detection is not loaded to the database at all. This keeps false positives out of counts, filters, and the verification UI.

**Non-label classes** (defined in `backend/app/ml/label_exclusion.py`): `bait`, `blank`, `empty`, `false detection`, `none`, `vide` (French for empty). These are always stripped, regardless of project settings.

**The rule:** a detection is skipped when the classifier returned output AND after filtering out non-label classes, zero classifications remain. Detections with no classifier output (unclassified animals) are still loaded with `label=NULL`. Person and vehicle detections are never classified and are always loaded.

**Observation type:** files where all detections were skipped get `observation_type="blank"`. They will not appear in the verification grid and will be counted as blank images on the dashboard.

**Raw JSON preservation:** the JSON on disk (`results.json`) is never modified. It contains all original detections including those classified as blank. The skip only applies during the in-memory DB load step.

**Key files:**

| File | What it does |
|------|-------------|
| `backend/app/ml/label_exclusion.py` | `NON_LABEL_CLASSES` set, `is_non_label_detection()` helper |
| `backend/app/ml/json_pipeline.py` | Skip logic in `load_json_to_database()` and `_load_to_database()` |

## Best frame selection (videos)

After video detection (phase 1) and frame extraction, a single representative frame number is selected per video. The algorithm:

1. Score each frame by summing animal detection confidences (>= 0.3)
2. Among top candidates (within 10% of best score), pick the sharpest (Laplacian variance)
3. Blank videos (no detections): sample ~10 evenly-spaced frames, pick the sharpest

See `backend/app/ml/best_frame.py`.

**Storage:** No separate frame JPEG is saved — `best_frame_path` points to the frame inside `video_frames/`: `{deployment_folder}/.addaxai/video_frames/{video_name}/frame{N:06d}.jpg`. The `files` table stores `best_frame_number` (0-based index) and `best_frame_path` (absolute path to the JPEG). Both are `NULL` for images.

**Usage:** The best frame is the canonical image representation of a video. Use it anywhere you'd use a photo for an image file:
- Thumbnails in the UI
- Human verification workflows
- Depth estimation
- Any future per-file visual feature

If you're building a feature that works on images, check `file.best_frame_path` for videos instead of extracting frames yourself.

## Creating a custom classification model

To add a new classification model to AddaxAI, create an `inference.py` file in your model's directory that implements the `ModelInference` class.

**Template:** See `/backend/templates/inference_template.py` for a complete template with examples.

**Required interface:**
```python
class ModelInference:
    def __init__(self, model_dir: Path, model_path: Path):
        # Store paths and initialize
        pass

    def check_gpu(self) -> bool:
        # Return True if GPU available
        pass

    def load_model(self) -> None:
        # Load model once at startup
        pass

    def get_crop(self, image: Image.Image, bbox: tuple[float, float, float, float]) -> Image.Image:
        # Crop and preprocess image for your model
        pass

    def get_classification(self, crop: Image.Image) -> list[tuple[str, float]]:
        # Return [(class_name, confidence), ...] for ALL classes
        pass

    def get_class_names(self) -> dict[str, str]:
        # Return {"1": "label1", "2": "label2", ...} (1-indexed)
        pass
```

**Benefits of class-based approach:**
- No global variables or `global` keyword needed
- Clear ownership (`self.model`)
- Framework-agnostic (works with PyTorch, Keras, JAX, TensorFlow, etc.)
- IDE autocomplete and type checking work properly

**Examples:**
- NAM-ADS-v1: YOLOv8 (PyTorch) - `/Users/peter/AddaxAI/models/cls/NAM-ADS-v1/inference.py`
- TAS-BB-v1: MEWC-Keras (Keras/JAX) - `/Users/peter/AddaxAI/models/cls/TAS-BB-v1/inference.py`

## Label taxonomy and the hierarchical filter tree

The label filter in the UI can render as either a flat multiselect or a hierarchical tree (class > order > family > genus > species). The tree is built from the `label_taxonomy` table. If no taxonomy rows exist for a project's classification model, the frontend falls back to the flat list.

### Database table: `label_taxonomy`

See `backend/app/models/label_taxonomy.py`.

| Column | Purpose |
|--------|---------|
| `classification_model_id` | Links to the classification model |
| `name` | Display label — **must match `Detection.label`** (this is the join key) |
| `taxon_class` .. `taxon_species` | Formal taxonomy ranks (nullable) |
| `level` | Most specific non-empty rank: `"class"`, `"order"`, `"family"`, `"genus"`, or `"species"` |
| `is_custom` | `True` for user-created entries, `False` for model-sourced entries |

Unique constraint: `(classification_model_id, name)`. All taxonomy functions are idempotent — calling them twice inserts 0 the second time.

`Detection.label` is a plain text field, **not** a foreign key. The tree builder matches it against `label_taxonomy.name` by string equality. This means the `name` value must exactly match whatever string ends up in `Detection.label`.

### How taxonomy gets populated

Taxonomy is populated automatically during two worker phases. All population functions live in `backend/app/ml/taxonomy_db.py`.

#### 1. Custom models with `taxonomy.csv`

Custom classification models (e.g. EUR-DF, NAM-ADS) ship a `taxonomy.csv` alongside their weights:

```csv
model_class,class,order,family,genus,species
leopard,mammalia,carnivora,felidae,panthera,pardus
bird,aves,,,,
```

`populate_taxonomy_from_csv(model_id, csv_path, db)` reads this file and inserts one row per line. The `model_class` column becomes `label_taxonomy.name`. Entries with only partial taxonomy (e.g. "bird" with just `class=aves`) get `level="class"`.

#### 2. Taxonomic rollup entries

When taxonomic rollup is enabled and a detection's top-1 confidence is below threshold, confidences are summed up the taxonomy tree. If a higher-level taxon (e.g. "felidae" at family level) crosses the threshold, `Detection.label` is set to that taxon name.

`add_rollup_taxonomy_entry(model_id, name, level, taxonomy_lookup, db)` inserts a new `label_taxonomy` row for the rolled-up label so it appears in the tree under the correct branch. Called from `backend/app/ml/postprocessing.py` for each new rolled-up label.

### Where population is triggered

Both workers call `populate_taxonomy_from_csv` when a `taxonomy.csv` exists in the model directory:

| Worker | When | Code location |
|--------|------|---------------|
| `detection_worker.py` | After loading results to DB (phase 6) | ~line 520 |
| `postprocessing_worker.py` | After reprocessing all deployments | ~line 174 |

```python
if taxonomy_csv.exists():
    populate_taxonomy_from_csv(model_id, taxonomy_csv, db)
```

The detection worker runs this once per deployment. The postprocessing worker runs it when reprocessing (e.g. after changing model or settings). Since all functions are idempotent, running them multiple times is safe.

### How the filter tree is built

`build_label_filter_tree()` in `backend/app/api/crud/label_tree.py`:

1. Queries which labels actually have detections in the project
2. Joins against `label_taxonomy` to get taxonomy columns
3. Builds the hierarchy: class > order > family > genus > species
4. Annotates each leaf with detection or event counts
5. Labels with no taxonomy match go under an `"__other__"` node
6. Returns `null` if no taxonomy rows exist (frontend shows flat list)

Exposed via `GET /api/events/label-tree?project_id=<id>&count_by=<event|detection>`.

### The `is_custom` flag

All model-sourced entries (CSV, JSON, rollup) set `is_custom=False`. The flag exists for UI-driven taxonomy creation where users can add custom labels with taxonomy info. Custom entries work identically in the tree builder — it queries all `label_taxonomy` rows for the model regardless of `is_custom`.

### Key files

| File | Purpose |
|------|---------|
| `backend/app/models/label_taxonomy.py` | SQLAlchemy model |
| `backend/app/ml/taxonomy_db.py` | Population functions (CSV, JSON, rollup) |
| `backend/app/ml/taxonomic_rollup.py` | Rollup algorithm (sums confidences up tree) |
| `backend/app/ml/postprocessing.py` | Orchestrates rollup + calls `add_rollup_taxonomy_entry` |
| `backend/app/api/crud/label_tree.py` | Builds the filter tree from `label_taxonomy` |
| `backend/app/ml/taxonomy_parser.py` | Parses CSV into a tree structure (used for validation, not DB) |
| `backend/tests/ml/test_taxonomy_db.py` | Tests for all population functions |
| `backend/tests/api/test_label_tree.py` | Tests for tree building + API endpoint |

### Rules

**No ad-hoc database fixes.** Do not run one-time scripts to patch database state. If data is stale or incorrect, fix the code that produces it. The data will be corrected when the user re-runs the relevant operation (analysis, reprocessing, taxonomy population). The app must handle its own data integrity.

**Never overwrite verified detections.** When a user manually verifies or relabels a detection (`Detection.verified == True`), that human judgment takes priority over any machine output. Postprocessing, reprocessing, taxonomic rollup, smoothing, and any other automatic pipeline must skip verified detections. If you are writing code that updates `Detection.label`, `Detection.label_confidence`, or `Detection.category`, always check `verified` first and leave verified records untouched.