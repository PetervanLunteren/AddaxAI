# Developer Documentation

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

#### 2. SpeciesNet (no CSV, taxonomy embedded in results JSON)

SpeciesNet doesn't ship a `taxonomy.csv`. Instead, its `results.json` contains a `classification_category_descriptions` dict with semicolon-delimited taxonomy strings:

```json
{
  "classification_category_descriptions": {
    "0": "uuid;mammalia;cetartiodactyla;bovidae;bos;taurus;domestic cattle",
    "1": "uuid;mammalia;cetartiodactyla;bovidae;;;bovidae",
    "2": "uuid;;;;;;;blank"
  }
}
```

Format: `UUID;class;order;family;genus;species;common_name`

`populate_taxonomy_from_json(model_id, json_path, db)` parses these strings and uses the **common name** (last field) as `label_taxonomy.name`. Entries with no taxonomy fields (e.g. "blank") are skipped.

#### 3. Taxonomic rollup entries

When taxonomic rollup is enabled and a detection's top-1 confidence is below threshold, confidences are summed up the taxonomy tree. If a higher-level taxon (e.g. "felidae" at family level) crosses the threshold, `Detection.label` is set to that taxon name.

`add_rollup_taxonomy_entry(model_id, name, level, taxonomy_lookup, db)` inserts a new `label_taxonomy` row for the rolled-up label so it appears in the tree under the correct branch. Called from `backend/app/ml/postprocessing.py` for each new rolled-up label.

### Where population is triggered

Both workers use the same fallback pattern — try CSV first, fall back to JSON:

| Worker | When | Code location |
|--------|------|---------------|
| `detection_worker.py` | After loading results to DB (phase 6) | ~line 542 |
| `postprocessing_worker.py` | After reprocessing all deployments | ~line 174 |

```python
# Simplified pattern used in both workers:
if taxonomy_csv.exists():
    populate_taxonomy_from_csv(model_id, taxonomy_csv, db)
elif results_json.exists():
    populate_taxonomy_from_json(model_id, results_json, db)
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

