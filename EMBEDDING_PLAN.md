# Embedding plan for AddaxAI-WebUI

## Goal

Compute a DINOv2 embedding for every detection (animal, person, vehicle) as the final step of the analysis pipeline. Store embeddings in a dedicated table. Allow users to choose between three DINOv2 model sizes via the project settings page. This enables future clustering, similarity search, and outlier detection.

## Key decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Models | Three DINOv2 variants (ViT-S/B/L) | Same architecture family, simple script, good coverage |
| Manifest category | New `emb/` type alongside `det/` and `cls/` | Consistent with existing patterns |
| Environment | Reuse `env-addaxai-base` | Already has PyTorch, avoids ~3GB extra download |
| Storage | Separate `detection_embeddings` table | Keeps `detections` table fast for normal queries; allows re-embedding with different models |
| Precision | float16 | Halves storage, negligible impact on similarity quality (>0.9999 cosine sim vs float32) |
| Weight downloads | HuggingFace repos | Reuses existing `hf_downloader.py`, consistent with det/cls |
| Subprocess pattern | Batch script (like MegaDetector) | Runs once per deployment, processes everything, exits. Simple. |
| Image caching | Cache decoded PIL images per source file | Avoids redundant disk I/O for multiple detections in same frame |
| Confidence filter | Embed ALL detections | More complete for clustering/outlier analysis |
| Failure semantics | Fatal — fail hard | Per CONVENTIONS.md: "Crash early and loudly. Never allow silent failures." |
| Progress UI | Same format as det/cls phases | tqdm info: ETA, percentage, speed |
| Feature gating | None | Keep it simple, add later if needed |

---

## What exists today

### Pipeline phases (detection_worker.py `_process_batch_job`)
1. Video detection (MegaDetector subprocess in env-addaxai-base)
2. Video classification (SpeciesNet batch or per-detection worker subprocess)
3. Image detection (MegaDetector subprocess)
4. Image classification (same as step 2)
5. Merge JSONs (video + image results into results.json)
6. Load to database (parse JSON, create File + Detection records)
7. Postprocessing (event smoothing, taxonomic rollup)

### Model manifest system
- Manifests in `~/AddaxAI/models/{det,cls}/{model_id}/manifest.json`
- Schema: `backend/app/ml/schemas/model_manifest.py` (Pydantic)
- Manager: `backend/app/ml/manifest_manager.py` — scans `det/` and `cls/` dirs
- Storage: `backend/app/ml/model_storage.py` — downloads from HuggingFace, resolves `det`/`cls` paths
- Catalog: `backend/app/ml/catalog_updater.py` — syncs remote `models.json` → creates local stubs

### Model storage path resolution (model_storage.py)
Currently hardcoded: `model_type = "det" if manifest.model_category == "detection" else "cls"`. Needs to handle `"embedding"` → `"emb"`.

### Database
- `detections` table: id, file_id, job_id, category, confidence, bbox_*, species, species_confidence, classification_method, frame_number, created_at
- `projects` table: detection_model_id, classification_model_id (no embedding_model_id yet)

### Settings page
- React Hook Form + Zod validation
- Detection model dropdown → classification model dropdown → species selection
- Model status badges (ready / needs_weights / needs_env)
- Preparation triggers async download + env build with WebSocket progress

---

## Implementation plan

### Step 1: database schema changes

**Modify:** `backend/app/models/project.py` — add after `classification_model_id`:
```python
embedding_model_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
```

**New file:** `backend/app/models/detection_embedding.py`
```python
class DetectionEmbedding(Base):
    __tablename__ = "detection_embeddings"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    detection_id: Mapped[str] = mapped_column(ForeignKey("detections.id", ondelete="CASCADE"), index=True)
    job_id: Mapped[str | None] = mapped_column(ForeignKey("jobs.id"), nullable=True)
    embedding_model_id: Mapped[str] = mapped_column(String(100), index=True)
    vector: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)  # float16 bytes
    dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    l2_norm: Mapped[float] = mapped_column(Float, nullable=False)  # pre-computed for cosine similarity
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
```
Composite index on `(detection_id, embedding_model_id)` for upsert lookups.

Add `Detection.embeddings` relationship for eager loading:
```python
# In backend/app/models/detection.py
embeddings: Mapped[list["DetectionEmbedding"]] = relationship("DetectionEmbedding", back_populates="detection", cascade="all, delete-orphan")
```
```python
# In backend/app/models/detection_embedding.py
detection: Mapped["Detection"] = relationship("Detection", back_populates="embeddings")
```

**New file:** `backend/alembic/versions/YYYYMMDD_HHMM_add_embedding_support.py`
- Add `embedding_model_id` VARCHAR(100) to `projects`
- Create `detection_embeddings` table
- Uses `batch_alter_table` (required for SQLite)

---

### Step 2: model manifest schema

**Modify:** `backend/app/ml/schemas/model_manifest.py` — add optional fields:
```python
embedding_dim: int | None = None    # 384, 768, or 1024
input_size: int | None = None       # e.g., 224
torch_hub_model: str | None = None  # e.g., "dinov2_vits14" (for architecture loading)
```

These are optional so existing det/cls manifests remain valid.

---

### Step 3: model infrastructure (manifest manager, storage, catalog)

**Modify:** `backend/app/ml/manifest_manager.py`
- Change `for model_type in ["det", "cls"]:` → `["det", "cls", "emb"]`
- Update category mapping to include `"emb": "embedding"`
- Add `get_embedding_models()` method (same pattern as `get_classification_models()`)

**Modify:** `backend/app/ml/model_storage.py`
- Replace hardcoded `"det" if ... == "detection" else "cls"` with a helper:
  ```python
  def _model_type_dir(self, manifest: ModelManifest) -> str:
      return {"detection": "det", "classification": "cls", "embedding": "emb"}[manifest.model_category]
  ```
- No new download method needed — HuggingFace downloader handles everything

**Modify:** `backend/app/ml/catalog_updater.py`
- `fetch_catalog()`: handle `"emb"` key (backward compat if missing from old catalogs)
- `get_local_models()`: add `"emb"` to scan loop
- `compare_models()`: add `"emb"` to comparison loop
- `sync()`: include `"emb"` count in fresh-install detection

---

### Step 4: model catalog entries

**Modify:** `models.json` — add `"emb"` section under `"models"`:

| model_id | friendly_name | embedding_dim | input_size | torch_hub_model | env |
|----------|--------------|---------------|------------|-----------------|-----|
| DINOV2-VITS14 | DINOv2 ViT-S/14 | 384 | 224 | dinov2_vits14 | addaxai-base |
| DINOV2-VITB14 | DINOv2 ViT-B/14 | 768 | 224 | dinov2_vitb14 | addaxai-base |
| DINOV2-VITL14 | DINOv2 ViT-L/14 | 1024 | 224 | dinov2_vitl14 | addaxai-base |

All entries: `type: "embedding"`, `emoji: "🧬"`, developer: "Meta AI (FAIR)", license: "Apache 2.0". HuggingFace repos to be created by user with the pretrained weights.

---

### Step 5: backend API changes

**Modify:** `backend/app/api/routers/ml_models.py`
- Add `GET /api/ml/models/embedding` endpoint (list embedding models)
  - Include "No embeddings" option (model_id="none") as first item
  - Sort remaining by embedding_dim (smallest first)
- Existing `/api/ml/models/{model_id}/status` and `/api/ml/models/{model_id}/prepare` already work for any model type

**Modify:** `backend/app/api/schemas/project.py`
- Add `embedding_model_id: str | None = Field(None)` to project schemas

**Modify:** `backend/app/api/routers/projects.py`
- Normalize `"none"` → `None` for `embedding_model_id` (same pattern as classification_model_id)
- Validate embedding_model_id exists in manifest manager (if not None)
- Ensure `embedding_model_id` is copied in the duplicate project endpoint and included in `/stats` responses

---

### Step 6: embedding script

**New file:** `backend/app/ml/inference/embedding_script.py`

Standalone script that runs as a subprocess in env-addaxai-base. Same pattern as MegaDetector.

**CLI interface:**
```
python embedding_script.py \
    --input /path/to/embedding_input.json \
    --output /path/to/embeddings.npz \
    --weights /path/to/dinov2_vits14_pretrain.pth \
    --model-arch dinov2_vits14 \
    --embedding-dim 384 \
    --input-size 224
```

**Input JSON:**
```json
{
  "detections": [
    {
      "detection_id": "uuid-1",
      "image_path": "/absolute/path/to/image.jpg",
      "bbox": [0.1, 0.2, 0.3, 0.4]
    }
  ]
}
```

**Processing steps:**
1. Parse arguments
2. Detect device: CUDA → MPS → CPU
   ```python
   if torch.cuda.is_available():
       device = torch.device("cuda")
   elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
       device = torch.device("mps")
       os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
   else:
       device = torch.device("cpu")
   ```
3. Load DINOv2 architecture via `torch.hub.load("facebookresearch/dinov2", model_name, pretrained=False)`
4. Load local weights via `torch.load(weights_path, map_location=device)`
5. Preprocessing: Resize(input_size) → ToTensor → Normalize(ImageNet stats)
6. Group detections by `image_path` — cache decoded PIL images per source file
7. For each batch:
   - Crop bbox regions from (cached) source images, preprocess
   - Stack into batch tensor, forward pass → CLS token
   - Convert to float16 numpy arrays
   - Print tqdm progress to stderr (same format as MegaDetector for progress parsing)
8. Save as .npz: keys are detection_ids, values are float16 arrays

**Batch sizes:** GPU=64, MPS=32, CPU=8 (auto-selected based on device, overridable via `--batch-size`).

**Output:** `.npz` file where each key is a detection_id and each value is a float16 numpy array.

---

### Step 7: embedding model wrapper

**New file:** `backend/app/ml/inference/embedding_model.py`

Thin wrapper around subprocess invocation, same pattern as MegaDetector wrapper.

```python
class EmbeddingModel:
    def __init__(self, model_path: Path, manifest: ModelManifest, env_manager: EnvironmentManager):
        self.model_path = model_path
        self.manifest = manifest
        self.python_path = env_manager.get_python("env-addaxai-base")
        self.script_path = Path(__file__).parent / "embedding_script.py"

    def compute_embeddings(
        self,
        input_json_path: Path,
        output_npz_path: Path,
        progress_callback: Callable | None = None,
    ) -> int:
        """
        Run embedding subprocess. Returns number of embeddings computed.
        Parses tqdm output from stderr for progress (same pattern as MegaDetector).
        """
        cmd = [
            str(self.python_path),
            str(self.script_path),
            "--input", str(input_json_path),
            "--output", str(output_npz_path),
            "--weights", str(self.model_path),
            "--model-arch", self.manifest.torch_hub_model,
            "--embedding-dim", str(self.manifest.embedding_dim),
            "--input-size", str(self.manifest.input_size),
        ]
        # Run subprocess, parse stderr for tqdm progress
        ...
```

---

### Step 8: embedding utilities

**New file:** `backend/app/ml/embedding_utils.py`

```python
def build_embedding_input(deployment_id: str, deployment_folder: Path, artifacts_folder: Path, db: Session) -> dict:
    """
    Query all detections for deployment, resolve image paths.
    Returns JSON-serializable dict for the embedding script.

    For video detections (frame_number is not None): resolve frame path from artifacts.
    For image detections: use File.file_path directly.
    """
    ...

def save_embeddings_to_db(
    npz_path: Path,
    job_id: str,
    embedding_model_id: str,
    embedding_dim: int,
    db: Session,
) -> int:
    """
    Load .npz file and bulk-insert DetectionEmbedding rows.
    Computes l2_norm per vector during insertion.
    Deletes existing embeddings for the same (detection_id, embedding_model_id) first.
    Returns count of inserted records.
    Flushes every 500 records to avoid excessive WAL growth.
    """
    ...
```

---

### Step 9: pipeline integration

**Modify:** `backend/app/workers/detection_worker.py`

Insert phase 8 in `_process_batch_job`, after phase 7 (postprocessing):

```python
# ============================================================
# PHASE 8: Embedding (DINOv2)
# ============================================================
embedding_model_id = project.embedding_model_id
if embedding_model_id:
    logger.info(f"Phase 8: Computing embeddings with {embedding_model_id}")
    await deployment_progress_callback("Computing embeddings...", 0.0, "embedding", 0.0)

    emb_manifest = manifest_manager.get_model(embedding_model_id)
    emb_model_path = model_storage.get_model_file(emb_manifest)

    from app.ml.inference.embedding_model import EmbeddingModel
    from app.ml.embedding_utils import build_embedding_input, save_embeddings_to_db

    embedding_model = EmbeddingModel(emb_model_path, emb_manifest, env_manager)

    input_data = build_embedding_input(deployment.id, folder_path, artifacts_folder, db)
    input_json_path = artifacts_folder / "embedding_input.json"
    output_npz_path = artifacts_folder / "embeddings.npz"

    with open(input_json_path, "w") as f:
        json.dump(input_data, f)

    # Run embedding subprocess
    embedded_count = await loop.run_in_executor(
        None,
        lambda: embedding_model.compute_embeddings(
            input_json_path, output_npz_path, progress_callback
        ),
    )

    # Write embeddings to DB
    if output_npz_path.exists():
        save_embeddings_to_db(
            output_npz_path, job.id, embedding_model_id,
            emb_manifest.embedding_dim, db
        )

    # Clean up intermediate files
    input_json_path.unlink(missing_ok=True)
    output_npz_path.unlink(missing_ok=True)

    logger.info(f"Embedding complete: {embedded_count} detections embedded")
```

This phase is **fatal** — if it fails, the deployment fails. No try/except swallowing errors.

**Data flow:**
```
DB (Detection records) → build_embedding_input() → embedding_input.json
    → embedding_script.py subprocess → embeddings.npz
    → save_embeddings_to_db() → INSERT INTO detection_embeddings
```

---

### Step 10: frontend changes

**Modify:** `frontend/src/api/types.ts`
- Add `embedding_model_id: string | null` to `ProjectCreate`, `ProjectUpdate`, `ProjectResponse`

**Modify:** `frontend/src/api/models.ts`
- Add `listEmbeddingModels: () => api.get<ModelInfo[]>("/api/ml/models/embedding")`

**Modify:** `frontend/src/pages/SettingsPage.tsx`
- Add `embedding_model_id: z.string().optional().nullable()` to Zod schema
- Add `useQuery` for embedding models list
- Add `useQuery` for embedding model status (when selected and not "none")
- Add new `FormField` section after classification model:
  - Label: "Embedding model"
  - Description: "Computes feature vectors for each detection crop. Used for similarity search and clustering."
  - Dropdown with model options (emoji + name + description_short)
  - Status badge + "Prepare" button (same pattern as classification)

**Modify:** `frontend/src/hooks/useTaskProgress.ts`
- Add `"embedding"` to recognized phase names
- Progress bar shows same info as det/cls: ETA, percentage, speed

---

## Performance estimates

| Metric | ViT-S/14 | ViT-B/14 | ViT-L/14 |
|--------|---------|---------|---------|
| CUDA GPU | ~500 crops/s | ~300 crops/s | ~100 crops/s |
| Apple MPS | ~200 crops/s | ~120 crops/s | ~50 crops/s |
| CPU | ~20 crops/s | ~10 crops/s | ~5 crops/s |
| Storage per detection (float16) | 768 B | 1.5 KB | 2 KB |

For a typical deployment of 5,000 images with ~2 detections each (10,000 detections):
- ViT-S on GPU: ~20 seconds
- ViT-S on CPU: ~8 minutes
- DB storage overhead: ~7.5 MB (ViT-S, float16)

---

## Testing strategy

### Unit tests
- `test_embedding_utils.py`: verify `build_embedding_input()` resolves image vs frame paths correctly; verify `save_embeddings_to_db()` inserts rows with correct dimensions and handles delete-before-insert
- `test_embedding_model.py`: verify `EmbeddingModel` builds correct subprocess command
- `test_detection_embedding_model.py`: verify ORM model, cascade deletes, composite index behavior

### Integration tests
- `test_embedding_script.py`: run embedding_script.py on a small set of test images, verify .npz output has correct shapes, dtypes (float16), and detection_ids
- `test_model_storage_emb.py`: verify model_storage handles `"embedding"` category correctly in path resolution
- `test_manifest_manager_emb.py`: verify manifest manager scans `emb/` dir and returns embedding models
- `test_catalog_updater_emb.py`: verify catalog updater syncs `emb` entries from models.json

### API tests
- `test_ml_models_api.py`: verify `GET /api/ml/models/embedding` returns models with "none" option first
- `test_projects_api.py`: verify project create/update accepts and validates embedding_model_id

### Migration test
- Verify alembic migration creates table and column correctly, both upgrade and downgrade

### Frontend tests
- Settings page renders embedding model selector
- Queue/progress UI shows "embedding" phase with correct label

---

## File list

### New files (5)
| File | Purpose |
|------|---------|
| `backend/app/models/detection_embedding.py` | DetectionEmbedding ORM model |
| `backend/app/ml/inference/embedding_script.py` | Standalone DINOv2 subprocess script |
| `backend/app/ml/inference/embedding_model.py` | Subprocess wrapper class |
| `backend/app/ml/embedding_utils.py` | build_embedding_input + save_embeddings_to_db |
| `backend/alembic/versions/*_add_embedding_support.py` | DB migration |

### Modified files (13)
| File | Changes |
|------|---------|
| `backend/app/models/project.py` | Add `embedding_model_id` column |
| `backend/app/ml/schemas/model_manifest.py` | Add `embedding_dim`, `input_size`, `torch_hub_model` |
| `backend/app/ml/manifest_manager.py` | Scan `emb/` dir, add `get_embedding_models()` |
| `backend/app/ml/model_storage.py` | Handle `"embedding"` category in path resolution |
| `backend/app/ml/catalog_updater.py` | Include `"emb"` in scan/compare/sync |
| `backend/app/workers/detection_worker.py` | Add phase 8 embedding block |
| `backend/app/api/routers/ml_models.py` | Add `GET /models/embedding` endpoint |
| `backend/app/api/schemas/project.py` | Add `embedding_model_id` field |
| `backend/app/api/routers/projects.py` | Validate embedding_model_id on create/update |
| `models.json` | Add `"emb"` section with 3 DINOv2 entries |
| `frontend/src/api/types.ts` | Add `embedding_model_id` to project types |
| `frontend/src/api/models.ts` | Add `listEmbeddingModels()` |
| `frontend/src/pages/SettingsPage.tsx` | Add embedding model dropdown + status + prepare |
| `frontend/src/hooks/useTaskProgress.ts` | Recognize `"embedding"` phase |

### Implementation order
1. DB schema (models + migration)
2. Model manifest schema
3. Model infrastructure (manifest_manager, model_storage, catalog_updater)
4. Model catalog (models.json)
5. Backend API (ml_models router, project schemas, project router)
6. Embedding script + model wrapper + utils
7. Pipeline integration (detection_worker.py)
8. Frontend (types, API, settings page, progress hook)
9. Tests
