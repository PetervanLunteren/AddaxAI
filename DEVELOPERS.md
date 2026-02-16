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

After video detection (phase 1), a single representative frame is selected per video and saved to `.addaxai/frames/{video_stem}.jpg`. The algorithm:

1. Score each frame by summing animal detection confidences (>= 0.3)
2. Among top candidates (within 10% of best score), pick the sharpest (Laplacian variance)
3. Blank videos (no detections): sample ~10 evenly-spaced frames, pick the sharpest

See `backend/app/ml/best_frame.py`.

**Storage:** The frame JPEG is at `{deployment_folder}/.addaxai/frames/{video_stem}.jpg`. The `files` table stores `best_frame_number` (0-based index) and `best_frame_path` (absolute path to the JPEG). Both are `NULL` for images.

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
        # Return {"1": "species1", "2": "species2", ...} (1-indexed)
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

