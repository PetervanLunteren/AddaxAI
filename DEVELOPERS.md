# Developer Documentation

**Key principles:**
1. **Crash early and loudly** - Fail hard in development so bugs cannot hide. Never allow silent failures.
2. **Explicit configuration** - No defaults. If something is missing, stop and surface it immediately.
3. **Type hints everywhere** - Make expectations clear and support safe refactoring.
4. **Short and clear documentation** - Keep explanations concise without losing clarity.
5. **Open source friendly** - Never commit secrets or anything that should not be public.
6. **No backward compatibility** - The project is in motion and has no users. Refactor freely when needed.
7. **Prefer simple solutions** - Use straightforward approaches that follow the conventions. Avoid cleverness when simplicity works.
8. **Follow the established conventions** - Keep structure predictable so the codebase stays readable and easy to maintain. 
9. **No quick fixes** - Fix issues in a way that holds for all future deployments, not only the current device.
10. **GitHub** - Always commit manually. Never commit automatically. 
11. **Clean repo** - Value simplicity and cleanliness. No redundant MD files. 
12. **No Title Case** - Use natural English capitalisation. That means only capitalising the first word of sentences and proper nouns (like "Peter van Lunteren", "Utrecht", "MegaDetector", "SpeciesNet", "Today, I was walking in the park.",  "Things I love about Amsterdam.", "Cities visited").

**Remember:** It's better to crash during development than to hide bugs that cause problems later. We'll add resilience (retries, fallbacks, graceful degradation) after the core functionality works.

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

