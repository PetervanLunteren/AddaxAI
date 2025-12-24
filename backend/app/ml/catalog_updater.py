"""
Model catalog updater - fetches central manifest and creates stubs for new models.

Following DEVELOPERS.md principles:
- Fail silently if offline (non-critical operation)
- Never overwrite existing model directories
- Log all operations for debugging
"""

import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.logging_config import get_logger
from app.ml.schemas.model_manifest import ModelManifest

logger = get_logger(__name__)


class ModelCatalogUpdater:
    """
    Fetches central model catalog and creates local directory stubs for new models.

    Only creates manifest.json files - does not download weights.
    """

    def __init__(self, models_dir: Path | None = None, catalog_url: str | None = None):
        """
        Initialize catalog updater.

        Args:
            models_dir: Directory where models are stored (default: ~/AddaxAI/models)
            catalog_url: URL to fetch catalog from (default: from config)
        """
        user_data_dir = Path.home() / "AddaxAI"
        self.models_dir = models_dir or (user_data_dir / "models")
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Default to GitHub raw URL
        self.catalog_url = catalog_url or (
            "https://raw.githubusercontent.com/PetervanLunteren/AddaxAI-WebUI/main/models.json"
        )

    def fetch_catalog(self, timeout: int = 2) -> dict[str, Any] | None:
        """
        Fetch model catalog from remote URL.

        Args:
            timeout: Request timeout in seconds (default: 2)

        Returns:
            Catalog dict if successful, None if failed

        Raises:
            Never raises - logs errors and returns None
        """
        try:
            logger.info(f"Fetching model catalog from {self.catalog_url}")

            with urllib.request.urlopen(self.catalog_url, timeout=timeout) as response:
                data = response.read()

            catalog = json.loads(data)

            # Validate basic structure
            if "models" not in catalog:
                logger.error("Invalid catalog structure: missing 'models'")
                return None

            if "det" not in catalog["models"] or "cls" not in catalog["models"]:
                logger.error("Invalid catalog structure: missing 'det' or 'cls' in models")
                return None

            logger.info(
                f"Fetched catalog: "
                f"{len(catalog['models']['det'])} det, {len(catalog['models']['cls'])} cls models"
            )
            return catalog

        except urllib.error.URLError as e:
            logger.warning(f"Failed to fetch model catalog (offline or unreachable): {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse model catalog JSON: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching model catalog: {e}", exc_info=True)
            return None

    def get_local_models(self) -> dict[str, set[str]]:
        """
        Scan local models directory and return existing model IDs.

        Returns:
            Dict with 'det' and 'cls' keys, values are sets of model_ids
        """
        local_models: dict[str, set[str]] = {"det": set(), "cls": set()}

        for model_type in ["det", "cls"]:
            type_dir = self.models_dir / model_type
            if not type_dir.exists():
                continue

            for model_dir in type_dir.iterdir():
                if model_dir.is_dir() and (model_dir / "manifest.json").exists():
                    local_models[model_type].add(model_dir.name)

        logger.debug(
            f"Found {len(local_models['det'])} local det models, "
            f"{len(local_models['cls'])} local cls models"
        )
        return local_models

    def compare_models(self, remote_catalog: dict[str, Any], local_models: dict[str, set[str]]) -> list[dict[str, Any]]:
        """
        Compare remote catalog with local models and return new models.

        Args:
            remote_catalog: Catalog fetched from remote URL
            local_models: Local model IDs from get_local_models()

        Returns:
            List of new model info (dicts with model_id, friendly_name, model_type)
        """
        new_models: list[dict[str, Any]] = []

        for model_type in ["det", "cls"]:
            remote_model_list = remote_catalog["models"][model_type]

            for manifest_data in remote_model_list:
                model_id = manifest_data["model_id"]

                if model_id not in local_models[model_type]:
                    new_models.append(
                        {
                            "model_id": model_id,
                            "friendly_name": manifest_data.get("friendly_name", model_id),
                            "emoji": manifest_data.get("emoji", "🤖"),
                            "model_type": model_type,
                            "manifest": manifest_data,
                        }
                    )
                    logger.info(f"New model found: {model_type}/{model_id}")

        return new_models

    def create_model_stub(self, model_type: str, manifest_data: dict[str, Any]) -> None:
        """
        Create model directory with manifest.json.

        Args:
            model_type: 'det' or 'cls'
            manifest_data: Model manifest dict

        Raises:
            Never raises - logs errors
        """
        model_id = manifest_data["model_id"]
        model_dir = self.models_dir / model_type / model_id

        try:
            # Check if directory already exists (safety check)
            if model_dir.exists():
                logger.warning(f"Model directory already exists, skipping: {model_dir}")
                return

            # Create directory
            model_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created model directory: {model_dir}")

            # Write manifest.json
            manifest_path = model_dir / "manifest.json"
            with open(manifest_path, "w") as f:
                json.dump(manifest_data, f, indent=2)

            logger.info(f"Created manifest for {model_type}/{model_id}")

        except Exception as e:
            logger.error(f"Failed to create model stub for {model_type}/{model_id}: {e}", exc_info=True)

    async def sync(self) -> dict[str, Any]:
        """
        Main sync method: fetch catalog, compare, create stubs.

        Returns:
            Dict with sync results:
            {
                "new_models": [
                    {"model_id": "...", "friendly_name": "..."},
                    ...
                ],
                "checked_at": "2025-12-24T10:00:00Z",
                "error": "error message" (if failed)
            }

        Note: This is async to allow non-blocking execution in FastAPI lifespan
        """
        result: dict[str, Any] = {
            "new_models": [],
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }

        try:
            # Fetch remote catalog
            catalog = self.fetch_catalog()
            if catalog is None:
                result["error"] = "Failed to fetch catalog"
                return result

            # Get local models
            local_models = self.get_local_models()

            # Check if this is a fresh install (no models at all)
            total_local = len(local_models["det"]) + len(local_models["cls"])
            is_fresh_install = total_local == 0

            # Compare and find new models
            new_models = self.compare_models(catalog, local_models)

            # Create stubs for new models
            for new_model in new_models:
                self.create_model_stub(new_model["model_type"], new_model["manifest"])

                # Only add to notification list if not a fresh install
                if not is_fresh_install:
                    result["new_models"].append({
                        "model_id": new_model["model_id"],
                        "friendly_name": new_model["friendly_name"],
                        "emoji": new_model["emoji"],
                    })

            if new_models:
                if is_fresh_install:
                    logger.info(f"Model catalog sync complete: {len(new_models)} models initialized (fresh install)")
                else:
                    logger.info(f"Model catalog sync complete: {len(result['new_models'])} new models added")
            else:
                logger.info("Model catalog sync complete: no new models")

            return result

        except Exception as e:
            logger.error(f"Model catalog sync failed: {e}", exc_info=True)
            result["error"] = str(e)
            return result
