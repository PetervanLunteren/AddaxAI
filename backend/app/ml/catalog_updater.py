"""
Model catalog updater - fetches central manifest and creates stubs for new models.

Following DEVELOPERS.md principles:
- Fail silently if offline (non-critical operation)
- Never overwrite existing model directories
- Log all operations for debugging
"""

import json
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.logging_config import get_logger

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

            det_count = len(catalog['models']['det'])
            cls_count = len(catalog['models']['cls'])
            emb_count = len(catalog["models"].get("emb", []))
            logger.info(
                f"Fetched catalog: {det_count} det, "
                f"{cls_count} cls, {emb_count} emb models"
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
        local_models: dict[str, set[str]] = {"det": set(), "cls": set(), "emb": set()}

        for model_type in ["det", "cls", "emb"]:
            type_dir = self.models_dir / model_type
            if not type_dir.exists():
                continue

            for model_dir in type_dir.iterdir():
                if model_dir.is_dir() and (model_dir / "manifest.json").exists():
                    local_models[model_type].add(model_dir.name)

        logger.debug(
            f"Found {len(local_models['det'])} local det models, "
            f"{len(local_models['cls'])} local cls models, "
            f"{len(local_models['emb'])} local emb models"
        )
        return local_models

    def download_taxonomy(self, model_id: str, model_dir: Path) -> None:
        """
        Download taxonomy.csv from HuggingFace repo.

        Args:
            model_id: Model ID (used to construct HF repo URL)
            model_dir: Local directory to save taxonomy.csv

        Raises:
            Never raises - logs errors and continues
        """
        # Construct HuggingFace URL
        hf_repo = f"Addax-Data-Science/{model_id}"
        taxonomy_url = f"https://huggingface.co/{hf_repo}/resolve/main/taxonomy.csv?download=true"
        taxonomy_path = model_dir / "taxonomy.csv"

        try:
            logger.info(f"Downloading taxonomy.csv from {taxonomy_url}")

            with urllib.request.urlopen(taxonomy_url, timeout=5) as response:
                data = response.read()

            with open(taxonomy_path, "wb") as f:
                f.write(data)

            logger.info(f"Downloaded taxonomy.csv for {model_id}")

        except urllib.error.HTTPError as e:
            if e.code == 404:
                logger.debug(f"No taxonomy.csv found for {model_id} (404)")
            else:
                logger.warning(f"Failed to download taxonomy.csv for {model_id}: HTTP {e.code}")
        except Exception as e:
            logger.warning(f"Failed to download taxonomy.csv for {model_id}: {e}")

    def write_manifest(
        self, model_type: str, manifest_data: dict[str, Any]
    ) -> str:
        """
        Idempotently sync the local manifest.json for a model with the
        central catalog. Creates the model directory and downloads
        taxonomy.csv on first appearance, refreshes the manifest in place
        when the catalog has newer content (citation, URL, license,
        friendly_name etc.), no-ops when content is identical.

        Returns one of "created" / "updated" / "unchanged" so the caller
        can decide what to surface in the UI. Never raises; logs and
        returns "unchanged" on unexpected I/O errors so a single bad
        entry can't take the whole sync down.
        """
        model_id = manifest_data["model_id"]
        model_dir = self.models_dir / model_type / model_id
        manifest_path = model_dir / "manifest.json"
        is_new_dir = not model_dir.exists()

        try:
            # Compare existing content. Identical bytes-or-equivalent
            # JSON means nothing to do; the catalog hasn't moved.
            if manifest_path.exists():
                try:
                    with open(manifest_path) as f:
                        existing = json.load(f)
                    if existing == manifest_data:
                        return "unchanged"
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning(
                        f"Existing manifest at {manifest_path} unreadable, "
                        f"will overwrite: {e}"
                    )

            model_dir.mkdir(parents=True, exist_ok=True)
            with open(manifest_path, "w") as f:
                json.dump(manifest_data, f, indent=2)

            if is_new_dir:
                logger.info(f"Created manifest stub for {model_type}/{model_id}")
                # Taxonomy ships alongside the model on first appearance.
                # No re-download on refresh: taxonomy lives in the HF
                # repo, not the catalog, and catalog refresh shouldn't
                # imply HF revision drift (that's the model-revision
                # drift TODO).
                if model_type == "cls":
                    self.download_taxonomy(model_id, model_dir)
                return "created"

            logger.info(f"Refreshed manifest for {model_type}/{model_id}")
            return "updated"

        except Exception as e:
            logger.error(
                f"Failed to sync manifest for {model_type}/{model_id}: {e}",
                exc_info=True,
            )
            return "unchanged"

    async def sync(self) -> dict[str, Any]:
        """
        Fetch the central catalog, then for every entry write the local
        manifest.json: create on first appearance, refresh in place when
        the catalog moved (citation, URL, license, friendly_name, etc.),
        no-op when identical. Idempotent: safe to run on every startup.

        Returns:
            {
                "new_models":       [{"model_id", "friendly_name", "emoji"}, ...],
                "refreshed_models": [{"model_id", "friendly_name"}, ...],
                "checked_at":       "<UTC ISO timestamp>",
                "error":            "<message>" (only if fetch failed),
            }

        Note: async so the lifespan startup task doesn't block boot.
        """
        result: dict[str, Any] = {
            "new_models": [],
            "refreshed_models": [],
            "checked_at": datetime.now(UTC).isoformat(),
        }

        try:
            catalog = self.fetch_catalog()
            if catalog is None:
                result["error"] = "Failed to fetch catalog"
                return result

            local_models = self.get_local_models()
            total_local = sum(len(s) for s in local_models.values())
            is_fresh_install = total_local == 0

            for model_type in ["det", "cls", "emb"]:
                for manifest_data in catalog["models"].get(model_type, []):
                    state = self.write_manifest(model_type, manifest_data)

                    if state == "created" and not is_fresh_install:
                        # Surface as "new model" toast on existing
                        # installs only. Fresh installs just want the
                        # catalog to populate silently.
                        result["new_models"].append(
                            {
                                "model_id": manifest_data["model_id"],
                                "friendly_name": manifest_data.get(
                                    "friendly_name", manifest_data["model_id"]
                                ),
                                "emoji": manifest_data.get("emoji", "🤖"),
                            }
                        )
                    elif state == "updated":
                        result["refreshed_models"].append(
                            {
                                "model_id": manifest_data["model_id"],
                                "friendly_name": manifest_data.get(
                                    "friendly_name", manifest_data["model_id"]
                                ),
                            }
                        )

            if is_fresh_install:
                # On first launch every entry is "created"; no point
                # listing them; the setup wizard handles weight downloads.
                total_entries = sum(
                    len(catalog["models"].get(t, []))
                    for t in ("det", "cls", "emb")
                )
                logger.info(
                    f"Model catalog sync complete: catalog initialized "
                    f"({total_entries} entries)"
                )
            else:
                logger.info(
                    f"Model catalog sync complete: "
                    f"{len(result['new_models'])} new, "
                    f"{len(result['refreshed_models'])} refreshed"
                )

            return result

        except Exception as e:
            logger.error(f"Model catalog sync failed: {e}", exc_info=True)
            result["error"] = str(e)
            return result
