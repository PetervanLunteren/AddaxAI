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
from app.ml.schemas.model_manifest import resolve_hf_repo

logger = get_logger(__name__)

# Names of envs whose drift we surface in the toast. Kept here rather
# than in EnvironmentManager because env_manager treats env_name as an
# opaque parameter; this list tracks which ones the app actually ships.
_DRIFT_CHECKED_ENVS: tuple[str, ...] = (
    "addaxai-base",
    "pytorch",
    "pywildlife",
    "tensorflow-v1",
    "tensorflow-v2",
)


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
            "https://raw.githubusercontent.com/PetervanLunteren/AddaxAI/main/models.json"
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

    def download_taxonomy(
        self, model_id: str, model_dir: Path, hf_repo: str | None = None
    ) -> None:
        """
        Download taxonomy.csv from HuggingFace repo.

        Args:
            model_id: Model ID (used to construct HF repo URL)
            model_dir: Local directory to save taxonomy.csv
            hf_repo: Explicit repo override from the manifest. None means
                     follow the `<DEFAULT_HF_ORG>/<model_id>` convention.

        Raises:
            Never raises - logs errors and continues
        """
        taxonomy_url = (
            f"https://huggingface.co/{resolve_hf_repo(model_id, hf_repo)}"
            f"/resolve/main/taxonomy.csv?download=true"
        )
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
        central catalog. Creates the model directory, refreshes the
        manifest in place when the catalog has newer content (citation,
        URL, license, friendly_name etc.), no-ops when content is
        identical, and fetches taxonomy.csv whenever it is missing.

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
            # JSON means the catalog hasn't moved and the file is left
            # alone, but we still fall through to the taxonomy check.
            unchanged = False
            if manifest_path.exists():
                try:
                    with open(manifest_path) as f:
                        existing = json.load(f)
                    unchanged = existing == manifest_data
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning(
                        f"Existing manifest at {manifest_path} unreadable, "
                        f"will overwrite: {e}"
                    )

            if not unchanged:
                model_dir.mkdir(parents=True, exist_ok=True)
                with open(manifest_path, "w") as f:
                    json.dump(manifest_data, f, indent=2)

            # Taxonomy ships in the HF repo, not the catalog, so fetch it
            # whenever it is missing rather than only on first creation.
            # This check must sit outside the `unchanged` branch: a stub
            # whose taxonomy never landed (model published before its
            # taxonomy.csv existed, or first synced while offline) has a
            # perfectly unchanged manifest, so an early return would leave
            # it broken forever, on a flat label list with no rollup.
            # Present file means no request, so once a model's taxonomy is
            # on disk this costs nothing; a repo that genuinely has no
            # taxonomy.csv pays one cheap 404 per launch.
            if model_type == "cls" and not (model_dir / "taxonomy.csv").exists():
                self.download_taxonomy(
                    model_id, model_dir, manifest_data.get("hf_repo")
                )

            if unchanged:
                return "unchanged"

            if is_new_dir:
                logger.info(f"Created manifest stub for {model_type}/{model_id}")
                return "created"

            logger.info(f"Refreshed manifest for {model_type}/{model_id}")
            return "updated"

        except Exception as e:
            logger.error(
                f"Failed to sync manifest for {model_type}/{model_id}: {e}",
                exc_info=True,
            )
            return "unchanged"

    def check_model_drift(
        self, model_type: str, manifest_data: dict[str, Any]
    ) -> bool | None:
        """
        Compare the local manifest's recorded `hf_revision_sha` to the
        live HuggingFace commit SHA for the same repo.

        Returns:
            True  if the sentinel SHA disagrees with the upstream
                  (drift; user should re-download).
            False if they agree (in sync).
            None  if drift can't be evaluated: legacy install with no
                  recorded SHA, no `hf_repo` declared, network failure,
                  or any other transient issue. Treat as "unknown but
                  valid" and skip.

        Never raises: HF auth / network / 404 errors collapse to None
        so a drift pass can't take down startup.
        """
        from huggingface_hub import HfApi
        from huggingface_hub.utils import HfHubHTTPError

        model_id = manifest_data["model_id"]
        model_dir = self.models_dir / model_type / model_id
        manifest_path = model_dir / "manifest.json"
        if not manifest_path.exists():
            return None

        try:
            with open(manifest_path) as f:
                local = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(
                f"Could not read local manifest at {manifest_path}: {e}"
            )
            return None

        recorded = local.get("hf_revision_sha")
        if not recorded:
            # Legacy install. Skip per "unknown but valid" rule.
            return None

        hf_repo = resolve_hf_repo(model_id, local.get("hf_repo"))
        try:
            info = HfApi().model_info(hf_repo)
            remote = getattr(info, "sha", None)
        except HfHubHTTPError as e:
            logger.warning(
                f"HF model_info({hf_repo}) HTTP error during drift check: {e}"
            )
            return None
        except Exception as e:
            logger.warning(
                f"HF model_info({hf_repo}) failed during drift check: {e}"
            )
            return None

        if not remote:
            return None

        return recorded != remote

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
                "drifted_models":   [{"model_id", "friendly_name", "emoji"}, ...],
                "drifted_envs":     [{"env_name"}, ...],
                "checked_at":       "<UTC ISO timestamp>",
                "error":            "<message>" (only if fetch failed),
            }

        Note: async so the lifespan startup task doesn't block boot.
        """
        result: dict[str, Any] = {
            "new_models": [],
            "refreshed_models": [],
            "drifted_models": [],
            "drifted_envs": [],
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

                    # Drift check: compare the local manifest's recorded
                    # HF revision SHA to the upstream. Only models that
                    # have actually been downloaded carry a recorded
                    # SHA, so this naturally skips catalog-only stubs.
                    # Skip on fresh installs too: there's nothing on
                    # disk that could be drifted.
                    if not is_fresh_install:
                        drifted = self.check_model_drift(model_type, manifest_data)
                        if drifted:
                            result["drifted_models"].append(
                                {
                                    "model_id": manifest_data["model_id"],
                                    "friendly_name": manifest_data.get(
                                        "friendly_name", manifest_data["model_id"]
                                    ),
                                    "emoji": manifest_data.get("emoji", "🤖"),
                                }
                            )

            # Env drift: hash each shipped env's bundled YAML and
            # compare to the sentinel written when the env was built.
            # Done outside the catalog loop because envs are shipped
            # by the app, not by the central models.json.
            if not is_fresh_install:
                from app.ml.environment_manager import EnvironmentManager
                env_manager = EnvironmentManager()
                for env_name in _DRIFT_CHECKED_ENVS:
                    try:
                        drifted = env_manager.check_yaml_drift(env_name)
                    except Exception as e:
                        logger.warning(
                            f"Env drift check for {env_name} raised: {e}"
                        )
                        drifted = None
                    if drifted:
                        result["drifted_envs"].append({"env_name": env_name})

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
                    f"{len(result['refreshed_models'])} refreshed, "
                    f"{len(result['drifted_models'])} model(s) drifted, "
                    f"{len(result['drifted_envs'])} env(s) drifted"
                )

            return result

        except Exception as e:
            logger.error(f"Model catalog sync failed: {e}", exc_info=True)
            result["error"] = str(e)
            return result
