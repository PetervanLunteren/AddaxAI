"""
Model catalog updater - fetches central manifest and creates stubs for new models.

Following DEVELOPERS.md principles:
- Fail silently if offline (non-critical operation)
- Never overwrite existing model directories
- Log all operations for debugging
"""

import asyncio
import json
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.ml.hf_downloader import hf_auth_headers
from app.ml.model_storage import find_stale_files
from app.ml.schemas.model_manifest import resolve_hf_repo

logger = get_logger(__name__)


def _bundled_catalog_path() -> Path | None:
    """
    The models.json shipped inside the app, or None if it is not there.

    Two locations, the same two `app/__init__.py` reads VERSION from: the
    PyInstaller bundle root when frozen, the repo root when running from
    source.
    """
    candidates: list[Path] = []
    if hasattr(sys, "_MEIPASS"):
        candidates.append(Path(sys._MEIPASS) / "models.json")
    candidates.append(Path(__file__).resolve().parents[3] / "models.json")
    for path in candidates:
        if path.is_file():
            return path
    return None


def _validate_catalog(catalog: Any) -> dict[str, Any] | None:
    """The catalog, or None when it is not shaped like one."""
    if not isinstance(catalog, dict) or "models" not in catalog:
        logger.error("Invalid catalog structure: missing 'models'")
        return None
    if "det" not in catalog["models"] or "cls" not in catalog["models"]:
        logger.error("Invalid catalog structure: missing 'det' or 'cls' in models")
        return None
    return catalog

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


def find_drifted_envs() -> list[dict[str, str]]:
    """
    Which shipped envs no longer match the YAML this app version carries.

    Reads a 64-byte sentinel and hashes a small YAML per env, all local,
    so this is cheap enough to call per request. `GET /api/ml/updates`
    does exactly that rather than serving the startup snapshot: rebuilding
    an env fixes the sentinel, and a user who then reloads the window must
    not be told again to rebuild what they just rebuilt.

    An env that is not installed, or predates the sentinel, reports
    nothing (`check_yaml_drift` returns None) and is skipped.
    """
    from app.ml.environment_manager import EnvironmentManager

    env_manager = EnvironmentManager()
    drifted: list[dict[str, str]] = []
    for env_name in _DRIFT_CHECKED_ENVS:
        try:
            has_drifted = env_manager.check_yaml_drift(env_name)
        except Exception as e:
            logger.warning(f"Env drift check for {env_name} raised: {e}")
            continue
        if has_drifted:
            drifted.append({"env_name": env_name})
    return drifted


class ModelCatalogUpdater:
    """
    Fetches central model catalog and creates local directory stubs for new models.

    Only creates manifest.json files - does not download weights.
    """

    def __init__(self, models_dir: Path | None = None, catalog_url: str | None = None):
        """
        Initialize catalog updater.

        Args:
            models_dir: Directory where models are stored (default: settings.models_dir)
            catalog_url: URL to fetch catalog from (default: from config)
        """
        self.models_dir = models_dir or get_settings().models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Default to GitHub raw URL
        self.catalog_url = catalog_url or (
            "https://raw.githubusercontent.com/PetervanLunteren/AddaxAI/main/models.json"
        )

    def fetch_catalog(self, timeout: int = 2) -> dict[str, Any] | None:
        """
        Fetch the model catalog, falling back to the copy shipped in the app.

        The bundled copy is not a cache, it is what keeps a blocked
        catalog host from emptying the app: manifest.json is written from
        this catalog and nothing else writes it, and ManifestManager
        skips any model directory without one. So a first launch behind a
        firewall used to download the weights and then show no models at
        all. The bundled file lists what this app version shipped with,
        which is the honest answer when upstream cannot be reached.

        Args:
            timeout: Request timeout in seconds (default: 2)

        Returns:
            Catalog dict, or None when neither source yields a valid one

        Raises:
            Never raises - logs errors and returns None
        """
        try:
            logger.info(f"Fetching model catalog from {self.catalog_url}")

            with urllib.request.urlopen(self.catalog_url, timeout=timeout) as response:
                data = response.read()

            catalog = _validate_catalog(json.loads(data))
            if catalog is not None:
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
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse model catalog JSON: {e}")
        except Exception as e:
            logger.error(f"Unexpected error fetching model catalog: {e}", exc_info=True)

        return self._bundled_catalog()

    def _bundled_catalog(self) -> dict[str, Any] | None:
        """The catalog shipped with the app, or None if it cannot be read."""
        path = _bundled_catalog_path()
        if path is None:
            logger.error("No bundled models.json to fall back on")
            return None
        try:
            catalog = _validate_catalog(json.loads(path.read_text(encoding="utf-8")))
        except Exception as e:
            logger.error(f"Failed to read bundled catalog {path}: {e}", exc_info=True)
            return None
        if catalog is not None:
            logger.warning(f"Using the model catalog shipped with the app: {path}")
        return catalog

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
            f"{get_settings().hf_base_url}/{resolve_hf_repo(model_id, hf_repo)}"
            f"/resolve/main/taxonomy.csv?download=true"
        )
        taxonomy_path = model_dir / "taxonomy.csv"

        try:
            logger.info(f"Downloading taxonomy.csv from {taxonomy_url}")

            request = urllib.request.Request(taxonomy_url, headers=hf_auth_headers())
            with urllib.request.urlopen(request, timeout=5) as response:
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

    async def _find_stale_files(
        self, model_type: str, manifest_data: dict[str, Any]
    ) -> list[str] | None:
        """
        Repo-relative paths of this model's local files that no longer
        match HuggingFace, or None when the model is not installed or the
        question could not be answered.

        Only models the user actually downloaded can be stale, so a
        catalog stub (a manifest with no weights next to it) is skipped
        without any HTTP call. Deliberately not `check_weights_ready`:
        that answers "can inference run", and reports False for an install
        that has its weights but is missing a support file, which is
        exactly the install whose missing files this should restore.
        """
        model_dir = self.models_dir / model_type / manifest_data["model_id"]
        if not (model_dir / manifest_data["model_fname"]).is_file():
            return None

        hf_repo = resolve_hf_repo(
            manifest_data["model_id"], manifest_data.get("hf_repo")
        )
        # One blocking HTTPS call plus a handful of local file reads, per
        # installed model. Off the event loop so a slow or black-holed
        # network cannot make the whole API unresponsive during startup.
        return await asyncio.to_thread(find_stale_files, model_dir, hf_repo)

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
                    installed models with at least one file that no longer
                    matches upstream. The file names go to the log rather
                    than over the wire: nothing renders them, and this
                    snapshot goes stale the moment upstream moves, so the
                    update endpoint recomputes the list itself.
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

                    # Compare the installed files against the upstream repo.
                    # Skipped on fresh installs: nothing is on disk yet.
                    if not is_fresh_install:
                        stale = await self._find_stale_files(model_type, manifest_data)
                        if stale:
                            logger.info(
                                f"{model_type}/{manifest_data['model_id']} has "
                                f"{len(stale)} file(s) to update: {', '.join(stale)}"
                            )
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
            # by the app, not by the central models.json. Recorded here
            # only so the count reaches the log; the endpoint recomputes
            # it per request so a rebuild takes effect immediately.
            if not is_fresh_install:
                result["drifted_envs"] = find_drifted_envs()

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
                    f"{len(result['drifted_models'])} model(s) with files to update, "
                    f"{len(result['drifted_envs'])} env(s) drifted"
                )

            return result

        except Exception as e:
            logger.error(f"Model catalog sync failed: {e}", exc_info=True)
            result["error"] = str(e)
            return result
