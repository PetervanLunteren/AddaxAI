"""The once-per-install licence step for models that run on ultralytics.

Six catalog models load their weights through the ultralytics package,
which is licensed under AGPL-3.0. AddaxAI is MIT and does not install that
package for anyone who has not asked for it: those models run in their own
environment, ``env-ultralytics``, and the app builds it only after the user
has accepted the licence once. The environment name is the key, so a model
needs the step exactly when its catalog entry says ``"env": "ultralytics"``.
There is no list of model ids to keep in step.

The acceptance is a self-explaining file in the user data dir, the pattern
of the revocation opt-out in ``environment_manager.py``. A file rather than
a setting because the two doors that build the environment, the prepare
route and the headless ``--setup`` command, both read it with nothing
threaded through. Nothing else reads it: no status endpoint reports it,
because the prepare route's refusal is the signal the UI acts on.

"Reset app" removes the file (``setup._WIPE_FILES``): reset means a fresh
install, and a fresh install asks again.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.ml.schemas.model_manifest import ModelManifest

logger = get_logger(__name__)

# The gate's key. It must equal the env folder name under app/ml/envs/;
# tests/ml/test_agpl_acceptance.py holds that.
AGPL_ENV: Final = "ultralytics"
AGPL_LICENCE_URL: Final = "https://www.ultralytics.com/license"
AGPL_MARKER_FILENAME: Final = ".agpl-accepted"
# What the prepare route's refusal is called on the wire, so the UI can
# offer the licence step instead of showing an error.
AGPL_ERROR_KIND: Final = "agpl_acceptance_required"
CLI_ACCEPT_FLAG: Final = "--accept-agpl"

_MARKER_NOTE = """\
The user of this AddaxAI installation accepted the licence of the
ultralytics package (AGPL-3.0, {url}) on {created}.

AddaxAI installs ultralytics only for the models that need it, in their
own environment (env-ultralytics). Delete this file to be asked again
before that environment is built.
"""


def agpl_marker_path() -> Path:
    """Absolute path of the acceptance marker, whether or not it exists."""
    return get_settings().user_data_dir / AGPL_MARKER_FILENAME


def agpl_accepted() -> bool:
    return agpl_marker_path().is_file()


def accept_agpl() -> Path:
    """Record the acceptance. Idempotent, rewritten on every call."""
    marker = agpl_marker_path()
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        _MARKER_NOTE.format(
            url=AGPL_LICENCE_URL, created=datetime.now(UTC).strftime("%Y-%m-%d")
        ),
        encoding="utf-8",
    )
    logger.info(f"ultralytics AGPL-3.0 licence accepted; marker written to {marker}")
    return marker


def needs_agpl_acceptance(manifest: ModelManifest) -> bool:
    """True when this model runs on ultralytics and nobody accepted that."""
    return manifest.env == AGPL_ENV and not agpl_accepted()


def agpl_refusal_message(model_id: str) -> str:
    """The one sentence the prepare route and the CLI both give."""
    return (
        f"{model_id} runs on the ultralytics package, which is licensed under "
        f"AGPL-3.0 ({AGPL_LICENCE_URL}). Accept that licence before this model "
        f"is set up."
    )
