"""
The one channel the backend has to tell the user why it would not start.

When `init_db()` refuses a database the process exits before the API or
the frontend exist, so nothing inside the app can show the reason. The
lifespan writes it here instead; Electron reads the file after the
backend dies and renders it on the startup error page.

Electron deletes the file just before it spawns the backend, so whatever
is here always belongs to the current launch and this module never has
to clear it.
"""

from app.core.config import Settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)

STARTUP_ERROR_FILENAME = ".startup-error.txt"

# Shown when startup died of something we have no specific wording for
# (a full disk, a permissions problem, a migration that crashed). The
# real exception is in the log; the page just has to stop the user
# guessing that the app is broken beyond repair.
GENERIC_STARTUP_FAILURE = (
    "The database could not be opened.\n\n"
    "The log has the details. If this keeps happening, restore a backup "
    "or start fresh with an empty database."
)


def write_startup_error(settings: Settings, message: str) -> None:
    """Record why startup failed, for the Electron error page.

    Best effort. We are already on the way out, and failing to write
    this file must never replace the real error with an IO error.
    """
    try:
        path = settings.user_data_dir / STARTUP_ERROR_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(message, encoding="utf-8")
    except Exception as e:
        logger.error(f"Could not write startup error file: {e}", exc_info=True)
