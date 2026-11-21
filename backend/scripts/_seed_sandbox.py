"""
The one way a seed script opens its database.

Seeds write millions of rows, so they must never touch the real data
folder. This refuses to run unless ``ADDAXAI_USER_DATA_DIR`` points at a
throwaway folder, then builds the schema with the app's own ``init_db``,
so the backend accepts the result.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

REPO_BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_BACKEND))

if TYPE_CHECKING:
    from sqlalchemy.orm import Session


def open_seed_sandbox() -> Session:
    """Return a session on a fresh or existing throwaway database.

    Exits with a message when ``ADDAXAI_USER_DATA_DIR`` is unset or is
    the real ``~/AddaxAI``.
    """
    data_dir = os.environ.get("ADDAXAI_USER_DATA_DIR")
    if not data_dir:
        raise SystemExit("Set ADDAXAI_USER_DATA_DIR to a throwaway folder first.")
    if Path(data_dir).resolve() == (Path.home() / "AddaxAI").resolve():
        raise SystemExit("Refusing to seed the real ~/AddaxAI.")
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    from app.db.base import get_session_factory, init_db

    init_db()
    return get_session_factory()()
