"""
Run the pending migrations on a copy of every database backup on this
machine, and check that no row was lost.

Why this exists: the CI tests prove a migration on an empty database
(`test_migrations.py`) and on one synthetic row per table
(`test_migration_keeps_rows.py`). Real databases are messier: years of
rows, old NULLs, whatever an earlier bug left behind. `~/AddaxAI/backups/`
holds real databases from the last days and from before every upgrade,
each stamped at the revision it was made at. This script upgrades a copy
of each and compares rows before and after. Nothing under `~/AddaxAI` is
touched.

Run it before releasing a version that ships a migration:

    backend/venv/bin/python backend/scripts/check_migration_on_backups.py

Exit code 0 means every backup came through, or there were no backups to
try (said so on the last line: then the CI tests are the only evidence).
Exit code 1 means a backup did not come through; do not release until
you know why.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

REPO_BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_BACKEND))

BACKUPS = Path(os.environ.get("ADDAXAI_USER_DATA_DIR", Path.home() / "AddaxAI")) / "backups"


def main() -> int:
    from sqlalchemy import create_engine

    import app.core.config
    import app.db.base
    import app.db.migrations
    from app.core.config import Settings
    from app.db.migration_check import lost_rows, snapshot, sqlite_health
    from app.db.migrations import (
        get_current_revision,
        get_head_revision,
        schema_problems,
        upgrade_to_head,
    )

    backups = sorted(BACKUPS.glob("*.db")) if BACKUPS.is_dir() else []
    if not backups:
        print(f"No backups in {BACKUPS}: nothing to try, the CI tests are the only evidence.")
        return 0

    head = get_head_revision()
    failures = 0
    with tempfile.TemporaryDirectory(prefix="addaxai-migration-check-") as tmp:
        for src in backups:
            dst = Path(tmp) / src.name
            shutil.copy(src, dst)
            settings = Settings(user_data_dir=Path(tmp), database_url=f"sqlite:///{dst}")
            for module in (app.core.config, app.db.base, app.db.migrations):
                module.get_settings = lambda s=settings: s  # noqa: E731

            engine = create_engine(settings.database_url, future=True)
            start_rev = get_current_revision(engine)
            engine.dispose()
            if start_rev == head:
                print(f"{src.name}: already at {head}, skipped")
                continue

            before = snapshot(str(dst))
            started = time.time()
            try:
                upgrade_to_head()
            except Exception as exc:  # noqa: BLE001 - report and keep going
                failures += 1
                print(f"{src.name}: FAILED during upgrade from {start_rev}: {exc}")
                continue
            elapsed = time.time() - started

            engine = create_engine(settings.database_url, future=True)
            problems = schema_problems(engine)
            end_rev = get_current_revision(engine)
            engine.dispose()
            problems += sqlite_health(str(dst))
            problems += lost_rows(before, snapshot(str(dst)))

            rows = sum(t.count for t in before.values())
            if problems:
                failures += 1
                print(f"{src.name}: FAILED ({start_rev} -> {end_rev}, {rows} rows)")
                for p in problems:
                    print(f"    {p}")
            else:
                print(f"{src.name}: ok ({start_rev} -> {end_rev}, {rows} rows, {elapsed:.1f}s)")
            dst.unlink(missing_ok=True)

    print(f"{len(backups)} backup(s), {failures} failed.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
