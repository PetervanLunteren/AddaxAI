"""
Collect size and statistics numbers from an AddaxAI database.

Read-only: it opens the database in read-only mode and changes nothing.
The result is a small text file with numbers only. It holds no file names,
no folder paths, no project or deployment names and no species labels.

Run it with the Python that AddaxAI installed (Windows, one line):

    "%USERPROFILE%\\AddaxAI\\envs\\env-addaxai-base\\python.exe" addaxai-db-stats.py

The database is found by itself. To point at another one, give its path:

    ... python.exe addaxai-db-stats.py D:\\somewhere\\addaxai.db
"""

import os
import sqlite3
import sys
import time
import traceback
from pathlib import Path

OUT_NAME = "addaxai-db-stats.txt"

# Each step: a title and the query that fills it. A step that fails is noted
# in the output and the rest still runs.
STEPS = [
    (
        "Query planner statistics",
        "SELECT tbl, idx, stat FROM sqlite_stat1 ORDER BY tbl, idx",
    ),
    (
        "Projects (numbered, no names)",
        "SELECT dense_rank() OVER (ORDER BY p.id) AS project, p.mode, "
        "count(d.id) AS deployments "
        "FROM projects p LEFT JOIN deployments d ON d.project_id = p.id "
        "GROUP BY p.id ORDER BY project",
    ),
    (
        "Files per deployment (numbered, no names)",
        "SELECT dense_rank() OVER (ORDER BY d.project_id) AS project, "
        "row_number() OVER (ORDER BY d.project_id, d.id) AS deployment, "
        "count(f.id) AS files, "
        "sum(f.file_type = 'video') AS videos "
        "FROM deployments d LEFT JOIN files f ON f.deployment_id = d.id "
        "GROUP BY d.id ORDER BY deployment",
    ),
    (
        "Detections per deployment (numbered, no names)",
        "SELECT dense_rank() OVER (ORDER BY d.project_id) AS project, "
        "row_number() OVER (ORDER BY d.project_id, d.id) AS deployment, "
        "count(det.id) AS detections, "
        "sum(det.confidence >= 0.2) AS conf_020_or_more, "
        "sum(det.verified) AS verified, "
        "count(DISTINCT det.label_taxonomy_id) AS distinct_labels "
        "FROM deployments d "
        "LEFT JOIN files f ON f.deployment_id = d.id "
        "LEFT JOIN detections det ON det.file_id = f.id "
        "GROUP BY d.id ORDER BY deployment",
    ),
    (
        "Embeddings per model",
        "SELECT embedding_model_id, dimension, count(*) AS n "
        "FROM detection_embeddings GROUP BY 1, 2",
    ),
    (
        "Size per table and index",
        "SELECT name, count(*) AS pages, sum(pgsize) AS bytes, "
        "sum(unused) AS unused_bytes "
        "FROM dbstat GROUP BY name ORDER BY bytes DESC",
    ),
]
# General, rows per table, then the queries above.
TOTAL_STEPS = len(STEPS) + 2


class Progress:
    """Print a line that keeps moving, so a long step never looks stuck."""

    def __init__(self) -> None:
        self.step = 0
        self.title = ""
        self.started = time.perf_counter()
        self.last_print = 0.0

    def start(self, title: str) -> None:
        self.finish()
        self.step += 1
        self.title = title
        self.started = time.perf_counter()
        self.last_print = 0.0
        self._show(force=True)

    def tick(self) -> int:
        # Called by SQLite while a query runs. Returning 0 lets it continue.
        self._show(force=False)
        return 0

    def finish(self) -> None:
        if self.step:
            seconds = time.perf_counter() - self.started
            line = f"Step {self.step} of {TOTAL_STEPS}: {self.title} - done in {seconds:.0f} s"
            print("\r" + line.ljust(78), flush=True)

    def _show(self, force: bool) -> None:
        now = time.perf_counter()
        if not force and now - self.last_print < 1.0:
            return
        self.last_print = now
        seconds = now - self.started
        line = f"Step {self.step} of {TOTAL_STEPS}: {self.title} - working, {seconds:.0f} s"
        print("\r" + line.ljust(78), end="", flush=True)


def find_database() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    data_dir = os.environ.get("ADDAXAI_USER_DATA_DIR")
    base = Path(data_dir) if data_dir else Path.home() / "AddaxAI"
    return base / "addaxai.db"


def open_read_only(db_path: Path) -> sqlite3.Connection:
    """Open without any way to write. Try the normal read-only mode first."""
    uri = db_path.resolve().as_uri()
    try:
        conn = sqlite3.connect(uri + "?mode=ro", uri=True)
        conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
        return conn
    except sqlite3.OperationalError:
        # Happens when the helper files of the database cannot be made, for
        # example in a folder without write rights. Safe while AddaxAI is closed.
        conn = sqlite3.connect(uri + "?immutable=1", uri=True)
        conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
        return conn


def collect(db_path: Path, lines: list, progress: Progress) -> None:
    conn = open_read_only(db_path)
    conn.set_progress_handler(progress.tick, 20000)

    progress.start("General")
    lines.append("## General")
    lines.append(f"sqlite_version\t{sqlite3.sqlite_version}")
    lines.append(f"python_version\t{sys.version.split()[0]}")
    lines.append(f"platform\t{sys.platform}")
    lines.append(f"file_size_bytes\t{db_path.stat().st_size}")
    for pragma in ("page_size", "page_count", "freelist_count", "auto_vacuum"):
        value = conn.execute("PRAGMA " + pragma).fetchone()[0]
        lines.append(f"{pragma}\t{value}")
    try:
        revision = conn.execute("SELECT version_num FROM alembic_version").fetchone()[0]
    except sqlite3.Error as err:
        revision = f"unknown ({err})"
    lines.append(f"schema_version\t{revision}")

    progress.start("Rows per table")
    lines.append("")
    lines.append("## Rows per table")
    lines.append("table\trows")
    names = [
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
    ]
    for name in names:
        try:
            rows = conn.execute('SELECT count(*) FROM "' + name + '"').fetchone()[0]
        except sqlite3.Error as err:
            rows = f"failed: {err}"
        lines.append(f"{name}\t{rows}")

    for title, sql in STEPS:
        progress.start(title)
        lines.append("")
        lines.append(f"## {title}")
        started = time.perf_counter()
        try:
            cur = conn.execute(sql)
            lines.append("\t".join(col[0] for col in cur.description))
            for row in cur:
                lines.append("\t".join("" if v is None else str(v) for v in row))
        except sqlite3.Error as err:
            lines.append(f"This step failed: {err}")
        lines.append(f"(took {time.perf_counter() - started:.1f} s)")

    progress.finish()
    conn.close()


def main() -> int:
    out_path = Path(__file__).resolve().parent / OUT_NAME
    db_path = find_database()
    lines = []
    ok = True

    print("AddaxAI database numbers")
    print(f"Database: {db_path}")
    print("This only reads. A large database can take a minute or two.")
    print("The counter below keeps moving while it works. Please do not close the window.")
    print()

    if not db_path.is_file():
        ok = False
        lines.append(f"No database found at {db_path}")
        print(f"No database found at {db_path}")
    else:
        try:
            collect(db_path, lines, Progress())
        except Exception:
            ok = False
            lines.append("")
            lines.append("## The script stopped with an error")
            lines.append(traceback.format_exc())
            print("\nSomething went wrong. The details are in the file below.")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print()
    print("Finished." if ok else "Not finished, but the file still helps.")
    print("Please mail this file to Peter:")
    print(out_path)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
