"""The large-database seed still builds a database the pages can read.

It runs rarely and by hand, so a schema change would otherwise break it
unnoticed until the next time someone measures. A tiny run here keeps it
honest: every table the dashboard, Labels page and sort worker read must
come out filled.
"""

import os
import sqlite3
import subprocess
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2]
SCRIPT = BACKEND / "scripts" / "seed_large_image_database.py"
TAXONOMY = BACKEND / "tests" / "data" / "seed_taxonomy.csv"


def _seed(data_dir: Path, home: Path) -> subprocess.CompletedProcess:
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("ADDAXAI_DATABASE_URL", "ADDAXAI_MODELS_DIR")
    }
    env["ADDAXAI_USER_DATA_DIR"] = str(data_dir)
    env["HOME"] = str(home)
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--classification-model",
            "TEST-MODEL",
            "--taxonomy-csv",
            str(TAXONOMY),
            "--scale",
            "0.0005",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_a_tiny_seed_fills_every_table_the_pages_read(tmp_path: Path) -> None:
    data_dir = tmp_path / "seed"
    result = _seed(data_dir, home=tmp_path / "home")
    assert result.returncode == 0, result.stdout + result.stderr

    conn = sqlite3.connect(data_dir / "addaxai.db")

    def count(sql: str) -> int:
        return conn.execute(sql).fetchone()[0]

    assert count("SELECT count(*) FROM projects") == 5
    for table in (
        "files",
        "detections",
        "detection_embeddings",
        "events",
        "event_files",
        "event_observations",
        "sqlite_stat1",
    ):
        assert count(f"SELECT count(*) FROM {table}") > 0, table
    # Every box points at a label row, or the filter tree cannot find it.
    assert count("SELECT count(*) FROM detections WHERE label_taxonomy_id IS NULL") == 0
    assert count("SELECT count(*) FROM detections WHERE label IS NOT NULL") > 0
    assert count("SELECT count(*) FROM files WHERE observation_type = 'unclassified'") == 0
    assert count("SELECT count(*) FROM detections WHERE confidence < 0.01") > 0
    assert count("SELECT count(*) FROM files WHERE verified") > 0
    conn.close()


def test_the_seed_refuses_the_real_data_folder(tmp_path: Path) -> None:
    home = tmp_path / "home"
    result = _seed(home / "AddaxAI", home=home)
    assert result.returncode != 0
    assert "Refusing to seed the real ~/AddaxAI" in result.stderr
    assert not (home / "AddaxAI" / "addaxai.db").exists()
