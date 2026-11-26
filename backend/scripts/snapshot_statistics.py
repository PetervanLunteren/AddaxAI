"""
Snapshot every statistic and export table of a database, for a before/after diff.

Runs the app in-process against a throwaway copy of a database and writes,
per project, every statistics response, the project / deployment / site
numbers, the export tables as CSV and the request timings. Two runs on two
versions of the app, on two fresh copies of the same database, are then
compared with ``diff -r``.

    cp ~/AddaxAI/addaxai.db "$S/addaxai.db"
    ADDAXAI_USER_DATA_DIR="$S" ADDAXAI_DISABLE_MODEL_UPDATES=1 \\
        backend/venv/bin/python backend/scripts/snapshot_statistics.py --out "$OUT"

It refuses to run on the real ``~/AddaxAI`` (``_seed_sandbox``). It only
sends GET requests and reads, but opening the database migrates the copy
to the current schema, so start every run from a fresh copy.

Timings go to ``timings.json``, outside the per-project folders, so a diff
of the numbers is not drowned in timing noise.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from _seed_sandbox import open_seed_sandbox

# Header names a later version renamed, mapped back to the names of the
# baseline, so the diff shows changed numbers, not changed spellings.
RENAMED_KEYS: dict[str, str] = {}

TIMING_RUNS = 3


def _baseline_key(key: str) -> str:
    """A response key in the baseline's words: the stored unit was called an
    event before it became a sequence (``sequence_count`` was ``event_count``)."""
    return key.replace("sequences", "events").replace("sequence", "event")


def _normalise(value: Any) -> Any:
    if isinstance(value, dict):
        return {_baseline_key(k): _normalise(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalise(v) for v in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_normalise(payload), indent=1, sort_keys=True) + "\n")


def _grouping_table(db: Any) -> tuple[str, str]:
    """The stored grouping table and its observation table, by either name."""
    from sqlalchemy import inspect

    tables = set(inspect(db.get_bind()).get_table_names())
    if "sequences" in tables:
        return "sequences", "sequence_observations"
    return "events", "event_observations"


def _oracle(db: Any, project_id: str) -> dict[str, int]:
    """Two counts the statistics must be explainable from: stored groups with
    an observation, and stored (group, species) pairs."""
    from sqlalchemy import text

    groups, observations = _grouping_table(db)
    fk = "sequence_id" if groups == "sequences" else "event_id"
    row = db.execute(
        text(
            f"SELECT count(DISTINCT o.{fk}), "
            f"count(DISTINCT o.{fk} || ':' || coalesce(o.label_taxonomy_id, o.label)) "
            f"FROM {observations} o JOIN {groups} g ON g.id = o.{fk} "
            "JOIN deployments d ON d.id = g.deployment_id WHERE d.project_id = :p"
        ),
        {"p": project_id},
    ).one()
    return {"groups_with_observation": row[0], "group_species_pairs": row[1]}


def _date_window(db: Any, project_id: str) -> dict[str, str] | None:
    """The middle third of the project's capture dates, to exercise date filters."""
    from sqlalchemy import text

    first, last = db.execute(
        text(
            "SELECT min(f.captured_at_local), max(f.captured_at_local) FROM files f "
            "JOIN deployments d ON d.id = f.deployment_id WHERE d.project_id = :p"
        ),
        {"p": project_id},
    ).one()
    if first is None:
        return None
    from datetime import datetime

    a, b = datetime.fromisoformat(str(first)), datetime.fromisoformat(str(last))
    third = (b - a) / 3
    return {
        "date_from": (a + third).date().isoformat(),
        "date_to": (a + 2 * third).date().isoformat(),
    }


def _get(client: Any, url: str, params: dict[str, Any]) -> dict[str, Any]:
    response = client.get(url, params=params)
    body: Any
    try:
        body = response.json()
    except ValueError:
        body = response.text
    return {"status": response.status_code, "body": body}


def _timed(client: Any, url: str, params: dict[str, Any]) -> tuple[dict[str, Any], float]:
    """The response, and the median time in ms over ``TIMING_RUNS`` calls."""
    times = []
    result: dict[str, Any] = {}
    for _ in range(TIMING_RUNS):
        start = time.perf_counter()
        result = _get(client, url, params)
        times.append((time.perf_counter() - start) * 1000)
    return result, round(statistics.median(times), 1)


def _sequences_prefix(client: Any, project_id: str) -> str:
    """The stored-group routes: ``/api/sequences`` after the rename, else ``/api/events``."""
    for prefix in ("/api/sequences", "/api/events"):
        # An unknown /api path falls through to the app's HTML page with a
        # 200, so only a JSON answer counts.
        response = client.get(f"{prefix}/count", params={"project_id": project_id})
        if response.status_code == 200 and response.headers.get(
            "content-type", ""
        ).startswith("application/json"):
            return prefix
    raise SystemExit("Neither /api/sequences nor /api/events answers.")


def _statistics_requests(
    project_id: str, species: list[str]
) -> list[tuple[str, str, dict[str, Any]]]:
    """(name, url, params) of every statistic, without scope filters."""
    p = {"project_id": project_id}
    out: list[tuple[str, str, dict[str, Any]]] = [
        ("overview", "/api/statistics/overview", p),
        ("capture-date-coverage", "/api/statistics/capture-date-coverage", p),
        ("activity-pattern", "/api/statistics/activity-pattern", p),
        ("detection-trend", "/api/statistics/detection-trend", p),
        ("rate-trend", "/api/statistics/rate-trend", p),
        ("species-rates", "/api/statistics/species-rates", p),
        ("summary", "/api/statistics/summary", p),
        ("demographics", "/api/statistics/demographics", p),
        ("categories", "/api/statistics/categories", p),
        ("observation-rate-map", "/api/statistics/observation-rate-map", p),
        ("timeline", "/api/statistics/timeline", p),
        (
            "verification-progress-by-label",
            "/api/statistics/verification-progress-by-label",
            p,
        ),
    ]
    for mode in ("events", "max_n"):
        for rank in (None, "species", "class"):
            for wildlife in (False, True):
                params = {**p, "count_mode": mode, "wildlife_only": wildlife}
                if rank:
                    params["taxonomic_rank"] = rank
                name = f"species-{mode}-{rank or 'raw'}-{'wild' if wildlife else 'all'}"
                out.append((name, "/api/statistics/species", params))
    if species:
        pair = {**p, "species_a": species[0]}
        if len(species) > 1:
            pair["species_b"] = species[1]
        out.append(("activity-overlap", "/api/statistics/activity-overlap", pair))
    return out


def snapshot_project(
    client: Any, db: Any, project: Any, out: Path, *, exports: bool
) -> dict[str, float]:
    """Write one project's snapshot; return its timings."""
    pid = project.id
    folder = out / pid
    timings: dict[str, float] = {}
    prefix = _sequences_prefix(client, pid)

    species_resp = _get(
        client,
        "/api/statistics/species",
        {"project_id": pid, "wildlife_only": True, "taxonomic_rank": "species"},
    )
    species = (
        [row["species"] for row in species_resp["body"][:2]]
        if species_resp["status"] == 200
        else []
    )
    requests = _statistics_requests(pid, species)
    requests += [
        ("project-stats", f"/api/projects/{pid}/stats", {}),
        (
            "independent-observation-stats",
            f"/api/projects/{pid}/independent-observation-stats",
            {},
        ),
        ("deployments-bulk-stats", "/api/deployments/bulk-stats", {"project_id": pid}),
        ("sequences-count", f"{prefix}/count", {"project_id": pid}),
        ("sequences-verification-stats", f"{prefix}/verification-stats", {"project_id": pid}),
        (
            "sequences-label-counts",
            f"{prefix}/label-counts",
            {"project_id": pid, "count_by": prefix.removeprefix("/api/").rstrip("s")},
        ),
    ]
    for name, url, params in requests:
        result, ms = _timed(client, url, params)
        timings[name] = ms
        _write_json(folder / "all" / f"{name}.json", result)

    window = _date_window(db, pid)
    sites = sorted(s.id for s in project.sites)
    scoped = {"window": window} if window else {}
    if sites:
        scoped["first-site"] = {"site_ids": sites[0]}
    for scope_name, extra in scoped.items():
        for name, url, params in _statistics_requests(pid, species):
            _write_json(
                folder / scope_name / f"{name}.json", _get(client, url, {**params, **extra})
            )

    for dep in sorted(project.deployments, key=lambda d: d.id):
        _write_json(
            folder / "deployments" / f"{dep.id}.json",
            _get(client, f"/api/deployments/{dep.id}/info", {}),
        )
    for site_id in sites:
        _write_json(
            folder / "sites" / f"{site_id}.json", _get(client, f"/api/sites/{site_id}/info", {})
        )

    _write_json(folder / "oracle.json", _oracle(db, pid))
    if exports:
        _write_exports(db, project, folder / "exports")
    return timings


def _write_exports(db: Any, project: Any, folder: Path) -> None:
    """Every table of the Export page, and the Camtrap DP observations, as CSV."""
    from app.api.crud import export as export_crud

    folder.mkdir(parents=True, exist_ok=True)
    for title, headers, rows in export_crud.build_spreadsheet_sheets(db, project, None):
        _write_csv(folder / f"{title.lower()}.csv", headers, rows)

    # The order the Camtrap DP worker writes in: event observations summarise
    # what media() saw, so media() must be exhausted first.
    builder = export_crud.CamtrapDpBuilder(db, project, None)
    _, media_headers, obs_headers = export_crud.camtrap_dp_headers()
    media_rows: list[list[Any]] = []
    obs_rows: list[list[Any]] = []
    for media_row, observation_rows in builder.media():
        media_rows.append(media_row)
        obs_rows.extend(observation_rows)
    # Named for the stored unit, so it was renamed with it.
    grouped = getattr(builder, "sequence_observations", None) or builder.event_observations
    obs_rows.extend(grouped())
    _write_csv(folder / "camtrap-media.csv", media_headers, media_rows)
    _write_csv(folder / "camtrap-observations.csv", obs_headers, obs_rows)


def _write_csv(path: Path, headers: list[str], rows: Any) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([RENAMED_KEYS.get(h, h) for h in headers])
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True, help="Folder to write into")
    parser.add_argument(
        "--no-exports",
        action="store_true",
        help="Skip the export tables, for a database too large to dump",
    )
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"{args.out} is not empty.")

    db = open_seed_sandbox()  # refuses the real ~/AddaxAI, migrates the copy

    from fastapi.testclient import TestClient

    from app.main import app
    from app.models import Project

    # No `with`: the app's startup (backups, catalog sync, thumbnails) stays off.
    client = TestClient(app)
    timings: dict[str, dict[str, float]] = {}
    for project in db.query(Project).order_by(Project.id):
        print(f"{project.id} {project.name}", file=sys.stderr)
        timings[project.id] = snapshot_project(
            client, db, project, args.out, exports=not args.no_exports
        )
    (args.out / "timings.json").write_text(json.dumps(timings, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
