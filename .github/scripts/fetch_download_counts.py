"""Snapshot GitHub release-asset download counts into a tidy CSV.

Run by the "Download metrics" workflow on a weekly schedule. GitHub's
``download_count`` is a live cumulative counter with no history, so we append a
dated row per asset to ``metrics/download_counts.csv`` to build a trend over
time.

Stdlib only (no dependencies): reads ``GITHUB_REPOSITORY`` and ``GITHUB_TOKEN``
from the environment (both provided automatically inside GitHub Actions),
fetches every release + asset, and appends one row per asset for today's UTC
date. The header is written only when the CSV does not exist yet.

The CSV is tidy/long: ``snapshot_date,release_tag,asset_name,download_count``.
Each row is a running total, so week-over-week deltas give new downloads.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

CSV_PATH = Path("metrics/download_counts.csv")
FIELDNAMES = ["snapshot_date", "release_tag", "asset_name", "download_count"]
API_ROOT = "https://api.github.com"
PER_PAGE = 100


def build_rows(releases: list[dict], snapshot_date: str) -> list[dict]:
    """Turn the releases API response into tidy rows, one per asset.

    Pure and side-effect free (this is the unit-tested part). A release with no
    assets contributes no rows. ``download_count`` defaults to 0 defensively.
    """
    rows: list[dict] = []
    for release in releases:
        tag = release.get("tag_name", "")
        for asset in release.get("assets", []):
            rows.append(
                {
                    "snapshot_date": snapshot_date,
                    "release_tag": tag,
                    "asset_name": asset.get("name", ""),
                    "download_count": asset.get("download_count", 0),
                }
            )
    return rows


def fetch_releases(repo: str, token: str) -> list[dict]:
    """Fetch all releases (paginated) for ``owner/repo``."""
    releases: list[dict] = []
    page = 1
    while True:
        url = (
            f"{API_ROOT}/repos/{repo}/releases"
            f"?per_page={PER_PAGE}&page={page}"
        )
        request = urllib.request.Request(url, headers=_headers(token))
        with urllib.request.urlopen(request, timeout=30) as response:
            batch = json.load(response)
        if not batch:
            break
        releases.extend(batch)
        if len(batch) < PER_PAGE:
            break
        page += 1
    return releases


def _headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "addaxai-download-metrics",
    }


def append_rows(rows: list[dict], csv_path: Path = CSV_PATH) -> None:
    """Append rows to the CSV, writing the header only on first creation."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    repo = os.environ.get("GITHUB_REPOSITORY")
    token = os.environ.get("GITHUB_TOKEN")
    if not repo or not token:
        print(
            "GITHUB_REPOSITORY and GITHUB_TOKEN must be set", file=sys.stderr
        )
        return 1

    snapshot_date = datetime.now(UTC).strftime("%Y-%m-%d")
    try:
        releases = fetch_releases(repo, token)
    except urllib.error.URLError as e:
        print(f"Failed to fetch releases: {e}", file=sys.stderr)
        return 1

    rows = build_rows(releases, snapshot_date)
    if not rows:
        print("No release assets found; nothing to record.")
        return 0

    append_rows(rows)
    total = sum(int(r["download_count"]) for r in rows)
    print(
        f"{snapshot_date}: recorded {len(rows)} asset rows across "
        f"{len(releases)} releases ({total:,} total downloads)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
