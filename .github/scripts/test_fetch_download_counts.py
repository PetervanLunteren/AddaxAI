"""Unit test for the pure releases-JSON -> tidy-rows transform.

Covers only ``build_rows`` (no network, no file I/O): the fetch and CSV append
are thin I/O wrappers we deliberately don't test. Run with:

    python -m pytest .github/scripts/ -q
"""

from fetch_download_counts import build_rows

# A realistic slice of the GitHub releases API response: one release with
# several assets, one release with none.
RELEASES = [
    {
        "tag_name": "v7.0.1-beta.16",
        "assets": [
            {"name": "AddaxAI-Setup.exe", "download_count": 42},
            {"name": "AddaxAI-arm64.dmg", "download_count": 18},
            {"name": "AddaxAI-amd64.deb", "download_count": 7},
        ],
    },
    {
        "tag_name": "v7.0.1-beta.15",
        "assets": [
            {"name": "AddaxAI-Setup.exe", "download_count": 130},
        ],
    },
    {"tag_name": "v0.0.1-notes-only", "assets": []},
]


def test_build_rows_one_row_per_asset():
    rows = build_rows(RELEASES, "2026-07-20")

    # 3 + 1 + 0 assets -> 4 rows; the assetless release contributes nothing.
    assert len(rows) == 4
    assert all(r["snapshot_date"] == "2026-07-20" for r in rows)

    assert rows[0] == {
        "snapshot_date": "2026-07-20",
        "release_tag": "v7.0.1-beta.16",
        "asset_name": "AddaxAI-Setup.exe",
        "download_count": 42,
    }
    # Same filename in a different release stays a distinct row (keyed by tag).
    assert rows[3] == {
        "snapshot_date": "2026-07-20",
        "release_tag": "v7.0.1-beta.15",
        "asset_name": "AddaxAI-Setup.exe",
        "download_count": 130,
    }


def test_build_rows_defaults_missing_fields():
    rows = build_rows([{"tag_name": "v1", "assets": [{"name": "x"}]}], "2026-01-01")
    assert rows == [
        {
            "snapshot_date": "2026-01-01",
            "release_tag": "v1",
            "asset_name": "x",
            "download_count": 0,
        }
    ]


def test_build_rows_empty():
    assert build_rows([], "2026-01-01") == []
