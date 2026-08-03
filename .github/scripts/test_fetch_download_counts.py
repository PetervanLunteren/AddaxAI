"""Unit test for the pure releases-JSON -> tidy-rows transform.

Covers only ``build_rows`` and the version parsing it filters on (no network,
no file I/O): the fetch and CSV append are thin I/O wrappers we deliberately
don't test. Run with:

    python -m pytest .github/scripts/ -q
"""

from fetch_download_counts import MIN_VERSION, build_rows, parse_version

# A realistic slice of the GitHub releases API response: one release with
# several assets, one release with none. All at or above MIN_VERSION, so the
# filter is not what these first tests are about.
RELEASES = [
    {
        "tag_name": "v7.0.3",
        "assets": [
            {"name": "AddaxAI-Setup.exe", "download_count": 42},
            {"name": "AddaxAI-arm64.dmg", "download_count": 18},
            {"name": "AddaxAI-amd64.deb", "download_count": 7},
        ],
    },
    {
        "tag_name": "v7.0.2",
        "assets": [
            {"name": "AddaxAI-Setup.exe", "download_count": 130},
        ],
    },
    {"tag_name": "v7.1.0-notes-only", "assets": []},
]


def test_build_rows_one_row_per_asset():
    rows = build_rows(RELEASES, "2026-07-20")

    # 3 + 1 + 0 assets -> 4 rows; the assetless release contributes nothing.
    assert len(rows) == 4
    assert all(r["snapshot_date"] == "2026-07-20" for r in rows)

    assert rows[0] == {
        "snapshot_date": "2026-07-20",
        "release_tag": "v7.0.3",
        "asset_name": "AddaxAI-Setup.exe",
        "download_count": 42,
    }
    # Same filename in a different release stays a distinct row (keyed by tag).
    assert rows[3] == {
        "snapshot_date": "2026-07-20",
        "release_tag": "v7.0.2",
        "asset_name": "AddaxAI-Setup.exe",
        "download_count": 130,
    }


def test_build_rows_defaults_missing_fields():
    rows = build_rows(
        [{"tag_name": "v7.0.2", "assets": [{"name": "x"}]}], "2026-01-01"
    )
    assert rows == [
        {
            "snapshot_date": "2026-01-01",
            "release_tag": "v7.0.2",
            "asset_name": "x",
            "download_count": 0,
        }
    ]


def test_build_rows_empty():
    assert build_rows([], "2026-01-01") == []


def test_build_rows_skips_anything_older_than_the_minimum():
    """The whole point of the filter: v6 and the v7 betas add nothing."""
    releases = [
        {"tag_name": "v7.0.2", "assets": [{"name": "keep", "download_count": 1}]},
        {"tag_name": "v7.0.1-beta.24", "assets": [{"name": "beta", "download_count": 9}]},
        {"tag_name": "v6.37", "assets": [{"name": "old", "download_count": 500}]},
        {"tag_name": "v.5.4", "assets": [{"name": "ancient", "download_count": 20}]},
        {"tag_name": "v1.0", "assets": [{"name": "first", "download_count": 3}]},
    ]

    rows = build_rows(releases, "2026-08-03")

    assert [r["asset_name"] for r in rows] == ["keep"]


def test_build_rows_skips_tags_that_are_not_versions():
    releases = [
        {"tag_name": "beta", "assets": [{"name": "a", "download_count": 1}]},
        {"tag_name": "test-migration", "assets": [{"name": "b", "download_count": 2}]},
        {"tag_name": "", "assets": [{"name": "c", "download_count": 3}]},
    ]

    assert build_rows(releases, "2026-08-03") == []


def test_parse_version_reads_the_tag_styles_used_in_this_repo():
    # Plain, pre-release, two-part, and the old "v.5.4" spelling.
    assert parse_version("v7.0.3") == (7, 0, 3)
    assert parse_version("v7.0.1-beta.16") == (7, 0, 1)
    assert parse_version("v6.37") == (6, 37)
    assert parse_version("v.5.4") == (5, 4)

    # Not versions.
    assert parse_version("beta") is None
    assert parse_version("test-migration") is None
    assert parse_version("") is None


def test_minimum_is_the_first_v7_release_with_assets_we_care_about():
    """Guards the constant itself: a stray edit here changes what is recorded."""
    assert MIN_VERSION == (7, 0, 2)
    assert parse_version("v7.0.2") == MIN_VERSION
    assert parse_version("v7.0.1-beta.24") < MIN_VERSION
    assert parse_version("v7.0.3") > MIN_VERSION
