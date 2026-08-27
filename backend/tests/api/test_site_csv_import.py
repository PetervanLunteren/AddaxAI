"""Tests for the site CSV bulk import endpoints."""

from app.models import Project, Site
from app.utils.timezone_from_coords import tz_from_coords
from tests.conftest import make_project, make_site

HEADER = "name,latitude,longitude,elevation_m,habitat_type,notes"
CLEAN = f"{HEADER}\nCAM01,52.09,5.12,12,Forest,near the path\nCAM02,52.10,5.15,,,\n"


def _post(client, path, project_id, body, filename="sites.csv"):
    data = body if isinstance(body, bytes) else body.encode("utf-8")
    return client.post(
        f"{path}?project_id={project_id}", files={"file": (filename, data, "text/csv")}
    )


def _preview(client, project_id, body, **kw):
    return _post(client, "/api/sites/import/preview", project_id, body, **kw)


def _import(client, project_id, body, **kw):
    return _post(client, "/api/sites/import", project_id, body, **kw)


def _messages(payload):
    return [p["message"] for p in payload["problems"]]


# ---------------------------------------------------------------------------
# Preview: the happy path
# ---------------------------------------------------------------------------


def test_preview_returns_rows_and_no_problems(client, db):
    p = make_project(db)
    body = _preview(client, p.id, CLEAN).json()
    assert body["problems"] == []
    assert [r["name"] for r in body["rows"]] == ["CAM01", "CAM02"]
    assert body["rows"][0] == {
        "row": 2,
        "name": "CAM01",
        "latitude": 52.09,
        "longitude": 5.12,
        "elevation_m": 12.0,
        "habitat_type": "Forest",
        "notes": "near the path",
        "tags": {},
    }


def test_preview_leaves_blank_optional_columns_null(client, db):
    p = make_project(db)
    row = _preview(client, p.id, CLEAN).json()["rows"][1]
    assert row["elevation_m"] is None
    assert row["habitat_type"] is None
    assert row["notes"] is None


def test_preview_accepts_a_file_with_only_the_required_columns(client, db):
    p = make_project(db)
    body = _preview(client, p.id, "name,latitude,longitude\nCAM01,52.09,5.12\n").json()
    assert body["problems"] == []
    assert len(body["rows"]) == 1


def test_preview_accepts_columns_in_any_order(client, db):
    p = make_project(db)
    body = _preview(client, p.id, "longitude,name,latitude\n5.12,CAM01,52.09\n").json()
    assert body["problems"] == []
    assert body["rows"][0]["name"] == "CAM01"


def test_preview_accepts_a_semicolon_delimited_file(client, db):
    p = make_project(db)
    body = _preview(client, p.id, "name;latitude;longitude\nCAM01;52.09;5.12\n").json()
    assert body["problems"] == []
    assert body["rows"][0]["name"] == "CAM01"


def test_preview_strips_the_byte_order_mark(client, db):
    p = make_project(db)
    body = _preview(client, p.id, "﻿" + CLEAN).json()
    assert body["problems"] == []
    assert len(body["rows"]) == 2


def test_preview_accepts_unicode_names(client, db):
    p = make_project(db)
    body = _preview(client, p.id, "name,latitude,longitude\nZürich-Süd,47.3,8.5\n").json()
    assert body["rows"][0]["name"] == "Zürich-Süd"


def test_preview_writes_nothing(client, db):
    p = make_project(db)
    _preview(client, p.id, CLEAN)
    assert db.query(Site).filter(Site.project_id == p.id).count() == 0


# ---------------------------------------------------------------------------
# Preview: file-level problems
# ---------------------------------------------------------------------------


def test_preview_reports_a_missing_required_column(client, db):
    p = make_project(db)
    body = _preview(client, p.id, "name,latitude\nCAM01,52.09\n").json()
    assert body["rows"] == []
    assert [pr["column"] for pr in body["problems"]] == ["longitude"]


def test_preview_reports_an_unrecognised_column(client, db):
    p = make_project(db)
    body = _preview(client, p.id, "name,latitude,longitude,camera_id\nA,1,2,7\n").json()
    assert body["rows"] == []
    assert "Allowed columns are" in body["problems"][0]["message"]


def test_preview_reports_an_empty_file(client, db):
    p = make_project(db)
    body = _preview(client, p.id, "").json()
    assert body["rows"] == []
    assert body["problems"][0]["row"] is None


def test_preview_reports_a_header_only_file(client, db):
    p = make_project(db)
    body = _preview(client, p.id, HEADER + "\n").json()
    assert "no sites" in body["problems"][0]["message"]


def test_preview_reports_invalid_utf8(client, db):
    p = make_project(db)
    body = _preview(client, p.id, "name,latitude,longitude\nZ\xfcrich,1,2\n".encode("latin-1"))
    assert "UTF-8" in body.json()["problems"][0]["message"]


def test_preview_accepts_carriage_return_only_line_endings(client, db):
    """Excel for Mac's "CSV (Macintosh)" format. Used to be HTTP 500."""
    p = make_project(db)
    body = f"{HEADER}\rCAM01,52.09,5.12,,,\rCAM02,52.10,5.15,,,\r"
    resp = _preview(client, p.id, body)
    assert resp.status_code == 200
    assert resp.json()["problems"] == []
    assert [r["name"] for r in resp.json()["rows"]] == ["CAM01", "CAM02"]


def test_preview_reports_an_unbalanced_quotation_mark(client, db):
    """Used to be HTTP 500. The message has to name the likely cause."""
    p = make_project(db)
    body = f'{HEADER}\nCAM01,52.09,5.12,,,\n"open\n' + "x,1,2,,,\n" * 40000
    resp = _preview(client, p.id, body)
    assert resp.status_code == 200
    assert resp.json()["rows"] == []
    assert "quotation mark" in resp.json()["problems"][0]["message"]


def test_import_writes_nothing_for_an_unbalanced_quotation_mark(client, db):
    p = make_project(db)
    body = f'{HEADER}\nCAM01,52.09,5.12,,,\n"open\n' + "x,1,2,,,\n" * 40000
    resp = _import(client, p.id, body)
    assert resp.status_code == 200
    assert resp.json()["imported"] == 0
    assert db.query(Site).filter(Site.project_id == p.id).count() == 0


def test_preview_rejects_a_file_over_the_size_limit(client, db):
    p = make_project(db)
    padding = "x" * 200
    big = HEADER + "\n" + "".join(f"CAM{i},52.0,5.0,,,{padding}\n" for i in range(12_000))
    resp = _preview(client, p.id, big)
    assert resp.status_code == 400
    assert "2 MB" in resp.json()["detail"]


def test_preview_unknown_project_returns_404(client, db):
    resp = _preview(client, "does-not-exist", CLEAN)
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Preview: row-level problems
# ---------------------------------------------------------------------------


def test_preview_reports_an_empty_name(client, db):
    p = make_project(db)
    body = _preview(client, p.id, f"{HEADER}\n,52.09,5.12,,,\n").json()
    assert body["rows"] == []
    assert body["problems"][0]["column"] == "name"
    assert body["problems"][0]["row"] == 2


def test_preview_reports_a_name_over_255_characters(client, db):
    p = make_project(db)
    body = _preview(client, p.id, f"{HEADER}\n{'x' * 256},52.09,5.12,,,\n").json()
    assert "shorter name" in body["problems"][0]["message"]


def test_preview_reports_an_empty_latitude(client, db):
    p = make_project(db)
    body = _preview(client, p.id, f"{HEADER}\nCAM01,,5.12,,,\n").json()
    assert body["problems"][0]["column"] == "latitude"
    assert "empty" in body["problems"][0]["message"]


def test_preview_reports_a_comma_decimal_separator(client, db):
    p = make_project(db)
    body = _preview(client, p.id, f'{HEADER}\nCAM01,"52,09",5.12,,,\n').json()
    assert "dot as the decimal separator" in body["problems"][0]["message"]
    assert body["problems"][0]["value"] == "52,09"


def test_preview_reports_a_latitude_out_of_range(client, db):
    p = make_project(db)
    body = _preview(client, p.id, f"{HEADER}\nCAM01,95,5.12,,,\n").json()
    assert "between -90 and 90" in body["problems"][0]["message"]


def test_preview_reports_a_longitude_out_of_range(client, db):
    p = make_project(db)
    body = _preview(client, p.id, f"{HEADER}\nCAM01,52.09,200,,,\n").json()
    assert "between -180 and 180" in body["problems"][0]["message"]


def test_preview_reports_a_non_finite_coordinate(client, db):
    p = make_project(db)
    body = _preview(client, p.id, f"{HEADER}\nCAM01,nan,5.12,,,\n").json()
    assert body["problems"][0]["column"] == "latitude"


def test_preview_reports_null_island(client, db):
    p = make_project(db)
    body = _preview(client, p.id, f"{HEADER}\nCAM01,0,0,,,\n").json()
    assert "0, 0 is not allowed" in body["problems"][0]["message"]


def test_preview_reports_a_non_numeric_elevation(client, db):
    p = make_project(db)
    body = _preview(client, p.id, f"{HEADER}\nCAM01,52.09,5.12,high,,\n").json()
    assert body["problems"][0]["column"] == "elevation_m"


def test_preview_reports_notes_over_1000_characters(client, db):
    p = make_project(db)
    body = _preview(client, p.id, f"{HEADER}\nCAM01,52.09,5.12,,,{'x' * 1001}\n").json()
    assert body["problems"][0]["column"] == "notes"


def test_preview_reports_a_row_with_too_many_values(client, db):
    p = make_project(db)
    body = _preview(client, p.id, f"{HEADER}\nCAM01,52.09,5.12,,,,extra\n").json()
    assert "more values than the header" in body["problems"][0]["message"]


def test_preview_reports_every_problem_not_only_the_first(client, db):
    p = make_project(db)
    body = _preview(client, p.id, f"{HEADER}\n,bad,5.12,,,\nCAM02,52.1,also-bad,,,\n").json()
    assert len(body["problems"]) == 3
    assert body["rows"] == []


def test_preview_keeps_good_rows_alongside_problems(client, db):
    p = make_project(db)
    body = _preview(client, p.id, f"{HEADER}\nCAM01,52.09,5.12,,,\nCAM02,bad,5.15,,,\n").json()
    assert [r["name"] for r in body["rows"]] == ["CAM01"]
    assert len(body["problems"]) == 1


def test_preview_sorts_file_level_problems_first(client, db):
    p = make_project(db)
    make_site(db, project_id=p.id, name="CAM02")
    body = _preview(client, p.id, f"{HEADER}\nCAM01,bad,5.12,,,\nCAM02,52.1,5.15,,,\n").json()
    assert [pr["row"] for pr in body["problems"]] == [2, 3]


# ---------------------------------------------------------------------------
# Preview: duplicate names
# ---------------------------------------------------------------------------


def test_preview_reports_a_duplicate_name_inside_the_file(client, db):
    """Both rows are flagged, so the user does not have to find the other one."""
    p = make_project(db)
    body = _preview(client, p.id, f"{HEADER}\nCAM01,52.09,5.12,,,\nCAM01,52.10,5.15,,,\n").json()
    assert body["rows"] == []
    assert [pr["row"] for pr in body["problems"]] == [2, 3]
    assert all("more than one row" in pr["message"] for pr in body["problems"])


def test_preview_reports_a_name_that_already_exists_in_the_project(client, db):
    p = make_project(db)
    make_site(db, project_id=p.id, name="CAM01")
    body = _preview(client, p.id, f"{HEADER}\nCAM01,52.09,5.12,,,\n").json()
    assert "already exists in this project" in body["problems"][0]["message"]


def test_preview_allows_a_name_that_exists_only_in_another_project(client, db):
    other = make_project(db)
    make_site(db, project_id=other.id, name="CAM01")
    p = make_project(db)
    body = _preview(client, p.id, f"{HEADER}\nCAM01,52.09,5.12,,,\n").json()
    assert body["problems"] == []


def test_preview_treats_names_as_case_sensitive(client, db):
    p = make_project(db)
    make_site(db, project_id=p.id, name="CAM01")
    body = _preview(client, p.id, f"{HEADER}\ncam01,52.09,5.12,,,\n").json()
    assert body["problems"] == []


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------


def test_import_writes_every_row(client, db):
    p = make_project(db)
    body = _import(client, p.id, CLEAN).json()
    assert body == {"imported": 2, "problems": []}

    sites = db.query(Site).filter(Site.project_id == p.id).order_by(Site.name).all()
    assert [s.name for s in sites] == ["CAM01", "CAM02"]
    assert sites[0].latitude == 52.09
    assert sites[0].elevation_m == 12.0
    assert sites[0].habitat_type == "Forest"
    assert sites[0].notes == "near the path"
    assert sites[0].tags == {}
    assert sites[1].elevation_m is None


def test_import_writes_nothing_when_any_row_is_invalid(client, db):
    p = make_project(db)
    body = _import(client, p.id, f"{HEADER}\nCAM01,52.09,5.12,,,\nCAM02,bad,5.15,,,\n").json()
    assert body["imported"] == 0
    assert len(body["problems"]) == 1
    assert db.query(Site).filter(Site.project_id == p.id).count() == 0


def test_import_writes_nothing_for_a_broken_header(client, db):
    p = make_project(db)
    body = _import(client, p.id, "name,latitude\nCAM01,52.09\n").json()
    assert body["imported"] == 0
    assert db.query(Site).filter(Site.project_id == p.id).count() == 0


def test_import_revalidates_against_the_database(client, db):
    """A clashing site created after the preview must still block the import."""
    p = make_project(db)
    assert _preview(client, p.id, CLEAN).json()["problems"] == []

    make_site(db, project_id=p.id, name="CAM01")
    body = _import(client, p.id, CLEAN).json()

    assert body["imported"] == 0
    assert "already exists" in body["problems"][0]["message"]
    assert db.query(Site).filter(Site.project_id == p.id).count() == 1


def test_import_sets_the_project_timezone_from_the_first_row(client, db):
    p = make_project(db, timezone=None)
    _import(client, p.id, CLEAN)
    # Whatever the offline lookup answers for the first row's coordinates,
    # asserted against the helper so the test does not pin its dataset.
    assert db.get(Project, p.id).timezone == tz_from_coords(52.09, 5.12)


def test_import_does_not_overwrite_an_existing_project_timezone(client, db):
    p = make_project(db, timezone="UTC")
    _import(client, p.id, CLEAN)
    assert db.get(Project, p.id).timezone == "UTC"


def test_import_unknown_project_returns_404(client, db):
    resp = _import(client, "does-not-exist", CLEAN)
    assert resp.status_code == 404
    assert db.query(Site).count() == 0


# ---------------------------------------------------------------------------
# Tags: one tag:<name> column per tag key
# ---------------------------------------------------------------------------


TAGGED = (
    "name,latitude,longitude,tag:tenure,tag: camera\n"
    "CAM01,52.09,5.12,Aboriginal land,Reconyx\n"
    "CAM02,52.10,5.15,,Browning\n"
)


def test_tag_columns_become_tags(client, db):
    p = make_project(db)
    body = _preview(client, p.id, TAGGED).json()
    assert body["problems"] == []
    # The space after the colon is not part of the tag name.
    assert body["rows"][0]["tags"] == {"tenure": "Aboriginal land", "camera": "Reconyx"}
    # An empty cell is no tag, not a tag with an empty value.
    assert body["rows"][1]["tags"] == {"camera": "Browning"}


def test_import_writes_the_tags_on_the_site(client, db):
    p = make_project(db)
    assert _import(client, p.id, TAGGED).json()["imported"] == 2
    by_name = {s.name: s.tags for s in db.query(Site).filter(Site.project_id == p.id)}
    assert by_name == {
        "CAM01": {"tenure": "Aboriginal land", "camera": "Reconyx"},
        "CAM02": {"camera": "Browning"},
    }


def test_a_tag_column_needs_a_name(client, db):
    from app.services.csv_import import TAG_KEY_EMPTY

    p = make_project(db)
    body = _preview(client, p.id, "name,latitude,longitude,tag:\nCAM01,52.09,5.12,x\n").json()
    assert body["rows"] == []
    assert _messages(body) == [TAG_KEY_EMPTY]


def test_a_tag_name_has_the_editor_limit(client, db):
    from app.services.csv_import import TAG_KEY_TOO_LONG

    p = make_project(db)
    header = "name,latitude,longitude,tag:" + "k" * 41
    body = _preview(client, p.id, f"{header}\nCAM01,52.09,5.12,x\n").json()
    assert _messages(body) == [TAG_KEY_TOO_LONG]


def test_a_tag_value_has_the_editor_limit(client, db):
    from app.services.csv_import import TAG_VALUE_TOO_LONG

    p = make_project(db)
    body = _preview(
        client, p.id, f"name,latitude,longitude,tag:note\nCAM01,52.09,5.12,{'v' * 151}\n"
    ).json()
    assert body["rows"] == []
    assert [(q["row"], q["column"], q["message"]) for q in body["problems"]] == [
        (2, "tag:note", TAG_VALUE_TOO_LONG)
    ]


def test_the_same_tag_twice_is_a_duplicate_column(client, db):
    p = make_project(db)
    body = _preview(
        client, p.id, "name,latitude,longitude,tag:camera,tag: camera\nCAM01,52.09,5.12,a,b\n"
    ).json()
    assert body["rows"] == []
    assert body["problems"][0]["column"] == "tag: camera"
    assert "more than once" in body["problems"][0]["message"]


def test_unknown_column_message_names_the_tag_form(client, db):
    p = make_project(db)
    body = _preview(client, p.id, "name,latitude,longitude,camera\nCAM01,52.09,5.12,a\n").json()
    assert "tag:<name>" in body["problems"][0]["message"]


def test_the_tag_prefix_is_read_in_any_letter_case(client, db):
    """Excel capitalises the first letter of a header cell on its own, so
    Tag:Camera has to mean the same as tag:Camera."""
    p = make_project(db)
    body = _preview(client, p.id, "name,latitude,longitude,Tag:Camera\nCAM01,52.09,5.12,x\n").json()
    assert body["problems"] == []
    assert body["rows"][0]["tags"] == {"Camera": "x"}
