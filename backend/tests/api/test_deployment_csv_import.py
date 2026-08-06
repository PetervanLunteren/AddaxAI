"""Tests for the deployment CSV bulk import endpoints.

What these create is a DeploymentQueue entry, not a Deployment. The queue has
no date columns, which is why the CSV has none either.
"""

from app.models import DeploymentQueue
from app.services.csv_import_deployments import SITE_NOT_FOUND
from app.services.folder_scanner import OUTPUT_DIR_MARKER
from tests.conftest import make_deployment, make_project, make_site

HEADER = "folder,site,notes"


def _folder(tmp_path, name, images=1, videos=0):
    """A real folder holding dummy media files. Only names are ever read."""
    folder = tmp_path / name
    folder.mkdir(parents=True)
    for i in range(images):
        (folder / f"img{i}.jpg").write_bytes(b"not really a jpeg")
    for i in range(videos):
        (folder / f"vid{i}.mp4").write_bytes(b"not really a video")
    return str(folder)


def _queue_entry(db, project_id, folder_path, status="pending"):
    entry = DeploymentQueue(project_id=project_id, folder_path=folder_path, status=status)
    db.add(entry)
    db.flush()
    return entry


def _post(client, path, project_id, body, filename="deployments.csv"):
    data = body if isinstance(body, bytes) else body.encode("utf-8")
    return client.post(
        f"{path}?project_id={project_id}", files={"file": (filename, data, "text/csv")}
    )


def _preview(client, project_id, body, **kw):
    return _post(client, "/api/deployment-queue/import/preview", project_id, body, **kw)


def _import(client, project_id, body, **kw):
    return _post(client, "/api/deployment-queue/import", project_id, body, **kw)


def _entries(db, project_id):
    return db.query(DeploymentQueue).filter(DeploymentQueue.project_id == project_id).all()


# ---------------------------------------------------------------------------
# Preview: the happy path and the media counts
# ---------------------------------------------------------------------------


def test_preview_returns_rows_with_counts(client, db, tmp_path):
    p = make_project(db)
    make_site(db, project_id=p.id, name="CAM01")
    folder = _folder(tmp_path, "cam01", images=3, videos=2)

    body = _preview(client, p.id, f"{HEADER}\n{folder},CAM01,first season\n").json()

    assert body["problems"] == []
    assert body["rows"] == [
        {
            "row": 2,
            "folder": folder,
            "site": "CAM01",
            "notes": "first season",
            "image_count": 3,
            "video_count": 2,
        }
    ]


def test_preview_counts_media_in_subfolders(client, db, tmp_path):
    p = make_project(db)
    folder = _folder(tmp_path, "cam01", images=1)
    _folder(tmp_path, "cam01/nested", images=4, videos=1)

    body = _preview(client, p.id, f"{HEADER}\n{folder},,\n").json()

    assert body["rows"][0]["image_count"] == 5
    assert body["rows"][0]["video_count"] == 1


def test_preview_ignores_non_media_files(client, db, tmp_path):
    p = make_project(db)
    folder = _folder(tmp_path, "cam01", images=2)
    (tmp_path / "cam01" / "notes.txt").write_text("hello")
    (tmp_path / "cam01" / "results.json").write_text("{}")

    body = _preview(client, p.id, f"{HEADER}\n{folder},,\n").json()
    assert body["rows"][0]["image_count"] == 2


def test_preview_skips_addaxai_output_folders(client, db, tmp_path):
    """A previous run's copies must not be counted as new input media."""
    p = make_project(db)
    folder = _folder(tmp_path, "cam01", images=2)
    output = _folder(tmp_path, "cam01/addaxai-output", images=7)
    (tmp_path / "cam01" / "addaxai-output" / OUTPUT_DIR_MARKER).write_text("")
    assert output  # the folder really does hold images

    body = _preview(client, p.id, f"{HEADER}\n{folder},,\n").json()
    assert body["rows"][0]["image_count"] == 2


def test_preview_skips_dot_folders(client, db, tmp_path):
    p = make_project(db)
    folder = _folder(tmp_path, "cam01", images=2)
    _folder(tmp_path, "cam01/.addaxai", images=5)

    body = _preview(client, p.id, f"{HEADER}\n{folder},,\n").json()
    assert body["rows"][0]["image_count"] == 2


def test_preview_writes_nothing(client, db, tmp_path):
    p = make_project(db)
    folder = _folder(tmp_path, "cam01")
    _preview(client, p.id, f"{HEADER}\n{folder},,\n")
    assert _entries(db, p.id) == []


# ---------------------------------------------------------------------------
# Preview: folder problems
# ---------------------------------------------------------------------------


def test_preview_reports_a_folder_that_does_not_exist(client, db, tmp_path):
    p = make_project(db)
    missing = str(tmp_path / "nope")
    body = _preview(client, p.id, f"{HEADER}\n{missing},,\n").json()

    assert body["rows"] == []
    assert "was not found" in body["problems"][0]["message"]
    assert body["problems"][0]["column"] == "folder"
    assert body["problems"][0]["value"] == missing


def test_preview_reports_a_path_that_is_a_file(client, db, tmp_path):
    p = make_project(db)
    a_file = tmp_path / "cam01.jpg"
    a_file.write_bytes(b"x")

    body = _preview(client, p.id, f"{HEADER}\n{a_file},,\n").json()
    assert "is a file, not a folder" in body["problems"][0]["message"]


def test_preview_reports_a_relative_path(client, db):
    p = make_project(db)
    body = _preview(client, p.id, f"{HEADER}\ndata/cam01,,\n").json()
    assert "not a full path" in body["problems"][0]["message"]


def test_preview_reports_an_empty_folder_cell(client, db):
    p = make_project(db)
    body = _preview(client, p.id, f"{HEADER}\n,CAM01,\n").json()
    assert "Folder is empty" in body["problems"][0]["message"]


def test_preview_reports_a_folder_with_no_media(client, db, tmp_path):
    p = make_project(db)
    folder = _folder(tmp_path, "cam01", images=0)
    body = _preview(client, p.id, f"{HEADER}\n{folder},,\n").json()
    assert "no images or videos" in body["problems"][0]["message"]


def test_preview_accepts_a_trailing_slash(client, db, tmp_path):
    p = make_project(db)
    folder = _folder(tmp_path, "cam01")
    body = _preview(client, p.id, f"{HEADER}\n{folder}/,,\n").json()
    assert body["problems"] == []


def test_preview_reports_a_duplicate_folder_inside_the_file(client, db, tmp_path):
    p = make_project(db)
    folder = _folder(tmp_path, "cam01")
    body = _preview(client, p.id, f"{HEADER}\n{folder},,\n{folder},,\n").json()

    assert body["rows"] == []
    assert [pr["row"] for pr in body["problems"]] == [2, 3]
    assert all("more than one row" in pr["message"] for pr in body["problems"])


def test_preview_reports_a_duplicate_that_differs_only_by_a_trailing_slash(
    client, db, tmp_path
):
    p = make_project(db)
    folder = _folder(tmp_path, "cam01")
    body = _preview(client, p.id, f"{HEADER}\n{folder},,\n{folder}/,,\n").json()
    assert len(body["problems"]) == 2


def test_preview_reports_a_folder_already_pending_in_the_queue(client, db, tmp_path):
    p = make_project(db)
    folder = _folder(tmp_path, "cam01")
    _queue_entry(db, p.id, folder)

    body = _preview(client, p.id, f"{HEADER}\n{folder},,\n").json()
    assert "already in the queue" in body["problems"][0]["message"]


def test_preview_reports_a_folder_already_processing_in_the_queue(client, db, tmp_path):
    p = make_project(db)
    folder = _folder(tmp_path, "cam01")
    _queue_entry(db, p.id, folder, status="processing")

    body = _preview(client, p.id, f"{HEADER}\n{folder},,\n").json()
    assert "already in the queue" in body["problems"][0]["message"]


def test_preview_allows_a_folder_whose_queue_entry_is_completed(client, db, tmp_path):
    """A completed entry no longer owns its folder, so it can be queued again."""
    p = make_project(db)
    folder = _folder(tmp_path, "cam01")
    _queue_entry(db, p.id, folder, status="completed")

    body = _preview(client, p.id, f"{HEADER}\n{folder},,\n").json()
    assert body["problems"] == []


def test_preview_reports_a_folder_already_used_by_a_deployment(client, db, tmp_path):
    p = make_project(db)
    folder = _folder(tmp_path, "cam01")
    make_deployment(db, project_id=p.id, folder_path=folder)

    body = _preview(client, p.id, f"{HEADER}\n{folder},,\n").json()
    assert "already used by a deployment" in body["problems"][0]["message"]


def test_preview_ignores_a_queue_entry_in_another_project(client, db, tmp_path):
    other = make_project(db)
    folder = _folder(tmp_path, "cam01")
    _queue_entry(db, other.id, folder)

    p = make_project(db)
    body = _preview(client, p.id, f"{HEADER}\n{folder},,\n").json()
    assert body["problems"] == []


def test_preview_ignores_a_deployment_in_another_project(client, db, tmp_path):
    other = make_project(db)
    folder = _folder(tmp_path, "cam01")
    make_deployment(db, project_id=other.id, folder_path=folder)

    p = make_project(db)
    body = _preview(client, p.id, f"{HEADER}\n{folder},,\n").json()
    assert body["problems"] == []


# ---------------------------------------------------------------------------
# Preview: spaces and quoted paths
#
# A path with a space needs quoting in a shell, so people quote it in their
# spreadsheet too. The space itself never needed anything.
# ---------------------------------------------------------------------------


def test_preview_accepts_a_path_containing_spaces(client, db, tmp_path):
    p = make_project(db)
    folder = _folder(tmp_path, "Kifaru Plains/deployment_001")
    body = _preview(client, p.id, f"{HEADER}\n{folder},,\n").json()
    assert body["problems"] == []
    assert body["rows"][0]["folder"] == folder


def test_preview_strips_single_quotes_around_a_folder(client, db, tmp_path):
    p = make_project(db)
    folder = _folder(tmp_path, "Kifaru Plains/deployment_001")
    body = _preview(client, p.id, f"{HEADER}\n'{folder}',,\n").json()
    assert body["problems"] == []
    assert body["rows"][0]["folder"] == folder


def test_preview_strips_double_quotes_around_a_folder(client, db, tmp_path):
    """csv.reader eats plain double quotes, so these are the escaped kind."""
    p = make_project(db)
    folder = _folder(tmp_path, "cam01")
    body = _preview(client, p.id, f'{HEADER}\n"""{folder}""",,\n').json()
    assert body["problems"] == []
    assert body["rows"][0]["folder"] == folder


def test_preview_strips_quotes_before_looking_for_duplicates(client, db, tmp_path):
    p = make_project(db)
    folder = _folder(tmp_path, "cam01")
    body = _preview(client, p.id, f"{HEADER}\n{folder},,\n'{folder}',,\n").json()
    assert [pr["row"] for pr in body["problems"]] == [2, 3]
    assert all("more than one row" in pr["message"] for pr in body["problems"])


def test_preview_reports_a_folder_written_as_two_quotes_as_empty(client, db):
    p = make_project(db)
    body = _preview(client, p.id, f"{HEADER}\n'',CAM01,\n").json()
    assert "Folder is empty" in body["problems"][0]["message"]


def test_preview_strips_a_trailing_quote_with_no_opening_one(client, db, tmp_path):
    """The shape a spreadsheet actually produces. A leading apostrophe is
    Excel's text marker, so typing '/data/x' into a cell exports /data/x'
    with the opening quote already gone."""
    p = make_project(db)
    folder = _folder(tmp_path, "Kifaru Plains/deployment_001")
    body = _preview(client, p.id, f"{HEADER}\n{folder}',,\n").json()
    assert body["problems"] == []
    assert body["rows"][0]["folder"] == folder


def test_preview_strips_a_leading_quote_with_no_closing_one(client, db, tmp_path):
    p = make_project(db)
    folder = _folder(tmp_path, "cam01")
    body = _preview(client, p.id, f"{HEADER}\n'{folder},,\n").json()
    assert body["problems"] == []
    assert body["rows"][0]["folder"] == folder


def test_preview_does_not_unescape_backslashes(client, db, tmp_path):
    """Dragging a folder into a terminal gives `Kifaru\\ Plains`. A backslash
    is legal in a POSIX filename, so un-escaping could point at a different
    folder. Deliberately left as a plain not-found."""
    p = make_project(db)
    folder = _folder(tmp_path, "Kifaru Plains/deployment_001")
    escaped = folder.replace(" ", "\\ ")
    body = _preview(client, p.id, f"{HEADER}\n{escaped},,\n").json()
    assert "was not found" in body["problems"][0]["message"]


def test_preview_does_not_strip_quotes_from_a_site_name(client, db, tmp_path):
    """Folder only. A site name that really contains quotes is likelier than
    a path that does."""
    p = make_project(db)
    make_site(db, project_id=p.id, name="CAM01")
    folder = _folder(tmp_path, "cam01")
    body = _preview(client, p.id, f"{HEADER}\n{folder},'CAM01',\n").json()
    assert body["problems"][0]["column"] == "site"


def test_import_stores_the_unquoted_path(client, db, tmp_path):
    p = make_project(db)
    folder = _folder(tmp_path, "Kifaru Plains/deployment_001")
    assert _import(client, p.id, f"{HEADER}\n'{folder}',,\n").json()["imported"] == 1
    assert _entries(db, p.id)[0].folder_path == folder


# ---------------------------------------------------------------------------
# Preview: overlapping folders within one file
#
# Listing a season folder and its camera subfolders would ingest the same
# images into two deployments. Compared as strings, within this one file only.
# ---------------------------------------------------------------------------


def test_preview_blocks_a_row_that_contains_another_row(client, db, tmp_path):
    p = make_project(db)
    parent = _folder(tmp_path, "season")
    child = _folder(tmp_path, "season/cam01")

    body = _preview(client, p.id, f"{HEADER}\n{parent},,\n{child},,\n").json()

    assert body["rows"] == []
    assert [pr["row"] for pr in body["problems"]] == [2, 3]
    assert "contains another folder" in body["problems"][0]["message"]
    assert "is inside another folder" in body["problems"][1]["message"]


def test_preview_names_the_folder_it_overlaps(client, db, tmp_path):
    """The row number already identifies the user's own row, so the value
    carries what it collides with instead."""
    p = make_project(db)
    parent = _folder(tmp_path, "season")
    child = _folder(tmp_path, "season/cam01")

    body = _preview(client, p.id, f"{HEADER}\n{parent},,\n{child},,\n").json()

    assert body["problems"][0]["value"] == child
    assert body["problems"][1]["value"] == parent


def test_preview_blocks_every_child_of_a_listed_parent(client, db, tmp_path):
    p = make_project(db)
    parent = _folder(tmp_path, "season")
    one = _folder(tmp_path, "season/cam01")
    two = _folder(tmp_path, "season/cam02")

    body = _preview(client, p.id, f"{HEADER}\n{parent},,\n{one},,\n{two},,\n").json()
    assert [pr["row"] for pr in body["problems"]] == [2, 3, 4]


def test_preview_allows_siblings_that_share_a_name_prefix(client, db, tmp_path):
    """cam1 is not the parent of cam10. This is why the check appends a
    separator before comparing."""
    p = make_project(db)
    one = _folder(tmp_path, "cam1")
    ten = _folder(tmp_path, "cam10")

    body = _preview(client, p.id, f"{HEADER}\n{one},,\n{ten},,\n").json()
    assert body["problems"] == []
    assert len(body["rows"]) == 2


def test_preview_allows_plain_siblings(client, db, tmp_path):
    p = make_project(db)
    one = _folder(tmp_path, "season/cam01")
    two = _folder(tmp_path, "season/cam02")

    body = _preview(client, p.id, f"{HEADER}\n{one},,\n{two},,\n").json()
    assert body["problems"] == []


def test_the_same_folder_twice_reports_duplicate_not_overlap(client, db, tmp_path):
    """Pins the check order: an exact match is the clearer diagnosis."""
    p = make_project(db)
    folder = _folder(tmp_path, "cam01")

    body = _preview(client, p.id, f"{HEADER}\n{folder},,\n{folder},,\n").json()
    assert all("more than one row" in pr["message"] for pr in body["problems"])


def test_a_missing_nested_folder_reports_not_found_not_overlap(client, db, tmp_path):
    """Pins the check order: a typo should get the actionable message."""
    p = make_project(db)
    parent = _folder(tmp_path, "season")
    missing = str(tmp_path / "season" / "nope")

    body = _preview(client, p.id, f"{HEADER}\n{parent},,\n{missing},,\n").json()
    messages = {pr["row"]: pr["message"] for pr in body["problems"]}
    assert "was not found" in messages[3]


def test_preview_allows_a_folder_nested_inside_an_existing_deployment(
    client, db, tmp_path
):
    """Deliberately allowed. Checking against stored deployments was dropped:
    it would block anyone who split a season folder and later imported one of
    its subfolders, and the safe fix is not obvious. See TODO.md."""
    p = make_project(db)
    parent = _folder(tmp_path, "season")
    child = _folder(tmp_path, "season/cam01")
    make_deployment(db, project_id=p.id, folder_path=parent)

    body = _preview(client, p.id, f"{HEADER}\n{child},,\n").json()
    assert body["problems"] == []


def test_preview_accepts_carriage_return_only_line_endings(client, db, tmp_path):
    p = make_project(db)
    folder = _folder(tmp_path, "cam01")
    resp = _preview(client, p.id, f"{HEADER}\r{folder},,\r")
    assert resp.status_code == 200
    assert resp.json()["problems"] == []


def test_preview_reports_an_unbalanced_quotation_mark(client, db):
    p = make_project(db)
    body = f'{HEADER}\n"open\n' + "x,1,2\n" * 40000
    resp = _preview(client, p.id, body)
    assert resp.status_code == 200
    assert "quotation mark" in resp.json()["problems"][0]["message"]


# ---------------------------------------------------------------------------
# Preview: the site column
# ---------------------------------------------------------------------------


def test_the_unknown_site_message_keeps_the_phrase_the_dialog_links():
    """ImportDeploymentsDialog turns "Import your sites first" inside this
    message into a link to the Sites page. It finds the phrase by matching
    the text, so rewording the message without updating
    `problemLink.phrase` there would silently drop the link."""
    assert "Import your sites first" in SITE_NOT_FOUND


def test_preview_reports_an_unknown_site_name(client, db, tmp_path):
    p = make_project(db)
    folder = _folder(tmp_path, "cam01")
    body = _preview(client, p.id, f"{HEADER}\n{folder},Nowhere,\n").json()

    assert body["rows"] == []
    assert body["problems"][0]["column"] == "site"
    assert "Import your sites first" in body["problems"][0]["message"]
    assert body["problems"][0]["value"] == "Nowhere"


def test_preview_matches_site_names_case_sensitively(client, db, tmp_path):
    p = make_project(db)
    make_site(db, project_id=p.id, name="CAM01")
    folder = _folder(tmp_path, "cam01")

    body = _preview(client, p.id, f"{HEADER}\n{folder},cam01,\n").json()
    assert body["problems"][0]["column"] == "site"


def test_preview_trims_spaces_around_a_site_name(client, db, tmp_path):
    p = make_project(db)
    make_site(db, project_id=p.id, name="CAM01")
    folder = _folder(tmp_path, "cam01")

    body = _preview(client, p.id, f"{HEADER}\n{folder},  CAM01  ,\n").json()
    assert body["problems"] == []


def test_preview_ignores_a_site_from_another_project(client, db, tmp_path):
    other = make_project(db)
    make_site(db, project_id=other.id, name="CAM01")
    p = make_project(db)
    folder = _folder(tmp_path, "cam01")

    body = _preview(client, p.id, f"{HEADER}\n{folder},CAM01,\n").json()
    assert body["problems"][0]["column"] == "site"


def test_preview_allows_an_empty_site(client, db, tmp_path):
    p = make_project(db)
    folder = _folder(tmp_path, "cam01")
    body = _preview(client, p.id, f"{HEADER}\n{folder},,\n").json()
    assert body["problems"] == []
    assert body["rows"][0]["site"] is None


def test_preview_reports_the_folder_and_the_site_of_one_row_together(client, db, tmp_path):
    p = make_project(db)
    missing = str(tmp_path / "nope")
    body = _preview(client, p.id, f"{HEADER}\n{missing},Nowhere,\n").json()
    assert {pr["column"] for pr in body["problems"]} == {"folder", "site"}


# ---------------------------------------------------------------------------
# Preview: file-level problems
# ---------------------------------------------------------------------------


def test_preview_reports_a_missing_folder_column(client, db):
    p = make_project(db)
    body = _preview(client, p.id, "site,notes\nCAM01,\n").json()
    assert body["rows"] == []
    assert [pr["column"] for pr in body["problems"]] == ["folder"]


def test_preview_reports_an_unrecognised_column(client, db):
    p = make_project(db)
    body = _preview(client, p.id, "folder,start_date\n/tmp/x,2026-01-01\n").json()
    assert "Allowed columns are: folder, site, notes." in body["problems"][0]["message"]


def test_preview_reports_a_header_only_file(client, db):
    p = make_project(db)
    body = _preview(client, p.id, HEADER + "\n").json()
    assert "no deployments" in body["problems"][0]["message"]


def test_preview_accepts_a_semicolon_delimited_file(client, db, tmp_path):
    p = make_project(db)
    folder = _folder(tmp_path, "cam01")
    body = _preview(client, p.id, f"folder;site;notes\n{folder};;\n").json()
    assert body["problems"] == []


def test_preview_accepts_a_file_with_only_the_folder_column(client, db, tmp_path):
    p = make_project(db)
    folder = _folder(tmp_path, "cam01")
    body = _preview(client, p.id, f"folder\n{folder}\n").json()
    assert body["problems"] == []
    assert body["rows"][0]["site"] is None


def test_preview_unknown_project_returns_404(client, db):
    assert _preview(client, "does-not-exist", f"{HEADER}\n/tmp/x,,\n").status_code == 404


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------


def test_import_creates_pending_entries_with_the_counts(client, db, tmp_path):
    p = make_project(db)
    site = make_site(db, project_id=p.id, name="CAM01")
    one = _folder(tmp_path, "cam01", images=3, videos=1)
    two = _folder(tmp_path, "cam02", images=2)

    body = _import(client, p.id, f"{HEADER}\n{one},CAM01,season one\n{two},,\n").json()
    assert body == {"imported": 2, "problems": []}

    entries = sorted(_entries(db, p.id), key=lambda e: e.folder_path)
    assert [e.folder_path for e in entries] == [one, two]
    assert entries[0].site_id == site.id
    assert entries[0].image_count == 3
    assert entries[0].video_count == 1
    assert entries[0].notes == "season one"
    assert entries[0].status == "pending"
    assert entries[1].site_id is None
    assert entries[1].notes is None


def test_import_leaves_offset_and_mtime_fallback_at_their_defaults(client, db, tmp_path):
    p = make_project(db)
    folder = _folder(tmp_path, "cam01")
    _import(client, p.id, f"{HEADER}\n{folder},,\n")

    entry = _entries(db, p.id)[0]
    assert entry.datetime_offset_seconds is None
    assert entry.use_file_mtime_fallback is False
    assert entry.tags == {}


def test_import_stores_the_normalised_folder_path(client, db, tmp_path):
    p = make_project(db)
    folder = _folder(tmp_path, "cam01")
    _import(client, p.id, f"{HEADER}\n{folder}/,,\n")
    assert _entries(db, p.id)[0].folder_path == folder


def test_import_writes_nothing_when_one_folder_is_missing(client, db, tmp_path):
    p = make_project(db)
    good = _folder(tmp_path, "cam01")
    missing = str(tmp_path / "nope")

    body = _import(client, p.id, f"{HEADER}\n{good},,\n{missing},,\n").json()
    assert body["imported"] == 0
    assert _entries(db, p.id) == []


def test_import_writes_nothing_when_a_site_is_unknown(client, db, tmp_path):
    p = make_project(db)
    folder = _folder(tmp_path, "cam01")
    body = _import(client, p.id, f"{HEADER}\n{folder},Nowhere,\n").json()
    assert body["imported"] == 0
    assert _entries(db, p.id) == []


def test_import_revalidates_the_folders_on_disk(client, db, tmp_path):
    """A folder that disappears after the preview must still block the import."""
    p = make_project(db)
    folder = _folder(tmp_path, "cam01")
    assert _preview(client, p.id, f"{HEADER}\n{folder},,\n").json()["problems"] == []

    (tmp_path / "cam01" / "img0.jpg").unlink()
    (tmp_path / "cam01").rmdir()

    body = _import(client, p.id, f"{HEADER}\n{folder},,\n").json()
    assert body["imported"] == 0
    assert "was not found" in body["problems"][0]["message"]
    assert _entries(db, p.id) == []


def test_import_blocks_a_second_import_of_the_same_file(client, db, tmp_path):
    p = make_project(db)
    folder = _folder(tmp_path, "cam01")
    csv = f"{HEADER}\n{folder},,\n"

    assert _import(client, p.id, csv).json()["imported"] == 1
    body = _import(client, p.id, csv).json()

    assert body["imported"] == 0
    assert "already in the queue" in body["problems"][0]["message"]
    assert len(_entries(db, p.id)) == 1


def test_import_unknown_project_returns_404(client, db, tmp_path):
    folder = _folder(tmp_path, "cam01")
    resp = _import(client, "does-not-exist", f"{HEADER}\n{folder},,\n")
    assert resp.status_code == 404
    assert db.query(DeploymentQueue).count() == 0
