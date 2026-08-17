"""An unreadable folder must never be reported as an empty one.

``os.walk`` defaults to ``onerror=None``, which discards every error
``os.scandir`` raises and yields nothing for that directory. A folder the
app cannot list then looks exactly like a folder with no media in it, and
every caller says "0 images".

That is what a beta tester hit on an external USB drive: the same path
answered "0 images, 0 videos" and then "8969 images" sixteen seconds
later, with `[Errno 5] Input/output error` on that drive elsewhere in the
same logs. The dangerous half was not the preview, it was the analysis
worker sharing the flaw: a stalled drive there means ingesting part of a
deployment and reporting success.

These tests make the folder unreadable with ``chmod 000`` and assert the
error comes out rather than being rounded down to zero.
"""

import pytest
from fastapi import status
from PIL import Image

from app.services.folder_scanner import count_media_files, scan_folder, walk_media_files
from app.workers.detection_worker import scan_folder_for_media


def _write_jpeg(path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), (1, 2, 3)).save(path, format="JPEG")


@pytest.fixture
def tree_with_unreadable_subdir(tmp_path, make_unreadable):
    """A readable folder holding a readable image and an unlistable subfolder.

    The locking (and the skip when chmod is not enforced) lives in the
    shared ``make_unreadable`` fixture in tests/conftest.py.
    """
    _write_jpeg(tmp_path / "IMG_0001.jpg")
    locked = tmp_path / "locked"
    _write_jpeg(locked / "IMG_0002.jpg")
    make_unreadable(locked)
    return tmp_path


def test_walk_media_files_raises_instead_of_returning_a_short_list(
    tree_with_unreadable_subdir,
):
    with pytest.raises(OSError):
        walk_media_files(tree_with_unreadable_subdir)


def test_count_media_files_raises_rather_than_counting_zero(
    tree_with_unreadable_subdir,
):
    with pytest.raises(OSError):
        count_media_files(tree_with_unreadable_subdir)


def test_scan_folder_propagates_the_error(tree_with_unreadable_subdir):
    with pytest.raises(OSError):
        scan_folder(str(tree_with_unreadable_subdir))


def test_worker_scan_raises_rather_than_analysing_part_of_a_deployment(
    tree_with_unreadable_subdir,
):
    """The one that could corrupt data.

    A short list here means MegaDetector runs over some of the folder and
    the deployment is written to the database as complete.
    """
    with pytest.raises(OSError):
        scan_folder_for_media(tree_with_unreadable_subdir)


def test_preview_folder_endpoint_refuses_an_unreadable_folder(
    client, tree_with_unreadable_subdir
):
    """The regression itself: this used to answer 200 with 0 images.

    A denied listing keeps the permission wording, which is the actionable
    one (on macOS, access to removable volumes is a privacy setting). What
    matters here is that it is an error at all.
    """
    response = client.get(
        "/api/deployments/preview-folder",
        params={"path": str(tree_with_unreadable_subdir)},
    )

    assert response.status_code == status.HTTP_403_FORBIDDEN


def test_preview_folder_endpoint_reports_a_drive_error_as_a_drive_error(
    client, tmp_path, monkeypatch
):
    """The Errno 5 case, which is what the beta tester's drive actually threw.

    Injected rather than provoked: there is no portable way to make a real
    filesystem return EIO. Permission errors keep their own branch above,
    so this one has to be exercised separately or the 503 path never runs
    in CI.
    """
    _write_jpeg(tmp_path / "IMG_0001.jpg")

    def _raise_eio(*args, **kwargs):
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(
        "app.services.folder_scanner.walk_media_files", _raise_eio
    )

    response = client.get(
        "/api/deployments/preview-folder", params={"path": str(tmp_path)}
    )

    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert "Could not read this folder" in response.json()["detail"]


def test_preview_folder_endpoint_still_reports_a_genuinely_empty_folder(
    client, tmp_path
):
    """The other half of the rule: empty must still read as empty.

    Without this, "raise on anything unusual" could quietly turn every
    empty folder into an error and nobody would notice.
    """
    empty = tmp_path / "empty"
    empty.mkdir()

    response = client.get(
        "/api/deployments/preview-folder", params={"path": str(empty)}
    )

    assert response.status_code == status.HTTP_200_OK
    assert response.json()["total_count"] == 0


# The CSV import's own case lives in tests/api/test_deployment_csv_import.py,
# next to the helpers that drive that endpoint.
