"""`MediaScope` is the one owner of "which files are in this export".

The three media modules each run their own file query, and annotation
writes any in-scope file that separation did not place to a fallback
name at the media root. So the three must read the same scope, and
these tests pin that they do: separation, the preview and annotation
land on the same file set under every filter, the event vote keeps
reading the whole burst, and the deployment folder name goes into the
layout only when asked.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from pathlib import Path

from PIL import Image

from app.ml.postprocessing_outputs._output_context import OutputContext
from app.ml.postprocessing_outputs._scope import MediaScope
from app.ml.postprocessing_outputs.annotated_copies import (
    write_annotated_copies,
)
from app.ml.postprocessing_outputs.output_preview import (
    build_output_preview,
)
from app.ml.postprocessing_outputs.separate_folders import (
    separate_into_folders,
)
from app.models import Event
from app.models.event import event_files
from tests.conftest import (
    make_deployment,
    make_detection,
    make_file,
    make_project,
)


def _jpeg(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (120, 120), (180, 200, 220)).save(path, format="JPEG")
    return str(path)


def _link_event(db, deployment_id: str, file_ids: list[str]) -> Event:
    ev = Event(
        id=str(uuid.uuid4()),
        deployment_id=deployment_id,
        event_start_local=datetime(2024, 1, 1, 8, 0, 0),
        event_end_local=datetime(2024, 1, 1, 8, 1, 0),
        file_count=len(file_ids),
    )
    db.add(ev)
    db.flush()
    for seq, fid in enumerate(file_ids):
        db.execute(
            event_files.insert().values(
                event_id=ev.id, file_id=fid, sequence_number=seq
            )
        )
    db.flush()
    return ev


def _two_camera_project(db, tmp_path: Path):
    """Two deployments, `cam01` and `cam02`, each with one dog photo at
    `IMG_1.jpg`, plus a liked, verified, later photo under cam01."""
    project = make_project(db, name="scope", counting_threshold=0.5)
    cam01 = make_deployment(
        db, project_id=project.id, folder_path=str(tmp_path / "x" / "cam01")
    )
    cam02 = make_deployment(
        db, project_id=project.id, folder_path=str(tmp_path / "y" / "cam02")
    )
    files = {}
    for key, dep, rel, kw in (
        ("a", cam01, "sub/IMG_1.jpg", {}),
        (
            "liked",
            cam01,
            "sub/IMG_2.jpg",
            {
                "favorited": True,
                "verified": True,
                "captured_at_local": datetime(2024, 6, 1, 12, 0),
            },
        ),
        ("b", cam02, "IMG_1.jpg", {}),
    ):
        src = _jpeg(Path(dep.folder_path) / rel)
        f = make_file(
            db, deployment_id=dep.id, file_path=src, observation_type="animal", **kw
        )
        make_detection(db, file_id=f.id, confidence=0.9, label="dog")
        files[key] = f
    db.flush()
    return project, cam01, cam02, files


def _names_under(root: Path) -> set[str]:
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


def _run_all_three(db, project_id, target: Path, scope: MediaScope):
    """Separation, the preview and annotation with one scope. Returns
    (copied names, preview, annotated names)."""
    sep_root = target / "sep"
    separate_into_folders(
        db,
        project_id,
        OutputContext(output_root=sep_root),
        group_by="flat",
        media_threshold=0.5,
        scope=scope,
        deployment_folder_in_path=True,
    )
    preview = build_output_preview(
        db,
        project_id,
        group_by="flat",
        media_threshold=0.5,
        scope=scope,
        deployment_folder_in_path=True,
    )
    # An empty context: every in-scope file goes to the fallback name at
    # the root, so any out-of-scope file would show up there.
    ann_root = target / "ann"
    write_annotated_copies(
        db,
        project_id,
        OutputContext(output_root=ann_root),
        draw_bboxes=True,
        anonymise=False,
        media_threshold=0.5,
        scope=scope,
    )
    return _names_under(sep_root), preview, _names_under(ann_root)


def test_no_scope_copies_every_file_with_the_deployment_folder_in_the_path(
    db, tmp_path
):
    project, _, _, _ = _two_camera_project(db, tmp_path)

    copied, preview, annotated = _run_all_three(
        db, project.id, tmp_path / "out", MediaScope()
    )

    assert copied == {
        "dog/cam01/sub/IMG_1.jpg",
        "dog/cam01/sub/IMG_2.jpg",
        "dog/cam02/IMG_1.jpg",
    }
    assert preview.in_scope_files == 3
    assert dict(preview.by_media_tree) == {"dog/cam01/sub": 2, "dog/cam02": 1}
    assert annotated == {"IMG_1.jpg", "IMG_2.jpg", "IMG_1_2.jpg"}


def test_a_deployment_scope_keeps_the_other_camera_out_of_all_three(db, tmp_path):
    project, _, cam02, _ = _two_camera_project(db, tmp_path)

    copied, preview, annotated = _run_all_three(
        db, project.id, tmp_path / "out", MediaScope(deployment_ids=(cam02.id,))
    )

    assert copied == {"dog/cam02/IMG_1.jpg"}
    assert preview.in_scope_files == 1
    assert dict(preview.by_media_tree) == {"dog/cam02": 1}
    assert annotated == {"IMG_1.jpg"}


def test_liked_verified_and_date_filters_narrow_to_the_same_file(db, tmp_path):
    project, _, _, files = _two_camera_project(db, tmp_path)
    expected = {"dog/cam01/sub/IMG_2.jpg"}

    for scope in (
        MediaScope(favorited_only=True),
        MediaScope(verified_only=True),
        MediaScope(date_from=date(2024, 6, 1), date_to=date(2024, 6, 1)),
    ):
        out = tmp_path / "out" / uuid.uuid4().hex
        copied, preview, annotated = _run_all_three(db, project.id, out, scope)
        assert copied == expected, scope
        assert preview.in_scope_files == 1, scope
        assert annotated == {"IMG_2.jpg"}, scope


def test_the_event_vote_reads_files_the_filters_will_not_copy(db, tmp_path):
    """A liked photo in a burst whose other, unliked photos were verified
    as wolf is filed under wolf: the vote is over the whole event, the
    filter only decides which files are copied."""
    project = make_project(db, name="vote", counting_threshold=0.5)
    dep = make_deployment(
        db, project_id=project.id, folder_path=str(tmp_path / "cam")
    )
    liked = make_file(
        db,
        deployment_id=dep.id,
        file_path=_jpeg(tmp_path / "cam" / "liked.jpg"),
        observation_type="animal",
        favorited=True,
    )
    make_detection(db, file_id=liked.id, confidence=0.9, label="dog")
    others = []
    for i in range(2):
        f = make_file(
            db,
            deployment_id=dep.id,
            file_path=_jpeg(tmp_path / "cam" / f"other{i}.jpg"),
            observation_type="animal",
        )
        make_detection(db, file_id=f.id, confidence=0.6, label="wolf", verified=True)
        others.append(f)
    _link_event(db, dep.id, [liked.id, *(f.id for f in others)])

    target = tmp_path / "out"
    separate_into_folders(
        db,
        project.id,
        OutputContext(output_root=target),
        group_by="flat",
        media_threshold=0.5,
        group_events=True,
        scope=MediaScope(favorited_only=True),
    )

    assert _names_under(target) == {"wolf/liked.jpg"}


def test_the_layout_switch_is_off_by_default_and_flips_with_species_last(
    db, tmp_path
):
    project, _, _, _ = _two_camera_project(db, tmp_path)

    plain = tmp_path / "plain"
    separate_into_folders(
        db,
        project.id,
        OutputContext(output_root=plain),
        group_by="flat",
        media_threshold=0.5,
    )
    # Folder-run layout: subfolders relative to each deployment folder,
    # so the two cameras' root files meet and one gets a suffix.
    assert _names_under(plain) == {
        "dog/sub/IMG_1.jpg",
        "dog/sub/IMG_2.jpg",
        "dog/IMG_1.jpg",
    }

    last = tmp_path / "last"
    separate_into_folders(
        db,
        project.id,
        OutputContext(output_root=last),
        group_by="flat",
        media_threshold=0.5,
        species_last=True,
        deployment_folder_in_path=True,
    )
    assert _names_under(last) == {
        "cam01/sub/dog/IMG_1.jpg",
        "cam01/sub/dog/IMG_2.jpg",
        "cam02/dog/IMG_1.jpg",
    }


def test_from_payload_reads_the_job_row_shape():
    scope = MediaScope.from_payload(
        {
            "deployment_ids": ["d1", "d2"],
            "favorited_only": True,
            "date_from": "2024-06-01",
            "date_to": None,
        }
    )
    assert scope == MediaScope(
        deployment_ids=("d1", "d2"), favorited_only=True, date_from=date(2024, 6, 1)
    )
    assert MediaScope.from_payload({}) == MediaScope()
    assert MediaScope.from_payload({"run_id": "x"}).file_clauses() == []
