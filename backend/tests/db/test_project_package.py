"""Tests for project packages (`app/services/project_package.py`).

They live under `tests/db/` because a package is a second SQLite file
attached to the first, which the shared in-memory engine cannot do. Each
test stands up one or two "machines": a user data dir with its own
database at head, exactly what `init_db` would leave behind, minus the
startup chores.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.core.job_cancellation import JobCancelledError
from app.db.migrations import get_head_revision, upgrade_to_head
from app.ml.taxonomy_db import BUILTIN_MODEL_ID
from app.models import (
    Deployment,
    Detection,
    DetectionEmbedding,
    EventObservation,
    File,
    LabelTaxonomy,
    Project,
    Site,
)
from app.services import project_package as pkg
from tests.conftest import (
    make_deployment,
    make_detection,
    make_event_with_files,
    make_file,
    make_job,
    make_project,
    make_site,
)
from tests.db.conftest import Machine, insert_row, make_machine

# ---------------------------------------------------------------------------
# Machines
# ---------------------------------------------------------------------------


@pytest.fixture()
def lead(tmp_path: Path) -> Machine:
    return make_machine(tmp_path / "lead")


@pytest.fixture()
def member(tmp_path: Path) -> Machine:
    return make_machine(tmp_path / "member")


def _report(_message: str, _done: int, _total: int) -> None:
    pass


@dataclass
class Seed:
    project_id: str
    site_id: str
    deployments: list[str]


def seed_project(db: Session, folders: list[Path], *, name: str = "Wolves") -> Seed:
    """A project with one site, one deployment per folder, each with two
    files, one labelled detection with an embedding, and one event with
    an observation. The label is the local builtin "animal" row, the one
    row every machine has under its own id."""
    project = make_project(db, name=name, classification_model_id="EUR-DF-v1-3")
    site = make_site(db, project_id=project.id, name="North")
    animal = db.execute(
        select(LabelTaxonomy).where(
            LabelTaxonomy.classification_model_id == BUILTIN_MODEL_ID,
            LabelTaxonomy.name == "animal",
        )
    ).scalar_one()
    job = make_job(db, payload={"output_dir": "/lead/only"})
    deployments = []
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
        dep = make_deployment(
            db, site_id=site.id, folder_path=str(folder), notes="from the lead"
        )
        event = make_event_with_files(
            db, deployment_id=dep.id, event_start_local=datetime(2024, 1, 1, 12)
        )
        first = db.execute(
            select(File).where(File.deployment_id == dep.id)
        ).scalar_one()
        first.file_path = str(folder / "sub" / "IMG_0001.jpg")
        (folder / "sub").mkdir(exist_ok=True)
        (folder / "sub" / "IMG_0001.jpg").write_bytes(b"jpg")
        (folder / "IMG_0002.jpg").write_bytes(b"jpg2")
        make_file(db, deployment_id=dep.id, file_path=str(folder / "IMG_0002.jpg"))
        det = make_detection(
            db, file_id=first.id, label="animal", label_taxonomy_id=animal.id,
            job_id=job.id,
        )
        db.add(
            DetectionEmbedding(
                detection_id=det.id,
                job_id=job.id,
                embedding_model_id="DINOV2-VITS14",
                vector=b"\x00\x01" * 4,
                dimension=4,
                l2_norm=1.0,
            )
        )
        db.add(
            EventObservation(
                event_id=event.id, label_taxonomy_id=animal.id, category="animal", max_n=1
            )
        )
        deployments.append(dep.id)
    db.commit()
    return Seed(project_id=project.id, site_id=site.id, deployments=deployments)


def export(lead: Machine, project_id: str, deployment_ids: list[str], out: Path,
           *, include_media: bool = False) -> Path:
    result = pkg.export_package(
        lead.db_path, project_id, deployment_ids, out,
        include_media=include_media, report=_report,
    )
    return Path(result.package_dir)


def import_into(machine: Machine, package_dir: Path, *, replace: bool = False):
    with machine.session() as db:
        return pkg.import_package(
            db, machine.settings, package_dir, replace_existing=replace,
            report=_report,
        )


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def test_export_writes_a_complete_package(lead: Machine, tmp_path: Path) -> None:
    with lead.session() as db:
        seed = seed_project(db, [tmp_path / "cams" / "A"])
    out = tmp_path / "out"
    out.mkdir()

    package_dir = export(lead, seed.project_id, seed.deployments, out)

    assert package_dir == out / "Wolves.addaxai-bundle"
    manifest = json.loads((package_dir / "manifest.json").read_text())
    assert manifest["format"] == 1
    assert manifest["project_id"] == seed.project_id
    assert manifest["alembic_revision"] == get_head_revision()
    assert manifest["includes_media"] is False
    assert manifest["deployments"] == [
        {"id": seed.deployments[0], "media_dir": None}
    ]
    assert manifest["project_db_sha256"] == pkg._sha256(package_dir / "project.db")

    with sqlite3.connect(package_dir / "project.db") as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        assert conn.execute("SELECT version_num FROM alembic_version").fetchall() == [
            (get_head_revision(),)
        ]
        counts = {
            t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]  # noqa: S608
            for t in (
                "projects", "sites", "deployments", "files", "detections",
                "detection_embeddings", "events", "event_files",
                "event_observations", "label_taxonomy", "jobs",
                "deployment_queue", "audit_log",
            )
        }
        assert counts == {
            "projects": 1, "sites": 1, "deployments": 1, "files": 2,
            "detections": 1, "detection_embeddings": 1, "events": 1,
            "event_files": 1, "event_observations": 1,
            # Only the "animal" row is referenced; person and vehicle stay.
            "label_taxonomy": 1, "jobs": 1,
            "deployment_queue": 0, "audit_log": 0,
        }
        # Machine-local columns travel as NULL.
        assert conn.execute("SELECT thumbnail_path FROM projects").fetchone() == (None,)
        assert conn.execute("SELECT payload, result, error FROM jobs").fetchone() == (
            None, None, None,
        )


def test_export_copies_only_the_selected_deployments(lead: Machine, tmp_path: Path) -> None:
    with lead.session() as db:
        seed = seed_project(
            db, [tmp_path / "cams" / "A", tmp_path / "cams" / "B"]
        )
    out = tmp_path / "out"
    out.mkdir()

    package_dir = export(lead, seed.project_id, [seed.deployments[0]], out)

    with sqlite3.connect(package_dir / "project.db") as conn:
        assert conn.execute("SELECT id FROM deployments").fetchall() == [(seed.deployments[0],)]
        assert conn.execute("SELECT COUNT(*) FROM files").fetchone() == (2,)


def test_export_with_media_copies_each_folder_whole(lead: Machine, tmp_path: Path) -> None:
    folders = [tmp_path / "cams" / "100RECNX", tmp_path / "other" / "100RECNX"]
    with lead.session() as db:
        seed = seed_project(db, folders)
    (folders[0] / ".addaxai" / "projects" / seed.project_id).mkdir(parents=True)
    (folders[0] / ".addaxai" / "projects" / seed.project_id / "results.json").write_text("{}")
    out = tmp_path / "out"
    out.mkdir()

    package_dir = export(
        lead, seed.project_id, seed.deployments, out, include_media=True
    )

    manifest = json.loads((package_dir / "manifest.json").read_text())
    media_dirs = {d["id"]: d["media_dir"] for d in manifest["deployments"]}
    # Two folders with one basename get distinct package folders.
    assert sorted(media_dirs.values()) == ["100RECNX", "100RECNX_2"]
    first = package_dir / "media" / media_dirs[seed.deployments[0]]
    assert (first / "sub" / "IMG_0001.jpg").read_bytes() == b"jpg"
    assert (first / ".addaxai" / "projects" / seed.project_id / "results.json").is_file()


def test_export_refuses_a_missing_folder_and_never_overwrites(
    lead: Machine, tmp_path: Path
) -> None:
    with lead.session() as db:
        seed = seed_project(db, [tmp_path / "cams" / "A"])
    out = tmp_path / "out"
    out.mkdir()
    shutil.rmtree(tmp_path / "cams" / "A")

    with pytest.raises(pkg.PackageError, match="Folder not found"):
        export(lead, seed.project_id, [seed.deployments[0]], out, include_media=True)
    # A package that did not finish is removed, not left half-written.
    assert not (out / "Wolves.addaxai-bundle").exists()

    # One bundle per team member into one folder: the second gets _2.
    first = export(lead, seed.project_id, [seed.deployments[0]], out)
    second = export(lead, seed.project_id, [seed.deployments[0]], out)
    assert first.name == "Wolves.addaxai-bundle"
    assert second.name == "Wolves_2.addaxai-bundle"
    assert (out / "Wolves.addaxai-bundle" / "manifest.json").is_file()


def test_export_refuses_a_folder_run(lead: Machine, tmp_path: Path) -> None:
    with lead.session() as db:
        seed = seed_project(db, [tmp_path / "cams" / "A"])
        db.get(Project, seed.project_id).mode = "folder_run"
        db.commit()
    with pytest.raises(pkg.PackageError, match="Only projects"):
        export(lead, seed.project_id, [seed.deployments[0]], tmp_path)


def test_connect_closes_the_connection_even_on_an_error(tmp_path: Path) -> None:
    """``sqlite3``'s own ``with`` only commits. A connection left open
    through a cancel kept Windows from deleting the half-written bundle,
    and the next export into that folder was refused."""
    with pytest.raises(RuntimeError):
        with pkg._connect(tmp_path / "x.db") as conn:
            conn.execute("CREATE TABLE t (a)")
            raise RuntimeError("cancelled")
    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")


def test_package_folder_name() -> None:
    assert pkg.package_folder_name("Wolves 2026") == "Wolves 2026.addaxai-bundle"
    assert pkg.package_folder_name("a/b:c") == "a_b_c.addaxai-bundle"
    assert pkg.package_folder_name(" . ") == "project.addaxai-bundle"
    long_name = pkg.package_folder_name("Wolves " * 36)
    assert long_name == ("Wolves " * 9)[:60].strip() + ".addaxai-bundle"


# ---------------------------------------------------------------------------
# Import: new project
# ---------------------------------------------------------------------------


def test_import_creates_the_project_with_every_id_kept(
    lead: Machine, member: Machine, tmp_path: Path
) -> None:
    with lead.session() as db:
        seed = seed_project(db, [tmp_path / "cams" / "A"])
    package_dir = export(lead, seed.project_id, seed.deployments, tmp_path)

    with member.session() as db:
        preview = pkg.inspect_package(db, package_dir)
    assert preview.problems == []
    assert preview.project_exists is False
    assert preview.revision_status == "current"
    assert [d.exists for d in preview.deployments] == [False]

    result = import_into(member, package_dir)

    assert result.project_created is True
    assert result.deployments_created == 1
    for table in ("sites", "deployments", "files", "detections",
                  "detection_embeddings", "events", "event_files", "jobs"):
        assert [r[0] for r in member.rows(table)] == [r[0] for r in lead.rows(table)], table
    with member.session() as db:
        imported = db.get(Project, seed.project_id)
        assert imported.name == "Wolves"
        assert imported.classification_model_id == "EUR-DF-v1-3"
        assert imported.thumbnail_path is None
        assert db.get(Site, seed.site_id).name == "North"
        dep = db.get(Deployment, seed.deployments[0])
        assert dep.folder_path == str(tmp_path / "cams" / "A")
        assert dep.folder_status == "valid"  # the folder exists on this machine too
        # The detection points at the member's own "animal" row, which has
        # a different id than the lead's.
        det = db.execute(select(Detection)).scalar_one()
        member_animal = db.execute(
            select(LabelTaxonomy).where(LabelTaxonomy.name == "animal")
        ).scalar_one()
        assert det.label_taxonomy_id == member_animal.id
        assert det.job_id is not None
        obs = db.execute(select(EventObservation)).scalar_one()
        assert obs.label_taxonomy_id == member_animal.id
    assert member.count("label_taxonomy") == 3
    # A backup was taken first.
    assert list((member.settings.user_data_dir / "backups").glob("*before-import*"))


def test_import_creates_missing_labels_with_their_taxonomy(
    lead: Machine, member: Machine, tmp_path: Path
) -> None:
    with lead.session() as db:
        seed = seed_project(db, [tmp_path / "cams" / "A"])
        custom = LabelTaxonomy(
            classification_model_id="", name="waddan", level="genus",
            taxon_class="Mammalia", taxon_genus="Waddan", common_name="Waddan",
            scientific_name="Waddan", is_custom=True,
        )
        db.add(custom)
        db.flush()
        det = db.execute(select(Detection)).scalar_one()
        det.label = "waddan"
        det.label_taxonomy_id = custom.id
        db.commit()
        custom_id = custom.id
    package_dir = export(lead, seed.project_id, seed.deployments, tmp_path)

    import_into(member, package_dir)

    assert member.count("label_taxonomy") == 4  # three builtin rows plus waddan
    with member.session() as db:
        row = db.get(LabelTaxonomy, custom_id)
        assert row is not None and row.taxon_genus == "Waddan" and row.is_custom
        assert db.execute(select(Detection)).scalar_one().label_taxonomy_id == custom_id

    # Importing the same package again maps the label by id, creates nothing.
    import_into(member, package_dir, replace=True)
    assert member.count("label_taxonomy") == 4


def test_import_matches_a_label_by_model_and_name_case_insensitively(
    lead: Machine, member: Machine, tmp_path: Path
) -> None:
    with lead.session() as db:
        seed = seed_project(db, [tmp_path / "cams" / "A"])
        lead_fox = LabelTaxonomy(
            classification_model_id="EUR-DF-v1-3", name="Red Fox", level="species",
            taxon_species="vulpes", is_custom=False,
        )
        db.add(lead_fox)
        db.flush()
        db.execute(select(Detection)).scalar_one().label_taxonomy_id = lead_fox.id
        db.commit()
    with member.session() as db:
        member_fox = LabelTaxonomy(
            classification_model_id="EUR-DF-v1-3", name="red fox", level="species",
            taxon_species="vulpes", is_custom=False,
        )
        db.add(member_fox)
        db.commit()
        member_fox_id = member_fox.id
    package_dir = export(lead, seed.project_id, seed.deployments, tmp_path)

    import_into(member, package_dir)

    assert member.count("label_taxonomy") == 4  # three builtin rows plus the fox
    with member.session() as db:
        assert db.execute(select(Detection)).scalar_one().label_taxonomy_id == member_fox_id


# ---------------------------------------------------------------------------
# Import: same project, skip and replace
# ---------------------------------------------------------------------------


def test_import_into_the_same_project_adds_skips_or_replaces(
    lead: Machine, member: Machine, tmp_path: Path
) -> None:
    with lead.session() as db:
        seed = seed_project(
            db, [tmp_path / "cams" / "A", tmp_path / "cams" / "B"]
        )
    # Member gets A only, works on it, and adds a camera C of their own.
    package_a = export(lead, seed.project_id, [seed.deployments[0]], tmp_path / "to-member")
    import_into(member, package_a)
    with member.session() as db:
        det = db.execute(select(Detection)).scalar_one()
        det.verified = True
        det.label = "verified by member"
        dep_a = db.get(Deployment, seed.deployments[0])
        dep_a.notes = "checked by member"
        dep_a.folder_path = "/member/cams/A"
        site_south = make_site(db, project_id=seed.project_id, name="South")
        dep_c = make_deployment(db, site_id=site_south.id, folder_path="/member/cams/C")
        make_file(db, deployment_id=dep_c.id, file_path="/member/cams/C/x.jpg")
        db.get(Project, seed.project_id).counting_threshold = 0.77  # ignored on merge
        db.commit()
        dep_c_id = dep_c.id
        site_south_id = site_south.id
    package_back = export(
        member, seed.project_id, [seed.deployments[0], dep_c_id], tmp_path / "back"
    )

    with lead.session() as db:
        preview = pkg.inspect_package(db, package_back)
    assert preview.project_exists is True
    assert preview.problems == []
    assert {d.id: d.exists for d in preview.deployments} == {
        seed.deployments[0]: True, dep_c_id: False,
    }

    # Skip: A untouched, C added.
    result = import_into(lead, package_back, replace=False)
    assert (result.deployments_created, result.deployments_replaced,
            result.deployments_skipped) == (1, 0, 1)
    assert result.project_created is False
    with lead.session() as db:
        assert db.get(Deployment, seed.deployments[0]).notes == "from the lead"
        assert db.get(Deployment, dep_c_id).site_id == site_south_id
        assert db.get(Project, seed.project_id).counting_threshold != 0.77
        assert db.execute(
            select(Detection).where(Detection.label == "verified by member")
        ).scalar_one_or_none() is None

    # Replace: A takes the member's rows and metadata, keeps the lead's folder.
    result = import_into(lead, package_back, replace=True)
    assert (result.deployments_created, result.deployments_replaced,
            result.deployments_skipped) == (0, 2, 0)
    with lead.session() as db:
        dep_a = db.get(Deployment, seed.deployments[0])
        assert dep_a.notes == "checked by member"
        assert dep_a.folder_path == str(tmp_path / "cams" / "A")
        det = db.execute(
            select(Detection).join(File).where(File.deployment_id == seed.deployments[0])
        ).scalar_one()
        assert det.verified is True and det.label == "verified by member"
        files = db.execute(
            select(File.file_path).where(File.deployment_id == seed.deployments[0])
        ).scalars().all()
        assert sorted(files) == sorted([
            str(tmp_path / "cams" / "A" / "sub" / "IMG_0001.jpg"),
            str(tmp_path / "cams" / "A" / "IMG_0002.jpg"),
        ])
        # B was never in the package and is untouched.
        assert db.get(Deployment, seed.deployments[1]).notes == "from the lead"
    assert lead.count("deployments") == 3
    assert lead.count("files") == 5


# ---------------------------------------------------------------------------
# Import: refusals
# ---------------------------------------------------------------------------


def test_preview_refuses_a_project_name_clash(
    lead: Machine, member: Machine, tmp_path: Path
) -> None:
    with lead.session() as db:
        seed = seed_project(db, [tmp_path / "cams" / "A"])
    with member.session() as db:
        make_project(db, name="Wolves")
        db.commit()
    package_dir = export(lead, seed.project_id, [seed.deployments[0]], tmp_path)

    with member.session() as db:
        preview = pkg.inspect_package(db, package_dir)
    assert preview.problems == [
        'A different project named "Wolves" already exists on this computer. '
        "Rename it and import again."
    ]
    with pytest.raises(pkg.PackageError, match="different project named"):
        import_into(member, package_dir)
    assert member.count("deployments") == 0


def test_preview_refuses_a_site_name_clash(
    lead: Machine, member: Machine, tmp_path: Path
) -> None:
    with lead.session() as db:
        seed = seed_project(db, [tmp_path / "cams" / "A"])
    package_dir = export(lead, seed.project_id, [seed.deployments[0]], tmp_path)
    import_into(member, package_dir)
    # Both sides invent "South" independently.
    with lead.session() as db:
        make_site(db, project_id=seed.project_id, name="South")
        db.commit()
    with member.session() as db:
        south = make_site(db, project_id=seed.project_id, name="South")
        dep = make_deployment(db, site_id=south.id, folder_path="/m/C")
        db.commit()
        dep_id = dep.id
    package_back = export(member, seed.project_id, [dep_id], tmp_path / "back")

    with lead.session() as db:
        preview = pkg.inspect_package(db, package_back)
    assert preview.problems == [
        'A different site named "South" already exists in this project. '
        "Rename it and import again."
    ]


def test_preview_refuses_a_newer_package_and_upgrades_an_older_one(
    lead: Machine, member: Machine, tmp_path: Path
) -> None:
    with lead.session() as db:
        seed = seed_project(db, [tmp_path / "cams" / "A"])
    package_dir = export(lead, seed.project_id, [seed.deployments[0]], tmp_path)
    db_path = package_dir / "project.db"

    def stamp(revision: str) -> None:
        with sqlite3.connect(db_path) as conn:
            conn.execute("UPDATE alembic_version SET version_num = ?", (revision,))
        manifest = json.loads((package_dir / "manifest.json").read_text())
        manifest["project_db_sha256"] = pkg._sha256(db_path)
        manifest["app_version"] = "9.9.9"
        (package_dir / "manifest.json").write_text(json.dumps(manifest))

    stamp("ffffffffffff")
    with member.session() as db:
        preview = pkg.inspect_package(db, package_dir)
    assert preview.revision_status == "newer"
    assert "Update AddaxAI" in preview.problems[0]

    # One revision back: the chain is replayed on a copy, the package
    # itself is left as it was.
    from alembic import command
    from app.db.migrations import _alembic_config

    stamp(get_head_revision())
    command.downgrade(_alembic_config(f"sqlite:///{db_path}"), "-1")
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA journal_mode=DELETE")
        older = conn.execute("SELECT version_num FROM alembic_version").fetchone()[0]
    stamp(older)
    with member.session() as db:
        preview = pkg.inspect_package(db, package_dir)
    assert preview.revision_status == "older"
    assert preview.problems == []
    assert "brought up to date" in preview.warnings[0]
    result = import_into(member, package_dir)
    assert result.deployments_created == 1
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT version_num FROM alembic_version").fetchone()[0] == older


def test_import_refuses_a_damaged_or_incomplete_package(
    lead: Machine, member: Machine, tmp_path: Path
) -> None:
    with lead.session() as db:
        seed = seed_project(db, [tmp_path / "cams" / "A"])
    package_dir = export(lead, seed.project_id, [seed.deployments[0]], tmp_path)

    with member.session() as db:
        with pytest.raises(pkg.PackageError, match="not an AddaxAI bundle"):
            pkg.inspect_package(db, tmp_path)
        with pytest.raises(pkg.PackageError, match="not a folder"):
            pkg.inspect_package(db, package_dir / "manifest.json")

    with sqlite3.connect(package_dir / "project.db") as conn:
        conn.execute("UPDATE projects SET description = 'tampered'")
        conn.execute("PRAGMA journal_mode=DELETE")
    with pytest.raises(pkg.PackageError, match="does not match its manifest"):
        import_into(member, package_dir)
    assert member.count("projects") == 0

    (package_dir / "project.db").unlink()
    with member.session() as db:
        with pytest.raises(pkg.PackageError, match="incomplete"):
            pkg.inspect_package(db, package_dir)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def test_import_points_deployments_into_the_package_media(
    lead: Machine, member: Machine, tmp_path: Path
) -> None:
    folder = tmp_path / "cams" / "A"
    with lead.session() as db:
        seed = seed_project(db, [folder])
        first = db.execute(
            select(File).where(File.file_path.like("%IMG_0001%"))
        ).scalar_one()
        first.best_frame_path = str(
            folder / ".addaxai" / "projects" / seed.project_id / "video_frames" / "f.jpg"
        )
        db.commit()
    package_dir = export(lead, seed.project_id, [seed.deployments[0]], tmp_path, include_media=True)

    import_into(member, package_dir)

    media = package_dir / "media" / "A"
    with member.session() as db:
        dep = db.get(Deployment, seed.deployments[0])
        assert dep.folder_path == str(media)
        assert dep.folder_status == "valid"
        paths = dict(
            db.execute(select(File.file_path, File.best_frame_path)).all()
        )
        assert sorted(paths) == sorted([
            str(media / "sub" / "IMG_0001.jpg"), str(media / "IMG_0002.jpg"),
        ])
        assert paths[str(media / "sub" / "IMG_0001.jpg")] == str(
            media / ".addaxai" / "projects" / seed.project_id / "video_frames" / "f.jpg"
        )


def test_import_marks_a_folder_that_is_not_here_for_reconnecting(
    lead: Machine, member: Machine, tmp_path: Path
) -> None:
    with lead.session() as db:
        seed = seed_project(db, [tmp_path / "cams" / "A"])
        db.get(Deployment, seed.deployments[0]).folder_path = "/somewhere/else/A"
        db.commit()
    package_dir = export(lead, seed.project_id, [seed.deployments[0]], tmp_path)

    import_into(member, package_dir)

    with member.session() as db:
        dep = db.get(Deployment, seed.deployments[0])
        assert dep.folder_path == "/somewhere/else/A"
        assert dep.folder_status == "needs_relink"


def test_rewrite_converts_the_other_systems_separators_and_guards_the_prefix(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "x.db"
    upgrade_to_head(f"sqlite:///{db_path}")
    engine = create_engine(f"sqlite:///{db_path}", future=True)
    with engine.begin() as conn:
        insert_row(conn, "projects", id="p", name="p")
        insert_row(conn, "deployments", id="d", project_id="p", folder_path="C:\\cams\\A")
        insert_row(
            conn, "files", id="f1", deployment_id="d",
            file_path="C:\\cams\\A\\100RECNX\\IMG_0001.JPG",
        )
        insert_row(
            conn, "files", id="f2", deployment_id="d",
            file_path="D:\\elsewhere\\IMG_0002.JPG",
        )
    engine.dispose()

    with sqlite3.connect(db_path) as conn:
        pkg._rewrite_paths(conn, "d", "C:\\cams\\A", "/Volumes/stick/A")
        rows = dict(conn.execute("SELECT id, file_path FROM files").fetchall())
        assert rows["f1"] == "/Volumes/stick/A/100RECNX/IMG_0001.JPG"
        # Not under the old folder: left alone.
        assert rows["f2"] == "D:\\elsewhere\\IMG_0002.JPG"
        assert conn.execute(
            "SELECT folder_path, folder_status FROM deployments"
        ).fetchone() == ("/Volumes/stick/A", "needs_relink")


def test_a_cancel_during_the_photo_copy_stops_it_and_leaves_nothing(
    lead: Machine, tmp_path: Path
) -> None:

    folder = tmp_path / "cams" / "A"
    with lead.session() as db:
        seed = seed_project(db, [folder])
    for i in range(20):
        (folder / f"extra_{i}.jpg").write_bytes(b"x")
    copied: list[int] = []

    def report(message: str, done: int, _total: int) -> None:
        if message.startswith("Copying photos"):
            copied.append(done)
            if done == 3:
                raise JobCancelledError()

    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(JobCancelledError):
        pkg.export_package(
            lead.db_path, seed.project_id, seed.deployments, out,
            include_media=True, report=report,
        )
    assert copied == [1, 2, 3]
    assert list(out.iterdir()) == []


def test_a_label_id_that_exists_locally_is_reused_whatever_its_name(
    lead: Machine, member: Machine, tmp_path: Path
) -> None:
    """A custom label renamed on one side keeps its identity by id."""
    with lead.session() as db:
        seed = seed_project(db, [tmp_path / "cams" / "A"])
        label_id = str(uuid.uuid4())
        db.add(LabelTaxonomy(
            id=label_id, classification_model_id="", name="stump", level="unknown",
            is_custom=True,
        ))
        db.flush()
        db.execute(select(Detection)).scalar_one().label_taxonomy_id = label_id
        db.commit()
    with member.session() as db:
        db.add(LabelTaxonomy(
            id=label_id, classification_model_id="", name="rock", level="unknown",
            is_custom=True,
        ))
        db.commit()
    package_dir = export(lead, seed.project_id, [seed.deployments[0]], tmp_path)

    import_into(member, package_dir)

    assert member.count("label_taxonomy") == 4  # three builtin rows plus rock
    with member.session() as db:
        assert db.get(LabelTaxonomy, label_id).name == "rock"
        assert db.execute(select(Detection)).scalar_one().label_taxonomy_id == label_id


def test_upgrade_to_head_with_a_url_touches_only_that_file(tmp_path: Path) -> None:
    """`upgrade_to_head(url)` touches only the file it is given."""
    other = tmp_path / "other.db"
    upgrade_to_head(f"sqlite:///{other}")
    with sqlite3.connect(other) as conn:
        assert conn.execute("SELECT version_num FROM alembic_version").fetchone() == (
            get_head_revision(),
        )
    assert not (tmp_path / "addaxai.db").exists()


# ---------------------------------------------------------------------------
# The jobs
# ---------------------------------------------------------------------------


def _run_job(machine: Machine, monkeypatch, coro_fn, job_type: str, payload: dict):
    """Create a job on `machine` and run the worker against it, the way
    the WebSocket "ready" handler would."""
    import asyncio

    from app.workers import project_package_worker as worker

    db = machine.session()
    monkeypatch.setattr(worker, "get_db", lambda: iter([db]))
    monkeypatch.setattr(worker, "get_settings", lambda: machine.settings)
    job = make_job(db, job_type=job_type, payload=payload)
    db.commit()
    job_id = job.id
    asyncio.run(coro_fn(job_id))
    with machine.session() as check:
        return check.get(type(job), job_id)


def test_export_and_import_jobs_round_trip(
    lead: Machine, member: Machine, tmp_path: Path, monkeypatch
) -> None:
    from app.workers.project_package_worker import (
        process_package_export_job,
        process_package_import_job,
    )

    with lead.session() as db:
        seed = seed_project(db, [tmp_path / "cams" / "A"])
    out = tmp_path / "out"
    out.mkdir()

    job = _run_job(
        lead, monkeypatch, process_package_export_job, "package_export",
        {
            "project_id": seed.project_id,
            "deployment_ids": seed.deployments,
            "output_dir": str(out),
            "include_media": True,
        },
    )
    assert job.status == "completed", job.error
    assert job.result["deployments"] == 1 and job.result["includes_media"] is True
    package_dir = Path(job.result["package_dir"])
    assert (package_dir / "manifest.json").is_file()

    job = _run_job(
        member, monkeypatch, process_package_import_job, "package_import",
        {"path": str(package_dir), "replace_existing": False},
    )
    assert job.status == "completed", job.error
    assert job.result["project_created"] is True
    assert job.result["deployments_created"] == 1
    assert member.count("files") == 2


def test_a_cancelled_export_leaves_no_package(
    lead: Machine, tmp_path: Path, monkeypatch
) -> None:
    from app.core.job_cancellation import request_cancel
    from app.workers import project_package_worker as worker

    with lead.session() as db:
        seed = seed_project(db, [tmp_path / "cams" / "A"])
    out = tmp_path / "out"
    out.mkdir()

    # Cancel before the first progress report lands.
    real_relay = worker._progress_relay

    def relay_then_cancel(job_id, loop, *, cancellable):
        request_cancel(job_id)
        return real_relay(job_id, loop, cancellable=cancellable)

    monkeypatch.setattr(worker, "_progress_relay", relay_then_cancel)
    job = _run_job(
        lead, monkeypatch, worker.process_package_export_job, "package_export",
        {
            "project_id": seed.project_id,
            "deployment_ids": seed.deployments,
            "output_dir": str(out),
            "include_media": False,
        },
    )
    assert job.status == "cancelled"
    assert list(out.iterdir()) == []


def test_a_failed_import_marks_the_job_failed_and_writes_nothing(
    lead: Machine, member: Machine, tmp_path: Path, monkeypatch
) -> None:
    from app.workers.project_package_worker import process_package_import_job

    with lead.session() as db:
        seed = seed_project(db, [tmp_path / "cams" / "A"])
    package_dir = export(lead, seed.project_id, seed.deployments, tmp_path)
    with sqlite3.connect(package_dir / "project.db") as conn:
        conn.execute("UPDATE projects SET description = 'tampered'")
        conn.execute("PRAGMA journal_mode=DELETE")

    job = _run_job(
        member, monkeypatch, process_package_import_job, "package_import",
        {"path": str(package_dir), "replace_existing": False},
    )
    assert job.status == "failed"
    assert "does not match its manifest" in job.error
    assert member.count("projects") == 0


# ---------------------------------------------------------------------------
# Bug hunt regressions
# ---------------------------------------------------------------------------


def test_export_refuses_a_folder_inside_its_own_camera_folders(
    lead: Machine, tmp_path: Path
) -> None:
    """With photos, a package inside a deployment folder copied itself
    into itself until the path got too long."""
    folder = tmp_path / "cams" / "A"
    with lead.session() as db:
        seed = seed_project(db, [folder])
    for inside in (folder, folder / "exports"):
        inside.mkdir(exist_ok=True)
        with pytest.raises(pkg.PackageError, match="outside the camera folders"):
            export(lead, seed.project_id, seed.deployments, inside, include_media=True)
        assert not (inside / "Wolves.addaxai-bundle").exists()
    # Next to the folder is fine.
    export(lead, seed.project_id, seed.deployments, tmp_path / "cams", include_media=True)


def test_a_damaged_data_file_or_manifest_is_refused_in_words(
    lead: Machine, member: Machine, tmp_path: Path
) -> None:
    with lead.session() as db:
        seed = seed_project(db, [tmp_path / "cams" / "A"])
    package_dir = export(lead, seed.project_id, seed.deployments, tmp_path)
    db_path = package_dir / "project.db"
    good = db_path.read_bytes()

    for damaged in (b"", good[:20000], b"not a database" * 400):
        db_path.write_bytes(damaged)
        with member.session() as db:
            with pytest.raises(pkg.PackageError, match="project.db cannot be read"):
                pkg.inspect_package(db, package_dir)

    db_path.write_bytes(good)
    (package_dir / "manifest.json").write_text("{not json")
    with member.session() as db:
        with pytest.raises(pkg.PackageError) as err:
            pkg.inspect_package(db, package_dir)
    assert str(err.value) == (
        "The bundle is damaged: its manifest.json cannot be read. "
        "Export the bundle again."
    )


def test_replace_from_a_package_with_photos_points_into_that_package(
    lead: Machine, member: Machine, tmp_path: Path
) -> None:
    """A second hand-out with photos replaces the first one's. Keeping
    the first package's path broke the deployment once it was deleted."""
    with lead.session() as db:
        seed = seed_project(db, [tmp_path / "cams" / "A"])
    first = export(lead, seed.project_id, seed.deployments, tmp_path / "one",
                   include_media=True)
    second = export(lead, seed.project_id, seed.deployments, tmp_path / "two",
                    include_media=True)
    import_into(member, first)

    import_into(member, second, replace=True)

    with member.session() as db:
        dep = db.get(Deployment, seed.deployments[0])
        assert dep.folder_path == str(second / "media" / "A")
        assert dep.folder_status == "valid"


def test_replace_without_photos_keeps_the_local_folder(
    lead: Machine, member: Machine, tmp_path: Path
) -> None:
    """The return trip: a data-only package never moves the lead's folder,
    and neither does a package whose photos went missing."""
    folder = tmp_path / "cams" / "A"
    with lead.session() as db:
        seed = seed_project(db, [folder])
    with_photos = export(lead, seed.project_id, seed.deployments, tmp_path / "one",
                         include_media=True)
    shutil.rmtree(with_photos / "media")
    manifest = json.loads((with_photos / "manifest.json").read_text())
    assert manifest["deployments"][0]["media_dir"] == "A"

    import_into(lead, with_photos, replace=True)

    with lead.session() as db:
        assert db.get(Deployment, seed.deployments[0]).folder_path == str(folder)
