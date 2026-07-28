"""What the data migrations do to real rows.

`test_migrations.py` proves the chain builds the right *schema*. It runs
against an empty database, so every `UPDATE` and `DELETE` in the chain
matches zero rows there and their correctness is never exercised. This
module is the other half: it seeds the data shape each migration was
written to handle, runs that one migration, and checks what the rows
became.

That matters more than it sounds. A wrong data migration leaves the
schema matching the models perfectly, so the startup schema check waves
it through; the only symptom is a user whose verification work is
quietly wrong.

**Adding a data migration?** Add a test here. The recipe is the same
every time, and there are six worked examples below:

    upgrade_to("<the revision before yours>")   # the input schema
    ...insert rows with insert_row()...          # the data shape
    upgrade_to("<your revision>")                # one step
    ...assert what the rows became...

Raw SQL throughout, never the ORM factories in `tests/conftest.py`.
Those describe the schema at head; these tests write into the schema as
it was, where the columns are different.
"""

from sqlalchemy import text

from tests.db.conftest import insert_row, seed_deployment, upgrade_to


def _scalar(engine, sql: str, **params):
    with engine.connect() as conn:
        return conn.execute(text(sql), params).scalar()


# ---------------------------------------------------------------------------
# a1b2c3d4e5f6 — collapse frame File rows onto their parent video
# ---------------------------------------------------------------------------


def test_a1b2c3d4e5f6_moves_detections_onto_the_parent_video(engine) -> None:
    """Frame rows go away; the detections that pointed at them do not.

    Pre-2026-05 the pipeline created a File row per extracted video
    frame and hung detections off those. This migration repoints each
    detection at the parent video and deletes the frame rows. Losing the
    detection instead of moving it would silently empty out every video
    analysed before that refactor.
    """
    upgrade_to("2540e6edbee2")

    with engine.begin() as conn:
        _, deployment_id = seed_deployment(conn)
        video_id = insert_row(
            conn,
            "files",
            deployment_id=deployment_id,
            file_path="/videos/clip.mp4",
            file_type="video",
        )
        frame_id = insert_row(
            conn,
            "files",
            deployment_id=deployment_id,
            file_path="/videos/clip.mp4/frame000042.jpg",
            file_type="frame",
            source_video_id=video_id,
        )
        detection_id = insert_row(
            conn,
            "detections",
            file_id=frame_id,
            category="animal",
            confidence=0.9,
            frame_number=42,
        )

    upgrade_to("a1b2c3d4e5f6")

    assert _scalar(
        engine, "SELECT file_id FROM detections WHERE id = :i", i=detection_id
    ) == video_id
    assert _scalar(
        engine, "SELECT frame_number FROM detections WHERE id = :i", i=detection_id
    ) == 42
    assert _scalar(engine, "SELECT COUNT(*) FROM files WHERE file_type = 'frame'") == 0
    assert _scalar(engine, "SELECT COUNT(*) FROM files WHERE id = :i", i=video_id) == 1


# ---------------------------------------------------------------------------
# e5f6a7b8c9d0 / f6a7b8c9d0e1 / a7b8c9d0e1f2 — folder-run step renames
# ---------------------------------------------------------------------------
#
# Three migrations with one shape: rewrite `folder_run_state.$.step` for
# the projects sitting on a step that no longer exists, and leave every
# other project alone. A saved folder run whose step was not rewritten
# would open on a step the UI cannot render.


def _seed_two_projects(engine, step: str) -> tuple[str, str]:
    """One project parked on `step`, one already on a step that survives."""
    with engine.begin() as conn:
        stale = insert_row(
            conn,
            "projects",
            name="Stale",
            folder_run_state=f'{{"step": "{step}", "folder_path": "/keep/me"}}',
        )
        other = insert_row(
            conn,
            "projects",
            name="Other",
            folder_run_state='{"step": "save"}',
        )
    return stale, other


def _assert_step_rewritten(engine, stale: str, other: str, new_step: str) -> None:
    def step_of(project_id: str) -> str:
        return _scalar(
            engine,
            "SELECT json_extract(folder_run_state, '$.step') FROM projects "
            "WHERE id = :i",
            i=project_id,
        )

    assert step_of(stale) == new_step
    assert step_of(other) == "save", "a project on another step was rewritten"
    # The rest of the blob has to survive: it holds the run's settings.
    assert _scalar(
        engine,
        "SELECT json_extract(folder_run_state, '$.folder_path') FROM projects "
        "WHERE id = :i",
        i=stale,
    ) == "/keep/me"


def test_e5f6a7b8c9d0_moves_the_run_step_to_model(engine) -> None:
    upgrade_to("d4e5f6a7b8c9")
    stale, other = _seed_two_projects(engine, "run")

    upgrade_to("e5f6a7b8c9d0")

    _assert_step_rewritten(engine, stale, other, "model")


def test_f6a7b8c9d0e1_moves_the_folder_step_to_model(engine) -> None:
    upgrade_to("e5f6a7b8c9d0")
    stale, other = _seed_two_projects(engine, "folder")

    upgrade_to("f6a7b8c9d0e1")

    _assert_step_rewritten(engine, stale, other, "model")


def test_a7b8c9d0e1f2_renames_the_review_step_to_edit(engine) -> None:
    upgrade_to("f6a7b8c9d0e1")
    stale, other = _seed_two_projects(engine, "review")

    upgrade_to("a7b8c9d0e1f2")

    _assert_step_rewritten(engine, stale, other, "edit")


# ---------------------------------------------------------------------------
# c9d0e1f2a3b4 — split display_name into scientific_name + common_name
# ---------------------------------------------------------------------------


def test_c9d0e1f2a3b4_keeps_scientific_names_and_backfills_common_ones(
    engine,
) -> None:
    """The rename must be lossless, and the new column must be filled.

    `display_name` becomes `scientific_name` verbatim. `common_name` is
    derived from the class label (underscores to spaces, first letter
    capitalised), falling back to the category when there is no label,
    which is how an unclassified animal is rendered.
    """
    upgrade_to("b8c9d0e1f2a3")

    with engine.begin() as conn:
        _, deployment_id = seed_deployment(conn)
        file_id = insert_row(
            conn,
            "files",
            deployment_id=deployment_id,
            file_path="/img/1.jpg",
            file_type="image",
        )
        taxon_id = insert_row(
            conn,
            "label_taxonomy",
            classification_model_id="SPECIESNET-v4",
            name="red_deer",
            level="species",
            is_custom=0,
            display_name="Cervus elaphus",
        )
        classified = insert_row(
            conn,
            "detections",
            file_id=file_id,
            category="animal",
            confidence=0.9,
            label="red_deer",
            display_name="Cervus elaphus",
        )
        unclassified = insert_row(
            conn,
            "detections",
            file_id=file_id,
            category="animal",
            confidence=0.9,
            label=None,
        )

    upgrade_to("c9d0e1f2a3b4")

    def taxon(col: str) -> str:
        return _scalar(
            engine, f"SELECT {col} FROM label_taxonomy WHERE id = :i", i=taxon_id
        )

    def detection(col: str, det_id: str) -> str:
        return _scalar(
            engine, f"SELECT {col} FROM detections WHERE id = :i", i=det_id
        )

    # The rename carried the value across untouched.
    assert taxon("scientific_name") == "Cervus elaphus"
    assert taxon("common_name") == "Red deer"

    assert detection("scientific_name", classified) == "Cervus elaphus"
    assert detection("common_name", classified) == "Red deer"
    # No label, so the category is the best name available.
    assert detection("common_name", unclassified) == "Animal"


# ---------------------------------------------------------------------------
# f2a3b4c5d6e7 — event counts, and the deletion that lost data once
# ---------------------------------------------------------------------------


def test_f2a3b4c5d6e7_keeps_counts_and_boxed_rows_when_deleting_boxless_ones(
    engine,
) -> None:
    """The migration that destroyed user data on 2026-05-27.

    Box-less detections were the old way of recording "I counted three
    of these" and are replaced here by `event_observations.human_count`.
    The migration copies the count over and then deletes those rows.

    Two things must hold. The human's count has to survive the deletion,
    and an ordinary detection with a bounding box must not be caught up
    in it. The second assertion is the one that matters: `DELETE FROM
    detections WHERE bbox_x IS NULL` is one careless edit away from
    taking real detections with it.
    """
    upgrade_to("e1f2a3b4c5d6")

    with engine.begin() as conn:
        _, deployment_id = seed_deployment(conn)
        verified_file = insert_row(
            conn,
            "files",
            deployment_id=deployment_id,
            file_path="/img/counted.jpg",
            file_type="image",
            verified=1,
        )
        event_id = insert_row(
            conn, "events", deployment_id=deployment_id, file_count=1
        )
        insert_row(conn, "event_files", event_id=event_id, file_id=verified_file)
        observation_id = insert_row(
            conn,
            "event_observations",
            event_id=event_id,
            category="animal",
            label="deer",
            max_n=3,
            max_n_file_id=verified_file,
        )
        # The human's count: three box-less verified rows for one species.
        for _ in range(3):
            insert_row(
                conn,
                "detections",
                file_id=verified_file,
                category="animal",
                confidence=1.0,
                label="deer",
                verified=1,
                bbox_x=None,
            )
        # An ordinary detection. This one must still be here afterwards.
        boxed = insert_row(
            conn,
            "detections",
            file_id=verified_file,
            category="animal",
            confidence=0.8,
            label="deer",
            bbox_x=0.1,
            bbox_y=0.1,
            bbox_width=0.2,
            bbox_height=0.2,
        )
        # A second event whose MaxN file was never verified: it must not
        # come out of this signed off.
        unverified_file = insert_row(
            conn,
            "files",
            deployment_id=deployment_id,
            file_path="/img/untouched.jpg",
            file_type="image",
            verified=0,
        )
        unsigned_event = insert_row(
            conn, "events", deployment_id=deployment_id, file_count=1
        )
        insert_row(
            conn, "event_files", event_id=unsigned_event, file_id=unverified_file
        )
        insert_row(
            conn,
            "event_observations",
            event_id=unsigned_event,
            category="animal",
            label="fox",
            max_n=1,
            max_n_file_id=unverified_file,
        )

    upgrade_to("f2a3b4c5d6e7")

    # The count the user entered survived the deletion.
    assert _scalar(
        engine,
        "SELECT human_count FROM event_observations WHERE id = :i",
        i=observation_id,
    ) == 3

    # The box-less rows are gone...
    assert _scalar(
        engine, "SELECT COUNT(*) FROM detections WHERE bbox_x IS NULL"
    ) == 0
    # ...and the real detection is not.
    assert _scalar(
        engine, "SELECT COUNT(*) FROM detections WHERE id = :i", i=boxed
    ) == 1

    # Sign-off carried over from the old derived rule, and only where it
    # was actually earned.
    assert _scalar(
        engine, "SELECT verified FROM events WHERE id = :i", i=event_id
    ) == 1
    assert _scalar(
        engine, "SELECT verified FROM events WHERE id = :i", i=unsigned_event
    ) == 0
