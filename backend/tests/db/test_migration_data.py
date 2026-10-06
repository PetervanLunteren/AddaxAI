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


def _rows(engine, sql: str, **params):
    with engine.connect() as conn:
        return conn.execute(text(sql), params).all()


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


# ---------------------------------------------------------------------------
# 5e6f7a8b9c0d — recompute observation_type under the strongest-detection rule
# ---------------------------------------------------------------------------


def _seed_file_with_detections(
    conn,
    deployment_id,
    dets,
    stored,
    *,
    file_type="image",
    best_frame_number=None,
):
    """One file with `dets` = [(category, confidence, verified), ...] and
    an `observation_type` as the old rule would have stored it.

    A fourth element on a detection tuple is its `frame_number`, for the
    video cases. Images leave it NULL, which is what they have in life.
    """
    file_id = insert_row(
        conn,
        "files",
        deployment_id=deployment_id,
        file_type=file_type,
        observation_type=stored,
        best_frame_number=best_frame_number,
    )
    for det in dets:
        category, conf, verified = det[0], det[1], det[2]
        insert_row(
            conn,
            "detections",
            file_id=file_id,
            category=category,
            confidence=conf,
            verified=verified,
            frame_number=det[3] if len(det) > 3 else None,
        )
    return file_id


def test_5e6f7a8b9c0d_recomputes_observation_type(engine):
    """The old rule ranked categories, so an animal always beat a person.
    The new rule ranks the detections themselves. Every stored value is
    therefore suspect, and nothing recomputes a denormalised column on
    its own, so the migration has to."""
    upgrade_to("4d5e6f7a8b9c")

    with engine.begin() as conn:
        project_id = insert_row(
            conn, "projects", name="Test project", counting_threshold=0.2
        )
        deployment_id = insert_row(
            conn, "deployments", project_id=project_id
        )

        # The case that motivated the change: a confident person and a
        # weaker animal. Priority said "animal"; the person is stronger.
        mixed = _seed_file_with_detections(
            conn, deployment_id,
            [("person", 0.95, 0), ("animal", 0.80, 0)],
            stored="animal",
        )
        # The mirror: nothing is biased against animals.
        animal_wins = _seed_file_with_detections(
            conn, deployment_id,
            [("person", 0.80, 0), ("animal", 0.95, 0)],
            stored="animal",
        )
        # A plain person file: only the spelling changes, human -> person.
        person_only = _seed_file_with_detections(
            conn, deployment_id, [("person", 0.90, 0)], stored="human",
        )
        # A human decision outranks a more confident model guess.
        verified_person = _seed_file_with_detections(
            conn, deployment_id,
            [("animal", 0.99, 0), ("person", 0.10, 1)],
            stored="animal",
        )
        # Everything below the project threshold: no trusted content.
        all_below = _seed_file_with_detections(
            conn, deployment_id, [("animal", 0.10, 0)], stored="animal",
        )
        # No detections at all.
        no_dets = _seed_file_with_detections(
            conn, deployment_id, [], stored="blank",
        )

    upgrade_to("5e6f7a8b9c0d")

    def obs(file_id):
        return _scalar(
            engine,
            "SELECT observation_type FROM files WHERE id = :i",
            i=file_id,
        )

    assert obs(mixed) == "person"
    assert obs(animal_wins) == "animal"
    assert obs(person_only) == "person"
    assert obs(verified_person) == "person"
    assert obs(all_below) == "blank"
    assert obs(no_dets) == "blank"


def test_5e6f7a8b9c0d_passes_a_novel_detector_category_through(engine):
    """A detector that is not MegaDetector keeps its own category. The
    old rule dropped anything outside animal/person/vehicle, so such a
    file read as blank."""
    upgrade_to("4d5e6f7a8b9c")

    with engine.begin() as conn:
        project_id = insert_row(
            conn, "projects", name="Marine", counting_threshold=0.2
        )
        deployment_id = insert_row(conn, "deployments", project_id=project_id)
        shark = _seed_file_with_detections(
            conn, deployment_id,
            [("fish", 0.40, 0), ("shark", 0.90, 0)],
            stored="blank",
        )

    upgrade_to("5e6f7a8b9c0d")

    assert _scalar(
        engine,
        "SELECT observation_type FROM files WHERE id = :i",
        i=shark,
    ) == "shark"


# ---------------------------------------------------------------------------
# 6f7a8b9c0d1e — video observation_type comes from the best frame
# ---------------------------------------------------------------------------


def _downgrade_to(revision: str) -> None:
    """Run the chain back down to `revision`.

    Test-local for the same reason `upgrade_to` is: the app only ever
    upgrades to head, so a partial move does not belong in
    `app.db.migrations`.
    """
    from alembic import command
    from app.db.migrations import _alembic_config

    command.downgrade(_alembic_config(), revision)


def test_6f7a8b9c0d1e_videos_follow_their_best_frame(engine):
    """A video is saved as one frame, so only that frame's boxes may say
    what it is. The column is denormalised, so the migration has to
    recompute it; images must come out untouched."""
    upgrade_to("5e6f7a8b9c0d")

    with engine.begin() as conn:
        project_id = insert_row(
            conn, "projects", name="Video project", counting_threshold=0.2
        )
        deployment_id = insert_row(conn, "deployments", project_id=project_id)

        # The whole point: the strongest box sits on a frame that was
        # never written to disk, so the video reads blank.
        off_frame_only = _seed_file_with_detections(
            conn, deployment_id,
            [("animal", 0.90, 0, 7)],
            stored="animal",
            file_type="video",
            best_frame_number=3,
        )
        # The strongest box on the saved frame decides, and the stronger
        # box on another frame does not.
        best_frame_wins = _seed_file_with_detections(
            conn, deployment_id,
            [("animal", 0.95, 0, 7), ("person", 0.60, 0, 3)],
            stored="animal",
            file_type="video",
            best_frame_number=3,
        )
        # A human decision is never out of reach, whatever frame it is on.
        verified_off_frame = _seed_file_with_detections(
            conn, deployment_id,
            [("animal", 0.10, 1, 7), ("person", 0.60, 0, 3)],
            stored="person",
            file_type="video",
            best_frame_number=3,
        )
        # Frame extraction failed, so there is no visible surface at all.
        no_best_frame = _seed_file_with_detections(
            conn, deployment_id,
            [("animal", 0.90, 0, 4)],
            stored="animal",
            file_type="video",
            best_frame_number=None,
        )
        # An image has frame_number and best_frame_number both NULL. The
        # UPDATE is scoped to videos precisely so `NULL = NULL` can never
        # be evaluated for it.
        image = _seed_file_with_detections(
            conn, deployment_id, [("animal", 0.90, 0)], stored="animal",
        )

    upgrade_to("6f7a8b9c0d1e")

    def obs(file_id):
        return _scalar(
            engine,
            "SELECT observation_type FROM files WHERE id = :i",
            i=file_id,
        )

    assert obs(off_frame_only) == "blank"
    assert obs(best_frame_wins) == "person"
    assert obs(verified_off_frame) == "animal"
    assert obs(no_best_frame) == "blank"
    assert obs(image) == "animal"


def test_6f7a8b9c0d1e_downgrade_restores_the_every_frame_rule(engine):
    """Downgrade means undo this revision, not undo two: it restores the
    ungated strongest-detection rule, not the category priority that
    5e6f7a8b9c0d replaced."""
    upgrade_to("5e6f7a8b9c0d")

    with engine.begin() as conn:
        project_id = insert_row(
            conn, "projects", name="Round trip", counting_threshold=0.2
        )
        deployment_id = insert_row(conn, "deployments", project_id=project_id)
        video = _seed_file_with_detections(
            conn, deployment_id,
            [("animal", 0.90, 0, 7)],
            stored="animal",
            file_type="video",
            best_frame_number=3,
        )

    upgrade_to("6f7a8b9c0d1e")
    assert _scalar(
        engine, "SELECT observation_type FROM files WHERE id = :i", i=video
    ) == "blank"

    _downgrade_to("5e6f7a8b9c0d")
    assert _scalar(
        engine, "SELECT observation_type FROM files WHERE id = :i", i=video
    ) == "animal"


# ---------------------------------------------------------------------------
# a1b2c3d4e5f7 — observation cohorts: several rows per species per event
# ---------------------------------------------------------------------------


def test_a1b2c3d4e5f7_allows_two_cohorts_of_one_species(engine):
    """The whole point of the migration is the constraint it drops, and
    nothing else notices a dropped constraint: `schema_problems()` only
    reports what the models declare and the DB lacks. So this inserts the
    second row the old schema refused. Existing rows read NULL (unknown)
    in the new columns, and the event gets its notes column."""
    upgrade_to("9c0d1e2f3a4b")

    with engine.begin() as conn:
        project_id, deployment_id = seed_deployment(conn)
        event_id = insert_row(conn, "events", deployment_id=deployment_id)
        taxonomy_id = insert_row(
            conn, "label_taxonomy", name="deer", level="species",
            classification_model_id="m", project_id=project_id,
        )
        first = insert_row(
            conn, "event_observations", event_id=event_id,
            label="deer", label_taxonomy_id=taxonomy_id,
            category="animal", max_n=2,
        )

    upgrade_to("a1b2c3d4e5f7")

    with engine.begin() as conn:
        second = insert_row(
            conn, "event_observations", event_id=event_id,
            label="deer", label_taxonomy_id=taxonomy_id,
            category="animal", max_n=0, human_count=1, sex="female",
        )

    assert _scalar(
        engine,
        "SELECT COUNT(*) FROM event_observations "
        "WHERE event_id = :e AND label_taxonomy_id = :t",
        e=event_id, t=taxonomy_id,
    ) == 2
    assert _scalar(
        engine,
        "SELECT sex IS NULL AND life_stage IS NULL AND behavior IS NULL "
        "FROM event_observations WHERE id = :i",
        i=first,
    ) == 1
    assert _scalar(
        engine, "SELECT notes IS NULL FROM events WHERE id = :i", i=event_id
    ) == 1
    assert _scalar(
        engine, "SELECT sex FROM event_observations WHERE id = :i", i=second
    ) == "female"


# ---------------------------------------------------------------------------
# c3d4e5f6a7b9 — custom labels become global (per-project rows merged)
# ---------------------------------------------------------------------------


def test_c3d4e5f6a7b9_merges_custom_labels_into_one_global_row(engine) -> None:
    """Two projects' custom "deer" become one global row, ranks kept.

    Custom labels used to be per-project. This migration merges same-name
    rows into one shared row (``project_id IS NULL``), keeping the one with
    taxonomy ranks and repointing the other's detections and observations.
    Losing a detection's label, or keeping two "deer" rows, would both be
    wrong.
    """
    upgrade_to("b2c3d4e5f6a8")

    with engine.begin() as conn:
        p1 = insert_row(conn, "projects", name="P1")
        p2 = insert_row(conn, "projects", name="P2")
        dep1 = insert_row(conn, "deployments", project_id=p1)
        dep2 = insert_row(conn, "deployments", project_id=p2)
        f1 = insert_row(conn, "files", deployment_id=dep1, file_path="/a.jpg")
        f2 = insert_row(conn, "files", deployment_id=dep2, file_path="/b.jpg")
        ev2 = insert_row(conn, "events", deployment_id=dep2)

        # Bare custom "deer" in P1 (older), rank-filled custom "deer" in P2.
        bare = insert_row(
            conn, "label_taxonomy", name="deer", level="unknown",
            classification_model_id="", project_id=p1, is_custom=1,
            created_at_utc="2026-01-01 00:00:00",
        )
        ranked = insert_row(
            conn, "label_taxonomy", name="Deer", level="species",
            classification_model_id="", project_id=p2, is_custom=1,
            created_at_utc="2026-02-01 00:00:00",
            taxon_class="mammalia", taxon_species="elaphus",
        )
        det1 = insert_row(
            conn, "detections", file_id=f1, category="animal",
            confidence=0.9, label="deer", label_taxonomy_id=bare,
        )
        det2 = insert_row(
            conn, "detections", file_id=f2, category="animal",
            confidence=0.9, label="deer", label_taxonomy_id=ranked,
        )
        obs = insert_row(
            conn, "event_observations", event_id=ev2, label="deer",
            label_taxonomy_id=bare, category="animal", max_n=1,
        )

    upgrade_to("c3d4e5f6a7b9")

    # One global "deer" row survives, and it is the rank-filled one.
    assert _scalar(
        engine, "SELECT COUNT(*) FROM label_taxonomy WHERE lower(name)='deer'"
    ) == 1
    survivor = _scalar(
        engine, "SELECT id FROM label_taxonomy WHERE lower(name)='deer'"
    )
    assert survivor == ranked
    assert _scalar(
        engine, "SELECT project_id FROM label_taxonomy WHERE id=:i", i=ranked
    ) is None
    # Both detections and the observation now point at the survivor.
    assert _scalar(
        engine, "SELECT label_taxonomy_id FROM detections WHERE id=:i", i=det1
    ) == ranked
    assert _scalar(
        engine, "SELECT label_taxonomy_id FROM detections WHERE id=:i", i=det2
    ) == ranked
    assert _scalar(
        engine, "SELECT label_taxonomy_id FROM event_observations WHERE id=:i", i=obs
    ) == ranked


def test_c3d4e5f6a7b9_clears_project_id_on_a_lone_custom_label(engine) -> None:
    """A custom label with no same-name sibling just goes global in place."""
    upgrade_to("b2c3d4e5f6a8")

    with engine.begin() as conn:
        p1 = insert_row(conn, "projects", name="P1")
        lone = insert_row(
            conn, "label_taxonomy", name="stump", level="unknown",
            classification_model_id="", project_id=p1, is_custom=1,
        )

    upgrade_to("c3d4e5f6a7b9")

    assert _scalar(
        engine, "SELECT project_id FROM label_taxonomy WHERE id=:i", i=lone
    ) is None
    assert _scalar(
        engine, "SELECT COUNT(*) FROM label_taxonomy WHERE name='stump'"
    ) == 1


# ---------------------------------------------------------------------------
# c3d4e5f6a7b9 — exhaustive custom-label merge coverage
# ---------------------------------------------------------------------------
#
# The migration above has focused tests; this one throws the whole matrix at
# it on one database: many names, many projects, ranked vs bare, mixed case,
# an already-global row in a group, lone labels, detections and observations
# on winners and losers, and a bulk of unique labels. It then asserts the
# global invariants that must hold however the data came in.


def _ranks(**kw):
    """Only the taxon_* columns, so a row can be seeded with or without ranks."""
    return {k: v for k, v in kw.items() if v is not None}


def test_c3d4e5f6a7b9_merge_matrix(engine) -> None:
    upgrade_to("b2c3d4e5f6a8")

    # Track, per lowercased name: the id we expect to survive, and every
    # (detection_id / observation_id) that must end up pointing at it.
    expected_winner: dict[str, str] = {}
    expected_det: dict[str, list[str]] = {}
    expected_obs: dict[str, list[str]] = {}

    with engine.begin() as conn:
        # A handful of projects, each with a deployment + file + event so we
        # can hang detections and observations off real parents.
        projects = []
        for i in range(5):
            pid = insert_row(conn, "projects", name=f"P{i}")
            dep = insert_row(conn, "deployments", project_id=pid)
            fil = insert_row(conn, "files", deployment_id=dep, file_path=f"/{i}.jpg")
            ev = insert_row(conn, "events", deployment_id=dep)
            projects.append({"id": pid, "file": fil, "event": ev})

        def custom(name, pid, created, *, project_id="__set__", **ranks):
            """Insert a custom label row; project_id defaults to the project."""
            return insert_row(
                conn, "label_taxonomy", name=name, level="unknown",
                classification_model_id="", is_custom=1,
                project_id=(pid if project_id == "__set__" else project_id),
                created_at_utc=created, **_ranks(**ranks),
            )

        def det(name, p, taxonomy_id):
            d = insert_row(
                conn, "detections", file_id=p["file"], category="animal",
                confidence=0.9, label=name, label_taxonomy_id=taxonomy_id,
            )
            expected_det.setdefault(name.lower(), []).append(d)
            return d

        def obs(name, p, taxonomy_id):
            o = insert_row(
                conn, "event_observations", event_id=p["event"], label=name,
                category="animal", max_n=1, label_taxonomy_id=taxonomy_id,
            )
            expected_obs.setdefault(name.lower(), []).append(o)
            return o

        # 1. "deer": bare(oldest) / ranked / bare -> ranked wins.
        d_bare = custom("deer", projects[0]["id"], "2026-01-01 00:00:00")
        d_rank = custom("deer", projects[1]["id"], "2026-02-01 00:00:00",
                        taxon_class="mammalia", taxon_species="elaphus")
        d_bare3 = custom("deer", projects[2]["id"], "2026-03-01 00:00:00")
        expected_winner["deer"] = d_rank
        det("deer", projects[0], d_bare)
        det("deer", projects[1], d_rank)
        det("deer", projects[2], d_bare3)
        obs("deer", projects[0], d_bare)      # observation on a loser
        obs("deer", projects[1], d_rank)      # observation on the winner

        # 2. case-insensitive "Fox"/"fox"/"FOX", no ranks -> oldest wins.
        f1 = custom("Fox", projects[0]["id"], "2026-02-01 00:00:00")
        f2 = custom("fox", projects[1]["id"], "2026-01-01 00:00:00")  # oldest
        f3 = custom("FOX", projects[3]["id"], "2026-03-01 00:00:00")
        expected_winner["fox"] = f2
        det("Fox", projects[0], f1)
        det("fox", projects[1], f2)
        det("FOX", projects[3], f3)

        # 3. "boar": two ranked rows -> older ranked wins.
        b_new = custom("boar", projects[1]["id"], "2026-02-01 00:00:00",
                       taxon_class="mammalia")
        b_old = custom("boar", projects[2]["id"], "2026-01-01 00:00:00",
                       taxon_class="mammalia", taxon_genus="sus")
        expected_winner["boar"] = b_old
        det("boar", projects[1], b_new)
        det("boar", projects[2], b_old)

        # 4. "rock": an already-global bare row + a per-project ranked row ->
        #    ranked wins and goes global, the old global row merges in.
        r_global = custom("rock", None, "2026-01-01 00:00:00", project_id=None)
        r_rank = custom("rock", projects[0]["id"], "2026-02-01 00:00:00",
                        taxon_class="aves")
        expected_winner["rock"] = r_rank
        det("rock", projects[0], r_global)    # detection on the old global row
        obs("rock", projects[0], r_rank)

        # 5. "stump": lone per-project label, no duplicate -> project_id cleared.
        s_lone = custom("stump", projects[4]["id"], "2026-01-01 00:00:00")
        expected_winner["stump"] = s_lone
        det("stump", projects[4], s_lone)

        # 6. "ghost": a duplicate pair neither side referenced by anything.
        g1 = custom("ghost", projects[0]["id"], "2026-01-01 00:00:00")
        custom("ghost", projects[1]["id"], "2026-02-01 00:00:00")
        expected_winner["ghost"] = g1

        # 7. Loads of unique labels across projects (no merge, just go global).
        bulk_ids = []
        for i in range(60):
            p = projects[i % 5]
            bulk_ids.append(
                custom(f"bulk_{i:03d}", p["id"], "2026-04-01 00:00:00")
            )

        # 8. Non-custom rows that must be left completely alone.
        insert_row(
            conn, "label_taxonomy", name="leopard", level="species",
            classification_model_id="SOME-MODEL", is_custom=0, project_id=None,
            taxon_class="mammalia",
        )
        insert_row(
            conn, "label_taxonomy", name="animal", level="unknown",
            classification_model_id="__builtin__", is_custom=0, project_id=None,
        )

    upgrade_to("c3d4e5f6a7b9")

    # --- Per-group: the expected row survived, losers gone, links moved ---
    for name, winner in expected_winner.items():
        rows = _rows(
            engine,
            "SELECT id FROM label_taxonomy WHERE lower(name)=:n AND is_custom=1",
            n=name,
        )
        assert [r[0] for r in rows] == [winner], f"{name}: wrong survivor set {rows}"
        for det_id in expected_det.get(name, []):
            got = _scalar(
                engine,
                "SELECT label_taxonomy_id FROM detections WHERE id=:i", i=det_id,
            )
            assert got == winner, f"{name}: detection {det_id} -> {got}, want {winner}"
        for obs_id in expected_obs.get(name, []):
            got = _scalar(
                engine,
                "SELECT label_taxonomy_id FROM event_observations WHERE id=:i",
                i=obs_id,
            )
            assert got == winner, f"{name}: observation {obs_id} -> {got}, want {winner}"

    # --- Global invariants over every custom row ---
    custom_rows = _rows(
        engine,
        "SELECT id, lower(name), project_id FROM label_taxonomy WHERE is_custom=1",
    )
    # Every custom row is global now.
    assert all(r[2] is None for r in custom_rows), "a custom row kept a project_id"
    # No two custom rows share a name (the unique key would forbid it anyway).
    names = [r[1] for r in custom_rows]
    assert len(names) == len(set(names)), "duplicate custom names survived the merge"
    # The lone + bulk rows all survive (unique names, nothing merged away).
    assert "stump" in names
    assert sum(n.startswith("bulk_") for n in names) == 60

    # --- Non-custom rows untouched ---
    assert _scalar(
        engine, "SELECT project_id FROM label_taxonomy WHERE name='leopard'"
    ) is None
    assert _scalar(
        engine,
        "SELECT classification_model_id FROM label_taxonomy WHERE name='leopard'",
    ) == "SOME-MODEL"
    assert _scalar(
        engine,
        "SELECT COUNT(*) FROM label_taxonomy WHERE name='animal' AND is_custom=0",
    ) == 1

    # --- The database is structurally sound ---
    assert _scalar(engine, "PRAGMA integrity_check") == "ok"
    assert _rows(engine, "PRAGMA foreign_key_check") == []


def test_c3d4e5f6a7b9_is_a_noop_when_already_global(engine) -> None:
    """Running it against an all-global custom set changes nothing."""
    upgrade_to("b2c3d4e5f6a8")
    with engine.begin() as conn:
        insert_row(
            conn, "label_taxonomy", name="deer", level="unknown",
            classification_model_id="", is_custom=1, project_id=None,
        )
        insert_row(
            conn, "label_taxonomy", name="fox", level="unknown",
            classification_model_id="", is_custom=1, project_id=None,
        )

    upgrade_to("c3d4e5f6a7b9")

    rows = _rows(
        engine,
        "SELECT lower(name), project_id FROM label_taxonomy WHERE is_custom=1 "
        "ORDER BY name",
    )
    assert [(r[0], r[1]) for r in rows] == [("deer", None), ("fox", None)]
