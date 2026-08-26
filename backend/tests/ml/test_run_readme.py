"""Tests for the always-on run README writer.

The README is informational, not load-bearing — these tests pin
that the file lands at the expected path, contains the key sections,
and surfaces enough metadata that a user revisiting the folder weeks
later can reconstruct what produced the deliverables.
"""

from pathlib import Path

import pytest

from app.ml.postprocessing_outputs.run_readme import (
    SUMMARY_FILENAME,
    write_run_readme,
)
from tests.conftest import (
    make_deployment,
    make_detection,
    make_file,
    make_project,
)


def _placeholder(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return str(path)


def test_readme_lands_at_canonical_path(db, tmp_path):
    project = make_project(db, name="readme-basic", timezone="UTC")
    make_deployment(
        db, project_id=project.id, folder_path=str(tmp_path / "src")
    )

    target = tmp_path / "out"
    result = write_run_readme(db, project.id, target, media_threshold=0.5)

    assert result.output_path.endswith(SUMMARY_FILENAME)
    path = target / SUMMARY_FILENAME
    assert path.is_file()
    assert result.bytes_written == path.stat().st_size


def test_readme_summarises_geofence_without_dumping_the_list(db, tmp_path):
    """A big geofence exclusion list is summarised to a count, not dumped
    in full — the line that used to make the README ~40 KB."""
    excluded = [f"species_{i}" for i in range(1500)]
    project = make_project(
        db, name="readme-geo", timezone="UTC", excluded_classes=excluded
    )
    make_deployment(
        db, project_id=project.id, folder_path=str(tmp_path / "src")
    )

    target = tmp_path / "out"
    write_run_readme(db, project.id, target, media_threshold=0.5)
    text = (target / SUMMARY_FILENAME).read_text()

    assert "1,500 excluded" in text
    # The full list is NOT dumped.
    assert "species_999" not in text
    assert (target / SUMMARY_FILENAME).stat().st_size < 5000


def test_readme_carries_run_metadata(db, tmp_path):
    project = make_project(
        db,
        name="readme-meta",
        timezone="Europe/Amsterdam",
        counting_threshold=0.5,
        country_code="NLD",
    )
    make_deployment(
        db,
        project_id=project.id,
        folder_path=str(tmp_path / "src" / "Kruger"),
    )

    target = tmp_path / "out"
    write_run_readme(db, project.id, target, media_threshold=0.5)

    text = (target / SUMMARY_FILENAME).read_text("utf-8")

    # Header carries the project name.
    assert "readme-meta" in text
    # AddaxAI version is surfaced — read from the canonical exporter.
    from app import __version__

    assert __version__ in text
    # Source folder, timezone, country code all appear.
    assert "Kruger" in text
    assert "Europe/Amsterdam" in text
    assert "NLD" in text
    # Project id appears so the readme can be cross-referenced with logs.
    assert project.id in text


def test_readme_top_species_matches_the_summary_table(db, tmp_path):
    """The readme's top species are the Summary sheet's rows, so the two
    outputs in one folder cannot disagree: a video's off-frame boxes and a
    box a person marked false count in neither."""
    project = make_project(db, name="readme-species-summary", counting_threshold=0.5)
    dep = make_deployment(db, project_id=project.id)
    img = make_file(
        db,
        deployment_id=dep.id,
        file_path=_placeholder(tmp_path / "src" / "IMG.jpg"),
        observation_type="animal",
    )
    make_detection(db, file_id=img.id, confidence=0.9, label="deer")
    make_detection(db, file_id=img.id, confidence=0.9, label="false detection",
                   verified=True, classification_method="human")
    clip = make_file(
        db,
        deployment_id=dep.id,
        file_path=_placeholder(tmp_path / "src" / "clip.mp4"),
        file_type="video",
        file_format="mp4",
        best_frame_number=3,
        observation_type="animal",
    )
    make_detection(db, file_id=clip.id, confidence=0.9, label="deer", frame_number=7)
    make_detection(db, file_id=clip.id, confidence=0.9, label="fox", frame_number=3)

    target = tmp_path / "out"
    write_run_readme(db, project.id, target, media_threshold=0.5)
    text = (target / SUMMARY_FILENAME).read_text("utf-8")
    section = text[text.find("Top species"):].split("\n\n")[0]

    assert "false detection" not in section
    assert "deer" in section and "fox" in section
    # One deer on the image; the clip's deer is on a frame nobody can open.
    deer_line = next(line for line in section.splitlines() if "deer" in line)
    assert deer_line.split()[-1] == "1"


def test_readme_lists_top_species(db, tmp_path):
    project = make_project(
        db, name="readme-species", counting_threshold=0.5
    )
    dep = make_deployment(db, project_id=project.id)
    for n, label in enumerate(["dog", "wolf", "cat"]):
        file = make_file(
            db,
            deployment_id=dep.id,
            file_path=_placeholder(tmp_path / "src" / f"IMG_{n}.jpg"),
            observation_type="animal",
        )
        # More detections of "dog" than the others so we can pin
        # the ordering.
        for _ in range(3 if label == "dog" else 1):
            make_detection(
                db,
                file_id=file.id,
                category="animal",
                confidence=0.9,
                label=label,
            )

    target = tmp_path / "out"
    write_run_readme(db, project.id, target, media_threshold=0.5)
    text = (target / SUMMARY_FILENAME).read_text("utf-8")

    # Slice out the Top species section so the substring search
    # doesn't false-match "cat" inside "Classification".
    species_section_idx = text.find("Top species")
    assert species_section_idx != -1
    species_section = text[species_section_idx:]

    # All three labels appear inside the section, and dog leads
    # because it has the most detections.
    dog_pos = species_section.find("dog")
    wolf_pos = species_section.find("wolf")
    cat_pos = species_section.find("cat")
    assert dog_pos != -1 and wolf_pos != -1 and cat_pos != -1
    assert dog_pos < wolf_pos and dog_pos < cat_pos


def test_readme_includes_settings_block(db, tmp_path):
    project = make_project(
        db,
        name="readme-settings",
        smoothing_strength="aggressive",
        taxonomic_rollup=False,
        independence_interval=900,
        video_fps=2.0,
    )
    make_deployment(db, project_id=project.id, folder_path=str(tmp_path))

    target = tmp_path / "out"
    # The reported confidence is the Save step's media confidence, not
    # a project setting; data exports are reported as complete.
    write_run_readme(db, project.id, target, media_threshold=0.42)
    text = (target / SUMMARY_FILENAME).read_text("utf-8")

    # Confidence values are rendered as whole percentages for humans.
    assert "42%" in text
    assert "Media output threshold" in text
    assert "complete, no confidence filter" in text
    assert "aggressive" in text
    # taxonomic_rollup False surfaces somehow — exact string varies
    # by Python repr but "False" is universally present.
    assert "False" in text
    assert "900" in text
    assert "2.0" in text


def test_readme_names_the_media_filter_when_it_excluded_something(db, tmp_path):
    """A filtered-out kind is counted from the database, so it reads "Videos
    0" — identical to a folder that never held any. This line is the only
    thing telling those two apart; without it the file quietly implies the
    folder had no videos."""
    project = make_project(
        db, name="readme-filter", timezone="UTC", media_filter="images"
    )
    make_deployment(
        db, project_id=project.id, folder_path=str(tmp_path / "src")
    )

    target = tmp_path / "out"
    write_run_readme(db, project.id, target, media_threshold=0.5)
    text = (target / SUMMARY_FILENAME).read_text()

    assert "Media filter" in text
    assert "only images" in text


def test_readme_omits_the_media_filter_by_default(db, tmp_path):
    """Nothing was excluded, so the line would be noise on every run."""
    project = make_project(db, name="readme-nofilter", timezone="UTC")
    make_deployment(
        db, project_id=project.id, folder_path=str(tmp_path / "src")
    )

    target = tmp_path / "out"
    write_run_readme(db, project.id, target, media_threshold=0.5)

    assert "Media filter" not in (target / SUMMARY_FILENAME).read_text()


def test_readme_unknown_project_raises(db, tmp_path):
    with pytest.raises(ValueError, match="not found"):
        write_run_readme(db, "no-such-id", tmp_path / "out", media_threshold=0.5)


def test_readme_lists_files_that_could_not_be_read(db, tmp_path):
    """A file the detector could not open has no File row, so it is in no
    table and in no detection list. Without this section the run details
    describe a smaller folder than the one the user pointed at, and nothing
    on disk says which files went missing."""
    project = make_project(db, name="readme-skipped", timezone="UTC")
    dep = make_deployment(
        db, project_id=project.id, folder_path=str(tmp_path / "src")
    )
    dep.warnings = [
        {
            "type": "video_processing_failure",
            "path": "corrupt-video/broken.mp4",
            "reason": "Error: found no frames in file",
        },
        # A dateless file was read perfectly well and IS in the data, so it
        # must not be listed as skipped.
        {"type": "missing_timestamp", "path": "/abs/undated.mp4"},
    ]
    db.commit()

    target = tmp_path / "out"
    write_run_readme(db, project.id, target, media_threshold=0.5)
    text = (target / SUMMARY_FILENAME).read_text()

    assert "Files skipped (unreadable)" in text
    assert "corrupt-video/broken.mp4" in text
    assert "found no frames" in text
    assert "undated.mp4" not in text


def test_readme_has_no_skipped_section_when_nothing_was_skipped(db, tmp_path):
    project = make_project(db, name="readme-clean", timezone="UTC")
    make_deployment(
        db, project_id=project.id, folder_path=str(tmp_path / "src")
    )

    target = tmp_path / "out"
    write_run_readme(db, project.id, target, media_threshold=0.5)
    text = (target / SUMMARY_FILENAME).read_text()

    assert "Files skipped" not in text
    assert "could not be read" not in text
