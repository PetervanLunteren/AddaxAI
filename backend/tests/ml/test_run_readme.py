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


def test_readme_unknown_project_raises(db, tmp_path):
    with pytest.raises(ValueError, match="not found"):
        write_run_readme(db, "no-such-id", tmp_path / "out", media_threshold=0.5)
