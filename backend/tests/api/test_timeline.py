"""Tests for the Insights → Deployment timeline CRUD + endpoint."""

from __future__ import annotations

from datetime import date, datetime

from app.api.crud.timeline import NO_SITE_LABEL, get_deployment_timeline
from app.api.crud.trap_nights import compute_trap_nights_for_deployments
from tests.conftest import make_deployment, make_file, make_project, make_site


def _make_dep_with_files(db, *, site_id, project_id, folder, dates, **kw):
    """Build a deployment with one or more files in the given folder."""
    first_date = min(dates)
    last_date = max(dates)
    dep = make_deployment(
        db,
        site_id=site_id,
        project_id=project_id,
        start_date_local=kw.pop("start_date_local", first_date),
        end_date_local=kw.pop("end_date_local", last_date),
        **kw,
    )
    for d in dates:
        make_file(
            db,
            deployment_id=dep.id,
            file_path=f"{folder}/{d.isoformat()}.jpg",
            captured_at_local=datetime.combine(d, datetime.min.time()),
        )
    return dep


def test_empty_project_returns_zeroed_payload(db):
    project = make_project(db)
    response = get_deployment_timeline(db, project.id)
    assert response.sites == []
    assert response.concurrent_cameras == []
    assert response.metrics.site_count == 0
    assert response.metrics.deployment_count == 0
    assert response.metrics.total_trap_nights == 0
    assert response.metrics.median_deployment_length_days is None
    assert response.metrics.max_concurrent_cameras == 0
    assert response.date_range_from is None
    assert response.date_range_to is None


def test_single_deployment_single_folder(db):
    project = make_project(db)
    site = make_site(db, project_id=project.id, name="Alpha")
    _make_dep_with_files(
        db,
        site_id=site.id,
        project_id=project.id,
        folder="/data/alpha/sd1",
        dates=[date(2024, 1, 1), date(2024, 1, 5), date(2024, 1, 10)],
    )

    response = get_deployment_timeline(db, project.id)

    assert len(response.sites) == 1
    site_row = response.sites[0]
    assert site_row.site_name == "Alpha"
    assert len(site_row.deployments) == 1
    dep_row = site_row.deployments[0]
    assert len(dep_row.intervals) == 1
    iv = dep_row.intervals[0]
    assert iv.start == date(2024, 1, 1)
    assert iv.end == date(2024, 1, 10)
    assert iv.trap_nights == 10
    assert dep_row.file_count == 3
    assert response.metrics.total_trap_nights == 10
    assert response.metrics.max_concurrent_cameras == 1


def test_single_deployment_two_disjoint_folders_renders_two_bars(db):
    project = make_project(db)
    site = make_site(db, project_id=project.id, name="Beta")
    dep = make_deployment(
        db,
        site_id=site.id,
        project_id=project.id,
        start_date_local=date(2012, 1, 1),
        end_date_local=date(2020, 2, 1),
    )
    # Folder A: Jan 2012
    for day in (date(2012, 1, 1), date(2012, 1, 15), date(2012, 2, 1)):
        make_file(
            db,
            deployment_id=dep.id,
            file_path=f"/data/beta/card_a/{day.isoformat()}.jpg",
            captured_at_local=datetime.combine(day, datetime.min.time()),
        )
    # Folder B: Jan 2020
    for day in (date(2020, 1, 1), date(2020, 1, 15), date(2020, 2, 1)):
        make_file(
            db,
            deployment_id=dep.id,
            file_path=f"/data/beta/card_b/{day.isoformat()}.jpg",
            captured_at_local=datetime.combine(day, datetime.min.time()),
        )

    response = get_deployment_timeline(db, project.id)

    dep_row = response.sites[0].deployments[0]
    assert len(dep_row.intervals) == 2
    ivs = sorted(dep_row.intervals, key=lambda iv: iv.start)
    assert ivs[0].start == date(2012, 1, 1)
    assert ivs[0].end == date(2012, 2, 1)
    assert ivs[0].trap_nights == 32
    assert ivs[1].start == date(2020, 1, 1)
    assert ivs[1].end == date(2020, 2, 1)
    assert ivs[1].trap_nights == 32
    assert response.metrics.total_trap_nights == 64


def test_two_deployments_same_site_with_gap(db):
    project = make_project(db)
    site = make_site(db, project_id=project.id, name="Camera Stand")
    _make_dep_with_files(
        db,
        site_id=site.id,
        project_id=project.id,
        folder="/data/stand/y1",
        dates=[date(2023, 3, 1), date(2023, 4, 30)],
    )
    _make_dep_with_files(
        db,
        site_id=site.id,
        project_id=project.id,
        folder="/data/stand/y2",
        dates=[date(2023, 7, 1), date(2023, 8, 31)],
    )

    response = get_deployment_timeline(db, project.id)

    assert len(response.sites) == 1
    assert len(response.sites[0].deployments) == 2
    # No overlap, so concurrent count never goes above 1.
    assert response.metrics.max_concurrent_cameras == 1


def test_two_overlapping_deployments_different_sites_raise_concurrent_count(db):
    project = make_project(db)
    site_a = make_site(db, project_id=project.id, name="A")
    site_b = make_site(db, project_id=project.id, name="B")
    _make_dep_with_files(
        db,
        site_id=site_a.id,
        project_id=project.id,
        folder="/data/a",
        dates=[date(2024, 5, 1), date(2024, 5, 31)],
    )
    _make_dep_with_files(
        db,
        site_id=site_b.id,
        project_id=project.id,
        folder="/data/b",
        dates=[date(2024, 5, 15), date(2024, 6, 15)],
    )

    response = get_deployment_timeline(db, project.id)

    assert response.metrics.max_concurrent_cameras == 2
    # There is at least one point where the count rises to 2 on May 15.
    saw_two = any(p.count == 2 for p in response.concurrent_cameras)
    assert saw_two


def test_site_less_deployment_uses_no_site_label(db):
    project = make_project(db)
    _make_dep_with_files(
        db,
        site_id=None,
        project_id=project.id,
        folder="/data/orphan",
        dates=[date(2024, 2, 1), date(2024, 2, 10)],
    )

    response = get_deployment_timeline(db, project.id)

    assert len(response.sites) == 1
    assert response.sites[0].site_id is None
    assert response.sites[0].site_name == NO_SITE_LABEL
    # Site without a real site_id doesn't count in the site tally.
    assert response.metrics.site_count == 0
    assert response.metrics.deployment_count == 1


def test_no_site_row_is_always_last(db):
    project = make_project(db)
    site = make_site(db, project_id=project.id, name="Zebra")
    _make_dep_with_files(
        db,
        site_id=site.id,
        project_id=project.id,
        folder="/data/zebra",
        dates=[date(2024, 1, 1), date(2024, 1, 10)],
    )
    _make_dep_with_files(
        db,
        site_id=None,
        project_id=project.id,
        folder="/data/orphan",
        dates=[date(2024, 1, 1), date(2024, 1, 10)],
    )

    response = get_deployment_timeline(db, project.id)

    assert len(response.sites) == 2
    assert response.sites[0].site_id == site.id
    assert response.sites[1].site_id is None


def test_clip_window_trims_intervals_and_trap_nights(db):
    project = make_project(db)
    site = make_site(db, project_id=project.id, name="Clip")
    _make_dep_with_files(
        db,
        site_id=site.id,
        project_id=project.id,
        folder="/data/clip",
        dates=[date(2024, 1, 1), date(2024, 3, 31)],
    )

    response = get_deployment_timeline(
        db,
        project.id,
        date_from=date(2024, 2, 1),
        date_to=date(2024, 2, 29),
    )

    dep_row = response.sites[0].deployments[0]
    assert len(dep_row.intervals) == 1
    iv = dep_row.intervals[0]
    assert iv.start == date(2024, 2, 1)
    assert iv.end == date(2024, 2, 29)
    assert iv.trap_nights == 29
    assert response.metrics.total_trap_nights == 29


def test_one_source_of_truth_matches_trap_nights_helper(db):
    """The timeline's total_trap_nights must match the Dashboard helper exactly."""
    project = make_project(db)
    site_a = make_site(db, project_id=project.id, name="A")
    site_b = make_site(db, project_id=project.id, name="B")
    dep1 = _make_dep_with_files(
        db,
        site_id=site_a.id,
        project_id=project.id,
        folder="/data/a/sd1",
        dates=[date(2024, 1, 1), date(2024, 1, 20)],
    )
    dep2 = _make_dep_with_files(
        db,
        site_id=site_a.id,
        project_id=project.id,
        folder="/data/a/sd2",
        dates=[date(2024, 3, 1), date(2024, 3, 31)],
    )
    dep3 = _make_dep_with_files(
        db,
        site_id=site_b.id,
        project_id=project.id,
        folder="/data/b/sd1",
        dates=[date(2024, 2, 15), date(2024, 4, 1)],
    )
    # Also mix in a deployment with two disjoint folders.
    dep4 = make_deployment(
        db,
        site_id=site_b.id,
        project_id=project.id,
        start_date_local=date(2023, 12, 1),
        end_date_local=date(2024, 5, 1),
    )
    for day in (date(2023, 12, 1), date(2023, 12, 15)):
        make_file(
            db,
            deployment_id=dep4.id,
            file_path=f"/data/b/sd_early/{day.isoformat()}.jpg",
            captured_at_local=datetime.combine(day, datetime.min.time()),
        )
    for day in (date(2024, 5, 1), date(2024, 5, 1)):
        make_file(
            db,
            deployment_id=dep4.id,
            file_path=f"/data/b/sd_late/{day.isoformat()}.jpg",
            captured_at_local=datetime.combine(day, datetime.min.time()),
        )

    response = get_deployment_timeline(db, project.id)
    helper = compute_trap_nights_for_deployments(
        db, [dep1.id, dep2.id, dep3.id, dep4.id]
    )

    assert response.metrics.total_trap_nights == sum(helper.values())


def test_rollover_chain_collapses_into_one_interval(db):
    """Reconyx rollover: two subfolders sharing an exact boundary day
    (100MEDIA.end == 101MEDIA.start) are one camera session. The
    timeline response collapses the chain into a single interval so the
    Gantt renders one bar, the tooltip shows the whole-chain dates, and
    the concurrent-cameras sweep does not spike on the shared day."""
    project = make_project(db)
    site = make_site(db, project_id=project.id, name="Rollover")
    dep = make_deployment(
        db,
        site_id=site.id,
        project_id=project.id,
        start_date_local=date(2024, 6, 1),
        end_date_local=date(2024, 6, 30),
    )
    # 100MEDIA Jun 1 .. Jun 15
    for day in (1, 8, 15):
        make_file(
            db,
            deployment_id=dep.id,
            file_path=f"/data/rollover/100MEDIA/img_{day:02d}.jpg",
            captured_at_local=datetime.combine(
                date(2024, 6, day), datetime.min.time()
            ),
        )
    # 101MEDIA Jun 15 .. Jun 30 (shares Jun 15 with 100MEDIA)
    for day in (15, 22, 30):
        make_file(
            db,
            deployment_id=dep.id,
            file_path=f"/data/rollover/101MEDIA/img_{day:02d}.jpg",
            captured_at_local=datetime.combine(
                date(2024, 6, day), datetime.min.time()
            ),
        )

    response = get_deployment_timeline(db, project.id)

    intervals = response.sites[0].deployments[0].intervals
    assert len(intervals) == 1
    assert intervals[0].start == date(2024, 6, 1)
    assert intervals[0].end == date(2024, 6, 30)
    assert intervals[0].trap_nights == 30
    assert response.metrics.max_concurrent_cameras == 1


def test_parallel_subfolders_in_one_deployment_raise_concurrent_count(db):
    """Motivating case for the sum-with-boundary-subtraction algorithm:
    one deployment row actually wraps multiple cameras running in
    parallel. The timeline response should now show the parallel
    subfolders as separate bars, count them in the concurrent-cameras
    sweep, and sum their trap-nights correctly (no merge undercount)."""
    project = make_project(db)
    site = make_site(db, project_id=project.id, name="Backlog")
    dep = make_deployment(
        db,
        site_id=site.id,
        project_id=project.id,
        start_date_local=date(2024, 1, 1),
        end_date_local=date(2024, 1, 10),
    )
    # Five parallel cameras, each spanning Jan 1..Jan 10 (10 days).
    for cam in range(5):
        for day in (1, 5, 10):
            make_file(
                db,
                deployment_id=dep.id,
                file_path=f"/data/backlog/cam_{cam}/img_{day:02d}.jpg",
                captured_at_local=datetime.combine(
                    date(2024, 1, day), datetime.min.time()
                ),
            )

    response = get_deployment_timeline(db, project.id)

    dep_row = response.sites[0].deployments[0]
    # Five subfolders, no boundary matches → five separate intervals.
    assert len(dep_row.intervals) == 5
    # Each interval is Jan 1..Jan 10 = 10 days.
    assert all(iv.trap_nights == 10 for iv in dep_row.intervals)
    # Concurrent count peaks at 5 on the overlap days.
    assert response.metrics.max_concurrent_cameras == 5
    # Total trap-nights = 5 * 10 = 50 (no merging, no boundary match).
    assert response.metrics.total_trap_nights == 50


def test_site_ids_filter_supports_no_site_sentinel(db, client):
    """The endpoint honours the NO_SITE_SENTINEL on the URL."""
    project = make_project(db)
    site = make_site(db, project_id=project.id, name="Alpha")
    _make_dep_with_files(
        db,
        site_id=site.id,
        project_id=project.id,
        folder="/data/alpha",
        dates=[date(2024, 1, 1), date(2024, 1, 10)],
    )
    _make_dep_with_files(
        db,
        site_id=None,
        project_id=project.id,
        folder="/data/orphan",
        dates=[date(2024, 1, 1), date(2024, 1, 10)],
    )

    response = client.get(
        "/api/statistics/timeline",
        params={"project_id": project.id, "site_ids": "null"},
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["sites"]) == 1
    assert body["sites"][0]["site_id"] is None
