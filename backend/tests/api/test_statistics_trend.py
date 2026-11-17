"""The observation rate over time: the Explore trend and the Trend page.

The Trend page takes the site as the sampling unit. Each site gets its own
rate per 100 trap nights, and a period is the mean of those rates with a
95% confidence interval. January 2024 starts on a Monday, so the ISO
weeks here are 1-7, 8-14, 15-21 and 22-28 January.
"""

import uuid
from datetime import date, datetime

import pytest

from app.api.crud import rate_trend
from app.api.crud.rate_trend import build_periods, period_of, pick_granularity
from app.models.event_observation import EventObservation
from app.models.label_taxonomy import LabelTaxonomy
from tests.conftest import (
    make_deployment,
    make_event_with_files,
    make_file,
    make_project,
    make_site,
)


def _jan(day: int) -> date:
    return date(2024, 1, day)


def _camera(db, *, project, site=None, first: date, last: date):
    """A deployment whose camera ran from `first` to `last`.

    Every file of a deployment made here sits in one folder, so the
    deployment is one camera session from its first to its last file.
    """
    dep = make_deployment(
        db, project_id=project.id, site_id=site.id if site else None,
        start_date_local=first,
    )
    for day in (first, last):
        make_file(
            db, deployment_id=dep.id,
            captured_at_local=datetime(day.year, day.month, day.day, 12),
        )
    return dep


def _taxon(db, name: str) -> LabelTaxonomy:
    row = LabelTaxonomy(
        id=str(uuid.uuid4()), classification_model_id="test-model", name=name,
        level="species", scientific_name=f"{name} sci", common_name=name.capitalize(),
    )
    db.add(row)
    db.flush()
    return row


def _sighting(db, dep, day: date, count: int, *, label="deer", taxon=None):
    event = make_event_with_files(
        db, deployment_id=dep.id,
        event_start_local=datetime(day.year, day.month, day.day, 6),
    )
    db.add(
        EventObservation(
            event_id=event.id, label=label,
            label_taxonomy_id=taxon.id if taxon else None,
            category="animal", max_n=count,
        )
    )
    db.flush()


def _three_sites(db):
    """Three sites, each with one camera for two full weeks (14 nights).

    Week 1: A sees 7 deer, B sees 14. Week 2: A sees 14. C sees nothing.
    """
    project = make_project(db)
    sites = [make_site(db, project_id=project.id, name=n) for n in "ABC"]
    deps = [
        _camera(db, project=project, site=s, first=_jan(1), last=_jan(14))
        for s in sites
    ]
    _sighting(db, deps[0], _jan(3), 7)
    _sighting(db, deps[1], _jan(2), 14)
    _sighting(db, deps[0], _jan(10), 14)
    return project, sites, deps


# ---------------------------------------------------------------------------
# Periods
# ---------------------------------------------------------------------------


def test_a_week_runs_monday_to_sunday_and_a_month_by_the_calendar():
    assert period_of(_jan(10), "day") == ("2024-01-10", _jan(10), _jan(10))
    assert period_of(_jan(10), "week") == ("2024-W02", _jan(8), _jan(14))
    assert period_of(date(2024, 2, 10), "month") == (
        "2024-02", date(2024, 2, 1), date(2024, 2, 29),
    )


def test_periods_cross_a_year_without_gaps():
    """30 December 2024 is a Monday in ISO week 1 of 2025."""
    weeks = build_periods(date(2024, 12, 20), date(2025, 1, 8), "week")
    months = build_periods(date(2024, 11, 30), date(2025, 2, 1), "month")

    assert [p.key for p in weeks] == ["2024-W51", "2024-W52", "2025-W01", "2025-W02"]
    assert weeks[2].start == date(2024, 12, 30)
    assert [p.key for p in months] == ["2024-11", "2024-12", "2025-01", "2025-02"]
    for periods in (weeks, months):
        for before, after in zip(periods, periods[1:], strict=False):
            assert (after.start - before.end).days == 1


def test_the_span_picks_the_period_size():
    def pick(days: int, *, allow_day: bool):
        last = date.fromordinal(_jan(1).toordinal() + days - 1)
        return pick_granularity(_jan(1), last, allow_day=allow_day)

    assert [pick(d, allow_day=True) for d in (1, 90, 91, 365, 366)] == [
        "day", "day", "week", "week", "month",
    ]
    # The Trend page never draws days: across sites they are mostly zeros.
    assert [pick(d, allow_day=False) for d in (1, 90, 365, 366)] == [
        "week", "week", "week", "month",
    ]


# ---------------------------------------------------------------------------
# Trend page: the mean of site rates per period
# ---------------------------------------------------------------------------


def test_a_week_is_the_mean_of_its_site_rates(db):
    project, *_ = _three_sites(db)

    response = rate_trend.get_rate_trend(db, project.id)

    assert response.granularity == "week"
    assert (response.first_date, response.last_date) == (_jan(1), _jan(14))
    week_1, week_2 = response.periods
    assert (week_1.period, week_1.start, week_1.end) == ("2024-W01", _jan(1), _jan(7))
    # Per 100 trap nights: A 7/7, B 14/7, C 0/7 -> 100, 200, 0.
    assert week_1.n_sites == 3
    assert week_1.mean == pytest.approx(100.0)
    assert week_1.sd == pytest.approx(100.0)
    # 4.303 * 100 / sqrt(3) = 248.43, more than the mean.
    assert week_1.ci_low == 0.0
    assert week_1.ci_high == pytest.approx(348.4338, abs=1e-3)
    # A 14/7, B 0, C 0 -> 200, 0, 0.
    assert week_2.mean == pytest.approx(200 / 3)
    assert week_2.sd == pytest.approx(115.4701, abs=1e-3)


def test_a_site_that_saw_nothing_pulls_the_mean_down(db):
    """Leaving out sites with effort and no sightings would put week 1 at
    150 (the mean of A and B) instead of 100."""
    project, sites, _ = _three_sites(db)

    with_c = rate_trend.get_rate_trend(db, project.id).periods[0]
    without_c = rate_trend.get_rate_trend(
        db, project.id, site_ids=[sites[0].id, sites[1].id]
    ).periods[0]

    assert (with_c.n_sites, with_c.mean) == (3, pytest.approx(100.0))
    assert (without_c.n_sites, without_c.mean) == (2, pytest.approx(150.0))


@pytest.mark.parametrize(("last_day", "counts"), [(12, False), (13, True)])
def test_a_site_counts_in_a_week_from_six_active_days(db, last_day, counts):
    """80% of a week is 5.6 days: 5 active days (8-12 January) is too
    little, 6 (8-13 January) is enough."""
    project, *_ = _three_sites(db)
    late = make_site(db, project_id=project.id, name="D")
    dep = _camera(db, project=project, site=late, first=_jan(8), last=_jan(last_day))
    _sighting(db, dep, _jan(9), 60)

    week_1, week_2 = rate_trend.get_rate_trend(db, project.id).periods

    assert week_1.n_sites == 3
    assert week_2.n_sites == (4 if counts else 3)


def test_two_cameras_at_one_site_are_one_site_with_double_effort(db):
    project, sites, _ = _three_sites(db)
    second = _camera(db, project=project, site=sites[0], first=_jan(1), last=_jan(14))
    _sighting(db, second, _jan(4), 7)

    week_1 = rate_trend.get_rate_trend(db, project.id).periods[0]

    # A: 14 observations over 14 trap nights is still 100 per 100.
    assert week_1.n_sites == 3
    assert week_1.mean == pytest.approx(100.0)


def test_a_week_nobody_ran_long_enough_is_a_gap_not_a_zero(db):
    project = make_project(db)
    site = make_site(db, project_id=project.id)
    first = _camera(db, project=project, site=site, first=_jan(1), last=_jan(7))
    _camera(db, project=project, site=site, first=_jan(22), last=_jan(28))
    _sighting(db, first, _jan(2), 7)

    periods = rate_trend.get_rate_trend(db, project.id).periods

    assert [p.period for p in periods] == ["2024-W01", "2024-W02", "2024-W03", "2024-W04"]
    assert [p.n_sites for p in periods] == [1, 0, 0, 1]
    assert [p.mean for p in periods] == [pytest.approx(100.0), None, None, 0.0]
    # One site has a mean and nothing to measure a spread with.
    assert (periods[0].sd, periods[0].ci_low, periods[0].ci_high) == (None, None, None)


def test_a_week_cut_by_the_date_filter_is_a_gap(db):
    """From Thursday 4 January only 4 days of week 1 are in view. Every
    camera ran all 4, but 4 of 7 is under 80%: a piece of a week is too
    thin to be a point. The period still says which days were in view."""
    project, *_ = _three_sites(db)

    week_1, week_2 = rate_trend.get_rate_trend(
        db, project.id, date_from="2024-01-04", date_to="2024-01-14"
    ).periods

    assert (week_1.start, week_1.end, week_1.n_sites, week_1.mean) == (
        _jan(4), _jan(7), 0, None,
    )
    assert (week_2.n_sites, week_2.mean) == (3, pytest.approx(200 / 3))


def test_a_single_day_at_the_end_of_the_range_is_no_spike(db):
    """The bug hunt's case: a range ending on a Monday left week 3 with one
    day, and every site counted on a single trap night."""
    project, sites, _ = _three_sites(db)
    week_3 = [
        _camera(db, project=project, site=site, first=_jan(15), last=_jan(21))
        for site in sites
    ]
    _sighting(db, week_3[0], _jan(15), 1)

    periods = rate_trend.get_rate_trend(
        db, project.id, date_from="2024-01-01", date_to="2024-01-15"
    ).periods

    assert [p.period for p in periods] == ["2024-W01", "2024-W02", "2024-W03"]
    assert [p.n_sites for p in periods] == [3, 3, 0]
    assert periods[2].mean is None


def test_the_date_filter_narrows_the_periods(db):
    project, *_ = _three_sites(db)

    response = rate_trend.get_rate_trend(
        db, project.id, date_from="2024-01-08", date_to="2024-01-14"
    )

    assert [p.period for p in response.periods] == ["2024-W02"]
    assert response.periods[0].mean == pytest.approx(200 / 3)


def test_a_deployment_without_a_site_is_no_sampling_unit(db):
    project, *_ = _three_sites(db)
    loose = _camera(db, project=project, first=_jan(1), last=_jan(14))
    _sighting(db, loose, _jan(5), 700)

    response = rate_trend.get_rate_trend(db, project.id)

    assert response.periods[0].n_sites == 3
    assert response.periods[0].mean == pytest.approx(100.0)
    assert [s.mean for s in response.species] == [pytest.approx(250 / 3)]


def test_months_are_drawn_on_request(db):
    project, *_ = _three_sites(db)

    response = rate_trend.get_rate_trend(db, project.id, granularity="month")

    assert response.granularity == "month"
    # 14 of January's 31 days is under 80%: no site counts.
    assert [(p.period, p.n_sites, p.mean) for p in response.periods] == [
        ("2024-01", 0, None)
    ]


# ---------------------------------------------------------------------------
# Trend page: the species table
# ---------------------------------------------------------------------------


def test_the_species_table_is_the_mean_over_every_site_with_effort(db):
    project, *_ = _three_sites(db)

    (deer,) = rate_trend.get_rate_trend(db, project.id).species

    # Whole range, 14 nights each: A 21, B 14, C 0 -> 150, 100, 0.
    assert (deer.label, deer.n_sites) == ("deer", 3)
    assert deer.mean == pytest.approx(250 / 3)
    assert deer.sd == pytest.approx(76.3763, abs=1e-3)
    assert deer.ci_low == 0.0
    assert deer.ci_high == pytest.approx(250 / 3 + 4.303 * 76.3763 / 3**0.5, abs=1e-2)


def test_the_species_table_counts_a_site_with_a_single_night(db):
    """The chart's 80% rule does not apply to the table."""
    project, *_ = _three_sites(db)
    brief = make_site(db, project_id=project.id, name="D")
    _camera(db, project=project, site=brief, first=_jan(8), last=_jan(8))

    response = rate_trend.get_rate_trend(db, project.id)

    assert response.species[0].n_sites == 4
    assert response.species[0].mean == pytest.approx(250 / 4)
    assert response.periods[1].n_sites == 3


def test_observations_without_a_label_get_a_row_per_category(db):
    """A detector-only project has no labels. Person, vehicle and animal
    used to land in one "unknown" row with their counts added up."""
    project, _, deps = _three_sites(db)
    for category, count in (("person", 7), ("vehicle", 14)):
        event = make_event_with_files(
            db, deployment_id=deps[2].id, event_start_local=datetime(2024, 1, 5, 6)
        )
        db.add(
            EventObservation(
                event_id=event.id, label=None, label_taxonomy_id=None,
                category=category, max_n=count,
            )
        )
    db.flush()

    rows = {s.label: s for s in rate_trend.get_rate_trend(db, project.id).species}

    assert set(rows) == {"deer", "person", "vehicle"}
    # Only C, over 14 nights: person 50, vehicle 100 per 100 -> mean of 3 sites.
    assert rows["person"].mean == pytest.approx(50 / 3)
    assert rows["vehicle"].mean == pytest.approx(100 / 3)
    assert (rows["person"].common_name, rows["person"].label_taxonomy_id) == ("person", None)


def test_the_species_table_has_a_row_per_label_highest_first(db):
    project, _, deps = _three_sites(db)
    pig = _taxon(db, "pig")
    _sighting(db, deps[2], _jan(6), 70, label="pig", taxon=pig)

    species = rate_trend.get_rate_trend(db, project.id).species

    assert [(s.label, s.common_name, s.scientific_name) for s in species] == [
        ("pig", "Pig", "pig sci"),
        ("deer", "deer", "deer"),
    ]
    # Pig: only C, 70 over 14 nights -> 0, 0, 500.
    assert species[0].mean == pytest.approx(500 / 3)
    assert species[0].label_taxonomy_id == pig.id


def test_picked_labels_narrow_the_chart_and_the_table(db):
    project, _, deps = _three_sites(db)
    pig = _taxon(db, "pig")
    _sighting(db, deps[2], _jan(6), 70, label="pig", taxon=pig)

    response = rate_trend.get_rate_trend(db, project.id, label_taxonomy_ids=[pig.id])

    assert [s.label for s in response.species] == ["pig"]
    # Week 1: A 0, B 0, C 70/7 -> 0, 0, 1000.
    assert response.periods[0].mean == pytest.approx(1000 / 3)
    assert response.periods[0].n_sites == 3


def test_a_project_without_effort_has_nothing_to_draw(db):
    project = make_project(db)
    make_site(db, project_id=project.id)

    response = rate_trend.get_rate_trend(db, project.id)

    assert (response.first_date, response.last_date) == (None, None)
    assert (response.periods, response.species) == ([], [])


# ---------------------------------------------------------------------------
# Explore: observations per period
# ---------------------------------------------------------------------------


def test_the_explore_trend_counts_observations_per_period(db):
    """A count adds up over a longer period: one deer a day is 7 a week.

    The card used to divide every period by the trap nights of the whole
    scope, so its "rate" changed with the period size. A count does not
    pretend to be a rate; the Trend page has the rate.
    """
    project = make_project(db)
    site = make_site(db, project_id=project.id)
    dep = _camera(db, project=project, site=site, first=_jan(1), last=_jan(28))
    for day in range(1, 29):
        _sighting(db, dep, _jan(day), 1)

    for granularity, per_period in (("day", [1] * 28), ("week", [7] * 4), ("month", [28])):
        response = rate_trend.get_detection_trend(db, project.id, granularity=granularity)

        assert response.granularity == granularity
        assert [p.count for p in response.points] == per_period
    week_2 = rate_trend.get_detection_trend(db, project.id, granularity="week").points[1]
    assert (week_2.period, week_2.start, week_2.end) == ("2024-W02", _jan(8), _jan(14))


def test_the_explore_trend_runs_between_its_first_and_last_observation(db):
    """Without a date filter the periods run from the first to the last day
    with an observation, and a day in between without one is 0."""
    project, *_ = _three_sites(db)

    response = rate_trend.get_detection_trend(db, project.id)

    assert response.granularity == "day"
    assert (response.first_date, response.last_date) == (_jan(2), _jan(10))
    assert [p.count for p in response.points] == [14, 7, 0, 0, 0, 0, 0, 0, 14]


def test_the_date_filter_bounds_the_explore_trend(db):
    project, *_ = _three_sites(db)

    response = rate_trend.get_detection_trend(
        db, project.id, date_from="2024-01-01", date_to="2024-01-14", granularity="week"
    )

    assert [(p.period, p.count) for p in response.points] == [
        ("2024-W01", 21), ("2024-W02", 14),
    ]


def test_the_explore_trend_counts_deployments_without_a_site(db):
    """Like the Explore tiles, which count every deployment in scope."""
    project, *_ = _three_sites(db)
    loose = _camera(db, project=project, first=_jan(1), last=_jan(14))
    _sighting(db, loose, _jan(5), 5)

    week_1, week_2 = rate_trend.get_detection_trend(
        db, project.id, granularity="week"
    ).points

    assert (week_1.count, week_2.count) == (26, 14)


def test_the_explore_trend_reads_no_file_rows(db, monkeypatch):
    """The dashboard has to be quick: effort means every file row of the
    scope, and the count does not need it."""
    project, *_ = _three_sites(db)

    def refuse(*_args, **_kwargs):
        raise AssertionError("the Explore trend must not load effort")

    monkeypatch.setattr(rate_trend, "compute_intervals_for_deployments", refuse)

    assert rate_trend.get_detection_trend(db, project.id).points


def test_a_corrupt_capture_year_is_reported_not_walked(db):
    """Year 25 stretches the project over two millennia. Both charts get
    the two dates to name in their notice, and no periods."""
    project, _, deps = _three_sites(db)
    # A sighting on the corrupt date also puts a file there.
    _sighting(db, deps[0], date(25, 6, 1), 1)

    explore = rate_trend.get_detection_trend(db, project.id)
    per_site = rate_trend.get_rate_trend(db, project.id)

    assert (explore.first_date, explore.last_date) == (date(25, 6, 1), _jan(10))
    assert (per_site.first_date, per_site.last_date) == (date(25, 6, 1), _jan(14))
    assert explore.points == []
    assert (per_site.periods, per_site.species) == ([], [])


# ---------------------------------------------------------------------------
# The endpoints
# ---------------------------------------------------------------------------


def test_the_rate_trend_endpoint_returns_the_chart_and_the_table(client, db):
    project, sites, _ = _three_sites(db)

    response = client.get(
        "/api/statistics/rate-trend",
        params={
            "project_id": project.id,
            "site_ids": f"{sites[0].id},{sites[1].id}",
            "granularity": "week",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["granularity"] == "week"
    assert [p["n_sites"] for p in body["periods"]] == [2, 2]
    assert body["periods"][0]["mean"] == pytest.approx(150.0)
    assert body["species"][0]["label"] == "deer"


def test_the_trend_page_refuses_days_and_the_explore_card_takes_them(client, db):
    project, *_ = _three_sites(db)
    params = {"project_id": project.id, "granularity": "day"}

    assert client.get("/api/statistics/rate-trend", params=params).status_code == 422
    explore = client.get("/api/statistics/detection-trend", params=params)

    assert explore.status_code == 200
    assert len(explore.json()["points"]) == 9


def test_an_empty_label_list_and_a_bad_date_are_refused(client, db):
    project, *_ = _three_sites(db)

    for extra in ({"label_taxonomy_ids": ""}, {"date_from": "banana"}):
        response = client.get(
            "/api/statistics/rate-trend", params={"project_id": project.id, **extra}
        )
        assert response.status_code == 422, extra
