#!/usr/bin/env python3
"""
Seed a 'Sun-time demo (Oslo)' project so the Activity overlap plot
shows a visible difference between clock-time and sun-time modes.

Usage:
    cd backend
    source venv/bin/activate
    python scripts/seed_sun_time_demo.py

What it creates:
    Project    : "Sun-time demo (Oslo)", timezone Europe/Oslo
    Site       : Nordmarka (59.91N, 10.75E)
    Deployment : 2024-03-01 to 2024-11-30 (9 months, no polar dates)
    Species    : one crepuscular species "Moose (demo)"
    Events     : ~400 observations, each clustered at either
                 (sunrise + 1h) or (sunset - 1h) with +/- 30 min jitter

Because Oslo's sunrise swings from ~03:55 in June to ~08:15 in November,
the clock-mode curve smears across a wide band while the sun-mode curve
sharpens into two tight peaks at the anchor sunrise + 1h and anchor
sunset - 1h. That is the visible demonstration of what the Vazquez
transform does.

Re-running the script is a no-op when the project already exists.
Delete the project via the Projects page when you are done.
"""

from __future__ import annotations

import random
import sys
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from astral import LocationInfo
from astral.sun import sun

from app.db.base import get_session_factory
from app.models.deployment import Deployment
from app.models.event import Event
from app.models.event_observation import EventObservation
from app.models.project import Project
from app.models.site import Site

PROJECT_NAME = "Sun-time demo (Oslo)"
TIMEZONE = "Europe/Oslo"
LAT = 59.91
LON = 10.75
START_DATE = date(2024, 3, 1)
END_DATE = date(2024, 11, 30)
SPECIES_LABEL = "Moose (demo)"
EVENT_COUNT = 400
RANDOM_SEED = 42

# Peak offsets. A biologically crepuscular species active ~1h after first
# light and ~1h before last light, with 30 min of jitter each side.
POST_SUNRISE_OFFSET_H = 1.0
PRE_SUNSET_OFFSET_H = 1.0
JITTER_H = 0.5


def _clock_time_for_date(d: date, rng: random.Random) -> datetime | None:
    """Pick a clock time on date `d` tied to that day's sunrise / sunset.

    Returns `None` on polar dates. Oslo never hits polar across the
    seed date range, so this is defensive only.
    """
    loc = LocationInfo("x", "x", TIMEZONE, LAT, LON)
    try:
        s = sun(loc.observer, date=d, tzinfo=ZoneInfo(TIMEZONE))
    except ValueError:
        return None
    sunrise_h = s["sunrise"].hour + s["sunrise"].minute / 60 + s["sunrise"].second / 3600
    sunset_h = s["sunset"].hour + s["sunset"].minute / 60 + s["sunset"].second / 3600
    if rng.random() < 0.5:
        t_hour = sunrise_h + POST_SUNRISE_OFFSET_H + rng.uniform(-JITTER_H, JITTER_H)
    else:
        t_hour = sunset_h - PRE_SUNSET_OFFSET_H + rng.uniform(-JITTER_H, JITTER_H)
    t_hour = t_hour % 24
    hh = int(t_hour)
    mm = int((t_hour - hh) * 60)
    return datetime(d.year, d.month, d.day, hh, mm)


def main() -> int:
    session_factory = get_session_factory()
    db = session_factory()
    try:
        existing = db.query(Project).filter(Project.name == PROJECT_NAME).first()
        if existing is not None:
            print(f"Project '{PROJECT_NAME}' already exists (id={existing.id}).")
            print("Delete it via the Projects page first if you want to re-seed.")
            return 0

        rng = random.Random(RANDOM_SEED)

        project = Project(
            name=PROJECT_NAME,
            timezone=TIMEZONE,
            description=(
                "Synthetic demo dataset. One crepuscular species active "
                "about 1 hour after sunrise and 1 hour before sunset over "
                "9 months. Clock-time activity overlap shows a smeared "
                "bimodal curve because sunrise drifts by ~4 hours across "
                "the year; sun-time overlap sharpens it into two tight "
                "peaks."
            ),
        )
        db.add(project)
        db.flush()

        site = Site(
            project_id=project.id,
            name="Nordmarka",
            latitude=LAT,
            longitude=LON,
        )
        db.add(site)
        db.flush()

        deployment = Deployment(
            site_id=site.id,
            start_date_local=START_DATE,
            end_date_local=END_DATE,
        )
        db.add(deployment)
        db.flush()

        total_days = (END_DATE - START_DATE).days + 1
        events_inserted = 0
        for _ in range(EVENT_COUNT):
            offset_days = rng.randint(0, total_days - 1)
            d = START_DATE + timedelta(days=offset_days)
            event_time = _clock_time_for_date(d, rng)
            if event_time is None:
                continue
            event = Event(
                deployment_id=deployment.id,
                event_start_local=event_time,
                event_end_local=event_time + timedelta(minutes=1),
                file_count=0,
            )
            db.add(event)
            db.flush()
            observation = EventObservation(
                event_id=event.id,
                label=SPECIES_LABEL,
                label_taxonomy_id=None,
                category="animal",
                max_n=1,
            )
            db.add(observation)
            events_inserted += 1

        db.commit()
        print(f"Seeded '{PROJECT_NAME}'")
        print(f"  project_id = {project.id}")
        print(f"  site_id    = {site.id}")
        print(f"  events     = {events_inserted}")
        print()
        print("Open the project, go to Plots -> Activity overlap, pick")
        print("'Moose (demo)' in Species A, and toggle between Clock and Sun.")
        return 0
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
