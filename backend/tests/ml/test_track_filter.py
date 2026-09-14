"""The post-filter's values come from the detector's domain, and a
detector without a domain is refused rather than filtered blindly."""

import pytest

from app.ml.schemas.model_manifest import ModelManifest
from app.ml.track_filter import TRACK_FILTER_BY_DOMAIN, TrackFilter, track_filter_for


def _manifest(**overrides):
    fields = dict(
        model_id="X", friendly_name="X", env="marine", model_fname="x.pt",
        description="x", developer="x", info_url="https://example.org",
        min_app_version="7.8.0",
    )
    fields.update(overrides)
    return ModelManifest(**fields)


def test_each_domain_has_its_measured_values():
    """Camera traps: the benchmark's knee (most of the gain, a still
    animal rarely lost). Underwater: SharkTrack's published rule."""
    assert TRACK_FILTER_BY_DOMAIN["camera_trap"] == TrackFilter(min_motion=0.06, exempt_conf=0.5)
    assert TRACK_FILTER_BY_DOMAIN["underwater"] == TrackFilter(min_motion=0.08, exempt_conf=0.7)


def test_the_filter_follows_the_detectors_domain():
    assert track_filter_for(_manifest(domain="camera_trap")).exempt_conf == 0.5
    assert track_filter_for(_manifest(domain="underwater")).exempt_conf == 0.7


def test_a_detector_without_a_domain_is_refused_by_name():
    with pytest.raises(ValueError, match="X declares domain None"):
        track_filter_for(_manifest())


def test_an_unknown_domain_is_refused_by_name_not_dropped_silently():
    """The schema accepts any string so a typo in the catalog does not make
    the model vanish from every picker; the run refuses it instead."""
    assert _manifest(domain="Camera_trap").domain == "Camera_trap"
    with pytest.raises(ValueError, match="X declares domain 'Camera_trap'"):
        track_filter_for(_manifest(domain="Camera_trap"))
