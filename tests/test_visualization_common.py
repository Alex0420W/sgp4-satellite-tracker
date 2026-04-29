"""Tests for ``sat_tracker.visualization.common`` — pure math, no plotting libs.

This file deliberately does not import ``cartopy`` or ``plotly``; it exercises
the orbit-math helpers (time step, window length, precompute, antimeridian
split) that both renderers depend on.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from skyfield.api import load

from sat_tracker.config import Config
from sat_tracker.coordinates import CoordinateConverter, GroundPosition
from sat_tracker.tle_fetcher import Tle
from sat_tracker.visualization.common import (
    Track,
    default_time_step_seconds,
    default_window_seconds,
    precompute_track,
    split_at_antimeridian,
)


# Fixture TLE epoch is 2024-01-01 day 001.54791667 — propagation starts here.
_NEAR_EPOCH_UTC = datetime(2024, 1, 1, 14, 0, 0, tzinfo=timezone.utc)

# Reused valid line1 from the ISS fixture so synthetic TLEs satisfy the
# 69-char length and prefix checks. (The Tle dataclass doesn't re-validate
# at construction; that's the fetcher's job. We only need parseable mean
# motion in line2 for the helpers under test.)
_VALID_LINE1 = (
    "1 25544U 98067A   24001.54791667  .00016717  00000-0  30571-3 0  9993"
)


def _line2_with_mean_motion(mm_field: str) -> str:
    """Build a valid-length line2 with the given 11-char mean-motion field.

    ``mm_field`` must be exactly 11 chars and parse to a float when read
    via ``float(line2[52:63])``.
    """
    assert len(mm_field) == 11, f"mm_field length {len(mm_field)}"
    prefix = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 "
    rev_and_checksum = "000010"  # 6 chars: 5 rev count + 1 checksum digit
    line = prefix + mm_field + rev_and_checksum
    assert len(line) == 69, f"synthetic line2 length {len(line)}"
    return line


@pytest.fixture(scope="module")
def bundled_timescale():
    return load.timescale(builtin=True)


@pytest.fixture
def config(tmp_path: Path) -> Config:
    return Config(
        cache_dir=tmp_path,
        cache_ttl_hours=6,
        tle_source_url="http://test.invalid/",
        user_agent="sat-tracker-tests/1.0",
        http_timeout_seconds=5,
        log_level="DEBUG",
        min_elevation_deg=10.0,
    )


@pytest.fixture
def converter(config: Config, bundled_timescale) -> CoordinateConverter:
    return CoordinateConverter(
        config, timescale=bundled_timescale, eop_degraded=True
    )


@pytest.fixture
def iss(iss_tle: tuple[str, str, str]) -> Tle:
    name, line1, line2 = iss_tle
    return Tle(name=name, line1=line1, line2=line2)


def _make_ground_position(
    *, lat: float, lon: float, alt: float = 400.0
) -> GroundPosition:
    return GroundPosition(
        time_utc=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        latitude_deg=lat,
        longitude_deg=lon,
        altitude_km=alt,
        eop_degraded=False,
    )


# -- step / window from mean motion ------------------------------------------


def test_default_time_step_seconds_iss(iss: Tle) -> None:
    step = default_time_step_seconds(iss)
    # ISS mean motion ~15.5 rev/day → period ~5571 s → step ~31 s.
    assert 25.0 < step < 40.0


def test_default_time_step_seconds_geo() -> None:
    geo = Tle(
        name="GEO-TEST",
        line1=_VALID_LINE1,
        line2=_line2_with_mean_motion(" 1.00000000"),
    )
    step = default_time_step_seconds(geo)
    # 86400 s / 1.0 rev/day / 180 = 480 s
    assert 470.0 < step < 490.0


def test_default_time_step_seconds_clamped_to_max() -> None:
    """Pathologically slow orbit: step would be 4800 s without the clamp."""
    weird = Tle(
        name="WEIRD",
        line1=_VALID_LINE1,
        line2=_line2_with_mean_motion(" 0.10000000"),
    )
    assert default_time_step_seconds(weird) == 900.0


def test_default_time_step_seconds_rejects_zero_mean_motion() -> None:
    bad = Tle(
        name="BAD",
        line1=_VALID_LINE1,
        line2=_line2_with_mean_motion(" 0.00000000"),
    )
    with pytest.raises(ValueError, match="mean motion"):
        default_time_step_seconds(bad)


def test_default_window_seconds_iss(iss: Tle) -> None:
    window = default_window_seconds(iss)
    # ISS period ~5571 s
    assert 5400 < window < 5800


# -- antimeridian split ------------------------------------------------------


def test_split_at_antimeridian_no_crossing() -> None:
    samples = [
        _make_ground_position(lat=0.0, lon=lon)
        for lon in (-90, -45, 0, 45, 90)
    ]
    segments = split_at_antimeridian(samples)
    assert len(segments) == 1
    assert len(segments[0]) == 5


def test_split_at_antimeridian_with_crossing() -> None:
    """+179 -> -179 produces |Δ|=358 > 180, must split into two segments."""
    samples = [
        _make_ground_position(lat=0.0, lon=160.0),
        _make_ground_position(lat=0.0, lon=170.0),
        _make_ground_position(lat=0.0, lon=179.0),
        _make_ground_position(lat=0.0, lon=-179.0),
        _make_ground_position(lat=0.0, lon=-170.0),
    ]
    segments = split_at_antimeridian(samples)
    assert len(segments) == 2
    assert [s.longitude_deg for s in segments[0]] == [160.0, 170.0, 179.0]
    assert [s.longitude_deg for s in segments[1]] == [-179.0, -170.0]


def test_split_at_antimeridian_multiple_crossings() -> None:
    """Track that wraps the world more than once produces N+1 segments."""
    samples = [
        _make_ground_position(lat=0.0, lon=170.0),
        _make_ground_position(lat=0.0, lon=-170.0),  # wrap 1
        _make_ground_position(lat=0.0, lon=-100.0),
        _make_ground_position(lat=0.0, lon=170.0),   # wrap 2 (Δ = -270)
        _make_ground_position(lat=0.0, lon=-170.0),  # wrap 3
    ]
    segments = split_at_antimeridian(samples)
    assert len(segments) == 4


def test_split_at_antimeridian_empty() -> None:
    assert split_at_antimeridian([]) == []


def test_split_at_antimeridian_single() -> None:
    pt = _make_ground_position(lat=0.0, lon=0.0)
    assert split_at_antimeridian([pt]) == [[pt]]


def test_split_at_antimeridian_does_not_split_at_exactly_180() -> None:
    """|Δ| == 180 exactly is NOT > 180; must remain one segment."""
    samples = [
        _make_ground_position(lat=0.0, lon=90.0),
        _make_ground_position(lat=0.0, lon=-90.0),
    ]
    segments = split_at_antimeridian(samples)
    assert len(segments) == 1


# -- precompute_track --------------------------------------------------------


def test_precompute_track_iss_one_period(
    iss: Tle, converter: CoordinateConverter
) -> None:
    track = precompute_track(
        iss,
        converter,
        start_utc=_NEAR_EPOCH_UTC,
        duration_seconds=default_window_seconds(iss),
    )
    assert isinstance(track, Track)
    assert track.catalog_number == 25544
    assert track.name == "ISS (ZARYA)"
    # Target ~180 samples per orbit (default).
    assert 170 < len(track.samples) < 200
    # Times must be monotonically increasing.
    for prev, curr in zip(track.samples, track.samples[1:]):
        assert prev.time_utc < curr.time_utc
    # Latitude bounded by inclination (51.6° + tiny tolerance for
    # ellipsoidal latitude nuance).
    for s in track.samples:
        assert -52.0 <= s.latitude_deg <= 52.0


def test_precompute_track_eop_degraded_propagates(
    iss: Tle, converter: CoordinateConverter
) -> None:
    """The fixture converter has eop_degraded=True; that flag must reach
    every sample and the Track-level property."""
    track = precompute_track(
        iss, converter, start_utc=_NEAR_EPOCH_UTC, duration_seconds=300.0
    )
    assert track.eop_degraded is True
    assert all(s.eop_degraded for s in track.samples)


def test_precompute_track_naive_datetime_raises(
    iss: Tle, converter: CoordinateConverter
) -> None:
    naive = datetime(2024, 1, 1, 14, 0, 0)
    with pytest.raises(ValueError, match="timezone-aware"):
        precompute_track(
            iss, converter, start_utc=naive, duration_seconds=300.0
        )


def test_precompute_track_zero_duration_raises(
    iss: Tle, converter: CoordinateConverter
) -> None:
    with pytest.raises(ValueError, match="duration_seconds"):
        precompute_track(
            iss, converter, start_utc=_NEAR_EPOCH_UTC, duration_seconds=0
        )


def test_precompute_track_custom_step_count(
    iss: Tle, converter: CoordinateConverter
) -> None:
    track = precompute_track(
        iss,
        converter,
        start_utc=_NEAR_EPOCH_UTC,
        duration_seconds=600.0,
        step_seconds=60.0,
    )
    # n_steps = int(600 // 60) + 1 = 11
    assert len(track.samples) == 11


def test_precompute_track_tle_epoch_parsed(
    iss: Tle, converter: CoordinateConverter
) -> None:
    """ISS fixture: 2024-001.54791667 ≈ 2024-01-01 13:09:00 UTC."""
    track = precompute_track(
        iss, converter, start_utc=_NEAR_EPOCH_UTC, duration_seconds=60.0
    )
    expected = datetime(2024, 1, 1, 13, 9, 0, tzinfo=timezone.utc)
    delta_s = abs((track.tle_epoch_utc - expected).total_seconds())
    assert delta_s < 60.0
