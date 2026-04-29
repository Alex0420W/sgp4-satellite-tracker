"""Tests for ``sat_tracker.coordinates``.

Tests use Skyfield's bundled timescale (``builtin=True``) injected into the
converter — no network, no IERS data on disk. One test explicitly exercises
the fallback path by monkeypatching ``Loader`` to fail the fresh-data fetch.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from skyfield.api import load

from sat_tracker import coordinates as coords_module
from sat_tracker.config import Config
from sat_tracker.coordinates import (
    CoordinateConverter,
    GroundPosition,
    _ecef_to_geodetic,
)
from sat_tracker.propagator import propagate
from sat_tracker.tle_fetcher import Tle


_EPOCH_UTC = datetime(2024, 1, 1, 13, 9, 0, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def bundled_timescale():
    """Skyfield bundled-data Timescale, built once per module."""
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
def offline_converter(config: Config, bundled_timescale) -> CoordinateConverter:
    return CoordinateConverter(
        config, timescale=bundled_timescale, eop_degraded=True
    )


@pytest.fixture
def iss(iss_tle: tuple[str, str, str]) -> Tle:
    name, line1, line2 = iss_tle
    return Tle(name=name, line1=line1, line2=line2)


# -- ECEF -> geodetic math: deterministic, no Skyfield required ---------------


def test_ecef_to_geodetic_equator() -> None:
    """Equator + altitude: latitude 0, longitude 0."""
    a = 6378.137
    lat, lon, alt = _ecef_to_geodetic(a + 500.0, 0.0, 0.0)
    assert lat == pytest.approx(0.0, abs=1e-9)
    assert lon == pytest.approx(0.0, abs=1e-9)
    assert alt == pytest.approx(500.0, abs=1e-6)


def test_ecef_to_geodetic_north_pole() -> None:
    """Point on +z axis: latitude 90, altitude is z minus polar radius."""
    b = 6378.137 * (1.0 - 1.0 / 298.257223563)
    lat, _lon, alt = _ecef_to_geodetic(0.0, 0.0, b + 1000.0)
    assert lat == pytest.approx(90.0, abs=1e-9)
    assert alt == pytest.approx(1000.0, abs=1e-6)


def test_ecef_to_geodetic_round_trip_at_60_north() -> None:
    """Construct a known geodetic point on the ellipsoid; verify recovery."""
    lat_target = 60.0
    a = 6378.137
    f = 1.0 / 298.257223563
    e2 = 2.0 * f - f * f
    sin_lat = math.sin(math.radians(lat_target))
    cos_lat = math.cos(math.radians(lat_target))
    n = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
    x = n * cos_lat
    z = n * (1.0 - e2) * sin_lat

    lat, lon, alt = _ecef_to_geodetic(x, 0.0, z)
    assert lat == pytest.approx(60.0, abs=1e-6)
    assert lon == pytest.approx(0.0, abs=1e-9)
    assert alt == pytest.approx(0.0, abs=1e-6)


# -- Full TEME -> ground pipeline -- requires bundled timescale ----------------


def test_iss_ground_position_is_plausible(
    iss: Tle, offline_converter: CoordinateConverter
) -> None:
    state = propagate(iss, _EPOCH_UTC)
    ground = offline_converter.teme_to_ground(state)

    assert isinstance(ground, GroundPosition)
    # ISS orbital inclination is 51.6° — sub-satellite latitude is bounded
    # by inclination, with a tiny tolerance for ellipsoid effects.
    assert -52.0 <= ground.latitude_deg <= 52.0
    assert -180.0 <= ground.longitude_deg <= 180.0
    # ISS orbits ~400 km above the surface.
    assert 350.0 <= ground.altitude_km <= 450.0
    # time_utc must pass through unchanged from input StateVector.
    assert ground.time_utc == state.time_utc


def test_eop_degraded_flag_propagates_to_position(
    iss: Tle, offline_converter: CoordinateConverter
) -> None:
    state = propagate(iss, _EPOCH_UTC)
    ground = offline_converter.teme_to_ground(state)
    assert ground.eop_degraded is True


def test_longitude_normalised_across_24h(
    iss: Tle, offline_converter: CoordinateConverter
) -> None:
    """Longitudes from a full day of ISS positions stay in [-180, 180]."""
    times = [_EPOCH_UTC + timedelta(minutes=10 * i) for i in range(144)]
    for when in times:
        state = propagate(iss, when)
        ground = offline_converter.teme_to_ground(state)
        assert -180.0 <= ground.longitude_deg <= 180.0
        assert -90.0 <= ground.latitude_deg <= 90.0


def test_eop_fallback_on_load_failure(
    config: Config,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Simulating an IERS download failure must trigger bundled fallback."""
    real_loader_cls = coords_module.Loader

    class _FailingLoader:
        def __init__(self, path: str) -> None:
            self._real = real_loader_cls(path)

        def timescale(self, builtin: bool = True):
            if not builtin:
                raise OSError("simulated IERS network failure")
            return self._real.timescale(builtin=True)

    monkeypatch.setattr(coords_module, "Loader", _FailingLoader)

    with caplog.at_level(logging.WARNING, logger="sat_tracker.coordinates"):
        converter = CoordinateConverter(config)

    assert converter.eop_degraded is True
    assert "simulated IERS network failure" in caplog.text
    assert "eop_degraded=True" in caplog.text
    assert "bundled" in caplog.text.lower()
