"""Tests for ``sat_tracker.passes``.

Tests use Skyfield's bundled timescale (no network) injected via the
shared :class:`CoordinateConverter`, plus an injected ``ephemeris=None``
to exercise the sunlit-fallback path without hitting the IERS / JPL
ephemeris download.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from skyfield.api import load

from sat_tracker.config import Config
from sat_tracker.coordinates import CoordinateConverter
from sat_tracker.passes import (
    GroundStation,
    Pass,
    PassPredictor,
)
from sat_tracker.tle_fetcher import Tle


# Fixture TLE epoch is 2024-01-01 day 001.54791667. Use a window starting
# right after the epoch so SGP4 propagation is fresh.
_NEAR_EPOCH_UTC = datetime(2024, 1, 1, 14, 0, 0, tzinfo=timezone.utc)


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
def predictor(
    config: Config, converter: CoordinateConverter
) -> PassPredictor:
    # ephemeris=None forces the sunlit-fallback path so tests don't hit
    # the network for de421.bsp.
    return PassPredictor(config, converter, ephemeris=None)


@pytest.fixture
def iss(iss_tle: tuple[str, str, str]) -> Tle:
    name, line1, line2 = iss_tle
    return Tle(name=name, line1=line1, line2=line2)


@pytest.fixture
def cape_canaveral() -> GroundStation:
    return GroundStation(
        latitude_deg=28.467,
        longitude_deg=-80.567,
        altitude_km=0.005,
        name="Cape Canaveral",
    )


# -- GroundStation validation -------------------------------------------------


def test_ground_station_rejects_bad_latitude() -> None:
    with pytest.raises(ValueError, match="latitude"):
        GroundStation(latitude_deg=91.0, longitude_deg=0.0, altitude_km=0.0)


def test_ground_station_rejects_bad_longitude() -> None:
    with pytest.raises(ValueError, match="longitude"):
        GroundStation(latitude_deg=0.0, longitude_deg=181.0, altitude_km=0.0)


def test_ground_station_name_optional() -> None:
    s = GroundStation(latitude_deg=0.0, longitude_deg=0.0, altitude_km=0.0)
    assert s.name is None


# -- Predictor input validation -----------------------------------------------


def test_naive_datetime_raises(
    predictor: PassPredictor, iss: Tle, cape_canaveral: GroundStation
) -> None:
    naive = datetime(2024, 1, 1, 12, 0, 0)
    with pytest.raises(ValueError, match="timezone-aware"):
        predictor.predict_passes(
            iss, cape_canaveral, start_utc=naive, hours=24
        )


def test_non_positive_hours_raises(
    predictor: PassPredictor, iss: Tle, cape_canaveral: GroundStation
) -> None:
    with pytest.raises(ValueError, match="positive"):
        predictor.predict_passes(
            iss, cape_canaveral, start_utc=_NEAR_EPOCH_UTC, hours=0
        )


# -- Mean-motion gate ---------------------------------------------------------


def test_geo_mean_motion_gate_skips_prediction(
    predictor: PassPredictor,
    iss: Tle,
    cape_canaveral: GroundStation,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Synthetic GEO TLE: line2 mean-motion field set to 1.0 rev/day."""
    # Reuse the ISS line1 (catnr/format are valid); only line2 mean-motion
    # matters for the gate. Construction bypasses the fetcher's checksum
    # validation, which is fine — we're testing the gate, not validation.
    # Mean motion field at positions 52-62 contains " 1.00000000" (1.0 rev/day).
    # Layout: <52-char prefix ending with separator space> <11-char mm>
    # <5-char rev count> <1-char checksum> = 69 chars total.
    geo_line2 = (
        "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 "  # 52 chars
        " 1.00000000000010"                                      # 17 chars
    )
    assert len(geo_line2) == 69, f"synthetic line2 length {len(geo_line2)}"
    geo_tle = Tle(name="FAKE-GEO", line1=iss.line1, line2=geo_line2)

    with caplog.at_level(logging.WARNING, logger="sat_tracker.passes"):
        passes = predictor.predict_passes(
            geo_tle,
            cape_canaveral,
            start_utc=_NEAR_EPOCH_UTC,
            hours=24,
        )

    assert passes == []
    # Warning must include the actual mean motion so log-readers can verify.
    assert "1.0000" in caplog.text
    assert "GEO" in caplog.text or "deep-space" in caplog.text


# -- ISS pass prediction smoke tests -----------------------------------------


def test_iss_24h_window_returns_some_passes(
    predictor: PassPredictor, iss: Tle, cape_canaveral: GroundStation
) -> None:
    """Sanity: ISS over a mid-latitude station should produce >=1 passes/24h."""
    passes = predictor.predict_passes(
        iss, cape_canaveral, start_utc=_NEAR_EPOCH_UTC, hours=24
    )

    assert len(passes) >= 1, "ISS should pass over Cape Canaveral at least once/day"
    assert len(passes) <= 30, "implausibly many ISS passes — check inputs"

    for p in passes:
        assert isinstance(p, Pass)
        assert p.satellite_catnr == 25544
        # AOS strictly before LOS, both within the requested window.
        assert p.aos_utc < p.los_utc
        assert _NEAR_EPOCH_UTC <= p.aos_utc
        assert p.los_utc <= _NEAR_EPOCH_UTC + timedelta(hours=24)
        # Max elevation within the pass interval.
        assert p.aos_utc <= p.max_elevation_time_utc <= p.los_utc
        # Threshold compliance and physical bounds.
        assert p.max_elevation_deg >= 10.0
        assert p.max_elevation_deg <= 90.0
        for az in (
            p.azimuth_at_aos_deg,
            p.azimuth_at_max_deg,
            p.azimuth_at_los_deg,
        ):
            assert 0.0 <= az < 360.0
        # All time fields must be tz-aware UTC.
        for t in (p.aos_utc, p.los_utc, p.max_elevation_time_utc):
            assert t.tzinfo is not None
            assert t.utcoffset() == timedelta(0)


def test_no_passes_in_one_second_window(
    predictor: PassPredictor, iss: Tle, cape_canaveral: GroundStation
) -> None:
    """A 1-second window cannot contain a complete (rise + set) pass."""
    passes = predictor.predict_passes(
        iss,
        cape_canaveral,
        start_utc=_NEAR_EPOCH_UTC,
        hours=1.0 / 3600.0,
    )
    assert passes == []


# -- Sunlit fallback ----------------------------------------------------------


def test_sunlit_is_none_when_ephemeris_unavailable(
    predictor: PassPredictor, iss: Tle, cape_canaveral: GroundStation
) -> None:
    """The injected ephemeris=None must produce sunlit=None on every pass."""
    passes = predictor.predict_passes(
        iss, cape_canaveral, start_utc=_NEAR_EPOCH_UTC, hours=24
    )
    assert len(passes) >= 1
    for p in passes:
        assert p.sunlit is None


def test_eop_degraded_flag_propagates_from_converter(
    predictor: PassPredictor, iss: Tle, cape_canaveral: GroundStation
) -> None:
    passes = predictor.predict_passes(
        iss, cape_canaveral, start_utc=_NEAR_EPOCH_UTC, hours=24
    )
    assert len(passes) >= 1
    for p in passes:
        # The converter fixture was built with eop_degraded=True.
        assert p.eop_degraded is True


# -- External validation: heavens-above.com ----------------------------------


# TLE used by Heavens-Above to compute the reference values below. Pinned in
# source for test determinism — re-running this test in 6 months produces the
# same prediction regardless of cache state. Epoch is 2026-118.52255674
# (2026-04-28 12:32:29 UTC), within ~1.9 days of the reference pass at
# 2026-04-30 09:48 UTC. SGP4 accuracy at this epoch offset is sub-km, well
# inside the ±60 second time tolerance.
_HA_TLE_NAME = "ISS (ZARYA)"
_HA_TLE_LINE1 = (
    "1 25544U 98067A   26118.52255674  .00007533  00000+0  14468-3 0  9997"
)
_HA_TLE_LINE2 = (
    "2 25544  51.6321 185.9244 0007073   0.6055 359.4941 15.49004967564056"
)

# Heavens-Above ground station for the reference lookup.
_HA_STATION = GroundStation(
    latitude_deg=40.575,
    longitude_deg=-105.0983,
    altitude_km=0.0,
)

# Reference values from Heavens-Above (UTC, after the MDT-vs-MST conversion
# documented in the test docstring).
_HA_EXPECTED_AOS_UTC = datetime(2026, 4, 30, 9, 46, 53, tzinfo=timezone.utc)
_HA_EXPECTED_AOS_AZ_DEG = 325.0
_HA_EXPECTED_MAX_UTC = datetime(2026, 4, 30, 9, 48, 32, tzinfo=timezone.utc)
_HA_EXPECTED_MAX_EL_DEG = 13.0
_HA_EXPECTED_MAX_AZ_DEG = 355.0
_HA_EXPECTED_LOS_UTC = datetime(2026, 4, 30, 9, 50, 11, tzinfo=timezone.utc)
_HA_EXPECTED_LOS_AZ_DEG = 24.0

# Tolerances per stage 6 spec.
_HA_TOL_TIME_S = 60.0
_HA_TOL_ELEVATION_DEG = 2.0
_HA_TOL_AZIMUTH_DEG = 5.0


def _angular_diff_deg(actual: float, expected: float) -> float:
    """Smallest signed difference on the 0..360 circle, returned as |abs|."""
    return abs(((actual - expected) + 540.0) % 360.0 - 180.0)


def test_heavens_above_reference_pass(
    predictor: PassPredictor,
) -> None:
    """Validate one pass prediction against heavens-above.com.

    Lookup metadata
    ---------------
    URL:           https://heavens-above.com/passdetails.aspx
                   ?lat=40.575&lng=-105.0983&loc=Unnamed&alt=0&tz=MST
                   &satid=25544&mjd=61160.4087065698&type=V
    Lookup date:   2026-04-28
    Pass date:     2026-04-30
    TLE epoch:     2026-118.52255674 (2026-04-28 12:32:29 UTC)

    Timezone reasoning (load-bearing — keep this on the record)
    -----------------------------------------------------------
    The Heavens-Above URL parameter says ``tz=MST``, but Colorado in late
    April 2026 is on **MDT** (UTC-6), not MST (UTC-7). US DST 2026 runs
    from 2026-03-08 to 2026-11-01, so April 30 is squarely inside DST.
    Heavens-Above's site labels the abbreviation incorrectly during DST
    (a long-known quirk on amateur observation forums) but its displayed
    times are real local clock time, i.e. MDT.

    Cross-check: the URL's ``mjd=61160.4087065698`` corresponds to
    JD 2461160.9087, which is 2026-04-30 09:48 UTC. The displayed local
    pass time of ~03:48 + 6h = 09:48 UTC matches. So the conversion is
    ``UTC = displayed_local + 6h`` (i.e. MDT semantics), and that's what
    the expected values below were derived under.

    Tolerances
    ----------
    Time:      ±60 seconds  (applied to AOS / max / LOS)
    Elevation: ±2 degrees   (max-elevation only)
    Azimuth:   ±5 degrees   (azimuth precision degrades at low elevation)
    """
    tle = Tle(name=_HA_TLE_NAME, line1=_HA_TLE_LINE1, line2=_HA_TLE_LINE2)

    # Search a 1-hour window centred on the expected pass. Wide enough to
    # cover ±15 minutes of prediction drift; narrow enough that we won't
    # collide with a neighbouring ISS pass.
    start = datetime(2026, 4, 30, 9, 30, 0, tzinfo=timezone.utc)
    passes = predictor.predict_passes(
        tle, _HA_STATION, start_utc=start, hours=1.0
    )

    assert len(passes) >= 1, (
        f"expected at least one ISS pass in [09:30, 10:30] UTC on "
        f"2026-04-30 over the Heavens-Above station; got {len(passes)}"
    )
    # Pick the pass closest to the expected AOS — defensive in case a
    # neighbouring pass also lands in the window.
    p = min(
        passes,
        key=lambda x: abs(
            (x.aos_utc - _HA_EXPECTED_AOS_UTC).total_seconds()
        ),
    )

    # Build full expected/actual dicts up front so the failure message
    # can dump everything at once — avoids the bisect-by-rerunning loop.
    expected = {
        "aos_utc": _HA_EXPECTED_AOS_UTC.isoformat(),
        "aos_az_deg": _HA_EXPECTED_AOS_AZ_DEG,
        "max_utc": _HA_EXPECTED_MAX_UTC.isoformat(),
        "max_elevation_deg": _HA_EXPECTED_MAX_EL_DEG,
        "max_az_deg": _HA_EXPECTED_MAX_AZ_DEG,
        "los_utc": _HA_EXPECTED_LOS_UTC.isoformat(),
        "los_az_deg": _HA_EXPECTED_LOS_AZ_DEG,
    }
    actual = {
        "aos_utc": p.aos_utc.isoformat(),
        "aos_az_deg": p.azimuth_at_aos_deg,
        "max_utc": p.max_elevation_time_utc.isoformat(),
        "max_elevation_deg": p.max_elevation_deg,
        "max_az_deg": p.azimuth_at_max_deg,
        "los_utc": p.los_utc.isoformat(),
        "los_az_deg": p.azimuth_at_los_deg,
    }

    failures: list[str] = []

    def _check_time(label: str, expected_dt: datetime, actual_dt: datetime) -> None:
        delta_s = abs((actual_dt - expected_dt).total_seconds())
        if delta_s > _HA_TOL_TIME_S:
            failures.append(
                f"  {label}: expected {expected_dt.isoformat()}, "
                f"got {actual_dt.isoformat()} "
                f"(|Δ| {delta_s:.1f}s, tolerance ±{_HA_TOL_TIME_S:.0f}s)"
            )

    def _check_az(label: str, expected_az: float, actual_az: float) -> None:
        delta = _angular_diff_deg(actual_az, expected_az)
        if delta > _HA_TOL_AZIMUTH_DEG:
            failures.append(
                f"  {label}: expected {expected_az:.1f}°, "
                f"got {actual_az:.1f}° "
                f"(|Δ| {delta:.2f}°, tolerance ±{_HA_TOL_AZIMUTH_DEG:.0f}°)"
            )

    _check_time("AOS time", _HA_EXPECTED_AOS_UTC, p.aos_utc)
    _check_time("MAX time", _HA_EXPECTED_MAX_UTC, p.max_elevation_time_utc)
    _check_time("LOS time", _HA_EXPECTED_LOS_UTC, p.los_utc)

    el_delta = abs(p.max_elevation_deg - _HA_EXPECTED_MAX_EL_DEG)
    if el_delta > _HA_TOL_ELEVATION_DEG:
        failures.append(
            f"  MAX elevation: expected {_HA_EXPECTED_MAX_EL_DEG:.1f}°, "
            f"got {p.max_elevation_deg:.2f}° "
            f"(|Δ| {el_delta:.2f}°, tolerance ±{_HA_TOL_ELEVATION_DEG:.0f}°)"
        )

    _check_az("AOS azimuth", _HA_EXPECTED_AOS_AZ_DEG, p.azimuth_at_aos_deg)
    _check_az("MAX azimuth", _HA_EXPECTED_MAX_AZ_DEG, p.azimuth_at_max_deg)
    _check_az("LOS azimuth", _HA_EXPECTED_LOS_AZ_DEG, p.azimuth_at_los_deg)

    if failures:
        msg_lines = [
            "Heavens-Above reference comparison FAILED.",
            "",
            "Expected:",
            *(f"  {k:20s} {v}" for k, v in expected.items()),
            "",
            "Actual:",
            *(f"  {k:20s} {v}" for k, v in actual.items()),
            "",
            "Out-of-tolerance differences:",
            *failures,
        ]
        raise AssertionError("\n".join(msg_lines))
