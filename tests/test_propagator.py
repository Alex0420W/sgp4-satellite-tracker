"""Tests for ``sat_tracker.propagator``.

Test layering follows the plan from stage 3:

* A magnitude-only smoke test that catches gross errors (units, frame mix-ups).
* A regression test that pins our wrapper's output against a direct
  ``sgp4.api.Satrec`` call — guards against jd/fr math bugs and vector ordering
  mistakes. We trust the underlying ``sgp4`` library's correctness, which is
  validated against Vallado's full SGP4-VER suite in its own CI.
* Behavioural tests for tz handling and SGP4 error-code translation.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest
from sgp4.api import Satrec, jday

from sat_tracker.propagator import (
    PropagationError,
    StateVector,
    propagate,
)
from sat_tracker.tle_fetcher import Tle

# The fixture TLE has epoch = day 001.54791667 of 2024
# = 2024-01-01 + 0.54791667 days
# ~ 2024-01-01 13:09:00.000 UTC. Used as a near-epoch propagation target.
_EPOCH_UTC = datetime(2024, 1, 1, 13, 9, 0, tzinfo=timezone.utc)


@pytest.fixture
def iss(iss_tle: tuple[str, str, str]) -> Tle:
    name, line1, line2 = iss_tle
    return Tle(name=name, line1=line1, line2=line2)


def test_iss_at_epoch_is_in_leo(iss: Tle) -> None:
    """Smoke: ISS state at its own epoch must look like a LEO satellite."""
    state = propagate(iss, _EPOCH_UTC)

    assert isinstance(state, StateVector)
    assert state.time_utc == _EPOCH_UTC

    x, y, z = state.position_km
    radius_km = math.sqrt(x * x + y * y + z * z)
    # Earth radius ~6378 km; ISS altitude ~400 km → radius ~6778 km.
    # Wide window — this is a sanity check, not a precision test.
    assert 6500 < radius_km < 7000, f"radius {radius_km:.1f} km not LEO-shaped"

    vx, vy, vz = state.velocity_km_s
    speed_km_s = math.sqrt(vx * vx + vy * vy + vz * vz)
    # Circular LEO speed ~7.66 km/s.
    assert 7.0 < speed_km_s < 8.5, f"speed {speed_km_s:.2f} km/s not LEO-shaped"


def test_wrapper_matches_direct_sgp4_call(iss: Tle) -> None:
    """Regression: our jd/fr math and vector ordering must match the library.

    Calls ``Satrec.sgp4`` directly with manually-computed ``jd, fr`` for the
    same instant, and asserts our wrapper produces bit-identical output. This
    is the primary defence against subtle wrapping bugs (timezone leaks,
    fractional-second rounding, position/velocity swaps).
    """
    sat = Satrec.twoline2rv(iss.line1, iss.line2)
    jd, fr = jday(2024, 1, 1, 13, 9, 0.0)
    err, expected_pos, expected_vel = sat.sgp4(jd, fr)
    assert err == 0  # If this fails the fixture itself is broken.

    state = propagate(iss, _EPOCH_UTC)

    assert state.position_km == pytest.approx(expected_pos, abs=1e-9)
    assert state.velocity_km_s == pytest.approx(expected_vel, abs=1e-9)


def test_naive_datetime_raises(iss: Tle) -> None:
    naive = datetime(2024, 1, 1, 13, 9, 0)  # no tzinfo
    with pytest.raises(ValueError, match="[Nn]aive datetime"):
        propagate(iss, naive)


def test_non_utc_tz_is_converted(iss: Tle) -> None:
    """tz-aware non-UTC datetime must produce the same answer as its UTC equivalent."""
    plus_five = timezone(timedelta(hours=5))
    same_moment_in_plus_five = datetime(2024, 1, 1, 18, 9, 0, tzinfo=plus_five)

    in_utc = propagate(iss, _EPOCH_UTC)
    in_local = propagate(iss, same_moment_in_plus_five)

    assert in_local.time_utc == in_utc.time_utc
    for a, b in zip(in_local.position_km, in_utc.position_km):
        assert a == pytest.approx(b, abs=1e-9)


def test_sgp4_error_code_raises_propagation_error(
    iss: Tle, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-zero SGP4 error codes must surface as ``PropagationError``."""

    def _fake_sgp4(self, jd, fr):  # noqa: ANN001, ARG001
        # Pretend the satellite has decayed (error code 6).
        return (6, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

    monkeypatch.setattr(Satrec, "sgp4", _fake_sgp4)

    with pytest.raises(PropagationError, match="decayed"):
        propagate(iss, _EPOCH_UTC)
