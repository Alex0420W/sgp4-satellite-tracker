"""Shared pytest fixtures for the sat_tracker test suite."""

from __future__ import annotations

import pytest

# A real, historical ISS (ZARYA) TLE. Stable test input — do not refresh.
# Epoch: 2024-01-01 (day 001 of 2024). Use only in tests; not for live tracking.
_ISS_TLE_NAME = "ISS (ZARYA)"
_ISS_TLE_LINE1 = (
    "1 25544U 98067A   24001.54791667  .00016717  00000-0  30571-3 0  9993"
)
_ISS_TLE_LINE2 = (
    "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.49960452433330"
)


@pytest.fixture
def iss_tle() -> tuple[str, str, str]:
    """Return a fixed (name, line1, line2) ISS TLE for deterministic tests."""
    return _ISS_TLE_NAME, _ISS_TLE_LINE1, _ISS_TLE_LINE2
