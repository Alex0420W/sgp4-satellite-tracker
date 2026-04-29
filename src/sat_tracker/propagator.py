"""SGP4 orbit propagation.

Takes a validated :class:`~sat_tracker.tle_fetcher.Tle` and a UTC datetime and
returns the satellite's position and velocity in the **TEME** (True Equator,
Mean Equinox) reference frame — the quasi-inertial frame SGP4 natively outputs
in.

Design notes:
    * Stateless — exposed as a module-level function rather than a class.
      ``Satrec`` objects are cheap to construct and the library has no pooling
      to take advantage of.
    * Strict on naive datetimes (``ValueError``); tolerant of tz-aware
      datetimes in any timezone (silently converted to UTC). The trap we're
      guarding against is implicit-local-time, not non-UTC labelling.
    * Non-zero SGP4 error codes are wrapped as :class:`PropagationError` with
      a human-readable explanation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from sgp4.api import Satrec, jday

from sat_tracker.tle_fetcher import Tle

logger = logging.getLogger(__name__)


class PropagationError(RuntimeError):
    """Raised when SGP4 fails to propagate a TLE (non-zero error code)."""


@dataclass(frozen=True)
class StateVector:
    """Satellite state at a specific instant, expressed in the TEME frame.

    Attributes:
        time_utc: Timezone-aware UTC datetime the state is evaluated at.
        position_km: ``(x, y, z)`` Earth-centred TEME position, in kilometres.
        velocity_km_s: ``(vx, vy, vz)`` TEME velocity, in kilometres/second.
    """

    time_utc: datetime
    position_km: tuple[float, float, float]
    velocity_km_s: tuple[float, float, float]


# Map SGP4 error codes to human-readable messages. Codes from Vallado's SGP4
# reference implementation; mirrored by python-sgp4's Satrec.sgp4().
_SGP4_ERROR_MESSAGES: dict[int, str] = {
    1: "mean eccentricity out of range (TLE may be corrupt)",
    2: "mean motion is negative (TLE is invalid)",
    3: "perturbed eccentricity out of range during propagation",
    4: "semi-latus rectum < 0 (orbit is degenerate at this time)",
    5: "epoch elements are sub-orbital",
    6: "satellite has decayed (re-entered) before this time",
}


def propagate(tle: Tle, when: datetime) -> StateVector:
    """Propagate a TLE to the given moment and return its TEME state.

    Args:
        tle: A validated TLE.
        when: Target instant. Must be timezone-aware. Non-UTC timezones are
            converted to UTC internally.

    Returns:
        A :class:`StateVector` in the TEME frame at ``when`` (in UTC).

    Raises:
        ValueError: If ``when`` is a naive datetime (no tzinfo).
        PropagationError: If SGP4 returns a non-zero error code.
    """
    when_utc = _require_aware_then_utc(when)

    sat = Satrec.twoline2rv(tle.line1, tle.line2)
    seconds_with_micros = when_utc.second + when_utc.microsecond / 1_000_000
    jd, fr = jday(
        when_utc.year,
        when_utc.month,
        when_utc.day,
        when_utc.hour,
        when_utc.minute,
        seconds_with_micros,
    )

    error, position, velocity = sat.sgp4(jd, fr)
    if error != 0:
        explanation = _SGP4_ERROR_MESSAGES.get(error, f"unknown error code {error}")
        raise PropagationError(
            f"SGP4 propagation failed for catnr={tle.catalog_number} at "
            f"{when_utc.isoformat()}: {explanation} (code {error})."
        )

    logger.debug(
        "Propagated catnr=%d to %s",
        tle.catalog_number,
        when_utc.isoformat(),
    )
    return StateVector(
        time_utc=when_utc,
        position_km=(position[0], position[1], position[2]),
        velocity_km_s=(velocity[0], velocity[1], velocity[2]),
    )


def _require_aware_then_utc(when: datetime) -> datetime:
    """Reject naive datetimes; convert any tz-aware datetime to UTC.

    Naive datetimes are rejected because we cannot tell whether the caller
    meant UTC or local time, and silently treating them as UTC produces
    very-wrong-looking-correct positions (a 5-hour timezone offset translates
    to ~140,000 km of error for a LEO satellite).
    """
    if when.tzinfo is None:
        raise ValueError(
            "Naive datetime passed to propagate() — must be timezone-aware. "
            "Use datetime.now(timezone.utc) or attach tzinfo explicitly."
        )
    return when.astimezone(timezone.utc)
