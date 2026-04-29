"""Pure-numerical helpers shared by the cartopy and plotly renderers.

Two responsibilities:

1. **Track precomputation.** Walk a uniform time grid, propagate the TLE at
   each step, convert to WGS84, and accumulate :class:`GroundPosition`
   samples into a :class:`Track`. Both renderers consume Tracks rather than
   running propagation themselves — keeps the orbit math out of the
   plotting code.

2. **Antimeridian splitting.** Split a track at ±180° longitude wraparounds
   so plotting libraries don't draw a horizontal line across the world map
   when a satellite crosses the dateline.

This module imports nothing from cartopy or plotly. Unit tests exercise the
math without pulling in heavy plotting dependencies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from sat_tracker.coordinates import (
    CoordinateConverter,
    EcefPosition,
    GroundPosition,
)
from sat_tracker.propagator import propagate
from sat_tracker.tle_fetcher import Tle


# Shared palette used by all renderers (cartopy 2D, plotly 2D, plotly 3D)
# so the same satellite is the same colour across every visualisation.
DEFAULT_TRACK_COLORS: tuple[str, ...] = (
    "#d62728",  # red       — first / single-satellite default
    "#1f77b4",  # blue
    "#2ca02c",  # green
    "#9467bd",  # purple
    "#ff7f0e",  # orange
)

logger = logging.getLogger(__name__)


# Visual-smoothness target. 180 samples around one full orbit gives a
# polyline that looks continuous at any reasonable zoom on a world map
# (one sample per ~2° of true anomaly). Exposed via the
# ``target_samples_per_orbit`` keyword if a caller needs a different
# density.
_TARGET_SAMPLES_PER_ORBIT: int = 180

# Step-size guard rails. ``min_step`` protects against pathological mean
# motions in malformed TLEs (a step of 0.1s would explode memory).
# ``max_step`` caps deep-space cases where 180-samples-per-orbit yields
# multi-minute steps; defensive only — real usage rarely binds.
_MIN_STEP_SECONDS: float = 5.0
_MAX_STEP_SECONDS: float = 900.0  # 15 minutes


@dataclass(frozen=True)
class Orbit3D:
    """Computed 3D Cartesian orbit positions for one satellite.

    Companion to :class:`Track`: same time-grid semantics and same source
    propagation, but the per-sample data is the satellite's actual 3D
    position in an Earth-fixed (ECEF) or inertial (TEME) Cartesian frame
    rather than the sub-satellite point on the ellipsoid.

    Attributes:
        catalog_number: NORAD catalog number from the source TLE.
        name: Satellite name from the source TLE (may be ``None``).
        samples: Tuple of :class:`EcefPosition` objects in chronological
            order. Tuple-not-list for the same hashability/immutability
            discipline as :class:`Track`.
        tle_epoch_utc: Epoch parsed from the TLE's line 1.
        frame: ``"ecef"`` (default — Earth-fixed, ITRF) or ``"teme"``
            (inertial, True Equator Mean Equinox). The 3D renderer reads
            this to decide whether to also rotate the Earth sphere or
            leave it static.
    """

    catalog_number: int
    name: Optional[str]
    samples: tuple[EcefPosition, ...]
    tle_epoch_utc: datetime
    frame: str = "ecef"

    @property
    def eop_degraded(self) -> bool:
        return any(s.eop_degraded for s in self.samples)


@dataclass(frozen=True)
class Track:
    """Computed ground positions for one satellite over a time window.

    Attributes:
        catalog_number: NORAD catalog number from the source TLE.
        name: Satellite name from the source TLE (may be ``None``).
        samples: Tuple of :class:`GroundPosition` objects in chronological
            order. Tuple (rather than list) for hashability and immutability
            — same discipline as the rest of the project's frozen data.
        tle_epoch_utc: Epoch parsed from the TLE's line 1 — useful for
            self-describing plot subtitles.
    """

    catalog_number: int
    name: Optional[str]
    samples: tuple[GroundPosition, ...]
    tle_epoch_utc: datetime

    @property
    def eop_degraded(self) -> bool:
        """True iff any sample was computed with degraded EOP data.

        Constant within a single Track since the converter's timescale
        doesn't change mid-precompute, but exposed as a property so
        consumers don't need to reach into individual samples.
        """
        return any(s.eop_degraded for s in self.samples)


def default_time_step_seconds(
    tle: Tle,
    *,
    target_samples_per_orbit: int = _TARGET_SAMPLES_PER_ORBIT,
    min_step: float = _MIN_STEP_SECONDS,
    max_step: float = _MAX_STEP_SECONDS,
) -> float:
    """Compute a time step that gives ~``target_samples_per_orbit`` per period.

    Args:
        tle: Validated TLE.
        target_samples_per_orbit: Desired sample density per orbital period.
        min_step: Lower clamp in seconds.
        max_step: Upper clamp in seconds.

    Returns:
        Step size in seconds, clamped to ``[min_step, max_step]``.

    Raises:
        ValueError: If the TLE's mean motion is non-positive (malformed TLE).
    """
    period_seconds = _orbital_period_seconds(tle)
    step = period_seconds / target_samples_per_orbit
    return max(min_step, min(max_step, step))


def default_window_seconds(tle: Tle) -> float:
    """One full orbital period from the TLE's mean motion (seconds).

    The natural default window for a single-orbit ground track. For a GEO
    satellite this is ~24h, which produces a small libration figure — that's
    correct physical behaviour, not a degenerate case.
    """
    return _orbital_period_seconds(tle)


def precompute_track(
    tle: Tle,
    converter: CoordinateConverter,
    *,
    start_utc: datetime,
    duration_seconds: float,
    step_seconds: Optional[float] = None,
) -> Track:
    """Propagate and convert at fixed intervals across the time window.

    Args:
        tle: Validated TLE.
        converter: Active :class:`CoordinateConverter`. Provides the loaded
            EOP data and controls the ``eop_degraded`` flag flowing through
            into each sample (and the resulting Track).
        start_utc: Start of the window. Must be timezone-aware. Non-UTC
            timezones are converted internally.
        duration_seconds: Window length, must be positive.
        step_seconds: Override for the time step. If ``None``, computed
            from mean motion via :func:`default_time_step_seconds`.

    Returns:
        A :class:`Track` with samples at uniform intervals from ``start_utc``
        through ``start_utc + duration_seconds`` (inclusive of both endpoints,
        modulo the step).

    Raises:
        ValueError: If ``start_utc`` is naive, or ``duration_seconds`` /
            ``step_seconds`` are non-positive.
    """
    if start_utc.tzinfo is None:
        raise ValueError(
            "start_utc must be timezone-aware. Naive datetimes are rejected "
            "to avoid local-time-as-UTC silent errors."
        )
    if duration_seconds <= 0:
        raise ValueError(
            f"duration_seconds must be positive, got {duration_seconds}"
        )

    start_utc = start_utc.astimezone(timezone.utc)
    if step_seconds is None:
        step_seconds = default_time_step_seconds(tle)
    if step_seconds <= 0:
        raise ValueError(f"step_seconds must be positive, got {step_seconds}")

    n_steps = int(duration_seconds // step_seconds) + 1
    samples: list[GroundPosition] = []
    for i in range(n_steps):
        when = start_utc + timedelta(seconds=i * step_seconds)
        state = propagate(tle, when)
        ground = converter.teme_to_ground(state)
        samples.append(ground)

    logger.debug(
        "Precomputed %d samples for catnr=%d over %.0fs (step=%.1fs)",
        n_steps,
        tle.catalog_number,
        duration_seconds,
        step_seconds,
    )
    return Track(
        catalog_number=tle.catalog_number,
        name=tle.name,
        samples=tuple(samples),
        tle_epoch_utc=_parse_tle_epoch(tle),
    )


def precompute_orbit(
    tle: Tle,
    converter: CoordinateConverter,
    *,
    start_utc: datetime,
    duration_seconds: float,
    step_seconds: Optional[float] = None,
    frame: str = "ecef",
) -> Orbit3D:
    """Propagate and convert at fixed intervals for the 3D orbit renderer.

    Mirrors :func:`precompute_track` but emits 3D Cartesian positions
    (kilometres from Earth's centre) rather than geodetic ground points.
    Reuses the same propagator + converter so a caller building both 2D and
    3D plots from the same window pays for SGP4 evaluation twice but pays
    nothing for stage setup (timescale, EOP).

    Args:
        tle: Validated TLE.
        converter: Active :class:`CoordinateConverter`.
        start_utc: Window start. Must be timezone-aware.
        duration_seconds: Window length in seconds. Must be positive.
        step_seconds: Override for the time step. ``None`` → derive from
            mean motion via :func:`default_time_step_seconds`.
        frame: ``"ecef"`` (default, Earth-fixed) or ``"teme"`` (inertial).
            ``"teme"`` skips the TEME→ITRF rotation and stores the raw
            inertial position; useful for orbital-mechanics demos but
            visually unintuitive with a fixed Earth.

    Returns:
        An :class:`Orbit3D` with samples at uniform intervals across the
        window.

    Raises:
        ValueError: If ``start_utc`` is naive, ``duration_seconds`` /
            ``step_seconds`` are non-positive, or ``frame`` is not one of
            ``"ecef"`` / ``"teme"``.
    """
    if frame not in ("ecef", "teme"):
        raise ValueError(
            f"frame must be 'ecef' or 'teme', got {frame!r}"
        )
    if start_utc.tzinfo is None:
        raise ValueError(
            "start_utc must be timezone-aware. Naive datetimes are rejected "
            "to avoid local-time-as-UTC silent errors."
        )
    if duration_seconds <= 0:
        raise ValueError(
            f"duration_seconds must be positive, got {duration_seconds}"
        )

    start_utc = start_utc.astimezone(timezone.utc)
    if step_seconds is None:
        step_seconds = default_time_step_seconds(tle)
    if step_seconds <= 0:
        raise ValueError(f"step_seconds must be positive, got {step_seconds}")

    n_steps = int(duration_seconds // step_seconds) + 1
    samples: list[EcefPosition] = []
    for i in range(n_steps):
        when = start_utc + timedelta(seconds=i * step_seconds)
        state = propagate(tle, when)
        if frame == "ecef":
            samples.append(converter.teme_to_ecef(state))
        else:
            # Inertial frame: store the raw TEME position straight from the
            # propagator. eop_degraded is irrelevant here (no rotation
            # applied) but we mirror the converter's flag for consistency
            # with the dataclass shape.
            samples.append(
                EcefPosition(
                    time_utc=state.time_utc,
                    x_km=float(state.position_km[0]),
                    y_km=float(state.position_km[1]),
                    z_km=float(state.position_km[2]),
                    eop_degraded=converter.eop_degraded,
                )
            )

    logger.debug(
        "Precomputed %d 3D samples (frame=%s) for catnr=%d over %.0fs",
        n_steps,
        frame,
        tle.catalog_number,
        duration_seconds,
    )
    return Orbit3D(
        catalog_number=tle.catalog_number,
        name=tle.name,
        samples=tuple(samples),
        tle_epoch_utc=_parse_tle_epoch(tle),
        frame=frame,
    )


def split_at_antimeridian(
    samples: list[GroundPosition],
) -> list[list[GroundPosition]]:
    """Split a track into segments at ±180° longitude crossings.

    Detects ``|lon[i+1] - lon[i]| > 180`` between successive samples as a
    longitude wrap. The threshold is robust at any reasonable time step:
    no real satellite ground point can move more than 180° in one step,
    so a delta exceeding 180° must be a wraparound.

    Plotting libraries draw a straight line between successive points
    without wrapping; without this split, a track that crosses the dateline
    draws a horizontal line across the entire world map. Each returned
    segment must be plotted as its own polyline.

    Args:
        samples: Chronologically-ordered ground positions.

    Returns:
        List of segments. Each segment is a contiguous chunk of samples with
        no antimeridian crossing inside it. An empty input yields an empty
        output; a single sample yields one one-element segment.
    """
    if not samples:
        return []
    if len(samples) == 1:
        return [list(samples)]

    segments: list[list[GroundPosition]] = [[samples[0]]]
    for prev, curr in zip(samples, samples[1:]):
        if abs(curr.longitude_deg - prev.longitude_deg) > 180.0:
            segments.append([curr])
        else:
            segments[-1].append(curr)
    return segments


def _orbital_period_seconds(tle: Tle) -> float:
    """Period in seconds, parsed from TLE mean motion (rev/day)."""
    mean_motion_rev_per_day = float(tle.line2[52:63])
    if mean_motion_rev_per_day <= 0:
        raise ValueError(
            f"non-positive mean motion {mean_motion_rev_per_day} "
            f"(malformed TLE, catnr={tle.catalog_number})"
        )
    return 86400.0 / mean_motion_rev_per_day


def _parse_tle_epoch(tle: Tle) -> datetime:
    """Parse the TLE epoch from line 1 positions 18-31.

    Format: ``YYDDD.DDDDDDDD`` where ``YY`` is a 2-digit year (00-56 maps to
    2000s; 57-99 to 1900s — the standard NORAD pivot) and the rest is
    fractional day-of-year.
    """
    yy = int(tle.line1[18:20])
    year = 2000 + yy if yy < 57 else 1900 + yy
    doy_frac = float(tle.line1[20:32])
    return datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(
        days=doy_frac - 1.0
    )
