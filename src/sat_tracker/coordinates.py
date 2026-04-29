"""TEME -> ITRF (ECEF) -> WGS84 geodetic coordinate transformations.

Converts the satellite state vectors emitted by ``sat_tracker.propagator`` into
sub-satellite points (geodetic latitude/longitude/altitude on the WGS84
ellipsoid) suitable for printing, mapping, or pass prediction.

Pipeline:
    StateVector(TEME)          (from propagator)
        |
        | TEME -> ITRF rotation by GMST(jd_ut1)  (via Skyfield)
        v
    ECEF Cartesian (km)
        |
        | Bowring 1976 closed-form ECEF -> geodetic
        v
    GroundPosition(lat, lon, alt)

Earth Orientation Parameters (EOP):
    Skyfield's ``Loader.timescale(builtin=False)`` downloads fresh
    USNO/IERS data into ``config.cache_dir``. On failure we fall back to
    Skyfield's bundled timescale and flag every emitted GroundPosition with
    ``eop_degraded=True``. Polar motion (xp, yp) is NOT used by this MVP —
    leaving it zero costs ~5-10 m at the equator, well within tolerance for
    visualization and pass-prediction at MVP scope.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
from skyfield.api import Loader
from skyfield.sgp4lib import TEME_to_ITRF
from skyfield.timelib import Timescale

from sat_tracker.config import Config
from sat_tracker.propagator import StateVector

logger = logging.getLogger(__name__)

# WGS84 defining constants (NIMA TR8350.2).
_WGS84_A_KM: float = 6378.137
_WGS84_F: float = 1.0 / 298.257223563


@dataclass(frozen=True)
class EcefPosition:
    """ECEF (ITRF) Cartesian position of a satellite at a given UTC instant.

    Distinct from :class:`GroundPosition` — this is the satellite's actual
    3D position in the Earth-fixed frame (kilometres from Earth's centre),
    not the sub-satellite point on the ellipsoid surface. Used by the 3D
    orbit renderer; not reduced to geodetic lat/lon/alt.

    Attributes:
        time_utc: Instant this position represents (copied from the source
            :class:`~sat_tracker.propagator.StateVector`).
        x_km, y_km, z_km: ECEF Cartesian coordinates in kilometres.
        eop_degraded: True when the underlying timescale is the bundled
            (offline) one. Same semantics as :class:`GroundPosition`.
    """

    time_utc: datetime
    x_km: float
    y_km: float
    z_km: float
    eop_degraded: bool


@dataclass(frozen=True)
class GroundPosition:
    """Geodetic sub-satellite point on the WGS84 ellipsoid.

    Attributes:
        time_utc: The instant this position represents — copied verbatim from
            the originating :class:`~sat_tracker.propagator.StateVector`. Not
            the time the conversion was performed.
        latitude_deg: Geodetic latitude on WGS84, in degrees, range [-90, 90].
        longitude_deg: Geodetic longitude on WGS84, normalised to [-180, 180].
        altitude_km: Height above the WGS84 ellipsoid surface, in kilometres.
        eop_degraded: True when the conversion used Skyfield's bundled
            timescale rather than fresh IERS data. Accuracy in this mode is
            typically still <10 m for current-era times; degrades for
            predictions past the bundled-data horizon (~1 year). Downstream
            users with strict accuracy requirements (low-elevation pass
            geometry, sub-metre tracking, long-horizon predictions) should
            check this flag.
    """

    time_utc: datetime
    latitude_deg: float
    longitude_deg: float
    altitude_km: float
    eop_degraded: bool


class CoordinateConverter:
    """Convert TEME state vectors to WGS84 ground positions.

    Holds a cached Skyfield :class:`Timescale` (constructing one is expensive;
    reusing it across many conversions matters once we start propagating long
    ground tracks). On construction the converter tries to load fresh IERS
    Earth Orientation data and falls back to Skyfield's bundled timescale on
    any failure, logging a warning and setting :attr:`eop_degraded`.
    """

    def __init__(
        self,
        config: Config,
        *,
        timescale: Optional[Timescale] = None,
        eop_degraded: bool = False,
    ) -> None:
        """Construct a converter.

        Args:
            config: Active configuration. ``cache_dir`` is reused to hold
                Skyfield's downloaded EOP files alongside the TLE cache.
            timescale: Optional pre-built Skyfield :class:`Timescale` — the
                test seam. When supplied, no network access is attempted.
            eop_degraded: Used together with ``timescale`` to declare whether
                the injected timescale is the bundled (degraded) one. Ignored
                when ``timescale`` is ``None``.
        """
        self._config = config
        if timescale is not None:
            self._timescale = timescale
            self._eop_degraded = eop_degraded
        else:
            self._timescale, self._eop_degraded = self._load_timescale()

    @property
    def eop_degraded(self) -> bool:
        """Whether this converter is operating on bundled (offline) EOP data."""
        return self._eop_degraded

    @property
    def timescale(self) -> Timescale:
        """The loaded Skyfield Timescale.

        Intentionally public — downstream modules (notably
        :mod:`sat_tracker.passes`) reuse this Timescale rather than
        re-downloading EOP data. The shared object also makes the
        :attr:`eop_degraded` flag consistent across modules.
        """
        return self._timescale

    def teme_to_ground(self, state: StateVector) -> GroundPosition:
        """Convert a TEME state vector to a WGS84 ground position.

        Args:
            state: Satellite state in the TEME frame at some UTC instant.

        Returns:
            A :class:`GroundPosition` evaluated at the same UTC instant.
        """
        ecef = self.teme_to_ecef(state)

        lat_deg, lon_deg, alt_km = _ecef_to_geodetic(
            ecef.x_km, ecef.y_km, ecef.z_km
        )
        # Normalise longitude into the canonical [-180, 180] range.
        lon_deg = ((lon_deg + 180.0) % 360.0) - 180.0

        return GroundPosition(
            time_utc=state.time_utc,
            latitude_deg=lat_deg,
            longitude_deg=lon_deg,
            altitude_km=alt_km,
            eop_degraded=self._eop_degraded,
        )

    def teme_to_ecef(self, state: StateVector) -> EcefPosition:
        """Convert a TEME state vector to an ECEF (ITRF) Cartesian position.

        The intermediate result of :meth:`teme_to_ground`, exposed for the 3D
        renderer which needs the satellite's actual 3D position in the
        Earth-fixed frame rather than its sub-satellite ellipsoid point.

        Args:
            state: Satellite state in the TEME frame at some UTC instant.

        Returns:
            An :class:`EcefPosition` at the same UTC instant.
        """
        t = self._timescale.from_datetime(state.time_utc)
        r_teme = np.array(state.position_km, dtype=float)
        v_teme = np.array(state.velocity_km_s, dtype=float)
        # Polar motion (xp, yp) defaults to zero — see module docstring.
        r_itrf, _v_itrf = TEME_to_ITRF(t.ut1, r_teme, v_teme)
        return EcefPosition(
            time_utc=state.time_utc,
            x_km=float(r_itrf[0]),
            y_km=float(r_itrf[1]),
            z_km=float(r_itrf[2]),
            eop_degraded=self._eop_degraded,
        )

    def _load_timescale(self) -> tuple[Timescale, bool]:
        cache_dir = self._config.cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        loader = Loader(str(cache_dir))
        try:
            ts = loader.timescale(builtin=False)
        except Exception as exc:  # network / parse / filesystem
            logger.warning(
                "EOP refresh failed (%s); falling back to Skyfield bundled "
                "timescale. Subsequent GroundPosition values have "
                "eop_degraded=True. Accuracy: typically <10m for current-era "
                "times; may exceed 100m for predictions far past the bundled-"
                "data horizon (~1 year past the Skyfield release).",
                exc,
            )
            return loader.timescale(builtin=True), True

        logger.debug("Loaded fresh IERS timescale into %s", cache_dir)
        return ts, False


def _ecef_to_geodetic(
    x_km: float, y_km: float, z_km: float
) -> tuple[float, float, float]:
    """ECEF Cartesian -> WGS84 geodetic latitude, longitude, altitude.

    Bowring 1976 closed-form approximation. Sub-millimetre error for any
    altitude of practical interest; non-iterative.

    Args:
        x_km, y_km, z_km: ECEF (ITRF) Cartesian coordinates, in kilometres.

    Returns:
        ``(latitude_deg, longitude_deg, altitude_km)``. Longitude is in the
        raw ``atan2`` range ``[-180, 180]`` and may need re-normalisation
        depending on caller conventions.
    """
    a = _WGS84_A_KM
    f = _WGS84_F
    b = a * (1.0 - f)
    e2 = 2.0 * f - f * f
    e_prime_sq = (a * a - b * b) / (b * b)

    p = math.hypot(x_km, y_km)
    lon = math.atan2(y_km, x_km)

    # Polar singularity: longitude is undefined; return 0 by convention.
    if p < 1e-9:
        lat = math.copysign(math.pi / 2.0, z_km)
        alt = abs(z_km) - b
        return math.degrees(lat), 0.0, alt

    theta = math.atan2(z_km * a, p * b)
    sin_theta_cubed = math.sin(theta) ** 3
    cos_theta_cubed = math.cos(theta) ** 3

    lat = math.atan2(
        z_km + e_prime_sq * b * sin_theta_cubed,
        p - e2 * a * cos_theta_cubed,
    )

    sin_lat = math.sin(lat)
    n_radius = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
    alt = p / math.cos(lat) - n_radius

    return math.degrees(lat), math.degrees(lon), alt
