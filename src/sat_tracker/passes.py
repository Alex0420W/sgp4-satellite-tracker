"""Pass prediction: when is a satellite visible from a ground station?

Given a TLE and a ground station, predicts the satellite's upcoming passes
above a configurable minimum-elevation threshold within a search window.
For each pass we report AOS / max-elevation / LOS times and azimuths plus
a sunlit-vs-eclipsed visibility flag.

Two SGP4 propagation paths in this codebase
-------------------------------------------
This module uses Skyfield's high-level :meth:`EarthSatellite.find_events`
for AOS / culmination / LOS root-finding. :mod:`sat_tracker.propagator`
uses the ``sgp4`` library directly (``Satrec.sgp4()``) for single-instant
TEME state evaluation.

Both paths produce identical positions — Skyfield wraps the same ``sgp4``
library internally — but expose different APIs. We use ``propagator.py``
for the "current position" CLI because it gives us TEME state vectors
directly. We use Skyfield's ``find_events`` here because reimplementing
robust AOS/LOS root-finding (with proper handling of short passes,
high-latitude circumpolar visibility, and numerical edge cases at the
elevation threshold) is significantly more work than reimplementing
single-instant propagation, and Skyfield's implementation is battle-tested.

This is a deliberate architectural choice, not an oversight; see also the
inline comment at the ``find_events`` call site and the architecture
section of README.md.

Sunlit calculation
------------------
The sunlit flag uses Skyfield's ``is_sunlit``, which requires a planetary
ephemeris (de421.bsp, ~16 MB). On ephemeris load failure we set
``sunlit = None`` and continue — pass *timing* (AOS/LOS/max-elevation) is
unaffected; only the visibility flag is degraded. The fallback warning
log explicitly says so to prevent users from misinterpreting which data
is reliable.

Geostationary / deep-space satellites
-------------------------------------
Mean motion < 1.5 rev/day satellites do not "pass" — they sit at fixed
alt/az relative to a fixed observer (modulo small libration). We detect
this case before searching ``find_events`` and return an empty pass list
with a warning log that includes the actual mean motion so log-readers
can verify the gate's decision without re-running.

Pass prediction caching
-----------------------
Pass prediction takes ~150-300 ms per call for a 24h window. The
complexity of correct cache invalidation (which depends on TLE freshness,
station coordinates, window length, and elevation threshold) is not
justified by that latency for the MVP. If a real consumer (e.g. a web
dashboard hitting prediction on every page load) appears, this is the
place to revisit.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from skyfield.api import EarthSatellite, Loader, wgs84

from sat_tracker.config import Config
from sat_tracker.coordinates import CoordinateConverter
from sat_tracker.tle_fetcher import Tle

logger = logging.getLogger(__name__)


# Mean motions below this threshold are considered deep-space / GEO and we
# refuse to search for passes. 1.5 rev/day comfortably brackets GEO (~1.0)
# and most GTO/MEO orbits without excluding the LEO regime where pass
# prediction is meaningful.
_MIN_LEO_MEAN_MOTION_REV_PER_DAY: float = 1.5

# Skyfield's find_events emits these event-type integers in order along
# each pass. Pinned as constants to make the grouping logic readable.
_EVENT_RISE: int = 0
_EVENT_CULMINATE: int = 1
_EVENT_SET: int = 2

# Sentinel for the ``ephemeris`` constructor argument so that explicit
# ``None`` (meaning "don't load anything") is distinguishable from "user
# didn't pass anything; auto-load from disk/network."
_AUTO_LOAD_EPHEMERIS: Any = object()


@dataclass(frozen=True)
class GroundStation:
    """Observer location for pass prediction (WGS84 coordinates).

    Attributes:
        latitude_deg: Geodetic latitude on WGS84, in degrees, range [-90, 90].
        longitude_deg: Geodetic longitude on WGS84, in degrees, [-180, 180].
        altitude_km: Height above the WGS84 ellipsoid surface, in kilometres.
        name: Optional human-readable station name. ``None`` for ad-hoc
            lat/lon lookups; meaningful when the future named-station
            registry lands.
    """

    latitude_deg: float
    longitude_deg: float
    altitude_km: float
    name: Optional[str] = None

    def __post_init__(self) -> None:
        if not -90.0 <= self.latitude_deg <= 90.0:
            raise ValueError(
                f"latitude_deg={self.latitude_deg} outside range [-90, 90]"
            )
        if not -180.0 <= self.longitude_deg <= 180.0:
            raise ValueError(
                f"longitude_deg={self.longitude_deg} outside range [-180, 180]"
            )


@dataclass(frozen=True)
class Pass:
    """A single satellite pass above a ground station.

    All time fields are timezone-aware UTC datetimes — same discipline as
    :class:`~sat_tracker.propagator.StateVector` and
    :class:`~sat_tracker.coordinates.GroundPosition`.

    Attributes:
        satellite_catnr: NORAD catalog number from the source TLE.
        satellite_name: Satellite name from the source TLE (may be ``None``).
        aos_utc: Acquisition of Signal — instant elevation crosses the
            minimum threshold on the way up.
        los_utc: Loss of Signal — instant elevation crosses the threshold
            on the way down.
        max_elevation_deg: Peak topocentric elevation reached in this pass,
            in degrees.
        max_elevation_time_utc: Instant of peak elevation.
        azimuth_at_aos_deg: Compass azimuth (0=N, 90=E) at AOS.
        azimuth_at_max_deg: Compass azimuth at peak elevation.
        azimuth_at_los_deg: Compass azimuth at LOS.
        sunlit: ``True`` if satellite is in sunlight at peak elevation;
            ``False`` if in Earth's shadow; ``None`` if undetermined
            (planetary ephemeris unavailable). Pass *timing* values are
            independent of this flag.
        eop_degraded: Constant for all passes within a single prediction
            call — reflects the predictor's loaded :class:`Timescale` at
            construction time. The Timescale is loaded once and used for
            the entire window; there is no mid-window EOP refresh.
    """

    satellite_catnr: int
    satellite_name: Optional[str]
    aos_utc: datetime
    los_utc: datetime
    max_elevation_deg: float
    max_elevation_time_utc: datetime
    azimuth_at_aos_deg: float
    azimuth_at_max_deg: float
    azimuth_at_los_deg: float
    sunlit: Optional[bool]
    eop_degraded: bool


class PassPredictor:
    """Predict satellite passes over a ground station.

    Reuses the :class:`Timescale` already loaded by
    :class:`~sat_tracker.coordinates.CoordinateConverter` (avoids
    re-downloading EOP data) and optionally loads a planetary ephemeris
    for the sunlit flag. Falls back gracefully on ephemeris failure: pass
    timing is unaffected; ``sunlit`` becomes ``None``.
    """

    def __init__(
        self,
        config: Config,
        converter: CoordinateConverter,
        *,
        ephemeris: Any = _AUTO_LOAD_EPHEMERIS,
        min_elevation_deg: Optional[float] = None,
    ) -> None:
        """Construct a predictor.

        Args:
            config: Active configuration. ``cache_dir`` is reused for
                Skyfield's ephemeris file.
            converter: An already-constructed coordinate converter — its
                :attr:`~CoordinateConverter.timescale` and
                :attr:`~CoordinateConverter.eop_degraded` are reused.
            ephemeris: Test seam. Default behaviour (no value passed) is
                to auto-load ``de421.bsp`` from ``cache_dir`` via
                Skyfield's Loader. Pass an already-loaded ephemeris to
                reuse it; pass ``None`` explicitly to skip loading and
                force the sunlit-fallback path.
            min_elevation_deg: Override for the elevation threshold. When
                ``None``, defaults to ``config.min_elevation_deg``.
        """
        self._config = config
        self._converter = converter
        self._timescale = converter.timescale
        self._min_elevation_deg = (
            config.min_elevation_deg
            if min_elevation_deg is None
            else min_elevation_deg
        )
        if ephemeris is _AUTO_LOAD_EPHEMERIS:
            self._ephemeris = self._try_load_ephemeris()
        else:
            self._ephemeris = ephemeris

    @property
    def min_elevation_deg(self) -> float:
        """Active elevation threshold in degrees."""
        return self._min_elevation_deg

    def _try_load_ephemeris(self) -> Any:
        cache_dir = self._config.cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        loader = Loader(str(cache_dir))
        try:
            eph = loader("de421.bsp")
            logger.debug("Loaded planetary ephemeris de421.bsp")
            return eph
        except Exception as exc:
            logger.warning(
                "Planetary ephemeris (de421.bsp) load failed (%s). "
                "Pass *timing* (AOS, LOS, max-elevation) is UNAFFECTED — "
                "only the sunlit/eclipsed visibility flag will be set to "
                "None on returned Pass objects.",
                exc,
            )
            return None

    def predict_passes(
        self,
        tle: Tle,
        station: GroundStation,
        *,
        start_utc: datetime,
        hours: float,
    ) -> list[Pass]:
        """Predict passes within ``[start_utc, start_utc + hours]``.

        Args:
            tle: Validated TLE.
            station: Observer position.
            start_utc: Start of the search window. Must be timezone-aware.
                Non-UTC timezones are converted internally.
            hours: Window length in hours. Must be positive.

        Returns:
            A list of :class:`Pass` objects ordered by AOS time. Empty if
            no complete passes (rise + culminate + set within window) are
            found, or if the satellite is in the deep-space / GEO regime.

        Raises:
            ValueError: If ``start_utc`` is naive, or ``hours`` is non-positive.
        """
        if start_utc.tzinfo is None:
            raise ValueError(
                "start_utc must be timezone-aware (UTC). Naive datetimes "
                "are rejected to avoid local-time-as-UTC silent errors."
            )
        if hours <= 0:
            raise ValueError(f"hours must be positive, got {hours}")
        start_utc = start_utc.astimezone(timezone.utc)
        end_utc = start_utc + timedelta(hours=hours)

        # Mean motion gate: refuse pass prediction for GEO / deep-space.
        # TLE line 2 positions 52..62 hold mean motion in revolutions/day.
        mean_motion = float(tle.line2[52:63])
        if mean_motion < _MIN_LEO_MEAN_MOTION_REV_PER_DAY:
            logger.warning(
                "satellite catnr=%d has mean motion %.4f rev/day "
                "(< %.1f gate) — pass prediction skipped (GEO / deep-space "
                "satellites do not rise or set from a fixed observer).",
                tle.catalog_number,
                mean_motion,
                _MIN_LEO_MEAN_MOTION_REV_PER_DAY,
            )
            return []

        # Use Skyfield's wrapped SGP4 here for AOS/LOS root-finding —
        # see module docstring on the two SGP4 paths in this codebase.
        sat = EarthSatellite(
            tle.line1, tle.line2, tle.name or "", self._timescale
        )
        observer = wgs84.latlon(
            station.latitude_deg,
            station.longitude_deg,
            elevation_m=station.altitude_km * 1000.0,
        )

        t0 = self._timescale.from_datetime(start_utc)
        t1 = self._timescale.from_datetime(end_utc)
        times, events = sat.find_events(
            observer, t0, t1, altitude_degrees=self._min_elevation_deg
        )

        return self._group_events_into_passes(
            sat=sat,
            observer=observer,
            tle=tle,
            times=times,
            events=events,
        )

    def _group_events_into_passes(
        self,
        *,
        sat: EarthSatellite,
        observer: Any,
        tle: Tle,
        times: Any,
        events: Any,
    ) -> list[Pass]:
        passes: list[Pass] = []
        difference = sat - observer

        i = 0
        n = len(events)
        while i < n:
            # Skip any leading partial pass (window opens mid-pass).
            if events[i] != _EVENT_RISE:
                i += 1
                continue
            # Need full (rise, culminate, set) triple. If the window cuts
            # off before the satellite sets, drop the partial trailing pass.
            if i + 2 >= n:
                break
            if events[i + 1] != _EVENT_CULMINATE or events[i + 2] != _EVENT_SET:
                logger.warning(
                    "unexpected Skyfield event sequence at index %d: %s "
                    "— skipping",
                    i,
                    list(events[i : i + 3]),
                )
                i += 1
                continue

            t_rise = times[i]
            t_max = times[i + 1]
            t_set = times[i + 2]

            alt_max, az_max, _ = difference.at(t_max).altaz()
            _, az_rise, _ = difference.at(t_rise).altaz()
            _, az_set, _ = difference.at(t_set).altaz()

            sunlit = self._compute_sunlit(sat, t_max, tle)

            passes.append(
                Pass(
                    satellite_catnr=tle.catalog_number,
                    satellite_name=tle.name,
                    aos_utc=_skyfield_to_utc(t_rise),
                    los_utc=_skyfield_to_utc(t_set),
                    max_elevation_deg=float(alt_max.degrees),
                    max_elevation_time_utc=_skyfield_to_utc(t_max),
                    azimuth_at_aos_deg=float(az_rise.degrees),
                    azimuth_at_max_deg=float(az_max.degrees),
                    azimuth_at_los_deg=float(az_set.degrees),
                    sunlit=sunlit,
                    eop_degraded=self._converter.eop_degraded,
                )
            )
            i += 3

        return passes

    def _compute_sunlit(
        self, sat: EarthSatellite, t_max: Any, tle: Tle
    ) -> Optional[bool]:
        if self._ephemeris is None:
            return None
        try:
            return bool(sat.at(t_max).is_sunlit(self._ephemeris))
        except Exception as exc:
            logger.warning(
                "is_sunlit() failed at %s for catnr=%d: %s",
                t_max.utc_iso(),
                tle.catalog_number,
                exc,
            )
            return None


def _skyfield_to_utc(t: Any) -> datetime:
    """Convert a Skyfield Time to a tz-aware UTC datetime.

    ``Time.utc_datetime()`` may return naive or aware depending on Skyfield
    version. ``replace(tzinfo=timezone.utc)`` is a no-op when the datetime
    is already aware-UTC and correct when it isn't.
    """
    return t.utc_datetime().replace(tzinfo=timezone.utc)
