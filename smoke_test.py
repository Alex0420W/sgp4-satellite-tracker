from datetime import datetime, timezone
from sat_tracker.config import load_config, configure_logging
from sat_tracker.tle_fetcher import TleFetcher
from sat_tracker.propagator import propagate
from sat_tracker.coordinates import CoordinateConverter

cfg = load_config()
configure_logging(cfg)

with TleFetcher(cfg) as f:
    tle = f.fetch(25544)

converter = CoordinateConverter(cfg)
state = propagate(tle, datetime.now(timezone.utc))
ground = converter.teme_to_ground(state)

print(f"ISS at {ground.time_utc.isoformat()}:")
print(f"  lat:  {ground.latitude_deg:+.4f} deg")
print(f"  lon:  {ground.longitude_deg:+.4f} deg")
print(f"  alt:  {ground.altitude_km:.1f} km")
print(f"  EOP:  {'degraded (bundled)' if ground.eop_degraded else 'fresh IERS data'}")