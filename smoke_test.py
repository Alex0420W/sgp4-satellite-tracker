from datetime import datetime, timezone
from sat_tracker.config import load_config, configure_logging
from sat_tracker.tle_fetcher import TleFetcher
from sat_tracker.propagator import propagate
import math

cfg = load_config()
configure_logging(cfg)

with TleFetcher(cfg) as f:
    tle = f.fetch(25544)

now = datetime.now(timezone.utc)
state = propagate(tle, now)

print(f"At {state.time_utc.isoformat()}:")
print(f"  position TEME (km):   {state.position_km}")
print(f"  velocity TEME (km/s): {state.velocity_km_s}")

x, y, z = state.position_km
print(f"  altitude: {math.sqrt(x*x + y*y + z*z) - 6378:.1f} km")