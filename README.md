# sgp4-satellite-tracker

Real-time satellite tracking and orbit propagation using TLE data from CelesTrak and the SGP4 model, with 3D visualization and pass prediction.

## Status

Stage 1 (MVP scaffolding) — in progress.

## Requirements

- Python 3.10 or newer

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Usage

```bash
sat-tracker                  # current ISS sub-satellite point
sat-tracker --catnr 20580    # Hubble
sat-tracker --watch 5        # poll every 5 seconds, append output
sat-tracker --verbose        # also print TEME state vectors
sat-tracker --help           # full option list

python -m sat_tracker        # equivalent if the console script isn't on PATH
```

## Configuration

All configuration is via environment variables. All have sensible defaults.

| Variable | Default | Purpose |
|---|---|---|
| `SAT_TRACKER_CACHE_DIR` | `./data` | Directory for cached TLE files |
| `SAT_TRACKER_CACHE_TTL_HOURS` | `6` | Refetch a cached TLE if it is older than this |
| `SAT_TRACKER_TLE_SOURCE_URL` | `https://celestrak.org/NORAD/elements/gp.php` | CelesTrak GP query endpoint |
| `SAT_TRACKER_USER_AGENT` | `sat-tracker/0.1 (+github.com/myusername/sat-tracker)` | Sent on every HTTP request — CelesTrak asks clients to identify themselves |
| `SAT_TRACKER_HTTP_TIMEOUT_SECONDS` | `10` | HTTP request timeout |
| `SAT_TRACKER_LOG_LEVEL` | `INFO` | Root log level (`DEBUG`/`INFO`/`WARNING`/`ERROR`) |

## Project structure

```
src/sat_tracker/
    config.py         # env-var driven configuration
    tle_fetcher.py    # CelesTrak fetch + local cache
    propagator.py     # SGP4 propagation
    coordinates.py    # TEME -> lat/lon/altitude
    cli.py            # entry point
tests/                # pytest suite
data/                 # local TLE cache (gitignored)
```

## Development

```bash
pytest
```

## Roadmap

- Stage 1 — MVP: fetch, propagate, print ISS position (current).
- Stage 2 — Visualization: 2D ground tracks and 3D Earth view.
- Stage 3 — Pass prediction.
- Stage 4 — Web dashboard.

## License

See [LICENSE](LICENSE).
