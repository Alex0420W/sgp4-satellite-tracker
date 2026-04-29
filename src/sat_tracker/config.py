"""Configuration loaded from environment variables.

All settings have sensible defaults so the application runs out of the box.
Override any value by exporting the matching ``SAT_TRACKER_*`` environment
variable before invoking the CLI or importing the package.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

_DEFAULT_CACHE_DIR = Path("data")
_DEFAULT_CACHE_TTL_HOURS = 6
_DEFAULT_TLE_SOURCE_URL = "https://celestrak.org/NORAD/elements/gp.php"
_DEFAULT_USER_AGENT = "sat-tracker/0.1 (+github.com/Alex0420W/sgp4-satellite-tracker)"
_DEFAULT_HTTP_TIMEOUT_SECONDS = 10
_DEFAULT_LOG_LEVEL = "INFO"


@dataclass(frozen=True)
class Config:
    """Resolved runtime configuration.

    Attributes:
        cache_dir: Directory where fetched TLE files are cached.
        cache_ttl_hours: Maximum age of a cached TLE before it is refetched.
        tle_source_url: CelesTrak GP query endpoint.
        user_agent: HTTP ``User-Agent`` header sent on every outbound request.
        http_timeout_seconds: Per-request HTTP timeout.
        log_level: Root logger level name (e.g. ``"INFO"``).
    """

    cache_dir: Path
    cache_ttl_hours: int
    tle_source_url: str
    user_agent: str
    http_timeout_seconds: int
    log_level: str


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(
            f"Environment variable {name}={raw!r} is not a valid integer."
        ) from exc


def load_config() -> Config:
    """Build a :class:`Config` from the current process environment.

    Returns:
        A frozen :class:`Config` instance reflecting the current environment.

    Raises:
        ValueError: If an integer-valued env var is set to a non-numeric value.
    """
    return Config(
        cache_dir=Path(
            os.environ.get("SAT_TRACKER_CACHE_DIR", str(_DEFAULT_CACHE_DIR))
        ),
        cache_ttl_hours=_env_int(
            "SAT_TRACKER_CACHE_TTL_HOURS", _DEFAULT_CACHE_TTL_HOURS
        ),
        tle_source_url=os.environ.get(
            "SAT_TRACKER_TLE_SOURCE_URL", _DEFAULT_TLE_SOURCE_URL
        ),
        user_agent=os.environ.get("SAT_TRACKER_USER_AGENT", _DEFAULT_USER_AGENT),
        http_timeout_seconds=_env_int(
            "SAT_TRACKER_HTTP_TIMEOUT_SECONDS", _DEFAULT_HTTP_TIMEOUT_SECONDS
        ),
        log_level=os.environ.get("SAT_TRACKER_LOG_LEVEL", _DEFAULT_LOG_LEVEL).upper(),
    )


def configure_logging(config: Config) -> None:
    """Apply ``config.log_level`` to the root logger.

    Idempotent — safe to call from both library code and the CLI entry point.

    Args:
        config: The active configuration.
    """
    logging.basicConfig(
        level=config.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
