"""Smoke tests for the cartopy ground-track renderer.

Verifies the renderer produces a non-trivial image file at the expected
dimensions. Pixel content is not validated — that would require either a
golden-image diff (brittle across cartopy/matplotlib versions) or a
human-in-the-loop review (handled separately during portfolio iteration).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from skyfield.api import load

from sat_tracker.config import Config
from sat_tracker.coordinates import CoordinateConverter
from sat_tracker.tle_fetcher import Tle
from sat_tracker.visualization.common import (
    default_window_seconds,
    precompute_track,
)

# Skip the whole module if the [viz] extras aren't installed — the
# renderer module imports them at the top level via defer-import inside
# the render function, so the import itself is cheap, but we want to keep
# CI green on a no-extras environment.
cartopy = pytest.importorskip("cartopy")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")  # headless backend for CI / local pytest

from sat_tracker.visualization.ground_track import render_ground_track


_NEAR_EPOCH_UTC = datetime(2024, 1, 1, 14, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def bundled_timescale():
    return load.timescale(builtin=True)


@pytest.fixture
def config(tmp_path: Path) -> Config:
    return Config(
        cache_dir=tmp_path,
        cache_ttl_hours=6,
        tle_source_url="http://test.invalid/",
        user_agent="sat-tracker-tests/1.0",
        http_timeout_seconds=5,
        log_level="DEBUG",
        min_elevation_deg=10.0,
    )


@pytest.fixture
def converter(config: Config, bundled_timescale) -> CoordinateConverter:
    return CoordinateConverter(
        config, timescale=bundled_timescale, eop_degraded=True
    )


@pytest.fixture
def iss(iss_tle: tuple[str, str, str]) -> Tle:
    name, line1, line2 = iss_tle
    return Tle(name=name, line1=line1, line2=line2)


@pytest.fixture
def iss_track(iss: Tle, converter: CoordinateConverter):
    return precompute_track(
        iss,
        converter,
        start_utc=_NEAR_EPOCH_UTC,
        duration_seconds=default_window_seconds(iss),
    )


def _png_dimensions(path: Path) -> tuple[int, int]:
    """Return (width, height) of a PNG by parsing its IHDR chunk.

    Avoids a Pillow dependency just for this assertion. PNG layout: 8-byte
    signature, then 4-byte length, 4-byte type ('IHDR'), 4-byte width,
    4-byte height, all big-endian.
    """
    raw = path.read_bytes()
    assert raw[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
    width = int.from_bytes(raw[16:20], "big")
    height = int.from_bytes(raw[20:24], "big")
    return width, height


def test_render_ground_track_writes_png(iss_track, tmp_path: Path) -> None:
    out = tmp_path / "track.png"
    written = render_ground_track(iss_track, out)
    assert written == out
    assert out.exists()
    # Anything under ~10 KB would mean a blank or near-blank image.
    assert out.stat().st_size > 10_000
    width, height = _png_dimensions(out)
    # figsize=(14, 7) at dpi=150 with bbox_inches="tight" — exact pixel
    # dimensions vary with backend trim; just sanity-check the order of
    # magnitude and aspect ratio.
    assert width > 1000
    assert height > 400
    assert width > height  # wider than tall (PlateCarree world map)


def test_render_ground_track_creates_parent_dir(iss_track, tmp_path: Path) -> None:
    out = tmp_path / "nested" / "deeper" / "track.png"
    render_ground_track(iss_track, out)
    assert out.exists()


def test_render_ground_track_multi_track(
    iss_track, iss: Tle, converter: CoordinateConverter, tmp_path: Path
) -> None:
    """Two tracks on one figure: must not blow up and must produce a file."""
    second = precompute_track(
        iss,
        converter,
        start_utc=_NEAR_EPOCH_UTC,
        duration_seconds=default_window_seconds(iss) / 2,
    )
    out = tmp_path / "multi.png"
    render_ground_track([iss_track, second], out)
    assert out.stat().st_size > 10_000


def test_render_ground_track_empty_track_raises(
    iss: Tle, converter: CoordinateConverter, tmp_path: Path
) -> None:
    from sat_tracker.visualization.common import Track

    empty = Track(
        catalog_number=99999,
        name="EMPTY",
        samples=(),
        tle_epoch_utc=_NEAR_EPOCH_UTC,
    )
    with pytest.raises(ValueError, match="no samples"):
        render_ground_track(empty, tmp_path / "should_not_exist.png")


def test_render_ground_track_empty_list_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one"):
        render_ground_track([], tmp_path / "should_not_exist.png")


def test_render_ground_track_custom_title(iss_track, tmp_path: Path) -> None:
    out = tmp_path / "titled.png"
    render_ground_track(iss_track, out, title="Custom Title For Test")
    assert out.exists()
    # Just verify the call accepts the override; pixel content not asserted.


def test_render_ground_track_now_marker_outside_window(
    iss_track, tmp_path: Path
) -> None:
    """A 'now' time well outside the track window must be silently skipped
    rather than place a misleading pin on the map."""
    far_future = datetime(2099, 1, 1, tzinfo=timezone.utc)
    out = tmp_path / "no_now.png"
    render_ground_track(iss_track, out, current_time_utc=far_future)
    # Renderer just drops the marker; file still produced.
    assert out.stat().st_size > 10_000
