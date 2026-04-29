"""Smoke tests for the plotly 3D orbit renderer.

Verifies the renderer produces non-trivial HTML / PNG files at the
expected dimensions. 3D geometry is not validated from rendered
output — it's validated upstream by ``test_visualization_common``
and ``test_coordinates``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from skyfield.api import load

from sat_tracker.config import Config
from sat_tracker.coordinates import CoordinateConverter
from sat_tracker.passes import GroundStation
from sat_tracker.tle_fetcher import Tle
from sat_tracker.visualization.common import (
    Orbit3D,
    default_window_seconds,
    precompute_orbit,
)

# Skip the whole module if [viz] extras aren't installed.
pytest.importorskip("plotly")
pytest.importorskip("kaleido")

from sat_tracker.visualization.orbit_3d import render_orbit_3d


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
def iss_orbit(iss: Tle, converter: CoordinateConverter) -> Orbit3D:
    return precompute_orbit(
        iss,
        converter,
        start_utc=_NEAR_EPOCH_UTC,
        duration_seconds=default_window_seconds(iss),
    )


def _png_dimensions(path: Path) -> tuple[int, int]:
    raw = path.read_bytes()
    assert raw[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
    width = int.from_bytes(raw[16:20], "big")
    height = int.from_bytes(raw[20:24], "big")
    return width, height


def test_render_orbit_3d_writes_html(iss_orbit: Orbit3D, tmp_path: Path) -> None:
    out = tmp_path / "orbit.html"
    written = render_orbit_3d(iss_orbit, out)
    assert written == out
    assert out.exists()
    # plotly HTML w/ embedded data + CDN reference: typically >100 KB.
    assert out.stat().st_size > 100_000
    head = out.read_bytes()[:64]
    assert b"<!DOCTYPE html>" in head or b"<html" in head


def test_render_orbit_3d_writes_png(iss_orbit: Orbit3D, tmp_path: Path) -> None:
    out = tmp_path / "orbit.png"
    written = render_orbit_3d(iss_orbit, out)
    assert written == out
    assert out.exists()
    assert out.stat().st_size > 10_000
    width, height = _png_dimensions(out)
    assert width > 800
    assert height > 600


def test_render_orbit_3d_default_extension_is_html(
    iss_orbit: Orbit3D, tmp_path: Path
) -> None:
    """Output without a suffix should default to HTML, not error."""
    out = tmp_path / "no_suffix"
    written = render_orbit_3d(iss_orbit, out)
    assert written.suffix == ".html"
    assert written.exists()


def test_render_orbit_3d_creates_parent_dir(
    iss_orbit: Orbit3D, tmp_path: Path
) -> None:
    out = tmp_path / "nested" / "deeper" / "orbit.html"
    render_orbit_3d(iss_orbit, out)
    assert out.exists()


def test_render_orbit_3d_multi_satellite(
    iss_orbit: Orbit3D,
    iss: Tle,
    converter: CoordinateConverter,
    tmp_path: Path,
) -> None:
    """Two orbits on one figure: must not blow up and must produce a file."""
    second = precompute_orbit(
        iss,
        converter,
        start_utc=_NEAR_EPOCH_UTC,
        duration_seconds=default_window_seconds(iss) / 2,
    )
    out = tmp_path / "multi.html"
    render_orbit_3d([iss_orbit, second], out)
    assert out.stat().st_size > 100_000


def test_render_orbit_3d_with_ground_station(
    iss_orbit: Orbit3D, tmp_path: Path
) -> None:
    station = GroundStation(
        latitude_deg=40.59,
        longitude_deg=-105.08,
        altitude_km=1.5,
        name="Fort Collins",
    )
    out = tmp_path / "with_gs.png"
    render_orbit_3d(
        iss_orbit,
        out,
        current_time_utc=_NEAR_EPOCH_UTC,
        ground_station=station,
    )
    assert out.stat().st_size > 10_000


def test_render_orbit_3d_static_strips_animation(
    iss_orbit: Orbit3D, tmp_path: Path
) -> None:
    """time_slider=True with PNG output must not error — slider is stripped
    automatically for static exports."""
    out = tmp_path / "no_slider.png"
    render_orbit_3d(
        iss_orbit,
        out,
        current_time_utc=_NEAR_EPOCH_UTC,
        time_slider=True,  # caller asks for it; renderer drops it for PNG
    )
    assert out.exists()


def test_render_orbit_3d_empty_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one"):
        render_orbit_3d([], tmp_path / "should_not_exist.html")


def test_render_orbit_3d_no_samples_raises(tmp_path: Path) -> None:
    empty = Orbit3D(
        catalog_number=99999,
        name="EMPTY",
        samples=(),
        tle_epoch_utc=_NEAR_EPOCH_UTC,
        frame="ecef",
    )
    with pytest.raises(ValueError, match="no samples"):
        render_orbit_3d(empty, tmp_path / "should_not_exist.html")
