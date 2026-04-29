"""Smoke tests for the plotly figure builders.

The builders in ``visualization/figures.py`` are the in-memory entry
point used by the Streamlit dashboard — they must not write anything to
disk and must return a ``plotly.graph_objects.Figure``. The CLI-facing
``render_*`` shims are tested separately in ``test_ground_track``,
``test_orbit_3d``, etc.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import pytest
from skyfield.api import load

from sat_tracker.config import Config
from sat_tracker.coordinates import CoordinateConverter
from sat_tracker.passes import GroundStation
from sat_tracker.tle_fetcher import Tle
from sat_tracker.visualization.common import (
    default_window_seconds,
    precompute_orbit,
    precompute_track,
)

# Skip the whole module if [viz] extras aren't installed.
pytest.importorskip("plotly")

from sat_tracker.visualization.figures import (
    build_ground_track_figure,
    build_orbit_3d_figure,
)


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


@pytest.fixture
def iss_orbit(iss: Tle, converter: CoordinateConverter):
    return precompute_orbit(
        iss,
        converter,
        start_utc=_NEAR_EPOCH_UTC,
        duration_seconds=default_window_seconds(iss),
    )


def _no_files_added(tmp_path: Path):
    """Snapshot tmp_path's contents; assert no new files exist on exit."""
    before = set(os.listdir(tmp_path))

    class _Checker:
        def __enter__(self_inner):
            return self_inner

        def __exit__(self_inner, *args):
            after = set(os.listdir(tmp_path))
            assert after == before, (
                f"figure builder wrote unexpected files: {after - before}"
            )

    return _Checker()


# -- 2D builder ---------------------------------------------------------------


def test_build_ground_track_figure_returns_figure(iss_track) -> None:
    import plotly.graph_objects as go

    fig = build_ground_track_figure(iss_track)
    assert isinstance(fig, go.Figure)
    # Single track: line + start + end traces (no "now" without explicit time).
    assert len(fig.data) >= 3


def test_build_ground_track_figure_does_no_disk_io(
    iss_track, tmp_path: Path
) -> None:
    """The builder must be pure — no disk writes for the dashboard path."""
    with _no_files_added(tmp_path):
        os.chdir(tmp_path)  # in case anything tries to write to cwd
        build_ground_track_figure(iss_track)


def test_build_ground_track_figure_with_now_marker(iss_track) -> None:
    fig = build_ground_track_figure(
        iss_track, current_time_utc=_NEAR_EPOCH_UTC
    )
    # Should have at least one trace named "... now" for the gold star.
    names = [getattr(t, "name", "") or "" for t in fig.data]
    assert any(name.endswith("now") for name in names), names


def test_build_ground_track_figure_multi_track(
    iss_track, iss, converter
) -> None:
    """Two distinct satellites should produce two legendgroups."""
    # Synthesize a second TLE with a different catnr so the labels
    # differ ("ISS [25544]" vs "FAKE [99999]").
    fake_line1 = "1 99999U 24001A   24001.54791667  .00000000  00000-0  00000-0 0  9999"
    fake = Tle(name="FAKE-SAT", line1=fake_line1, line2=iss.line2)
    # Re-use iss_track for the first; build a track for fake.
    second = precompute_track(
        fake,
        converter,
        start_utc=_NEAR_EPOCH_UTC,
        duration_seconds=default_window_seconds(fake) / 2,
    )
    fig = build_ground_track_figure([iss_track, second])
    legendgroups = {
        getattr(t, "legendgroup", None) for t in fig.data
        if getattr(t, "legendgroup", None) is not None
    }
    assert len(legendgroups) >= 2, legendgroups


def test_build_ground_track_figure_empty_raises() -> None:
    with pytest.raises(ValueError, match="at least one"):
        build_ground_track_figure([])


# -- 3D builder ---------------------------------------------------------------


def test_build_orbit_3d_figure_returns_figure(iss_orbit) -> None:
    import plotly.graph_objects as go

    fig = build_orbit_3d_figure(iss_orbit)
    assert isinstance(fig, go.Figure)
    # Earth sphere + coastlines + graticule + orbit line + start + end =
    # at least 6 traces.
    assert len(fig.data) >= 6


def test_build_orbit_3d_figure_does_no_disk_io(
    iss_orbit, tmp_path: Path
) -> None:
    with _no_files_added(tmp_path):
        os.chdir(tmp_path)
        build_orbit_3d_figure(iss_orbit)


def test_build_orbit_3d_figure_with_ground_station(iss_orbit) -> None:
    station = GroundStation(
        latitude_deg=51.4779,
        longitude_deg=-0.0014,
        altitude_km=0.0,
        name="Greenwich",
    )
    fig = build_orbit_3d_figure(
        iss_orbit,
        current_time_utc=_NEAR_EPOCH_UTC,
        ground_station=station,
    )
    names = [getattr(t, "name", "") or "" for t in fig.data]
    assert any("GS:" in name for name in names), names


def test_build_orbit_3d_figure_camera_eye(iss_orbit) -> None:
    """Custom camera_eye must reach the figure layout."""
    fig = build_orbit_3d_figure(
        iss_orbit, camera_eye=(2.5, 0.0, 0.5)
    )
    eye = fig.layout.scene.camera.eye
    assert eye.x == pytest.approx(2.5)
    assert eye.y == pytest.approx(0.0)
    assert eye.z == pytest.approx(0.5)


def test_build_orbit_3d_figure_empty_raises() -> None:
    with pytest.raises(ValueError, match="at least one"):
        build_orbit_3d_figure([])
