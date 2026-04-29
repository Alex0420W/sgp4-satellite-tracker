"""Interactive 3D orbit rendering via plotly.

Companion to the 2D ground-track renderers. Earth is drawn as a WGS84
ellipsoid with a cream land/ocean base, charcoal coastline polylines pulled
from cartopy's already-downloaded Natural Earth assets, and a faint 30°
graticule overlay. Satellite trajectories are 3D polylines in ECEF (or
TEME) Cartesian coordinates with prominent endpoints and a "now" marker
matching the conventions of the 2D renderers.

Plotly is defer-imported inside :func:`render_orbit_3d` so importing this
module does not require the ``[viz]`` extra. cartopy is also defer-imported
purely for shapefile reading.

This module exposes a single public entry point: :func:`render_orbit_3d`.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence, Union

from sat_tracker.passes import GroundStation
from sat_tracker.visualization.common import DEFAULT_TRACK_COLORS, Orbit3D

logger = logging.getLogger(__name__)

# WGS84 defining constants (kilometres). Match coordinates._WGS84_*.
_WGS84_A_KM: float = 6378.137
_WGS84_F: float = 1.0 / 298.257223563
_WGS84_B_KM: float = _WGS84_A_KM * (1.0 - _WGS84_F)

# Earth-sphere appearance tuned to match the 2D ground-track basemap.
_OCEAN_BASE = "#a8c8e6"   # same as cfeature.OCEAN colour in ground_track.py
_COASTLINE_COLOR = "#3a3a3a"
_COASTLINE_WIDTH = 0.7
_COASTLINE_OPACITY = 0.6
_GRATICULE_COLOR = "#444"
_GRATICULE_WIDTH = 0.3
_GRATICULE_OPACITY = 0.18
_GRATICULE_SPACING_DEG = 30

# Sphere mesh density. 73x145 → 5° latitude × 2.5° longitude — smooth at
# any reasonable zoom, ~10K vertices is well within plotly's comfort.
_SPHERE_LAT_STEPS = 73
_SPHERE_LON_STEPS = 145

_HOVER_TIME_FORMAT = "%Y-%m-%d %H:%M:%S UTC"


@dataclass(frozen=True)
class _Camera:
    """Plotly scene camera position in scene-data units (1.0 ≈ Earth radius)."""

    eye_x: float
    eye_y: float
    eye_z: float


_DEFAULT_CAMERA = _Camera(eye_x=1.6, eye_y=1.6, eye_z=0.8)


def render_orbit_3d(
    orbits: Union[Orbit3D, Sequence[Orbit3D]],
    output_path: Union[str, Path],
    *,
    title: Optional[str] = None,
    colors: Optional[Sequence[str]] = None,
    line_width: float = 4.0,
    current_time_utc: Optional[datetime] = None,
    ground_station: Optional[GroundStation] = None,
    show_los_line: bool = True,
    time_slider: bool = True,
    width: int = 1100,
    height: int = 800,
    show_coastlines: bool = True,
    show_graticule: bool = True,
    camera: str = "default",
) -> Path:
    """Render one or more 3D orbits to an interactive HTML file.

    Args:
        orbits: A single :class:`Orbit3D` or a sequence of them.
        output_path: Destination file. Suffix selects format: ``.html``
            (default — interactive plot) or any ``kaleido``-supported static
            format (``.png``, ``.svg``, ``.pdf``). Static export is wired
            in step 5 of stage 8; for now any non-HTML suffix raises
            ``NotImplementedError`` to fail loudly.
        title: Optional title override. Default is built from the
            satellite name(s) and catalog number(s).
        colors: Optional per-orbit colour list. Defaults to
            :data:`~sat_tracker.visualization.common.DEFAULT_TRACK_COLORS`
            so the same satellite is the same colour as in the 2D plots.
        line_width: Orbit polyline width.
        current_time_utc: Optional "now" instant. For each orbit, marks
            the sample closest to this time with a star — same discipline
            as the 2D renderers (suppressed if outside the rendered
            window).
        width, height: Pixel size for the figure.
        show_coastlines: When True (default), overlays Natural Earth
            coastline polylines on the sphere.
        show_graticule: When True (default), overlays a faint 30°
            lat/lon grid.
        camera: ``"default"`` for the static (1.6, 1.6, 0.8) eye position;
            ``"follow"`` to position the camera along the same vector as
            the "now" sample of the first orbit (only meaningful when
            ``current_time_utc`` is provided).
        ground_station: Optional :class:`GroundStation`. When provided,
            its location is shown as a small marker on Earth's surface.
        show_los_line: When ``True`` (default) and both ``ground_station``
            and ``current_time_utc`` are provided, draw a thin line from
            the station to the satellite's "now" position — only when
            the satellite is geometrically above the horizon (positive
            elevation). Hidden otherwise.
        time_slider: When ``True`` (default) and writing HTML, attach a
            slider that scrubs the "now" marker along the first orbit
            (multi-sat HTML uses the first orbit's time grid as the
            slider axis; the marker on every other orbit advances
            synchronously). Static exports ignore the slider entirely.

    Returns:
        Resolved :class:`~pathlib.Path` of the written file.

    Raises:
        ValueError: If ``orbits`` is empty or any orbit has no samples.
        NotImplementedError: If ``output_path`` is not an HTML file.
            Static export lands in a later step of stage 8.
        ImportError: If plotly or cartopy are not installed.
    """
    orbit_list = _normalize_orbits(orbits)
    color_list = list(colors) if colors else list(DEFAULT_TRACK_COLORS)

    output_path = Path(output_path)
    suffix = output_path.suffix.lower()
    if suffix == "":
        output_path = output_path.with_suffix(".html")
        suffix = ".html"
    is_static = suffix not in {".html", ".htm"}

    import plotly.graph_objects as go

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = go.Figure()

    # 1. Earth sphere (back-most).
    _add_earth_sphere(fig)

    # 2. Coastlines + graticule (overlay; thin lines).
    if show_coastlines:
        _add_coastlines(fig)
    if show_graticule:
        _add_graticule(fig)

    # 3. Orbit polylines + endpoints + "now" markers (front-most).
    # Track the index of each orbit's "now" marker trace so the slider
    # frames can address them by position.
    now_marker_indices: list[Optional[int]] = []
    any_eop_degraded = False
    for idx, orbit in enumerate(orbit_list):
        color = color_list[idx % len(color_list)]
        any_eop_degraded = any_eop_degraded or orbit.eop_degraded
        marker_idx = _add_orbit_traces(
            fig,
            orbit,
            color=color,
            line_width=line_width,
            current_time_utc=current_time_utc,
            always_emit_now_marker=time_slider,
        )
        now_marker_indices.append(marker_idx)

    # 4. Ground station + line-of-sight (if requested).
    if ground_station is not None:
        _add_ground_station(
            fig,
            ground_station,
            orbit_list,
            current_time_utc=current_time_utc,
            show_los_line=show_los_line,
        )

    # Camera + scene framing.
    cam = _resolve_camera(camera, orbit_list, current_time_utc)
    _layout_scene(
        fig,
        title=title or _default_title(orbit_list),
        subtitle=_subtitle(orbit_list, any_eop_degraded),
        camera=cam,
        width=width,
        height=height,
        multi=len(orbit_list) > 1,
    )

    # Slider only applies to interactive HTML — static exports must be
    # deterministic, so we skip the slider machinery entirely on PNG/SVG/PDF.
    if not is_static and time_slider and any(
        idx is not None for idx in now_marker_indices
    ):
        _attach_time_slider(fig, orbit_list, now_marker_indices)

    if is_static:
        # Drop bottom margin we'd reserved for the slider gutter — without
        # the slider, the extra padding would just leave dead space below
        # the globe in the static export.
        fig.update_layout(margin=dict(l=0, r=0, t=70, b=0))
        fig.write_image(str(output_path), width=width, height=height)
    else:
        fig.write_html(str(output_path), include_plotlyjs="cdn")
    logger.debug(
        "Rendered 3D orbit for %d satellite(s) to %s",
        len(orbit_list),
        output_path,
    )
    return output_path


# -- Earth sphere -------------------------------------------------------------


def _add_earth_sphere(fig) -> None:
    """Add a WGS84 ellipsoid as a flat ocean-blue Surface trace."""
    import numpy as np
    import plotly.graph_objects as go

    lats = np.linspace(-math.pi / 2, math.pi / 2, _SPHERE_LAT_STEPS)
    lons = np.linspace(-math.pi, math.pi, _SPHERE_LON_STEPS)
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")

    # Parametric ellipsoid (geodetic latitude, longitude on the surface).
    cos_lat = np.cos(lat_grid)
    sin_lat = np.sin(lat_grid)
    cos_lon = np.cos(lon_grid)
    sin_lon = np.sin(lon_grid)
    e2 = 2.0 * _WGS84_F - _WGS84_F * _WGS84_F
    n_radius = _WGS84_A_KM / np.sqrt(1.0 - e2 * sin_lat * sin_lat)
    x = n_radius * cos_lat * cos_lon
    y = n_radius * cos_lat * sin_lon
    z = n_radius * (1.0 - e2) * sin_lat

    # surfacecolor uses a single-stop colourscale to keep the entire sphere
    # one flat colour. Lighting kept dim/diffuse so we don't get a strong
    # specular highlight that would distract from the orbit line.
    fig.add_trace(
        go.Surface(
            x=x,
            y=y,
            z=z,
            surfacecolor=np.zeros_like(x),
            colorscale=[[0, _OCEAN_BASE], [1, _OCEAN_BASE]],
            showscale=False,
            hoverinfo="skip",
            lighting=dict(
                ambient=0.85,
                diffuse=0.4,
                specular=0.1,
                roughness=0.95,
                fresnel=0.05,
            ),
            lightposition=dict(x=10000, y=10000, z=8000),
            opacity=1.0,
            name="Earth",
            showlegend=False,
        )
    )


def _add_coastlines(fig) -> None:
    """Overlay Natural Earth coastlines as 3D polylines on the sphere.

    Reads the ne_110m_coastline shapefile that cartopy has already
    downloaded (the 2D renderer triggers the download). Each LineString
    becomes a contiguous polyline; segments are joined into a single
    Scatter3d trace with NaN separators (plotly's idiom for "lift the
    pen" between segments).
    """
    import numpy as np
    import plotly.graph_objects as go
    import cartopy.io.shapereader as shpreader

    fname = shpreader.natural_earth(
        resolution="110m", category="physical", name="coastline"
    )
    reader = shpreader.Reader(fname)

    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for geom in reader.geometries():
        # Pull (lon, lat) sequences from each LineString. MultiLineString
        # would need a .geoms iteration, but ne_110m_coastline only ships
        # plain LineStrings so this stays simple.
        if geom.geom_type != "LineString":
            continue
        lons, lats = geom.xy
        for lon_deg, lat_deg in zip(lons, lats):
            x, y, z = _wgs84_surface_xyz(lat_deg, lon_deg, lift_km=2.0)
            xs.append(x)
            ys.append(y)
            zs.append(z)
        # NaN separator: plotly draws a discontinuity between this geom
        # and the next, so they don't visually concatenate.
        xs.append(float("nan"))
        ys.append(float("nan"))
        zs.append(float("nan"))

    fig.add_trace(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            line=dict(color=_COASTLINE_COLOR, width=_COASTLINE_WIDTH),
            opacity=_COASTLINE_OPACITY,
            hoverinfo="skip",
            showlegend=False,
            name="Coastlines",
        )
    )


def _add_graticule(fig) -> None:
    """Faint 30° lat/lon grid on the Earth sphere."""
    import numpy as np
    import plotly.graph_objects as go

    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []

    # Meridians: hold longitude fixed, sweep latitude.
    lat_sweep = np.linspace(-89.0, 89.0, 90)
    for lon_deg in range(-180, 181, _GRATICULE_SPACING_DEG):
        for lat_deg in lat_sweep:
            x, y, z = _wgs84_surface_xyz(float(lat_deg), float(lon_deg), lift_km=1.5)
            xs.append(x)
            ys.append(y)
            zs.append(z)
        xs.append(float("nan"))
        ys.append(float("nan"))
        zs.append(float("nan"))

    # Parallels: hold latitude fixed, sweep longitude.
    lon_sweep = np.linspace(-180.0, 180.0, 180)
    for lat_deg in range(-60, 61, _GRATICULE_SPACING_DEG):
        for lon_deg in lon_sweep:
            x, y, z = _wgs84_surface_xyz(float(lat_deg), float(lon_deg), lift_km=1.5)
            xs.append(x)
            ys.append(y)
            zs.append(z)
        xs.append(float("nan"))
        ys.append(float("nan"))
        zs.append(float("nan"))

    fig.add_trace(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            line=dict(color=_GRATICULE_COLOR, width=_GRATICULE_WIDTH),
            opacity=_GRATICULE_OPACITY,
            hoverinfo="skip",
            showlegend=False,
            name="Graticule",
        )
    )


def _wgs84_surface_xyz(
    lat_deg: float, lon_deg: float, *, lift_km: float = 0.0
) -> tuple[float, float, float]:
    """Convert (geodetic lat, lon, height) → ECEF Cartesian (km).

    ``lift_km`` is added as a small height-above-ellipsoid so coastline /
    graticule overlays sit just outside the sphere surface and don't
    z-fight with the Surface trace.
    """
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    e2 = 2.0 * _WGS84_F - _WGS84_F * _WGS84_F
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    n_radius = _WGS84_A_KM / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
    x = (n_radius + lift_km) * cos_lat * math.cos(lon)
    y = (n_radius + lift_km) * cos_lat * math.sin(lon)
    z = (n_radius * (1.0 - e2) + lift_km) * sin_lat
    return x, y, z


# -- Orbit traces -------------------------------------------------------------


def _add_orbit_traces(
    fig,
    orbit: Orbit3D,
    *,
    color: str,
    line_width: float,
    current_time_utc: Optional[datetime],
    always_emit_now_marker: bool = False,
) -> Optional[int]:
    """Add orbit polyline + endpoint markers + optional "now" marker.

    Returns the index in ``fig.data`` of the "now" marker trace if one
    was emitted, else ``None``. The slider machinery uses this index to
    address the right trace when scrubbing.
    """
    import plotly.graph_objects as go

    label = f"{orbit.name or '<unnamed>'} [{orbit.catalog_number}]"
    xs = [s.x_km for s in orbit.samples]
    ys = [s.y_km for s in orbit.samples]
    zs = [s.z_km for s in orbit.samples]
    hover = [
        (
            f"{label}<br>"
            f"{s.time_utc.strftime(_HOVER_TIME_FORMAT)}<br>"
            f"x={s.x_km:+.0f}, y={s.y_km:+.0f}, z={s.z_km:+.0f} km<br>"
            f"|r|={math.sqrt(s.x_km ** 2 + s.y_km ** 2 + s.z_km ** 2):.0f} km"
        )
        for s in orbit.samples
    ]
    fig.add_trace(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            line=dict(color=color, width=line_width),
            name=label,
            legendgroup=label,
            hovertext=hover,
            hoverinfo="text",
        )
    )

    start = orbit.samples[0]
    end = orbit.samples[-1]
    fig.add_trace(
        go.Scatter3d(
            x=[start.x_km],
            y=[start.y_km],
            z=[start.z_km],
            mode="markers",
            marker=dict(
                size=7,
                color=color,
                line=dict(color="white", width=2),
                symbol="circle",
            ),
            name=f"{label} start",
            legendgroup=label,
            showlegend=False,
            hovertext=(
                f"{label} start<br>"
                f"{start.time_utc.strftime(_HOVER_TIME_FORMAT)}"
            ),
            hoverinfo="text",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[end.x_km],
            y=[end.y_km],
            z=[end.z_km],
            mode="markers",
            marker=dict(
                size=4,
                color=color,
                line=dict(color="white", width=1),
                symbol="diamond",
                opacity=0.85,
            ),
            name=f"{label} end",
            legendgroup=label,
            showlegend=False,
            hovertext=(
                f"{label} end<br>"
                f"{end.time_utc.strftime(_HOVER_TIME_FORMAT)}"
            ),
            hoverinfo="text",
        )
    )

    # "Now" marker. Emitted if (a) the caller passed current_time_utc and
    # the closest sample is within the window, or (b) the slider needs a
    # placeholder trace it can scrub regardless of caller-provided "now".
    sample = None
    if current_time_utc is not None:
        sample = _closest_sample(orbit.samples, current_time_utc)
    if sample is None and always_emit_now_marker:
        # Use the middle sample as the slider's initial position.
        sample = orbit.samples[len(orbit.samples) // 2]
    if sample is None:
        return None

    fig.add_trace(
        go.Scatter3d(
            x=[sample.x_km],
            y=[sample.y_km],
            z=[sample.z_km],
            mode="markers",
            marker=dict(
                size=10,
                color="#ffd400",
                line=dict(color=color, width=2),
                # Plotly 3D doesn't support the 'star' symbol; a slightly
                # larger gold diamond approximates the visual hierarchy
                # of the 2D star.
                symbol="diamond",
            ),
            name=f"{label} now",
            legendgroup=label,
            showlegend=False,
            hovertext=(
                f"{label} now<br>"
                f"{sample.time_utc.strftime(_HOVER_TIME_FORMAT)}<br>"
                f"|r|={math.sqrt(sample.x_km ** 2 + sample.y_km ** 2 + sample.z_km ** 2):.0f} km"
            ),
            hoverinfo="text",
        )
    )
    return len(fig.data) - 1


def _closest_sample(samples, when: datetime):
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    closest = min(samples, key=lambda s: abs((s.time_utc - when).total_seconds()))
    if len(samples) >= 2:
        step = abs(
            (samples[1].time_utc - samples[0].time_utc).total_seconds()
        )
    else:
        step = float("inf")
    if abs((closest.time_utc - when).total_seconds()) > step:
        return None
    return closest


# -- Layout / camera ----------------------------------------------------------


def _resolve_camera(
    mode: str,
    orbits: Sequence[Orbit3D],
    current_time_utc: Optional[datetime],
) -> _Camera:
    if mode == "default" or current_time_utc is None:
        return _DEFAULT_CAMERA
    if mode != "follow":
        raise ValueError(f"camera must be 'default' or 'follow', got {mode!r}")
    sample = _closest_sample(orbits[0].samples, current_time_utc)
    if sample is None:
        return _DEFAULT_CAMERA
    # Eye is along the same vector as the satellite's "now" position but
    # ~2.2 Earth radii out, with a 10° downward tilt for a clearer view of
    # the inclination.
    r = math.sqrt(sample.x_km ** 2 + sample.y_km ** 2 + sample.z_km ** 2)
    if r == 0:
        return _DEFAULT_CAMERA
    nx, ny, nz = sample.x_km / r, sample.y_km / r, sample.z_km / r
    # Apply a 10° rotation about the local east axis to tilt down.
    horiz = math.hypot(nx, ny)
    tilt = math.radians(10.0)
    new_z = nz * math.cos(tilt) - horiz * math.sin(tilt)
    scale_xy = (nz * math.sin(tilt) + horiz * math.cos(tilt)) / max(horiz, 1e-9)
    return _Camera(
        eye_x=nx * scale_xy * 2.2,
        eye_y=ny * scale_xy * 2.2,
        eye_z=new_z * 2.2,
    )


def _layout_scene(
    fig,
    *,
    title: str,
    subtitle: str,
    camera: _Camera,
    width: int,
    height: int,
    multi: bool,
) -> None:
    fig.update_layout(
        title=dict(
            text=(
                f"<b>{title}</b><br>"
                f"<span style='font-size:11px'>{subtitle}</span>"
            ),
            x=0.5,
            xanchor="center",
        ),
        width=width,
        height=height,
        margin=dict(l=0, r=0, t=70, b=80),
        showlegend=multi,
        legend=dict(
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.85)",
        ),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",  # 1 km on x = 1 km on y = 1 km on z
            bgcolor="#0a0e1a",   # dark "space" surround makes Earth pop
            camera=dict(
                eye=dict(
                    x=camera.eye_x, y=camera.eye_y, z=camera.eye_z
                ),
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
            ),
        ),
        paper_bgcolor="#0a0e1a",
        font=dict(color="#e8e8e8"),
    )


# -- Ground station -----------------------------------------------------------


def _add_ground_station(
    fig,
    station: GroundStation,
    orbits: Sequence[Orbit3D],
    *,
    current_time_utc: Optional[datetime],
    show_los_line: bool,
) -> None:
    """Add a ground-station marker, plus an optional line-of-sight line.

    The marker sits exactly on the ellipsoid surface at the station's
    geodetic lat/lon (altitude is added if non-zero, but at MVP scope the
    visual difference is sub-pixel).

    The line-of-sight line is drawn only when:
      - ``show_los_line`` is True,
      - ``current_time_utc`` is provided,
      - the first orbit has a sample within the rendered window, AND
      - that satellite is geometrically above the station's local horizon
        (positive elevation). Below-horizon satellites have no line of
        sight, so drawing one would be physically misleading.
    """
    import plotly.graph_objects as go

    sx, sy, sz = _wgs84_surface_xyz(
        station.latitude_deg,
        station.longitude_deg,
        lift_km=max(station.altitude_km, 5.0),
    )
    label = station.name or (
        f"{station.latitude_deg:+.3f}°, {station.longitude_deg:+.3f}°"
    )
    fig.add_trace(
        go.Scatter3d(
            x=[sx],
            y=[sy],
            z=[sz],
            mode="markers",
            marker=dict(
                size=6,
                color="#39ff88",
                line=dict(color="white", width=1.2),
                symbol="diamond",
            ),
            name=f"GS: {label}",
            hovertext=(
                f"Ground station<br>"
                f"{label}<br>"
                f"alt {station.altitude_km:.2f} km"
            ),
            hoverinfo="text",
            showlegend=True,
        )
    )

    if not (show_los_line and current_time_utc is not None and orbits):
        return

    sample = _closest_sample(orbits[0].samples, current_time_utc)
    if sample is None:
        return

    if not _is_above_horizon(
        station, sx_km=sample.x_km, sy_km=sample.y_km, sz_km=sample.z_km
    ):
        return

    fig.add_trace(
        go.Scatter3d(
            x=[sx, sample.x_km],
            y=[sy, sample.y_km],
            z=[sz, sample.z_km],
            mode="lines",
            line=dict(color="#39ff88", width=2, dash="dot"),
            opacity=0.7,
            hoverinfo="skip",
            showlegend=False,
            name="LOS",
        )
    )


def _is_above_horizon(
    station: GroundStation, *, sx_km: float, sy_km: float, sz_km: float
) -> bool:
    """True iff the ECEF point (sx, sy, sz) is above the station's local
    horizon — i.e. the elevation angle is positive.

    Tests the dot product of the station-to-satellite vector with the
    station's local "up" (geodetic normal). Positive → above horizon.
    """
    # Station ECEF position on the ellipsoid surface (no lift).
    bx, by, bz = _wgs84_surface_xyz(
        station.latitude_deg, station.longitude_deg, lift_km=0.0
    )
    # Local up vector at the station (geodetic normal to the ellipsoid).
    lat = math.radians(station.latitude_deg)
    lon = math.radians(station.longitude_deg)
    ux = math.cos(lat) * math.cos(lon)
    uy = math.cos(lat) * math.sin(lon)
    uz = math.sin(lat)
    dx = sx_km - bx
    dy = sy_km - by
    dz = sz_km - bz
    return (dx * ux + dy * uy + dz * uz) > 0.0


# -- Time slider --------------------------------------------------------------


def _attach_time_slider(
    fig,
    orbits: Sequence[Orbit3D],
    now_marker_indices: Sequence[Optional[int]],
) -> None:
    """Attach a slider that scrubs the "now" marker(s) along the first
    orbit's time grid.

    The slider's frame index addresses the first orbit's samples directly.
    For each frame, every other orbit's marker is updated to its
    closest-in-time sample (each orbit may have a slightly different time
    grid because mean motions differ).
    """
    if not orbits:
        return
    first = orbits[0]
    n_frames = len(first.samples)

    # Build per-frame marker positions for every orbit.
    frames_data: list[list[dict]] = []  # frames[i] = list of trace updates
    slider_steps: list[dict] = []

    for i, primary_sample in enumerate(first.samples):
        when = primary_sample.time_utc
        traces_in_frame = []
        for orbit, marker_idx in zip(orbits, now_marker_indices):
            if marker_idx is None:
                traces_in_frame.append(None)
                continue
            sample = _closest_sample(orbit.samples, when)
            if sample is None:
                # No close sample — keep the trace at its current spot
                # by sending an empty update.
                traces_in_frame.append({"x": [], "y": [], "z": []})
                continue
            traces_in_frame.append(
                {
                    "x": [sample.x_km],
                    "y": [sample.y_km],
                    "z": [sample.z_km],
                    "hovertext": [
                        f"{orbit.name or '<unnamed>'} [{orbit.catalog_number}]<br>"
                        f"{sample.time_utc.strftime(_HOVER_TIME_FORMAT)}<br>"
                        f"|r|={math.sqrt(sample.x_km ** 2 + sample.y_km ** 2 + sample.z_km ** 2):.0f} km"
                    ],
                }
            )
        frames_data.append(traces_in_frame)

    # Plotly's frames API expects a list of go.Frame objects, each with
    # `data` (list of partial-trace updates) and `traces` (list of trace
    # indices the data entries apply to).
    import plotly.graph_objects as go

    frame_objects = []
    valid_indices = [i for i in now_marker_indices if i is not None]
    for i, frame in enumerate(frames_data):
        data_for_frame = []
        traces_for_frame = []
        for marker_idx, update in zip(now_marker_indices, frame):
            if marker_idx is None or update is None:
                continue
            data_for_frame.append(go.Scatter3d(**update))
            traces_for_frame.append(marker_idx)
        frame_objects.append(
            go.Frame(
                data=data_for_frame,
                traces=traces_for_frame,
                name=str(i),
            )
        )
        slider_steps.append(
            {
                "method": "animate",
                "label": first.samples[i].time_utc.strftime("%H:%M"),
                "args": [
                    [str(i)],
                    {
                        "mode": "immediate",
                        "frame": {"duration": 0, "redraw": True},
                        "transition": {"duration": 0},
                    },
                ],
            }
        )

    fig.frames = frame_objects

    # Attach slider + play/pause buttons. Initial slider value: the frame
    # whose timestamp is closest to the "current" instant if one was used,
    # else frame 0.
    initial_active = n_frames // 2 if n_frames > 0 else 0

    fig.update_layout(
        sliders=[
            {
                "active": initial_active,
                "currentvalue": {
                    "prefix": "t = ",
                    "font": {"size": 11, "color": "#ddd"},
                },
                "pad": {"t": 30, "b": 10, "l": 60, "r": 30},
                "len": 0.85,
                "x": 0.075,
                "y": 0.0,
                "steps": slider_steps,
                "bgcolor": "rgba(255,255,255,0.15)",
                "activebgcolor": "#ffd400",
                "tickcolor": "#ddd",
                "font": {"color": "#ddd"},
            }
        ],
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.04,
                "y": -0.005,
                "xanchor": "right",
                "yanchor": "top",
                "pad": {"t": 30, "r": 10},
                "bgcolor": "rgba(255,255,255,0.15)",
                "font": {"color": "#ddd"},
                "buttons": [
                    {
                        "label": "▶",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "mode": "immediate",
                                "frame": {"duration": 80, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "❚❚",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": False},
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
            }
        ],
    )


# -- Helpers ------------------------------------------------------------------


def _normalize_orbits(orbits: Union[Orbit3D, Sequence[Orbit3D]]) -> list[Orbit3D]:
    if isinstance(orbits, Orbit3D):
        orbit_list = [orbits]
    else:
        orbit_list = list(orbits)
    if not orbit_list:
        raise ValueError("orbits must contain at least one Orbit3D")
    for orbit in orbit_list:
        if not orbit.samples:
            raise ValueError(
                f"orbit for catnr={orbit.catalog_number} has no samples"
            )
    return orbit_list


def _default_title(orbits: Sequence[Orbit3D]) -> str:
    if len(orbits) == 1:
        o = orbits[0]
        return f"{o.name or '<unnamed>'} [{o.catalog_number}] 3D orbit"
    names = ", ".join(
        f"{o.name or '<unnamed>'} [{o.catalog_number}]" for o in orbits
    )
    return f"3D orbits: {names}"


def _subtitle(orbits: Sequence[Orbit3D], any_eop_degraded: bool) -> str:
    earliest = min(o.samples[0].time_utc for o in orbits)
    latest = max(o.samples[-1].time_utc for o in orbits)
    epochs = sorted(
        {o.tle_epoch_utc.strftime("%Y-%m-%d %H:%M UTC") for o in orbits}
    )
    epoch_label = (
        f"TLE epoch {epochs[0]}"
        if len(epochs) == 1
        else f"TLE epochs {', '.join(epochs)}"
    )
    parts = [
        f"frame: {orbits[0].frame.upper()}",
        epoch_label,
        (
            f"{earliest.strftime('%Y-%m-%d %H:%M UTC')} → "
            f"{latest.strftime('%Y-%m-%d %H:%M UTC')}"
        ),
    ]
    if any_eop_degraded:
        parts.append("EOP degraded")
    return "  |  ".join(parts)


__all__ = ("render_orbit_3d",)
