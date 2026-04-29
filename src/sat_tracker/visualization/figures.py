"""Plotly figure builders — pure, return ``go.Figure`` without disk I/O.

The :mod:`sat_tracker.visualization.interactive` and
:mod:`sat_tracker.visualization.orbit_3d` modules both used to combine
figure construction with file output (``write_html`` / ``write_image``).
That coupling was fine for the CLI but blocks the Streamlit dashboard,
which needs an in-memory :class:`plotly.graph_objects.Figure` to pass to
``st.plotly_chart``.

This module factors the figure-building half out so it can be called from
either context. The CLI-facing ``render_*`` functions in those two
modules now build a figure here and write it to disk; the dashboard
imports the builders directly.

Plotly is defer-imported so importing this module does not require the
``[viz]`` extra to be installed — the failure surfaces only when a
caller actually invokes one of the builders.
"""

from __future__ import annotations

import functools
import logging
import math
from datetime import datetime, timezone
from typing import Optional, Sequence, Union

from sat_tracker.passes import GroundStation
from sat_tracker.visualization.common import (
    DEFAULT_TRACK_COLORS,
    Orbit3D,
    Track,
    split_at_antimeridian,
)

logger = logging.getLogger(__name__)


_HOVER_TIME_FORMAT = "%Y-%m-%d %H:%M:%S UTC"
_SUBTITLE_TIME_FORMAT = "%Y-%m-%d %H:%M UTC"

# WGS84 defining constants (kilometres). Match coordinates._WGS84_*.
_WGS84_A_KM: float = 6378.137
_WGS84_F: float = 1.0 / 298.257223563
_WGS84_B_KM: float = _WGS84_A_KM * (1.0 - _WGS84_F)

# 3D Earth-sphere appearance — tuned to read as "Earth from space"
# against a dark space surround. Deeper than the 2D map's ocean colour
# because at 3D camera distances the lighter shade washed out and the
# globe stopped looking like Earth.
_OCEAN_BASE = "#2c4a6e"
# White continents on dark ocean — high-contrast monochrome look,
# reminiscent of aviation chart aesthetics. Pure white survives
# plotly's heavy ambient lighting wash without losing legibility.
_LAND_BASE = "#ffffff"
# Off-white coastlines pop against the dark ocean — same readability
# strategy NASA Blue Marble uses for low-bandwidth web previews.
_COASTLINE_COLOR = "#e8e8d8"
_COASTLINE_WIDTH = 1.6
_COASTLINE_OPACITY = 0.85
# Graticule: faint white, just visible enough to give an orientation
# reference without competing with the orbit.
_GRATICULE_COLOR = "#c8d4e8"
_GRATICULE_WIDTH = 0.6
_GRATICULE_OPACITY = 0.18
_GRATICULE_SPACING_DEG = 30
# Sphere mesh density: 145 × 289 = ~42K cells, every 1.25° lat / 1.25°
# lon. Matched to the ne_50m Natural Earth shapefile detail. Going
# higher (e.g. 217×433 + ne_10m) would expose finer islands but adds
# ~2 s to cold-start mask computation; 145×289 is the sweet spot.
_SPHERE_LAT_STEPS = 145
_SPHERE_LON_STEPS = 289
# Natural Earth resolution used for both the land-fill mask and the
# coastline overlay. "50m" = 1:50,000,000 scale; visibly crisper than
# the previous "110m" without a noticeable performance hit.
_NATURAL_EARTH_RESOLUTION = "50m"


# -- Public builders ----------------------------------------------------------


def build_ground_track_figure(
    tracks: Union[Track, Sequence[Track]],
    *,
    title: Optional[str] = None,
    colors: Optional[Sequence[str]] = None,
    line_width: float = 2.5,
    current_time_utc: Optional[datetime] = None,
    width: int = 1200,
    height: int = 650,
):
    """Build the 2D plotly ground-track figure (no disk I/O).

    See :func:`sat_tracker.visualization.interactive.render_interactive_ground_track`
    for full argument documentation. This builder takes the same
    parameters minus ``output_path`` and returns the constructed
    :class:`plotly.graph_objects.Figure` for callers that want to embed
    it (Streamlit's ``st.plotly_chart``, Jupyter, etc.) instead of
    writing it to disk.

    Returns:
        A :class:`plotly.graph_objects.Figure` instance ready to render.
    """
    track_list = _normalize_tracks(tracks)
    color_list = list(colors) if colors else list(DEFAULT_TRACK_COLORS)

    import plotly.graph_objects as go

    fig = go.Figure()
    any_eop_degraded = False

    for idx, track in enumerate(track_list):
        color = color_list[idx % len(color_list)]
        any_eop_degraded = any_eop_degraded or track.eop_degraded
        label = f"{track.name or '<unnamed>'} [{track.catalog_number}]"
        _add_track_traces(
            fig,
            track,
            label=label,
            color=color,
            line_width=line_width,
            current_time_utc=current_time_utc,
        )

    fig.update_geos(
        projection_type="equirectangular",
        showcoastlines=True,
        coastlinecolor="#444",
        coastlinewidth=0.7,
        showland=True,
        landcolor="#f4f1e8",
        showocean=True,
        oceancolor=_OCEAN_BASE,
        showcountries=True,
        countrycolor="rgba(136,136,136,0.4)",
        countrywidth=0.4,
        lataxis=dict(showgrid=True, gridcolor="rgba(120,120,120,0.3)", dtick=30),
        lonaxis=dict(showgrid=True, gridcolor="rgba(120,120,120,0.3)", dtick=60),
    )

    if title is None:
        title = _ground_track_default_title(track_list)
    subtitle = _ground_track_subtitle(track_list, any_eop_degraded)
    # Deep navy title matches the 3D figure's white-on-dark hierarchy
    # without requiring a fully dark 2D map; the cream basemap reads
    # better with a strong dark title than a mid-grey one.
    fig.update_layout(
        title=dict(
            text=(
                f"<b>{title}</b><br>"
                f"<span style='font-size:11px;color:#3a4a5e'>"
                f"{subtitle}</span>"
            ),
            x=0.5,
            xanchor="center",
            font=dict(color="#1a3a5e", size=15),
        ),
        width=width,
        height=height,
        margin=dict(l=10, r=10, t=80, b=10),
        legend=dict(
            yanchor="bottom",
            y=0.02,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="#1a3a5e",
            borderwidth=1,
            font=dict(color="#1a3a5e", size=11),
        ),
        showlegend=len(track_list) > 1,
    )
    return fig


def build_orbit_3d_figure(
    orbits: Union[Orbit3D, Sequence[Orbit3D]],
    *,
    title: Optional[str] = None,
    colors: Optional[Sequence[str]] = None,
    line_width: float = 4.0,
    current_time_utc: Optional[datetime] = None,
    ground_station: Optional[GroundStation] = None,
    show_los_line: bool = True,
    show_coastlines: bool = True,
    show_graticule: bool = True,
    width: int = 1100,
    height: int = 800,
    camera_eye: Optional[tuple[float, float, float]] = None,
):
    """Build the 3D plotly orbit figure (no disk I/O, no slider).

    Mirrors :func:`sat_tracker.visualization.orbit_3d.render_orbit_3d`
    minus the slider/animation machinery and disk output. The dashboard
    layers its own animation frames on top of the figure this builder
    returns.

    Args:
        camera_eye: Optional ``(x, y, z)`` tuple in plotly scene-data
            units (1.0 ≈ Earth radius). Default is ``(1.6, 1.6, 0.8)``.
            Other args mirror :func:`render_orbit_3d`.

    Returns:
        A :class:`plotly.graph_objects.Figure` with:

        * Earth sphere trace (index 0)
        * Optional coastline polyline trace
        * Optional graticule polyline trace
        * Per-orbit line + start + end + optional "now" marker traces
        * Optional ground-station marker + LOS line traces
    """
    orbit_list = _normalize_orbits(orbits)
    color_list = list(colors) if colors else list(DEFAULT_TRACK_COLORS)

    import plotly.graph_objects as go

    fig = go.Figure()
    _add_earth_sphere(fig)
    if show_coastlines:
        _add_coastlines(fig)
    if show_graticule:
        _add_graticule(fig)

    any_eop_degraded = False
    for idx, orbit in enumerate(orbit_list):
        color = color_list[idx % len(color_list)]
        any_eop_degraded = any_eop_degraded or orbit.eop_degraded
        _add_orbit_traces(
            fig,
            orbit,
            color=color,
            line_width=line_width,
            current_time_utc=current_time_utc,
        )

    if ground_station is not None:
        _add_ground_station(
            fig,
            ground_station,
            orbit_list,
            current_time_utc=current_time_utc,
            show_los_line=show_los_line,
        )

    eye = camera_eye if camera_eye is not None else (1.6, 1.6, 0.8)
    if title is None:
        title = _orbit_3d_default_title(orbit_list)
    subtitle = _orbit_3d_subtitle(orbit_list, any_eop_degraded)
    fig.update_layout(
        title=dict(
            text=(
                f"<b>{title}</b><br>"
                f"<span style='font-size:11px;color:#cdd6e0'>"
                f"{subtitle}</span>"
            ),
            x=0.5,
            xanchor="center",
            font=dict(color="#f0f0f5"),
        ),
        width=width,
        height=height,
        margin=dict(l=0, r=0, t=70, b=0),
        showlegend=len(orbit_list) > 1,
        legend=dict(
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(20,30,50,0.85)",
            bordercolor="rgba(180,200,230,0.4)",
            borderwidth=1,
            font=dict(color="#f0f0f5", size=12),
        ),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
            bgcolor="#0a0e1a",
            camera=dict(
                eye=dict(x=eye[0], y=eye[1], z=eye[2]),
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
            ),
        ),
        paper_bgcolor="#0a0e1a",
        font=dict(color="#e8e8e8"),
    )
    return fig


# -- 2D ground-track helpers --------------------------------------------------


def _normalize_tracks(tracks: Union[Track, Sequence[Track]]) -> list[Track]:
    if isinstance(tracks, Track):
        track_list = [tracks]
    else:
        track_list = list(tracks)
    if not track_list:
        raise ValueError("tracks must contain at least one Track")
    for track in track_list:
        if not track.samples:
            raise ValueError(
                f"track for catnr={track.catalog_number} has no samples; "
                f"nothing to render"
            )
    return track_list


def _add_track_traces(
    fig,
    track: Track,
    *,
    label: str,
    color: str,
    line_width: float,
    current_time_utc: Optional[datetime],
) -> None:
    import plotly.graph_objects as go

    segments = split_at_antimeridian(list(track.samples))
    first_segment = True
    for segment in segments:
        if len(segment) < 2:
            continue
        lons = [s.longitude_deg for s in segment]
        lats = [s.latitude_deg for s in segment]
        hover = [
            (
                f"{label}<br>"
                f"{s.time_utc.strftime(_HOVER_TIME_FORMAT)}<br>"
                f"lat {s.latitude_deg:+.3f}°, lon {s.longitude_deg:+.3f}°<br>"
                f"alt {s.altitude_km:.1f} km"
            )
            for s in segment
        ]
        fig.add_trace(
            go.Scattergeo(
                lon=lons,
                lat=lats,
                mode="lines",
                line=dict(color=color, width=line_width),
                name=label,
                legendgroup=label,
                showlegend=first_segment,
                hovertext=hover,
                hoverinfo="text",
            )
        )
        first_segment = False

    start = track.samples[0]
    end = track.samples[-1]
    fig.add_trace(
        go.Scattergeo(
            lon=[start.longitude_deg],
            lat=[start.latitude_deg],
            mode="markers",
            marker=dict(
                size=14,
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
        go.Scattergeo(
            lon=[end.longitude_deg],
            lat=[end.latitude_deg],
            mode="markers",
            marker=dict(
                size=9,
                color=color,
                line=dict(color="white", width=1),
                symbol="triangle-up",
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

    if current_time_utc is not None:
        sample = _closest_track_sample(track.samples, current_time_utc)
        if sample is not None:
            fig.add_trace(
                go.Scattergeo(
                    lon=[sample.longitude_deg],
                    lat=[sample.latitude_deg],
                    mode="markers",
                    marker=dict(
                        size=20,
                        color="#ffffff",
                        line=dict(color=color, width=2),
                        symbol="star",
                    ),
                    name=f"{label} now",
                    legendgroup=label,
                    showlegend=False,
                    hovertext=(
                        f"{label} now<br>"
                        f"{sample.time_utc.strftime(_HOVER_TIME_FORMAT)}<br>"
                        f"alt {sample.altitude_km:.1f} km"
                    ),
                    hoverinfo="text",
                )
            )


def _closest_track_sample(samples, when: datetime):
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    closest = min(samples, key=lambda s: abs((s.time_utc - when).total_seconds()))
    if len(samples) >= 2:
        step = abs((samples[1].time_utc - samples[0].time_utc).total_seconds())
    else:
        step = float("inf")
    if abs((closest.time_utc - when).total_seconds()) > step:
        return None
    return closest


def _ground_track_default_title(tracks: Sequence[Track]) -> str:
    if len(tracks) == 1:
        t = tracks[0]
        return f"{t.name or '<unnamed>'} [{t.catalog_number}] ground track"
    names = ", ".join(
        f"{t.name or '<unnamed>'} [{t.catalog_number}]" for t in tracks
    )
    return f"Ground tracks: {names}"


def _ground_track_subtitle(
    tracks: Sequence[Track], any_eop_degraded: bool
) -> str:
    earliest = min(t.samples[0].time_utc for t in tracks)
    latest = max(t.samples[-1].time_utc for t in tracks)
    epochs = sorted(
        {t.tle_epoch_utc.strftime(_SUBTITLE_TIME_FORMAT) for t in tracks}
    )
    epoch_label = (
        f"TLE epoch {epochs[0]}"
        if len(epochs) == 1
        else f"TLE epochs {', '.join(epochs)}"
    )
    parts = [
        epoch_label,
        (
            f"{earliest.strftime(_SUBTITLE_TIME_FORMAT)} → "
            f"{latest.strftime(_SUBTITLE_TIME_FORMAT)}"
        ),
    ]
    if any_eop_degraded:
        parts.append("EOP degraded")
    return "  |  ".join(parts)


# -- 3D orbit helpers ---------------------------------------------------------


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


def _add_earth_sphere(fig) -> None:
    import numpy as np
    import plotly.graph_objects as go

    lats = np.linspace(-math.pi / 2, math.pi / 2, _SPHERE_LAT_STEPS)
    lons = np.linspace(-math.pi, math.pi, _SPHERE_LON_STEPS)
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    cos_lat = np.cos(lat_grid)
    sin_lat = np.sin(lat_grid)
    cos_lon = np.cos(lon_grid)
    sin_lon = np.sin(lon_grid)
    e2 = 2.0 * _WGS84_F - _WGS84_F * _WGS84_F
    n_radius = _WGS84_A_KM / np.sqrt(1.0 - e2 * sin_lat * sin_lat)
    x = n_radius * cos_lat * cos_lon
    y = n_radius * cos_lat * sin_lon
    z = n_radius * (1.0 - e2) * sin_lat

    # Per-cell land/ocean mask drives the surfacecolor; combined with a
    # two-stop colorscale [ocean, land] this fills continents at zero
    # per-frame cost. Plotly's smooth interpolation between the two
    # stops produces a soft natural coastline transition for free.
    surface_color = _land_mask(_SPHERE_LAT_STEPS, _SPHERE_LON_STEPS)

    fig.add_trace(
        go.Surface(
            x=x,
            y=y,
            z=z,
            surfacecolor=surface_color,
            cmin=0.0,
            cmax=1.0,
            colorscale=[[0, _OCEAN_BASE], [1, _LAND_BASE]],
            showscale=False,
            hoverinfo="skip",
            lighting=dict(
                ambient=0.65,
                diffuse=0.55,
                specular=0.08,
                roughness=0.92,
                fresnel=0.05,
            ),
            lightposition=dict(x=10000, y=10000, z=8000),
            opacity=1.0,
            name="Earth",
            showlegend=False,
        )
    )


@functools.lru_cache(maxsize=4)
def _land_mask(lat_steps: int, lon_steps: int):
    """Build a (lat_steps × lon_steps) array: 1.0 where land, 0.0 where ocean.

    Uses cartopy's bundled Natural Earth ``ne_110m_land`` polygon set
    plus shapely's prepared-geometry contains for fast point-in-polygon
    over ~10K cells. Cached for the process so the cost is paid once
    per worker — not on every figure build.

    On any failure (shapely missing, shapefile unavailable, etc.) falls
    back to all-ocean, which produces the original featureless-sphere
    look rather than crashing the dashboard.
    """
    import numpy as np

    try:
        import cartopy.io.shapereader as shpreader
        from shapely.geometry import Point
        from shapely.ops import unary_union
        from shapely.prepared import prep
    except ImportError as exc:
        logger.warning(
            "land-mask deps missing (%s); Earth sphere will be flat ocean", exc
        )
        return np.zeros((lat_steps, lon_steps))

    try:
        land_path = shpreader.natural_earth(
            resolution=_NATURAL_EARTH_RESOLUTION,
            category="physical",
            name="land",
        )
        land_geoms = list(shpreader.Reader(land_path).geometries())
        land_union = unary_union(land_geoms)
        land_prep = prep(land_union)
    except Exception as exc:
        logger.warning(
            "Natural Earth land shapefile unavailable (%s); using flat ocean",
            exc,
        )
        return np.zeros((lat_steps, lon_steps))

    lats_deg = np.linspace(-90.0, 90.0, lat_steps)
    lons_deg = np.linspace(-180.0, 180.0, lon_steps)
    mask = np.zeros((lat_steps, lon_steps))
    for i, lat in enumerate(lats_deg):
        for j, lon in enumerate(lons_deg):
            if land_prep.contains(Point(float(lon), float(lat))):
                mask[i, j] = 1.0
    return mask


def _add_coastlines(fig) -> None:
    import plotly.graph_objects as go
    import cartopy.io.shapereader as shpreader

    fname = shpreader.natural_earth(
        resolution=_NATURAL_EARTH_RESOLUTION,
        category="physical",
        name="coastline",
    )
    reader = shpreader.Reader(fname)
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for geom in reader.geometries():
        if geom.geom_type != "LineString":
            continue
        lons, lats = geom.xy
        for lon_deg, lat_deg in zip(lons, lats):
            x, y, z = _wgs84_surface_xyz(lat_deg, lon_deg, lift_km=2.0)
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
            line=dict(color=_COASTLINE_COLOR, width=_COASTLINE_WIDTH),
            opacity=_COASTLINE_OPACITY,
            hoverinfo="skip",
            showlegend=False,
            name="Coastlines",
        )
    )


def _add_graticule(fig) -> None:
    import numpy as np
    import plotly.graph_objects as go

    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
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


def _add_orbit_traces(
    fig,
    orbit: Orbit3D,
    *,
    color: str,
    line_width: float,
    current_time_utc: Optional[datetime],
) -> None:
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
            hoverinfo="text",
            hovertext=(
                f"{label} start<br>"
                f"{start.time_utc.strftime(_HOVER_TIME_FORMAT)}"
            ),
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
            hoverinfo="text",
            hovertext=(
                f"{label} end<br>"
                f"{end.time_utc.strftime(_HOVER_TIME_FORMAT)}"
            ),
        )
    )

    if current_time_utc is not None:
        sample = _closest_orbit_sample(orbit.samples, current_time_utc)
        if sample is not None:
            fig.add_trace(
                go.Scatter3d(
                    x=[sample.x_km],
                    y=[sample.y_km],
                    z=[sample.z_km],
                    mode="markers",
                    marker=dict(
                        size=10,
                        color="#ffffff",
                        line=dict(color=color, width=2),
                        symbol="diamond",
                    ),
                    name=f"{label} now",
                    legendgroup=label,
                    showlegend=False,
                    hoverinfo="text",
                    hovertext=(
                        f"{label} now<br>"
                        f"{sample.time_utc.strftime(_HOVER_TIME_FORMAT)}<br>"
                        f"|r|={math.sqrt(sample.x_km ** 2 + sample.y_km ** 2 + sample.z_km ** 2):.0f} km"
                    ),
                )
            )


def _closest_orbit_sample(samples, when: datetime):
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    closest = min(samples, key=lambda s: abs((s.time_utc - when).total_seconds()))
    if len(samples) >= 2:
        step = abs((samples[1].time_utc - samples[0].time_utc).total_seconds())
    else:
        step = float("inf")
    if abs((closest.time_utc - when).total_seconds()) > step:
        return None
    return closest


def _add_ground_station(
    fig,
    station: GroundStation,
    orbits: Sequence[Orbit3D],
    *,
    current_time_utc: Optional[datetime],
    show_los_line: bool,
) -> None:
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

    sample = _closest_orbit_sample(orbits[0].samples, current_time_utc)
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
    bx, by, bz = _wgs84_surface_xyz(
        station.latitude_deg, station.longitude_deg, lift_km=0.0
    )
    lat = math.radians(station.latitude_deg)
    lon = math.radians(station.longitude_deg)
    ux = math.cos(lat) * math.cos(lon)
    uy = math.cos(lat) * math.sin(lon)
    uz = math.sin(lat)
    dx = sx_km - bx
    dy = sy_km - by
    dz = sz_km - bz
    return (dx * ux + dy * uy + dz * uz) > 0.0


def _orbit_3d_default_title(orbits: Sequence[Orbit3D]) -> str:
    if len(orbits) == 1:
        o = orbits[0]
        return f"{o.name or '<unnamed>'} [{o.catalog_number}] 3D orbit"
    names = ", ".join(
        f"{o.name or '<unnamed>'} [{o.catalog_number}]" for o in orbits
    )
    return f"3D orbits: {names}"


def _orbit_3d_subtitle(orbits: Sequence[Orbit3D], any_eop_degraded: bool) -> str:
    earliest = min(o.samples[0].time_utc for o in orbits)
    latest = max(o.samples[-1].time_utc for o in orbits)
    epochs = sorted(
        {o.tle_epoch_utc.strftime(_SUBTITLE_TIME_FORMAT) for o in orbits}
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
            f"{earliest.strftime(_SUBTITLE_TIME_FORMAT)} → "
            f"{latest.strftime(_SUBTITLE_TIME_FORMAT)}"
        ),
    ]
    if any_eop_degraded:
        parts.append("EOP degraded")
    return "  |  ".join(parts)


__all__ = ("build_ground_track_figure", "build_orbit_3d_figure")
