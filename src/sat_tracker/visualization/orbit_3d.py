"""Interactive 3D orbit rendering via plotly (file output + slider).

Thin disk-output shim around
:func:`sat_tracker.visualization.figures.build_orbit_3d_figure`. The
figure-construction logic (Earth sphere, coastlines, graticule, orbit
traces, ground station) lives in ``figures.py`` so the Streamlit
dashboard can import the in-memory figure directly. This module wraps
that builder with two CLI-only concerns: suffix-routed disk output and
the HTML-only time-slider machinery.

The dashboard composes its own client-side animation; it does not use
the ``_attach_time_slider`` path here.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence, Union

from sat_tracker.passes import GroundStation
from sat_tracker.visualization.common import Orbit3D
from sat_tracker.visualization.figures import build_orbit_3d_figure

logger = logging.getLogger(__name__)

_HOVER_TIME_FORMAT = "%Y-%m-%d %H:%M:%S UTC"


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
    """Render one or more 3D orbits to an interactive HTML file or static export.

    Args:
        orbits: A single :class:`Orbit3D` or a sequence of them.
        output_path: Destination file. Suffix selects format: ``.html``
            (default — interactive plot) or any ``kaleido``-supported
            static format (``.png``, ``.svg``, ``.pdf``).
        title: Optional title override.
        colors: Optional per-orbit colour list. Defaults to
            :data:`~sat_tracker.visualization.common.DEFAULT_TRACK_COLORS`.
        line_width: Orbit polyline width.
        current_time_utc: Optional "now" instant. For each orbit, marks
            the sample closest to this time with a star.
        ground_station: Optional :class:`GroundStation`. Renders a
            station marker on Earth's surface and (if
            ``show_los_line`` and ``current_time_utc`` are both set)
            a line of sight when the satellite is above the local
            horizon.
        show_los_line: See above.
        time_slider: When ``True`` and writing HTML, attach a slider
            that scrubs the "now" marker. Static exports always strip
            the slider.
        width, height: Pixel size for the figure.
        show_coastlines: Overlay Natural Earth coastlines on the sphere.
        show_graticule: Overlay a faint 30° lat/lon grid.
        camera: ``"default"`` for the static (1.6, 1.6, 0.8) eye
            position; ``"follow"`` to position along the same vector
            as the "now" sample of the first orbit.

    Returns:
        Resolved :class:`~pathlib.Path` of the written file.

    Raises:
        ValueError: If ``orbits`` is empty or any orbit has no samples.
        ImportError: If plotly or cartopy are not installed.
    """
    output_path = Path(output_path)
    suffix = output_path.suffix.lower()
    if suffix == "":
        output_path = output_path.with_suffix(".html")
        suffix = ".html"
    is_static = suffix not in {".html", ".htm"}

    # Normalize first so empty-input/zero-sample paths raise ValueError
    # before we touch any plotly internals (matches the builder's
    # error contract; the existing tests pin it).
    if isinstance(orbits, Orbit3D):
        orbit_list: list[Orbit3D] = [orbits]
    else:
        orbit_list = list(orbits)
    if not orbit_list:
        raise ValueError("orbits must contain at least one Orbit3D")
    for orbit in orbit_list:
        if not orbit.samples:
            raise ValueError(
                f"orbit for catnr={orbit.catalog_number} has no samples"
            )

    # Resolve camera: "follow" depends on current_time_utc and the first
    # orbit's geometry, so it has to happen here rather than in the
    # builder (which is dashboard-friendly and shouldn't know about the
    # CLI's "follow" affordance).
    camera_eye = _resolve_camera_eye(camera, orbit_list, current_time_utc)

    # Slider needs "now" markers it can scrub. If the caller asked for a
    # slider but didn't pass current_time_utc, synthesize one pointing
    # at the middle of the first orbit's window so the builder still
    # emits the marker traces.
    builder_current = current_time_utc
    if time_slider and not is_static and builder_current is None:
        first = orbit_list[0]
        mid = first.samples[len(first.samples) // 2]
        builder_current = mid.time_utc

    fig = build_orbit_3d_figure(
        orbit_list,
        title=title,
        colors=colors,
        line_width=line_width,
        current_time_utc=builder_current,
        ground_station=ground_station,
        show_los_line=show_los_line,
        show_coastlines=show_coastlines,
        show_graticule=show_graticule,
        width=width,
        height=height,
        camera_eye=camera_eye,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not is_static and time_slider:
        # The builder emits per-orbit "now" markers as traces named
        # "<label> now". Find them by name and attach the slider.
        marker_indices = _find_now_marker_indices(fig, orbit_list)
        if any(idx is not None for idx in marker_indices):
            _attach_time_slider(fig, orbit_list, marker_indices)
            # Reserve a bit of bottom margin for the slider gutter.
            fig.update_layout(margin=dict(l=0, r=0, t=70, b=80))

    if is_static:
        fig.write_image(str(output_path), width=width, height=height)
    else:
        fig.write_html(str(output_path), include_plotlyjs="cdn")

    logger.debug(
        "Rendered 3D orbit to %s", output_path
    )
    return output_path


# -- Camera resolution --------------------------------------------------------


def _resolve_camera_eye(
    mode: str,
    orbits: Sequence[Orbit3D],
    current_time_utc: Optional[datetime],
) -> Optional[tuple[float, float, float]]:
    """Translate the CLI ``camera`` keyword into a builder ``camera_eye``.

    ``"default"`` returns ``None`` so the builder picks its own (1.6,
    1.6, 0.8). ``"follow"`` projects along the satellite's "now" radial
    out to ~2.2 Earth radii with a 10° downward tilt for inclination
    visibility.
    """
    if mode == "default" or current_time_utc is None:
        return None
    if mode != "follow":
        raise ValueError(
            f"camera must be 'default' or 'follow', got {mode!r}"
        )
    if not orbits:
        return None
    first = orbits[0]
    sample = _closest_sample(first.samples, current_time_utc)
    if sample is None:
        return None
    r = math.sqrt(sample.x_km ** 2 + sample.y_km ** 2 + sample.z_km ** 2)
    if r == 0:
        return None
    nx, ny, nz = sample.x_km / r, sample.y_km / r, sample.z_km / r
    horiz = math.hypot(nx, ny)
    tilt = math.radians(10.0)
    new_z = nz * math.cos(tilt) - horiz * math.sin(tilt)
    scale_xy = (nz * math.sin(tilt) + horiz * math.cos(tilt)) / max(horiz, 1e-9)
    return (nx * scale_xy * 2.2, ny * scale_xy * 2.2, new_z * 2.2)


# -- Slider machinery ---------------------------------------------------------


def _find_now_marker_indices(
    fig, orbits: Sequence[Orbit3D]
) -> list[Optional[int]]:
    """Locate each orbit's "now" marker trace by name suffix.

    The figure builder names marker traces ``"<label> now"`` where
    ``label`` is ``"{name} [{catnr}]"``. Finding them this way decouples
    the slider from the builder's internal trace ordering.
    """
    indices: list[Optional[int]] = []
    for orbit in orbits:
        label = f"{orbit.name or '<unnamed>'} [{orbit.catalog_number}]"
        target = f"{label} now"
        found: Optional[int] = None
        for i, trace in enumerate(fig.data):
            if getattr(trace, "name", None) == target:
                found = i
                break
        indices.append(found)
    return indices


def _attach_time_slider(
    fig,
    orbits: Sequence[Orbit3D],
    now_marker_indices: Sequence[Optional[int]],
) -> None:
    """Attach a time slider that scrubs each orbit's "now" marker."""
    if not orbits:
        return
    first = orbits[0]
    n_frames = len(first.samples)

    import plotly.graph_objects as go

    frame_objects = []
    slider_steps: list[dict] = []
    for i, primary_sample in enumerate(first.samples):
        when = primary_sample.time_utc
        data_for_frame = []
        traces_for_frame = []
        for orbit, marker_idx in zip(orbits, now_marker_indices):
            if marker_idx is None:
                continue
            sample = _closest_sample(orbit.samples, when)
            if sample is None:
                data_for_frame.append(go.Scatter3d(x=[], y=[], z=[]))
            else:
                data_for_frame.append(
                    go.Scatter3d(
                        x=[sample.x_km],
                        y=[sample.y_km],
                        z=[sample.z_km],
                        hovertext=[
                            f"{orbit.name or '<unnamed>'} [{orbit.catalog_number}]<br>"
                            f"{sample.time_utc.strftime(_HOVER_TIME_FORMAT)}<br>"
                            f"|r|={math.sqrt(sample.x_km ** 2 + sample.y_km ** 2 + sample.z_km ** 2):.0f} km"
                        ],
                    )
                )
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

    fig.update_layout(
        sliders=[
            {
                "active": n_frames // 2 if n_frames > 0 else 0,
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


def _closest_sample(samples, when: datetime):
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


__all__ = ("render_orbit_3d",)
