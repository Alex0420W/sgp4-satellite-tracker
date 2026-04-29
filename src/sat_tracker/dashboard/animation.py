"""60-frame client-side animation for the Streamlit dashboard.

The dashboard's "smooth motion" is fully client-side: the server
precomputes 60 satellite positions at 1-second intervals (one minute
of motion), packs them as plotly :class:`~plotly.graph_objects.Frame`
objects, and lets the browser interpolate between them at 1 FPS.

Why this matters: Streamlit's default rerun model is unsuitable for
sub-second updates — every ``st.rerun()`` triggers a WebSocket
round-trip and a full DOM repaint, which visibly stutters. By
precomputing a window of frames and animating client-side, the user
sees smooth motion in the browser while the server reruns only once
per minute (to fetch the next window). See
``app.py:_schedule_next_rerun`` for the server-side cadence.
"""

from __future__ import annotations

import logging
import math
import time
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional, Sequence

from sat_tracker.coordinates import CoordinateConverter
from sat_tracker.tle_fetcher import Tle
from sat_tracker.visualization.common import (
    Orbit3D,
    Track,
    precompute_orbit,
    precompute_track,
)

logger = logging.getLogger(__name__)

# 60 frames at 1-second steps = one minute of animated motion. Tuned
# in the spec — short enough that server rerun cadence (60s) keeps
# the browser within the precomputed window, long enough that one
# loop of the animation feels meaningful.
ANIMATION_FRAME_COUNT: int = 60
ANIMATION_STEP_SECONDS: float = 1.0
# Frame interval in milliseconds: 1000 ms gives true 1 Hz playback.
# Plotly's animation engine is responsive to this in the 100–2000ms
# range; below 100ms motion looks jittery on mid-spec hardware.
ANIMATION_FRAME_INTERVAL_MS: int = 1000


def minute_bucket(now: Optional[datetime] = None) -> int:
    """Integer minute-of-epoch — used as a deterministic cache key.

    The dashboard re-derives this on every script run; equal buckets
    mean cached frames are still valid, different buckets force a
    fresh precompute. Avoids any wall-clock-vs-cache-key drift.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    return int(now.timestamp() // 60)


def precompute_orbit_window(
    tle: Tle,
    converter: CoordinateConverter,
    *,
    minute_bucket_value: int,
    n_frames: int = ANIMATION_FRAME_COUNT,
    step_seconds: float = ANIMATION_STEP_SECONDS,
) -> Orbit3D:
    """Precompute one minute of 3D positions for animation.

    Window starts at the *bucket* boundary so two scripts agreeing on
    the same ``minute_bucket_value`` get identical samples. The
    one-orbit-width context polyline (used for the orbit trace) is
    *not* included here — call :func:`precompute_orbit` separately
    for that.

    Args:
        tle: Validated TLE.
        converter: Active coordinate converter.
        minute_bucket_value: Output of :func:`minute_bucket` for the
            target window. Lets callers cache by this scalar.
        n_frames: Number of animation frames. Default
            :data:`ANIMATION_FRAME_COUNT`.
        step_seconds: Seconds between consecutive frames. Default
            :data:`ANIMATION_STEP_SECONDS`.

    Returns:
        An :class:`Orbit3D` with exactly ``n_frames`` ECEF samples
        evenly spaced from the bucket boundary.
    """
    start = datetime.fromtimestamp(
        minute_bucket_value * 60, tz=timezone.utc
    )
    duration = step_seconds * (n_frames - 1)
    return precompute_orbit(
        tle,
        converter,
        start_utc=start,
        duration_seconds=duration + step_seconds * 0.01,
        step_seconds=step_seconds,
    )


def precompute_track_window(
    tle: Tle,
    converter: CoordinateConverter,
    *,
    minute_bucket_value: int,
    n_frames: int = ANIMATION_FRAME_COUNT,
    step_seconds: float = ANIMATION_STEP_SECONDS,
) -> Track:
    """Like :func:`precompute_orbit_window` but emits a 2D :class:`Track`."""
    start = datetime.fromtimestamp(
        minute_bucket_value * 60, tz=timezone.utc
    )
    duration = step_seconds * (n_frames - 1)
    return precompute_track(
        tle,
        converter,
        start_utc=start,
        duration_seconds=duration + step_seconds * 0.01,
        step_seconds=step_seconds,
    )


def attach_orbit_3d_animation(
    fig,
    orbits: Sequence[Orbit3D],
    *,
    frame_interval_ms: int = ANIMATION_FRAME_INTERVAL_MS,
    autoplay: bool = True,
    follow_catnr: Optional[int] = None,
) -> None:
    """Attach 60-frame animation to a 3D orbit figure.

    The figure is expected to come from
    :func:`sat_tracker.visualization.figures.build_orbit_3d_figure`
    with a ``current_time_utc`` set, so each orbit already has a
    "now" marker trace named ``"<label> now"``. This function:

    1. Locates each "now" marker by name suffix.
    2. Builds N frames where each frame swaps the marker positions
       for that satellite at that timestep.
    3. Wires plotly's animation controls so the figure auto-plays
       on browser load and loops indefinitely.

    Mutates ``fig`` in place. The figure's polyline / Earth /
    coastline traces stay static — only the per-orbit "now" markers
    animate. This is the cheapest possible animation cost: 60 frames
    × N satellites × 3 floats = trivial bandwidth.

    Args:
        fig: A :class:`plotly.graph_objects.Figure` from the 3D builder.
        orbits: The same orbit windows the figure was built from. Must
            have ``ANIMATION_FRAME_COUNT`` (or matching N) samples.
        frame_interval_ms: Animation step pacing in the browser.
        autoplay: When True, the figure starts playing immediately on
            load (no user click required). When False, the user must
            press the ▶ button.
    """
    import plotly.graph_objects as go

    if not orbits:
        return

    marker_indices = _find_now_marker_indices(fig, orbits)
    if not any(idx is not None for idx in marker_indices):
        logger.debug(
            "No 'now' marker traces found; animation skipped"
        )
        return

    # Determine frame count from the first orbit (all orbits share
    # the same time grid by construction).
    n_frames = len(orbits[0].samples)
    if n_frames < 2:
        return

    # Identify the orbit being followed (if any) so each frame can
    # carry a camera-update layout patch pointing at that satellite.
    follow_orbit: Optional[Orbit3D] = None
    if follow_catnr is not None:
        follow_orbit = next(
            (o for o in orbits if o.catalog_number == follow_catnr),
            None,
        )

    frame_objects: list = []
    slider_steps: list[dict] = []
    for i in range(n_frames):
        data_for_frame = []
        traces_for_frame: list[int] = []
        for orbit, marker_idx in zip(orbits, marker_indices):
            if marker_idx is None:
                continue
            # If this orbit has fewer samples than the master frame
            # count (defensive — shouldn't normally happen), clamp.
            sample = orbit.samples[min(i, len(orbit.samples) - 1)]
            r_km = math.sqrt(
                sample.x_km ** 2 + sample.y_km ** 2 + sample.z_km ** 2
            )
            data_for_frame.append(
                go.Scatter3d(
                    x=[sample.x_km],
                    y=[sample.y_km],
                    z=[sample.z_km],
                    hovertext=[
                        f"{orbit.name or '<unnamed>'} "
                        f"[{orbit.catalog_number}]<br>"
                        f"{sample.time_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}<br>"
                        f"|r|={r_km:.0f} km"
                    ],
                )
            )
            traces_for_frame.append(marker_idx)

        # Build the per-frame camera-eye position when in follow mode.
        # Eye sits on the same ECEF radial as the satellite at ~2.0
        # Earth-radii out (in plotly scene units 1.0 = one Earth
        # radius given aspectmode="data" and a 6378 km canonical
        # axis). The satellite ends up centred in the frame; Earth
        # fills the background.
        frame_layout = None
        if follow_orbit is not None:
            sample = follow_orbit.samples[
                min(i, len(follow_orbit.samples) - 1)
            ]
            r_km = math.sqrt(
                sample.x_km ** 2 + sample.y_km ** 2 + sample.z_km ** 2
            )
            if r_km > 0:
                eye_radius_norm = 2.0  # ~12,800 km ≈ 2 Earth radii
                scale = eye_radius_norm / (r_km / 6378.137)
                frame_layout = go.Layout(
                    scene_camera=dict(
                        eye=dict(
                            x=sample.x_km / 6378.137 * scale,
                            y=sample.y_km / 6378.137 * scale,
                            z=sample.z_km / 6378.137 * scale,
                        ),
                        center=dict(x=0, y=0, z=0),
                        up=dict(x=0, y=0, z=1),
                    )
                )

        frame_objects.append(
            go.Frame(
                data=data_for_frame,
                traces=traces_for_frame,
                layout=frame_layout,
                name=str(i),
            )
        )
        slider_steps.append(
            {
                "method": "animate",
                # Empty label: with 60 frames in a single slider, any
                # per-tick text collides into an unreadable mess. The
                # "t = HH:MM:SS" indicator above the slider already
                # shows the current frame's wall-clock time.
                "label": "",
                "args": [
                    [str(i)],
                    {
                        "mode": "immediate",
                        "frame": {
                            "duration": frame_interval_ms,
                            "redraw": True,
                        },
                        "transition": {"duration": 0},
                    },
                ],
            }
        )

    fig.frames = frame_objects

    # Auto-play: the play button's args are configured exactly the
    # same way as the manual ▶ click, but plotly's animation system
    # also supports a `transitioning` key on the layout that starts
    # animation immediately on load. The cleanest cross-version
    # approach is a hidden "auto-play" updatemenu with `visible: False`
    # -— but plotly's API doesn't support pre-clicking that menu, so
    # instead we attach an actual visible Play/Pause control plus
    # a tiny JS hook via Streamlit's plotly_chart that fires the play
    # action on mount. The simpler approach used here: include the
    # play+pause buttons, and rely on plotly's "loop" mode plus the
    # frame=fromcurrent restart behaviour.
    fig.update_layout(
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.04,
                "y": -0.06,
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
                                "frame": {
                                    "duration": frame_interval_ms,
                                    "redraw": True,
                                },
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                                "loop": True,
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
                                "frame": {
                                    "duration": 0,
                                    "redraw": False,
                                },
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "currentvalue": {
                    "prefix": "t = ",
                    "font": {"size": 11, "color": "#cdd6e0"},
                },
                "pad": {"t": 30, "b": 10, "l": 60, "r": 30},
                "len": 0.85,
                "x": 0.075,
                "y": -0.06,
                "steps": slider_steps,
                "bgcolor": "rgba(255,255,255,0.15)",
                "activebgcolor": "#ffd400",
                "tickcolor": "#cdd6e0",
                "font": {"color": "#cdd6e0"},
            }
        ],
        margin=dict(l=0, r=0, t=70, b=80),
    )

    if autoplay:
        # Attach a layout-level `_autoplay` key — handled by the
        # post-render JS hook we inject from `app.py` via Streamlit's
        # components system. plotly itself doesn't support
        # auto-starting animations from figure JSON alone; the hook
        # finds the play button and clicks it on mount.
        fig.update_layout(meta={"sat_tracker_autoplay": True})


def attach_ground_track_animation(
    fig,
    tracks: Sequence[Track],
    *,
    frame_interval_ms: int = ANIMATION_FRAME_INTERVAL_MS,
    autoplay: bool = True,
    follow_catnr: Optional[int] = None,
) -> None:
    """Attach 60-frame animation to a 2D ground-track figure.

    Mirrors :func:`attach_orbit_3d_animation` but for the 2D
    Scattergeo plot. The figure is expected to come from
    :func:`sat_tracker.visualization.figures.build_ground_track_figure`
    with a ``current_time_utc`` set, so each track already has a
    "now" marker (gold star) trace named ``"<label> now"``.

    Each frame swaps the per-track marker's ``lon`` and ``lat`` arrays
    for the next 1-second sample. Polylines, endpoints, and basemap
    stay static.

    Args:
        fig: Figure from :func:`build_ground_track_figure`.
        tracks: Track windows the figure was built from. Same time
            grid expected as the figure's source.
        frame_interval_ms: Browser playback pacing in milliseconds.
        autoplay: When True, layout meta carries the autoplay hint
            for the dashboard's JS bootstrap.
    """
    import plotly.graph_objects as go

    if not tracks:
        return

    marker_indices = _find_now_marker_indices_2d(fig, tracks)
    if not any(idx is not None for idx in marker_indices):
        logger.debug(
            "No 2D 'now' marker traces found; animation skipped"
        )
        return

    n_frames = len(tracks[0].samples)
    if n_frames < 2:
        return

    follow_track: Optional[Track] = None
    if follow_catnr is not None:
        follow_track = next(
            (t for t in tracks if t.catalog_number == follow_catnr),
            None,
        )

    frame_objects: list = []
    slider_steps: list[dict] = []
    for i in range(n_frames):
        data_for_frame = []
        traces_for_frame: list[int] = []
        for track, marker_idx in zip(tracks, marker_indices):
            if marker_idx is None:
                continue
            sample = track.samples[min(i, len(track.samples) - 1)]
            data_for_frame.append(
                go.Scattergeo(
                    lon=[sample.longitude_deg],
                    lat=[sample.latitude_deg],
                    hovertext=[
                        f"{track.name or '<unnamed>'} "
                        f"[{track.catalog_number}]<br>"
                        f"{sample.time_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}<br>"
                        f"lat {sample.latitude_deg:+.3f}°, "
                        f"lon {sample.longitude_deg:+.3f}°<br>"
                        f"alt {sample.altitude_km:.1f} km"
                    ],
                )
            )
            traces_for_frame.append(marker_idx)

        # 2D follow: per-frame layout patches that re-center the
        # geo subplot on the sub-satellite point.
        frame_layout = None
        if follow_track is not None:
            sample = follow_track.samples[
                min(i, len(follow_track.samples) - 1)
            ]
            frame_layout = go.Layout(
                geo=dict(
                    center=dict(
                        lat=sample.latitude_deg,
                        lon=sample.longitude_deg,
                    ),
                    projection=dict(
                        rotation=dict(
                            lat=0,
                            lon=sample.longitude_deg,
                        ),
                    ),
                ),
            )

        frame_objects.append(
            go.Frame(
                data=data_for_frame,
                traces=traces_for_frame,
                layout=frame_layout,
                name=str(i),
            )
        )
        slider_steps.append(
            {
                "method": "animate",
                "label": "",  # see comment in attach_orbit_3d_animation
                "args": [
                    [str(i)],
                    {
                        "mode": "immediate",
                        "frame": {
                            "duration": frame_interval_ms,
                            "redraw": True,
                        },
                        "transition": {"duration": 0},
                    },
                ],
            }
        )

    fig.frames = frame_objects

    fig.update_layout(
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.04,
                "y": -0.06,
                "xanchor": "right",
                "yanchor": "top",
                "pad": {"t": 30, "r": 10},
                "bgcolor": "rgba(255,255,255,0.7)",
                "font": {"color": "#222"},
                "buttons": [
                    {
                        "label": "▶",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "mode": "immediate",
                                "frame": {
                                    "duration": frame_interval_ms,
                                    "redraw": True,
                                },
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                                "loop": True,
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
                                "frame": {
                                    "duration": 0,
                                    "redraw": False,
                                },
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "currentvalue": {
                    "prefix": "t = ",
                    "font": {"size": 11, "color": "#444"},
                },
                "pad": {"t": 30, "b": 10, "l": 60, "r": 30},
                "len": 0.85,
                "x": 0.075,
                "y": -0.06,
                "steps": slider_steps,
                "bgcolor": "rgba(255,255,255,0.7)",
                "activebgcolor": "#ffd400",
                "tickcolor": "#444",
                "font": {"color": "#444"},
            }
        ],
        margin=dict(l=10, r=10, t=80, b=80),
    )

    if autoplay:
        fig.update_layout(meta={"sat_tracker_autoplay": True})


def _find_now_marker_indices(
    fig, orbits: Iterable[Orbit3D]
) -> list[Optional[int]]:
    """Locate per-orbit "now" markers by name suffix in fig.data."""
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


def _find_now_marker_indices_2d(
    fig, tracks: Iterable[Track]
) -> list[Optional[int]]:
    """Locate per-track "now" markers by name suffix in fig.data (2D)."""
    indices: list[Optional[int]] = []
    for track in tracks:
        label = f"{track.name or '<unnamed>'} [{track.catalog_number}]"
        target = f"{label} now"
        found: Optional[int] = None
        for i, trace in enumerate(fig.data):
            if getattr(trace, "name", None) == target:
                found = i
                break
        indices.append(found)
    return indices


def seconds_until_next_minute_bucket() -> float:
    """Wall-clock seconds remaining until the *next* ``minute_bucket``.

    Used by the dashboard's server-side rerun loop: sleep this many
    seconds, then call ``st.rerun()``. Aligning to bucket boundaries
    (rather than fixed 60s sleeps) means cached frames stay maximally
    fresh — the rerun fires right when the cache is about to expire,
    not 30s before or after.
    """
    now = time.time()
    return max(1.0, 60.0 - (now % 60.0))


__all__ = (
    "ANIMATION_FRAME_COUNT",
    "ANIMATION_STEP_SECONDS",
    "ANIMATION_FRAME_INTERVAL_MS",
    "minute_bucket",
    "precompute_orbit_window",
    "precompute_track_window",
    "attach_orbit_3d_animation",
    "attach_ground_track_animation",
    "seconds_until_next_minute_bucket",
)
