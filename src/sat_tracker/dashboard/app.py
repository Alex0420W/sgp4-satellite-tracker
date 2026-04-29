"""Streamlit dashboard entry point — step (d): sidebar + multi-sat tracking.

Single page with two tabs (3D / 2D) and a satellite-picker sidebar.
ISS is tracked by default; users can add/remove satellites via:

* curated-list search box
* free-form NORAD catalog number entry
* CelesTrak group quick-buttons (Stations, GPS-OPS, Starlink top-50, …)

All tracked satellites animate together at 1 Hz over a shared
60-frame window. Colours come from the Paul-Tol vibrant palette
(colourblind-safe distinguishability) keyed by tracked-list index.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone

import streamlit as st
import streamlit.components.v1 as components

from sat_tracker.config import Config, configure_logging, load_config
from sat_tracker.coordinates import CoordinateConverter
from sat_tracker.dashboard.animation import (
    ANIMATION_FRAME_INTERVAL_MS,
    attach_ground_track_animation,
    attach_orbit_3d_animation,
    minute_bucket,
    precompute_orbit_window,
    precompute_track_window,
    seconds_until_next_minute_bucket,
)
from sat_tracker.dashboard.components import (
    MAX_TRACKED_SATELLITES,
    TOL_VIBRANT_PALETTE,
    get_active_station,
    get_following,
    init_session_state,
    render_sidebar,
    set_following,
)
from sat_tracker.passes import GroundStation, Pass, PassPredictor
from sat_tracker.tle_fetcher import TleFetcher, TleFetchError
from sat_tracker.visualization.common import (
    Orbit3D,
    Track,
    default_window_seconds,
    precompute_orbit,
    precompute_track,
)
from sat_tracker.visualization.figures import (
    build_ground_track_figure,
    build_orbit_3d_figure,
)

logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner=False)
def _config() -> Config:
    """Load runtime config, force a writable cache dir on cloud hosts.

    The default cache_dir is ``./data`` (relative to cwd). Streamlit
    Cloud's working directory is sometimes read-only at runtime, which
    breaks ``TleFetcher``'s atomic-write into the cache. We re-home the
    cache to ``$TMPDIR/sat_tracker_cache`` on every run — this directory
    is reliably writable on every Linux/macOS host and survives across
    reruns within the same container.
    """
    import tempfile
    from dataclasses import replace
    from pathlib import Path

    cfg = load_config()
    cache_dir = Path(tempfile.gettempdir()) / "sat_tracker_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cfg = replace(cfg, cache_dir=cache_dir)
    configure_logging(cfg)
    return cfg


@st.cache_resource(show_spinner=False)
def _converter() -> CoordinateConverter:
    """CoordinateConverter singleton.

    Constructs once per server process. If the live IERS fetch fails,
    falls back to Skyfield's bundled timescale (``eop_degraded=True``)
    and schedules a single-shot background retry — when that retry
    succeeds, ``_converter()``'s cache entry is invalidated so the
    *next* script run picks up a fresh timescale.
    """
    converter = CoordinateConverter(_config())
    if converter.eop_degraded:
        _schedule_eop_refresh()
    return converter


# Tracks whether a background EOP refresh thread is in flight, so we
# don't spawn duplicates across reruns. ``threading.Event`` instead of
# a plain bool because Streamlit's script reruns share the module-level
# state but run on separate worker threads.
_eop_refresh_in_progress = threading.Event()


def _schedule_eop_refresh() -> None:
    """Spawn one background thread to attempt a fresh IERS fetch.

    The retry runs Skyfield's loader once with ``builtin=False``. On
    success, we clear the ``_converter`` cache so the next script
    rerun rebuilds with the fresh timescale; the ``EOP: bundled
    (refreshing…)`` badge then flips to ``EOP: fresh (IERS)``.
    """
    if _eop_refresh_in_progress.is_set():
        return
    _eop_refresh_in_progress.set()

    def _refresh():
        try:
            from skyfield.api import Loader

            loader = Loader(str(_config().cache_dir))
            loader.timescale(builtin=False)
            logger.info("Background EOP refresh succeeded")
            # Drop the cached converter so the next rerun rebuilds.
            _converter.clear()
        except Exception as exc:
            logger.warning("Background EOP refresh failed: %s", exc)
        finally:
            _eop_refresh_in_progress.clear()

    threading.Thread(target=_refresh, daemon=True).start()


@st.cache_resource(show_spinner=False)
def _fetcher() -> TleFetcher:
    return TleFetcher(_config())


@st.cache_resource(show_spinner=False)
def _pass_predictor() -> PassPredictor:
    """PassPredictor singleton — reuses the loaded Skyfield Timescale."""
    return PassPredictor(_config(), _converter())


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_tle(catnr: int, hour_bucket: int):
    """Cache TLE by (catnr, hour-bucket); ``hour_bucket`` is invalidation-only."""
    return _fetcher().fetch(catnr)


@st.cache_data(ttl=120, show_spinner=False)
def _cached_orbit_context(catnr: int, hour_bucket: int) -> Orbit3D:
    tle = _cached_tle(catnr, hour_bucket)
    period = default_window_seconds(tle)
    start = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(
        seconds=period / 2
    )
    return precompute_orbit(
        tle, _converter(), start_utc=start, duration_seconds=period
    )


@st.cache_data(ttl=120, show_spinner=False)
def _cached_track_context(catnr: int, hour_bucket: int) -> Track:
    tle = _cached_tle(catnr, hour_bucket)
    period = default_window_seconds(tle)
    start = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(
        seconds=period / 2
    )
    return precompute_track(
        tle, _converter(), start_utc=start, duration_seconds=period
    )


@st.cache_data(ttl=90, show_spinner=False)
def _cached_animation_orbit(catnr: int, minute_bucket_value: int) -> Orbit3D:
    tle = _cached_tle(catnr, minute_bucket_value // 60)
    return precompute_orbit_window(
        tle, _converter(), minute_bucket_value=minute_bucket_value
    )


@st.cache_data(ttl=90, show_spinner=False)
def _cached_animation_track(catnr: int, minute_bucket_value: int) -> Track:
    tle = _cached_tle(catnr, minute_bucket_value // 60)
    return precompute_track_window(
        tle, _converter(), minute_bucket_value=minute_bucket_value
    )


@st.cache_data(ttl=86_400, show_spinner=False)
def _cached_passes(
    catnr: int,
    station_repr: tuple,
    day_bucket: int,
) -> list[Pass]:
    """Predict next 24h of passes — cached per (catnr, station, day-bucket)."""
    del day_bucket  # invalidation-only
    lat, lon, alt, name = station_repr
    station = GroundStation(
        latitude_deg=lat, longitude_deg=lon, altitude_km=alt, name=name
    )
    tle = _cached_tle(catnr, int(datetime.now(timezone.utc).timestamp() // 3600))
    return _pass_predictor().predict_passes(
        tle,
        station,
        start_utc=datetime.now(timezone.utc),
        hours=24.0,
    )


def _passes_provider(
    catnr: int, station: GroundStation, day_bucket: int
) -> list[Pass]:
    """Adapt the cached pass-prediction call into the components callback."""
    station_repr = (
        station.latitude_deg,
        station.longitude_deg,
        station.altitude_km,
        station.name,
    )
    return _cached_passes(catnr, station_repr, day_bucket)


def _follow_eye_for(sample) -> tuple[float, float, float]:
    """Compute a plotly scene-camera eye placing the satellite centred.

    Eye sits on the same ECEF radial as the satellite at ~2.0 Earth
    radii from origin. Plotly's scene-camera "eye" is in normalized
    units where 1.0 = one Earth radius given ``aspectmode="data"``
    on a 6378 km axis.
    """
    import math
    r_km = math.sqrt(
        sample.x_km ** 2 + sample.y_km ** 2 + sample.z_km ** 2
    )
    if r_km == 0:
        return (1.45, 0.65, 0.55)
    eye_radius_norm = 2.0
    scale = eye_radius_norm / (r_km / 6378.137)
    return (
        sample.x_km / 6378.137 * scale,
        sample.y_km / 6378.137 * scale,
        sample.z_km / 6378.137 * scale,
    )


def _animation_bootstrap_js(
    div_id: str,
    storage_key: str,
    *,
    detect_user_camera_drag: bool = False,
) -> str:
    """Generate the inline JS that boots animation + (optionally) detects
    a user camera drag in follow mode.

    When ``detect_user_camera_drag`` is True, a ``plotly_relayout`` listener
    fires the moment the user manually rotates / pans / zooms the scene.
    The handler sets a sessionStorage flag that Streamlit's auto-rerun
    cycle reads on its next pass to clear ``following`` — i.e. manual
    interaction unlocks the camera (per fork (g)/option (a)).
    """
    drag_listener = (
        f"""
            // Detect user camera drag: any plotly_relayout that
            // mentions scene.camera keys is treated as manual control.
            // We set a sessionStorage flag; the Streamlit-side rerun
            // (every minute_bucket boundary) checks this flag and
            // clears the follow state.
            var FOLLOW_RELEASE_KEY = "sat-tracker-follow-release";
            // Ignore relayout events fired by our own animation
            // frames — those are programmatic, not user-driven.
            // Plotly sets `event.user` on user-initiated relayouts
            // in newer versions; older fall back to
            // `eventData["scene.camera"]` heuristic.
            gd.on("plotly_relayout", function(eventData) {{
                if (!eventData) return;
                // Skip animation-driven camera frames.
                if (gd._transitioning || gd._transitionData?._inTransition) {{
                    return;
                }}
                var hasCameraKey = false;
                for (var k in eventData) {{
                    if (k.indexOf("scene.camera") === 0
                        || k === "scene.camera") {{
                        hasCameraKey = true;
                        break;
                    }}
                }}
                if (hasCameraKey) {{
                    sessionStorage.setItem(FOLLOW_RELEASE_KEY, "1");
                    // Inform the parent Streamlit page so it can rerun
                    // and clear the follow state immediately.
                    try {{
                        window.parent.postMessage({{
                            type: "sat-tracker-follow-release"
                        }}, "*");
                    }} catch (e) {{}}
                }}
            }});
        """
        if detect_user_camera_drag
        else ""
    )
    return f"""
    <script>
    (function() {{
        var STORAGE_KEY = {storage_key!r};
        function startAnim() {{
            var gd = document.getElementById({div_id!r});
            if (!gd || !window.Plotly) {{ return setTimeout(startAnim, 100); }}
            if (!gd._fullLayout) {{ return setTimeout(startAnim, 100); }}

            var frames = (gd._transitionData || gd.frames || {{}})._frames
                       || gd.frames || [];
            var nFrames = frames.length;
            var saved = sessionStorage.getItem(STORAGE_KEY);
            var startIdx = 0;
            if (saved !== null) {{
                var idx = parseInt(saved, 10);
                if (!isNaN(idx) && idx >= 0 && idx < nFrames) {{ startIdx = idx; }}
            }}
            gd.on("plotly_animatingframe", function(ev) {{
                if (ev && ev.frame && ev.frame.name !== undefined) {{
                    sessionStorage.setItem(STORAGE_KEY, ev.frame.name);
                }}
            }});
            {drag_listener}
            var jump = (startIdx > 0)
                ? window.Plotly.animate(gd, [String(startIdx)], {{
                    mode: "immediate",
                    frame: {{duration: 0, redraw: true}},
                    transition: {{duration: 0}},
                }})
                : Promise.resolve();
            jump.then(function() {{
                window.Plotly.animate(gd, null, {{
                    frame: {{duration: {ANIMATION_FRAME_INTERVAL_MS}, redraw: true}},
                    transition: {{duration: 0}},
                    mode: "immediate",
                    fromcurrent: true,
                    loop: true,
                }});
            }});
        }}
        if (document.readyState === "complete") {{ startAnim(); }}
        else {{ window.addEventListener("load", startAnim); }}
    }})();
    </script>
    """


def _render_3d_tab(
    context_orbits: list[Orbit3D],
    anim_orbits: list[Orbit3D],
) -> None:
    if not context_orbits:
        _render_empty_state("3D")
        return
    follow_catnr = get_following()
    first_anim_sample = anim_orbits[0].samples[0]
    # If we are following a satellite, seed the figure's *initial*
    # camera at that satellite's first-frame radial so the figure
    # opens already framed correctly. The animation frames then
    # update the camera per step.
    initial_camera = (1.45, 0.65, 0.55)
    if follow_catnr is not None:
        followed = next(
            (o for o in anim_orbits if o.catalog_number == follow_catnr),
            None,
        )
        if followed is not None and followed.samples:
            initial_camera = _follow_eye_for(followed.samples[0])
    fig = build_orbit_3d_figure(
        context_orbits,
        current_time_utc=first_anim_sample.time_utc,
        height=720,
        camera_eye=initial_camera,
        line_width=5.0 if len(context_orbits) == 1 else 3.5,
        colors=TOL_VIBRANT_PALETTE,
    )
    attach_orbit_3d_animation(
        fig, anim_orbits, follow_catnr=follow_catnr
    )
    fig.update_layout(autosize=True, width=None)
    if len(context_orbits) > 7:
        # Legend would dominate at this size; chip list in the sidebar
        # is the source of truth.
        fig.update_layout(showlegend=False)

    div_id = f"sat-tracker-3d-{uuid.uuid4().hex}"
    fig_html = fig.to_html(
        include_plotlyjs="cdn",
        full_html=False,
        auto_play=False,
        div_id=div_id,
        config={"displaylogo": False, "scrollZoom": True, "responsive": True},
    )
    components.html(
        fig_html
        + _animation_bootstrap_js(
            div_id,
            "sat-tracker-anim-frame",
            detect_user_camera_drag=follow_catnr is not None,
        ),
        height=820,
        scrolling=False,
    )


def _render_2d_tab(
    context_tracks: list[Track],
    anim_tracks: list[Track],
) -> None:
    if not context_tracks:
        _render_empty_state("2D")
        return
    import plotly.graph_objects as go

    follow_catnr = get_following()
    first_anim_sample = anim_tracks[0].samples[0]
    fig = build_ground_track_figure(
        context_tracks,
        current_time_utc=first_anim_sample.time_utc,
        height=560,
        line_width=3.0 if len(context_tracks) <= 3 else 2.0,
        colors=TOL_VIBRANT_PALETTE,
    )

    # Overlay the active ground station as a small marker so users
    # can see "I'm tracking passes from this point on Earth."
    station = get_active_station()
    fig.add_trace(
        go.Scattergeo(
            lon=[station.longitude_deg],
            lat=[station.latitude_deg],
            mode="markers",
            marker=dict(
                size=11,
                color="#39ff88",
                line=dict(color="white", width=1.5),
                symbol="diamond",
            ),
            name=f"GS: {station.name or 'station'}",
            hovertext=(
                f"Ground station<br>"
                f"{station.name or '(unnamed)'}<br>"
                f"{station.latitude_deg:+.4f}°, {station.longitude_deg:+.4f}°"
            ),
            hoverinfo="text",
            showlegend=True,
        )
    )
    # 2D follow: re-centre the geo subplot on the followed sub-point
    # immediately so the figure opens framed correctly. Per-frame
    # layout patches advance the centre during animation.
    if follow_catnr is not None:
        followed = next(
            (t for t in anim_tracks if t.catalog_number == follow_catnr),
            None,
        )
        if followed is not None and followed.samples:
            s = followed.samples[0]
            fig.update_geos(
                center=dict(lat=s.latitude_deg, lon=s.longitude_deg),
                projection_rotation=dict(lat=0, lon=s.longitude_deg),
            )
    attach_ground_track_animation(
        fig, anim_tracks, follow_catnr=follow_catnr
    )
    fig.update_layout(autosize=True, width=None)
    if len(context_tracks) > 7:
        fig.update_layout(showlegend=False)

    div_id = f"sat-tracker-2d-{uuid.uuid4().hex}"
    fig_html = fig.to_html(
        include_plotlyjs="cdn",
        full_html=False,
        auto_play=False,
        div_id=div_id,
        config={"displaylogo": False, "scrollZoom": True, "responsive": True},
    )
    components.html(
        fig_html + _animation_bootstrap_js(div_id, "sat-tracker-anim-frame"),
        height=680,
        scrolling=False,
    )


def _render_following_banner(context_orbits: list[Orbit3D]) -> None:
    """Top-of-page badge + Stop button when in follow mode."""
    follow = get_following()
    if follow is None:
        return
    name = next(
        (o.name for o in context_orbits if o.catalog_number == follow), None
    )
    label = f"{name} [{follow}]" if name else f"catnr {follow}"
    col_msg, col_btn = st.columns([5, 1])
    with col_msg:
        st.markdown(
            f"""
            <div style="
                background:rgba(255, 212, 0, 0.12);
                border-left:4px solid #ffd400;
                padding:8px 14px;
                border-radius:4px;
                font-size:14px;
                margin-bottom:8px;
            "><b style="color:#ffd400">📍 Following:</b>
                {label} — camera locked. Drag to override (or click Stop).</div>
            """,
            unsafe_allow_html=True,
        )
    with col_btn:
        if st.button("Stop following", use_container_width=True):
            set_following(None)
            st.rerun()


def _render_empty_state(view_label: str) -> None:
    st.info(
        f"No satellites are being tracked. "
        f"Add one from the sidebar to populate the {view_label} view."
    )


def _resolve_satellites(
    catnrs: list[int],
    bucket: int,
) -> tuple[
    list[Orbit3D],
    list[Track],
    list[Orbit3D],
    list[Track],
    list[tuple[int, str]],
]:
    """Build the four parallel lists of figure inputs.

    Returns:
        Five-tuple (context_orbits, context_tracks, anim_orbits,
        anim_tracks, failed). The ``failed`` list holds
        ``(catnr, message)`` pairs so the UI can display the actual
        upstream error rather than just "fetch failed".
    """
    context_orbits: list[Orbit3D] = []
    context_tracks: list[Track] = []
    anim_orbits: list[Orbit3D] = []
    anim_tracks: list[Track] = []
    failed: list[tuple[int, str]] = []
    for catnr in catnrs:
        try:
            context_orbits.append(_cached_orbit_context(catnr, bucket // 60))
            context_tracks.append(_cached_track_context(catnr, bucket // 60))
            anim_orbits.append(_cached_animation_orbit(catnr, bucket))
            anim_tracks.append(_cached_animation_track(catnr, bucket))
        except TleFetchError as exc:
            logger.warning("TLE fetch failed for catnr=%d: %s", catnr, exc)
            failed.append((catnr, str(exc)))
        except Exception as exc:  # pragma: no cover — defensive
            logger.exception("Unexpected error for catnr=%d", catnr)
            failed.append((catnr, f"{type(exc).__name__}: {exc}"))
    return context_orbits, context_tracks, anim_orbits, anim_tracks, failed


def _render() -> None:
    st.set_page_config(
        page_title="sat-tracker",
        page_icon="🛰",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()
    render_sidebar(fetcher=_fetcher(), passes_provider=_passes_provider)

    st.title("sat-tracker")
    st.caption(
        "Live SGP4 satellite tracking. ECEF 3D and equirectangular 2D "
        "views share the same 60-frame / 1 Hz animation; the server "
        "refreshes the window once per minute."
    )

    catnrs: list[int] = list(st.session_state.get("tracked", []))
    bucket = minute_bucket()

    if catnrs:
        with st.spinner(
            f"Fetching TLEs + propagating {len(catnrs)} satellite(s)…"
        ):
            (
                context_orbits,
                context_tracks,
                anim_orbits,
                anim_tracks,
                failed,
            ) = _resolve_satellites(catnrs, bucket)
        if failed:
            # Show prominent inline error with the actual upstream
            # message. Toasts auto-dismiss after a few seconds and are
            # easy to miss; this stays on screen until the next rerun
            # or until the user removes the offending satellite. The
            # message includes the underlying CelesTrak / network
            # error so we can diagnose remotely from a screenshot.
            failed_lines = "\n".join(
                f"- **catnr {c}** — {msg}" for c, msg in failed
            )
            st.error(
                "TLE fetch failed for the following satellites. "
                "They are skipped in the figures:\n\n"
                f"{failed_lines}\n\n"
                "Most likely causes: CelesTrak rate-limit, network "
                "outage on the host, or a catnr CelesTrak doesn't "
                "carry (e.g. an L2 / heliocentric mission).",
                icon="⚠️",
            )
    else:
        context_orbits = []
        context_tracks = []
        anim_orbits = []
        anim_tracks = []

    _render_following_banner(context_orbits)

    tab_3d, tab_2d = st.tabs(["🌐 3D orbit", "🗺️ 2D ground track"])
    with tab_3d:
        _render_3d_tab(context_orbits, anim_orbits)
    with tab_2d:
        _render_2d_tab(context_tracks, anim_tracks)

    # Status row.
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Tracked satellites", value=len(context_orbits))
    with col2:
        if anim_orbits:
            st.metric(
                label="Animation window starts",
                value=anim_orbits[0].samples[0].time_utc.strftime("%H:%M:%S"),
            )
        else:
            st.metric(label="Animation window starts", value="—")
    with col3:
        any_degraded = any(o.eop_degraded for o in context_orbits)
        if any_degraded:
            label_value = (
                "bundled (refreshing…)"
                if _eop_refresh_in_progress.is_set()
                else "bundled (offline)"
            )
        else:
            label_value = "fresh (IERS)"
        st.metric(label="EOP", value=label_value)

    if any(o.eop_degraded for o in context_orbits):
        if _eop_refresh_in_progress.is_set():
            st.info(
                "EOP using Skyfield's bundled approximation while a fresh "
                "IERS fetch runs in the background. Indicator will flip to "
                "'fresh (IERS)' on completion.",
                icon="🔄",
            )
        else:
            st.warning(
                "EOP data is the bundled (offline) approximation. Position "
                "accuracy may be slightly degraded; see logs.",
                icon="⚠️",
            )

    sleep_seconds = seconds_until_next_minute_bucket()
    logger.debug(
        "Sleeping %.1fs until next minute_bucket rerun", sleep_seconds
    )
    time.sleep(sleep_seconds)
    st.rerun()


_render()
