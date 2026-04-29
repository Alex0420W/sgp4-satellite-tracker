"""Reusable Streamlit UI widgets for the dashboard.

Each public function takes only what it needs (Streamlit's
``session_state`` is the source of truth for tracked satellites; this
module reads/mutates ``session_state["tracked"]`` directly).

The curated satellite list is loaded once per process from
``static/curated_satellites.json``. Categories there are pedagogical
groupings, not strict CelesTrak group memberships — the
"Add CelesTrak group" buttons separately call ``TleFetcher.fetch_group``
which queries CelesTrak's actual ``GROUP=`` endpoint.
"""

from __future__ import annotations

import functools
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import streamlit as st

from datetime import datetime, timezone

from sat_tracker.passes import GroundStation, Pass
from sat_tracker.tle_fetcher import TleFetchError


# Default ground station: Greenwich. Politically neutral, instantly
# recognisable, deterministic.
DEFAULT_GROUND_STATION = GroundStation(
    latitude_deg=51.4779,
    longitude_deg=-0.0014,
    altitude_km=0.0,
    name="Greenwich",
)


# Preset ground stations selectable from the sidebar dropdown.
# Mix of observatories, launch sites, and major cities across both
# hemispheres. Altitudes are in km above WGS84 ellipsoid.
GROUND_STATION_PRESETS: tuple[GroundStation, ...] = (
    DEFAULT_GROUND_STATION,
    GroundStation(
        latitude_deg=40.5853,
        longitude_deg=-105.0844,
        altitude_km=1.524,
        name="Fort Collins",
    ),
    GroundStation(
        latitude_deg=28.5721,
        longitude_deg=-80.6480,
        altitude_km=0.003,
        name="Kennedy Space Center",
    ),
    GroundStation(
        latitude_deg=29.5589,
        longitude_deg=-95.0931,
        altitude_km=0.012,
        name="Johnson Space Center",
    ),
    GroundStation(
        latitude_deg=34.7420,
        longitude_deg=-120.5724,
        altitude_km=0.112,
        name="Vandenberg SFB",
    ),
    GroundStation(
        latitude_deg=45.9650,
        longitude_deg=63.3050,
        altitude_km=0.090,
        name="Baikonur",
    ),
    GroundStation(
        latitude_deg=5.2389,
        longitude_deg=-52.7689,
        altitude_km=0.014,
        name="Kourou (CSG)",
    ),
    GroundStation(
        latitude_deg=35.6762,
        longitude_deg=139.6503,
        altitude_km=0.040,
        name="Tokyo",
    ),
    GroundStation(
        latitude_deg=-33.9249,
        longitude_deg=18.4241,
        altitude_km=0.040,
        name="Cape Town",
    ),
    GroundStation(
        latitude_deg=-31.2754,
        longitude_deg=149.0672,
        altitude_km=1.165,
        name="Siding Spring Obs.",
    ),
)


def get_active_station() -> GroundStation:
    """Return the currently-selected ground station from session state.

    Defaults to Greenwich on first read; the picker mutates this via
    :func:`set_active_station` so subsequent reads pick up the change.
    """
    cached = st.session_state.get("active_station")
    if cached is None:
        return DEFAULT_GROUND_STATION
    # Stored as a 4-tuple to keep session_state hashable / serialisable.
    lat, lon, alt, name = cached
    return GroundStation(
        latitude_deg=lat,
        longitude_deg=lon,
        altitude_km=alt,
        name=name,
    )


def set_active_station(station: GroundStation) -> None:
    """Persist the selected ground station for the rest of the session."""
    st.session_state["active_station"] = (
        station.latitude_deg,
        station.longitude_deg,
        station.altitude_km,
        station.name,
    )

logger = logging.getLogger(__name__)

_CURATED_PATH = Path(__file__).parent / "static" / "curated_satellites.json"

# Maximum tracked satellites at once. Higher than this degrades plotly's
# WebGL renderer on mid-spec hardware (50 sats × 60 frames + sphere +
# coastlines ≈ 4000 animated points).
MAX_TRACKED_SATELLITES: int = 50

# Multi-satellite colour palette: Paul Tol's "vibrant" qualitative
# palette, designed for distinguishability under most colour-vision
# deficiencies. Cycles after 7; for >7 satellites we add the project's
# original four-colour core to extend.
TOL_VIBRANT_PALETTE: tuple[str, ...] = (
    "#EE7733",  # orange
    "#0077BB",  # blue
    "#33BBEE",  # cyan
    "#EE3377",  # magenta
    "#CC3311",  # red
    "#009988",  # teal
    "#BBBBBB",  # grey
    # Extended fillers if a user tracks >7 sats:
    "#9467bd",  # purple (matplotlib tab10)
    "#8c564b",  # brown
    "#e377c2",  # pink
)


# CelesTrak group buttons — name, group_name (CelesTrak GROUP=), label
# blurb. Counts shown in the UI come from a live group fetch (and are
# cached for an hour) rather than being hardcoded so the labels stay
# truthful as constellations evolve.
@dataclass(frozen=True)
class CelestrakGroupOption:
    """Predefined CelesTrak group selectable from the sidebar."""

    label: str
    group_name: str
    blurb: str


CELESTRAK_GROUPS: tuple[CelestrakGroupOption, ...] = (
    CelestrakGroupOption(
        label="Stations + ISS visitors",
        group_name="stations",
        blurb="ISS, Tiangong, and currently-docked supply / crew vehicles",
    ),
    CelestrakGroupOption(
        label="GPS-OPS",
        group_name="gps-ops",
        blurb="Operational GPS constellation",
    ),
    CelestrakGroupOption(
        label="GLONASS-OPS",
        group_name="glo-ops",
        blurb="Operational GLONASS constellation",
    ),
    CelestrakGroupOption(
        label="Galileo",
        group_name="galileo",
        blurb="Operational European GNSS constellation",
    ),
    CelestrakGroupOption(
        label="Weather (NOAA)",
        group_name="weather",
        blurb="Polar-orbiting NOAA / METOP / Suomi-NPP weather sats",
    ),
    CelestrakGroupOption(
        label="GEO weather",
        group_name="goes",
        blurb="Geostationary GOES weather imagers",
    ),
    CelestrakGroupOption(
        label="Starlink",
        group_name="starlink",
        blurb=f"Starlink constellation, capped at {MAX_TRACKED_SATELLITES} sats",
    ),
)


@functools.lru_cache(maxsize=1)
def load_curated_satellites() -> list[dict]:
    """Load the curated 200-entry satellite list from the JSON asset.

    Returns:
        A flat list of ``{"name": str, "catnr": int, "blurb": str,
        "category": str}`` dicts, one per satellite. Categories are
        flattened into a per-entry field so search results can show
        the category alongside the name.
    """
    with _CURATED_PATH.open() as f:
        data = json.load(f)
    flat: list[dict] = []
    for category, entries in data["categories"].items():
        for entry in entries:
            flat.append(
                {
                    "name": entry["name"],
                    "catnr": int(entry["catnr"]),
                    "blurb": entry.get("blurb", ""),
                    "category": category,
                }
            )
    return flat


def color_for_index(i: int) -> str:
    """Return a satellite's chip / orbit colour from the qualitative palette."""
    return TOL_VIBRANT_PALETTE[i % len(TOL_VIBRANT_PALETTE)]


# -- Session-state helpers ----------------------------------------------------


def init_session_state(default_catnrs: tuple[int, ...] = (25544,)) -> None:
    """Seed ``st.session_state`` keys used by the dashboard.

    Idempotent — only sets keys that don't already exist.
    """
    if "tracked" not in st.session_state:
        # Default tracked: ISS only. Stored as ordered tuple of catnrs
        # so the colour palette assignment is deterministic across reruns.
        st.session_state["tracked"] = list(default_catnrs)


def add_tracked(catnr: int) -> None:
    """Append a satellite to the tracked list, deduped, capped."""
    tracked = st.session_state.setdefault("tracked", [])
    if catnr in tracked:
        return
    if len(tracked) >= MAX_TRACKED_SATELLITES:
        st.toast(
            f"Tracked-satellite cap reached ({MAX_TRACKED_SATELLITES}). "
            f"Remove one before adding another.",
            icon="⚠️",
        )
        return
    tracked.append(catnr)


def remove_tracked(catnr: int) -> None:
    tracked = st.session_state.get("tracked", [])
    if catnr in tracked:
        tracked.remove(catnr)


def set_following(catnr: Optional[int]) -> None:
    """Set or clear the satellite the camera should follow.

    ``None`` returns the camera to free orbit (manual control).
    """
    if catnr is None:
        st.session_state.pop("following", None)
    else:
        st.session_state["following"] = catnr


def get_following() -> Optional[int]:
    """Return the currently-followed catnr, or ``None`` if free orbit."""
    return st.session_state.get("following")


def add_many_tracked(catnrs: list[int], *, source_label: str) -> None:
    """Bulk-add satellites; respects the cap and dedupes."""
    tracked = st.session_state.setdefault("tracked", [])
    added = 0
    skipped = 0
    capped = 0
    for catnr in catnrs:
        if catnr in tracked:
            skipped += 1
            continue
        if len(tracked) >= MAX_TRACKED_SATELLITES:
            capped += 1
            continue
        tracked.append(catnr)
        added += 1
    msg = f"Added {added} satellites from {source_label}"
    if skipped:
        msg += f" ({skipped} already tracked)"
    if capped:
        msg += f" — {capped} skipped at the {MAX_TRACKED_SATELLITES}-sat cap"
    st.toast(msg, icon="✅" if added else "ℹ️")


# -- Sidebar -----------------------------------------------------------------


def render_sidebar(
    *,
    fetcher,
    fetcher_error_label: str = "fetch failed",
    passes_provider=None,
) -> None:
    """Render the satellite-picker sidebar.

    Args:
        fetcher: An active :class:`TleFetcher` used by group buttons.
            Already cached as a singleton in ``app.py``.
        fetcher_error_label: Label shown in the toast when a group
            fetch fails.
        passes_provider: Optional callable
            ``(catnr, station, day_bucket) -> list[Pass]`` used by the
            pass-prediction panel. When omitted, the panel is hidden.
    """
    with st.sidebar:
        st.markdown("### Tracked satellites")
        _render_tracked_chips()

        st.divider()
        st.markdown("### Add satellite")
        _render_search_box()
        _render_catnr_box()

        st.divider()
        st.markdown("### Add CelesTrak group")
        _render_group_buttons(fetcher, fetcher_error_label)

        if passes_provider is not None:
            st.divider()
            _render_passes_panel(passes_provider)
            st.divider()
            _render_station_picker()


def _render_tracked_chips() -> None:
    """Show currently-tracked satellites as removable chips.

    Each chip carries a short label, a 🔭 follow toggle (locks the
    camera onto that satellite), and a ✕ remove button. Chip text
    drops parentheticals so "ISS (ZARYA)" → "ISS"; full names live
    in the figure legend.
    """
    tracked: list[int] = st.session_state.get("tracked", [])
    if not tracked:
        st.caption("No satellites tracked yet — add one below.")
        return

    curated = {c["catnr"]: c for c in load_curated_satellites()}
    following: Optional[int] = st.session_state.get("following")

    for i, catnr in enumerate(tracked):
        entry = curated.get(catnr)
        full_label = entry["name"] if entry else f"[{catnr}]"
        short_label = _shorten_for_chip(full_label)
        color = color_for_index(i)
        is_following = following == catnr
        chip_col, follow_col, remove_col = st.columns([4, 1, 1])
        with chip_col:
            badge = (
                " <span style='font-size:10px;color:#ffd400'>● following</span>"
                if is_following
                else ""
            )
            st.markdown(
                f"""
                <div style="
                    display:inline-block;
                    padding:4px 10px;
                    border-left:4px solid {color};
                    background:rgba(255,255,255,0.04);
                    border-radius:4px;
                    font-size:13px;
                    white-space:nowrap;
                    overflow:hidden;
                    text-overflow:ellipsis;
                    max-width:100%;
                ">{short_label}{badge}</div>
                """,
                unsafe_allow_html=True,
            )
        with follow_col:
            if st.button(
                "🔭" if not is_following else "■",
                key=f"follow_{catnr}",
                help=(
                    f"Stop following" if is_following
                    else f"Follow {full_label} — camera tracks this satellite"
                ),
                use_container_width=True,
            ):
                set_following(None if is_following else catnr)
                st.rerun()
        with remove_col:
            if st.button(
                "✕",
                key=f"remove_{catnr}",
                help=f"Stop tracking {full_label}",
                use_container_width=True,
            ):
                remove_tracked(catnr)
                # If we were following the removed sat, drop follow too.
                if following == catnr:
                    set_following(None)
                st.rerun()


def _shorten_for_chip(name: str) -> str:
    """Return chip-display label: drop parentheticals + clip at 14 chars."""
    # "ISS (ZARYA)" → "ISS"; "GPS BIIR-2 (PRN 13)" → "GPS BIIR-2"
    paren = name.find("(")
    short = name[:paren].strip() if paren > 0 else name
    if len(short) > 14:
        short = short[:13].rstrip() + "…"
    return short


def _render_search_box() -> None:
    """Curated-list search box with type-ahead matching.

    Wrapped in a form (``clear_on_submit=True``) so the selectbox
    auto-resets after each add — direct mutation of ``session_state``
    after widget instantiation raises ``StreamlitAPIException``.
    """
    curated = load_curated_satellites()
    options = [
        f"{entry['name']}  ·  [{entry['catnr']}]  ·  {entry['category']}"
        for entry in curated
    ]
    with st.form("search_form", clear_on_submit=True, border=False):
        selected = st.selectbox(
            "Search curated list",
            options=options,
            index=None,
            help=(
                f"Type to filter the {len(curated)}-entry curated list, "
                "then press Add."
            ),
            placeholder="ISS, Hubble, GPS, weather…",
        )
        submit = st.form_submit_button("Add", use_container_width=True)
        if submit and selected:
            idx = options.index(selected)
            entry = curated[idx]
            add_tracked(entry["catnr"])
            st.rerun()


def _render_catnr_box() -> None:
    """Free-form catalog-number input — power-user escape hatch.

    Uses a plain text_input + button rather than a form so we don't
    inherit the form's red error-style border + "Press Enter to submit"
    helper text, which made the field look misconfigured.

    Submit triggers: clicking Add **or** pressing Enter inside the
    input. Streamlit's text_input commits its new value on Enter and
    fires a rerun; we detect that by comparing the current value to
    the last value we processed.
    """
    pending_clear = st.session_state.pop("_catnr_clear_pending", False)
    if pending_clear:
        st.session_state["catnr_input"] = ""

    col_in, col_btn = st.columns([3, 1])
    with col_in:
        catnr_str = st.text_input(
            "NORAD catnr",
            key="catnr_input",
            placeholder="e.g. 25544",
            label_visibility="collapsed",
        )
    with col_btn:
        clicked = st.button(
            "Add",
            key="catnr_add_btn",
            use_container_width=True,
            disabled=not catnr_str,
        )

    # Detect Enter commit: the input's value changed from what we
    # last processed AND it is non-empty.
    last_processed = st.session_state.get("_catnr_last_processed", "")
    enter_commit = (
        catnr_str
        and catnr_str != last_processed
        and not clicked
    )

    if clicked or enter_commit:
        if not catnr_str:
            return
        try:
            catnr = int(catnr_str)
        except ValueError:
            st.toast(f"'{catnr_str}' is not a valid integer", icon="❌")
            st.session_state["_catnr_last_processed"] = catnr_str
            return
        add_tracked(catnr)
        st.session_state["_catnr_last_processed"] = catnr_str
        st.session_state["_catnr_clear_pending"] = True
        st.rerun()


def _render_group_buttons(fetcher, fetcher_error_label: str) -> None:
    """Predefined CelesTrak group quick-buttons."""
    counts = _get_cached_group_counts(fetcher)
    tracked = set(st.session_state.get("tracked", []))

    for group in CELESTRAK_GROUPS:
        n = counts.get(group.group_name)
        # Determine button state.
        if n is None:
            label = f"{group.label}"
            disabled = True
            help_text = "(Group count unavailable — fetch failed?)"
        else:
            cap_label = (
                f" (top {MAX_TRACKED_SATELLITES})"
                if n > MAX_TRACKED_SATELLITES
                else f" ({n})"
            )
            label = f"+ {group.label}{cap_label}"
            # Disable if the group's catnrs are already a subset of
            # tracked. Computing this requires looking at the group's
            # actual catnrs, which we'd have to fetch — defer to live
            # check inside the click handler instead. Keep enabled.
            disabled = False
            help_text = group.blurb
        if st.button(
            label,
            key=f"group_{group.group_name}",
            help=help_text,
            disabled=disabled,
            use_container_width=True,
        ):
            _on_add_group(fetcher, group, fetcher_error_label)
            st.rerun()


@st.cache_data(ttl=3600, show_spinner=False)
def _get_cached_group_counts(_fetcher) -> dict[str, int]:
    """Fetch CelesTrak group satellite counts (cached for an hour).

    Returns ``{}`` if any individual fetch fails — the buttons fall
    back to "count unavailable" labels.

    Note: ``_fetcher`` is named with a leading underscore so Streamlit
    skips hashing it — singletons can't be hashed and Streamlit would
    raise UnhashableParamError.
    """
    counts: dict[str, int] = {}
    for group in CELESTRAK_GROUPS:
        try:
            tles = _fetcher.fetch_group(group.group_name)
            counts[group.group_name] = len(tles)
        except TleFetchError as exc:
            logger.warning(
                "Group count fetch failed for %s: %s", group.group_name, exc
            )
    return counts


def _on_add_group(
    fetcher,
    group: CelestrakGroupOption,
    fetcher_error_label: str,
) -> None:
    """Click handler for a CelesTrak group button."""
    try:
        with st.spinner(f"Fetching {group.label}…"):
            tles = fetcher.fetch_group(group.group_name)
    except TleFetchError as exc:
        st.toast(f"{group.label}: {fetcher_error_label} ({exc})", icon="❌")
        return
    catnrs = [t.catalog_number for t in tles[:MAX_TRACKED_SATELLITES]]
    add_many_tracked(catnrs, source_label=group.label)


def _render_passes_panel(passes_provider) -> None:
    """Pass-prediction panel — only shown for a single satellite.

    When more than one satellite is tracked, a satellite picker
    appears so the user can choose which one to predict passes for.
    """
    tracked: list[int] = st.session_state.get("tracked", [])
    st.markdown("### Upcoming passes")

    if not tracked:
        st.caption("Track a satellite to see passes.")
        return

    if len(tracked) == 1:
        catnr = tracked[0]
    else:
        catnr = _render_pass_satellite_picker(tracked)
        if catnr is None:
            return

    station = get_active_station()
    label = station.name or "(unnamed)"
    st.caption(
        f"Over **{label}** "
        f"({station.latitude_deg:+.4f}°, {station.longitude_deg:+.4f}°"
        + (f", {station.altitude_km:.2f} km" if station.altitude_km > 0 else "")
        + ")"
    )

    day_bucket = int(datetime.now(timezone.utc).timestamp() // 86400)
    try:
        with st.spinner("Predicting passes…"):
            passes = passes_provider(catnr, station, day_bucket)
    except Exception as exc:
        # PassPredictor or upstream PropagationError; surface but don't
        # crash the whole page.
        logger.warning("Pass prediction failed for catnr=%d: %s", catnr, exc)
        st.error(
            f"Could not predict passes for {catnr}: {exc}",
            icon="⚠️",
        )
        return

    if not passes:
        st.info(
            "No complete passes in the next 24 h.\n\n"
            "(GEO satellites and deep-space targets never \"pass\" "
            "over a fixed observer.)",
            icon="🛰",
        )
        return

    now = datetime.now(timezone.utc)
    for pass_ in passes:
        _render_pass_card(pass_, now=now)


def _render_pass_satellite_picker(tracked: list[int]) -> Optional[int]:
    """Selectbox for choosing which tracked satellite to show passes for."""
    curated = {c["catnr"]: c for c in load_curated_satellites()}
    options: list[int] = list(tracked)

    def _label(catnr: int) -> str:
        entry = curated.get(catnr)
        full_label = entry["name"] if entry else f"catnr {catnr}"
        return f"{full_label} [{catnr}]"

    return st.selectbox(
        "Show passes for…",
        options=options,
        format_func=_label,
        key="passes_satellite_picker",
    )


def _render_pass_card(pass_: Pass, *, now: datetime) -> None:
    """Compact pass card for the sidebar."""
    aos_relative = _format_relative(pass_.aos_utc - now)
    duration_s = (pass_.los_utc - pass_.aos_utc).total_seconds()
    if pass_.sunlit is True:
        visibility_badge = "☀️ sunlit"
        badge_color = "#ffd400"
    elif pass_.sunlit is False:
        visibility_badge = "🌑 eclipse"
        badge_color = "#7a8aaa"
    else:
        visibility_badge = "❔ unknown"
        badge_color = "#999"
    st.markdown(
        f"""
        <div style="
            border-left:3px solid {badge_color};
            background:rgba(255,255,255,0.04);
            border-radius:4px;
            padding:8px 12px;
            margin:6px 0;
            font-size:13px;
            line-height:1.45;
        ">
          <div style="font-weight:600;font-size:13.5px">
            AOS in {aos_relative}
            <span style="float:right;opacity:0.85">{visibility_badge}</span>
          </div>
          <div style="opacity:0.85;font-size:12px;margin-top:2px">
            {pass_.aos_utc.strftime('%Y-%m-%d %H:%M:%S UTC')} →
            {pass_.los_utc.strftime('%H:%M:%S')}
          </div>
          <div style="opacity:0.85;font-size:12px;margin-top:2px">
            max el <b>{pass_.max_elevation_deg:.1f}°</b>
            · {duration_s:.0f}s
            · az {pass_.azimuth_at_aos_deg:.0f}°→{pass_.azimuth_at_los_deg:.0f}°
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_station_picker() -> None:
    """Ground-station picker: preset dropdown + custom-coord override.

    Selecting a preset immediately replaces the active station and
    triggers a rerun (so the passes panel rebuilds). The "Custom"
    option reveals lat/lon/alt input fields with inline validation.
    """
    st.markdown("### Ground station")
    active = get_active_station()
    preset_names = [s.name for s in GROUND_STATION_PRESETS] + ["Custom"]

    # Determine which preset matches the current active station, if any.
    matched = None
    for s in GROUND_STATION_PRESETS:
        if (
            s.name == active.name
            and abs(s.latitude_deg - active.latitude_deg) < 1e-4
            and abs(s.longitude_deg - active.longitude_deg) < 1e-4
        ):
            matched = s.name
            break
    default_index = (
        preset_names.index(matched)
        if matched is not None
        else preset_names.index("Custom")
    )

    selected = st.selectbox(
        "Preset",
        options=preset_names,
        index=default_index,
        key="station_preset_select",
    )

    if selected != "Custom":
        # User picked a preset; apply it if it's actually different.
        chosen = next(
            s for s in GROUND_STATION_PRESETS if s.name == selected
        )
        if matched != selected:
            set_active_station(chosen)
            st.rerun()
    else:
        _render_custom_station_inputs(active)

    # Always-visible status line: confirms what the dashboard is using.
    label = active.name or "(unnamed)"
    lat_dir = "N" if active.latitude_deg >= 0 else "S"
    lon_dir = "E" if active.longitude_deg >= 0 else "W"
    st.caption(
        f"**Tracking from:** {label} "
        f"({abs(active.latitude_deg):.4f}°{lat_dir}, "
        f"{abs(active.longitude_deg):.4f}°{lon_dir}"
        + (
            f", {active.altitude_km:.2f} km"
            if active.altitude_km > 0
            else ""
        )
        + ")"
    )


def _render_custom_station_inputs(active: GroundStation) -> None:
    """Lat/lon/alt inputs with inline validation. Apply on click."""
    with st.form("custom_station_form", clear_on_submit=False, border=False):
        col1, col2 = st.columns(2)
        with col1:
            lat = st.number_input(
                "Latitude (°)",
                min_value=-90.0,
                max_value=90.0,
                value=float(active.latitude_deg),
                step=0.0001,
                format="%.4f",
                key="custom_station_lat",
            )
        with col2:
            lon = st.number_input(
                "Longitude (°)",
                min_value=-180.0,
                max_value=180.0,
                value=float(active.longitude_deg),
                step=0.0001,
                format="%.4f",
                key="custom_station_lon",
            )
        col3, col4 = st.columns(2)
        with col3:
            alt_m = st.number_input(
                "Altitude (m)",
                min_value=0.0,
                max_value=5000.0,
                value=float(active.altitude_km * 1000.0),
                step=10.0,
                format="%.0f",
                key="custom_station_alt_m",
                help="Height above WGS84 ellipsoid (0–5000 m).",
            )
        with col4:
            name = st.text_input(
                "Label",
                value=(
                    active.name
                    if active.name
                    and active.name not in {s.name for s in GROUND_STATION_PRESETS}
                    else ""
                ),
                placeholder="My location",
                key="custom_station_name",
                max_chars=40,
            )
        submit = st.form_submit_button(
            "Apply custom station", use_container_width=True
        )
        if submit:
            set_active_station(
                GroundStation(
                    latitude_deg=float(lat),
                    longitude_deg=float(lon),
                    altitude_km=float(alt_m) / 1000.0,
                    name=name.strip() or None,
                )
            )
            st.rerun()


def _format_relative(delta) -> str:
    """Render a ``timedelta`` as ``2h 14m`` / ``45m`` / ``in progress``."""
    seconds = int(delta.total_seconds())
    if seconds <= 0:
        return "in progress"
    hours, rem = divmod(seconds, 3600)
    minutes = rem // 60
    if hours > 0:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m"


__all__ = (
    "MAX_TRACKED_SATELLITES",
    "TOL_VIBRANT_PALETTE",
    "CELESTRAK_GROUPS",
    "DEFAULT_GROUND_STATION",
    "GROUND_STATION_PRESETS",
    "load_curated_satellites",
    "color_for_index",
    "init_session_state",
    "add_tracked",
    "remove_tracked",
    "add_many_tracked",
    "set_following",
    "get_following",
    "get_active_station",
    "set_active_station",
    "render_sidebar",
)
