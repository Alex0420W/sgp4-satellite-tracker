"""Interactive ground-track rendering via plotly.

Plotly is defer-imported inside :func:`render_interactive_ground_track` so the
package can be imported on a machine without the ``[viz]`` extra installed.

Output is an HTML file by default. Pass ``output_path`` with a ``.png`` /
``.svg`` / ``.pdf`` extension and ``kaleido`` will render a static export
(useful for README screenshots, CI artefacts, etc.).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence, Union

from sat_tracker.visualization.common import Track, split_at_antimeridian

logger = logging.getLogger(__name__)

_HOVER_TIME_FORMAT = "%Y-%m-%d %H:%M:%S UTC"

# Same palette as the cartopy renderer so single-vs-multi screenshots look
# consistent regardless of which backend produced them.
_DEFAULT_COLORS: tuple[str, ...] = (
    "#d62728",
    "#1f77b4",
    "#2ca02c",
    "#9467bd",
    "#ff7f0e",
)


def render_interactive_ground_track(
    tracks: Union[Track, Sequence[Track]],
    output_path: Union[str, Path],
    *,
    title: Optional[str] = None,
    colors: Optional[Sequence[str]] = None,
    line_width: float = 2.5,
    current_time_utc: Optional[datetime] = None,
    width: int = 1200,
    height: int = 650,
) -> Path:
    """Render one or more tracks to an interactive HTML map (or static export).

    Args:
        tracks: A single :class:`Track` or a sequence of them.
        output_path: Destination file. Format is inferred from the suffix:
            ``.html`` (default — full interactive plot) or any static format
            supported by ``kaleido`` (``.png``, ``.svg``, ``.pdf``).
        title: Optional override for the figure title.
        colors: Optional per-track colour list. Cycles if shorter than the
            number of tracks. Defaults to the same palette as the cartopy
            renderer.
        line_width: Track line width.
        current_time_utc: Optional "now" instant. For each track, marks the
            sample closest to this time with a star. Skipped per-track when
            the closest sample is more than one step away.
        width: Plot width in pixels (interactive HTML respects this as the
            initial size; static export honours it directly).
        height: Plot height in pixels.

    Returns:
        Resolved :class:`~pathlib.Path` of the written file.

    Raises:
        ValueError: If ``tracks`` is empty or any track has no samples.
        ImportError: If plotly is not installed (or kaleido missing for
            static export).
    """
    track_list = _normalize_tracks(tracks)
    color_list = list(colors) if colors else list(_DEFAULT_COLORS)

    import plotly.graph_objects as go

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

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
        oceancolor="#a8c8e6",
        showcountries=True,
        countrycolor="rgba(136,136,136,0.4)",
        countrywidth=0.4,
        lataxis=dict(showgrid=True, gridcolor="rgba(120,120,120,0.3)", dtick=30),
        lonaxis=dict(showgrid=True, gridcolor="rgba(120,120,120,0.3)", dtick=60),
    )

    if title is None:
        title = _default_title(track_list)
    subtitle = _subtitle(track_list, any_eop_degraded)
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b><br><span style='font-size:11px'>{subtitle}</span>",
            x=0.5,
            xanchor="center",
        ),
        width=width,
        height=height,
        margin=dict(l=10, r=10, t=80, b=10),
        legend=dict(
            yanchor="bottom",
            y=0.02,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.85)",
        ),
        showlegend=len(track_list) > 1,
    )

    suffix = output_path.suffix.lower()
    if suffix in {".html", ".htm", ""}:
        if suffix == "":
            output_path = output_path.with_suffix(".html")
        fig.write_html(str(output_path), include_plotlyjs="cdn")
    else:
        fig.write_image(str(output_path), width=width, height=height)

    logger.debug(
        "Rendered interactive ground track for %d satellite(s) to %s",
        len(track_list),
        output_path,
    )
    return output_path


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
        sample = _closest_sample(track.samples, current_time_utc)
        if sample is not None:
            fig.add_trace(
                go.Scattergeo(
                    lon=[sample.longitude_deg],
                    lat=[sample.latitude_deg],
                    mode="markers",
                    marker=dict(
                        size=20,
                        color="#ffd400",
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


def _default_title(tracks: Sequence[Track]) -> str:
    if len(tracks) == 1:
        t = tracks[0]
        return f"{t.name or '<unnamed>'} [{t.catalog_number}] ground track"
    names = ", ".join(
        f"{t.name or '<unnamed>'} [{t.catalog_number}]" for t in tracks
    )
    return f"Ground tracks: {names}"


def _subtitle(tracks: Sequence[Track], any_eop_degraded: bool) -> str:
    earliest = min(t.samples[0].time_utc for t in tracks)
    latest = max(t.samples[-1].time_utc for t in tracks)
    epochs = sorted(
        {t.tle_epoch_utc.strftime("%Y-%m-%d %H:%M UTC") for t in tracks}
    )
    epoch_label = (
        f"TLE epoch {epochs[0]}"
        if len(epochs) == 1
        else f"TLE epochs {', '.join(epochs)}"
    )
    parts = [
        epoch_label,
        (
            f"{earliest.strftime('%Y-%m-%d %H:%M UTC')} → "
            f"{latest.strftime('%Y-%m-%d %H:%M UTC')}"
        ),
    ]
    if any_eop_degraded:
        parts.append("EOP degraded")
    return "  |  ".join(parts)
