"""Static world-map ground-track rendering via cartopy + matplotlib.

Both heavy plotting libraries are defer-imported inside
:func:`render_ground_track` so the package can be imported on a machine
without the ``[viz]`` extra installed — the failure only surfaces when a
caller actually tries to render.

The renderer takes one or more precomputed
:class:`~sat_tracker.visualization.common.Track` objects (propagation lives in
:mod:`sat_tracker.visualization.common`, not here) and writes a single image
file. Antimeridian splitting is applied per track so a satellite that crosses
the dateline doesn't draw a horizontal line across the map.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

from sat_tracker.visualization.common import (
    DEFAULT_TRACK_COLORS,
    Track,
    split_at_antimeridian,
)

logger = logging.getLogger(__name__)

_SUBTITLE_TIME_FORMAT = "%Y-%m-%d %H:%M UTC"


def render_ground_track(
    tracks: Union[Track, Sequence[Track]],
    output_path: Union[str, Path],
    *,
    figsize: tuple[float, float] = (14, 7),
    dpi: int = 150,
    title: Optional[str] = None,
    colors: Optional[Sequence[str]] = None,
    line_width: float = 2.5,
    show_endpoints: bool = True,
    current_time_utc: Optional[datetime] = None,
) -> Path:
    """Render one or more tracks to a world-map image file.

    Args:
        tracks: A single :class:`Track` or a sequence of them. Each track is
            drawn in its own colour and gets its own legend entry.
        output_path: Destination file. Output format is inferred from the
            extension by matplotlib (``.png``, ``.pdf``, ``.svg`` all work).
            Parent directories are created if missing.
        figsize: Figure size in inches.
        dpi: Resolution for raster outputs.
        title: Optional override for the figure title. By default a title
            is built from the satellite name(s) and catalog number(s).
        colors: Optional per-track colour list (any matplotlib colour
            strings). If shorter than ``tracks``, cycles. Defaults to a
            built-in palette tuned for contrast against the ocean basemap.
        line_width: Track polyline width in points.
        show_endpoints: When ``True``, mark each track's start (large
            circle) and end (small triangle) for direction.
        current_time_utc: Optional "now" instant. For each track, the sample
            with the closest ``time_utc`` is marked with a star — useful as
            a "where the satellite is right now" indicator on a hero image.
            If the closest sample is more than one step away from this time,
            no marker is drawn for that track (avoids placing a misleading
            "now" pin outside the rendered window).

    Returns:
        The resolved :class:`~pathlib.Path` of the written file.

    Raises:
        ValueError: If ``tracks`` is empty or any track has no samples.
        ImportError: If cartopy or matplotlib are not installed.
    """
    track_list = _normalize_tracks(tracks)
    color_list = list(colors) if colors else list(DEFAULT_TRACK_COLORS)

    # Defer-imports: cartopy + matplotlib are only required at render time.
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    projection = ccrs.PlateCarree()
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(1, 1, 1, projection=projection)
    ax.set_global()

    # Basemap layers, painted back-to-front.
    ax.add_feature(cfeature.OCEAN, facecolor="#a8c8e6", zorder=0)
    ax.add_feature(cfeature.LAND, facecolor="#f4f1e8", zorder=0)
    ax.add_feature(
        cfeature.BORDERS, linewidth=0.4, edgecolor="#888", alpha=0.4, zorder=1
    )
    ax.add_feature(
        cfeature.COASTLINE, linewidth=0.7, edgecolor="#444", zorder=2
    )

    gridlines = ax.gridlines(
        draw_labels=True,
        linewidth=0.3,
        linestyle="--",
        color="#777",
        alpha=0.5,
    )
    gridlines.top_labels = False
    gridlines.right_labels = False

    legend_handles: list[Line2D] = []
    any_eop_degraded = False
    for idx, track in enumerate(track_list):
        color = color_list[idx % len(color_list)]
        any_eop_degraded = any_eop_degraded or track.eop_degraded
        _draw_track(
            ax,
            track,
            color=color,
            line_width=line_width,
            projection=projection,
            show_endpoints=show_endpoints,
            current_time_utc=current_time_utc,
        )
        label = f"{track.name or '<unnamed>'} [{track.catalog_number}]"
        legend_handles.append(
            Line2D([], [], color=color, linewidth=line_width, label=label)
        )

    if len(track_list) > 1:
        ax.legend(
            handles=legend_handles,
            loc="lower left",
            fontsize=9,
            framealpha=0.85,
            facecolor="white",
        )

    if title is None:
        title = _default_title(track_list)
    fig.suptitle(title, fontsize=15, y=0.98)
    ax.set_title(_subtitle(track_list, any_eop_degraded), fontsize=9, pad=10)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    logger.debug(
        "Rendered ground track for %d satellite(s) to %s",
        len(track_list),
        output_path,
    )
    return output_path


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


def _draw_track(
    ax,
    track: Track,
    *,
    color: str,
    line_width: float,
    projection,
    show_endpoints: bool,
    current_time_utc: Optional[datetime],
) -> None:
    for segment in split_at_antimeridian(list(track.samples)):
        if len(segment) < 2:
            continue
        lons = [s.longitude_deg for s in segment]
        lats = [s.latitude_deg for s in segment]
        ax.plot(
            lons,
            lats,
            color=color,
            linewidth=line_width,
            transform=projection,
            zorder=3,
        )

    if show_endpoints:
        start = track.samples[0]
        end = track.samples[-1]
        # Start: prominent ringed circle. End: small triangle.
        ax.plot(
            start.longitude_deg,
            start.latitude_deg,
            marker="o",
            color=color,
            markersize=11,
            markeredgecolor="white",
            markeredgewidth=1.6,
            transform=projection,
            zorder=5,
        )
        ax.plot(
            end.longitude_deg,
            end.latitude_deg,
            marker="^",
            color=color,
            markersize=6,
            markeredgecolor="white",
            markeredgewidth=0.8,
            transform=projection,
            zorder=5,
            alpha=0.85,
        )

    if current_time_utc is not None:
        sample = _closest_sample(track.samples, current_time_utc)
        if sample is not None:
            ax.plot(
                sample.longitude_deg,
                sample.latitude_deg,
                marker="*",
                color="#ffd400",
                markersize=22,
                markeredgecolor=color,
                markeredgewidth=1.8,
                transform=projection,
                zorder=6,
            )


def _closest_sample(samples, when: datetime):
    """Return the sample closest to ``when``, or ``None`` if too far away.

    "Too far away" is defined as more than the local sampling step — outside
    that, marking a "now" position would be misleading because the rendered
    window doesn't actually cover the requested instant.
    """
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
    epochs = sorted({t.tle_epoch_utc.strftime(_SUBTITLE_TIME_FORMAT) for t in tracks})
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


# Backwards-compat alias kept intentionally thin: a few existing callers
# may still pass a single Track positionally — the unified function handles
# both cases.
__all__: Iterable[str] = ("render_ground_track",)
