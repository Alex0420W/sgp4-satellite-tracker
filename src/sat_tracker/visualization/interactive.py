"""Interactive ground-track rendering via plotly (file output).

Thin disk-output shim around
:func:`sat_tracker.visualization.figures.build_ground_track_figure`. The
figure-building logic lives in ``figures.py`` so the Streamlit dashboard
can import the in-memory figure without touching disk; this module wraps
that builder with the suffix-routed write logic the CLI ``plot``
subcommand expects.

Output is an HTML file by default. Pass ``output_path`` with a ``.png``
/ ``.svg`` / ``.pdf`` extension and ``kaleido`` will render a static
export (useful for README screenshots, CI artefacts, etc.).
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence, Union

from sat_tracker.visualization.common import Track
from sat_tracker.visualization.figures import build_ground_track_figure

logger = logging.getLogger(__name__)


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
    fig = build_ground_track_figure(
        tracks,
        title=title,
        colors=colors,
        line_width=line_width,
        current_time_utc=current_time_utc,
        width=width,
        height=height,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix in {".html", ".htm", ""}:
        if suffix == "":
            output_path = output_path.with_suffix(".html")
        fig.write_html(str(output_path), include_plotlyjs="cdn")
    else:
        fig.write_image(str(output_path), width=width, height=height)

    logger.debug(
        "Rendered interactive ground track to %s", output_path
    )
    return output_path
