"""Command-line entry point for sat-tracker.

Subcommands::

    sat-tracker now    [--catnr N] [--watch S] [--verbose] [--max-failures N]
    sat-tracker passes --lat L --lon L [--alt-km A] [--hours H] ...
    sat-tracker plot   [--catnr N] [--catnr N ...] --output FILE [...]

For backward compatibility, invoking ``sat-tracker`` (no subcommand) is
equivalent to ``sat-tracker now``. Bare flags without a subcommand are
also forwarded to ``now`` — so ``sat-tracker --catnr 25544`` keeps working.

Output goes to stdout; warnings, iteration failures, and the interrupt
notice go to stderr. Logging follows the standard library/app split — only
this module configures handlers (via
:func:`sat_tracker.config.configure_logging`).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from sat_tracker.config import Config, configure_logging, load_config
from sat_tracker.coordinates import CoordinateConverter, GroundPosition
from sat_tracker.passes import GroundStation, Pass, PassPredictor
from sat_tracker.propagator import PropagationError, StateVector, propagate
from sat_tracker.tle_fetcher import Tle, TleFetcher, TleFetchError

logger = logging.getLogger(__name__)

_DEFAULT_CATNR = 25544          # ISS (ZARYA)
_DEFAULT_MAX_FAILURES = 10
_DEFAULT_HOURS = 24.0

_KNOWN_SUBCOMMANDS = frozenset({"now", "passes", "plot", "dashboard"})

# Exit codes — distinct enough that watch-mode CI checks can disambiguate.
_EXIT_OK = 0
_EXIT_TLE_FETCH = 2
_EXIT_PROPAGATION = 3
_EXIT_WATCH_ABORTED = 4
_EXIT_PLOT = 5
_EXIT_DASHBOARD = 6
_EXIT_INTERRUPTED = 130


# -- 'now' helpers (position rendering) ---------------------------------------


def position_to_dict(
    tle: Tle, state: StateVector, ground: GroundPosition
) -> dict[str, Any]:
    """Extract a JSON-serialisable summary of a satellite position."""
    return {
        "name": tle.name,
        "catalog_number": tle.catalog_number,
        "time_utc": ground.time_utc.isoformat(),
        "latitude_deg": ground.latitude_deg,
        "longitude_deg": ground.longitude_deg,
        "altitude_km": ground.altitude_km,
        "eop_degraded": ground.eop_degraded,
        "teme_position_km": list(state.position_km),
        "teme_velocity_km_s": list(state.velocity_km_s),
    }


def render_position(
    tle: Tle,
    state: StateVector,
    ground: GroundPosition,
    verbose: bool = False,
) -> str:
    """Format a position block for human display."""
    data = position_to_dict(tle, state, ground)

    lat = data["latitude_deg"]
    lon = data["longitude_deg"]
    lat_dir = "N" if lat >= 0 else "S"
    lon_dir = "E" if lon >= 0 else "W"

    name = data["name"] or "<unnamed>"
    lines = [
        f"{name} [{data['catalog_number']}]",
        f"  Time:     {data['time_utc']}",
        f"  Position: {abs(lat):.4f}°{lat_dir}, "
        f"{abs(lon):.4f}°{lon_dir}",
        f"  Altitude: {data['altitude_km']:.1f} km",
    ]
    if data["eop_degraded"]:
        lines.append(
            "  Note:     EOP data is bundled (offline) — accuracy may be "
            "degraded; see logs."
        )
    if verbose:
        x, y, z = data["teme_position_km"]
        vx, vy, vz = data["teme_velocity_km_s"]
        lines.append(
            f"  TEME pos: ({x:+12.3f}, {y:+12.3f}, {z:+12.3f}) km"
        )
        lines.append(
            f"  TEME vel: ({vx:+8.4f}, {vy:+8.4f}, {vz:+8.4f}) km/s"
        )
    return "\n".join(lines)


def _run_once(
    fetcher: TleFetcher,
    converter: CoordinateConverter,
    catnr: int,
    verbose: bool,
) -> None:
    """One full pipeline execution: fetch -> propagate -> convert -> print."""
    tle = fetcher.fetch(catnr)
    state = propagate(tle, datetime.now(timezone.utc))
    ground = converter.teme_to_ground(state)
    print(render_position(tle, state, ground, verbose))


def _run_watch(
    fetcher: TleFetcher,
    converter: CoordinateConverter,
    *,
    catnr: int,
    interval_seconds: float,
    verbose: bool,
    max_failures: int,
    max_iterations: Optional[int] = None,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Loop the pipeline until interrupted or too many consecutive failures."""
    consecutive_failures = 0
    iteration = 0
    while max_iterations is None or iteration < max_iterations:
        try:
            _run_once(fetcher, converter, catnr, verbose)
            consecutive_failures = 0
        except (TleFetchError, PropagationError) as exc:
            consecutive_failures += 1
            print(
                f"iteration {iteration} failed: {exc}",
                file=sys.stderr,
            )
            if consecutive_failures >= max_failures:
                print(
                    f"aborting watch after {consecutive_failures} consecutive "
                    f"failures.",
                    file=sys.stderr,
                )
                return _EXIT_WATCH_ABORTED

        iteration += 1
        if max_iterations is None or iteration < max_iterations:
            print()
            sleep(interval_seconds)

    return _EXIT_OK


# -- 'passes' helpers (pass rendering) ----------------------------------------


def passes_to_list(passes: list[Pass]) -> list[dict[str, Any]]:
    """JSON-serialisable list of passes (data extraction, no formatting)."""
    return [
        {
            "name": p.satellite_name,
            "catalog_number": p.satellite_catnr,
            "aos_utc": p.aos_utc.isoformat(),
            "los_utc": p.los_utc.isoformat(),
            "duration_seconds": (p.los_utc - p.aos_utc).total_seconds(),
            "max_elevation_deg": p.max_elevation_deg,
            "max_elevation_time_utc": p.max_elevation_time_utc.isoformat(),
            "azimuth_at_aos_deg": p.azimuth_at_aos_deg,
            "azimuth_at_max_deg": p.azimuth_at_max_deg,
            "azimuth_at_los_deg": p.azimuth_at_los_deg,
            "sunlit": p.sunlit,
            "eop_degraded": p.eop_degraded,
        }
        for p in passes
    ]


def render_passes(
    passes: list[Pass], station: GroundStation, hours: float
) -> str:
    """Format the list of passes as a multi-line human-readable block."""
    station_label = _format_station(station)
    if not passes:
        return (
            f"No complete passes for the requested satellite over "
            f"{station_label} in the next {hours:g}h.\n"
            f"(Note: incomplete passes — those that started before the "
            f"window or finish after — are skipped.)"
        )

    name = passes[0].satellite_name or "<unnamed>"
    catnr = passes[0].satellite_catnr

    lines = [
        f"{name} [{catnr}] passes over {station_label}",
        f"{len(passes)} pass(es) found in {hours:g}h window.",
    ]

    for i, p in enumerate(passes, start=1):
        duration_s = (p.los_utc - p.aos_utc).total_seconds()
        if p.sunlit is True:
            visibility = "sunlit (visible)"
        elif p.sunlit is False:
            visibility = "in Earth's shadow"
        else:
            visibility = "visibility unknown (no ephemeris)"
        lines.extend(
            [
                "",
                f"Pass {i} (duration {duration_s:.0f}s, "
                f"max elevation {p.max_elevation_deg:.1f}°)",
                f"  AOS: {_fmt_iso(p.aos_utc)}  "
                f"az={p.azimuth_at_aos_deg:5.1f}°",
                f"  MAX: {_fmt_iso(p.max_elevation_time_utc)}  "
                f"az={p.azimuth_at_max_deg:5.1f}°  "
                f"el={p.max_elevation_deg:5.1f}°",
                f"  LOS: {_fmt_iso(p.los_utc)}  "
                f"az={p.azimuth_at_los_deg:5.1f}°",
                f"  Visibility: {visibility}",
            ]
        )

    if passes[0].eop_degraded:
        lines.append("")
        lines.append(
            "Note: EOP data is bundled (offline) — pass timing accuracy "
            "may be degraded; see logs."
        )
    return "\n".join(lines)


def _fmt_iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


def _format_station(station: GroundStation) -> str:
    lat_dir = "N" if station.latitude_deg >= 0 else "S"
    lon_dir = "E" if station.longitude_deg >= 0 else "W"
    coord = (
        f"{abs(station.latitude_deg):.4f}°{lat_dir}, "
        f"{abs(station.longitude_deg):.4f}°{lon_dir}"
    )
    return f"{station.name} ({coord})" if station.name else coord


# -- argparse setup -----------------------------------------------------------


def _normalize_argv(argv: list[str]) -> list[str]:
    """Prepend ``now`` when the user invokes the CLI without a subcommand.

    * Bare ``sat-tracker``                  -> ``sat-tracker now``
    * ``sat-tracker --catnr 25544``         -> ``sat-tracker now --catnr 25544``
    * ``sat-tracker now ...``               -> unchanged
    * ``sat-tracker passes ...``            -> unchanged
    * ``sat-tracker -h`` / ``--help``        -> unchanged (top-level help)
    """
    if not argv:
        return ["now"]
    first = argv[0]
    if first in _KNOWN_SUBCOMMANDS:
        return list(argv)
    if first in ("-h", "--help"):
        return list(argv)
    return ["now"] + list(argv)


def _build_parser() -> argparse.ArgumentParser:
    common_satellite = argparse.ArgumentParser(add_help=False)
    common_satellite.add_argument(
        "-c",
        "--catnr",
        type=int,
        default=_DEFAULT_CATNR,
        help=f"NORAD catalog number (default: {_DEFAULT_CATNR}, ISS).",
    )

    parser = argparse.ArgumentParser(
        prog="sat-tracker",
        description="Satellite tracking, propagation, and pass prediction.",
    )
    subparsers = parser.add_subparsers(dest="subcommand")

    # 'now' (default subcommand)
    now_parser = subparsers.add_parser(
        "now",
        parents=[common_satellite],
        help="Print the current ground position (default subcommand).",
        description="Print the current ground position of a satellite.",
    )
    now_parser.add_argument(
        "-w",
        "--watch",
        type=float,
        metavar="SECONDS",
        default=None,
        help="Loop, printing a new position every SECONDS until interrupted.",
    )
    now_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Also print TEME position/velocity vectors.",
    )
    now_parser.add_argument(
        "--max-failures",
        type=int,
        default=_DEFAULT_MAX_FAILURES,
        metavar="N",
        help=(
            f"In watch mode, abort after N consecutive failures "
            f"(default: {_DEFAULT_MAX_FAILURES})."
        ),
    )

    # 'passes'
    passes_parser = subparsers.add_parser(
        "passes",
        parents=[common_satellite],
        help="Predict upcoming passes over a ground station.",
        description=(
            "Predict satellite passes above the minimum elevation "
            "threshold within a time window."
        ),
    )
    passes_parser.add_argument(
        "--lat",
        type=float,
        required=True,
        metavar="DEG",
        help="Observer geodetic latitude in degrees (-90..90).",
    )
    passes_parser.add_argument(
        "--lon",
        type=float,
        required=True,
        metavar="DEG",
        help="Observer geodetic longitude in degrees (-180..180).",
    )
    passes_parser.add_argument(
        "--alt-km",
        type=float,
        default=0.0,
        metavar="KM",
        help="Observer altitude above WGS84 ellipsoid in km (default 0).",
    )
    passes_parser.add_argument(
        "--hours",
        type=float,
        default=_DEFAULT_HOURS,
        metavar="H",
        help=f"Search window length in hours (default {_DEFAULT_HOURS:g}).",
    )
    passes_parser.add_argument(
        "--station-name",
        default=None,
        metavar="NAME",
        help="Optional station label printed in the header.",
    )
    passes_parser.add_argument(
        "--min-elevation",
        type=float,
        default=None,
        metavar="DEG",
        help=(
            "Override SAT_TRACKER_MIN_ELEVATION_DEG for this run "
            "(degrees; default uses config)."
        ),
    )

    # 'plot'
    plot_parser = subparsers.add_parser(
        "plot",
        help="Render a ground track to an image or interactive HTML file.",
        description=(
            "Render one or more satellite ground tracks to a static image "
            "(cartopy + matplotlib) or interactive HTML map (plotly). "
            "Requires the [viz] extra: pip install -e '.[viz]'."
        ),
    )
    plot_parser.add_argument(
        "-c",
        "--catnr",
        type=int,
        action="append",
        metavar="N",
        help=(
            "NORAD catalog number. Repeat to plot multiple satellites "
            f"on the same map. Defaults to {_DEFAULT_CATNR} (ISS) when omitted."
        ),
    )
    plot_parser.add_argument(
        "-o",
        "--output",
        required=True,
        metavar="PATH",
        help=(
            "Output file. Suffix selects backend & format: .png/.pdf/.svg "
            "use cartopy; .html uses plotly (interactive)."
        ),
    )
    plot_parser.add_argument(
        "--hours",
        type=float,
        default=None,
        metavar="H",
        help=(
            "Window length in hours, centred on now. Default is one full "
            "orbital period of the first satellite."
        ),
    )
    plot_parser.add_argument(
        "--start-utc",
        default=None,
        metavar="ISO8601",
        help=(
            "Override window start (ISO 8601, e.g. 2026-04-28T12:00:00Z). "
            "Defaults to now − window/2."
        ),
    )
    plot_parser.add_argument(
        "--no-now-marker",
        action="store_true",
        help="Suppress the gold star 'current position' marker.",
    )
    plot_parser.add_argument(
        "--title",
        default=None,
        metavar="STR",
        help="Override the plot title.",
    )
    plot_parser.add_argument(
        "--3d",
        dest="three_d",
        action="store_true",
        help=(
            "Render an interactive 3D orbit view (Earth as a textured "
            "sphere with the orbit polyline in ECEF). Output suffix still "
            "selects format (.html for interactive, .png/.svg/.pdf for "
            "static). Default is the 2D ground track."
        ),
    )
    plot_parser.add_argument(
        "--gs-lat",
        type=float,
        default=None,
        metavar="DEG",
        help=(
            "(3D only) Ground-station geodetic latitude. Adds a station "
            "marker on Earth's surface and a line-of-sight line to the "
            "satellite at 'now' when above the horizon."
        ),
    )
    plot_parser.add_argument(
        "--gs-lon",
        type=float,
        default=None,
        metavar="DEG",
        help="(3D only) Ground-station geodetic longitude. Required with --gs-lat.",
    )
    plot_parser.add_argument(
        "--gs-alt-km",
        type=float,
        default=0.0,
        metavar="KM",
        help="(3D only) Ground-station altitude in km (default 0).",
    )
    plot_parser.add_argument(
        "--gs-name",
        default=None,
        metavar="NAME",
        help="(3D only) Optional station label printed in the legend.",
    )
    plot_parser.add_argument(
        "--no-time-slider",
        action="store_true",
        help=(
            "(3D HTML only) Suppress the time slider. Static exports never "
            "include a slider."
        ),
    )

    # 'dashboard'
    dashboard_parser = subparsers.add_parser(
        "dashboard",
        help="Launch the local Streamlit dashboard server.",
        description=(
            "Spawn `streamlit run` against src/sat_tracker/dashboard/app.py. "
            "Requires the [dashboard] extra: pip install -e '.[dashboard]'."
        ),
    )
    dashboard_parser.add_argument(
        "--port",
        type=int,
        default=8501,
        metavar="N",
        help="TCP port to bind the local server (default 8501).",
    )
    dashboard_parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not auto-open a browser tab on launch.",
    )
    return parser


# -- dispatch -----------------------------------------------------------------


def _dispatch_now(
    args: argparse.Namespace,
    fetcher: TleFetcher,
    converter: CoordinateConverter,
) -> int:
    if args.watch is None:
        _run_once(fetcher, converter, args.catnr, args.verbose)
        return _EXIT_OK

    try:
        return _run_watch(
            fetcher,
            converter,
            catnr=args.catnr,
            interval_seconds=args.watch,
            verbose=args.verbose,
            max_failures=args.max_failures,
        )
    except KeyboardInterrupt:
        print("\ninterrupted.", file=sys.stderr)
        return _EXIT_INTERRUPTED


def _dispatch_passes(
    args: argparse.Namespace,
    fetcher: TleFetcher,
    converter: CoordinateConverter,
    config: Config,
) -> int:
    tle = fetcher.fetch(args.catnr)
    station = GroundStation(
        latitude_deg=args.lat,
        longitude_deg=args.lon,
        altitude_km=args.alt_km,
        name=args.station_name,
    )
    predictor = PassPredictor(
        config,
        converter,
        min_elevation_deg=args.min_elevation,
    )
    passes = predictor.predict_passes(
        tle,
        station,
        start_utc=datetime.now(timezone.utc),
        hours=args.hours,
    )
    print(render_passes(passes, station, args.hours))
    return _EXIT_OK


def _dispatch_plot(
    args: argparse.Namespace,
    fetcher: TleFetcher,
    converter: CoordinateConverter,
) -> int:
    # Defer the visualization import — keeps `now` and `passes` callable on
    # machines without the [viz] extras installed.
    try:
        from sat_tracker.visualization.common import (
            default_window_seconds,
            precompute_orbit,
            precompute_track,
        )
    except ImportError as exc:  # pragma: no cover — defensive
        print(f"error: cannot import visualization module: {exc}", file=sys.stderr)
        return _EXIT_PLOT

    if (args.gs_lat is None) != (args.gs_lon is None):
        print(
            "error: --gs-lat and --gs-lon must be provided together",
            file=sys.stderr,
        )
        return _EXIT_PLOT

    catnrs: list[int] = args.catnr or [_DEFAULT_CATNR]
    tles: list[Tle] = [fetcher.fetch(c) for c in catnrs]

    now = datetime.now(timezone.utc).replace(microsecond=0)
    if args.hours is not None:
        if args.hours <= 0:
            print("error: --hours must be positive", file=sys.stderr)
            return _EXIT_PLOT
        duration_s = args.hours * 3600.0
    else:
        duration_s = default_window_seconds(tles[0])

    if args.start_utc is not None:
        start = _parse_iso_utc(args.start_utc)
        if start is None:
            print(
                f"error: --start-utc {args.start_utc!r} is not a valid "
                f"ISO 8601 timestamp",
                file=sys.stderr,
            )
            return _EXIT_PLOT
    else:
        start = now - timedelta(seconds=duration_s / 2)

    output_path = Path(args.output)
    suffix = output_path.suffix.lower()
    current_time = None if args.no_now_marker else now

    try:
        if args.three_d:
            orbits = [
                precompute_orbit(
                    tle, converter, start_utc=start, duration_seconds=duration_s
                )
                for tle in tles
            ]
            station = None
            if args.gs_lat is not None:
                from sat_tracker.passes import GroundStation
                station = GroundStation(
                    latitude_deg=args.gs_lat,
                    longitude_deg=args.gs_lon,
                    altitude_km=args.gs_alt_km,
                    name=args.gs_name,
                )
            from sat_tracker.visualization.orbit_3d import render_orbit_3d
            written = render_orbit_3d(
                orbits if len(orbits) > 1 else orbits[0],
                output_path,
                current_time_utc=current_time,
                title=args.title,
                ground_station=station,
                time_slider=not args.no_time_slider,
            )
        else:
            tracks = [
                precompute_track(
                    tle, converter, start_utc=start, duration_seconds=duration_s
                )
                for tle in tles
            ]
            if suffix in {".html", ".htm"}:
                from sat_tracker.visualization.interactive import (
                    render_interactive_ground_track,
                )
                written = render_interactive_ground_track(
                    tracks if len(tracks) > 1 else tracks[0],
                    output_path,
                    current_time_utc=current_time,
                    title=args.title,
                )
            else:
                from sat_tracker.visualization.ground_track import (
                    render_ground_track,
                )
                written = render_ground_track(
                    tracks if len(tracks) > 1 else tracks[0],
                    output_path,
                    current_time_utc=current_time,
                    title=args.title,
                )
    except ImportError as exc:
        print(
            f"error: missing visualization dependency ({exc}). Install with: "
            f"pip install -e '.[viz]'",
            file=sys.stderr,
        )
        return _EXIT_PLOT

    print(f"wrote {written}")
    return _EXIT_OK


def _dispatch_dashboard(args: argparse.Namespace) -> int:
    """Spawn ``streamlit run`` against the dashboard entry point.

    The dashboard imports its own config / converter / fetcher
    singletons, so we do not pre-initialise anything here. Running via
    subprocess (rather than e.g. ``streamlit.web.bootstrap``) keeps
    the local-launch path one shell call deep — the same as what
    Streamlit Cloud does.
    """
    import shutil
    import subprocess
    from pathlib import Path

    streamlit_bin = shutil.which("streamlit")
    if streamlit_bin is None:
        print(
            "error: 'streamlit' executable not found on PATH. Install with: "
            "pip install -e '.[dashboard]'",
            file=sys.stderr,
        )
        return _EXIT_DASHBOARD

    # Resolve the entry point relative to this file so the command works
    # regardless of cwd.
    app_path = (
        Path(__file__).resolve().parent / "dashboard" / "app.py"
    )
    if not app_path.is_file():
        print(
            f"error: dashboard app not found at {app_path}",
            file=sys.stderr,
        )
        return _EXIT_DASHBOARD

    cmd = [
        streamlit_bin,
        "run",
        str(app_path),
        f"--server.port={args.port}",
    ]
    if args.no_browser:
        cmd.append("--server.headless=true")
    try:
        return subprocess.call(cmd)
    except KeyboardInterrupt:
        return _EXIT_INTERRUPTED


def _parse_iso_utc(raw: str) -> Optional[datetime]:
    # Accept trailing 'Z' as a synonym for +00:00 (Python <3.11 quirk).
    text = raw.replace("Z", "+00:00") if raw.endswith("Z") else raw
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point.

    Returns:
        Process exit code. ``0`` success; ``2`` TLE fetch failure;
        ``3`` propagation failure; ``4`` watch aborted; ``5`` plot failure;
        ``130`` Ctrl-C.
    """
    if argv is None:
        argv = sys.argv[1:]
    argv = _normalize_argv(list(argv))
    args = _build_parser().parse_args(argv)

    config = load_config()
    configure_logging(config)

    # 'dashboard' is a thin subprocess shim — it does not need any of
    # the per-process TleFetcher / CoordinateConverter setup the other
    # subcommands need (the streamlit subprocess builds its own).
    if args.subcommand == "dashboard":
        return _dispatch_dashboard(args)

    converter = CoordinateConverter(config)

    try:
        with TleFetcher(config) as fetcher:
            if args.subcommand == "passes":
                return _dispatch_passes(args, fetcher, converter, config)
            if args.subcommand == "plot":
                return _dispatch_plot(args, fetcher, converter)
            # Default / 'now'
            return _dispatch_now(args, fetcher, converter)
    except TleFetchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return _EXIT_TLE_FETCH
    except PropagationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return _EXIT_PROPAGATION


if __name__ == "__main__":
    raise SystemExit(main())
