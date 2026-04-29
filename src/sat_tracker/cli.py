"""Command-line entry point for sat-tracker.

Wires the four pipeline modules together (config -> fetch -> propagate ->
convert -> render) and exposes them as a small CLI:

    sat-tracker                  # current ISS position, one shot
    sat-tracker --catnr 20580    # Hubble
    sat-tracker --watch 5        # poll every 5 seconds, append output
    sat-tracker --verbose        # also print TEME state vectors

Output goes to stdout; warnings, iteration failures, and the interrupt notice
go to stderr. Logging follows the standard library/app split — only this
module configures handlers (via :func:`sat_tracker.config.configure_logging`).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from sat_tracker.config import configure_logging, load_config
from sat_tracker.coordinates import CoordinateConverter, GroundPosition
from sat_tracker.propagator import PropagationError, StateVector, propagate
from sat_tracker.tle_fetcher import Tle, TleFetcher, TleFetchError

logger = logging.getLogger(__name__)

_DEFAULT_CATNR = 25544          # ISS (ZARYA)
_DEFAULT_MAX_FAILURES = 10

# Exit codes — distinct enough that watch-mode CI checks can disambiguate.
_EXIT_OK = 0
_EXIT_TLE_FETCH = 2
_EXIT_PROPAGATION = 3
_EXIT_WATCH_ABORTED = 4
_EXIT_INTERRUPTED = 130


def position_to_dict(
    tle: Tle, state: StateVector, ground: GroundPosition
) -> dict[str, Any]:
    """Extract a JSON-serialisable summary of a satellite position.

    Pure data, no formatting. A future ``--json`` flag (or any other consumer
    that wants structured output) can call ``json.dumps`` on the result of
    this function with no further plumbing.
    """
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
    """Format a position block for human display.

    Args:
        tle: Source TLE — name and catalog number appear in the header.
        state: TEME state vector — only consulted in verbose mode.
        ground: Geodetic sub-satellite point.
        verbose: When True, append TEME position and velocity vectors.

    Returns:
        A multi-line string ready for ``print``. No trailing newline.
    """
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
    """Loop the pipeline until interrupted or too many consecutive failures.

    Args:
        fetcher: Active TLE fetcher.
        converter: Active coordinate converter.
        catnr: NORAD catalog number to track.
        interval_seconds: Naive sleep between iterations.
        verbose: Forwarded to :func:`render_position`.
        max_failures: Bail out after this many *consecutive* failed iterations
            (transient blips reset the counter).
        max_iterations: Test seam — None for production (infinite loop), a
            small integer for tests. The loop exits cleanly after this many
            iterations.
        sleep: Test seam — replaceable with a no-op for fast tests.

    Returns:
        Exit code: ``0`` on clean termination (max_iterations reached), ``4``
        if aborted from too many consecutive failures.
    """
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
        # Don't sleep or print the separator after the last iteration.
        if max_iterations is None or iteration < max_iterations:
            print()
            sleep(interval_seconds)

    return _EXIT_OK


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sat-tracker",
        description="Print the current ground position of a satellite.",
    )
    parser.add_argument(
        "-c",
        "--catnr",
        type=int,
        default=_DEFAULT_CATNR,
        help=f"NORAD catalog number (default: {_DEFAULT_CATNR}, ISS).",
    )
    parser.add_argument(
        "-w",
        "--watch",
        type=float,
        metavar="SECONDS",
        default=None,
        help="Loop, printing a new position every SECONDS until interrupted.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Also print TEME position/velocity vectors.",
    )
    parser.add_argument(
        "--max-failures",
        type=int,
        default=_DEFAULT_MAX_FAILURES,
        metavar="N",
        help=(
            f"In watch mode, abort after N consecutive failures "
            f"(default: {_DEFAULT_MAX_FAILURES})."
        ),
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point.

    Args:
        argv: Optional argument list (excluding the program name). Defaults
            to ``sys.argv[1:]`` via argparse.

    Returns:
        Process exit code. ``0`` success; ``2`` TLE fetch failure at startup;
        ``3`` propagation failure at startup; ``4`` watch aborted from too
        many consecutive failures; ``130`` Ctrl-C.
    """
    args = _build_parser().parse_args(argv)

    config = load_config()
    configure_logging(config)

    converter = CoordinateConverter(config)

    try:
        with TleFetcher(config) as fetcher:
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
    except TleFetchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return _EXIT_TLE_FETCH
    except PropagationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return _EXIT_PROPAGATION


if __name__ == "__main__":
    raise SystemExit(main())
