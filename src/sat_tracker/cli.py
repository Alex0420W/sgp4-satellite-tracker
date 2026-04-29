"""Command-line entry point for sat-tracker.

The real implementation lands in stage 5 once the propagator and coordinate
modules exist. For now this stub keeps the ``[project.scripts]`` entry in
``pyproject.toml`` resolvable so the package is installable at every stage.
"""

from __future__ import annotations


def main() -> int:
    """Print a placeholder message and exit successfully.

    Returns:
        Process exit code. ``0`` for success.
    """
    print("sat-tracker: not yet implemented")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
