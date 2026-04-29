"""Allow ``python -m sat_tracker`` invocation.

Runs the same entry point as the ``sat-tracker`` console script. Useful in
environments where the venv's ``Scripts``/``bin`` directory isn't on PATH.
"""

from sat_tracker.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
