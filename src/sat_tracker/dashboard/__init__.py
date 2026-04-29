"""Streamlit dashboard for live satellite tracking.

The dashboard composes the existing tracker modules — ``tle_fetcher``,
``propagator``, ``coordinates``, ``passes``, ``visualization.figures`` —
into a web UI. Heavy computation lives in those modules; this
subpackage is orchestration only.

Run locally with::

    streamlit run src/sat_tracker/dashboard/app.py

or via the CLI subcommand::

    python -m sat_tracker dashboard
"""
