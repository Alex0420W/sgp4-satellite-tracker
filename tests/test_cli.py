"""Tests for ``sat_tracker.cli``.

The CLI is mostly wiring, so tests focus on:

* Pure formatting/extraction helpers (``position_to_dict``, ``render_position``).
* The ``_run_watch`` loop body — exit conditions, failure counter behaviour.
* Top-level ``main`` exit-code translation.

Watch-mode tests inject ``max_iterations`` and a no-op ``sleep`` so they
terminate deterministically and run fast.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from sat_tracker import cli as cli_module
from sat_tracker.cli import (
    _normalize_argv,
    _run_watch,
    main,
    passes_to_list,
    position_to_dict,
    render_passes,
    render_position,
)
from sat_tracker.coordinates import GroundPosition
from sat_tracker.passes import GroundStation, Pass
from sat_tracker.propagator import PropagationError, StateVector
from sat_tracker.tle_fetcher import Tle, TleFetchError


@pytest.fixture
def sample_tle(iss_tle: tuple[str, str, str]) -> Tle:
    name, line1, line2 = iss_tle
    return Tle(name=name, line1=line1, line2=line2)


@pytest.fixture
def sample_state() -> StateVector:
    return StateVector(
        time_utc=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        position_km=(1234.567, -2345.678, 3456.789),
        velocity_km_s=(7.1234, -1.2345, 0.5678),
    )


@pytest.fixture
def sample_ground() -> GroundPosition:
    return GroundPosition(
        time_utc=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        latitude_deg=42.1234,
        longitude_deg=-78.5678,
        altitude_km=412.3,
        eop_degraded=False,
    )


# -- Pure helpers -------------------------------------------------------------


def test_position_to_dict_has_expected_keys(
    sample_tle: Tle,
    sample_state: StateVector,
    sample_ground: GroundPosition,
) -> None:
    d = position_to_dict(sample_tle, sample_state, sample_ground)
    assert set(d.keys()) == {
        "name",
        "catalog_number",
        "time_utc",
        "latitude_deg",
        "longitude_deg",
        "altitude_km",
        "eop_degraded",
        "teme_position_km",
        "teme_velocity_km_s",
    }
    assert d["catalog_number"] == 25544
    assert d["latitude_deg"] == pytest.approx(42.1234)
    assert d["teme_position_km"] == [1234.567, -2345.678, 3456.789]


def test_render_position_basic_layout(
    sample_tle: Tle,
    sample_state: StateVector,
    sample_ground: GroundPosition,
) -> None:
    out = render_position(sample_tle, sample_state, sample_ground)
    assert "ISS (ZARYA)" in out
    assert "[25544]" in out
    assert "42.1234°N" in out
    assert "78.5678°W" in out
    assert "412.3 km" in out
    assert "TEME" not in out  # not verbose


def test_render_position_southern_western_hemispheres(
    sample_tle: Tle, sample_state: StateVector
) -> None:
    ground = GroundPosition(
        time_utc=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        latitude_deg=-30.0,
        longitude_deg=-60.0,
        altitude_km=400.0,
        eop_degraded=False,
    )
    out = render_position(sample_tle, sample_state, ground)
    assert "30.0000°S" in out
    assert "60.0000°W" in out


def test_render_position_includes_eop_note_when_degraded(
    sample_tle: Tle, sample_state: StateVector
) -> None:
    ground = GroundPosition(
        time_utc=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        latitude_deg=0.0,
        longitude_deg=0.0,
        altitude_km=400.0,
        eop_degraded=True,
    )
    out = render_position(sample_tle, sample_state, ground)
    assert "degraded" in out.lower()


def test_render_position_verbose_includes_teme_vectors(
    sample_tle: Tle,
    sample_state: StateVector,
    sample_ground: GroundPosition,
) -> None:
    out = render_position(sample_tle, sample_state, sample_ground, verbose=True)
    assert "TEME pos" in out
    assert "TEME vel" in out
    # numeric values should appear (formatted)
    assert "1234.567" in out
    assert "7.1234" in out


# -- Watch loop ---------------------------------------------------------------


def test_run_watch_runs_max_iterations_then_exits_zero(
    sample_tle: Tle,
    sample_state: StateVector,
    sample_ground: GroundPosition,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fetcher = MagicMock()
    fetcher.fetch.return_value = sample_tle
    converter = MagicMock()
    converter.teme_to_ground.return_value = sample_ground
    monkeypatch.setattr(cli_module, "propagate", lambda tle, when: sample_state)

    sleep_calls: list[float] = []

    rc = _run_watch(
        fetcher,
        converter,
        catnr=25544,
        interval_seconds=2.0,
        verbose=False,
        max_failures=10,
        max_iterations=3,
        sleep=sleep_calls.append,
    )

    assert rc == 0
    assert fetcher.fetch.call_count == 3
    # 2 sleeps between 3 iterations; no sleep after the final.
    assert sleep_calls == [2.0, 2.0]


def test_run_watch_aborts_after_consecutive_failures(
    capsys: pytest.CaptureFixture[str],
) -> None:
    fetcher = MagicMock()
    fetcher.fetch.side_effect = TleFetchError("simulated outage")
    converter = MagicMock()

    rc = _run_watch(
        fetcher,
        converter,
        catnr=25544,
        interval_seconds=0.0,
        verbose=False,
        max_failures=3,
        max_iterations=None,  # would loop forever without the failure guard
        sleep=lambda _: None,
    )

    err = capsys.readouterr().err
    assert rc == 4
    assert err.count("failed") >= 3
    assert "aborting" in err.lower()


def test_run_watch_resets_failure_counter_on_success(
    sample_tle: Tle,
    sample_state: StateVector,
    sample_ground: GroundPosition,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fetcher = MagicMock()
    fetcher.fetch.side_effect = [
        TleFetchError("transient"),
        sample_tle,
        sample_tle,
    ]
    converter = MagicMock()
    converter.teme_to_ground.return_value = sample_ground
    monkeypatch.setattr(cli_module, "propagate", lambda tle, when: sample_state)

    rc = _run_watch(
        fetcher,
        converter,
        catnr=25544,
        interval_seconds=0.0,
        verbose=False,
        max_failures=3,
        max_iterations=3,
        sleep=lambda _: None,
    )

    # One failure followed by two successes — counter reset, no abort.
    assert rc == 0


# -- main exit codes ----------------------------------------------------------


def test_main_returns_two_on_startup_tle_fetch_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FailingFetcher:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def __enter__(self) -> "_FailingFetcher":
            return self

        def __exit__(self, *exc_info: object) -> None:
            pass

        def fetch(self, catnr: int) -> Tle:
            raise TleFetchError("no internet, no cache")

    class _AnyConverter:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

    monkeypatch.setattr(cli_module, "TleFetcher", _FailingFetcher)
    monkeypatch.setattr(cli_module, "CoordinateConverter", _AnyConverter)

    assert main(["--catnr", "25544"]) == 2


def test_main_returns_three_on_propagation_error(
    sample_tle: Tle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _OkFetcher:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def __enter__(self) -> "_OkFetcher":
            return self

        def __exit__(self, *exc_info: object) -> None:
            pass

        def fetch(self, catnr: int) -> Tle:
            return sample_tle

    class _AnyConverter:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

    def _raise_propagation(tle: object, when: object) -> StateVector:
        raise PropagationError("decayed")

    monkeypatch.setattr(cli_module, "TleFetcher", _OkFetcher)
    monkeypatch.setattr(cli_module, "CoordinateConverter", _AnyConverter)
    monkeypatch.setattr(cli_module, "propagate", _raise_propagation)

    assert main(["--catnr", "25544"]) == 3


# -- argv normalisation (default-to-now) --------------------------------------


def test_normalize_argv_empty_defaults_to_now() -> None:
    assert _normalize_argv([]) == ["now"]


def test_normalize_argv_bare_flags_prepend_now() -> None:
    assert _normalize_argv(["--catnr", "25544"]) == ["now", "--catnr", "25544"]


def test_normalize_argv_keeps_known_subcommand() -> None:
    assert _normalize_argv(["passes", "--lat", "0"]) == ["passes", "--lat", "0"]
    assert _normalize_argv(["now", "--watch", "5"]) == ["now", "--watch", "5"]


def test_normalize_argv_keeps_top_level_help() -> None:
    assert _normalize_argv(["--help"]) == ["--help"]
    assert _normalize_argv(["-h"]) == ["-h"]


# -- passes rendering --------------------------------------------------------


@pytest.fixture
def sample_pass() -> Pass:
    return Pass(
        satellite_catnr=25544,
        satellite_name="ISS (ZARYA)",
        aos_utc=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        los_utc=datetime(2024, 1, 1, 12, 7, 13, tzinfo=timezone.utc),
        max_elevation_deg=47.3,
        max_elevation_time_utc=datetime(2024, 1, 1, 12, 3, 36, tzinfo=timezone.utc),
        azimuth_at_aos_deg=312.2,
        azimuth_at_max_deg=28.4,
        azimuth_at_los_deg=104.8,
        sunlit=True,
        eop_degraded=False,
    )


@pytest.fixture
def sample_station() -> GroundStation:
    return GroundStation(
        latitude_deg=40.59,
        longitude_deg=-105.08,
        altitude_km=1.5,
        name="Fort Collins",
    )


def test_passes_to_list_keys(sample_pass: Pass) -> None:
    out = passes_to_list([sample_pass])
    assert len(out) == 1
    d = out[0]
    assert d["catalog_number"] == 25544
    assert d["max_elevation_deg"] == pytest.approx(47.3)
    assert d["sunlit"] is True
    assert d["duration_seconds"] == pytest.approx(7 * 60 + 13)


def test_render_passes_basic(
    sample_pass: Pass, sample_station: GroundStation
) -> None:
    out = render_passes([sample_pass], sample_station, hours=24)
    assert "ISS (ZARYA)" in out
    assert "[25544]" in out
    assert "Fort Collins" in out
    assert "40.5900°N" in out
    assert "105.0800°W" in out
    assert "1 pass(es)" in out
    assert "47.3" in out
    assert "sunlit (visible)" in out


def test_render_passes_empty(sample_station: GroundStation) -> None:
    out = render_passes([], sample_station, hours=24)
    assert "No complete passes" in out
    assert "Fort Collins" in out


def test_render_passes_visibility_branches(
    sample_pass: Pass, sample_station: GroundStation
) -> None:
    eclipsed = Pass(**{**sample_pass.__dict__, "sunlit": False})
    unknown = Pass(**{**sample_pass.__dict__, "sunlit": None})
    assert "Earth's shadow" in render_passes([eclipsed], sample_station, 24)
    assert "no ephemeris" in render_passes([unknown], sample_station, 24)


def test_render_passes_eop_note_when_degraded(
    sample_pass: Pass, sample_station: GroundStation
) -> None:
    degraded = Pass(**{**sample_pass.__dict__, "eop_degraded": True})
    out = render_passes([degraded], sample_station, 24)
    assert "EOP" in out
    assert "degraded" in out
