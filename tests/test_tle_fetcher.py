"""Tests for ``sat_tracker.tle_fetcher``.

Network access is faked by injecting a ``_FakeSession`` into the fetcher's
constructor — the same seam users would use to swap in a custom session.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

import pytest
import requests

from sat_tracker.config import Config
from sat_tracker.tle_fetcher import Tle, TleFetcher, TleFetchError


class _FakeResponse:
    def __init__(self, text: str, status_code: int = 200) -> None:
        self.text = text
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class _FakeSession:
    """Minimal stand-in for ``requests.Session`` used in tests."""

    def __init__(
        self,
        response: Optional[_FakeResponse] = None,
        exc: Optional[BaseException] = None,
    ) -> None:
        self._response = response
        self._exc = exc
        self.headers: dict[str, str] = {}
        self.calls: int = 0

    def get(self, url: str, params=None, timeout=None) -> _FakeResponse:  # noqa: ARG002
        self.calls += 1
        if self._exc is not None:
            raise self._exc
        assert self._response is not None
        return self._response

    def close(self) -> None:
        pass


@pytest.fixture
def config(tmp_path: Path) -> Config:
    return Config(
        cache_dir=tmp_path,
        cache_ttl_hours=6,
        tle_source_url="http://test.invalid/gp.php",
        user_agent="sat-tracker-tests/1.0",
        http_timeout_seconds=5,
        log_level="DEBUG",
        min_elevation_deg=10.0,
    )


def _good_body(iss_tle: tuple[str, str, str]) -> str:
    name, line1, line2 = iss_tle
    return f"{name}\r\n{line1}\r\n{line2}\r\n"


def test_fetch_returns_tle_and_writes_cache(
    config: Config, iss_tle: tuple[str, str, str]
) -> None:
    session = _FakeSession(response=_FakeResponse(_good_body(iss_tle)))
    fetcher = TleFetcher(config, session=session)

    tle = fetcher.fetch(25544)

    assert isinstance(tle, Tle)
    assert tle.line1 == iss_tle[1]
    assert tle.line2 == iss_tle[2]
    assert tle.catalog_number == 25544
    assert (config.cache_dir / "tle_25544.txt").exists()
    assert session.calls == 1


def test_cache_hit_skips_network(
    config: Config, iss_tle: tuple[str, str, str]
) -> None:
    name, line1, line2 = iss_tle
    cache_file = config.cache_dir / "tle_25544.txt"
    cache_file.write_text(f"{name}\n{line1}\n{line2}\n", encoding="utf-8")

    session = _FakeSession(exc=AssertionError("network must not be called"))
    fetcher = TleFetcher(config, session=session)

    tle = fetcher.fetch(25544)

    assert tle.line1 == line1
    assert session.calls == 0


def test_stale_cache_triggers_refetch(
    config: Config, iss_tle: tuple[str, str, str]
) -> None:
    name, line1, line2 = iss_tle
    cache_file = config.cache_dir / "tle_25544.txt"
    cache_file.write_text(f"OLD\n{line1}\n{line2}\n", encoding="utf-8")
    old = time.time() - 24 * 3600  # 24h old, TTL is 6h
    os.utime(cache_file, (old, old))

    session = _FakeSession(response=_FakeResponse(_good_body(iss_tle)))
    fetcher = TleFetcher(config, session=session)

    tle = fetcher.fetch(25544)

    assert session.calls == 1
    assert tle.name == name  # refreshed cache replaces "OLD" name


def test_stale_cache_returned_on_network_failure(
    config: Config,
    iss_tle: tuple[str, str, str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    name, line1, line2 = iss_tle
    cache_file = config.cache_dir / "tle_25544.txt"
    cache_file.write_text(f"{name}\n{line1}\n{line2}\n", encoding="utf-8")
    old = time.time() - 24 * 3600
    os.utime(cache_file, (old, old))

    session = _FakeSession(exc=requests.ConnectionError("network down"))
    fetcher = TleFetcher(config, session=session)

    with caplog.at_level(logging.WARNING, logger="sat_tracker.tle_fetcher"):
        tle = fetcher.fetch(25544)

    assert tle.line1 == line1
    assert "stale" in caplog.text.lower()
    assert "network down" in caplog.text
    assert "age=" in caplog.text  # age included for debuggability


def test_no_gp_data_response_raises(config: Config) -> None:
    session = _FakeSession(response=_FakeResponse("No GP data found"))
    fetcher = TleFetcher(config, session=session)

    with pytest.raises(TleFetchError, match="no data"):
        fetcher.fetch(99999)


def test_malformed_line_length_raises(config: Config) -> None:
    bad_body = "BAD\nthis is not a tle line\nneither is this\n"
    session = _FakeSession(response=_FakeResponse(bad_body))
    fetcher = TleFetcher(config, session=session)

    with pytest.raises(TleFetchError, match="length"):
        fetcher.fetch(12345)
