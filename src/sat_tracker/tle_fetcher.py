"""Fetch Two-Line Element sets from CelesTrak with local file caching.

The fetcher honours a TTL on the cache, atomically writes refreshed TLEs to
disk, and gracefully falls back to a stale cache on network failure so the
caller still gets the best available data.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Optional

import requests

from sat_tracker.config import Config

logger = logging.getLogger(__name__)


class TleFetchError(RuntimeError):
    """Raised when a TLE cannot be obtained from cache or remote source."""


@dataclass(frozen=True)
class Tle:
    """A validated Two-Line Element set.

    Attributes:
        name: Optional satellite name from the (optional) line 0.
        line1: First TLE data line (69 characters).
        line2: Second TLE data line (69 characters).
    """

    name: Optional[str]
    line1: str
    line2: str

    @property
    def catalog_number(self) -> int:
        """NORAD catalog number, parsed from positions 2-7 of line 1."""
        return int(self.line1[2:7])


class TleFetcher:
    """Fetches TLEs from CelesTrak with TTL-based local caching.

    Each call to :meth:`fetch` consults the on-disk cache first. If the cached
    file is missing or older than ``config.cache_ttl_hours``, the fetcher hits
    the configured CelesTrak endpoint. If that remote fetch fails for any
    reason and a stale cache exists, the stale cache is returned with a warning
    log entry rather than raising — graceful degradation matches how operational
    aerospace tooling typically behaves.
    """

    def __init__(
        self,
        config: Config,
        session: Optional[requests.Session] = None,
    ) -> None:
        """Construct a fetcher.

        Args:
            config: Active configuration (cache dir, TTL, source URL, etc.).
            session: Optional pre-built ``requests.Session``. When ``None``, the
                fetcher creates one and is responsible for closing it. Passing
                an external session is the test seam used by the suite.
        """
        self._config = config
        self._owns_session = session is None
        self._session = session if session is not None else requests.Session()
        self._session.headers.update({"User-Agent": config.user_agent})

    def close(self) -> None:
        """Close the underlying session if we created it."""
        if self._owns_session:
            self._session.close()

    def __enter__(self) -> "TleFetcher":
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        self.close()

    def fetch_group(self, group_name: str) -> list[Tle]:
        """Fetch a CelesTrak satellite group and return its current TLEs.

        Cache layout for groups (hybrid):
            * One file per satellite at ``tle_<catnr>.txt`` (canonical TLE
              storage, shared with the single-catnr :meth:`fetch`). No
              duplicate storage.
            * One small manifest at ``tle_group_<group>.txt`` listing the
              catalog numbers in the group, one per line.

        Stale-cache nuance — *intentional behaviour*: per-catnr cache files
        persist beyond the group manifest's TTL. If a satellite leaves a
        group between two ``fetch_group`` calls, its individual
        ``tle_<catnr>.txt`` file remains in the cache until invalidated by
        a separate ``fetch(catnr)`` call. A subsequent ``fetch(catnr)`` for
        a since-departed satellite therefore returns the last-known TLE
        rather than 404'ing. This preserves cache reuse across the
        single-vs-group fetch paths and is a deliberate design decision.

        On remote fetch failure, falls back to the stale group cache (if
        any) with a warning log — matches the graceful-degradation pattern
        used by :meth:`fetch`.

        Args:
            group_name: CelesTrak ``GROUP`` identifier
                (e.g. ``"stations"``, ``"starlink"``, ``"gps-ops"``).

        Returns:
            A list of validated :class:`Tle` objects, one per satellite
            currently in the group.

        Raises:
            TleFetchError: If no manifest exists *and* the remote fetch fails.
        """
        manifest_path = self._group_manifest_path(group_name)

        if manifest_path.exists() and self._is_fresh(manifest_path):
            try:
                tles = self._read_group_from_cache(manifest_path)
                logger.debug(
                    "Group cache hit for group=%r (%d satellites)",
                    group_name,
                    len(tles),
                )
                return tles
            except (TleFetchError, ValueError, OSError) as exc:
                logger.warning(
                    "Manifest fresh but per-catnr read failed (%s); "
                    "refetching group=%r.",
                    exc,
                    group_name,
                )

        try:
            tles = self._fetch_remote_group(group_name)
        except (requests.RequestException, TleFetchError) as exc:
            if manifest_path.exists():
                try:
                    stale = self._read_group_from_cache(manifest_path)
                    age_hours = (
                        time.time() - manifest_path.stat().st_mtime
                    ) / 3600.0
                    logger.warning(
                        "Group fetch failed for group=%r (%s); serving stale "
                        "cache (age=%.1fh, %d satellites).",
                        group_name,
                        exc,
                        age_hours,
                        len(stale),
                    )
                    return stale
                except Exception as inner:  # noqa: BLE001
                    logger.warning(
                        "Stale group cache fallback failed (%s); raising "
                        "original network error.",
                        inner,
                    )
            raise TleFetchError(
                f"No cached group {group_name!r} and remote fetch failed: "
                f"{exc}"
            ) from exc

        catnrs: list[int] = []
        for tle in tles:
            self._write_cache(self._cache_path(tle.catalog_number), tle)
            catnrs.append(tle.catalog_number)
        self._write_manifest(manifest_path, catnrs)
        logger.info(
            "Group refreshed for group=%r (%d satellites)",
            group_name,
            len(tles),
        )
        return tles

    def _read_group_from_cache(self, manifest_path: Path) -> list[Tle]:
        """Read a group manifest and load each per-catnr TLE from cache."""
        catnrs = self._read_manifest(manifest_path)
        tles: list[Tle] = []
        for c in catnrs:
            cache_path = self._cache_path(c)
            if not cache_path.exists():
                raise TleFetchError(
                    f"per-catnr cache missing for {c} (manifest claims it "
                    f"belongs to the group)"
                )
            tles.append(self._read_cache(cache_path))
        return tles

    def _group_manifest_path(self, group_name: str) -> Path:
        # Sanitise so an arbitrary group name can't escape the cache dir.
        safe = group_name.replace("/", "_").replace("\\", "_")
        return self._config.cache_dir / f"tle_group_{safe}.txt"

    def _read_manifest(self, path: Path) -> list[int]:
        out: list[int] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.append(int(line))
        return out

    def _write_manifest(self, path: Path, catnrs: list[int]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        body = "\n".join(str(c) for c in catnrs) + "\n"
        tmp_path.write_text(body, encoding="utf-8")
        tmp_path.replace(path)  # atomic on the same filesystem

    def _fetch_remote_group(self, group_name: str) -> list[Tle]:
        params = {"GROUP": group_name, "FORMAT": "tle"}
        response = self._session.get(
            self._config.tle_source_url,
            params=params,
            timeout=self._config.http_timeout_seconds,
        )
        response.raise_for_status()
        return _parse_multi_tle_text(response.text)

    def fetch(self, catalog_number: int) -> Tle:
        """Return the current TLE for the given NORAD catalog number.

        Args:
            catalog_number: NORAD catalog number (e.g. ``25544`` for the ISS).

        Returns:
            A validated :class:`Tle`.

        Raises:
            TleFetchError: If no cached TLE exists *and* the remote fetch fails.
        """
        cache_path = self._cache_path(catalog_number)

        if cache_path.exists() and self._is_fresh(cache_path):
            logger.debug("TLE cache hit (catnr=%d)", catalog_number)
            return self._read_cache(cache_path)

        try:
            tle = self._fetch_remote(catalog_number)
        except (requests.RequestException, TleFetchError) as exc:
            if cache_path.exists():
                age_hours = (time.time() - cache_path.stat().st_mtime) / 3600.0
                logger.warning(
                    "TLE refresh failed for catnr=%d (%s); "
                    "serving stale cache (age=%.1fh).",
                    catalog_number,
                    exc,
                    age_hours,
                )
                return self._read_cache(cache_path)
            raise TleFetchError(
                f"No cached TLE for catnr={catalog_number} and remote fetch "
                f"failed: {exc}"
            ) from exc

        self._write_cache(cache_path, tle)
        logger.info("TLE refreshed for catnr=%d", catalog_number)
        return tle

    def _cache_path(self, catalog_number: int) -> Path:
        return self._config.cache_dir / f"tle_{catalog_number}.txt"

    def _is_fresh(self, path: Path) -> bool:
        age_seconds = time.time() - path.stat().st_mtime
        return age_seconds < self._config.cache_ttl_hours * 3600

    def _read_cache(self, path: Path) -> Tle:
        return _parse_tle_text(path.read_text(encoding="utf-8"))

    def _write_cache(self, path: Path, tle: Tle) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        body_lines = []
        if tle.name:
            body_lines.append(tle.name)
        body_lines.extend([tle.line1, tle.line2])
        tmp_path.write_text("\n".join(body_lines) + "\n", encoding="utf-8")
        tmp_path.replace(path)  # atomic on the same filesystem

    def _fetch_remote(self, catalog_number: int) -> Tle:
        params = {"CATNR": str(catalog_number), "FORMAT": "tle"}
        response = self._session.get(
            self._config.tle_source_url,
            params=params,
            timeout=self._config.http_timeout_seconds,
        )
        response.raise_for_status()
        return _parse_tle_text(response.text)


def _parse_tle_text(text: str) -> Tle:
    """Parse and validate a TLE response body.

    Args:
        text: Raw response body or cache file contents.

    Returns:
        A validated :class:`Tle`.

    Raises:
        TleFetchError: If the body is empty, the wrong shape, or any line
            fails structural or checksum validation.
    """
    if not text or not text.strip():
        raise TleFetchError("Empty TLE response.")

    if "no gp data found" in text.lower():
        raise TleFetchError(f"CelesTrak returned no data: {text.strip()!r}")

    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    if len(lines) not in (2, 3):
        raise TleFetchError(
            f"Expected 2 or 3 TLE lines, got {len(lines)}: {lines!r}"
        )

    if len(lines) == 3:
        name: Optional[str] = lines[0].strip()
        line1, line2 = lines[1], lines[2]
    else:
        name = None
        line1, line2 = lines[0], lines[1]

    _validate_tle_line(line1, expected_first_char="1")
    _validate_tle_line(line2, expected_first_char="2")

    catnr1 = line1[2:7]
    catnr2 = line2[2:7]
    if catnr1 != catnr2:
        raise TleFetchError(
            f"Catalog number mismatch between lines: {catnr1!r} vs {catnr2!r}"
        )

    return Tle(name=name, line1=line1, line2=line2)


def _parse_multi_tle_text(text: str) -> list[Tle]:
    """Parse a CelesTrak multi-TLE response (group endpoint).

    Accepts both 3-line entries (name + line1 + line2) and 2-line entries
    (line1 + line2 only). CelesTrak's group endpoint emits 3-line by
    default, but the parser is tolerant of 2-line in case a future endpoint
    or a custom source ever drops the names.

    Args:
        text: Raw response body.

    Returns:
        A list of validated :class:`Tle` in the order they appear in the
        response.

    Raises:
        TleFetchError: On empty body, "no GP data found" sentinel, truncated
            entries, or per-line validation failure (length / prefix /
            checksum / catalog-number mismatch).
    """
    if not text or not text.strip():
        raise TleFetchError("Empty group TLE response.")

    if "no gp data found" in text.lower():
        raise TleFetchError(
            f"CelesTrak returned no group data: {text.strip()!r}"
        )

    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]

    tles: list[Tle] = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("1 "):
            # Two-line entry, no name.
            if i + 1 >= len(lines):
                raise TleFetchError(
                    f"Truncated TLE at line index {i}: missing line 2"
                )
            name: Optional[str] = None
            line1, line2 = lines[i], lines[i + 1]
            i += 2
        else:
            # Three-line entry: name + line1 + line2.
            if i + 2 >= len(lines):
                raise TleFetchError(
                    f"Truncated TLE at line index {i}: missing data lines"
                )
            name = lines[i].strip()
            line1, line2 = lines[i + 1], lines[i + 2]
            i += 3

        _validate_tle_line(line1, expected_first_char="1")
        _validate_tle_line(line2, expected_first_char="2")
        if line1[2:7] != line2[2:7]:
            raise TleFetchError(
                f"Catalog number mismatch in group entry: "
                f"{line1[2:7]!r} vs {line2[2:7]!r}"
            )
        tles.append(Tle(name=name, line1=line1, line2=line2))

    return tles


def _validate_tle_line(line: str, expected_first_char: str) -> None:
    """Validate one TLE data line: length, prefix, and mod-10 checksum."""
    if len(line) != 69:
        raise TleFetchError(
            f"TLE line has length {len(line)}, expected 69: {line!r}"
        )
    if line[0] != expected_first_char or line[1] != " ":
        raise TleFetchError(
            f"TLE line does not start with {expected_first_char!r} + space: "
            f"{line!r}"
        )
    expected = _checksum(line[:-1])
    last = line[-1]
    if not last.isdigit() or int(last) != expected:
        raise TleFetchError(
            f"TLE checksum mismatch (computed {expected}, found {last!r}) "
            f"on line: {line!r}"
        )


def _checksum(s: str) -> int:
    """TLE mod-10 checksum: digits add themselves, '-' adds 1, all else 0."""
    total = 0
    for ch in s:
        if ch.isdigit():
            total += int(ch)
        elif ch == "-":
            total += 1
    return total % 10
