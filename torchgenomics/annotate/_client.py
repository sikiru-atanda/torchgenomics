"""NCBI HTTP client with rate limiting, retries, and response caching.

The client wraps both the NCBI Datasets v2 API and the legacy E-utilities with
a single `_get` / `_post` surface. It is network-required — there is no offline
fallback — but responses are cached in-process so repeated gene-detail or
ortholog lookups during a single `annotate_hits` call hit the API at most once.

Tests should inject a subclass that overrides `_get` / `_post` (see the
`FakeNCBIClient` fixture in tests/test_annotate.py).
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

DATASETS_BASE = "https://api.ncbi.nlm.nih.gov/datasets/v2alpha"
EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


class NCBIError(RuntimeError):
    """Raised when NCBI returns a non-retryable error or the retry budget is exhausted."""


class CropNotFoundError(NCBIError):
    """Raised when a common name / scientific name cannot be resolved to a taxid."""


class AssemblyNotFoundError(NCBIError):
    """Raised when no reference assembly is available for the given taxid."""


class _TokenBucket:
    """Simple thread-safe token bucket for request pacing."""

    def __init__(self, rate_per_sec: float) -> None:
        self._rate = float(rate_per_sec)
        self._min_interval = 1.0 / self._rate if self._rate > 0 else 0.0
        self._lock = threading.Lock()
        # Negative infinity so the first call never sleeps.
        self._last = float("-inf")

    def acquire(self, *, sleep=time.sleep, now=time.monotonic) -> None:
        """Block until at least `1/rate` seconds have passed since the last call."""
        with self._lock:
            t = now()
            wait = self._last + self._min_interval - t
            if wait > 0:
                sleep(wait)
                t = now()
            self._last = t


class NCBIClient:
    """HTTP client for NCBI Datasets v2 and E-utilities.

    Parameters
    ----------
    api_key : str or None
        NCBI API key. If None, falls back to the ``NCBI_API_KEY`` environment
        variable. With a key, rate limit is 10 req/s; without, 3 req/s.
    rate_per_sec : float or None
        Override the default rate limit. Useful in tests.
    timeout : float
        Per-request timeout in seconds.
    max_retries : int
        Number of retry attempts on 429 / 5xx.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        rate_per_sec: float | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.environ.get("NCBI_API_KEY")
        if rate_per_sec is None:
            rate_per_sec = 10.0 if self.api_key else 3.0
        self._bucket = _TokenBucket(rate_per_sec)
        self._timeout = timeout
        self._max_retries = max_retries
        self._cache: dict[str, Any] = {}
        self._session = None  # lazy — only built when a real request is made

    # ------------------------------------------------------------------ utilities

    def _get_session(self):  # type: ignore[no-untyped-def]
        if self._session is None:
            try:
                import requests
            except ImportError as e:
                raise NCBIError(
                    "torchgenomics.annotate requires the 'requests' package. "
                    "Install with `pip install requests`."
                ) from e
            self._session = requests.Session()
        return self._session

    def _cache_key(self, method: str, url: str, params: dict | None, body: dict | None) -> str:
        return f"{method}|{url}|{json.dumps(params or {}, sort_keys=True)}|{json.dumps(body or {}, sort_keys=True)}"

    def _datasets_headers(self) -> dict[str, str]:
        h = {"Accept": "application/json"}
        if self.api_key:
            h["api-key"] = self.api_key
        return h

    def _augment_eutils_params(self, params: dict | None) -> dict:
        p = dict(params or {})
        p.setdefault("retmode", "json")
        if self.api_key:
            p["api_key"] = self.api_key
        return p

    # ------------------------------------------------------------------ core I/O

    def _request(
        self,
        method: str,
        url: str,
        *,
        params: dict | None = None,
        body: dict | None = None,
        headers: dict | None = None,
    ) -> Any:
        """Issue a request with rate limiting, retries, and JSON caching."""
        key = self._cache_key(method, url, params, body)
        if key in self._cache:
            return self._cache[key]

        session = self._get_session()
        attempt = 0
        while True:
            self._bucket.acquire()
            try:
                if method == "GET":
                    resp = session.get(
                        url, params=params, headers=headers, timeout=self._timeout
                    )
                else:
                    resp = session.post(
                        url, params=params, json=body, headers=headers, timeout=self._timeout
                    )
            except Exception as e:  # noqa: BLE001 — network errors
                if attempt >= self._max_retries:
                    raise NCBIError(f"NCBI request failed: {e}") from e
                time.sleep(2.0**attempt)
                attempt += 1
                continue

            if resp.status_code == 200:
                payload = resp.json()
                self._cache[key] = payload
                return payload
            if resp.status_code in (429, 500, 502, 503, 504):
                if attempt >= self._max_retries:
                    raise NCBIError(
                        f"NCBI returned {resp.status_code} after {attempt + 1} attempts: {url}"
                    )
                time.sleep(2.0**attempt)
                attempt += 1
                continue
            raise NCBIError(
                f"NCBI returned HTTP {resp.status_code} for {url}: {resp.text[:200]}"
            )

    def _get(self, url: str, params: dict | None = None, *, eutils: bool = False) -> Any:
        if eutils:
            return self._request("GET", url, params=self._augment_eutils_params(params))
        return self._request("GET", url, params=params, headers=self._datasets_headers())

    def _post(self, url: str, body: dict) -> Any:
        return self._request("POST", url, body=body, headers=self._datasets_headers())
