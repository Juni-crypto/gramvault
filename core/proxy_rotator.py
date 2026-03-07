"""Proxy Rotator — Fetches and rotates free HTTPS proxies.

Used by instaloader to bypass Instagram's datacenter IP blocking.
Fetches proxy lists from public sources, validates them, and rotates.
"""

import random
import threading
import time

import requests

from utils.logger import log

# Public proxy list sources (raw text, one IP:PORT per line)
_PROXY_SOURCES = [
    "https://raw.githubusercontent.com/TheSpeedX/PROXY-List/master/http.txt",
    "https://raw.githubusercontent.com/clarketm/proxy-list/master/proxy-list-raw.txt",
    "https://raw.githubusercontent.com/ShiftyTR/Proxy-List/master/https.txt",
]

_TEST_URL = "https://www.instagram.com/favicon.ico"
_TEST_TIMEOUT = 5
_REFRESH_INTERVAL = 1800  # 30 minutes


class ProxyRotator:
    """Manages a pool of validated proxies."""

    def __init__(self, max_proxies: int = 20):
        self._proxies: list[str] = []
        self._index = 0
        self._lock = threading.Lock()
        self._max = max_proxies
        self._last_refresh = 0.0

    def _fetch_proxy_list(self) -> list[str]:
        """Fetch raw proxy list from public sources."""
        raw_proxies = set()
        for src in _PROXY_SOURCES:
            try:
                r = requests.get(src, timeout=10)
                if r.ok:
                    for line in r.text.strip().splitlines():
                        line = line.strip()
                        if line and ":" in line:
                            # Some lists have extra columns, take just ip:port
                            proxy = line.split()[0]
                            if proxy.count(":") == 1:
                                raw_proxies.add(proxy)
            except Exception as e:
                log.debug("Proxy source %s failed: %s", src, e)

        log.info("Fetched %d raw proxies from %d sources", len(raw_proxies), len(_PROXY_SOURCES))
        return list(raw_proxies)

    def _test_proxy(self, proxy: str) -> bool:
        """Quick test if proxy can reach Instagram."""
        try:
            r = requests.get(
                _TEST_URL,
                proxies={"http": f"http://{proxy}", "https": f"http://{proxy}"},
                timeout=_TEST_TIMEOUT,
            )
            return r.status_code < 500
        except Exception:
            return False

    def refresh(self):
        """Fetch and validate proxies."""
        raw = self._fetch_proxy_list()
        if not raw:
            log.warning("No proxies fetched from any source")
            return

        # Shuffle and test a sample (testing all would be too slow)
        random.shuffle(raw)
        candidates = raw[:80]

        working = []
        log.info("[proxy] Testing %d proxy candidates (this takes ~30s)...", len(candidates))
        for i, proxy in enumerate(candidates):
            if self._test_proxy(proxy):
                working.append(proxy)
                log.info("[proxy] Found working proxy %d/%d: %s", len(working), self._max, proxy)
                if len(working) >= self._max:
                    break
            if (i + 1) % 20 == 0:
                log.info("[proxy] Tested %d/%d, found %d working so far", i + 1, len(candidates), len(working))

        with self._lock:
            self._proxies = working
            self._index = 0
            self._last_refresh = time.time()

        log.info("[proxy] Proxy pool ready: %d working proxies", len(working))

    def get(self) -> str | None:
        """Get next proxy (round-robin). Returns 'ip:port' or None."""
        # Auto-refresh if stale or empty
        if not self._proxies or (time.time() - self._last_refresh > _REFRESH_INTERVAL):
            self.refresh()

        with self._lock:
            if not self._proxies:
                return None
            proxy = self._proxies[self._index % len(self._proxies)]
            self._index += 1
            return proxy

    def remove(self, proxy: str):
        """Remove a dead proxy from the pool."""
        with self._lock:
            if proxy in self._proxies:
                self._proxies.remove(proxy)
                log.debug("Removed dead proxy %s (%d remaining)", proxy, len(self._proxies))

    @property
    def count(self) -> int:
        return len(self._proxies)
