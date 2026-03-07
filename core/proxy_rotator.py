"""Proxy Rotator — Fetches and rotates free proxies via Proxifly.

Proxifly validates proxies every 5 minutes from 82 countries.
Used to bypass Instagram's datacenter IP blocking on EC2.
"""

import random
import threading
import time

import requests

from utils.logger import log

# Proxifly CDN — pre-validated, updated every 5 minutes
_PROXY_SOURCES = [
    "https://cdn.jsdelivr.net/gh/proxifly/free-proxy-list@main/proxies/protocols/http/data.txt",
    "https://cdn.jsdelivr.net/gh/proxifly/free-proxy-list@main/proxies/protocols/https/data.txt",
]

_TEST_URL = "https://www.instagram.com/favicon.ico"
_TEST_TIMEOUT = 5
_REFRESH_INTERVAL = 600  # 10 minutes (Proxifly updates every 5)


class ProxyRotator:
    """Manages a pool of validated proxies from Proxifly."""

    def __init__(self, max_proxies: int = 20):
        self._proxies: list[str] = []
        self._index = 0
        self._lock = threading.Lock()
        self._max = max_proxies
        self._last_refresh = 0.0

    def _fetch_proxy_list(self) -> list[str]:
        """Fetch pre-validated proxies from Proxifly CDN."""
        raw_proxies = set()
        for src in _PROXY_SOURCES:
            try:
                r = requests.get(src, timeout=10)
                if r.ok:
                    for line in r.text.strip().splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        # Strip protocol prefix (Proxifly format: http://ip:port)
                        proxy = line.split()[0]
                        proxy = proxy.replace("http://", "").replace("https://", "")
                        if proxy and proxy.count(":") == 1:
                            raw_proxies.add(proxy)
                    log.info("[proxy] Fetched %d proxies from %s", len(raw_proxies), src.split("/")[-2])
            except Exception as e:
                log.warning("[proxy] Source failed: %s", e)

        log.info("[proxy] Total: %d unique proxies from Proxifly", len(raw_proxies))
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
            log.warning("[proxy] No proxies fetched")
            return

        random.shuffle(raw)
        candidates = raw[:100]

        working = []
        log.info("[proxy] Testing %d candidates...", len(candidates))
        for i, proxy in enumerate(candidates):
            if self._test_proxy(proxy):
                working.append(proxy)
                log.info("[proxy] Working %d/%d: %s", len(working), self._max, proxy)
                if len(working) >= self._max:
                    break
            if (i + 1) % 25 == 0:
                log.info("[proxy] Progress: tested %d, found %d working", i + 1, len(working))

        with self._lock:
            self._proxies = working
            self._index = 0
            self._last_refresh = time.time()

        log.info("[proxy] Pool ready: %d working proxies", len(working))

    def get(self) -> str | None:
        """Get next proxy (round-robin). Returns 'ip:port' or None."""
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
                log.debug("[proxy] Removed %s (%d left)", proxy, len(self._proxies))

    @property
    def count(self) -> int:
        return len(self._proxies)
