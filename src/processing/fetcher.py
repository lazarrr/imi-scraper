"""
fetcher.py
~~~~~~~~~~
HTTP fetching + URL identity.

  • canonicalize_url() — identity key of a page: lower-case host, https,
    no fragment / trailing slash / tracking params, sorted query, and
    WITHOUT the site's tab-view params (?osnovne …) — those return the
    same content and used to create 4× duplicates with wrong links.
  • Fetcher            — requests session with retries + backoff,
    robots.txt, polite per-host delay, optional raw-body cache
    (--cached) and probe-verified tab deep-link params.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import requests
from requests.adapters import HTTPAdapter
from requests.utils import requote_uri
from urllib3.util.retry import Retry

from src.processing.site_profiles import profile_for
from src.utils.text import sha1

logger = logging.getLogger(__name__)

_TRACKING_RE = re.compile(r"^(utm_\w+|fbclid|gclid|mc_cid|mc_eid)$", re.I)
# Server-injected tab click:  $("#tabela_osn_click_inf").click();
# (double quotes — the page's own JS uses single quotes)
_INJECTED_CLICK_RE = re.compile(r'\$\("#([A-Za-z0-9_\-]+)"\)\.click\(\);')

DOC_EXTENSIONS = (".pdf", ".docx")


class FetchError(Exception):
    pass


# --------------------------------------------------------------------------- #
#  URL helpers                                                                 #
# --------------------------------------------------------------------------- #

def canonicalize_url(url: str, allowed_domains: Iterable[str] = ()) -> str:
    """Identity key for a URL (see module docstring)."""
    parts = urlsplit(requote_uri(url.strip()))
    host = (parts.hostname or "").lower()
    scheme = (parts.scheme or "https").lower()
    if any(host == d or host.endswith("." + d) for d in allowed_domains):
        scheme = "https"
    netloc = host if parts.port in (None, 80, 443) else f"{host}:{parts.port}"

    path = re.sub(r"/{2,}", "/", parts.path or "/")
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    if path == "/":
        path = ""

    profile = profile_for(url)
    query = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if not _TRACKING_RE.match(k) and not (profile and profile.is_view_param(k))
    ]
    query.sort()
    return urlunsplit((scheme, netloc, path, urlencode(query, doseq=True), ""))


def display_url(final_url: str) -> str:
    """Human-facing link: the URL as served, minus fragment / view params."""
    parts = urlsplit(final_url)
    profile = profile_for(final_url)
    query = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if not _TRACKING_RE.match(k) and not (profile and profile.is_view_param(k))
    ]
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query, doseq=True), ""))


def is_allowed_domain(url: str, allowed_domains: Iterable[str]) -> bool:
    host = (urlsplit(url).hostname or "").lower()
    return any(host == d or host.endswith("." + d) for d in allowed_domains)


def is_document_url(url: str) -> bool:
    return urlsplit(url).path.lower().endswith(DOC_EXTENSIONS)


def with_param(url: str, param: str) -> str:
    """Append a bare tab param:  https://x/informatika-studije?osnovne"""
    parts = urlsplit(url)
    query = f"{parts.query}&{param}" if parts.query else param
    return urlunsplit((parts.scheme, parts.netloc, parts.path, query, ""))


# --------------------------------------------------------------------------- #
#  Fetcher                                                                     #
# --------------------------------------------------------------------------- #

@dataclass
class FetchResult:
    url: str                # requested URL
    final_url: str          # after redirects
    status: int
    content_type: str
    body: bytes
    encoding: Optional[str]

    @property
    def kind(self) -> str:
        """'html' | 'pdf' | 'docx' | 'other' — by content type, then extension."""
        ct = self.content_type.lower()
        path = urlsplit(self.final_url).path.lower()
        if "pdf" in ct or path.endswith(".pdf"):
            return "pdf"
        if "officedocument.wordprocessingml" in ct or path.endswith(".docx"):
            return "docx"
        if "html" in ct or "xml" in ct or not ct:
            return "html"
        return "other"

    def text(self) -> str:
        return self.body.decode(self.encoding or "utf-8", errors="replace")


class Fetcher:
    def __init__(self, settings, cache_dir: Path, use_cache: bool = False):
        self.settings = settings
        self.delay = settings.CRAWL_DELAY
        self.timeout = settings.CRAWL_TIMEOUT
        self.user_agent = settings.CRAWL_USER_AGENT
        self.respect_robots = settings.RESPECT_ROBOTS
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        retry = Retry(
            total=3,
            backoff_factor=1.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET", "HEAD"),
            respect_retry_after_header=True,
        )
        self._session = requests.Session()
        self._session.mount("http://", HTTPAdapter(max_retries=retry))
        self._session.mount("https://", HTTPAdapter(max_retries=retry))
        self._session.headers.update({
            "User-Agent": self.user_agent,
            "Accept-Language": "sr,sr-Latn;q=0.9,en;q=0.5",
        })
        self._last_hit: dict[str, float] = {}
        self._robots: dict[str, Optional[RobotFileParser]] = {}

    # ------------------------------------------------------------------ #
    #  Politeness                                                          #
    # ------------------------------------------------------------------ #

    def _wait_turn(self, host: str) -> None:
        last = self._last_hit.get(host)
        if last is not None:
            remaining = self.delay - (time.monotonic() - last)
            if remaining > 0:
                time.sleep(remaining)
        self._last_hit[host] = time.monotonic()

    def allowed_by_robots(self, url: str) -> bool:
        if not self.respect_robots:
            return True
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin not in self._robots:
            parser = RobotFileParser()
            try:
                resp = self._session.get(origin + "/robots.txt", timeout=self.timeout)
                if resp.status_code == 200:
                    parser.parse(resp.text.splitlines())
                else:
                    parser = None  # no robots.txt → everything allowed
            except requests.RequestException:
                parser = None
            self._robots[origin] = parser
        parser = self._robots[origin]
        return True if parser is None else parser.can_fetch(self.user_agent, url)

    # ------------------------------------------------------------------ #
    #  Cache                                                               #
    # ------------------------------------------------------------------ #

    def _cache_paths(self, url: str) -> tuple[Path, Path]:
        key = sha1(url)
        return self.cache_dir / f"{key}.body", self.cache_dir / f"{key}.json"

    def _from_cache(self, url: str) -> Optional[FetchResult]:
        body_path, meta_path = self._cache_paths(url)
        if not (body_path.is_file() and meta_path.is_file()):
            return None
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return FetchResult(body=body_path.read_bytes(), **meta)

    def _to_cache(self, result: FetchResult) -> None:
        body_path, meta_path = self._cache_paths(result.url)
        body_path.write_bytes(result.body)
        meta = {k: getattr(result, k) for k in ("url", "final_url", "status", "content_type", "encoding")}
        meta_path.write_text(json.dumps(meta), encoding="utf-8")

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def fetch(self, url: str) -> FetchResult:
        url = requote_uri(url.strip())
        if self.use_cache:
            cached = self._from_cache(url)
            if cached is not None:
                return cached

        if not self.allowed_by_robots(url):
            raise FetchError(f"Blocked by robots.txt: {url}")

        self._wait_turn(urlsplit(url).netloc)
        try:
            resp = self._session.get(url, timeout=self.timeout, allow_redirects=True)
        except requests.RequestException as exc:
            raise FetchError(f"Request failed for {url}: {exc}") from exc
        if resp.status_code >= 400:
            raise FetchError(f"HTTP {resp.status_code} for {url}")

        encoding = resp.encoding if "charset" in resp.headers.get("Content-Type", "").lower() else None
        result = FetchResult(
            url=url,
            final_url=resp.url,
            status=resp.status_code,
            content_type=resp.headers.get("Content-Type", ""),
            body=resp.content,
            encoding=encoding or (resp.apparent_encoding if "html" in resp.headers.get("Content-Type", "") else None),
        )
        self._to_cache(result)
        return result

    def probe_tab_params(
        self, page_url: str, candidates: Iterable[str], valid_triggers: Iterable[str]
    ) -> dict[str, str]:
        """
        For each candidate ?param, fetch the page once and see which tab
        trigger the server injects.  Keep the param only if that trigger
        is one of this page's top-level tabs.  Returns {trigger: param}.
        """
        valid = set(valid_triggers)
        found: dict[str, str] = {}
        for param in candidates:
            try:
                html = self.fetch(with_param(page_url, param)).text()
            except FetchError as exc:
                logger.warning("Tab probe failed for %s?%s: %s", page_url, param, exc)
                continue
            for trigger in _INJECTED_CLICK_RE.findall(html):
                if trigger in valid and trigger not in found:
                    found[trigger] = param
        return found
