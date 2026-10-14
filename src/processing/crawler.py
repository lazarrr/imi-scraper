"""
crawler.py
~~~~~~~~~~
Crawl frontier.  Only what the links file asks for is fetched:

  depth=0 (default)  just the listed page
  depth=N            + same-domain page links, N levels deep
  docs=true          + linked PDF/DOCX matching CRAWL_DOC_PATTERNS
                       (capped — program pages link hundreds of syllabi)

URLs are de-duplicated by canonical form, so ?osnovne / trailing-slash
variants are fetched once.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterator, Optional

from src.processing.fetcher import canonicalize_url, is_allowed_domain, is_document_url
from src.processing.links import LinkSpec
from src.processing.schemas import Page


@dataclass
class CrawlItem:
    target: str
    depth: int
    docs: bool
    is_file: bool
    parent: Optional[str] = None


class Frontier:
    def __init__(self, links: list[LinkSpec], settings):
        self.settings = settings
        self.queue: deque[CrawlItem] = deque(
            CrawlItem(spec.target, spec.depth, spec.docs, spec.is_file) for spec in links
        )
        self.seen: set[str] = set()
        self.yielded = 0

    def key(self, item: CrawlItem) -> str:
        return item.target if item.is_file else canonicalize_url(
            item.target, self.settings.CRAWL_ALLOWED_DOMAINS
        )

    def __iter__(self) -> Iterator[CrawlItem]:
        while self.queue and self.yielded < self.settings.CRAWL_MAX_PAGES:
            item = self.queue.popleft()
            key = self.key(item)
            if key in self.seen:
                continue
            self.seen.add(key)
            self.yielded += 1
            yield item

    def _page_link_allowed(self, url: str) -> bool:
        lowered = url.lower()
        return (
            is_allowed_domain(url, self.settings.CRAWL_ALLOWED_DOMAINS)
            and not is_document_url(url)
            and not any(p.lower() in lowered for p in self.settings.CRAWL_EXCLUDE_PATTERNS)
        )

    def add_children(self, item: CrawlItem, page: Page) -> None:
        if item.is_file:
            return
        if item.depth > 0:
            for link in page.links:
                if self._page_link_allowed(link):
                    self.queue.append(CrawlItem(link, item.depth - 1, item.docs, False, page.url))
        if item.docs:
            docs = [
                d for d in page.doc_links
                if is_allowed_domain(d, self.settings.CRAWL_ALLOWED_DOMAINS)
                and any(p in d for p in self.settings.CRAWL_DOC_PATTERNS)
            ]
            for doc in docs[: self.settings.CRAWL_MAX_DOCS_PER_PAGE]:
                self.queue.append(CrawlItem(doc, 0, False, False, page.url))
