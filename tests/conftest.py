"""
Shared test fakes — no network, no DeepSeek, no 2.2 GB embedding model.
"""

from __future__ import annotations

import hashlib
import re
import threading
from pathlib import Path

import numpy as np
import pytest

from config.settings import Settings
from src.core.deepseek_client import Usage
from src.processing.fetcher import FetchResult
from src.processing.schemas import ChunkEnrichment, EnrichmentBatch, PageAnalysis, RegistryDraftEntry
from src.utils.text import fold

FIXTURES = Path(__file__).parent / "fixtures"

INF_URL = "https://imi.pmf.kg.ac.rs/informatika-studije"
MAT_URL = "https://imi.pmf.kg.ac.rs/matematika-studije"
NEWS_URL = "https://imi.pmf.kg.ac.rs/vesti/11836-novi-smerovi-na-oas-matematike-i-oas-informatike"

PAGES = {
    INF_URL: "informatika-studije.html",
    MAT_URL: "matematika-studije.html",
    NEWS_URL: "vest-11836.html",
}


def fixture_html(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


class FakeFetcher:
    """Serves the saved pages; tab probes return what the live site does."""

    PROBES = {
        INF_URL: {
            "tabela_osn_click_inf": "osnovne",
            "tabela_osn_u_akreditaciji_click_inf": "osnovne_u_akreditaciji",
            "tabela_mas_u_akreditaciji_click_inf": "master_u_akreditaciji",
            "tabela_dok_click_inf": "doktorske",
        },
        MAT_URL: {
            "tabela_osn_click": "osnovne",
            "tabela_osn_click_u_akreditaciji": "osnovne_u_akreditaciji",
            "tabela_mas_click": "master",
            "tabela_dok_click": "doktorske",
        },
    }

    def __init__(self):
        self.fetched: list[str] = []

    def fetch(self, url: str) -> FetchResult:
        self.fetched.append(url)
        base = url.split("?")[0].rstrip("/")
        name = PAGES.get(base)
        if name is None:
            from src.processing.fetcher import FetchError
            raise FetchError(f"HTTP 404 for {url}")
        return FetchResult(
            url=url, final_url=url, status=200, content_type="text/html; charset=UTF-8",
            body=(FIXTURES / name).read_bytes(), encoding="utf-8",
        )

    def probe_tab_params(self, page_url, candidates, valid_triggers):
        valid = set(valid_triggers)
        return {t: p for t, p in self.PROBES.get(page_url, {}).items() if t in valid}


class FakeClient:
    """Stands in for DeepSeekClient.  Page analysis drafts one real and one
    fabricated registry entry; enrichment marks 'Misija'-style text ok and
    one-line link text as low_value."""

    model = "fake-deepseek"
    thinking = False
    max_workers = 4

    def __init__(self):
        self.usage = Usage()
        self.calls: list[str] = []
        self._lock = threading.Lock()

    def complete_json(self, system, user, schema, max_tokens):
        with self._lock:
            self.calls.append(schema.__name__)
        if schema is PageAnalysis:
            drafts = []
            if "UNMAPPED PANELS: " in user and "panel5inf" in user.split("UNMAPPED PANELS: ")[1].split("\n")[0]:
                drafts = [
                    RegistryDraftEntry(
                        program="Informatika", level="doktorske", version_label="DAS Informatika — opšte",
                        panels=["panel5inf"], evidence=["Doktorske akademske studije traju 3 godine"],
                    ),
                    RegistryDraftEntry(
                        program="Informatika", level="osnovne", version_label="Izmišljen program",
                        panels=["panel_faq"], evidence=["ovaj citat ne postoji na stranici"],
                    ),
                ]
            return PageAnalysis(
                page_type="studijski_program" if "studije" in user else "vest",
                page_summary="Kratak opis stranice.",
                registry_draft=drafts,
            )
        if schema is EnrichmentBatch:
            items = []
            for local_id, text in re.findall(r'<chunk id="(c\d+)">\n(.*?)\n</chunk>', user, re.S):
                quality = "low_value" if len(text) < 80 else "ok"
                # the real FAQ still says "3 smera" — contradicts the facts (2 modules)
                outdated = "PROGRAM FACTS" in user and ("3 smera" in text or "tri smera" in text)
                items.append(ChunkEnrichment(
                    id=local_id, context="Deo stranice o studijama.",
                    questions=["Koliko ESPB ima predmet?"], keywords=["studije"],
                    quality=quality, issues=["tabela bez zaglavlja"] if "Zbir" in text else [],
                    outdated=outdated,
                    outdated_reason="Navodi 3 smera, a aktuelni program ima 2 modula." if outdated else "",
                ))
            return EnrichmentBatch(chunks=items)
        raise AssertionError(f"unexpected schema {schema}")


class FakeEmbedder:
    """Hashing bag-of-words vectors: deterministic, normalized, 256 dims."""

    dims = 256

    def _vec(self, text: str) -> list[float]:
        vec = np.zeros(self.dims, dtype=np.float32)
        for token in fold(text).split():
            vec[int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dims] += 1.0
        norm = np.linalg.norm(vec)
        return (vec / norm if norm else vec).tolist()

    def embed_passages(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


@pytest.fixture
def test_settings(tmp_path) -> Settings:
    return Settings(
        DATA_DIR=str(tmp_path / "data"),
        PROGRAMS_DRAFT_FILE=str(tmp_path / "programs.draft.yaml"),
        CRAWL_DELAY=0,
        DEEPSEEK_API_KEY=None,
    )
