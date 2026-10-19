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
from src.processing.schemas import (
    CuratedChunk,
    CurationResponse,
    DiscardedContent,
    PageAnalysis,
    ReconcileResponse,
    RegistryDraftEntry,
    SectionCuration,
    SourceEvidence,
)
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


_SECTION_RE = re.compile(r'<section id="([^"]+)" part="([^"]*)">\n(.*?)\n</section>', re.S)
_SOURCE_RE = re.compile(r"<source>\n(.*?)\n</source>", re.S)
_COURSE_NAME_RE = re.compile(r"Predmet: ([^|]+)")


def _first_quote(text: str) -> str | None:
    for line in text.splitlines():
        line = line.strip()
        if len(fold(line)) >= 4:
            return line[:300]
    return None


class FakeClient:
    """Stands in for DeepSeekClient.

    Page analysis drafts one real and one fabricated registry entry.  Content
    curation is deterministic and honest by default: it keeps the whole
    extracted text, turns table rows into row chunks plus an overview, and
    marks the FAQ's "3 smera" as outdated.  Hooks let tests inject invalid
    evidence, hallucinated facts, duplicate groups and section failures.
    """

    model = "fake-deepseek"
    thinking = False
    max_workers = 4

    def __init__(
        self,
        *,
        fail_sections: set[str] | None = None,
        curation_hook=None,
        reconcile_hook=None,
        discard_shorter_than: int = 40,
    ):
        self.usage = Usage()
        self.calls: list[str] = []
        self.fail_sections = fail_sections or set()
        self.curation_hook = curation_hook
        self.reconcile_hook = reconcile_hook
        self.discard_shorter_than = discard_shorter_than
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #

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
        if schema is CurationResponse:
            parsed = self._parse_sections(user)
            if self.curation_hook is not None:
                hooked = self.curation_hook(user, parsed)
                if hooked is not None:
                    return hooked
            return self._curate(parsed)
        if schema is ReconcileResponse:
            if self.reconcile_hook is not None:
                return self.reconcile_hook(user)
            return ReconcileResponse()
        raise AssertionError(f"unexpected schema {schema}")

    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_sections(user: str) -> list[dict]:
        out: list[dict] = []
        for sid, part, body in _SECTION_RE.findall(user):
            source = _SOURCE_RE.search(body)
            versioned = bool(re.search(r"^versioned: yes$", body, re.M))
            breadcrumb = re.search(r"^breadcrumb: (.*)$", body, re.M)
            version = re.search(r"^program_version: (.*)$", body, re.M)
            out.append({
                "id": sid,
                "part": part,
                "text": source.group(1) if source else body,
                "versioned": versioned,
                "breadcrumb": breadcrumb.group(1) if breadcrumb else sid,
                "version": version.group(1) if version else "",
            })
        return out

    def _curate(self, parsed: list[dict]) -> CurationResponse:
        sections: list[SectionCuration] = []
        for item in parsed:
            sid = item["id"]
            if sid in self.fail_sections:
                raise RuntimeError(f"forced failure for {sid}")
            text = item["text"]
            if len(text.strip()) < self.discard_shorter_than:
                sections.append(SectionCuration(
                    source_section_id=sid, status="no_relevant_content",
                    discarded_content=[DiscardedContent(
                        source_section_id=sid, description="prekratak/navigacioni tekst",
                        reason="boilerplate",
                    )],
                ))
                continue
            chunks: list[CuratedChunk] = []
            rows = [
                line.strip() for line in text.splitlines()
                if line.strip() and " | " in line and any(k in line for k in ("Predmet:", "Šifra:", "Semestar:"))
            ]
            if rows:
                for i in range(0, len(rows), 8):
                    group = rows[i:i + 8]
                    name = _COURSE_NAME_RE.search(group[0])
                    questions = (
                        [f"Koliko ESPB ima predmet {name.group(1).strip()}?",
                         f"U kom semestru se sluša {name.group(1).strip()}?"]
                        if name else ["Koji su predmeti u ovom semestru?"]
                    )
                    chunks.append(self._chunk(item, "\n".join(group), group[0], questions))
                names = []
                for row in rows:
                    match = _COURSE_NAME_RE.search(row)
                    if match and match.group(1).strip() not in names:
                        names.append(match.group(1).strip())
                if len(names) >= 2:
                    overview = f"{item['breadcrumb']} — predmeti: " + "; ".join(names[:20])
                    chunks.append(self._chunk(item, overview, rows[0], ["Koji predmeti postoje u ovom semestru?"]))
            else:
                quote = _first_quote(text)
                if quote is None:
                    sections.append(SectionCuration(
                        source_section_id=sid, status="no_relevant_content",
                        discarded_content=[DiscardedContent(
                            source_section_id=sid, description="nema citljivog teksta",
                            reason="empty",
                        )],
                    ))
                    continue
                chunks.append(self._chunk(
                    item, text, quote,
                    ["Šta je opisano u ovom delu stranice?", "Koje informacije su dostupne studentima?"],
                ))
            sections.append(SectionCuration(
                source_section_id=sid, status="processed", chunks=chunks,
            ))
        return CurationResponse(sections=sections)

    @staticmethod
    def _chunk(item: dict, content: str, quote: str, questions: list[str]) -> CuratedChunk:
        outdated = (
            not item["versioned"]
            and ("3 smera" in content or "tri smera" in content)
        )
        return CuratedChunk(
            content=content,
            context=f"{item['breadcrumb']} — deo stranice.",
            questions=questions[:4],
            keywords=["studije", "informatika"],
            source_evidence=[SourceEvidence(
                source_section_id=item["id"], source_excerpt=quote,
            )],
            quality="ok",
            outdated=outdated,
            outdated_reason="Navodi 3 smera, a aktuelni program ima 2 modula." if outdated else "",
            issues=["tabela bez zaglavlja"] if "Zbir" in content else [],
        )


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
