"""
enricher.py
~~~~~~~~~~~
DeepSeek page analysis (formerly the enrichment stage).

  analyze_page()  — page type / summary / date, and registry drafts for
                    program panels that are not in programs.yaml.

The content curation itself (context, questions, keywords, final chunks)
now lives in `curator.py`; this module is deliberately small: it is the
"what kind of page is this and which program versions does it describe"
pass, whose output the curator receives as page context.

Without a client (--no-llm, or no API key in a dry run) deterministic
defaults are used, so the rest of the pipeline still works.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

from src.core.deepseek_client import DeepSeekClient
from src.processing import prompts
from src.processing.registry import Registry
from src.processing.schemas import Page, PageAnalysis

logger = logging.getLogger(__name__)

_URL_TYPE_HINTS = (
    ("/vesti/", "vest"), ("kontakt", "kontakt"), ("upis", "upis"), ("raspored", "raspored"),
    ("ispit", "ispit"), ("oglas", "oglas"), ("osoblje", "osoblje"), ("projekt", "projekat"),
    ("studije", "studijski_program"),
)


class PageAnalyzer:
    def __init__(self, client: Optional[DeepSeekClient], settings, today: date):
        self.client = client
        self.settings = settings
        self.today = today

    # ------------------------------------------------------------------ #
    #  Heuristic fallback (--no-llm / API failure)                         #
    # ------------------------------------------------------------------ #

    def heuristic_analysis(self, page: Page, registry: Registry) -> PageAnalysis:
        if page.doc_type in ("pdf", "docx"):
            page_type = "dokument"
        elif registry.entries_for_page(page.url):
            page_type = "studijski_program"
        else:
            page_type = next((t for hint, t in _URL_TYPE_HINTS if hint in page.url.lower()), "opste")
        published = page.published_hint[:10] if page.published_hint else None
        return PageAnalysis(page_type=page_type, published_date=published)

    # ------------------------------------------------------------------ #
    #  Call A                                                             #
    # ------------------------------------------------------------------ #

    def analyze_page(self, page: Page, registry: Registry, unmapped: list[str]) -> PageAnalysis:
        if self.client is None:
            return self.heuristic_analysis(page, registry)
        mapped = [
            f"{e.version_label} (panels: {', '.join(e.panels)})"
            for e in registry.entries_for_page(page.url)
        ]
        sections = [s for s in page.sections if s.reachable and s.blocks]
        user = prompts.analyze_user(page, sections, self.today.isoformat(), unmapped, mapped)
        try:
            analysis = self.client.complete_json(
                prompts.ANALYZE_SYSTEM, user, PageAnalysis, max_tokens=4000
            )
        except Exception as exc:
            logger.warning("DeepSeek page analysis failed for %s: %s", page.url, exc)
            analysis = self.heuristic_analysis(page, registry)
            analysis.page_summary = ""
            return analysis
        # drafts may only reference panels that are actually unmapped
        allowed = set(unmapped)
        for draft in analysis.registry_draft:
            draft.panels = [p for p in draft.panels if p in allowed]
        analysis.registry_draft = [d for d in analysis.registry_draft if d.panels]
        return analysis
