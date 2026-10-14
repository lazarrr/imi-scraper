"""
enricher.py
~~~~~~~~~~~
DeepSeek validation + enrichment.

  Call A  analyze_page()  — page type / summary / date, and registry
                            drafts for program panels not in programs.yaml
  Call B  enrich()        — per chunk: context, questions, keywords,
                            quality flag and issues (batched per section,
                            batches run in parallel)

Without a client (--no-llm, or no API key in a dry run) deterministic
defaults are used, so the rest of the pipeline still works.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import Optional

from src.core.deepseek_client import DeepSeekClient, OutputTruncated
from src.processing import prompts
from src.processing.chunker import location_path
from src.processing.registry import Registry
from src.processing.schemas import Chunk, ChunkEnrichment, EnrichmentBatch, Page, PageAnalysis
from src.utils.text import truncate

logger = logging.getLogger(__name__)

_URL_TYPE_HINTS = (
    ("/vesti/", "vest"), ("kontakt", "kontakt"), ("upis", "upis"), ("raspored", "raspored"),
    ("ispit", "ispit"), ("oglas", "oglas"), ("osoblje", "osoblje"), ("projekt", "projekat"),
    ("studije", "studijski_program"),
)


class Enricher:
    def __init__(
        self,
        client: Optional[DeepSeekClient],
        settings,
        today: date,
        facts: Optional[list[str]] = None,
    ):
        self.client = client
        self.settings = settings
        self.today = today
        self.batch_size = settings.LLM_BATCH_CHUNKS
        # reviewed version maps — DeepSeek flags unversioned text that contradicts them
        self.facts = facts or []

    # ------------------------------------------------------------------ #
    #  Call A                                                              #
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

    # ------------------------------------------------------------------ #
    #  Call B                                                              #
    # ------------------------------------------------------------------ #

    def _run_batch(self, page: Page, analysis: PageAnalysis, batch: list[Chunk]) -> dict[str, ChunkEnrichment]:
        local = [(f"c{i}", chunk) for i, chunk in enumerate(batch)]
        first = batch[0]
        path = location_path(first.section.breadcrumb, first.heading_path)
        user = prompts.enrich_user(
            page, analysis, path, first.version, [(lid, c.text) for lid, c in local], self.facts
        )
        max_tokens = min(8000, 350 * len(batch) + 600)
        try:
            result = self.client.complete_json(prompts.ENRICH_SYSTEM, user, EnrichmentBatch, max_tokens)
        except OutputTruncated:
            if len(batch) == 1:
                logger.warning("Enrichment output truncated for one chunk on %s", page.url)
                return {}
            half = len(batch) // 2
            out = self._run_batch(page, analysis, batch[:half])
            out.update(self._run_batch(page, analysis, batch[half:]))
            return out
        except Exception as exc:
            logger.warning("DeepSeek enrichment failed on %s (%d chunks): %s", page.url, len(batch), exc)
            return {}

        by_local = {item.id: item for item in result.chunks}
        out: dict[str, ChunkEnrichment] = {}
        for lid, chunk in local:
            item = by_local.get(lid)
            if item is None:
                continue
            # versioned chunks are dated by the registry, never by the model
            outdated = item.outdated and chunk.version is None
            out[chunk.chunk_id] = ChunkEnrichment(
                id=chunk.chunk_id,
                context=truncate(item.context.strip(), 400),
                questions=[q.strip() for q in item.questions if q.strip()][:4],
                keywords=[k.strip() for k in item.keywords if k.strip()][:8],
                quality=item.quality,
                issues=[i.strip() for i in item.issues if i.strip()][:5],
                outdated=outdated,
                outdated_reason=truncate(item.outdated_reason.strip(), 200) if outdated else "",
            )
        return out

    def enrich(self, page: Page, analysis: PageAnalysis, chunks: list[Chunk]) -> dict[str, ChunkEnrichment]:
        if self.client is None or not chunks:
            return {}
        # batches never mix sections (they share the section header / version)
        batches: list[list[Chunk]] = []
        current: list[Chunk] = []
        for chunk in chunks:
            if current and (
                chunk.section.section_id != current[0].section.section_id
                or len(current) >= self.batch_size
            ):
                batches.append(current)
                current = []
            current.append(chunk)
        if current:
            batches.append(current)

        results: dict[str, ChunkEnrichment] = {}
        with ThreadPoolExecutor(max_workers=self.client.max_workers) as pool:
            for partial in pool.map(lambda b: self._run_batch(page, analysis, b), batches):
                results.update(partial)
        return results
