"""
ingest.py
~~~~~~~~~
Pipeline orchestration:

  links file → fetch → parse (sections, breadcrumbs, reachability)
    → versions (programs.yaml; DeepSeek drafts for unmapped panels)
    → DeepSeek page analysis
    → DeepSeek content curation (final chunks, or deterministic chunks
      with --no-llm) + version-map chunks
    → validation, dedupe, quality filter
    → header / document / embedded text / metadata
    → E5 embeddings → Chroma (upsert, then delete stale ids of the page)
    → report.md + chunks.jsonl + curation.json + state

A page is only committed when its curation is complete (or
`--allow-partial-write` is given), so a failed model call can never delete
the last known-good chunks of a page.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path
from typing import Optional

from src.core.deepseek_client import DeepSeekClient
from src.processing.chunker import (
    TokenCounter,
    build_header,
    chunk_section,
    location_path,
    pack_units,
)
from src.processing.crawler import CrawlItem, Frontier
from src.processing.curator import Curator, PageCuration
from src.processing.documents import DocxParser, PdfParser
from src.processing.enricher import PageAnalyzer
from src.processing.fetcher import Fetcher, canonicalize_url, display_url, with_param
from src.processing.html_parser import parse_html
from src.processing.links import parse_links_file
from src.processing.prompts import CURATION_SCHEMA_VERSION, PROMPT_VERSION
from src.processing.registry import Registry, check_draft, write_drafts
from src.processing.report import PageReport, RunReport, SectionRow
from src.processing.schemas import (
    Chunk,
    ChunkEnrichment,
    Page,
    PageAnalysis,
    Section,
    VersionInfo,
)
from src.processing.site_profiles import profile_for
from src.processing.state import IngestState
from src.utils.text import TextIndex, fold, sha1, slugify, split_units, truncate

logger = logging.getLogger(__name__)

# Bump when parsing / chunking / curation / embedded-text logic changes, so
# pages ingested by older code are re-processed instead of skipped.
PIPELINE_VERSION = "2026-10-19.1"

DROP_QUALITIES = ("boilerplate", "low_value")
DATED_PAGE_TYPES = ("vest", "oglas", "dogadjaj")

# "Predmet: Osnovi programiranja |"  /  "Izborni predmeti grupe G1: Engleski jezik A1 |"
_COURSE_RE = re.compile(r"(?:^|\| |— )(?:Predmet|Izborni predmeti?[^:|—\n]*): ([^|\n]+?)\s*(?=\||$)", re.M)
_PLACEHOLDER_RE = re.compile(r"^(izborni predmet\b|zbir|ukupno)", re.I)


def course_questions(chunk_text: str, limit: int = 2) -> list[str]:
    """Deterministic questions for course-table rows, embedded with the row
    (never shown to the chatbot) so "Koliko ESPB ima X?" hits X's row."""
    names: list[str] = []
    for name in _COURSE_RE.findall(chunk_text):
        name = name.strip()
        if name and not _PLACEHOLDER_RE.match(name) and name not in names:
            names.append(name)
    questions: list[str] = []
    for name in names[:limit]:
        questions += [f"Koliko ESPB ima predmet {name}?", f"U kom semestru se sluša {name}?"]
    return questions


@dataclass
class PageResult:
    chunks: list[Chunk] = field(default_factory=list)
    dropped: list[Chunk] = field(default_factory=list)
    analysis: Optional[PageAnalysis] = None
    drafts: list[dict] = field(default_factory=list)
    curation: Optional[PageCuration] = None
    complete: bool = True
    incomplete_reason: str = ""
    embed_split: int = 0


class IngestPipeline:
    def __init__(
        self,
        settings,
        *,
        links_file: Path,
        collection: str,
        db_path: Path,
        dry_run: bool = False,
        no_llm: bool = False,
        force: bool = False,
        cached: bool = False,
        keep_low_value: bool = False,
        include_unreachable: bool = False,
        allow_partial_write: bool = False,
        today: Optional[date] = None,
        fetcher=None,
        client=None,
        embedder=None,
        store=None,
        token_counter: Optional[TokenCounter] = None,
    ):
        self.settings = settings
        self.links_file = links_file
        self.collection = collection
        self.db_path = db_path
        self.dry_run = dry_run
        self.force = force
        self.keep_low_value = keep_low_value
        self.include_unreachable = include_unreachable
        self.allow_partial_write = allow_partial_write
        self.today = today or date.today()
        self.domains = settings.CRAWL_ALLOWED_DOMAINS

        data_dir = settings.data_dir
        self.data_dir = data_dir
        self.registry = Registry.load(settings.path(settings.PROGRAMS_FILE), self.today, self.domains)
        self.fetcher = fetcher or Fetcher(settings, data_dir / "cache" / "http", use_cache=cached)
        self.state = IngestState(data_dir / "state.json")

        if client is None and not no_llm:
            client = DeepSeekClient(settings, data_dir / "cache" / "llm.sqlite")
        self.client = None if no_llm else client
        self.analyzer = PageAnalyzer(self.client, settings, self.today)

        self.counter = token_counter or TokenCounter(settings.EMBEDDING_MODEL)
        self.curator: Optional[Curator] = (
            Curator(self.client, settings, self.registry, self.counter) if self.client else None
        )

        self._embedder = embedder
        self._store = store

    # ------------------------------------------------------------------ #
    #  Lazy heavy components                                               #
    # ------------------------------------------------------------------ #

    @property
    def embedder(self):
        if self._embedder is None:
            from src.core.embeddings import E5Embedder

            self._embedder = E5Embedder(
                self.settings.EMBEDDING_MODEL,
                device=self.settings.EMBEDDING_DEVICE,
                batch_size=self.settings.EMBEDDING_BATCH_SIZE,
                cache_path=self.data_dir / "cache" / "embeddings.sqlite",
            )
        return self._embedder

    @property
    def store(self):
        if self._store is None:
            from src.core.vector_store import ChromaStore

            self._store = ChromaStore(self.db_path, self.collection)
        return self._store

    # ------------------------------------------------------------------ #
    #  Loading                                                             #
    # ------------------------------------------------------------------ #

    def _load(self, item: CrawlItem) -> Page:
        if item.is_file:
            path = Path(item.target)
            if not path.is_file():
                raise FileNotFoundError(f"local file not found: {path}")
            data, url, shown = path.read_bytes(), path.resolve().as_uri(), str(path)
            if path.suffix.lower() == ".pdf":
                return PdfParser().parse(data, url, shown)
            if path.suffix.lower() == ".docx":
                return DocxParser().parse(data, url, shown)
            raise ValueError(f"unsupported local file type: {path.suffix}")

        result = self.fetcher.fetch(item.target)
        canonical = canonicalize_url(result.final_url, self.domains)
        shown = display_url(result.final_url)
        if result.kind == "html":
            return parse_html(result.text(), result.final_url, result.final_url, self.domains)
        if result.kind == "pdf":
            return PdfParser().parse(result.body, canonical, shown)
        if result.kind == "docx":
            return DocxParser().parse(result.body, canonical, shown)
        raise ValueError(f"unsupported content type {result.content_type!r}")

    def _probe_tabs(self, page: Page) -> None:
        if page.doc_type != "web" or not page.top_triggers:
            return
        profile = profile_for(page.display_url)
        if profile is None or not profile.tab_param_candidates:
            return
        params = None if self.force else self.state.tab_params(page.url)
        if params is None:
            params = self.fetcher.probe_tab_params(
                page.display_url, profile.tab_param_candidates, page.top_triggers.keys()
            )
            self.state.set_tab_params(page.url, params)
        page.tab_params = params

    def _signature(self, page: Page, text_hash: str) -> str:
        s = self.settings
        parts = [
            text_hash, PIPELINE_VERSION, PROMPT_VERSION, self.registry.page_hash(page.url),
            s.EMBEDDING_MODEL, s.CHUNK_MAX_TOKENS, s.CHUNK_TABLE_MAX_TOKENS,
            s.CHUNK_OVERLAP_TOKENS, s.MIN_CHUNK_CHARS,
            s.EMBED_QUESTIONS, s.EMBED_MAX_TOKENS, self.counter.exact,
            self.include_unreachable, self.keep_low_value,
            "curation" if self.curator else "deterministic",
            s.CURATION_INPUT_TOKENS, s.CURATION_MAX_OUTPUT_TOKENS,
            s.CURATION_CHUNK_MIN_TOKENS, s.CURATION_CHUNK_TARGET_TOKENS,
            s.CURATION_CHUNK_MAX_TOKENS, s.CURATION_STRICT_FACTS,
            s.CURATION_RECONCILE, s.CURATION_RECONCILE_MAX_CHUNKS,
            page.tab_params,
        ]
        return sha1(json.dumps(parts, sort_keys=True, default=str))

    # ------------------------------------------------------------------ #
    #  Versions                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _module_for(section: Section, version: Optional[VersionInfo]) -> Optional[str]:
        if version is None:
            return None
        label = fold(section.label)
        for module in version.modules:
            m = fold(module)
            if m and (m == label or m in label):
                return module
        return None

    # ------------------------------------------------------------------ #
    #  Chunk construction                                                  #
    # ------------------------------------------------------------------ #

    def _curated_to_chunks(self, page: Page, curation: PageCuration) -> list[Chunk]:
        chunks: list[Chunk] = []
        for result in curation.sections:
            for item in result.items:
                module = self._module_for(item.section, item.version)
                chunks.append(Chunk(
                    chunk_id=item.chunk_id,
                    page_url=page.url,
                    section=item.section,
                    text=item.content,
                    kind="curated",
                    index=item.ordinal,
                    heading_path=list(item.heading_path),
                    version=item.version,
                    module=module,
                    merged_modules=[module] if module else [],
                    enrichment=ChunkEnrichment(
                        id=item.chunk_id,
                        context=item.context,
                        questions=list(item.questions),
                        keywords=list(item.keywords),
                        quality=item.quality,
                        issues=list(item.issues),
                        outdated=item.outdated,
                        outdated_reason=item.outdated_reason,
                    ),
                    curated=True,
                    curation_status=result.status,
                    curation_issues=list(dict.fromkeys(result.validation_issues + item.issues)),
                    source_sections=list(item.source_sections),
                    evidence=list(item.evidence),
                    rewritten=item.rewritten,
                ))
        return chunks

    def _deterministic_chunks(
        self,
        page: Page,
        sections: list[Section],
        versions: dict[str, Optional[VersionInfo]],
    ) -> list[Chunk]:
        s = self.settings
        page_key = sha1(page.url)[:10]
        chunks: list[Chunk] = []
        for section in sections:
            version = versions.get(section.section_id)
            module = self._module_for(section, version)
            for raw in chunk_section(
                section, self.counter, s.CHUNK_MAX_TOKENS, s.CHUNK_OVERLAP_TOKENS,
                s.MIN_CHUNK_CHARS, s.CHUNK_TABLE_MAX_TOKENS,
            ):
                chunks.append(Chunk(
                    chunk_id=f"{page_key}-{sha1(section.dom_path)[:8]}-{raw.index:03d}",
                    page_url=page.url, section=section, text=raw.text, kind=raw.kind,
                    index=raw.index, heading_path=raw.heading_path, version=version,
                    module=module, merged_modules=[module] if module else [],
                    curation_status="deterministic",
                ))
        return chunks

    def _dedupe_chunks(
        self,
        page: Page,
        chunks: list[Chunk],
        result: PageResult,
        owners: dict[str, tuple[str, str]],
        also_on_updates: list[tuple[str, str]],
    ) -> list[Chunk]:
        """Collapse identical text (same version scope; cross-page alike)."""
        merged: dict[str, Chunk] = {}
        kept: list[Chunk] = []
        for chunk in chunks:
            folded = fold(chunk.text)
            scope = chunk.version.key if chunk.version else f"page:{page.url}"
            dedup_key = sha1(f"{scope}|{folded}")[:16]
            chunk.metadata["dedup_key"] = dedup_key
            if dedup_key in merged:
                canonical = merged[dedup_key]
                for module in chunk.merged_modules:
                    if module and module not in canonical.merged_modules:
                        canonical.merged_modules.append(module)
                for sid in chunk.source_sections:
                    if sid not in canonical.source_sections:
                        canonical.source_sections.append(sid)
                for evidence in chunk.evidence:
                    if evidence not in canonical.evidence:
                        canonical.evidence.append(evidence)
                chunk.dropped_reason = "duplicate"
                result.dropped.append(chunk)
                continue
            if chunk.version is None:
                owner = owners.get(sha1(folded))
                if owner is not None and owner[1] != page.url:
                    also_on_updates.append((owner[0], page.display_url))
                    chunk.dropped_reason = "duplicate-other-page"
                    result.dropped.append(chunk)
                    continue
                owners[sha1(folded)] = (chunk.chunk_id, page.url)
            merged[dedup_key] = chunk
            kept.append(chunk)
        return kept

    # ------------------------------------------------------------------ #
    #  Final text / metadata                                               #
    # ------------------------------------------------------------------ #

    def _embed_text(self, header: str, context: str, questions: list[str], text: str) -> tuple[str, bool]:
        """Deterministic-mode embed text (kept for --no-llm compatibility)."""
        budget = self.settings.EMBED_MAX_TOKENS
        if not self.settings.EMBED_QUESTIONS:
            questions = []
        variants = [(context, questions), (context, []), ("", [])]
        candidate = ""
        for ctx, qs in variants:
            lines = [header]
            if ctx:
                lines.append(ctx)
            if qs:
                lines.append("Pitanja: " + " ".join(qs))
            lines.append(text)
            candidate = "\n".join(lines)
            if self.counter.count_model_input("passage: " + candidate) <= budget:
                return candidate, (ctx, qs) != variants[0]
        return candidate, True

    def _fit_embed(
        self, header: str, context: str, questions: list[str], text: str
    ) -> Optional[tuple[str, str, list[str]]]:
        budget = self.settings.EMBED_MAX_TOKENS
        if not self.settings.EMBED_QUESTIONS:
            questions = []
        for ctx, qs in ((context, questions), (context, []), ("", [])):
            lines = [header]
            if ctx:
                lines.append(ctx)
            if qs:
                lines.append("Pitanja: " + " ".join(qs))
            lines.append(text)
            candidate = "\n".join(lines)
            if self.counter.count_model_input("passage: " + candidate) <= budget:
                return candidate, ctx, qs
        return None

    def _curated_embed_parts(
        self, header: str, context: str, questions: list[str], text: str
    ) -> list[tuple[str, str, str, list[str]]]:
        """(text, embed_text, context, questions) — never truncates content."""
        fit = self._fit_embed(header, context, questions, text)
        if fit is not None:
            embed, ctx, qs = fit
            return [(text, embed, ctx, qs)]

        header_tokens = self.counter.count_model_input("passage: " + header) + 4
        budget = max(32, self.settings.EMBED_MAX_TOKENS - header_tokens)
        units: list[tuple[str, bool]] = []
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            units.extend((sentence, i == 0) for i, sentence in enumerate(split_units(line)) if sentence)
        pieces = pack_units(units, self.counter, budget, 0) or [text]

        out: list[tuple[str, str, str, list[str]]] = []
        for piece in pieces:
            piece_fit = self._fit_embed(header, context, questions, piece)
            if piece_fit is None:
                piece_fit = self._fit_embed(header, "", [], piece)
            if piece_fit is None:
                raise ValueError(
                    f"curated chunk does not fit the embedding budget even after splitting: {piece[:80]!r}"
                )
            embed, ctx, qs = piece_fit
            out.append((piece, embed, ctx, qs))
        return out

    def _metadata(
        self,
        page: Page,
        analysis: PageAnalysis,
        chunk: Chunk,
        path: list[str],
        context: str,
        questions: list[str],
        truncated: bool,
    ) -> dict:
        section = chunk.section
        version = chunk.version
        enrichment = chunk.enrichment
        module = ", ".join(chunk.merged_modules) if chunk.merged_modules else chunk.module
        scope = version.key if version else f"page:{page.url}"

        source = (
            f"{page.url}#page={section.page}" if page.doc_type == "pdf" and section.page
            else f"{page.url}#{section.section_id}"
        )
        return {
            # ── fields the chatbot already uses ──────────────────────── #
            "source": source,
            "filename": " › ".join(path),
            "type": page.doc_type,
            "page": section.page,
            "url": self._url_for(page, section),
            # ── location ─────────────────────────────────────────────── #
            "page_url": page.url,
            "page_title": page.title,
            "breadcrumb": " › ".join(path),
            "section_id": section.section_id,
            "hidden_by_default": section.hidden_by_default,
            "ui_hidden": section.ui_hidden,
            # ── version ──────────────────────────────────────────────── #
            "program": version.program if version else chunk.metadata.get("program"),
            "level": version.level if version else chunk.metadata.get("level"),
            "module": module,
            "version_label": version.version_label if version else None,
            "accreditation_year": version.accreditation_year if version else None,
            # Always present: version-scoped chunks get the registry status,
            # everything else (FAQ, intro, news, version maps) is "unknown".
            "status": version.status if version else (chunk.metadata.get("status") or "unknown"),
            "valid_for": (version.valid_for or None) if version else None,
            "version_source": version.source if version else "none",
            # ── content ──────────────────────────────────────────────── #
            "doc_kind": "version_map" if chunk.kind == "version_map" else "chunk",
            "block_type": chunk.kind,
            "generated": chunk.generated,
            "page_type": analysis.page_type,
            "published_date": analysis.published_date,
            "page_summary": truncate(analysis.page_summary, 300) or None,
            "context": context or None,
            "questions": questions,
            "keywords": enrichment.keywords if enrichment else [],
            "quality": enrichment.quality if enrichment else None,
            "issues": enrichment.issues if enrichment else [],
            "outdated": bool(enrichment and enrichment.outdated),
            "outdated_reason": (enrichment.outdated_reason or None) if enrichment else None,
            # ── AI curation ──────────────────────────────────────────── #
            "curated": chunk.curated,
            "curation_status": chunk.curation_status or (
                "deterministic" if not chunk.curated else "processed"
            ),
            "curation_version": CURATION_SCHEMA_VERSION if chunk.curated else None,
            "curation_issues": chunk.curation_issues,
            "source_sections": chunk.source_sections,
            "source_evidence": [
                f"{e.get('source_section_id', '')}: {e.get('source_excerpt', '')}"
                for e in chunk.evidence[:3]
            ],
            "rewritten": chunk.rewritten,
            # ── bookkeeping ──────────────────────────────────────────── #
            "dedup_key": sha1(f"{scope}|{fold(chunk.text)}")[:16],
            "content_hash": sha1(chunk.text)[:16],
            "also_on": chunk.also_on,
            "embed_truncated": truncated,
            "ingested_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "prompt_version": PROMPT_VERSION,
            "llm_model": self.client.model if self.client else "none",
        }

    def _finalize(self, page: Page, analysis: PageAnalysis, chunk: Chunk) -> list[Chunk]:
        section = chunk.section
        version = chunk.version
        enrichment = chunk.enrichment

        if chunk.kind == "version_map":
            path = [page.title, "Pregled verzija studijskih programa"]
            header = (
                f"[{page.title} › Pregled verzija studijskog programa | "
                f"automatski generisano iz registra programa — proveriti na izvoru]"
            )
        else:
            path = location_path(section.breadcrumb, chunk.heading_path)
            if len(chunk.merged_modules) > 1 and section.label in path:
                path[path.index(section.label)] = " / ".join(chunk.merged_modules)
            extras = [version.valid_for if version else ""]
            if analysis.published_date and analysis.page_type in DATED_PAGE_TYPES:
                extras.append(f"objavljeno {analysis.published_date}")
            if enrichment is not None and enrichment.outdated:
                extras.append(f"⚠ moguće zastarelo: {enrichment.outdated_reason or 'proveriti'}")
            header = build_header(
                path,
                version.version_label if version else "",
                version.status_label if version else "",
                "; ".join(x for x in extras if x),
            )

        context = enrichment.context if enrichment else ""
        questions = enrichment.questions if enrichment else []
        embed_questions = list(questions)
        if chunk.kind in ("table", "curated"):
            seen = {fold(q) for q in embed_questions}
            embed_questions = [q for q in course_questions(chunk.text) if fold(q) not in seen] + embed_questions

        chunk.header = header

        if not chunk.curated:
            chunk.document = f"{header}\n{chunk.text}"
            chunk.embed_text, truncated = self._embed_text(header, context, embed_questions, chunk.text)
            chunk.metadata = self._metadata(
                page, analysis, chunk, path, context, questions, truncated
            )
            return [chunk]

        parts = self._curated_embed_parts(header, context, embed_questions, chunk.text)
        if len(parts) == 1:
            piece_text, embed_text, _used_context, _used_questions = parts[0]
            chunk.text = piece_text
            chunk.document = f"{header}\n{piece_text}"
            chunk.embed_text = embed_text
            # metadata keeps the full model labels even when they did not fit
            # into the embedding window; the document never contains them.
            chunk.metadata = self._metadata(
                page, analysis, chunk, path, context, questions, False
            )
            return [chunk]

        logger.info(
            "  chunk %s split into %d pieces for the embedding budget", chunk.chunk_id, len(parts)
        )
        out: list[Chunk] = []
        for index, (piece_text, embed_text, _used_context, _used_questions) in enumerate(parts, start=1):
            piece = replace(
                chunk,
                chunk_id=f"{chunk.chunk_id}-e{index}",
                index=chunk.index * 100 + index,
                text=piece_text,
                document=f"{header}\n{piece_text}",
                embed_text=embed_text,
                metadata={},
            )
            piece.metadata = self._metadata(
                page, analysis, piece, path, context, questions, False
            )
            piece.metadata["embed_piece"] = index
            out.append(piece)
        return out

    def _url_for(self, page: Page, section: Section) -> str:
        if page.doc_type == "pdf" and section.page:
            return f"{page.display_url}#page={section.page}"
        if page.doc_type == "web" and section.top_trigger in page.tab_params:
            return with_param(page.display_url, page.tab_params[section.top_trigger])
        return page.display_url

    # ------------------------------------------------------------------ #
    #  One page                                                            #
    # ------------------------------------------------------------------ #

    def _process_page(
        self,
        page: Page,
        report: PageReport,
        owners: dict[str, tuple[str, str]],
        also_on_updates: list[tuple[str, str]],
    ) -> PageResult:
        s = self.settings
        result = PageResult()

        # 1) sections a visitor can actually reach
        sections: list[Section] = []
        for section in page.sections:
            if not section.blocks:
                continue
            if not section.reachable and not self.include_unreachable:
                report.unreachable.append((section.section_id, " › ".join(section.breadcrumb), len(section.text)))
                continue
            sections.append(section)

        # 2) page analysis (+ drafts for program panels not in programs.yaml)
        unmapped = [
            sec.section_id for sec in sections
            if sec.panel_chain and self.registry.resolve(page.url, sec.panel_chain) is None
        ]
        analysis = self.analyzer.analyze_page(page, self.registry, unmapped)
        result.analysis = analysis
        report.page_type = analysis.page_type
        report.summary = analysis.page_summary
        report.published_date = analysis.published_date

        page_index = TextIndex(page.full_text)
        drafted: dict[str, object] = {}
        for draft in analysis.registry_draft:
            ok, problems = check_draft(draft, page_index)
            record = {"page": page.url, **draft.model_dump(), "evidence_ok": ok,
                      "problems": problems, "drafted_at": self.today.isoformat()}
            report.drafts.append(record)
            result.drafts.append(record)
            if ok:
                for panel in draft.panels:
                    drafted[panel] = draft
        pool = self.registry.entries_for_page(page.url) + list({id(d): d for d in drafted.values()}.values())

        def version_for(section: Section) -> Optional[VersionInfo]:
            entry = self.registry.resolve(page.url, section.panel_chain)
            if entry is not None:
                return self.registry.version_info(entry)
            for panel in reversed(section.panel_chain):
                if panel in drafted:
                    return self.registry.version_info(drafted[panel], "llm", pool, page.url)
            return None

        versions = {section.section_id: version_for(section) for section in sections}

        # 3) chunks — AI curation, or deterministic chunks with --no-llm
        if self.curator is not None:
            curation = self.curator.curate_page(page, analysis, sections, versions)
            result.curation = curation
            result.complete = curation.complete
            result.incomplete_reason = "; ".join(
                f"{sid}: {reason}" for sid, _path, reason in curation.failed_sections
            ) or ("" if curation.complete else "curation incomplete")
            report.curation_status = "complete" if curation.complete else "partial"
            report.curation = curation.stats
            report.discarded = curation.discarded
            report.discard_samples = curation.discard_samples
            report.rejected = curation.rejected
            report.contradictions = curation.contradictions
            report.missing_context = curation.missing_context
            report.failed_sections = curation.failed_sections
            chunks = self._curated_to_chunks(page, curation)
            for chunk in chunks:
                for issue in chunk.curation_issues:
                    report.issues.append((
                        chunk.chunk_id,
                        " › ".join(location_path(chunk.section.breadcrumb, chunk.heading_path)),
                        issue,
                    ))
        else:
            chunks = self._deterministic_chunks(page, sections, versions)
            report.curation_status = "deterministic"
            report.curation = {
                "sections_total": len(sections),
                "sections_processed": len(sections),
                "sections_failed": 0,
                "parts_total": len(sections),
                "chunks_curated": len(chunks),
                "chunks_rejected": 0,
                "chunks_deduplicated": 0,
                "chunks_rewritten": 0,
                "chunks_split": 0,
                "requests": 0,
                "source_chars": sum(len(section.text) for section in sections),
                "curated_chars": sum(len(chunk.text) for chunk in chunks),
                "source_tokens": sum(self.counter.count(section.text) for section in sections),
                "curated_tokens": sum(self.counter.count(chunk.text) for chunk in chunks),
            }

        # 4) cross-page / within-page duplicate collapse
        chunks = self._dedupe_chunks(page, chunks, result, owners, also_on_updates)

        # 5) quality filter
        kept: list[Chunk] = []
        for chunk in chunks:
            enrichment = chunk.enrichment
            if enrichment is not None:
                path = " › ".join(location_path(chunk.section.breadcrumb, chunk.heading_path))
                if not chunk.curated:       # curated issues were added above
                    report.issues += [(chunk.chunk_id, path, i) for i in enrichment.issues]
                if enrichment.outdated:
                    report.issues.append((
                        chunk.chunk_id, path,
                        f"ZASTARELO (skriva se u chatbotu): {enrichment.outdated_reason}",
                    ))
                if enrichment.quality in DROP_QUALITIES and not self.keep_low_value:
                    chunk.dropped_reason = enrichment.quality
                    result.dropped.append(chunk)
                    continue
            kept.append(chunk)

        # 6) version maps — the verified overview chunks
        page_key = sha1(page.url)[:10]
        for vm in self.registry.version_maps(page.url, page_index):
            report.version_maps.append(vm)
            section = Section(
                section_id=f"vmap-{slugify(vm['program'])}-{vm['level']}",
                dom_path=f"vmap/{slugify(vm['program'])}/{vm['level']}",
                label="Pregled verzija", breadcrumb=[page.title, "Pregled verzija"],
            )
            kept.append(Chunk(
                chunk_id=f"{page_key}-vmap-{slugify(vm['program'], 20)}-{vm['level']}",
                page_url=page.url, section=section, text=vm["text"], kind="version_map",
                index=0, generated=True,
                metadata={"program": vm["program"], "level": vm["level"],
                          "dedup_key": sha1(f"vmap|{page.url}|{vm['title']}")[:16]},
            ))

        # 7) header / document / embedded text / metadata
        finalized: list[Chunk] = []
        for chunk in kept:
            finalized.extend(self._finalize(page, analysis, chunk))
        result.embed_split = len(finalized) - len(kept)
        report.curation["chunks_final"] = len(finalized)
        report.curation["chunks_final_tokens"] = sum(
            self.counter.count(chunk.text) for chunk in finalized
        )
        if self.curator is None:
            duplicates = sum(
                1 for chunk in result.dropped
                if (chunk.dropped_reason or "").startswith("duplicate")
            )
            report.curation.update({
                "chunks_curated": len(finalized),
                "chunks_deduplicated": duplicates,
                "curated_chars": sum(len(chunk.text) for chunk in finalized),
                "curated_tokens": report.curation["chunks_final_tokens"],
            })
        if result.curation is not None:
            page_status = "complete" if result.curation.complete else "partial"
            for chunk in finalized:
                chunk.metadata["page_curation_status"] = page_status
        for chunk in result.dropped:
            chunk.header = build_header(location_path(chunk.section.breadcrumb, chunk.heading_path))

        # 8) report rows — counts include embedding splits
        per_section: dict[str, int] = {}
        for chunk in finalized:
            per_section[chunk.section.section_id] = per_section.get(chunk.section.section_id, 0) + 1
        for section in sections:
            version = versions.get(section.section_id)
            report.sections.append(SectionRow(
                section_id=section.section_id,
                path=" › ".join(section.breadcrumb[1:]) or section.label,
                version=version.version_label + (" (draft)" if version.source == "llm" else "") if version else "",
                status=version.status if version else "",
                hidden=section.hidden_by_default,
                reachable=section.reachable,
                chunks=per_section.get(section.section_id, 0),
                url=self._url_for(page, section),
            ))
        seen_sections: set[str] = set()
        for chunk in finalized:
            if chunk.section.section_id in seen_sections or len(report.previews) >= 40:
                continue
            seen_sections.add(chunk.section.section_id)
            report.previews.append({
                "id": chunk.chunk_id, "header": chunk.header, "text": chunk.text,
                "questions": chunk.enrichment.questions if chunk.enrichment else [],
            })
        for chunk in result.dropped:
            report.dropped[chunk.dropped_reason or "dropped"] += 1

        result.chunks = finalized
        return result

    # ------------------------------------------------------------------ #
    #  Run                                                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _record(chunk: Chunk, kept: bool, committed: bool = False) -> dict:
        return {
            "id": chunk.chunk_id,
            "page_url": chunk.page_url,
            "kept": kept,
            "committed": committed,
            "dropped_reason": chunk.dropped_reason,
            "header": chunk.header,
            "document": chunk.document or chunk.text,
            "embed_text": chunk.embed_text,
            "metadata": chunk.metadata if kept else {},
            "curated": chunk.curated,
            "curation_status": chunk.curation_status or None,
            "curation_issues": chunk.curation_issues,
            "source_sections": chunk.source_sections,
            "evidence": chunk.evidence,
            "rewritten": chunk.rewritten,
        }

    @staticmethod
    def _usage_snapshot(client) -> tuple[int, int]:
        usage = getattr(client, "usage", None) if client is not None else None
        if usage is None:
            return 0, 0
        return int(getattr(usage, "calls", 0)), int(getattr(usage, "cached_calls", 0))

    def _validate_replacement(self, page: Page, chunks: list[Chunk], embeddings: list) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunk/embedding count mismatch")
        ids = [chunk.chunk_id for chunk in chunks]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate chunk ids")
        for chunk in chunks:
            if not chunk.document.strip():
                raise ValueError(f"empty document for {chunk.chunk_id}")
            if not chunk.metadata or chunk.metadata.get("page_url") != page.url:
                raise ValueError(f"metadata missing/foreign for {chunk.chunk_id}")
            if self.counter.count_model_input("passage: " + chunk.embed_text) > self.settings.EMBED_MAX_TOKENS:
                raise ValueError(f"embedding input over budget for {chunk.chunk_id}")

    def _curation_record(self, page: Page, page_report: PageReport) -> dict:
        return {
            "page_url": page.url,
            "title": page.title,
            "outcome": page_report.outcome,
            "curation_status": page_report.curation_status,
            "stats": page_report.curation,
            "failed_sections": [
                {"section_id": sid, "path": path, "reason": reason}
                for sid, path, reason in page_report.failed_sections
            ],
            "rejected": page_report.rejected[:300],
            "discarded": dict(page_report.discarded),
            "discard_samples": page_report.discard_samples,
            "contradictions": page_report.contradictions,
            "missing_context": page_report.missing_context,
            "llm_calls": page_report.llm_calls,
            "llm_cached_calls": page_report.llm_cached_calls,
        }

    def run(self) -> tuple[RunReport, Path]:
        links = parse_links_file(self.links_file)
        frontier = Frontier(links, self.settings)
        llm_line = (
            f"{self.client.model} ({'thinking' if self.client.thinking else 'non-thinking'})"
            if self.client else "disabled (--no-llm)"
        )
        report = RunReport(
            links_file=str(self.links_file), collection=self.collection,
            db_path=str(self.db_path), dry_run=self.dry_run, llm=llm_line,
        )
        content_owner: dict[str, str] = {}
        chunk_owners: dict[str, tuple[str, str]] = {}
        also_on_updates: list[tuple[str, str]] = []
        records: list[dict] = []
        analyses: dict[str, dict] = {}
        curation_records: list[dict] = []
        drafts: list[dict] = []

        for item in frontier:
            page_report = PageReport(target=item.target)
            report.pages.append(page_report)
            logger.info("→ %s", item.target)
            try:
                page = self._load(item)
            except Exception as exc:
                page_report.outcome, page_report.error = "failed", str(exc)
                logger.error("  failed: %s", exc)
                continue

            page_report.url, page_report.title = page.url, page.title
            page_report.doc_type, page_report.flags = page.doc_type, list(page.flags)
            frontier.add_children(item, page)

            text_hash = sha1(page.full_text)
            if not page.full_text.strip():
                page_report.outcome = "empty"
                continue
            owner = content_owner.get(text_hash)
            if owner and owner != page.url:
                page_report.outcome, page_report.alias_of = "alias", owner
                continue
            content_owner[text_hash] = page.url

            curation_recorded = False
            try:
                self._probe_tabs(page)
                labels = {s.section_id: s.label for s in page.sections}
                page_report.tab_params = {
                    labels.get(page.top_triggers.get(trigger, ""), trigger): param
                    for trigger, param in page.tab_params.items()
                }
                signature = self._signature(page, text_hash)
                if not self.force and not self.dry_run and self.state.is_unchanged(
                    self.collection, page.url, signature
                ):
                    page_report.outcome = "skipped"
                    logger.info("  unchanged — skipped")
                    continue

                calls_before, cached_before = self._usage_snapshot(self.client)
                result = self._process_page(page, page_report, chunk_owners, also_on_updates)
                calls_after, cached_after = self._usage_snapshot(self.client)
                page_report.llm_calls = max(0, calls_after - calls_before)
                page_report.llm_cached_calls = max(0, cached_after - cached_before)

                analyses[page.url] = result.analysis.model_dump() if result.analysis else {}
                drafts += result.drafts

                if self.dry_run:
                    page_report.outcome = "dry-run"
                    page_report.stored = len(result.chunks)
                elif not result.complete and not self.allow_partial_write:
                    page_report.outcome = "partial"
                    page_report.error = (
                        result.incomplete_reason or "curation incomplete; existing data kept"
                    )
                    page_report.stored, page_report.deleted = 0, 0
                    # Invalidate any older "complete" signature: the page must
                    # be retried, otherwise a later run would skip it.
                    self.state.forget(self.collection, page.url)
                    logger.warning("  partial curation — not committed: %s", page_report.error)
                elif not result.chunks and page.full_text.strip():
                    page_report.outcome = "failed"
                    page_report.error = "curation produced no chunks; existing data kept"
                    self.state.forget(self.collection, page.url)
                    logger.warning("  no chunks produced — not committed")
                else:
                    started = time.perf_counter()
                    embeddings = self.embedder.embed_passages([c.embed_text for c in result.chunks])
                    page_report.embed_seconds = time.perf_counter() - started
                    self._validate_replacement(page, result.chunks, embeddings)
                    started = time.perf_counter()
                    stored, deleted = self.store.replace_page(
                        page.url,
                        [c.chunk_id for c in result.chunks],
                        embeddings,
                        [c.document for c in result.chunks],
                        [c.metadata for c in result.chunks],
                        delete_stale=result.complete,
                    )
                    page_report.write_seconds = time.perf_counter() - started
                    page_report.stored, page_report.deleted = stored, deleted
                    if result.complete:
                        page_report.outcome = "ok"
                        self.state.record(
                            self.collection, page.url, signature=signature, title=page.title,
                            chunks=stored, parent=item.parent,
                        )
                    else:
                        page_report.outcome = "partial"
                        page_report.error = result.incomplete_reason or "partial write requested"
                        self.state.forget(self.collection, page.url)
                committed = page_report.outcome in ("ok", "partial") and not self.dry_run
                records += [self._record(c, True, committed=committed) for c in result.chunks]
                records += [self._record(c, False, committed=False) for c in result.dropped]
                curation_records.append(self._curation_record(page, page_report))
                curation_recorded = True
                logger.info("  %d chunks (%s)", page_report.stored, page_report.outcome)
            except Exception as exc:
                logger.exception("  failed while processing %s", page.url)
                page_report.outcome, page_report.error = "failed", f"{type(exc).__name__}: {exc}"
                if page_report.curation_status and not curation_recorded:
                    curation_records.append(self._curation_record(page, page_report))

        if not self.dry_run:
            for chunk_id, url in also_on_updates:
                self.store.add_also_on(chunk_id, url)
            if any(p.outcome in ("ok", "partial") for p in report.pages):
                self.store.write_shared_config({
                    "embedding_model": self.settings.EMBEDDING_MODEL,
                    "passage_prefix": "passage: ",
                    "query_prefix": "query: ",
                    "distance": "cosine",
                    "collection": self.collection,
                    "prompt_version": PROMPT_VERSION,
                    "curation_schema_version": CURATION_SCHEMA_VERSION,
                    "written_by": "imi-scraper",
                })
        self.state.save()

        if drafts:
            draft_path = self.settings.path(self.settings.PROGRAMS_DRAFT_FILE)
            write_drafts(draft_path, drafts)
            report.drafts_file = str(draft_path)

        if self.client is not None:
            u = self.client.usage
            hit_rate = (u.cache_hit_tokens / u.prompt_tokens * 100) if u.prompt_tokens else 0
            report.usage_line = (
                f"DeepSeek usage: {u.calls} calls ({u.cached_calls} answered from local cache, "
                f"{u.failed_calls} failed), {u.prompt_tokens:,} prompt tokens ({hit_rate:.0f}% "
                f"server cache hits), {u.completion_tokens:,} completion tokens, "
                f"≈ ${u.cost_usd(self.settings):.3f} at peak prices; models: {', '.join(sorted(u.models)) or '-'}"
            )

        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = report.write(
            self.data_dir / "reports" / stamp, records, analyses, curation_records
        )
        return report, path
