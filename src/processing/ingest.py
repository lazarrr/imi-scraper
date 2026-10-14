"""
ingest.py
~~~~~~~~~
Pipeline orchestration:

  links file → fetch → parse (sections, breadcrumbs, reachability)
    → versions (programs.yaml; DeepSeek drafts for unmapped panels)
    → DeepSeek page analysis → chunk + merge duplicates
    → DeepSeek chunk enrichment + quality filter → version-map chunks
    → header / document / embedded text / metadata
    → E5 embeddings → Chroma (upsert, then delete stale ids of the page)
    → report.md + chunks.jsonl + state
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

from src.core.deepseek_client import DeepSeekClient
from src.processing.chunker import TokenCounter, build_header, chunk_section, location_path
from src.processing.crawler import CrawlItem, Frontier
from src.processing.documents import DocxParser, PdfParser
from src.processing.enricher import Enricher
from src.processing.fetcher import Fetcher, canonicalize_url, display_url, with_param
from src.processing.html_parser import parse_html
from src.processing.links import parse_links_file
from src.processing.prompts import PROMPT_VERSION
from src.processing.registry import Registry, check_draft, write_drafts
from src.processing.report import PageReport, RunReport, SectionRow
from src.processing.schemas import Chunk, Page, PageAnalysis, Section, VersionInfo
from src.processing.site_profiles import profile_for
from src.processing.state import IngestState
from src.utils.text import TextIndex, fold, sha1, slugify, truncate

logger = logging.getLogger(__name__)

# Bump when parsing / chunking / embedded-text logic changes, so pages that
# were ingested by older code are re-processed instead of skipped.
PIPELINE_VERSION = "2026-10-11.2"

DROP_QUALITIES = ("boilerplate", "low_value")
DATED_PAGE_TYPES = ("vest", "oglas", "dogadjaj")

# "Predmet: Osnovi programiranja |"  /  "Izborni predmeti grupe G1: Engleski jezik A1 |"
_COURSE_RE = re.compile(r"(?:^|\| |— )(?:Predmet|Izborni predmeti?[^:|—\n]*): ([^|\n]+?)\s*(?=\||$)", re.M)
_PLACEHOLDER_RE = re.compile(r"^(izborni predmet\b|zbir|ukupno)", re.I)


def course_questions(chunk_text: str, limit: int = 2) -> list[str]:
    """Deterministic questions for course-table rows, embedded with the row
    (never shown to the chatbot) so "Koliko ESPB ima X?" hits X's row —
    works with or without DeepSeek."""
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
        self.enricher = Enricher(self.client, settings, self.today, self.registry.all_version_maps())

        self.counter = token_counter or TokenCounter(settings.EMBEDDING_MODEL)
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
            "llm" if self.client else "no-llm", page.tab_params,
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
    #  Assembly                                                            #
    # ------------------------------------------------------------------ #

    def _embed_text(self, header: str, context: str, questions: list[str], text: str) -> tuple[str, bool]:
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

    def _url_for(self, page: Page, section: Section) -> str:
        if page.doc_type == "pdf" and section.page:
            return f"{page.display_url}#page={section.page}"
        if page.doc_type == "web" and section.top_trigger in page.tab_params:
            return with_param(page.display_url, page.tab_params[section.top_trigger])
        return page.display_url

    def _finalize(self, page: Page, analysis: PageAnalysis, chunk: Chunk) -> None:
        section = chunk.section
        version = chunk.version
        enrichment = chunk.enrichment

        if chunk.kind == "version_map":
            path = [page.title, "Pregled verzija studijskih programa"]
            header = f"[{page.title} › Pregled verzija studijskog programa | automatski generisano iz registra programa — proveriti na izvoru]"
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
        if chunk.kind == "table":
            seen = {fold(q) for q in embed_questions}
            embed_questions = [q for q in course_questions(chunk.text) if fold(q) not in seen] + embed_questions
        chunk.header = header
        chunk.document = f"{header}\n{chunk.text}"
        chunk.embed_text, truncated = self._embed_text(header, context, embed_questions, chunk.text)

        source = (
            f"{page.url}#page={section.page}" if page.doc_type == "pdf" and section.page
            else f"{page.url}#{section.section_id}"
        )
        module = ", ".join(chunk.merged_modules) if chunk.merged_modules else chunk.module
        meta = {
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
            "status": version.status if version else None,
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
            # ── bookkeeping ──────────────────────────────────────────── #
            "dedup_key": chunk.metadata.get("dedup_key"),
            "content_hash": sha1(chunk.text)[:16],
            "also_on": chunk.also_on,
            "embed_truncated": truncated,
            "ingested_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "prompt_version": PROMPT_VERSION,
            "llm_model": self.client.model if self.client else "none",
        }
        chunk.metadata = meta

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
        analysis = self.enricher.analyze_page(page, self.registry, unmapped)
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

        # 3) chunk + merge identical text within one version
        merged: dict[str, Chunk] = {}
        chunks: list[Chunk] = []
        per_section: dict[str, int] = {}
        page_key = sha1(page.url)[:10]
        for section in sections:
            version = version_for(section)
            module = self._module_for(section, version)
            for raw in chunk_section(section, self.counter, s.CHUNK_MAX_TOKENS,
                                     s.CHUNK_OVERLAP_TOKENS, s.MIN_CHUNK_CHARS,
                                     s.CHUNK_TABLE_MAX_TOKENS):
                folded = fold(raw.text)
                scope = version.key if version else f"page:{page.url}"
                dedup_key = sha1(f"{scope}|{folded}")[:16]
                chunk = Chunk(
                    chunk_id=f"{page_key}-{sha1(section.dom_path)[:8]}-{raw.index:03d}",
                    page_url=page.url, section=section, text=raw.text, kind=raw.kind,
                    index=raw.index, heading_path=raw.heading_path, version=version,
                    module=module, merged_modules=[module] if module else [],
                    metadata={"dedup_key": dedup_key},
                )
                if dedup_key in merged:
                    kept = merged[dedup_key]
                    if module and module not in kept.merged_modules:
                        kept.merged_modules.append(module)
                    chunk.dropped_reason = "duplicate"
                    result.dropped.append(chunk)
                    continue
                if version is None:
                    owner = owners.get(sha1(folded))
                    if owner is not None and owner[1] != page.url:
                        also_on_updates.append((owner[0], page.display_url))
                        chunk.dropped_reason = "duplicate-other-page"
                        result.dropped.append(chunk)
                        continue
                    owners[sha1(folded)] = (chunk.chunk_id, page.url)
                merged[dedup_key] = chunk
                chunks.append(chunk)
                per_section[section.section_id] = per_section.get(section.section_id, 0) + 1

        # 4) DeepSeek enrichment + quality filter
        enrichment = self.enricher.enrich(page, analysis, chunks)
        kept: list[Chunk] = []
        for chunk in chunks:
            chunk.enrichment = enrichment.get(chunk.chunk_id)
            if chunk.enrichment is not None:
                path = " › ".join(location_path(chunk.section.breadcrumb, chunk.heading_path))
                report.issues += [(chunk.chunk_id, path, i) for i in chunk.enrichment.issues]
                if chunk.enrichment.outdated:
                    report.issues.append((chunk.chunk_id, path,
                                          f"ZASTARELO (skriva se u chatbotu): {chunk.enrichment.outdated_reason}"))
                if chunk.enrichment.quality in DROP_QUALITIES and not self.keep_low_value:
                    chunk.dropped_reason = chunk.enrichment.quality
                    result.dropped.append(chunk)
                    per_section[chunk.section.section_id] -= 1
                    continue
            kept.append(chunk)

        # 5) version maps — the verified overview chunks
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

        # 6) header / document / embedded text / metadata
        for chunk in kept:
            self._finalize(page, analysis, chunk)
        for chunk in result.dropped:
            chunk.header = build_header(location_path(chunk.section.breadcrumb, chunk.heading_path))

        # 7) report rows
        for section in sections:
            version = version_for(section)
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
        for chunk in kept:
            if chunk.section.section_id in seen_sections or len(report.previews) >= 40:
                continue
            seen_sections.add(chunk.section.section_id)
            report.previews.append({
                "id": chunk.chunk_id, "header": chunk.header, "text": chunk.text,
                "questions": chunk.enrichment.questions if chunk.enrichment else [],
            })
        for chunk in result.dropped:
            report.dropped[chunk.dropped_reason or "dropped"] += 1

        result.chunks = kept
        return result

    # ------------------------------------------------------------------ #
    #  Run                                                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _record(chunk: Chunk, kept: bool) -> dict:
        return {
            "id": chunk.chunk_id,
            "page_url": chunk.page_url,
            "kept": kept,
            "dropped_reason": chunk.dropped_reason,
            "header": chunk.header,
            "document": chunk.document or chunk.text,
            "embed_text": chunk.embed_text,
            "metadata": chunk.metadata if kept else {},
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

                result = self._process_page(page, page_report, chunk_owners, also_on_updates)
                analyses[page.url] = result.analysis.model_dump() if result.analysis else {}
                drafts += result.drafts
                records += [self._record(c, True) for c in result.chunks]
                records += [self._record(c, False) for c in result.dropped]

                if self.dry_run:
                    page_report.outcome = "dry-run"
                    page_report.stored = len(result.chunks)
                else:
                    embeddings = self.embedder.embed_passages([c.embed_text for c in result.chunks])
                    stored, deleted = self.store.replace_page(
                        page.url,
                        [c.chunk_id for c in result.chunks],
                        embeddings,
                        [c.document for c in result.chunks],
                        [c.metadata for c in result.chunks],
                    )
                    page_report.stored, page_report.deleted = stored, deleted
                    self.state.record(
                        self.collection, page.url, signature=signature, title=page.title,
                        chunks=stored, parent=item.parent,
                    )
                logger.info("  %d chunks (%s)", page_report.stored, page_report.outcome)
            except Exception as exc:
                logger.exception("  failed while processing %s", page.url)
                page_report.outcome, page_report.error = "failed", f"{type(exc).__name__}: {exc}"

        if not self.dry_run:
            for chunk_id, url in also_on_updates:
                self.store.add_also_on(chunk_id, url)
            if any(p.outcome == "ok" for p in report.pages):
                self.store.write_shared_config({
                    "embedding_model": self.settings.EMBEDDING_MODEL,
                    "passage_prefix": "passage: ",
                    "query_prefix": "query: ",
                    "distance": "cosine",
                    "collection": self.collection,
                    "prompt_version": PROMPT_VERSION,
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
        path = report.write(self.data_dir / "reports" / stamp, records, analyses)
        return report, path
