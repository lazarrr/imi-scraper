"""
curator.py
~~~~~~~~~~
AI content curation — DeepSeek writes the final RAG chunks.

The old pipeline split first and asked the model only for labels.  This stage
inverts that: the parser's sections (with their full text, table rows,
breadcrumbs and registry version) go to DeepSeek, which decides what is
knowledge, rewrites where useful and returns self-contained chunks.  The
deterministic part of the pipeline keeps doing what it is good at:

  • section grouping / splitting under the request token budget — every
    section is processed, no silent truncation;
  • strict Pydantic validation of the reply;
  • evidence quotes checked against the extracted source;
  • numeric facts checked against the page corpus;
  • registry-derived version metadata (never the model's guess);
  • deterministic chunk ids and page-level duplicate reconciliation;
  • explicit partial failures — a failed section makes the page incomplete.

The curator returns `PageCuration`; `IngestPipeline` converts its items into
`Chunk` objects and owns storage, embeddings and reporting.
"""

from __future__ import annotations

import logging
import re
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Optional

from src.core.deepseek_client import DeepSeekClient, OutputTruncated
from src.processing import factcheck, prompts
from src.processing.chunker import TokenCounter, pack_units
from src.processing.registry import Registry
from src.processing.schemas import (
    CuratedChunk,
    CurationResponse,
    Page,
    PageAnalysis,
    ReconcileResponse,
    Section,
    SectionCuration,
    VersionInfo,
)
from src.utils.text import TextIndex, fold, normalize_text, sha1, truncate

logger = logging.getLogger(__name__)

# Request-level source budget is split into parts only when a single section
# exceeds it; grouping never mixes two parts of the same section in one call.
MIN_PART_TOKENS = 200
MAX_SPLIT_DEPTH = 4
MAX_EVIDENCE_QUOTE_CHARS = 400

# "3 smera", "tri izborna modula" — used for a deterministic outdated check.
_MODULE_COUNT_RE = re.compile(
    r"\b(\d{1,2})\s+(?:izborna\s+|obavezna\s+)?(?:modula|smera|modul|smer)\b", re.I
)
_MODULE_COUNT_WORD_RE = re.compile(
    r"\b(\w+)\s+(?:izborna\s+|obavezna\s+)?(?:modula|smera|modul|smer)\b", re.I
)
# Text that explicitly talks about an older program is not "outdated".
_LEGACY_CUE_RE = re.compile(
    r"stari\s+(?:program|programi|programu)|prethodni\s+program|"
    r"više se ne upisuje|akreditacij\w*\s+(?:19|20(?:0\d|1[0-8]))",
    re.I,
)


# --------------------------------------------------------------------------- #
#  Plumbing                                                                    #
# --------------------------------------------------------------------------- #

@dataclass
class SectionPayload:
    """One section (or one part of a long section) sent in one request."""

    section: Section
    version: Optional[VersionInfo]
    text: str
    part_label: str
    order: int
    block_types: str
    tokens: int

    def as_dict(self) -> dict:
        return {
            "section_id": self.section.section_id,
            "part_label": self.part_label,
            "breadcrumb": " › ".join(self.section.breadcrumb),
            "version_label": (
                f"{self.version.version_label} — {self.version.status_label}"
                + (f" ({self.version.valid_for})" if self.version.valid_for else "")
                if self.version
                else ""
            ),
            "versioned": self.version is not None,
            "block_types": self.block_types,
            "text": self.text,
        }


@dataclass
class PartOutcome:
    payload: SectionPayload
    section: Optional[SectionCuration]
    error: str = ""


@dataclass
class CuratedItem:
    """One validated, final knowledge chunk (not yet a store `Chunk`)."""

    chunk_id: str
    section: Section
    version: Optional[VersionInfo]
    content: str
    context: str
    questions: list[str]
    keywords: list[str]
    evidence: list[dict]
    source_sections: list[str]
    quality: str
    issues: list[str]
    outdated: bool
    outdated_reason: str
    order: int
    ordinal: int
    rewritten: bool
    heading_path: list[str] = field(default_factory=list)


@dataclass
class SectionCurationResult:
    section: Section
    version: Optional[VersionInfo]
    status: str = "processed"                  # processed | no_relevant_content | failed
    items: list[CuratedItem] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    discarded: list[dict] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)
    validation_issues: list[str] = field(default_factory=list)
    complete: bool = True
    parts: int = 1
    source_chars: int = 0
    source_tokens: int = 0
    split_chunks: int = 0
    validated_items: int = 0
    pending: list[tuple[SectionPayload, CuratedChunk]] = field(default_factory=list)


@dataclass
class PageCuration:
    sections: list[SectionCurationResult]
    complete: bool
    page_issues: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    missing_context: list[str] = field(default_factory=list)
    discarded: Counter = field(default_factory=Counter)
    discard_samples: list[dict] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)
    failed_sections: list[tuple[str, str, str]] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def items(self) -> list[CuratedItem]:
        out: list[CuratedItem] = []
        for section in self.sections:
            out.extend(section.items)
        return out


# --------------------------------------------------------------------------- #
#  Curator                                                                     #
# --------------------------------------------------------------------------- #

class Curator:
    def __init__(
        self,
        client: DeepSeekClient,
        settings,
        registry: Registry,
        counter: TokenCounter,
    ):
        self.client = client
        self.settings = settings
        self.registry = registry
        self.counter = counter
        self.facts = registry.all_version_maps()

        s = settings
        self.input_budget = max(1000, int(s.CURATION_INPUT_TOKENS))
        self.min_tokens = max(1, int(s.CURATION_CHUNK_MIN_TOKENS))
        self.target_tokens = max(self.min_tokens, int(s.CURATION_CHUNK_TARGET_TOKENS))
        # Leave room in E5's 512-token window for header/context/questions.
        self.chunk_max_tokens = max(
            self.min_tokens,
            min(int(s.CURATION_CHUNK_MAX_TOKENS), int(s.EMBED_MAX_TOKENS) - 180),
        )
        self.max_workers = max(
            1, min(int(s.CURATION_MAX_WORKERS), int(getattr(client, "max_workers", 4) or 1))
        )
        self.strict_facts = bool(s.CURATION_STRICT_FACTS)
        # A section this large gets its own request: grouping two of them in
        # one reply made the model omit one of the two (seen on the 2016/2018
        # informatics program tabs), because the expected output is too large.
        self.large_section_tokens = max(2000, int(s.CURATION_MAX_OUTPUT_TOKENS) // 2)
        self.system = prompts.curation_system(
            self.min_tokens, self.target_tokens, self.chunk_max_tokens
        )
        self._requests = 0
        self._retries = 0
        self._missing_retries = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    #  Public entry point                                                  #
    # ------------------------------------------------------------------ #

    def curate_page(
        self,
        page: Page,
        analysis: PageAnalysis,
        sections: list[Section],
        versions: dict[str, Optional[VersionInfo]],
    ) -> PageCuration:
        self._requests = 0
        self._retries = 0
        self._missing_retries = 0

        payloads = self._payloads(page, sections, versions)
        groups = self._group(payloads)
        facts = self._facts_for_page(page)
        current_versions = self._current_versions(page)
        logger.info(
            "  curation: %d sections / %d parts in %d request(s) for %s",
            len(sections), len(payloads), len(groups), page.url,
        )

        outcomes: list[PartOutcome] = []
        if groups:
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                for partial in pool.map(
                    lambda group: self._run_group(page, analysis, group, facts), groups
                ):
                    outcomes.extend(partial)

        results = self._merge(page, sections, versions, payloads, outcomes)
        page_corpus = self._page_corpus(page, analysis, sections, versions, facts)
        rejected: list[dict] = []
        for result in results:
            self._validate_result(
                result, payloads, page_corpus, page, rejected, current_versions
            )

        validated_total = sum(len(r.items) for r in results)
        deduped = self._dedupe(items=[i for r in results for i in r.items])
        contradictions: list[str] = []
        missing_context: list[str] = []
        page_issues: list[str] = []
        preliminary_complete = all(r.complete for r in results)
        if (
            self.settings.CURATION_RECONCILE
            and len(results) > 1          # cross-section check only
            and preliminary_complete      # a partial page will not be committed
        ):
            deduped, contradictions, missing_context, page_issues = self._reconcile(
                page, deduped, results
            )

        # Rebuild per-section item lists from the surviving, ordered items.
        kept_by_section: dict[str, list[CuratedItem]] = {}
        for item in deduped:
            kept_by_section.setdefault(item.section.section_id, []).append(item)
        for result in results:
            result.items = sorted(
                kept_by_section.get(result.section.section_id, []), key=lambda i: (i.order, i.ordinal)
            )

        for result in results:
            if result.status == "processed" and not result.items and not result.discarded:
                if result.validated_items:
                    # Legitimate: every chunk this section produced was merged
                    # into another section's canonical chunk by dedupe/reconcile.
                    result.complete = True
                    result.issues.append(
                        "all curated chunks were merged into another section's chunks (duplicate)"
                    )
                else:
                    result.status = "failed"
                    result.complete = False
                    result.issues.append("model returned no chunks and no discard explanation")

        failed = [
            (r.section.section_id, " › ".join(r.section.breadcrumb), "; ".join(r.issues) or "not processed")
            for r in results if not r.complete
        ]
        complete = not failed

        discarded = Counter()
        samples: list[dict] = []
        for result in results:
            for entry in result.discarded:
                discarded[entry.get("reason", "other")] += 1
                if len(samples) < 25:
                    samples.append({
                        "page_url": page.url,
                        "section_id": entry.get("source_section_id") or result.section.section_id,
                        "description": entry.get("description", ""),
                        "reason": entry.get("reason", "other"),
                    })

        stats = {
            "sections_total": len(results),
            "sections_processed": sum(1 for r in results if r.status == "processed"),
            "sections_no_relevant": sum(1 for r in results if r.status == "no_relevant_content"),
            "sections_failed": sum(1 for r in results if not r.complete),
            "parts_total": len(payloads),
            "requests": self._requests,
            "truncation_retries": self._retries,
            "missing_section_retries": self._missing_retries,
            "chunks_curated": len(deduped),
            "chunks_rejected": len(rejected),
            "chunks_deduplicated": max(0, validated_total - len(deduped)),
            "chunks_rewritten": sum(1 for i in deduped if i.rewritten),
            "chunks_split": sum(r.split_chunks for r in results),
            "source_chars": sum(len(s.text) for s in sections),
            "curated_chars": sum(len(i.content) for i in deduped),
            "source_tokens": sum(self.counter.count(s.text) for s in sections),
            "curated_tokens": sum(self.counter.count(i.content) for i in deduped),
            "discarded": dict(discarded),
        }

        return PageCuration(
            sections=results,
            complete=complete,
            page_issues=page_issues,
            contradictions=contradictions,
            missing_context=missing_context,
            discarded=discarded,
            discard_samples=samples,
            rejected=rejected,
            failed_sections=failed,
            stats=stats,
        )

    # ------------------------------------------------------------------ #
    #  Payload construction                                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _block_types(section: Section) -> str:
        kinds = sorted({b.kind for b in section.blocks if b.text})
        return ",".join(kinds) or "text"

    def _payloads(
        self,
        page: Page,
        sections: list[Section],
        versions: dict[str, Optional[VersionInfo]],
    ) -> list[SectionPayload]:
        payloads: list[SectionPayload] = []
        order = 0
        for section in sections:
            version = versions.get(section.section_id)
            text = section.text.strip()
            if not text:
                continue
            budget = max(MIN_PART_TOKENS, self.input_budget - 200)
            text_parts = self._split_text(text, budget)
            for index, part_text in enumerate(text_parts, start=1):
                payloads.append(SectionPayload(
                    section=section,
                    version=version,
                    text=part_text,
                    part_label=f"{index}/{len(text_parts)}",
                    order=order,
                    block_types=self._block_types(section),
                    tokens=self.counter.count(part_text),
                ))
                order += 1
        return payloads

    def _split_text(self, text: str, budget: int) -> list[str]:
        """Split by lines/sentences under the budget, never mid-table-row."""
        if self.counter.count(text) <= budget:
            return [text]
        units: list[tuple[str, bool]] = []
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            sentences = [(s, i == 0) for i, s in enumerate(_sentences(line)) if s]
            units.extend(sentences)
        return pack_units(units, self.counter, budget, 0) or [text]

    def _group(self, payloads: list[SectionPayload]) -> list[list[SectionPayload]]:
        """Contiguous groups under the input budget, one part per section."""
        groups: list[list[SectionPayload]] = []
        current: list[SectionPayload] = []
        used = 0
        for payload in payloads:
            same_section = any(p.section.section_id == payload.section.section_id for p in current)
            too_large = (
                payload.tokens >= self.large_section_tokens
                or used >= self.large_section_tokens
            )
            if current and (same_section or too_large or used + payload.tokens > self.input_budget):
                groups.append(current)
                current, used = [], 0
            current.append(payload)
            used += payload.tokens + 40
        if current:
            groups.append(current)
        return groups

    # ------------------------------------------------------------------ #
    #  LLM calls                                                           #
    # ------------------------------------------------------------------ #

    def _run_group(
        self,
        page: Page,
        analysis: PageAnalysis,
        group: list[SectionPayload],
        facts: list[str],
        depth: int = 0,
    ) -> list[PartOutcome]:
        if not group:
            return []
        user = prompts.curate_user(
            page, analysis, [payload.as_dict() for payload in group], facts
        )
        logger.debug(
            "curation request %s: %s",
            page.url,
            [(p.section.section_id, p.part_label, p.tokens) for p in group],
        )
        with self._lock:
            self._requests += 1
        try:
            response = self.client.complete_json(
                self.system, user, CurationResponse,
                max_tokens=int(self.settings.CURATION_MAX_OUTPUT_TOKENS),
            )
        except OutputTruncated:
            with self._lock:
                self._retries += 1
            if depth >= MAX_SPLIT_DEPTH:
                return [PartOutcome(p, None, "output truncated; cannot split further") for p in group]
            if len(group) > 1:
                mid = len(group) // 2
                return (
                    self._run_group(page, analysis, group[:mid], facts, depth + 1)
                    + self._run_group(page, analysis, group[mid:], facts, depth + 1)
                )
            halves = self._halve_payload(group[0])
            if halves is None:
                return [PartOutcome(group[0], None, "output truncated; single unsplittable section")]
            return (
                self._run_group(page, analysis, [halves[0]], facts, depth + 1)
                + self._run_group(page, analysis, [halves[1]], facts, depth + 1)
            )
        except Exception as exc:  # LLMError, network, validation after retries
            logger.warning("curation failed for %s: %s", page.url, exc)
            return [PartOutcome(p, None, f"{type(exc).__name__}: {exc}") for p in group]

        by_id: dict[str, list[SectionCuration]] = {}
        for section in response.sections:
            by_id.setdefault(section.source_section_id.strip(), []).append(section)
        logger.debug("curation response %s: %s", page.url, sorted(by_id))

        known = {p.section.section_id for p in group}
        for unknown in set(by_id) - known:
            logger.warning("curation returned unknown section %r for %s", unknown, page.url)

        outcomes: list[PartOutcome] = []
        missing: list[SectionPayload] = []
        for payload in group:
            sid = payload.section.section_id
            matches = by_id.get(sid, [])
            if not matches:
                missing.append(payload)
                continue
            # One payload per section per request, so a duplicate response is
            # a model error; merge its chunks instead of silently dropping them.
            merged = matches[0]
            if len(matches) > 1:
                for extra in matches[1:]:
                    merged.chunks.extend(extra.chunks)
                    merged.discarded_content.extend(extra.discarded_content)
                    merged.issues.extend(extra.issues)
                merged.issues.append(f"model returned section {sid} more than once; chunks merged")
            outcomes.append(PartOutcome(payload, merged))

        # A large reply can silently omit one section (seen with two large
        # sibling program tabs).  Retry it alone — or split it further — before
        # declaring the page incomplete.  Bounded by MAX_SPLIT_DEPTH.
        got = ", ".join(sorted(by_id)) or "empty sections list"
        for payload in missing:
            retried: list[PartOutcome] = []
            if depth < MAX_SPLIT_DEPTH:
                with self._lock:
                    self._missing_retries += 1
                if len(group) > 1:
                    retried = self._run_group(page, analysis, [payload], facts, depth + 1)
                else:
                    halves = self._halve_payload(payload)
                    if halves is not None:
                        retried = (
                            self._run_group(page, analysis, [halves[0]], facts, depth + 1)
                            + self._run_group(page, analysis, [halves[1]], facts, depth + 1)
                        )
            if retried:
                outcomes.extend(retried)
            else:
                outcomes.append(PartOutcome(
                    payload, None,
                    f"section missing from model response (got: {got})",
                ))
        return outcomes

    def _halve_payload(self, payload: SectionPayload) -> Optional[tuple[SectionPayload, SectionPayload]]:
        lines = payload.text.split("\n")
        if len(lines) >= 2:
            mid = len(lines) // 2
            left, right = "\n".join(lines[:mid]).strip(), "\n".join(lines[mid:]).strip()
        else:
            units = [(s, True) for s in _sentences(payload.text) if s]
            if len(units) < 2:
                return None
            budget = max(MIN_PART_TOKENS, self.counter.count(payload.text) // 2)
            parts = pack_units(units, self.counter, budget, 0)
            left, right = (parts[0].strip(), "\n".join(parts[1:]).strip()) if len(parts) > 1 else ("", "")
        if not left or not right:
            return None
        return (
            replace(payload, text=left, part_label=f"{payload.part_label}a",
                    tokens=self.counter.count(left)),
            replace(payload, text=right, part_label=f"{payload.part_label}b",
                    tokens=self.counter.count(right)),
        )

    # ------------------------------------------------------------------ #
    #  Merge + validate                                                    #
    # ------------------------------------------------------------------ #

    def _merge(
        self,
        page: Page,
        sections: list[Section],
        versions: dict[str, Optional[VersionInfo]],
        payloads: list[SectionPayload],
        outcomes: list[PartOutcome],
    ) -> list[SectionCurationResult]:
        by_section: dict[str, SectionCurationResult] = {}
        for section in sections:
            result = SectionCurationResult(
                section=section,
                version=versions.get(section.section_id),
                source_chars=len(section.text),
                source_tokens=self.counter.count(section.text),
            )
            if not section.text.strip():
                # Nothing was sent for this section, so nothing can be missing.
                result.status = "no_relevant_content"
                result.discarded.append({
                    "source_section_id": section.section_id,
                    "description": "prazna sekcija bez teksta",
                    "reason": "empty",
                })
            by_section[section.section_id] = result

        for outcome in outcomes:
            payload = outcome.payload
            result = by_section.get(payload.section.section_id)
            if result is None:
                continue
            result.parts = max(result.parts, self._parts_count(payload.part_label))
            if outcome.section is None:
                result.complete = False
                result.status = "failed"
                result.issues.append(outcome.error or "not processed")
                continue
            result.discarded.extend(
                {
                    "source_section_id": d.source_section_id or payload.section.section_id,
                    "description": truncate(normalize_text(d.description), 300),
                    "reason": d.reason,
                }
                for d in outcome.section.discarded_content
            )
            result.issues.extend(truncate(normalize_text(i), 300) for i in outcome.section.issues if i.strip())
            if outcome.section.status == "no_relevant_content" and outcome.section.chunks:
                result.issues.append("model said no_relevant_content but returned chunks; chunks kept")

            # Validated later, when the whole page's evidence indexes exist.
            result.pending.extend((payload, raw) for raw in outcome.section.chunks)

            if outcome.section.status == "no_relevant_content":
                result.status = "no_relevant_content"

        return list(by_section.values())

    @staticmethod
    def _parts_count(label: str) -> int:
        match = re.match(r"\d+/(\d+)", label)
        return int(match.group(1)) if match else 1

    def _page_corpus(
        self,
        page: Page,
        analysis: PageAnalysis,
        sections: list[Section],
        versions: dict[str, Optional[VersionInfo]],
        facts: list[str],
    ) -> str:
        chunks = [page.title, page.url, analysis.page_summary or ""]
        for section in sections:
            version = versions.get(section.section_id)
            chunks.append(section.text)
            chunks.append(" › ".join(section.breadcrumb))
            if version is not None:
                chunks.append(f"{version.version_label} {version.status_label} {version.valid_for}")
        chunks.extend(facts)
        return "\n".join(chunks)

    def _facts_for_page(self, page: Page) -> list[str]:
        """Reviewed version maps relevant to this page.

        Program pages get their own version maps (small request prefix); other
        pages get all reviewed maps, because FAQ/news text may contradict any
        of the current programs.
        """
        entries = self.registry.entries_for_page(page.url)
        if not entries:
            return self.facts
        maps = self.registry.version_maps(page.url, TextIndex(page.full_text))
        return [m["text"] for m in maps] or self.facts

    def _current_versions(self, page: Page) -> list[VersionInfo]:
        """Program versions currently enrolling on this page (registry only)."""
        out: list[VersionInfo] = []
        for entry in self.registry.entries_for_page(page.url):
            if entry.modules and self.registry.status(entry) == "enrolling":
                out.append(self.registry.version_info(entry))
        return out

    @staticmethod
    def _module_count_contradiction(content: str, current_versions: list[VersionInfo]) -> str:
        """Flag an unversioned text whose module/smer count contradicts the
        currently enrolling program(s).  Text with an explicit old-program cue
        is left alone — it is allowed to describe a historical version."""
        if _LEGACY_CUE_RE.search(content):
            return ""
        folded = fold(content)
        counts = {int(match.group(1)) for match in _MODULE_COUNT_RE.finditer(folded)}
        for match in _MODULE_COUNT_WORD_RE.finditer(folded):
            value = factcheck.NUMBER_WORDS.get(match.group(1))
            if value is not None:
                counts.add(value)
        if not counts:
            return ""
        allowed = {len(version.modules) for version in current_versions if version.modules}
        if counts & allowed:
            return ""
        current = next((version for version in current_versions if version.modules), None)
        if current is None:
            return ""
        count = sorted(counts)[0]
        return (
            f"navodi {count} modula/smera, a aktuelni program "
            f"{current.version_label} ima {len(current.modules)}"
        )

    def _validate_result(
        self,
        result: SectionCurationResult,
        payloads: list[SectionPayload],
        page_corpus: str,
        page: Page,
        rejected: list[dict],
        current_versions: list[VersionInfo],
    ) -> None:
        pending = list(result.pending)
        result.pending = []

        # Evidence may reference any section of the page (the model sometimes
        # merges context from a parent tab); every quote is still checked
        # against the actual extracted text of the section it names.
        text_index: dict[str, TextIndex] = {
            section.section_id: TextIndex(section.text) for section in _unique_sections(payloads)
        }

        base = f"{sha1(page.url)[:10]}-{sha1(result.section.dom_path)[:8]}"
        ordinal = len(result.items)
        for payload, raw in pending:
            valid = self._clean_chunk(raw, payload, result, text_index, page_corpus, base, ordinal)
            accepted = [entry for entry in valid if not entry[1]]
            if len(accepted) > 1:
                result.split_chunks += len(accepted) - 1
            for item, reason, detail in valid:
                if reason:
                    result.rejected.append({
                        "page_url": page.url,
                        "section_id": result.section.section_id,
                        "reason": reason,
                        "detail": detail,
                        "content_preview": truncate(normalize_text(raw.content), 240),
                    })
                    rejected.append(result.rejected[-1])
                    continue
                # Deterministic safety net for the known failure mode: the FAQ
                # says "3 smera" while the currently enrolling program has 2.
                # The model is allowed to be cautious, but a count that
                # contradicts the registry must still be hidden by retrieval.
                if result.version is None and current_versions:
                    contradiction = self._module_count_contradiction(item.content, current_versions)
                    if contradiction:
                        item.outdated = True
                        item.outdated_reason = truncate(contradiction, 200)
                        if contradiction not in item.issues:
                            item.issues.append(contradiction)
                result.items.append(item)  # type: ignore[arg-type]
                ordinal += 1

        # Registry module names present on the page must not silently vanish.
        if result.version is not None and result.version.modules:
            missing = factcheck.missing_registry_modules(
                result.version.modules,
                result.section.text,
                [item.content for item in result.items],
            )
            if missing:
                message = "registry modules not mentioned in curated chunks: " + ", ".join(missing)
                result.validation_issues.append(message)
                result.issues.append(message)

        # Aggressive compression is allowed, but a section that keeps almost
        # nothing is a human-review signal (the facts may have been summarised
        # away).  This is a warning, not a rejection.
        source_chars = len(result.section.text)
        curated_chars = sum(len(item.content) for item in result.items)
        if result.items and source_chars >= 500:
            has_table = any(block.kind == "table" for block in result.section.blocks)
            threshold = 0.15 if has_table else 0.30
            if curated_chars < source_chars * threshold:
                message = (
                    f"low coverage: only {curated_chars}/{source_chars} source chars kept "
                    f"(< {threshold:.0%})"
                )
                result.validation_issues.append(message)
                result.issues.append(message)

        result.validated_items = len(result.items)
        failed_part = result.status == "failed" or not result.complete
        if result.items:
            result.status = "failed" if failed_part else "processed"
            result.complete = not failed_part
        elif result.discarded and not failed_part:
            result.status = "no_relevant_content"
            result.complete = True
        else:
            result.status = "failed"
            result.complete = False
            if not result.issues:
                result.issues.append("no chunks produced and no discarded content reported")

    def _clean_chunk(
        self,
        raw: CuratedChunk,
        payload: SectionPayload,
        result: SectionCurationResult,
        text_index: dict[str, TextIndex],
        page_corpus: str,
        base: str,
        ordinal: int,
    ) -> list[tuple[Optional[CuratedItem], str, str]]:
        """Validate one model chunk → zero or more final items, or a rejection."""
        content = normalize_text(raw.content).strip()
        if len(content) < 12:
            return [(None, "too_short", f"{len(content)} chars")]

        # ── evidence ─────────────────────────────────────────────────── #
        evidence: list[dict] = []
        for ref in raw.source_evidence:
            sid = ref.source_section_id.strip()
            excerpt = ref.source_excerpt.strip()
            if sid not in text_index:
                return [(
                    None, "evidence_unknown_section",
                    f"evidence references {sid!r}, not a section on this page",
                )]
            if len(fold(excerpt)) < 4 or not text_index[sid].contains(excerpt):
                return [(
                    None, "evidence_not_found",
                    f"quote not found in {sid}: {truncate(excerpt, 120)!r}",
                )]
            evidence.append({
                "source_section_id": sid,
                "source_excerpt": truncate(excerpt, MAX_EVIDENCE_QUOTE_CHARS),
            })
        if not evidence:
            return [(None, "no_evidence", "chunk has no source evidence")]

        # ── facts ────────────────────────────────────────────────────── #
        unsupported = factcheck.unsupported_facts(content, page_corpus)
        if unsupported and self.strict_facts:
            return [(
                None, "unsupported_facts",
                "numbers/codes not present in the source: " + ", ".join(unsupported[:8]),
            )]

        issues = [truncate(normalize_text(i), 240) for i in raw.issues if i.strip()]
        if unsupported:
            issues.append("nepotvrđene vrednosti iz izvora: " + ", ".join(unsupported[:8]))

        outdated = bool(raw.outdated) and result.version is None
        reason = truncate(normalize_text(raw.outdated_reason), 200) if outdated else ""

        source_sections = sorted({payload.section.section_id} | {e["source_section_id"] for e in evidence})
        rewritten = factcheck.is_rewritten(content, payload.section.text)

        # ── size ─────────────────────────────────────────────────────── #
        pieces = self._split_content(content, self.chunk_max_tokens)
        items: list[tuple[Optional[CuratedItem], str, str]] = []
        for index, piece in enumerate(pieces):
            item = CuratedItem(
                chunk_id=f"{base}-c{ordinal + index:03d}",
                section=result.section,
                version=result.version,
                content=piece,
                context=truncate(normalize_text(raw.context), 400),
                questions=self._clean_list(raw.questions, 4, 200),
                keywords=self._clean_list(raw.keywords, 8, 100),
                evidence=evidence,
                source_sections=source_sections,
                quality=raw.quality,
                issues=issues,
                outdated=outdated,
                outdated_reason=reason,
                order=payload.order,
                ordinal=ordinal + index,
                rewritten=rewritten,
                heading_path=self._heading_for(result.section, evidence),
            )
            items.append((item, "", ""))
        return items

    def _split_content(self, content: str, budget: int) -> list[str]:
        if self.counter.count(content) <= budget:
            return [content]
        units: list[tuple[str, bool]] = []
        for line in content.split("\n"):
            line = line.strip()
            if not line:
                continue
            units.extend((s, i == 0) for i, s in enumerate(_sentences(line)) if s)
        return pack_units(units, self.counter, budget, 0) or [content]

    @staticmethod
    def _heading_for(section: Section, evidence: list[dict]) -> list[str]:
        """Most specific heading path whose block contains a source quote.

        The model returns no heading, so it is recovered deterministically from
        the evidence.  This keeps the old `page › tab › sub-heading` header
        (e.g. `Informatika › Osnovne › Izborni modul: Nauka o podacima`).
        """
        blocks = [(fold(b.text), b.heading_path) for b in section.blocks if b.heading_path]
        best: list[str] = []
        for ref in evidence:
            needle = fold(ref.get("source_excerpt", ""))
            if not needle:
                continue
            for text, path in blocks:
                if needle in text and len(path) > len(best):
                    best = list(path)
        return best

    @staticmethod
    def _clean_list(values: list[str], limit: int, max_chars: int) -> list[str]:
        out: list[str] = []
        for value in values:
            text = truncate(normalize_text(value), max_chars)
            if text and text not in out:
                out.append(text)
            if len(out) >= limit:
                break
        return out

    # ------------------------------------------------------------------ #
    #  Deterministic dedupe + page reconciliation                          #
    # ------------------------------------------------------------------ #

    def _dedupe(self, items: list[CuratedItem]) -> list[CuratedItem]:
        seen: dict[str, CuratedItem] = {}
        kept: list[CuratedItem] = []
        for item in sorted(items, key=lambda i: (i.order, i.ordinal)):
            key = sha1(fold(item.content))
            canonical = seen.get(key)
            if canonical is None:
                seen[key] = item
                kept.append(item)
                continue
            for sid in item.source_sections:
                if sid not in canonical.source_sections:
                    canonical.source_sections.append(sid)
            canonical.evidence.extend(e for e in item.evidence if e not in canonical.evidence)
        return kept

    def _reconcile(
        self,
        page: Page,
        items: list[CuratedItem],
        results: list[SectionCurationResult],
    ) -> tuple[list[CuratedItem], list[str], list[str], list[str]]:
        if len(items) < 2:
            return items, [], [], []
        cap = int(self.settings.CURATION_RECONCILE_MAX_CHUNKS)
        if len(items) > cap:
            results[0].issues.append(
                f"page has {len(items)} chunks (> {cap}); page reconciliation skipped"
            )
            return items, [], [], [f"reconciliation skipped: {len(items)} chunks"]

        digest = [{
            "id": item.chunk_id,
            "section": " › ".join(item.section.breadcrumb),
            "version_label": item.version.version_label if item.version else "",
            "content": item.content,
        } for item in items]
        try:
            with self._lock:
                self._requests += 1
            response = self.client.complete_json(
                prompts.RECONCILE_SYSTEM,
                prompts.reconcile_user(page, digest),
                ReconcileResponse,
                max_tokens=min(4000, 300 + 80 * len(digest)),
            )
        except Exception as exc:
            logger.warning("reconciliation failed for %s: %s", page.url, exc)
            return items, [], [], [f"reconciliation failed: {type(exc).__name__}: {exc}"]

        by_id = {item.chunk_id: item for item in items}
        removed: set[str] = set()
        issues: list[str] = []
        contradictions: list[str] = []
        missing_context = [truncate(normalize_text(t), 240) for t in response.missing_context if t.strip()]

        for group in response.duplicate_groups:
            canonical = by_id.get(group.canonical_chunk_id)
            if canonical is None or canonical.chunk_id in removed:
                issues.append(f"reconciliation: unknown canonical id {group.canonical_chunk_id!r}")
                continue
            for duplicate_id in group.duplicate_chunk_ids:
                duplicate = by_id.get(duplicate_id)
                if duplicate is None or duplicate.chunk_id == canonical.chunk_id or duplicate_id in removed:
                    issues.append(f"reconciliation: unknown duplicate id {duplicate_id!r}")
                    continue
                if not _near_duplicate(canonical.content, duplicate.content):
                    issues.append(
                        f"reconciliation: {duplicate_id} is not a real duplicate of "
                        f"{canonical.chunk_id}; kept it"
                    )
                    continue
                for sid in duplicate.source_sections:
                    if sid not in canonical.source_sections:
                        canonical.source_sections.append(sid)
                canonical.evidence.extend(
                    e for e in duplicate.evidence if e not in canonical.evidence
                )
                removed.add(duplicate_id)

        for contradiction in response.contradictions:
            known_ids = [cid for cid in contradiction.chunk_ids if cid in by_id]
            if not known_ids:
                issues.append(
                    "reconciliation: contradiction references unknown chunk ids "
                    f"{contradiction.chunk_ids!r}; ignored"
                )
                continue
            ids = ", ".join(known_ids)
            text = f"{ids}: {truncate(normalize_text(contradiction.description), 300)}"
            contradictions.append(text)
            issues.append("contradiction: " + text)

        issues.extend(truncate(normalize_text(t), 240) for t in response.page_issues if t.strip())
        kept = [item for item in items if item.chunk_id not in removed]
        return kept, contradictions, missing_context, issues


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _unique_sections(payloads: list[SectionPayload]) -> list[Section]:
    seen: set[str] = set()
    out: list[Section] = []
    for payload in payloads:
        if payload.section.section_id not in seen:
            seen.add(payload.section.section_id)
            out.append(payload.section)
    return out


def _sentences(text: str) -> list[str]:
    from src.utils.text import split_units

    return [unit for unit, _ in split_units(text)]


def _near_duplicate(a: str, b: str) -> bool:
    """Conservative duplicate test used before deleting a reconciled chunk."""
    folded_a, folded_b = fold(a), fold(b)
    if folded_a == folded_b:
        return True
    tokens_a, tokens_b = set(folded_a.split()), set(folded_b.split())
    if not tokens_a or not tokens_b:
        return False
    jaccard = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
    length_ratio = min(len(tokens_a), len(tokens_b)) / max(len(tokens_a), len(tokens_b))
    return jaccard >= 0.9 and length_ratio >= 0.85
