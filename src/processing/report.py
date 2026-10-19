"""
report.py
~~~~~~~~~
Run report — the thing to read after every ingest:

  data/reports/<timestamp>/report.md         human-readable summary
  data/reports/<timestamp>/chunks.jsonl      every chunk: document, embedded
                                             text, metadata, drop reason,
                                             evidence, source sections and
                                             whether it was committed
  data/reports/<timestamp>/curation.json     per-page curation stats, rejected
                                             chunks, discarded content,
                                             incomplete sections, contradictions
  data/reports/<timestamp>/page_analysis.json raw DeepSeek page analyses
"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.utils.text import truncate


@dataclass
class SectionRow:
    section_id: str
    path: str
    version: str
    status: str
    hidden: bool
    reachable: bool
    chunks: int
    url: str


@dataclass
class PageReport:
    target: str
    url: str = ""
    title: str = ""
    doc_type: str = ""
    outcome: str = "ok"                 # ok | dry-run | skipped | alias | failed | empty | partial
    error: str = ""
    page_type: str = ""
    summary: str = ""
    published_date: Optional[str] = None
    alias_of: str = ""
    tab_params: dict = field(default_factory=dict)
    sections: list[SectionRow] = field(default_factory=list)
    unreachable: list[tuple[str, str, int]] = field(default_factory=list)
    stored: int = 0
    deleted: int = 0
    dropped: Counter = field(default_factory=Counter)
    drafts: list[dict] = field(default_factory=list)
    version_maps: list[dict] = field(default_factory=list)
    issues: list[tuple[str, str, str]] = field(default_factory=list)
    previews: list[dict] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    # ── AI curation ────────────────────────────────────────────────────── #
    curation_status: str = ""           # complete | partial | deterministic
    curation: dict = field(default_factory=dict)
    discarded: Counter = field(default_factory=Counter)
    discard_samples: list[dict] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    missing_context: list[str] = field(default_factory=list)
    failed_sections: list[tuple[str, str, str]] = field(default_factory=list)
    llm_calls: int = 0
    llm_cached_calls: int = 0
    embed_seconds: float = 0.0
    write_seconds: float = 0.0
    embed_split: int = 0


@dataclass
class RunReport:
    links_file: str
    collection: str
    db_path: str
    dry_run: bool
    llm: str
    started: float = field(default_factory=time.time)
    pages: list[PageReport] = field(default_factory=list)
    usage_line: str = ""
    drafts_file: str = ""

    # ------------------------------------------------------------------ #

    def totals(self) -> dict:
        outcomes = Counter(p.outcome for p in self.pages)
        dropped: Counter = Counter()
        for p in self.pages:
            dropped.update(p.dropped)
        return {
            "outcomes": outcomes,
            "stored": sum(p.stored for p in self.pages),
            "deleted": sum(p.deleted for p in self.pages),
            "dropped": dropped,
        }

    def curation_totals(self) -> dict:
        pages = [p for p in self.pages if p.curation_status in ("complete", "partial")]
        stats = [p.curation for p in pages]
        return {
            "pages": len(pages),
            "complete": sum(1 for p in pages if p.curation_status == "complete"),
            "partial": sum(1 for p in pages if p.curation_status == "partial"),
            "sections": sum(int(s.get("sections_total", 0)) for s in stats),
            "sections_processed": sum(int(s.get("sections_processed", 0)) for s in stats),
            "sections_failed": sum(int(s.get("sections_failed", 0)) for s in stats),
            "chunks": sum(int(s.get("chunks_curated", 0)) for s in stats),
            "chunks_final": sum(
                int(s.get("chunks_final", s.get("chunks_curated", 0))) for s in stats
            ),
            "rejected": sum(int(s.get("chunks_rejected", 0)) for s in stats),
            "deduplicated": sum(int(s.get("chunks_deduplicated", 0)) for s in stats),
            "rewritten": sum(int(s.get("chunks_rewritten", 0)) for s in stats),
            "split": sum(int(s.get("chunks_split", 0)) for s in stats),
            "requests": sum(int(s.get("requests", 0)) for s in stats),
            "truncation_retries": sum(int(s.get("truncation_retries", 0)) for s in stats),
            "missing_section_retries": sum(
                int(s.get("missing_section_retries", 0)) for s in stats
            ),
            "source_chars": sum(int(s.get("source_chars", 0)) for s in stats),
            "curated_chars": sum(int(s.get("curated_chars", 0)) for s in stats),
            "source_tokens": sum(int(s.get("source_tokens", 0)) for s in stats),
            "curated_tokens": sum(int(s.get("curated_tokens", 0)) for s in stats),
        }

    def summary_lines(self) -> list[str]:
        t = self.totals()
        o = t["outcomes"]
        dropped = ", ".join(f"{k} {v}" for k, v in t["dropped"].most_common()) or "none"
        lines = [
            f"Pages: {o.get('ok', 0) + o.get('dry-run', 0)} processed, {o.get('skipped', 0)} unchanged (skipped), "
            f"{o.get('partial', 0)} partial, {o.get('alias', 0)} aliases, "
            f"{o.get('empty', 0)} empty, {o.get('failed', 0)} failed",
            f"Chunks: {t['stored']} {'prepared (dry run)' if self.dry_run else 'stored'}, "
            f"{t['deleted']} stale deleted, dropped: {dropped}",
        ]
        c = self.curation_totals()
        if c["pages"]:
            discarded: Counter = Counter()
            for page in self.pages:
                discarded.update(page.discarded)
            shrink = (c["curated_chars"] / c["source_chars"] * 100) if c["source_chars"] else 0
            lines.append(
                f"Curation: {c['complete']}/{c['pages']} pages complete, {c['partial']} partial · "
                f"{c['sections_processed']}/{c['sections']} sections processed, {c['sections_failed']} failed · "
                f"{c['chunks_final']} final chunks ({c['chunks']} curated, {c['rewritten']} rewritten, "
                f"{c['deduplicated']} deduplicated, {c['rejected']} rejected, {c['split']} split) · "
                f"{c['source_chars']:,} → {c['curated_chars']:,} chars ({shrink:.0f}%) · "
                f"{c['requests']} LLM request(s) "
                f"({c['truncation_retries']} output-truncation, "
                f"{c['missing_section_retries']} missing-section retries) · discarded: "
                + (", ".join(f"{k} {v}" for k, v in discarded.most_common()) or "none")
            )
        if self.usage_line:
            lines.append(self.usage_line)
        return lines

    # ------------------------------------------------------------------ #

    def _page_markdown(self, p: PageReport) -> list[str]:
        out = [f"## {p.title or p.target}", ""]
        out.append(f"- URL: {p.url or p.target}")
        out.append(f"- Outcome: **{p.outcome}**" + (f" — {p.error}" if p.error else "")
                   + (f" (same content as {p.alias_of})" if p.alias_of else ""))
        if p.outcome in ("skipped", "alias"):
            return out + [""]
        out.append(f"- Type: {p.doc_type} / {p.page_type or '-'}"
                   + (f" · published {p.published_date}" if p.published_date else ""))
        if p.summary:
            out.append(f"- Summary: {p.summary}")
        if p.tab_params:
            out.append("- Verified tab links: " + ", ".join(f"`?{v}` → {k}" for k, v in p.tab_params.items()))
        if p.flags:
            out.append(f"- Flags: {', '.join(p.flags)}")
        dropped = ", ".join(f"{k} {v}" for k, v in p.dropped.most_common()) or "none"
        out.append(f"- Chunks: {p.stored} stored · {p.deleted} stale deleted · dropped: {dropped}")
        if p.curation_status:
            c = p.curation
            if p.curation_status == "deterministic":
                out.append("- Curation: **deterministic** (`--no-llm`; no AI-generated chunks)")
            else:
                shrink = (c.get("curated_chars", 0) / c.get("source_chars", 1) * 100) if c.get("source_chars") else 0
                out.append(
                    f"- Curation: **{p.curation_status}** · "
                    f"{c.get('sections_processed', 0)}/{c.get('sections_total', 0)} sections processed"
                    + (f", {c.get('sections_failed', 0)} failed" if c.get("sections_failed") else "")
                    + f" · {c.get('chunks_curated', 0)} chunks "
                    f"({c.get('chunks_rewritten', 0)} rewritten, {c.get('chunks_deduplicated', 0)} deduplicated, "
                    f"{c.get('chunks_rejected', 0)} rejected, {c.get('chunks_split', 0)} split)"
                    + f" · {c.get('source_chars', 0):,} → {c.get('curated_chars', 0):,} chars ({shrink:.0f}%)"
                    + f" · {c.get('requests', 0)} LLM request(s) "
                    f"({c.get('truncation_retries', 0)} truncation, "
                    f"{c.get('missing_section_retries', 0)} missing-section retries)"
                    + (f" · {p.llm_calls} API call(s), {p.llm_cached_calls} cache hit(s)" if c.get("requests") else "")
                )
        if p.embed_seconds or p.write_seconds:
            out.append(
                f"- Embedding/write: {p.embed_seconds:.1f} s / {p.write_seconds:.1f} s"
                + (f" · {p.embed_split} embedding split(s)" if p.embed_split else "")
            )
        out.append("")

        if p.sections:
            out += ["### Sections", "", "| Section | Path | Version | Status | Hidden | Chunks | Link |",
                    "|---|---|---|---|---|---|---|"]
            for s in p.sections:
                out.append(
                    f"| `{s.section_id}` | {s.path} | {s.version or '-'} | {s.status or '-'} | "
                    f"{'yes' if s.hidden else ''} | {s.chunks} | {s.url} |"
                )
            out.append("")
        if p.unreachable:
            out += ["### Not indexed — unreachable for visitors", "",
                    "Hidden panels that no tab or link opens (orphans / copy-paste leftovers).", ""]
            out += [f"- `{sid}` {path} ({chars} chars)" for sid, path, chars in p.unreachable]
            out.append("")
        if p.failed_sections:
            out += ["### Incomplete sections (the page was not committed)", ""]
            for sid, path, reason in p.failed_sections:
                out.append(f"- `{sid}` {path}: {reason}")
            out.append("")
        if p.rejected:
            out += ["### Chunks rejected by validation", "", "| Section | Reason | Detail | Preview |",
                    "|---|---|---|---|"]
            for entry in p.rejected[:80]:
                preview = str(entry.get("content_preview", "")).replace("|", "\\|")[:160]
                detail = str(entry.get("detail", "")).replace("|", "\\|")[:200]
                out.append(
                    f"| `{entry.get('section_id', '-')}` | {entry.get('reason', '-')} | "
                    f"{detail or '-'} | {preview or '-'} |"
                )
            if len(p.rejected) > 80:
                out.append(f"- … {len(p.rejected) - 80} more in curation.json")
            out.append("")
        if p.discarded:
            out += ["### Content discarded by curation", "",
                    "Reasons: " + ", ".join(f"{k} {v}" for k, v in p.discarded.most_common()), ""]
            for sample in p.discard_samples[:20]:
                out.append(
                    f"- `{sample.get('section_id', '-')}` ({sample.get('reason', '-')}): "
                    f"{sample.get('description', '')}"
                )
            out.append("")
        if p.contradictions or p.missing_context:
            out += ["### Reconciliation — contradictions / missing context", ""]
            out += [f"- ⚠ kontradikcija: {text}" for text in p.contradictions]
            out += [f"- missing context: {text}" for text in p.missing_context]
            out.append("")
        if p.version_maps:
            out += ["### Version maps (generated from programs.yaml)", ""]
            for vm in p.version_maps:
                out.append("```")
                out.append(vm["text"])
                out.append("```")
                if vm.get("missing_modules"):
                    out.append(f"⚠ module names not found on the page (omitted): {', '.join(vm['missing_modules'])}")
            out.append("")
        if p.drafts:
            out += ["### Unmapped program panels — drafts written to programs.draft.yaml", ""]
            for d in p.drafts:
                ok = "✅ evidence found" if d.get("evidence_ok") else "⚠ " + "; ".join(d.get("problems", []))
                out.append(f"- {d.get('version_label')} — panels {d.get('panels')} — {ok}")
            out.append("")
        if p.issues:
            out += ["### Issues flagged by DeepSeek", ""]
            out += [f"- `{cid}` {path}: {issue}" for cid, path, issue in p.issues[:60]]
            if len(p.issues) > 60:
                out.append(f"- … {len(p.issues) - 60} more in chunks.jsonl")
            out.append("")
        if p.previews:
            out += ["### Chunk previews (first chunk of each section)", ""]
            for pv in p.previews:
                out.append(f"**`{pv['id']}`** {pv['header']}")
                out.append("")
                out.append("> " + truncate(pv["text"], 400).replace("\n", "\n> "))
                if pv.get("questions"):
                    out.append(f"> _Pitanja:_ {' · '.join(pv['questions'])}")
                out.append("")
        return out

    def write(
        self,
        directory: Path,
        chunk_records: list[dict],
        analyses: dict,
        curation_records: list[dict] | None = None,
    ) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        lines = [
            f"# Ingest report — {time.strftime('%Y-%m-%d %H:%M', time.localtime(self.started))}",
            "",
            f"- Links file: `{self.links_file}`",
            f"- Target: collection `{self.collection}` at `{self.db_path}`"
            + (" — **dry run, nothing written**" if self.dry_run else ""),
            f"- DeepSeek: {self.llm}",
            f"- Duration: {time.time() - self.started:.0f} s",
        ]
        lines += [f"- {line}" for line in self.summary_lines()]
        if self.drafts_file:
            lines.append(f"- Registry drafts: `{self.drafts_file}` — review and move entries into programs.yaml")
        lines.append("")
        for page in self.pages:
            lines += self._page_markdown(page)

        (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")
        with (directory / "chunks.jsonl").open("w", encoding="utf-8") as fh:
            for record in chunk_records:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        (directory / "page_analysis.json").write_text(
            json.dumps(analyses, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if curation_records is not None:
            (directory / "curation.json").write_text(
                json.dumps(curation_records, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        return directory / "report.md"
