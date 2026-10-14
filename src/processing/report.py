"""
report.py
~~~~~~~~~
Run report — the thing to read after every ingest:

  data/reports/<timestamp>/report.md         human-readable summary
  data/reports/<timestamp>/chunks.jsonl      every chunk: document, embedded
                                             text, metadata, drop reason
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
    outcome: str = "ok"                 # ok | dry-run | skipped | alias | failed | empty
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

    def summary_lines(self) -> list[str]:
        t = self.totals()
        o = t["outcomes"]
        dropped = ", ".join(f"{k} {v}" for k, v in t["dropped"].most_common()) or "none"
        lines = [
            f"Pages: {o.get('ok', 0) + o.get('dry-run', 0)} processed, {o.get('skipped', 0)} unchanged (skipped), "
            f"{o.get('alias', 0)} aliases, {o.get('empty', 0)} empty, {o.get('failed', 0)} failed",
            f"Chunks: {t['stored']} {'prepared (dry run)' if self.dry_run else 'stored'}, "
            f"{t['deleted']} stale deleted, dropped: {dropped}",
        ]
        if self.usage_line:
            lines.append(self.usage_line)
        return lines

    # ------------------------------------------------------------------ #

    def _page_markdown(self, p: PageReport) -> list[str]:
        out = [f"## {p.title or p.target}", ""]
        out.append(f"- URL: {p.url or p.target}")
        out.append(f"- Outcome: **{p.outcome}**" + (f" — {p.error}" if p.error else "")
                   + (f" (same content as {p.alias_of})" if p.alias_of else ""))
        if p.outcome in ("failed", "skipped", "alias"):
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

    def write(self, directory: Path, chunk_records: list[dict], analyses: dict) -> Path:
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
        return directory / "report.md"
