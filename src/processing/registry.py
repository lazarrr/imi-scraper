"""
registry.py
~~~~~~~~~~~
Program-version registry (programs.yaml) — the authoritative answer to
"which study-program version does this section belong to?".

Why a registry: program pages hold several versions of the same program
(2026 / 2019 / 2016 / 2011 …).  Labels must be IDENTICAL on every run,
otherwise retrieval keeps mixing "2 modules" with "3 modules".  DeepSeek
drafts entries (programs.draft.yaml), a human reviews them, the code
enforces them.

Status is computed from enrollment years and today's date, so it moves
on by itself (an "upcoming" program becomes "enrolling" when its first
enrollment year arrives):

    upcoming   enrollment has not started yet
    enrolling  still open for enrollment
    phase_out  closed for enrollment, earlier students still study by it
               (enrollment_to + duration(level) + 2 years)
    legacy     older than that
    unknown    not enough information (e.g. no enrollment year)
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Optional

import yaml

from src.processing.fetcher import canonicalize_url
from src.processing.schemas import ProgramVersion, RegistryDraftEntry, VersionInfo
from src.utils.text import TextIndex, fold, sha1

logger = logging.getLogger(__name__)

LEVEL_DURATION = {"osnovne": 4, "master": 1, "doktorske": 3}
PHASE_OUT_GRACE_YEARS = 2
LEVEL_ABBR = {"osnovne": "OAS", "master": "MAS", "doktorske": "DAS"}
LEVEL_NAME = {
    "osnovne": "osnovne akademske studije",
    "master": "master akademske studije",
    "doktorske": "doktorske akademske studije",
}
STATUS_LABEL = {
    "enrolling": "aktuelni program (upis u toku)",
    "phase_out": "prethodni program (važi za ranije upisane studente)",
    "upcoming": "novi program (upis još nije počeo)",
    "legacy": "stari program",
    "unknown": "status programa nije poznat",
}


def enrollment_year(today: date) -> int:
    """Enrollment for academic year Y/Y+1 runs from June of year Y."""
    return today.year if today.month >= 6 else today.year - 1


def academic_year(year: int) -> str:
    return f"{year}/{(year + 1) % 100:02d}"


def _modules_phrase(modules: list[str]) -> str:
    noun = "modul" if len(modules) == 1 else "modula"
    return f"{len(modules)} {noun}: {', '.join(modules)}"


class _TrustAll:
    """Stands in for a TextIndex when the registry itself is the source."""

    @staticmethod
    def contains(_needle: str) -> bool:
        return True


class Registry:
    def __init__(self, entries: list[ProgramVersion], today: date, allowed_domains=()):
        self.today = today
        self.entries: list[ProgramVersion] = []
        self._panel_index: dict[tuple[str, str], ProgramVersion] = {}
        for entry in entries:
            entry = entry.model_copy(update={"page": canonicalize_url(entry.page, allowed_domains)})
            self.entries.append(entry)
            for panel in entry.panels:
                key = (entry.page, panel)
                if key in self._panel_index:
                    logger.warning("programs.yaml: panel %s on %s mapped twice", panel, entry.page)
                self._panel_index.setdefault(key, entry)

    # ------------------------------------------------------------------ #
    #  Loading                                                             #
    # ------------------------------------------------------------------ #

    @classmethod
    def load(cls, path: Path, today: date, allowed_domains=()) -> "Registry":
        if not path.is_file():
            logger.info("No registry at %s — program versions come from DeepSeek drafts only", path)
            return cls([], today, allowed_domains)
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
        if isinstance(raw, dict):
            raw = raw.get("programs", [])
        entries = [ProgramVersion.model_validate(item) for item in raw]
        return cls(entries, today, allowed_domains)

    # ------------------------------------------------------------------ #
    #  Lookup                                                              #
    # ------------------------------------------------------------------ #

    def entries_for_page(self, page_url: str) -> list[ProgramVersion]:
        return [e for e in self.entries if e.page == page_url]

    def mapped_panels(self, page_url: str) -> set[str]:
        return {p for e in self.entries_for_page(page_url) for p in e.panels}

    def page_hash(self, page_url: str) -> str:
        """Changes when this page's entries — or the enrollment year — change."""
        payload = [e.model_dump() for e in self.entries_for_page(page_url)]
        return sha1(json.dumps([payload, enrollment_year(self.today)], sort_keys=True, default=str))

    def resolve(self, page_url: str, panel_chain: list[str]) -> Optional[ProgramVersion]:
        """Entry of the nearest mapped ancestor-or-self panel."""
        for panel in reversed(panel_chain):
            entry = self._panel_index.get((page_url, panel))
            if entry is not None:
                return entry
        return None

    # ------------------------------------------------------------------ #
    #  Status                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def start_year(entry) -> Optional[int]:
        return entry.enrollment_from or entry.accreditation_year

    def _siblings(self, entry, pool=None) -> list:
        pool = self.entries if pool is None else pool
        return [e for e in pool if fold(e.program) == fold(entry.program) and e.level == entry.level]

    def end_year(self, entry, pool=None) -> Optional[int]:
        if entry.enrollment_to is not None:
            return entry.enrollment_to
        start = self.start_year(entry)
        if start is None:
            return None
        later = [
            s for s in (self.start_year(e) for e in self._siblings(entry, pool))
            if s is not None and s > start
        ]
        return min(later) - 1 if later else None

    def status(self, entry, pool=None) -> str:
        if getattr(entry, "status", None):
            return entry.status
        start = self.start_year(entry)
        if start is None:
            return "unknown"
        current = enrollment_year(self.today)
        if start > current:
            return "upcoming"
        end = self.end_year(entry, pool)
        if end is None or end >= current:
            return "enrolling"
        if end + LEVEL_DURATION.get(entry.level, 4) + PHASE_OUT_GRACE_YEARS >= current:
            return "phase_out"
        return "legacy"

    def valid_for(self, entry, pool=None) -> str:
        if getattr(entry, "valid_for", None):
            return entry.valid_for
        start = self.start_year(entry)
        if start is None:
            return ""
        status = self.status(entry, pool)
        if getattr(entry, "status", None) and entry.enrollment_to is None:
            # status was set by hand → only state what is known for sure
            return f"upis od {academic_year(start)}" if status in ("upcoming", "enrolling") else ""
        end = self.end_year(entry, pool)
        if status in ("upcoming", "enrolling") or end is None:
            return f"upis od {academic_year(start)}"
        if end <= start:
            return f"studenti upisani {academic_year(start)}"
        return f"studenti upisani {academic_year(start)}–{academic_year(end)}"

    def version_info(self, entry, source: str = "registry", pool=None, page_url: str = "") -> VersionInfo:
        status = self.status(entry, pool)
        page = getattr(entry, "page", None) or page_url
        return VersionInfo(
            program=entry.program,
            level=entry.level,
            version_label=entry.version_label,
            accreditation_year=entry.accreditation_year,
            status=status,
            status_label=STATUS_LABEL[status],
            valid_for=self.valid_for(entry, pool),
            modules=list(entry.modules),
            source=source,
            key=sha1(f"{page}|{fold(entry.program)}|{entry.level}|{fold(entry.version_label)}")[:12],
        )

    def all_version_maps(self) -> list[str]:
        """Version maps of every registry page, as reviewed (no page check) —
        the reference DeepSeek compares unversioned text against."""
        texts: list[str] = []
        for page in dict.fromkeys(e.page for e in self.entries):
            texts += [m["text"] for m in self.version_maps(page, _TrustAll())]
        return texts

    # ------------------------------------------------------------------ #
    #  Version maps (the verified overview chunk)                          #
    # ------------------------------------------------------------------ #

    def version_maps(self, page_url: str, page_index: TextIndex) -> list[dict]:
        """
        One overview per (program, level) with ≥ 2 versions on this page.
        Module names that do not appear on the page are left out and
        reported (`missing_modules`).
        """
        groups: dict[tuple[str, str], list[ProgramVersion]] = {}
        for entry in self.entries_for_page(page_url):
            groups.setdefault((entry.program, entry.level), []).append(entry)

        maps: list[dict] = []
        for (program, level), entries in groups.items():
            if len(entries) < 2:
                continue
            entries.sort(key=lambda e: (self.start_year(e) or 0, e.version_label), reverse=True)
            lines: list[str] = []
            legacy: list[str] = []
            missing: list[str] = []
            for entry in entries:
                status = self.status(entry)
                if status == "legacy":
                    legacy.append(entry.version_label)
                    continue
                modules = [m for m in entry.modules if page_index.contains(m)]
                missing += [m for m in entry.modules if m not in modules]
                parts = [f"{entry.version_label} — {STATUS_LABEL[status]}"]
                valid = self.valid_for(entry)
                if valid:
                    parts.append(valid)
                line = "; ".join(parts)
                if modules:
                    line += f": {_modules_phrase(modules)}"
                lines.append(f"• {line}.")
            if legacy:
                lines.append(f"• Stari programi (više se ne upisuju): {', '.join(legacy)}.")
            if not lines:
                continue
            title = f"{program} — {LEVEL_NAME[level]} ({LEVEL_ABBR[level]})"
            text = (
                f"Pregled verzija studijskog programa {title}, "
                f"stanje na dan {self.today:%d.%m.%Y}:\n" + "\n".join(lines)
            )
            maps.append({
                "program": program,
                "level": level,
                "title": title,
                "text": text,
                "missing_modules": missing,
            })
        return maps


# --------------------------------------------------------------------------- #
#  Drafts (DeepSeek proposals → programs.draft.yaml)                           #
# --------------------------------------------------------------------------- #

DRAFT_HEADER = (
    "# DRAFT program-version entries proposed by DeepSeek.\n"
    "# Review every entry (years, modules, panels), fix it, then MOVE it into\n"
    "# programs.yaml — only programs.yaml is authoritative.\n"
    "# evidence_ok: false  → at least one quote was NOT found on the page.\n"
)


def check_draft(draft: RegistryDraftEntry, page_index: TextIndex) -> tuple[bool, list[str]]:
    """Every evidence quote and module name must appear on the page."""
    problems = [f"evidence not found: {q!r}" for q in draft.evidence if not page_index.contains(q)]
    problems += [f"module not found: {m!r}" for m in draft.modules if not page_index.contains(m)]
    if not draft.evidence:
        problems.append("no evidence given")
    return (not problems), problems


def write_drafts(path: Path, drafts: list[dict]) -> int:
    """Merge new drafts into programs.draft.yaml (keyed by page + panels)."""
    existing: list[dict] = []
    if path.is_file():
        existing = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    index = {(d.get("page"), tuple(sorted(d.get("panels", [])))): d for d in existing}
    for draft in drafts:
        index[(draft.get("page"), tuple(sorted(draft.get("panels", []))))] = draft
    merged = list(index.values())
    path.write_text(
        DRAFT_HEADER + yaml.safe_dump(merged, allow_unicode=True, sort_keys=False, width=100),
        encoding="utf-8",
    )
    return len(merged)
