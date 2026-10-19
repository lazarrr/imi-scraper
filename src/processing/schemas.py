"""
schemas.py
~~~~~~~~~~
Data model of the pipeline.

Internal structures (dataclasses):
    Page ─┬─ Section (tab panel / heading group / PDF page) ── Block (text | table)
          └─ ...
    Chunk  — one vector-store entry (curated content + location + version + labels)

LLM I/O and registry (pydantic — validated):
    PageAnalysis, RegistryDraftEntry   — DeepSeek page analysis (call A)
    CuratedChunk, SectionCuration,
    CurationResponse                   — DeepSeek content curation (call B)
    ReconcileResponse                  — page-level duplicate/contradiction pass
    ProgramVersion                     — one entry of programs.yaml
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

Level = Literal["osnovne", "master", "doktorske"]
Status = Literal["enrolling", "phase_out", "upcoming", "legacy", "unknown"]
Quality = Literal["ok", "boilerplate", "garbled", "low_value"]

PAGE_TYPES = (
    "studijski_program", "vest", "oglas", "upis", "raspored", "ispit",
    "osoblje", "kontakt", "projekat", "dogadjaj", "dokument", "opste",
)


# --------------------------------------------------------------------------- #
#  Parsed page structure                                                       #
# --------------------------------------------------------------------------- #

@dataclass
class Block:
    kind: str                               # "text" | "table"
    text: str                               # normalized text (tables: rows joined by \n)
    rows: list[str] = field(default_factory=list)   # rendered table rows
    heading_path: list[str] = field(default_factory=list)
    overview: list[str] = field(default_factory=list)   # tables: one course list per caption


@dataclass
class Section:
    section_id: str                         # panel id / "main" / "page-3" / heading slug
    dom_path: str                           # structural path → stable chunk ids
    label: str                              # tab label, heading or "strana N"
    breadcrumb: list[str]                   # page title › tab › sub-tab
    parent_id: Optional[str] = None
    panel_chain: list[str] = field(default_factory=list)   # ancestor-or-self panel ids
    hidden_by_default: bool = False         # panel not shown until a tab is clicked
    ui_hidden: bool = False                 # tab title hidden with inline display:none
    reachable: bool = True                  # a visitor can open it (tab or link path exists)
    top_trigger: Optional[str] = None       # trigger id of the top-level tab above it
    page: int = 0                           # 1-based PDF page, 0 otherwise
    blocks: list[Block] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(b.text for b in self.blocks if b.text)


@dataclass
class Page:
    url: str                                # canonical URL (identity, `page_url`)
    display_url: str                        # human link (final URL, view params stripped)
    doc_type: str                           # "web" | "pdf" | "docx"
    title: str
    sections: list[Section]
    links: list[str] = field(default_factory=list)       # outgoing page links (absolute)
    doc_links: list[str] = field(default_factory=list)   # outgoing PDF/DOCX links
    top_triggers: dict[str, str] = field(default_factory=dict)  # trigger id -> panel id
    tab_params: dict[str, str] = field(default_factory=dict)    # trigger id -> verified ?param
    published_hint: Optional[str] = None    # date found by the extractor (generic pages)
    flags: list[str] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n".join(s.text for s in self.sections if s.blocks)

    def section(self, section_id: str) -> Optional[Section]:
        return next((s for s in self.sections if s.section_id == section_id), None)


@dataclass
class VersionInfo:
    program: str
    level: str
    version_label: str
    accreditation_year: Optional[int]
    status: str                             # enrolling | phase_out | upcoming | legacy | unknown
    status_label: str                       # Serbian, shown in the chunk header
    valid_for: str
    modules: list[str]
    source: str                             # "registry" | "llm"
    key: str                                # identity of the version (merge key)


@dataclass
class Chunk:
    chunk_id: str
    page_url: str
    section: Section
    text: str                               # normalized chunk text
    kind: str                               # "text" | "table" | "version_map"
    index: int
    heading_path: list[str] = field(default_factory=list)
    version: Optional[VersionInfo] = None
    module: Optional[str] = None
    merged_modules: list[str] = field(default_factory=list)
    enrichment: Optional["ChunkEnrichment"] = None
    also_on: list[str] = field(default_factory=list)
    generated: bool = False
    header: str = ""
    document: str = ""                      # what is stored in Chroma `documents`
    embed_text: str = ""                    # what is embedded (without "passage: ")
    metadata: dict = field(default_factory=dict)
    dropped_reason: Optional[str] = None
    # ── AI curation provenance (empty/false for deterministic chunks) ── #
    curated: bool = False
    curation_status: str = ""
    curation_issues: list[str] = field(default_factory=list)
    source_sections: list[str] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)
    rewritten: bool = False

    @property
    def content_hash(self) -> str:
        from src.utils.text import sha1
        return sha1(self.text)


# --------------------------------------------------------------------------- #
#  LLM I/O (DeepSeek JSON output)                                              #
# --------------------------------------------------------------------------- #

class _Lenient(BaseModel):
    model_config = ConfigDict(extra="ignore")


class RegistryDraftEntry(_Lenient):
    program: str
    level: Level
    version_label: str
    accreditation_year: Optional[int] = None
    enrollment_from: Optional[int] = None
    enrollment_to: Optional[int] = None
    modules: list[str] = Field(default_factory=list)
    panels: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)


class PageAnalysis(_Lenient):
    page_type: str = "opste"
    page_summary: str = ""
    published_date: Optional[str] = None
    relevant: bool = True
    registry_draft: list[RegistryDraftEntry] = Field(default_factory=list)

    @field_validator("page_type")
    @classmethod
    def _known_type(cls, v: str) -> str:
        return v if v in PAGE_TYPES else "opste"


class ChunkEnrichment(_Lenient):
    """Labels attached to one final chunk (curated or deterministic).

    Kept as a small internal carrier: `ingest._finalize()` and the report use
    the same fields for both pipeline modes."""

    id: str
    context: str = ""
    questions: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    quality: Quality = "ok"
    issues: list[str] = Field(default_factory=list)
    outdated: bool = False                  # contradicts the reviewed current facts
    outdated_reason: str = ""


# --------------------------------------------------------------------------- #
#  AI content curation (DeepSeek call B) — strict, validated contract         #
# --------------------------------------------------------------------------- #

DiscardReason = Literal[
    "boilerplate", "navigation", "duplicate", "empty", "promotional",
    "unreadable", "irrelevant", "other",
]
CurationStatus = Literal["processed", "no_relevant_content"]


class _Strict(BaseModel):
    # Curation output must have exactly the agreed shape: an unexpected field
    # means the model drifted, and the client will ask for a corrected reply.
    model_config = ConfigDict(extra="forbid")


class SourceEvidence(_Strict):
    source_section_id: str = Field(min_length=1)
    source_excerpt: str = Field(min_length=3)     # verbatim quote from the section


class CuratedChunk(_Strict):
    content: str = Field(min_length=1)            # the final knowledge text
    context: str = ""
    questions: list[str] = Field(default_factory=list, max_length=6)
    keywords: list[str] = Field(default_factory=list, max_length=12)
    source_evidence: list[SourceEvidence] = Field(min_length=1)
    quality: Quality = "ok"
    outdated: bool = False
    outdated_reason: str = ""
    issues: list[str] = Field(default_factory=list, max_length=8)


class DiscardedContent(_Strict):
    source_section_id: str = ""
    description: str = Field(min_length=1, max_length=400)
    reason: DiscardReason = "other"


class SectionCuration(_Strict):
    source_section_id: str = Field(min_length=1)
    status: CurationStatus = "processed"
    chunks: list[CuratedChunk] = Field(default_factory=list)
    discarded_content: list[DiscardedContent] = Field(default_factory=list, max_length=20)
    issues: list[str] = Field(default_factory=list, max_length=10)


class CurationResponse(_Strict):
    sections: list[SectionCuration] = Field(default_factory=list)
    page_issues: list[str] = Field(default_factory=list, max_length=20)


# --------------------------------------------------------------------------- #
#  Page-level reconciliation (duplicates / contradictions across sections)    #
# --------------------------------------------------------------------------- #

class DuplicateGroup(_Strict):
    canonical_chunk_id: str = Field(min_length=1)
    duplicate_chunk_ids: list[str] = Field(default_factory=list, min_length=1)
    reason: str = ""


class Contradiction(_Strict):
    chunk_ids: list[str] = Field(default_factory=list)
    description: str = Field(min_length=1)


class ReconcileResponse(_Strict):
    duplicate_groups: list[DuplicateGroup] = Field(default_factory=list)
    contradictions: list[Contradiction] = Field(default_factory=list)
    missing_context: list[str] = Field(default_factory=list, max_length=20)
    page_issues: list[str] = Field(default_factory=list, max_length=20)


# --------------------------------------------------------------------------- #
#  Program-version registry entry (programs.yaml)                              #
# --------------------------------------------------------------------------- #

class ProgramVersion(_Lenient):
    page: str
    program: str
    level: Level
    version_label: str
    accreditation_year: Optional[int] = None
    enrollment_from: Optional[int] = None   # 2026 means "upis 2026/27"
    enrollment_to: Optional[int] = None     # last enrollment year; inferred if empty
    modules: list[str] = Field(default_factory=list)
    panels: list[str] = Field(default_factory=list)
    status: Optional[Status] = None         # manual override
    valid_for: Optional[str] = None         # manual override of the generated text
