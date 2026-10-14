"""
prompts.py
~~~~~~~~~~
Prompt templates for DeepSeek.

Fixed instructions come first and page context before chunk text, so
DeepSeek's automatic prefix (KV) cache is hit across the batches of one
page.  Bump PROMPT_VERSION whenever a prompt changes — it invalidates
the LLM cache and the per-page ingest state.
"""

from __future__ import annotations

from src.processing.schemas import PAGE_TYPES, Page, PageAnalysis, Section, VersionInfo
from src.utils.text import truncate

PROMPT_VERSION = "2026-10-11.1"

EXCERPT_CHARS = 500
PDF_OUTLINE_PAGES = 12


# --------------------------------------------------------------------------- #
#  Call A — page analysis (+ registry drafts)                                  #
# --------------------------------------------------------------------------- #

ANALYZE_SYSTEM = f"""You are a data-quality assistant for the website of the Institute of Mathematics and Informatics (IMI), Faculty of Science, University of Kragujevac, Serbia. A scraper extracted ONE web page or document for a retrieval (RAG) knowledge base used by a student chatbot. You receive the page outline and return a json object describing the page.

Rules:
- Use ONLY information present in the input. Never invent names, years, modules or dates. Use null when unsure.
- Write all free text in Serbian, Latin script.
- "page_type": one of {", ".join(PAGE_TYPES)}.
- "page_summary": 1-2 sentences: what a student can learn from this page.
- "published_date": publication date (YYYY-MM-DD) if the page is a dated news item or announcement and the date is stated; otherwise null.
- "relevant": false only if the page has no useful content for students (login form, empty page, pure navigation).
- "registry_draft": ONLY for sections listed under UNMAPPED PANELS that describe one specific VERSION of a study program (osnovne / master / doktorske studije). One entry per program version; put in "panels" ALL unmapped panel ids that belong to that version (its main tab and its module sub-panels). Use [] when there are no such sections.
  - "program": e.g. "Informatika" or "Matematika".
  - "level": "osnovne" | "master" | "doktorske".
  - "version_label": short Serbian label, e.g. "OAS Informatika — akreditacija 2016".
  - "accreditation_year": year the version was accredited if stated ("akreditovane 2016" -> 2016), else null.
  - "enrollment_from": first enrollment year if stated ("Upis 2026/27" -> 2026), else null.
  - "modules": module / track names ("modul", "smer") of THAT version exactly as written; [] if none.
  - "evidence": 1-3 SHORT verbatim quotes (max 15 words each) copied character-for-character from the input that justify the year and the modules.

Return exactly one json object of this shape:
{{"page_type": "studijski_program", "page_summary": "...", "published_date": null, "relevant": true, "registry_draft": [{{"program": "Informatika", "level": "osnovne", "version_label": "OAS Informatika — akreditacija 2016", "accreditation_year": 2016, "enrollment_from": null, "modules": [], "panels": ["panel8inf"], "evidence": ["Osnovne akademske studije akreditovane 2016"]}}]}}
"""


def _excerpt(section: Section) -> str:
    rows = [r for b in section.blocks for r in b.rows]
    if rows and not any(b.kind == "text" for b in section.blocks):
        return truncate(" / ".join(rows[:3]), EXCERPT_CHARS)
    return truncate(section.text.replace("\n", " "), EXCERPT_CHARS)


def analyze_user(
    page: Page,
    sections: list[Section],
    today: str,
    unmapped: list[str],
    mapped_versions: list[str],
) -> str:
    lines = [
        f"TODAY: {today}",
        f"URL: {page.url}",
        f"TITLE: {page.title}",
        f"DOCUMENT TYPE: {page.doc_type}",
        f"UNMAPPED PANELS: {', '.join(unmapped) if unmapped else 'none'}",
    ]
    if mapped_versions:
        lines.append("ALREADY MAPPED VERSIONS (do not draft these again):")
        lines.extend(f"- {v}" for v in mapped_versions)
    if page.published_hint:
        lines.append(f"DATE FOUND BY EXTRACTOR: {page.published_hint}")
    lines.append("OUTLINE:")
    outline = sections if page.doc_type == "web" else sections[:PDF_OUTLINE_PAGES]
    for section in outline:
        if not section.blocks:
            continue
        lines.append(f"## section: {section.section_id}")
        lines.append(f"path: {' › '.join(section.breadcrumb)}")
        if section.hidden_by_default:
            lines.append("hidden_by_default: yes (opened by clicking a tab or link)")
        lines.append(f"excerpt: {_excerpt(section)}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
#  Call B — chunk enrichment                                                   #
# --------------------------------------------------------------------------- #

ENRICH_SYSTEM = """You enrich text chunks for a Serbian university RAG knowledge base (IMI, PMF Kragujevac). For EACH input chunk return an object with:
- "id": the chunk id, unchanged.
- "context": 1-2 sentences in Serbian (Latin script) that situate the chunk within its page: which study program / version / topic it belongs to and what it covers. Use only the page and section information and the chunk itself.
- "questions": 2-4 short, natural questions in Serbian (Latin script) that a student could ask and that THIS chunk answers. Never ask questions the chunk cannot answer.
- "keywords": up to 8 keywords or key phrases in Serbian (Latin script).
- "quality": "ok" | "boilerplate" (menus, navigation, cookie or login text, repeated footers, link lists without information) | "garbled" (broken or unreadable extraction) | "low_value" (no informational content for students).
- "issues": short notes in Serbian about data problems a human should check, e.g. "tabela bez zaglavlja", "pominje rok iz 2023 — moguće zastarelo", "nedovršena rečenica", "kontradikcija unutar teksta". Use [] if none.
- "outdated": true ONLY when PROGRAM VERSION says the chunk is not tied to a program version AND the chunk states, as currently valid, something that contradicts the PROGRAM FACTS (e.g. it says a program has 3 modules while the program currently enrolling has 2). Text that clearly describes an older program, or that the facts do not cover, is NOT outdated. Default false.
- "outdated_reason": when outdated is true, one short Serbian sentence naming the contradiction; otherwise "".
Never add facts that are not in the input.

Return exactly one json object: {"chunks": [{"id": "c0", "context": "...", "questions": ["..."], "keywords": ["..."], "quality": "ok", "issues": [], "outdated": false, "outdated_reason": ""}]} with one entry per input chunk, in the same order.
"""


def enrich_user(
    page: Page,
    analysis: PageAnalysis,
    path: list[str],
    version: VersionInfo | None,
    chunks: list[tuple[str, str]],
    facts: list[str] | None = None,
) -> str:
    if version is not None:
        version_line = f"{version.version_label} — {version.status_label}"
        if version.valid_for:
            version_line += f" ({version.valid_for})"
    else:
        version_line = "nije vezano za određenu verziju studijskog programa"
    lines = []
    if facts:
        lines.append("PROGRAM FACTS (reviewed, authoritative):")
        lines.extend(facts)
        lines.append("")
    lines += [
        f"PAGE: {page.title} — {page.url}",
        f"PAGE TYPE: {analysis.page_type}",
        f"PAGE SUMMARY: {analysis.page_summary or '-'}",
        f"SECTION: {' › '.join(path)}",
        f"PROGRAM VERSION: {version_line}",
        "CHUNKS:",
    ]
    for local_id, text in chunks:
        lines.append(f'<chunk id="{local_id}">\n{text}\n</chunk>')
    return "\n".join(lines)
