"""
documents.py
~~~~~~~~~~~~
PDF and DOCX → Page.

Header / footer stripping is ported from the chatbot
(imiChat/src/processing/document_loader.py), with one important change:
text is kept PER PAGE, so every PDF chunk carries its page number
(`page=N`, link `file.pdf#page=N`) — the old pipeline joined all pages
and lost them.
"""

from __future__ import annotations

import io
import re
from collections import Counter
from pathlib import PurePosixPath
from urllib.parse import unquote, urlsplit

import pdfplumber
from docx import Document

from src.processing.schemas import Block, Page, Section
from src.utils.text import normalize_text

_PAGE_NUMBER_RE = re.compile(r"[-–—]?\s*\d+\s*[-–—]?|[Pp]age\s+\d+(\s+of\s+\d+)?|[Ss]trana\s+\d+(\s+od\s+\d+)?")
_BAD_PDF_TITLES = re.compile(r"^(microsoft word|untitled|document|dokument|\s*$)", re.I)


def title_from_location(location: str) -> str:
    """Readable title from a URL or path: last segment, no extension."""
    path = urlsplit(location).path if "://" in location else location.replace("\\", "/")
    name = unquote(PurePosixPath(path).name)
    name = re.sub(r"\.(pdf|docx?)$", "", name, flags=re.I)
    name = re.sub(r"[_]+", " ", name)
    return normalize_text(name) or location


# --------------------------------------------------------------------------- #
#  PDF                                                                         #
# --------------------------------------------------------------------------- #

class PdfParser:
    # Fraction of page height reserved for header / footer zones
    HEADER_FRACTION: float = 0.08
    FOOTER_FRACTION: float = 0.08
    # A line on this share of pages (or more) is a running header / footer
    REPETITION_THRESHOLD: float = 0.40

    def _detect_repeated_lines(self, pdf) -> set[str]:
        page_line_sets: list[set[str]] = []
        for page in pdf.pages:
            raw = page.extract_text() or ""
            page_line_sets.append({line.strip() for line in raw.splitlines() if line.strip()})
        if not page_line_sets:
            return set()

        counter: Counter = Counter()
        for line_set in page_line_sets:
            counter.update(line_set)
        min_count = max(2, int(len(page_line_sets) * self.REPETITION_THRESHOLD))
        return {line for line, count in counter.items() if count >= min_count}

    def _extract_page_text(self, page, repeated_lines: set[str]) -> str:
        h, w = float(page.height), float(page.width)
        body = page.crop(
            (0, h * self.HEADER_FRACTION, w, h - h * self.FOOTER_FRACTION),
            relative=False,
        )
        lines: list[str] = []
        for line in (body.extract_text() or "").splitlines():
            stripped = line.strip()
            if stripped and stripped not in repeated_lines and not _PAGE_NUMBER_RE.fullmatch(stripped):
                lines.append(stripped)
        return "\n".join(lines)

    def parse(self, data: bytes, url: str, display: str) -> Page:
        sections: list[Section] = []
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            meta_title = str((pdf.metadata or {}).get("Title") or "")
            title = (
                normalize_text(meta_title)
                if meta_title and not _BAD_PDF_TITLES.match(meta_title)
                else title_from_location(display)
            )
            repeated = self._detect_repeated_lines(pdf)
            for number, page in enumerate(pdf.pages, start=1):
                text = normalize_text(self._extract_page_text(page, repeated))
                if not text:
                    continue
                sections.append(Section(
                    section_id=f"page-{number}",
                    dom_path=f"page-{number}",
                    label=f"strana {number}",
                    breadcrumb=[title, f"strana {number}"],
                    page=number,
                    blocks=[Block(kind="text", text=text)],
                ))

        return Page(
            url=url,
            display_url=display,
            doc_type="pdf",
            title=title,
            sections=sections,
            flags=[] if sections else ["no_text"],
        )


# --------------------------------------------------------------------------- #
#  DOCX                                                                        #
# --------------------------------------------------------------------------- #

class DocxParser:
    @staticmethod
    def _header_footer_texts(doc) -> set[str]:
        texts: set[str] = set()
        for section in doc.sections:
            for part in (
                section.header, section.footer,
                section.even_page_header, section.even_page_footer,
                section.first_page_header, section.first_page_footer,
            ):
                if part is None:
                    continue
                for para in part.paragraphs:
                    if para.text.strip():
                        texts.add(para.text.strip())
        return texts

    def parse(self, data: bytes, url: str, display: str) -> Page:
        doc = Document(io.BytesIO(data))
        title = normalize_text(doc.core_properties.title or "") or title_from_location(display)
        skip = self._header_footer_texts(doc)

        section = Section(section_id="document", dom_path="document", label=title, breadcrumb=[title])
        heading_path: list[str] = []
        lines: list[str] = []

        def flush() -> None:
            text = normalize_text("\n".join(lines))
            if text:
                section.blocks.append(Block(kind="text", text=text, heading_path=list(heading_path)))
            lines.clear()

        for item in doc.iter_inner_content():
            if hasattr(item, "rows"):                      # table
                flush()
                rows = []
                for row in item.rows:
                    cells = list(dict.fromkeys(normalize_text(c.text) for c in row.cells if c.text.strip()))
                    if cells:
                        rows.append(" | ".join(cells))
                if rows:
                    section.blocks.append(Block(kind="table", text="\n".join(rows), rows=rows,
                                                heading_path=list(heading_path)))
                continue
            text = item.text.strip()
            if not text or text in skip:
                continue
            style = (item.style.name if item.style is not None else "") or ""
            if style.lower().startswith(("heading", "naslov", "title")):
                flush()
                heading_path[:] = [normalize_text(text)]
            lines.append(text)
        flush()

        return Page(
            url=url,
            display_url=display,
            doc_type="docx",
            title=title,
            sections=[section] if section.blocks else [],
            flags=[] if section.blocks else ["no_text"],
        )
