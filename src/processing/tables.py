"""
tables.py
~~~~~~~~~
HTML table → self-contained text rows.

The IMI course tables use rowspan / colspan heavily (the semester cell
spans 5–6 rows, "Fond časova" spans p / v / don).  Flattening them with
get_text() produced lines like "1. 1. 25.IN1073 TM Osnovi programiranja
2 2 0 6" — numbers without meaning.  Here the table is expanded into a
grid first, header rows are detected per caption segment, and each data
row is rendered as

    I godina — Semestar: 1. | Šifra: 25.IN1073 | Tip: TM |
               Predmet: Osnovi programiranja | Fond časova predavanja: 2 |
               Fond časova vežbe: 2 | Fond časova DON: 0 | ESPB: 6

so every row can be retrieved (and understood) on its own.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from bs4 import Tag

from src.utils.text import normalize_text

_NUMERIC_RE = re.compile(r"^[\d\s.,()/+\-–—*%]+$")
_LAYOUT_CELL_CHARS = 250          # a cell this long means "layout table"


@dataclass
class TableRender:
    kind: str                     # "table" | "text" (layout tables)
    rows: list[str]
    overview: list[str] = field(default_factory=list)   # one compact line per caption


_SUM_LABELS = {"zbir", "ukupno", "svega", "suma"}


def _course_overview(caption: str, names: list[str], rows: list[list[str]]) -> Optional[str]:
    """'I godina — predmeti: Osnovi programiranja; Teorijske osnove informatike; …'
    — lets listing questions ("koji predmeti su na prvoj godini?") hit one
    chunk.  Names only: lists full of "(6 ESPB)" became retrieval hubs that
    outranked the actual course rows for every ESPB question."""
    course = next(
        (j for j, n in enumerate(names) if n and (n.casefold().startswith("predmet") or "predmeti" in n.casefold())),
        None,
    )
    if course is None:
        return None
    items = []
    for row in rows:
        name = row[course]
        if name and name.casefold().strip(" .:") not in _SUM_LABELS and name not in items:
            items.append(name)
    if len(items) < 2:
        return None
    return f"{caption + ' — ' if caption else ''}predmeti: " + "; ".join(items)


# --------------------------------------------------------------------------- #
#  Grid expansion                                                              #
# --------------------------------------------------------------------------- #

def _int_attr(tag: Tag, name: str) -> int:
    try:
        return max(1, min(int(str(tag.get(name, "1")).strip() or 1), 50))
    except ValueError:
        return 1


def _own_rows(table: Tag) -> list[Tag]:
    """<tr> elements of this table only (not of nested tables)."""
    return [tr for tr in table.find_all("tr") if tr.find_parent("table") is table]


def expand_grid(table: Tag) -> tuple[list[list[str]], list[list[int]]]:
    """
    Expand rowspan / colspan so every row has one value per column.

    Returns (texts, origins): origins[r][c] identifies the source cell, so
    a colspan value is rendered once while genuinely equal neighbours
    (e.g. "predavanja: 2 | vežbe: 2") are kept.
    """
    texts: list[list[str]] = []
    origins: list[list[int]] = []
    spans: dict[tuple[int, int], tuple[str, int]] = {}
    next_origin = 0

    for r, tr in enumerate(_own_rows(table)):
        row: list[str] = []
        row_origins: list[int] = []
        c = 0

        def take_spans() -> None:
            nonlocal c
            while (r, c) in spans:
                text, origin = spans.pop((r, c))
                row.append(text)
                row_origins.append(origin)
                c += 1

        for td in tr.find_all(["td", "th"], recursive=False):
            take_spans()
            text = normalize_text(td.get_text(" ", strip=True))
            rowspan, colspan = _int_attr(td, "rowspan"), _int_attr(td, "colspan")
            origin, next_origin = next_origin, next_origin + 1
            for i in range(colspan):
                row.append(text)
                row_origins.append(origin)
                for k in range(1, rowspan):
                    spans[(r + k, c + i)] = (text, origin)
            c += colspan
        take_spans()

        # spans further right, separated by a gap (malformed tables)
        for col in sorted(cc for (rr, cc) in spans if rr == r):
            while c < col:
                row.append("")
                row_origins.append(-1)
                c += 1
            text, origin = spans.pop((r, col))
            row.append(text)
            row_origins.append(origin)
            c += 1
        texts.append(row)
        origins.append(row_origins)

    width = max((len(row) for row in texts), default=0)
    for row, row_origins in zip(texts, origins):
        row.extend([""] * (width - len(row)))
        row_origins.extend([-1] * (width - len(row_origins)))
    return texts, origins


# --------------------------------------------------------------------------- #
#  Row classification                                                          #
# --------------------------------------------------------------------------- #

def _distinct(row: list[str]) -> list[str]:
    return list(dict.fromkeys(v for v in row if v))


def _is_caption(row: list[str], width: int) -> bool:
    """A single value spanning (most of) the row, e.g. "I godina"."""
    values = _distinct(row)
    return width > 2 and len(values) == 1 and row.count(values[0]) >= max(2, width // 2)


def _is_numeric_row(row: list[str]) -> bool:
    return any(v and any(ch.isdigit() for ch in v) and _NUMERIC_RE.match(v) for v in row)


def _is_layout_table(grid: list[list[str]]) -> bool:
    width = len(grid[0]) if grid else 0
    longest = max((len(v) for row in grid for v in row), default=0)
    return width <= 1 or longest >= _LAYOUT_CELL_CHARS


# --------------------------------------------------------------------------- #
#  Rendering                                                                   #
# --------------------------------------------------------------------------- #

def _column_names(
    grid: list[list[str]], header_rows: list[int], glossary: dict[str, str]
) -> list[str]:
    width = len(grid[0])
    names: list[str] = []
    for j in range(width):
        parts: list[str] = []
        for r in header_rows:
            value = grid[r][j]
            if value and (not parts or parts[-1] != glossary.get(value, value)):
                parts.append(glossary.get(value, value))
        names.append(" ".join(parts))
    return names


def _render_row(
    row: list[str],
    row_origins: list[int],
    names: Optional[list[str]],
    drop_columns: tuple[str, ...],
) -> str:
    # column names covered by each source cell (colspan)
    covered: dict[int, set[str]] = {}
    for j, origin in enumerate(row_origins):
        covered.setdefault(origin, set()).add(names[j] if names else "")

    pairs: list[str] = []
    rendered: set[int] = set()
    for j, value in enumerate(row):
        origin = row_origins[j]
        name = names[j] if names else ""
        if not value or origin in rendered:
            continue
        rendered.add(origin)
        if name in drop_columns and len(covered[origin]) == 1:
            continue
        if len(covered[origin]) > 1:          # e.g. "Zbir" spanning S..Predmet
            name = ""
        if name and name != value:
            pairs.append(f"{name}: {value}")
        elif not name:
            pairs.append(value)
    return " | ".join(pairs)


def render_table(
    table: Tag,
    glossary: Optional[dict[str, str]] = None,
    drop_columns: tuple[str, ...] = (),
) -> TableRender:
    glossary = glossary or {}
    texts, origins = expand_grid(table)
    keep = [i for i, row in enumerate(texts) if any(row)]
    grid = [texts[i] for i in keep]
    grid_origins = [origins[i] for i in keep]
    if not grid:
        return TableRender(kind="table", rows=[])

    if _is_layout_table(grid):
        rows = []
        for row in grid:
            rows.extend(v for v in _distinct(row) if v not in rows)
        return TableRender(kind="text", rows=rows)

    width = len(grid[0])
    if width == 2:
        # two columns: header detection is unreliable (key/value lists vs.
        # side-by-side lists) — keep the row association, no names
        rows = [" | ".join(_distinct(row)) for row in grid]
        return TableRender(kind="table", rows=[r for r in rows if r])

    # ── split into segments at caption rows ───────────────────────────── #
    segments: list[tuple[str, list[int]]] = []
    caption, current = "", []
    for r, row in enumerate(grid):
        if _is_caption(row, width):
            if current:
                segments.append((caption, current))
            caption, current = _distinct(row)[0], []
        else:
            current.append(r)
    if current:
        segments.append((caption, current))

    # ── per segment: detect header rows, render data rows ─────────────── #
    lines: list[str] = []
    overview: list[str] = []
    inherited: Optional[list[str]] = None
    for caption, rows in segments:
        first_numeric = next((i for i, r in enumerate(rows) if _is_numeric_row(grid[r])), None)
        if first_numeric is None:
            header_rows = rows[:1] if len(rows) >= 2 and len(_distinct(grid[rows[0]])) >= 2 else []
        else:
            header_rows = rows[:first_numeric][-3:]

        if header_rows:
            names = _column_names(grid, header_rows, glossary)
            inherited = names
        else:
            names = inherited

        data_rows = [r for r in rows if r not in header_rows]
        for r in data_rows:
            text = _render_row(grid[r], grid_origins[r], names, drop_columns)
            if text:
                lines.append(f"{caption} — {text}" if caption else text)
        if names:
            line = _course_overview(caption, names, [grid[r] for r in data_rows])
            if line:
                overview.append(line)

    if not lines and segments:
        # caption-only table — keep the captions as text
        lines = [c for c, _ in segments if c]
    return TableRender(kind="table", rows=lines, overview=overview)
