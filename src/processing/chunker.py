"""
chunker.py
~~~~~~~~~~
Section blocks → chunks that fit the embedding model.

  • Sizes are measured in tokens of the E5 model's own tokenizer
    (multilingual-e5-large silently truncates input after 512 tokens).
  • Prose is packed sentence by sentence with a small overlap.
  • Tables are packed row by row (rows are self-contained, no overlap).
  • A short heading-only block (e.g. "Izborni modul: Nauka o podacima"
    right before its table) is not a chunk of its own — it already is
    part of the next block's heading path, which goes into the header.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.processing.schemas import Block, Section
from src.utils.text import fold, join_units, split_units

logger = logging.getLogger(__name__)

_CHARS_PER_TOKEN = 3.2      # fallback estimate for Serbian Latin text


class TokenCounter:
    """Counts tokens with the embedding model's tokenizer (only the small
    tokenizer files are downloaded, not the model).  Falls back to a
    character estimate when the tokenizer is unavailable (offline)."""

    def __init__(self, model_name: str | None = None):
        self._tokenizer = None
        if model_name:
            try:
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(model_name)
                self._tokenizer.model_max_length = 10 ** 9   # we only count
            except Exception as exc:
                logger.warning(
                    "Tokenizer for %s unavailable (%s) — using a %.1f chars/token estimate",
                    model_name, exc, _CHARS_PER_TOKEN,
                )

    @property
    def exact(self) -> bool:
        return self._tokenizer is not None

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self._tokenizer is None:
            return max(1, int(len(text) / _CHARS_PER_TOKEN) + 1)
        return len(self._tokenizer(text, add_special_tokens=False)["input_ids"])

    def count_model_input(self, text: str) -> int:
        """Tokens the model actually sees: text + <s> and </s>."""
        return self.count(text) + 2


@dataclass
class RawChunk:
    section: Section
    text: str
    kind: str                       # "text" | "table"
    index: int                      # position within the section
    heading_path: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
#  Packing                                                                     #
# --------------------------------------------------------------------------- #

def _split_oversized(text: str, counter: TokenCounter, max_tokens: int) -> list[str]:
    """Split one unit that is too long on its own, word by word."""
    pieces, current = [], []
    for word in text.split():
        candidate = " ".join(current + [word])
        if current and counter.count(candidate) > max_tokens:
            pieces.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        pieces.append(" ".join(current))
    return pieces


def pack_units(
    units: list[tuple[str, bool]],
    counter: TokenCounter,
    max_tokens: int,
    overlap_tokens: int = 0,
) -> list[str]:
    chunks: list[str] = []
    current: list[tuple[str, bool]] = []
    current_tokens = 0
    for unit, new_line in units:
        tokens = counter.count(unit)
        if tokens > max_tokens:
            if current:
                chunks.append(join_units(current))
                current, current_tokens = [], 0
            chunks.extend(_split_oversized(unit, counter, max_tokens))
            continue
        if current and current_tokens + tokens > max_tokens:
            chunks.append(join_units(current))
            carry: list[tuple[str, bool]] = []
            carry_tokens = 0
            for prev_unit, prev_new_line in reversed(current):
                prev_tokens = counter.count(prev_unit)
                if carry_tokens + prev_tokens > overlap_tokens:
                    break
                carry.insert(0, (prev_unit, prev_new_line))
                carry_tokens += prev_tokens
            if carry_tokens + tokens > max_tokens:
                carry, carry_tokens = [], 0
            current, current_tokens = carry, carry_tokens
        current.append((unit, new_line))
        current_tokens += tokens
    if current:
        chunks.append(join_units(current))
    return chunks


def pack_overview(line: str, counter: TokenCounter, max_tokens: int) -> list[str]:
    """Split 'I godina — predmeti: A; B; …' by items, repeating the prefix."""
    prefix, sep, rest = line.partition("predmeti: ")
    if not sep or counter.count(line) <= max_tokens:
        return [line]
    parts, current = [], []
    for item in rest.split("; "):
        candidate = f"{prefix}{sep}{'; '.join(current + [item])}"
        if current and counter.count(candidate) > max_tokens:
            parts.append(f"{prefix}{sep}{'; '.join(current)}")
            current = [item]
        else:
            current.append(item)
    if current:
        parts.append(f"{prefix}{sep}{'; '.join(current)}")
    return parts


def _is_heading_only(block: Block, next_block: Block | None) -> bool:
    """A block that is nothing but a heading — its own, or the heading of
    the next block (it is in that block's header anyway)."""
    if block.kind != "text" or len(block.text) > 120:
        return False
    folded = fold(block.text)
    if block.heading_path and fold(block.heading_path[-1]) == folded:
        return True
    return next_block is not None and any(fold(h) == folded for h in next_block.heading_path)


def chunk_section(
    section: Section,
    counter: TokenCounter,
    max_tokens: int,
    overlap_tokens: int,
    min_chars: int = 30,
    table_max_tokens: int | None = None,
) -> list[RawChunk]:
    """
    Prose → ≤ max_tokens with overlap.  Table rows → ≤ table_max_tokens
    (1–2 rows: a question about one course must hit that course's row,
    not a 6-course mix).  Each table caption also gets one compact
    "predmeti: …" overview chunk for listing questions.
    """
    table_max_tokens = table_max_tokens or max_tokens
    chunks: list[RawChunk] = []
    blocks = section.blocks

    def add(text: str, kind: str, heading_path: list[str]) -> None:
        if len(text.strip()) >= min_chars:
            chunks.append(RawChunk(section=section, text=text, kind=kind,
                                   index=len(chunks), heading_path=list(heading_path)))

    for i, block in enumerate(blocks):
        next_block = blocks[i + 1] if i + 1 < len(blocks) else None
        if _is_heading_only(block, next_block):
            continue
        if block.kind == "table" and block.rows:
            for text in pack_units([(row, True) for row in block.rows], counter, table_max_tokens, 0):
                add(text, "table", block.heading_path)
            for line in block.overview:
                for text in pack_overview(line, counter, max_tokens):
                    add(text, "table_overview", block.heading_path)
        else:
            for text in pack_units(split_units(block.text), counter, max_tokens, overlap_tokens):
                add(text, "text", block.heading_path)
    return chunks


# --------------------------------------------------------------------------- #
#  Header                                                                      #
# --------------------------------------------------------------------------- #

def location_path(breadcrumb: list[str], heading_path: list[str]) -> list[str]:
    """Breadcrumb + headings, without consecutive repeats."""
    path: list[str] = []
    for item in breadcrumb + heading_path:
        item = item.strip()
        if item and (not path or fold(path[-1]) != fold(item)):
            path.append(item)
    return path


def build_header(path: list[str], version_label: str = "", status_label: str = "",
                 extra: str = "") -> str:
    parts = [" › ".join(path)]
    if version_label:
        parts.append(version_label)
    if status_label:
        parts.append(status_label)
    if extra:
        parts.append(extra)
    return "[" + " | ".join(p for p in parts if p) + "]"
