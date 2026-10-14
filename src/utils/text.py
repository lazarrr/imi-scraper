"""
text.py
~~~~~~~
Text helpers shared by the whole pipeline:

  • normalize_text()  — NFKC, Cyrillic → Latin (also fixes Cyrillic
                        homoglyphs inside Latin words, e.g. "mаtemаtike"),
                        whitespace cleanup.  Applied BEFORE hashing,
                        dedup and embedding so equal text hashes equally.
  • fold() / TextIndex — aggressive normalization for fuzzy "does this
                        quote really appear on the page?" checks.
  • split_units()     — line + sentence splitting for chunk packing.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

import cyrtranslit

CYRILLIC_RE = re.compile(r"[Ѐ-ӿ]")

# Sentence boundary: terminal punctuation followed by whitespace and an
# upper-case letter / digit / opening quote or bullet.
_SENTENCE_RE = re.compile(r"(?<=[.!?…])\s+(?=[A-ZČĆŠŽĐ0-9\"„(•\-])")

_DASHES_RE = re.compile(r"[‐-―−]")
_QUOTES_RE = re.compile(r"[\"'`´‘’‚‛“”„‟«»]")


# --------------------------------------------------------------------------- #
#  Normalization                                                               #
# --------------------------------------------------------------------------- #

def has_cyrillic(text: str) -> bool:
    return bool(CYRILLIC_RE.search(text or ""))


def to_latin(text: str) -> str:
    """Serbian Cyrillic → Latin.  Works per character, so mixed-script
    words (Latin text with Cyrillic а/ј/К homoglyphs) are fixed too."""
    if not has_cyrillic(text):
        return text
    return cyrtranslit.to_latin(text, "sr")


def normalize_text(text: str) -> str:
    """Canonical form of every stored / hashed / embedded text."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = (
        text.replace(" ", " ")
        .replace("​", "")
        .replace("﻿", "")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )
    text = to_latin(text)
    lines = [re.sub(r"[ \t\f\v]+", " ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fold(text: str) -> str:
    """Aggressive normalization for matching: Latin script, casefold,
    no diacritics, unified dashes, no quotes / punctuation."""
    t = normalize_text(text).casefold()
    t = t.replace("đ", "dj")
    t = unicodedata.normalize("NFKD", t)
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = _DASHES_RE.sub("-", t)
    t = _QUOTES_RE.sub("", t)
    t = re.sub(r"[^\w\s-]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


class TextIndex:
    """Folded page text + word trigrams for fast fuzzy containment checks."""

    def __init__(self, text: str):
        self.folded = fold(text)
        words = self.folded.split()
        self._trigrams = {" ".join(words[i:i + 3]) for i in range(len(words) - 2)}

    def contains(self, needle: str, min_chars: int = 4, min_coverage: float = 0.8) -> bool:
        """True if `needle` appears in the text (exact after folding, on
        word boundaries), or — for longer quotes — if ≥ `min_coverage`
        of its word trigrams appear (tolerates small extraction diffs)."""
        f = fold(needle)
        if len(f) < min_chars:
            return False
        if re.search(rf"(?<!\w){re.escape(f)}(?!\w)", self.folded):
            return True
        words = f.split()
        if len(words) < 4:
            return False
        grams = [" ".join(words[i:i + 3]) for i in range(len(words) - 2)]
        hits = sum(g in self._trigrams for g in grams)
        return hits / len(grams) >= min_coverage


# --------------------------------------------------------------------------- #
#  Splitting / misc                                                            #
# --------------------------------------------------------------------------- #

def split_units(text: str) -> list[tuple[str, bool]]:
    """Split text into lines, then sentences — the packing units for prose.
    Returns (unit, starts_new_line) so packing can restore the layout."""
    units: list[tuple[str, bool]] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        sentences = [s.strip() for s in _SENTENCE_RE.split(line) if s.strip()]
        units.extend((s, i == 0) for i, s in enumerate(sentences))
    return units


def join_units(units: list[tuple[str, bool]]) -> str:
    out = ""
    for i, (unit, new_line) in enumerate(units):
        out += unit if i == 0 else ("\n" if new_line else " ") + unit
    return out


def slugify(text: str, max_len: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", fold(text)).strip("-")
    return slug[:max_len].strip("-") or "x"


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def truncate(text: str, limit: int) -> str:
    text = text or ""
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"
