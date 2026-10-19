"""
factcheck.py
~~~~~~~~~~~~
Deterministic guards around AI curation.

The model is allowed to rewrite prose, but it may not invent numbers, course
codes, years or module names.  Everything here works on the *extracted source*
and on the model's own output — no LLM involvement — so a hallucinated deadline
or credit value is caught even when the JSON is perfectly valid.

Checks:
  * numeric_facts()             — digits, decimals, course codes and written
                                  Serbian numbers ("dva" counts as 2) in a text
  * unsupported_facts()         — sensitive tokens in curated content that do
                                  not occur anywhere in the source corpus
  * missing_registry_modules()  — registry module names present in the source
                                  section but dropped from every curated chunk
                                  of that section (reported, not silently lost)
  * is_rewritten()              — whether the curated text is not a verbatim
                                  slice of the source (reporting statistic)
"""

from __future__ import annotations

import re
from typing import Iterable

from src.utils.text import fold, normalize_text

# Serbian number words, already folded (no diacritics) so they match fold().
NUMBER_WORDS: dict[str, int] = {
    "nula": 0,
    "jedan": 1, "jedna": 1, "jedno": 1,
    "dva": 2, "dve": 2,
    "tri": 3,
    "cetiri": 4,
    "pet": 5,
    "sest": 6,
    "sedam": 7,
    "osam": 8,
    "devet": 9,
    "deset": 10,
    "jedanaest": 11,
    "dvanaest": 12,
    "trinaest": 13,
    "cetrnaest": 14,
    "petnaest": 15,
    "sesnaest": 16,
    "sedamnaest": 17,
    "osamnaest": 18,
    "devetnaest": 19,
    "dvadeset": 20,
}

_NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")
# 25.IN1073, 19.FIZ060, 15.MAT123 — the IMI course-code shape.
_CODE_RE = re.compile(r"\b\d{1,2}\.[A-Za-zČĆŠŽĐčćšžđ]{2,}\d{2,}\b")

# Folded Serbian month names, so "15. septembra 2026" and "15.09.2026" can be
# compared as the same date instead of a rewritten format being rejected.
MONTH_WORDS: dict[str, int] = {
    "januar": 1, "januara": 1,
    "februar": 2, "februara": 2,
    "mart": 3, "marta": 3,
    "april": 4, "aprila": 4,
    "maj": 5, "maja": 5,
    "jun": 6, "juna": 6,
    "jul": 7, "jula": 7,
    "avgust": 8, "avgusta": 8,
    "septembar": 9, "septembra": 9,
    "oktobar": 10, "oktobra": 10,
    "novembar": 11, "novembra": 11,
    "decembar": 12, "decembra": 12,
}


def numeric_facts(text: str) -> set[str]:
    """All sensitive numeric tokens in `text`, as comparable strings.

    Decimals are normalized to a dot, dates are covered by their parts, course
    codes are also kept whole, and written numbers ("dva") map to digits so
    "dva modula" in the source supports "2 modula" in curated prose.
    """
    if not text:
        return set()

    normalized = normalize_text(text)
    facts: set[str] = set()
    for match in _NUM_RE.finditer(normalized):
        token = match.group(0).replace(",", ".")
        facts.add(token)
        # Date/decimal components: "15.09" also counts as 15 and 09, and the
        # leading-zero form matches a month written as a word.
        for part in token.split("."):
            if part:
                facts.add(part)
                if part.isdigit():
                    facts.add(str(int(part)))
    for match in _CODE_RE.finditer(normalized):
        facts.add(fold(match.group(0)))

    folded = fold(text)
    for word, value in NUMBER_WORDS.items():
        if re.search(rf"(?<!\w){re.escape(word)}(?!\w)", folded):
            facts.add(str(value))
    for word, month in MONTH_WORDS.items():
        if re.search(rf"(?<!\w){re.escape(word)}(?!\w)", folded):
            facts.add(f"{month:02d}")
            facts.add(str(month))
    return facts


def unsupported_facts(content: str, source: str) -> list[str]:
    """Numeric/code tokens in `content` that do not occur in `source`.

    Deliberately conservative: the caller decides whether an unsupported token
    rejects the chunk or is only reported.  A token is considered supported if
    it appears anywhere in the (page-level) source corpus, so facts merged in
    from a neighbouring section are not penalized.  A dotted date/decimal is
    also supported when all of its components occur in the source, so
    "15.09.2026" and "15. septembra 2026" are treated as the same date.
    """
    content_facts = numeric_facts(content)
    source_facts = numeric_facts(source)
    unsupported: list[str] = []
    for token in sorted(content_facts):
        if token in source_facts:
            continue
        parts = [part for part in re.split(r"[./-]", token) if part]
        if len(parts) > 1 and all(part in source_facts for part in parts):
            continue
        unsupported.append(token)
    return unsupported


def missing_registry_modules(
    modules: Iterable[str], source_text: str, curated_texts: Iterable[str]
) -> list[str]:
    """Registry modules that the source section mentions but no curated chunk
    from that section keeps.  This is a data-completeness warning, not a hard
    rejection: the model may legitimately fold a module into a comparison."""
    folded_source = fold(source_text)
    folded_curated = [fold(t) for t in curated_texts]
    missing: list[str] = []
    for module in modules:
        folded = fold(module)
        if folded and folded in folded_source and not any(folded in text for text in folded_curated):
            missing.append(module)
    return missing


def is_rewritten(content: str, source_text: str) -> bool:
    """True when the curated sentence is not a verbatim slice of the source."""
    return fold(content) not in fold(source_text)
