"""
retrieval.py
~~~~~~~~~~~~
Version-aware result policy — the SAME rules the chatbot applies in
imiChat/src/core/vector_store.py (_apply_version_policy), so `run.py
search` / `run.py eval` show what the chatbot will actually get.

  1. drop chunks of legacy program versions and chunks DeepSeek flagged
     as outdated (they contradict the reviewed current facts), unless the
     question asks for an old program (mentions a year 2000–2018,
     "stari program", "akreditacija 2016" …)
  2. collapse chunks sharing a dedup_key (same text, same version)
  3. keep the best k
"""

from __future__ import annotations

import re

LEGACY_QUERY_RE = re.compile(
    r"\b20(0\d|1[0-8])\b"
    r"|\bstar(i|og|om|im|ih|e|a|u)\s+(studijsk\w+\s+)?(program\w*|akreditacij\w*|plan\w*)"
    r"|\bprethodn\w+\s+akreditacij\w*",
    re.IGNORECASE,
)


def wants_legacy(query: str) -> bool:
    return bool(LEGACY_QUERY_RE.search(query or ""))


# Applied INSIDE the Chroma query: legacy tables are ~half of all chunks and
# would otherwise crowd current content out of the candidate list.
# ($ne also keeps chunks that do not have the key at all.)
CURRENT_ONLY_WHERE = {"$and": [{"status": {"$ne": "legacy"}}, {"outdated": {"$ne": True}}]}


def where_filter(query: str, include_legacy: bool = False):
    return None if include_legacy or wants_legacy(query) else CURRENT_ONLY_WHERE


# Why a candidate did not make the final list — the UI shows these verbatim.
REASON_BELOW_THRESHOLD = "below_threshold"
REASON_LEGACY = "legacy_or_outdated"
REASON_DUPLICATE = "duplicate"
REASON_BEYOND_K = "beyond_k"

REASON_LABELS = {
    REASON_BELOW_THRESHOLD: "Below the similarity threshold",
    REASON_LEGACY: "Legacy / outdated version",
    REASON_DUPLICATE: "Duplicate text (same dedup_key)",
    REASON_BEYOND_K: "Ranked below the top-k cut-off",
}


def _candidate(
    idx: int,
    ids: list[str] | None,
    document: str,
    metadata: dict,
    distance: float,
    reason: str | None = None,
    detail: str = "",
) -> dict:
    return {
        "id": ids[idx] if ids is not None and idx < len(ids) else f"candidate-{idx}",
        "document": document,
        "metadata": metadata or {},
        "distance": float(distance),
        "similarity": 1.0 - float(distance),
        "reason": reason,
        "reason_detail": detail,
    }


def filter_candidates(
    query: str,
    ids: list[str] | None,
    documents: list[str],
    metadatas: list[dict],
    distances: list[float],
    k: int,
    max_distance: float | None = None,
    include_legacy: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Split raw Chroma candidates into (kept, rejected).

    The single implementation of the result policy — `apply_version_policy`
    delegates here, and so do the CLI, the demo script and the retrieval UI,
    so every consumer ranks identically and can explain each rejection.

    Order of the rules (same as the chatbot):
      1. cosine distance above `max_distance` (i.e. similarity below the
         threshold) → dropped
      2. `status == "legacy"` or `outdated is True` → dropped unless the
         question itself asks for an old program
      3. a `dedup_key` already seen in this run → dropped
      4. everything past the k-th survivor → dropped
    """
    keep_legacy = include_legacy or wants_legacy(query)
    kept: list[dict] = []
    rejected: list[dict] = []
    seen: dict[str, str] = {}          # dedup_key -> id of the survivor

    for idx, (document, metadata, distance) in enumerate(zip(documents, metadatas, distances)):
        meta = metadata or {}
        sim = 1.0 - float(distance)

        if max_distance is not None and float(distance) > max_distance:
            rejected.append(_candidate(
                idx, ids, document, meta, distance, REASON_BELOW_THRESHOLD,
                f"similarity {sim:.3f} < {1.0 - max_distance:.3f}",
            ))
            continue
        if not keep_legacy and (meta.get("status") == "legacy" or meta.get("outdated") is True):
            why = "status=legacy" if meta.get("status") == "legacy" else "outdated=true"
            rejected.append(_candidate(
                idx, ids, document, meta, distance, REASON_LEGACY,
                f"{why} (ask about an older program to include it)",
            ))
            continue
        key = meta.get("dedup_key")
        if key and key in seen:
            rejected.append(_candidate(
                idx, ids, document, meta, distance, REASON_DUPLICATE,
                f"same text as {seen[key]}",
            ))
            continue
        if len(kept) >= k:
            rejected.append(_candidate(
                idx, ids, document, meta, distance, REASON_BEYOND_K,
                f"rank {len(kept) + 1} > k={k}",
            ))
            continue
        if key:
            seen[key] = _candidate(idx, ids, document, meta, distance)["id"]
        kept.append(_candidate(idx, ids, document, meta, distance))

    return kept, rejected


def apply_version_policy(
    query: str,
    documents: list[str],
    metadatas: list[dict],
    distances: list[float],
    k: int,
    include_legacy: bool = False,
) -> tuple[list[str], list[dict], list[float]]:
    """Backwards-compatible wrapper around `filter_candidates`."""
    kept, _ = filter_candidates(
        query, None, documents, metadatas, distances, k, include_legacy=include_legacy
    )
    return (
        [h["document"] for h in kept],
        [h["metadata"] for h in kept],
        [h["distance"] for h in kept],
    )
