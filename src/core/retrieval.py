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


def apply_version_policy(
    query: str,
    documents: list[str],
    metadatas: list[dict],
    distances: list[float],
    k: int,
    include_legacy: bool = False,
) -> tuple[list[str], list[dict], list[float]]:
    keep_legacy = include_legacy or wants_legacy(query)
    docs: list[str] = []
    metas: list[dict] = []
    dists: list[float] = []
    seen: set[str] = set()
    for doc, meta, dist in zip(documents, metadatas, distances):
        meta = meta or {}
        if not keep_legacy and (meta.get("status") == "legacy" or meta.get("outdated") is True):
            continue
        key = meta.get("dedup_key")
        if key:
            if key in seen:
                continue
            seen.add(key)
        docs.append(doc)
        metas.append(meta)
        dists.append(dist)
        if len(docs) >= k:
            break
    return docs, metas, dists
