"""
retrieval_service.py
~~~~~~~~~~~~~~~~~~~~
The single retrieval path used by everything that queries the store: the CLI
(`run.py search` / `run.py eval`), `tools/vector_store_demo.py` and the local
JSON API behind the Flutter UI.

It returns *JSON-ready* dictionaries, including the candidates that were
dropped and **why** they were dropped (`below_threshold`, `legacy_or_outdated`,
`duplicate`, `beyond_k`) — that is what the UI's "rejected" panel renders.

Nothing here re-implements ranking: the rules live in `src.core.retrieval`
(`filter_candidates`) and the embedding/prefix conventions in
`src.core.embeddings` (E5 `query: ` / `passage: `), so the CLI and the UI can
never drift apart.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from src.core.retrieval import REASON_LABELS, filter_candidates, where_filter
from src.utils.text import normalize_text

logger = logging.getLogger(__name__)

# Metadata keys the UI shows as headline fields instead of generic chips.
HEADLINE_KEYS = (
    "page_title", "breadcrumb", "doc_kind", "page_type", "status", "type",
    "version_label", "version_source", "section_id", "filename", "block_type",
)
# ...everything else is rendered as a chip / detail row, in this order first.
CHIP_KEY_ORDER = (
    "curation_status", "curated", "rewritten", "keywords", "quality", "page",
    "generated", "ingested_at", "llm_model", "prompt_version", "embed_truncated",
    "ui_hidden", "hidden_by_default", "content_hash", "dedup_key",
)
# Purely mechanical keys the UI can hide behind "show internal metadata".
INTERNAL_KEYS = frozenset({"dedup_key", "content_hash", "source", "page_url", "url"})


class RetrievalService:
    """Store + embedder + result policy, with timing and rejection reasons.

    The embedder is optional so the store can be inspected (health, stats,
    collection listing) without paying for a 2 GB model load; `search()` and
    `ensure_embedder()` are the only paths that need it.
    """

    def __init__(
        self,
        store,
        embedder=None,
        *,
        model_name: str,
        threshold: float,
        fetch_k: int,
    ):
        self.store = store
        self.embedder = embedder
        self.model_name = model_name
        self.threshold = float(threshold)
        self.fetch_k = int(fetch_k)

    # ------------------------------------------------------------------ #
    #  Construction                                                        #
    # ------------------------------------------------------------------ #

    @property
    def model_loaded(self) -> bool:
        return self.embedder is not None

    def ensure_embedder(self, *, device: str = "cpu", cache_path: Path | None = None):
        """Load the E5 model on first use (~2 GB, tens of seconds on CPU)."""
        if self.embedder is not None:
            return self.embedder
        from src.core.embeddings import E5Embedder

        logger.info("Loading embedding model %s on %s …", self.model_name, device)
        self.embedder = E5Embedder(self.model_name, device=device, cache_path=cache_path)
        return self.embedder

    @classmethod
    def build(
        cls,
        settings,
        *,
        db_path: str | Path | None = None,
        collection: str | None = None,
        embedder=None,
        embedder_cache: Path | None = None,
        with_model: bool = True,
    ) -> "RetrievalService":
        """Open the store and — unless `with_model=False` — load the E5 model.

        `embedder` is injectable so tests can pass a fake and skip the 2 GB
        model download.
        """
        from src.core.vector_store import ChromaStore

        path = settings.path(str(db_path or settings.VECTOR_DB_PATH))
        store = ChromaStore(path, collection or settings.CHROMA_COLLECTION)
        service = cls(
            store,
            embedder,
            model_name=settings.EMBEDDING_MODEL,
            threshold=settings.SIMILARITY_THRESHOLD,
            fetch_k=settings.RETRIEVAL_FETCH_K,
        )
        if embedder is not None:
            service.model_name = getattr(embedder, "model_name", settings.EMBEDDING_MODEL)
        elif with_model:
            service.ensure_embedder(
                device=settings.EMBEDDING_DEVICE, cache_path=embedder_cache
            )
        return service

    # ------------------------------------------------------------------ #
    #  Introspection                                                       #
    # ------------------------------------------------------------------ #

    def describe(self) -> dict[str, Any]:
        """Where the data lives and how it was built — shown in the UI header."""
        return {
            "path": str(self.store.path),
            "collection": self.store.collection_name,
            "count": self.store.count(),
            "embedding_model": self.model_name,
            "distance": "cosine",
            "query_prefix": getattr(self.embedder, "QUERY_PREFIX", "query: "),
            "passage_prefix": getattr(self.embedder, "PASSAGE_PREFIX", "passage: "),
            "threshold": self.threshold,
            "fetch_k": self.fetch_k,
            "collections": self.store.list_collections(),
        }

    def stats(self, top_pages: int = 25) -> dict[str, Any]:
        """Facet counts over every chunk in the collection."""
        from collections import Counter

        metas = self.store.all_metadatas()
        facets = {
            field: dict(Counter(str(m.get(field, "—")) for m in metas).most_common())
            for field in ("doc_kind", "page_type", "status", "type", "version_label")
        }
        pages = Counter(m.get("page_url", "—") for m in metas)
        return {
            "chunks": len(metas),
            "pages": len(pages),
            "facets": facets,
            "top_pages": [
                {"url": url, "chunks": n} for url, n in pages.most_common(top_pages)
            ],
        }

    # ------------------------------------------------------------------ #
    #  Search                                                              #
    # ------------------------------------------------------------------ #

    def search(
        self,
        query: str,
        k: int | None = None,
        include_legacy: bool = False,
        threshold: float | None = None,
        max_rejected: int = 50,
    ) -> dict[str, Any]:
        """Rank the collection for `query`.

        Mirrors the chatbot exactly: normalize → E5 `query: ` embedding →
        Chroma over `max(k, fetch_k)` candidates inside the legacy/outdated
        `where` pre-filter → threshold → version policy → best k.
        """
        k = int(k or self.fetch_k)
        threshold = self.threshold if threshold is None else float(threshold)
        max_distance = 1.0 - threshold
        if self.embedder is None:
            raise RuntimeError(
                "the embedding model is not loaded — call ensure_embedder() first"
            )

        started = time.perf_counter()
        normalized = normalize_text(query)
        t0 = time.perf_counter()
        embedding = self.embedder.embed_query(normalized)
        embed_ms = (time.perf_counter() - t0) * 1000

        where = where_filter(normalized, include_legacy)
        t0 = time.perf_counter()
        raw = self.store.query(embedding, max(k, self.fetch_k), where=where)
        query_ms = (time.perf_counter() - t0) * 1000

        ids = (raw.get("ids") or [[]])[0]
        documents = (raw.get("documents") or [[]])[0]
        metadatas = (raw.get("metadatas") or [[]])[0]
        distances = (raw.get("distances") or [[]])[0]

        kept, rejected = filter_candidates(
            normalized, ids, documents, metadatas, distances, k,
            max_distance=max_distance, include_legacy=include_legacy,
        )
        for rank, hit in enumerate(kept, start=1):
            hit["rank"] = rank

        counts = {
            "fetched": len(ids),
            "kept": len(kept),
            "rejected": len(rejected),
            "below_threshold": sum(1 for r in rejected if r["reason"] == "below_threshold"),
            "legacy_or_outdated": sum(1 for r in rejected if r["reason"] == "legacy_or_outdated"),
            "duplicate": sum(1 for r in rejected if r["reason"] == "duplicate"),
            "beyond_k": sum(1 for r in rejected if r["reason"] == "beyond_k"),
        }

        return {
            "query": query,
            "normalized_query": normalized,
            "k": k,
            "threshold": threshold,
            "include_legacy": bool(include_legacy),
            "where_filter": where,
            "store": self.describe(),
            "timing": {
                "embed_ms": round(embed_ms, 1),
                "query_ms": round(query_ms, 1),
                "total_ms": round((time.perf_counter() - started) * 1000, 1),
            },
            "counts": counts,
            "results": [self._decorate(hit) for hit in kept],
            "rejected": [self._decorate(hit) for hit in rejected[:max_rejected]],
        }

    # ------------------------------------------------------------------ #
    #  Shaping for the UI                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _decorate(hit: dict) -> dict[str, Any]:
        """Add the derived fields the UI binds to, without touching the policy."""
        meta = hit.get("metadata") or {}
        document = hit.get("document") or ""
        body = document.split("\n", 1)[-1] if "\n" in document else document
        out = dict(hit)
        out["similarity"] = round(hit["similarity"], 4)
        out["distance"] = round(hit["distance"], 4)
        out["url"] = meta.get("url") or meta.get("page_url") or ""
        out["page_url"] = meta.get("page_url") or ""
        out["breadcrumb"] = meta.get("breadcrumb") or meta.get("filename") or ""
        out["page_title"] = meta.get("page_title") or ""
        out["doc_kind"] = meta.get("doc_kind") or meta.get("type") or "chunk"
        out["status"] = meta.get("status") or "unknown"
        out["body"] = body
        out["chars"] = len(document)
        out["words"] = len(document.split())
        out["reason_label"] = REASON_LABELS.get(hit.get("reason") or "", "")
        out["chips"] = _chips(meta)
        return out


def _chips(meta: dict) -> list[dict[str, str]]:
    """Ordered label/value chips for the result cards."""
    chips: list[dict[str, str]] = []
    for key in CHIP_KEY_ORDER:
        value = meta.get(key)
        if value in (None, "", False) or key in INTERNAL_KEYS:
            continue
        chips.append({"key": key, "value": str(value)})
    return chips
