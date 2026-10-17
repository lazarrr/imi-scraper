"""
vector_store.py
~~~~~~~~~~~~~~~
ChromaDB writer/reader, compatible with the chatbot:

  • same PersistentClient path, collection name and cosine space,
    created exactly like the chatbot does (no embedding function,
    explicit embeddings)
  • replace_page(): upsert the page's chunks FIRST, then delete the ids
    of that page that no longer exist — no window in which the page is
    missing, and no stale chunks after the page changes
  • metadata is sanitized to what Chroma accepts everywhere: scalar
    values only, no None (lists are joined with "; ")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MAX_META_STRING = 2000


def sanitize_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for key, value in meta.items():
        if key.startswith("_") or value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            value = "; ".join(str(v) for v in value if v not in (None, ""))
            if not value:
                continue
        if isinstance(value, bool) or isinstance(value, (int, float)):
            clean[key] = value
        else:
            text = str(value)
            clean[key] = text if len(text) <= MAX_META_STRING else text[: MAX_META_STRING - 1] + "…"
    return clean


class ChromaStore:
    def __init__(self, path: str | Path, collection_name: str):
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(self.path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------ #
    #  Writing                                                             #
    # ------------------------------------------------------------------ #

    def _batch_size(self) -> int:
        try:
            return max(1, int(self.client.get_max_batch_size()))
        except Exception:
            return 1000

    def page_ids(self, page_url: str) -> list[str]:
        return self.collection.get(where={"page_url": page_url}, include=[])["ids"]

    def replace_page(
        self,
        page_url: str,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> tuple[int, int]:
        """Upsert this page's chunks, then delete its stale ones.
        Returns (upserted, deleted)."""
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate chunk ids for {page_url}")
        metadatas = [sanitize_metadata(m) for m in metadatas]
        size = self._batch_size()
        for i in range(0, len(ids), size):
            self.collection.upsert(
                ids=ids[i:i + size],
                embeddings=embeddings[i:i + size],
                documents=documents[i:i + size],
                metadatas=metadatas[i:i + size],
            )
        keep = set(ids)
        stale = [x for x in self.page_ids(page_url) if x not in keep]
        for i in range(0, len(stale), size):
            self.collection.delete(ids=stale[i:i + size])
        return len(ids), len(stale)

    def remove_page(self, page_url: str) -> int:
        ids = self.page_ids(page_url)
        size = self._batch_size()
        for i in range(0, len(ids), size):
            self.collection.delete(ids=ids[i:i + size])
        return len(ids)

    def add_also_on(self, chunk_id: str, url: str) -> None:
        got = self.collection.get(ids=[chunk_id], include=["metadatas"])
        if not got["ids"]:
            return
        meta = dict(got["metadatas"][0] or {})
        urls = [u for u in str(meta.get("also_on", "")).split("; ") if u]
        if url not in urls:
            urls.append(url)
            meta["also_on"] = "; ".join(urls)
            self.collection.update(ids=[chunk_id], metadatas=[meta])

    # ------------------------------------------------------------------ #
    #  Reading                                                             #
    # ------------------------------------------------------------------ #

    def count(self) -> int:
        return self.collection.count()

    def list_collections(self) -> list[dict]:
        """Every collection in this store with its chunk count (UI selector)."""
        out: list[dict] = []
        for item in self.client.list_collections():
            name = item if isinstance(item, str) else getattr(item, "name", str(item))
            try:
                count = self.client.get_collection(name).count()
            except Exception:
                count = -1
            out.append({"name": name, "count": count, "active": name == self.collection_name})
        return sorted(out, key=lambda c: (-c["count"], c["name"]))

    def get_chunks(self, ids: list[str]) -> dict:
        """Full records for the given ids (drill-down in the UI)."""
        if not ids:
            return {"ids": [], "documents": [], "metadatas": []}
        return self.collection.get(ids=ids, include=["documents", "metadatas"])

    def db_size_bytes(self) -> int:
        """On-disk size of the store folder — shown in the UI header."""
        total = 0
        for path in self.path.rglob("*"):
            try:
                if path.is_file():
                    total += path.stat().st_size
            except OSError:
                continue
        return total

    def query(self, embedding: list[float], n_results: int, where: dict | None = None) -> dict:
        n = max(1, min(n_results, self.count() or 1))
        return self.collection.query(
            query_embeddings=[embedding],
            n_results=n,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

    def all_metadatas(self, page_size: int = 2000) -> list[dict]:
        out: list[dict] = []
        offset = 0
        while True:
            got = self.collection.get(include=["metadatas"], limit=page_size, offset=offset)
            out.extend(m or {} for m in got["metadatas"])
            if len(got["ids"]) < page_size:
                return out
            offset += page_size

    def write_shared_config(self, info: dict) -> None:
        """Record how the store was built, so both projects stay in sync."""
        path = self.path / "shared_config.json"
        data = {}
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                data = {}
        data.update(info)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
