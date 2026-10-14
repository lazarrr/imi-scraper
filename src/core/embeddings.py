"""
embeddings.py
~~~~~~~~~~~~~
intfloat/multilingual-e5-large — exactly as the chatbot uses it:
"passage: " prefix when indexing, "query: " prefix when searching,
normalized vectors (cosine).

Embeddings are cached in SQLite by (model, prefixed text), so re-running
an ingest after a small page change only embeds the changed chunks.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np

from src.utils.text import sha1

logger = logging.getLogger(__name__)


class EmbeddingCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path))
        self._conn.execute("CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, vec BLOB)")
        self._conn.commit()

    def get_many(self, keys: list[str]) -> dict[str, np.ndarray]:
        found: dict[str, np.ndarray] = {}
        for i in range(0, len(keys), 500):
            part = keys[i:i + 500]
            marks = ",".join("?" * len(part))
            for key, blob in self._conn.execute(
                f"SELECT key, vec FROM vectors WHERE key IN ({marks})", part
            ):
                found[key] = np.frombuffer(blob, dtype=np.float32)
        return found

    def put_many(self, items: list[tuple[str, np.ndarray]]) -> None:
        self._conn.executemany(
            "INSERT OR REPLACE INTO vectors (key, vec) VALUES (?, ?)",
            [(k, v.astype(np.float32).tobytes()) for k, v in items],
        )
        self._conn.commit()


class E5Embedder:
    # E5 models require explicit prefixes — omitting them degrades retrieval quality
    QUERY_PREFIX = "query: "
    PASSAGE_PREFIX = "passage: "

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        batch_size: int = 16,
        cache_path: Optional[Path] = None,
    ):
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model %s on %s …", model_name, device)
        self.model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name
        self.batch_size = batch_size
        self.cache = EmbeddingCache(cache_path) if cache_path else None

    def _key(self, prefixed: str) -> str:
        return sha1(f"{self.model_name}\x00{prefixed}")

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        prefixed = [self.PASSAGE_PREFIX + t for t in texts]
        keys = [self._key(p) for p in prefixed]
        cached = self.cache.get_many(keys) if self.cache else {}

        missing = [i for i, k in enumerate(keys) if k not in cached]
        if missing:
            vectors = self.model.encode(
                [prefixed[i] for i in missing],
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=len(missing) > 32,
                convert_to_numpy=True,
            )
            new_items = [(keys[i], vec) for i, vec in zip(missing, vectors)]
            if self.cache:
                self.cache.put_many(new_items)
            cached.update(dict(new_items))
        logger.info("Embedded %d passages (%d from cache)", len(texts), len(texts) - len(missing))
        return [cached[k].tolist() for k in keys]

    def embed_query(self, text: str) -> list[float]:
        vector = self.model.encode(
            [self.QUERY_PREFIX + text], normalize_embeddings=True, convert_to_numpy=True
        )[0]
        return vector.tolist()
