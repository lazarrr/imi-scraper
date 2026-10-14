"""
state.py
~~~~~~~~
Per-collection ingest state (data/state.json) — replaces the old
crawl_history.json.

A page is skipped on the next run only if its signature is unchanged:
extracted-text hash + prompt version + its registry entries (incl. the
enrollment year, statuses move with time) + chunking/embedding settings
+ whether DeepSeek was used.  Probed tab params are cached here too.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional


class IngestState:
    def __init__(self, path: Path):
        self.path = path
        self.data: dict = {"version": 2, "collections": {}, "tab_params": {}}
        if path.is_file():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict) and loaded.get("version") == 2:
                    self.data = loaded
            except json.JSONDecodeError:
                pass

    def _collection(self, collection: str) -> dict:
        return self.data["collections"].setdefault(collection, {})

    def page(self, collection: str, page_url: str) -> Optional[dict]:
        return self._collection(collection).get(page_url)

    def is_unchanged(self, collection: str, page_url: str, signature: str) -> bool:
        entry = self.page(collection, page_url)
        return entry is not None and entry.get("signature") == signature

    def record(self, collection: str, page_url: str, **fields) -> None:
        fields["ingested_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._collection(collection)[page_url] = fields

    def forget(self, collection: str, page_url: str) -> None:
        self._collection(collection).pop(page_url, None)

    def tab_params(self, page_url: str) -> Optional[dict]:
        return self.data["tab_params"].get(page_url)

    def set_tab_params(self, page_url: str, params: dict) -> None:
        self.data["tab_params"][page_url] = params

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.path)
