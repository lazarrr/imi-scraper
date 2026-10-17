# `data/vector_store/` — the project's ChromaDB index

This folder **is** the vector database. It is created automatically the first
time anything opens the store, and every ingest now writes here — nothing has
to be copied in from the chatbot checkout any more.

```
data/vector_store/
├── chroma.sqlite3                 # chunks, documents, metadata (SQLite)
├── <collection-uuid>/             # HNSW index files for that collection
│   ├── data_level0.bin
│   ├── header.bin
│   ├── length.bin
│   └── link_lists.bin
└── shared_config.json             # embedding model / distance / prefixes used
```

## How it is wired up

| Where | Setting | Value |
| --- | --- | --- |
| `config/settings.py` | `VECTOR_DB_PATH` | `./data/vector_store` |
| `.env` | `VECTOR_DB_PATH` | `./data/vector_store` |
| `.env` | `CHROMA_COLLECTION` | `university_knowledge` |

Relative paths resolve against the **project root** (`config.settings.Settings.path`),
so the store is always found no matter which directory you launch from.

## Commands that use it

```bash
python run.py ingest                 # crawl links.txt → embed → write into this folder
python run.py stats                  # what is inside right now
python run.py search "koliko modula ima informatika"
python tools/vector_store_demo.py    # standalone smoke test: write + query 5 similar chunks
python run.py ui                     # Flutter desktop app over this store (5 similar results)
```

## Notes

* Cosines space, explicit embeddings (`passage: ` / `query: ` E5 prefixes) — the
  same convention the chatbot uses, so the folder can be handed to it as-is.
* `data/` is git-ignored: the index is a build artifact. Only this README is
  tracked. Deleting the folder is a safe "start over" — the next ingest
  rebuilds it (raw HTML stays cached in `data/cache/http`).
* To share one index with the chatbot instead, set `VECTOR_DB_PATH` in **both**
  projects to the same absolute path.
