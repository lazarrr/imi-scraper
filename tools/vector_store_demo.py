#!/usr/bin/env python3
"""
vector_store_demo.py
~~~~~~~~~~~~~~~~~~~~
Smoke test for the project's own ChromaDB folder (`data/vector_store/`):
writes a handful of sample chunks, then queries the store and shows the
5 nearest neighbours — the same code path the real ingest uses, so if this
works, ingest will too.

    python tools/vector_store_demo.py                     # seed + 3 sample queries
    python tools/vector_store_demo.py -q "koliko traje master" -k 5
    python tools/vector_store_demo.py --collection university_knowledge --no-seed
    python tools/vector_store_demo.py --reset             # drop the demo collection first

The samples go into their own collection (default `demo_smoke_test`), so a
real index in the same folder is never touched.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import settings                            # noqa: E402

DEMO_COLLECTION = "demo_smoke_test"

# Deliberately overlapping topics, so ranking is actually visible.
SAMPLES: list[tuple[str, str]] = [
    ("master-informatika",
     "Master akademske studije informatike traju dve godine (4 semestra) i nose 120 ESPB bodova."),
    ("master-informatika",
     "Na master studijama informatike student bira izborna predmeta iz modula: softversko inženjerstvo, "
     "računarske mreže i informacioni sistemi."),
    ("master-matematika",
     "Master akademske studije matematike traju dve godine i obuhvataju module teorijska matematika "
     "i primenjena matematika."),
    ("osnovne-informatika",
     "Osnovne akademske studije informatike traju četiri godine (8 semestara) i nose 240 ESPB bodova."),
    ("upis-2026",
     "Upis na master studije u školskoj 2026/27. godini vrši se na osnovu prosečne ocene sa osnovnih "
     "studija i rezultata prijemnog ispita."),
    ("studentska-sluzba",
     "Studentska služba radi svakog radnog dana od 9 do 14 časova, u zgradi Fakulteta, kancelarija 12."),
    ("ispiti",
     "Prijava ispita u ispitnom roku obavlja se elektronski, najkasnije tri dana pre početka roka."),
    ("skolarina",
     "Školarina za samofinansirajuće studente master studija iznosi 90.000 dinara godišnje, "
     "a može se platiti u ratama."),
]

QUERIES = [
    "Koliko traju master studije informatike?",
    "Kolika je školarina na masteru?",
    "Kada radi studentska služba?",
]


# --------------------------------------------------------------------------- #
#  Console helpers                                                             #
# --------------------------------------------------------------------------- #

def bar(similarity: float, width: int = 24) -> str:
    filled = max(0, min(width, int(round(similarity * width))))
    return "█" * filled + "·" * (width - filled)


def print_hits(hits: list[dict], rejected: list[dict], counts: dict) -> None:
    if not hits:
        print("    (no results — try lowering the threshold with --threshold)")
    for hit in hits:
        meta = hit["metadata"]
        print(f"  [{hit['rank']:>2}] {hit['similarity']:.4f} |{bar(hit['similarity'])}|  "
              f"dist {hit['distance']:.4f}")
        print(f"       {meta.get('breadcrumb') or meta.get('filename') or meta.get('page_url', '—')}")
        print(f"       {hit['body'][:150]}{'…' if len(hit['body']) > 150 else ''}")
    print(f"    kept {counts['kept']} of {counts['fetched']} candidates · "
          f"below-threshold {counts['below_threshold']} · legacy {counts['legacy_or_outdated']} "
          f"· duplicates {counts['duplicate']} · beyond-k {counts['beyond_k']}")
    for item in rejected:
        if item["reason"] == "below_threshold":
            print(f"      ✗ {item['similarity']:.4f}  {item['reason_detail']}  "
                  f"{(item['metadata'].get('breadcrumb') or item['document'][:60])}")


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--collection", default=DEMO_COLLECTION,
                        help=f"collection to use (default {DEMO_COLLECTION})")
    parser.add_argument("--db-path", default=None,
                        help=f"store folder (default {settings.VECTOR_DB_PATH})")
    parser.add_argument("-q", "--query", action="append", default=None,
                        help="query to run (repeatable; default: built-in samples)")
    parser.add_argument("-k", type=int, default=settings.TOP_K_RESULTS,
                        help=f"how many neighbours to show (default {settings.TOP_K_RESULTS})")
    parser.add_argument("--threshold", type=float, default=settings.SIMILARITY_THRESHOLD)
    parser.add_argument("--no-seed", action="store_true", help="do not write the sample chunks")
    parser.add_argument("--reset", action="store_true", help="delete the demo collection first")
    args = parser.parse_args(argv)

    store_path = settings.path(args.db_path or settings.VECTOR_DB_PATH)
    print(f"\nChroma store : {store_path}")
    print(f"Collection   : {args.collection}")
    print(f"Exists now   : {'yes' if store_path.exists() else 'no — it will be created'}\n")

    from src.core.embeddings import E5Embedder
    from src.core.vector_store import ChromaStore

    cache = settings.data_dir / "cache" / "embeddings.sqlite"
    embedder = E5Embedder(settings.EMBEDDING_MODEL, device=settings.EMBEDDING_DEVICE,
                          batch_size=settings.EMBEDDING_BATCH_SIZE, cache_path=cache)
    store = ChromaStore(store_path, args.collection)

    if args.reset and store.count():
        store.collection.delete(ids=store.collection.get(include=[])["ids"])
        print(f"Reset: dropped {args.collection}\n")

    # ---- write ---------------------------------------------------------- #
    if not args.no_seed:
        ids = [f"demo-{i:02d}" for i in range(len(SAMPLES))]
        texts = [f"[{breadcrumb}] {text}" for breadcrumb, text in SAMPLES]
        metadatas = [
            {
                "page_url": f"https://imi.pmf.kg.ac.rs/demo/{breadcrumb}",
                "url": f"https://imi.pmf.kg.ac.rs/demo/{breadcrumb}",
                "page_title": breadcrumb.replace("-", " ").title(),
                "breadcrumb": f"Demo › {breadcrumb.replace('-', ' ').title()}",
                "doc_kind": "prose",
                "status": "enrolling",
                "source": f"tools/vector_store_demo.py#{i}",
                "written_by": "vector_store_demo",
            }
            for i, (breadcrumb, _) in enumerate(SAMPLES)
        ]
        upserted, deleted = store.replace_page(
            "https://imi.pmf.kg.ac.rs/demo", ids,
            embedder.embed_passages(texts), texts, metadatas,
        )
        print(f"Seeded {upserted} sample chunks into '{args.collection}' "
              f"({deleted} stale removed) — collection now holds {store.count()} chunks")
        store.write_shared_config({
            "embedding_model": settings.EMBEDDING_MODEL,
            "passage_prefix": getattr(embedder, "PASSAGE_PREFIX", "passage: "),
            "query_prefix": getattr(embedder, "QUERY_PREFIX", "query: "),
            "distance": "cosine",
            "collection": args.collection,
            "written_by": "tools/vector_store_demo.py",
        })
    else:
        print(f"Collection holds {store.count()} chunks (seeding skipped)")

    if store.count() == 0:
        print("\nNothing to query. Drop --no-seed to write the samples first.", file=sys.stderr)
        return 1

    # ---- read (through the same service the CLI and the UI use) ---------- #
    from src.core.retrieval_service import RetrievalService

    service = RetrievalService(
        store, embedder,
        model_name=settings.EMBEDDING_MODEL,
        threshold=args.threshold,
        fetch_k=settings.RETRIEVAL_FETCH_K,
    )
    for query in (args.query or QUERIES):
        run = service.search(query, k=args.k)
        print(f"\nQuery: {query}")
        print(f"  ({run['timing']['total_ms']:.0f} ms · embed {run['timing']['embed_ms']:.0f} ms "
              f"· threshold {run['threshold']:.2f})")
        print_hits(run["results"], run["rejected"], run["counts"])

    print(f"\nStore on disk: {store_path}")
    for path in sorted(store_path.iterdir()):
        kind = "dir " if path.is_dir() else "file"
        size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file()) if path.is_dir() \
            else path.stat().st_size
        print(f"  {kind} {path.name:<45} {size / 1024:>10,.0f} KB")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
