"""
run.py
~~~~~~
CLI entry point.

    python run.py                                       # ingest everything in links.txt
    python run.py ingest --links links.test.txt --dry-run --no-llm
    python run.py ingest --links links.test.txt --collection university_knowledge_test
    python run.py search "Koliko modula ima informatika?" --collection university_knowledge_test
    python run.py eval --collection university_knowledge_test
    python run.py stats
    python run.py remove https://imi.pmf.kg.ac.rs/neka-stranica
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

from config.settings import settings

logger = logging.getLogger("scraper")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "httpcore", "urllib3", "chromadb", "sentence_transformers",
                  "transformers", "openai", "trafilatura", "pdfminer", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _collection(args) -> str:
    return args.collection or settings.CHROMA_COLLECTION


def _db_path(args) -> Path:
    return settings.path(args.db_path or settings.VECTOR_DB_PATH)


# --------------------------------------------------------------------------- #
#  ingest                                                                      #
# --------------------------------------------------------------------------- #

def cmd_ingest(args) -> int:
    from src.core.deepseek_client import LLMError
    from src.processing.ingest import IngestPipeline

    links = settings.path(args.links or settings.LINKS_FILE)
    try:
        pipeline = IngestPipeline(
            settings,
            links_file=links,
            collection=_collection(args),
            db_path=_db_path(args),
            dry_run=args.dry_run,
            no_llm=args.no_llm,
            force=args.force,
            cached=args.cached,
            keep_low_value=args.keep_low_value,
            include_unreachable=args.include_unreachable,
        )
    except LLMError as exc:
        print(f"ERROR: {exc}")
        return 2

    report, path = pipeline.run()
    print()
    for line in report.summary_lines():
        print(line)
    if report.drafts_file:
        print(f"Registry drafts: {report.drafts_file}  (review → move into programs.yaml)")
    print(f"Report: {path}")
    if not args.dry_run:
        print("Restart the chatbot so it picks up the new data.")
    return 1 if any(p.outcome == "failed" for p in report.pages) else 0


# --------------------------------------------------------------------------- #
#  search / eval                                                               #
# --------------------------------------------------------------------------- #

class _Searcher:
    """Same retrieval the chatbot performs: E5 query prefix, threshold,
    version policy (src/core/retrieval.py)."""

    def __init__(self, args):
        from src.core.embeddings import E5Embedder
        from src.core.vector_store import ChromaStore

        self.store = ChromaStore(_db_path(args), _collection(args))
        if self.store.count() == 0:
            raise SystemExit(f"Collection '{_collection(args)}' at {_db_path(args)} is empty — run ingest first.")
        self.embedder = E5Embedder(
            settings.EMBEDDING_MODEL, device=settings.EMBEDDING_DEVICE,
            cache_path=None,
        )

    def search(self, query: str, k: int, include_legacy: bool = False):
        from src.core.retrieval import apply_version_policy, where_filter
        from src.utils.text import normalize_text

        query = normalize_text(query)
        result = self.store.query(
            self.embedder.embed_query(query),
            max(k, settings.RETRIEVAL_FETCH_K),
            where=where_filter(query, include_legacy),
        )
        docs, metas, dists = result["documents"][0], result["metadatas"][0], result["distances"][0]
        max_distance = 1.0 - settings.SIMILARITY_THRESHOLD
        kept = [(d, m, x) for d, m, x in zip(docs, metas, dists) if x <= max_distance]
        filtered = len(docs) - len(kept)
        docs, metas, dists = (list(t) for t in zip(*kept)) if kept else ([], [], [])
        docs, metas, dists = apply_version_policy(query, docs, metas, dists, k, include_legacy)
        return docs, metas, dists, filtered


def cmd_search(args) -> int:
    searcher = _Searcher(args)
    docs, metas, dists, filtered = searcher.search(args.query, args.k, args.include_legacy)
    print(f"\nQuery: {args.query}   ({filtered} results above distance threshold)\n")
    for rank, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
        status = meta.get("status") or "-"
        print(f"[{rank}] distance {dist:.3f} | status {status} | {meta.get('doc_kind', 'chunk')}")
        print(f"    {meta.get('breadcrumb') or meta.get('filename')}")
        print(f"    {meta.get('url')}")
        body = doc.split("\n", 1)[-1].replace("\n", " ")
        print(f"    {body[:300]}{'…' if len(body) > 300 else ''}\n")
    if not docs:
        print("No results.")
    return 0


def cmd_eval(args) -> int:
    import yaml

    from src.utils.text import fold

    path = settings.path(args.file or settings.EVAL_FILE)
    cases = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    searcher = _Searcher(args)
    hits, rr = 0, 0.0
    print(f"\n{'#':>2}  {'hit':4} {'rank':4}  question")
    for i, case in enumerate(cases, start=1):
        expect = case.get("expect", {})
        docs, metas, _, _ = searcher.search(case["question"], args.k, args.include_legacy)
        rank = None
        for r, (doc, meta) in enumerate(zip(docs, metas), start=1):
            haystacks = {
                "url": f"{meta.get('url', '')} {meta.get('page_url', '')}",
                "breadcrumb": f"{meta.get('breadcrumb', '')} {meta.get('version_label', '')}",
                "text": doc,
            }
            if any(fold(needle) in fold(haystacks[field]) for field in ("url", "breadcrumb", "text")
                   for needle in expect.get(field, [])):
                rank = r
                break
        if rank:
            hits += 1
            rr += 1.0 / rank
        print(f"{i:>2}  {'yes' if rank else 'NO':4} {rank or '-':>4}  {case['question']}")
    n = len(cases) or 1
    print(f"\nhit@{args.k}: {hits}/{len(cases)} = {hits / n:.0%}   MRR: {rr / n:.3f}")
    return 0


# --------------------------------------------------------------------------- #
#  remove / stats                                                              #
# --------------------------------------------------------------------------- #

def cmd_remove(args) -> int:
    from src.core.vector_store import ChromaStore
    from src.processing.fetcher import canonicalize_url
    from src.processing.state import IngestState

    target = args.url
    if Path(target).is_file():
        page_url = Path(target).resolve().as_uri()
    else:
        page_url = canonicalize_url(target, settings.CRAWL_ALLOWED_DOMAINS)
    store = ChromaStore(_db_path(args), _collection(args))
    removed = store.remove_page(page_url)
    state = IngestState(settings.data_dir / "state.json")
    state.forget(_collection(args), page_url)
    state.save()
    print(f"Removed {removed} chunks of {page_url} from '{_collection(args)}'.")
    return 0


def cmd_stats(args) -> int:
    from src.core.vector_store import ChromaStore

    store = ChromaStore(_db_path(args), _collection(args))
    metas = store.all_metadatas()
    print(f"\nCollection '{_collection(args)}' at {_db_path(args)}: {len(metas)} chunks\n")
    for field in ("status", "doc_kind", "page_type", "version_source", "type"):
        counts = Counter(str(m.get(field, "-")) for m in metas)
        print(f"{field:15} " + ", ".join(f"{k}: {v}" for k, v in counts.most_common()))
    pages = Counter(m.get("page_url", "-") for m in metas)
    print(f"\nPages ({len(pages)}):")
    for url, n in pages.most_common():
        print(f"  {n:5}  {url}")
    return 0


# --------------------------------------------------------------------------- #
#  main                                                                        #
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="IMI scraper → DeepSeek → ChromaDB")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command")

    def common(p):
        p.add_argument("--collection", help=f"Chroma collection (default {settings.CHROMA_COLLECTION})")
        p.add_argument("--db-path", help="Chroma persist directory (default VECTOR_DB_PATH)")

    p = sub.add_parser("ingest", help="crawl the links file and (re)index it")
    common(p)
    p.add_argument("--links", help=f"links file (default {settings.LINKS_FILE})")
    p.add_argument("--dry-run", action="store_true", help="parse + DeepSeek + report, no embeddings / DB writes")
    p.add_argument("--no-llm", action="store_true", help="skip DeepSeek (registry + deterministic labels only)")
    p.add_argument("--force", action="store_true", help="re-process pages even if unchanged")
    p.add_argument("--cached", action="store_true", help="reuse raw pages from data/cache/http")
    p.add_argument("--keep-low-value", action="store_true", help="keep chunks DeepSeek rated boilerplate/low_value")
    p.add_argument("--include-unreachable", action="store_true", help="also index panels no visitor can open")

    for name, help_text in (("search", "query the collection like the chatbot does"),
                            ("eval", "run the golden questions (hit@k)")):
        p = sub.add_parser(name, help=help_text)
        common(p)
        if name == "search":
            p.add_argument("query")
        else:
            p.add_argument("--file", help=f"questions file (default {settings.EVAL_FILE})")
        p.add_argument("-k", type=int, default=settings.TOP_K_RESULTS)
        p.add_argument("--include-legacy", action="store_true", help="do not filter legacy program versions")

    p = sub.add_parser("remove", help="delete all chunks of one page / file")
    common(p)
    p.add_argument("url")

    p = sub.add_parser("stats", help="collection statistics")
    common(p)
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0].startswith("-") and argv[0] not in ("-h", "--help", "-v", "--verbose"):
        argv = ["ingest"] + argv
    elif argv[0] in ("-v", "--verbose") and (len(argv) == 1 or argv[1].startswith("-")):
        argv = [argv[0], "ingest"] + argv[1:]
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    commands = {
        "ingest": cmd_ingest, "search": cmd_search, "eval": cmd_eval,
        "remove": cmd_remove, "stats": cmd_stats,
    }
    return commands[args.command or "ingest"](args)


if __name__ == "__main__":
    sys.exit(main())
