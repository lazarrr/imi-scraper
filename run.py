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
    python run.py ui                                    # Flutter desktop app over the store
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
            allow_partial_write=args.allow_partial_write,
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
    return 1 if any(p.outcome in ("failed", "partial") for p in report.pages) else 0


# --------------------------------------------------------------------------- #
#  search / eval                                                               #
# --------------------------------------------------------------------------- #

class _Searcher:
    """Same retrieval the chatbot performs: E5 query prefix, threshold,
    version policy — all of it shared with the UI via
    `src/core/retrieval_service.py`, so the CLI and the Flutter app can never
    disagree about a ranking."""

    def __init__(self, args, embedder=None):
        from src.core.retrieval_service import RetrievalService

        self.service = RetrievalService.build(
            settings,
            db_path=_db_path(args),
            collection=_collection(args),
            embedder=embedder,
            embedder_cache=settings.data_dir / "cache" / "embeddings.sqlite",
        )
        if self.service.store.count() == 0:
            raise SystemExit(
                f"Collection '{_collection(args)}' at {_db_path(args)} is empty — run ingest first."
            )

    def run_search(self, query: str, k: int, include_legacy: bool = False) -> dict:
        """Full structured run: results + rejected candidates + timings."""
        return self.service.search(query, k=k, include_legacy=include_legacy)

    def search(self, query: str, k: int, include_legacy: bool = False):
        """(documents, metadatas, distances, filtered) — the CLI's shape."""
        run = self.run_search(query, k, include_legacy)
        docs = [hit["document"] for hit in run["results"]]
        metas = [hit["metadata"] for hit in run["results"]]
        dists = [hit["distance"] for hit in run["results"]]
        return docs, metas, dists, run["counts"]["below_threshold"]


def cmd_search(args) -> int:
    searcher = _Searcher(args)
    run = searcher.run_search(args.query, args.k, args.include_legacy)
    counts = run["counts"]
    print(f"\nQuery: {args.query}   ({counts['fetched']} candidates from "
          f"{run['store']['collection']}, {len(run['results'])} kept, "
          f"{counts['below_threshold']} below threshold, "
          f"{counts['legacy_or_outdated']} legacy/outdated)\n")
    for hit in run["results"]:
        meta = hit["metadata"]
        print(f"[{hit['rank']}] similarity {hit['similarity']:.3f} | distance {hit['distance']:.3f} "
              f"| status {meta.get('status') or '-'} | {meta.get('doc_kind', 'chunk')}")
        print(f"    {meta.get('breadcrumb') or meta.get('filename')}")
        print(f"    {meta.get('url')}")
        print(f"    {hit['body'][:300]}{'…' if len(hit['body']) > 300 else ''}\n")
    if not run["results"]:
        print("No results.")
    print(f"Total {run['timing']['total_ms']:.0f} ms "
          f"(embed {run['timing']['embed_ms']:.0f} ms) · store {run['store']['path']}")
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
    for field in ("status", "doc_kind", "page_type", "version_source", "type",
                  "curation_status", "curated", "rewritten"):
        counts = Counter(str(m.get(field, "-")) for m in metas)
        print(f"{field:15} " + ", ".join(f"{k}: {v}" for k, v in counts.most_common()))
    pages = Counter(m.get("page_url", "-") for m in metas)
    print(f"\nPages ({len(pages)}):")
    for url, n in pages.most_common():
        print(f"  {n:5}  {url}")
    return 0


# --------------------------------------------------------------------------- #
#  ui                                                                          #
# --------------------------------------------------------------------------- #

UI_DIR = Path(__file__).resolve().parent / "ui" / "retrieval_explorer"
# Release first, then Debug: on some Xcode versions `flutter build macos
# --release` fails in the tool's universal-binary step while --debug succeeds.
BUILT_APPS = (
    UI_DIR / "build" / "macos" / "Build" / "Products" / "Release"
    / "retrieval_explorer.app" / "Contents" / "MacOS" / "retrieval_explorer",
    UI_DIR / "build" / "macos" / "Build" / "Products" / "Debug"
    / "retrieval_explorer.app" / "Contents" / "MacOS" / "retrieval_explorer",
)


def _api_reachable(url: str, timeout: float = 2.0) -> bool:
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(f"{url}/api/health", timeout=timeout) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError):
        return False


def _wait_for_api(url: str, timeout: float = 30.0) -> bool:
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        if _api_reachable(url):
            return True
        time.sleep(0.25)
    return False


def _app_command(args) -> list[str] | None:
    """Where to launch the desktop app from: explicit path, an already built
    bundle, or `flutter run` straight out of the sources."""
    import shutil

    if args.app_path:
        return [args.app_path]
    for candidate in BUILT_APPS:
        if candidate.is_file():
            return [str(candidate)]
    flutter = shutil.which("flutter")
    if not flutter:
        return None
    build_mode = "--release" if args.release else "--debug"
    return [flutter, "run", "-d", "macos", build_mode]


def cmd_ui(args) -> int:
    """Serve the local retrieval API and open the Flutter desktop app."""
    import subprocess
    import threading
    import time

    from tools.retrieval_api import configure, make_server, warm_model

    host = args.host or settings.UI_HOST
    port = args.port or settings.UI_PORT
    url = f"http://{host}:{port}"
    configure(str(_db_path(args)), args.collection)

    try:
        httpd = make_server(host, port)
    except OSError as exc:
        print(f"ERROR: cannot listen on {url} — {exc}")
        print("       Another instance may already be running (try --port 8766).")
        return 2

    threading.Thread(
        target=httpd.serve_forever,
        name="retrieval-api",
        daemon=True,
    ).start()
    if not args.no_warm:
        threading.Thread(target=warm_model, args=(args.collection,), daemon=True).start()

    if not _wait_for_api(url, timeout=20.0):
        print(f"ERROR: the API did not answer on {url}.")
        return 2

    print("\n" + "─" * 62)
    print(f"  Retrieval API : {url}")
    print(f"  Store         : {settings.path(args.db_path or settings.VECTOR_DB_PATH)}")
    print(f"  Collection    : {args.collection or settings.CHROMA_COLLECTION}")
    print(f"  Endpoints     : {url}/api/health · /api/search · /api/stats")
    print("─" * 62 + "\n")

    if args.no_app:
        print(f"Serving the API only — press Ctrl-C to stop ({url}).")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            httpd.shutdown()
        return 0

    if args.build:
        import shutil
        import subprocess as sp

        flutter = shutil.which("flutter")
        if not flutter:
            print("ERROR: `flutter` is not on PATH, cannot build the desktop app.")
            return 2
        print("Building the macOS app (flutter build macos --release) …")
        code = sp.call([flutter, "build", "macos", "--release"], cwd=str(UI_DIR))
        if code != 0:
            print("ERROR: the Flutter build failed — see the output above.")
            return code
        args.app_path = str(BUILT_APPS[0])

    command = _app_command(args)
    if command is None:
        print("Flutter is not on PATH and no built app was found, so the UI cannot be")
        print("started for you.  Either install Flutter, or build the app once:")
        print(f"    cd {UI_DIR} && flutter build macos --release")
        print("    python run.py ui --app-path <path>/retrieval_explorer")
        print(f"\nThe API stays up on {url} — press Ctrl-C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            return 0

    print(f"Launching: {' '.join(command)}")
    try:
        code = subprocess.call(command, cwd=str(UI_DIR))
    except KeyboardInterrupt:
        code = 0
    finally:
        httpd.shutdown()
    print(f"\nDesktop app closed (exit {code}). API stopped.")
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
    p.add_argument("--allow-partial-write", action="store_true",
                   help="write the successful sections even if some section failed curation "
                        "(the page stays uncommitted in state and is retried next run)")

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

    p = sub.add_parser("ui", help="open the Flutter desktop app for visualising retrieval")
    common(p)
    p.add_argument("--host", help=f"API host (default {settings.UI_HOST})")
    p.add_argument("--port", type=int, help=f"API port (default {settings.UI_PORT})")
    p.add_argument("--no-app", action="store_true", help="serve the API only, do not open the app")
    p.add_argument("--no-warm", action="store_true", help="do not preload the E5 model")
    p.add_argument("--release", action="store_true", help="run/build the app in release mode")
    p.add_argument("--build", action="store_true", help="build the macOS app first, then open it")
    p.add_argument("--app-path", help="path to an already built retrieval_explorer binary")
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
        "remove": cmd_remove, "stats": cmd_stats, "ui": cmd_ui,
    }
    try:
        return commands[args.command or "ingest"](args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
