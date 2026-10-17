#!/usr/bin/env python3
"""
retrieval_api.py
~~~~~~~~~~~~~~~~
Local JSON API over the project's Chroma store — the backend of the Flutter
retrieval explorer (`ui/retrieval_explorer`).

No web framework: `http.server.ThreadingHTTPServer` + the existing
`RetrievalService`, so the UI adds zero Python dependencies.

    python tools/retrieval_api.py                 # 127.0.0.1:8765, warms the model
    python tools/retrieval_api.py --port 9000
    python tools/retrieval_api.py --no-warm       # load the E5 model on first query

Endpoints
    GET  /api/health                     store path, chunk count, collections, readiness
    GET  /api/stats                      facet counts over all chunks
    GET  /api/collections                collections inside the store folder
    POST /api/search                     {"query", "k", "include_legacy", "threshold", "collection"}
    GET  /api/chunk/<id>                 one full record (drill-down)
    GET  /                               tiny HTML page documenting the endpoints
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import settings                      # noqa: E402
from src.core.retrieval_service import RetrievalService    # noqa: E402

logger = logging.getLogger("retrieval_api")

# Chroma + the E5 model are not safe to hit concurrently from many threads.
_MODEL_LOCK = threading.Lock()
_SERVICE_LOCK = threading.Lock()
_SERVICES: dict[str, RetrievalService] = {}
_STATE = {
    "model_loaded": False,
    "model_error": None,
    "warming": False,
    "started_at": time.time(),
}

# Overridden by `run.py ui --db-path … --collection …`.
_DEFAULTS: dict[str, str | None] = {"db_path": None, "collection": None}


def configure(db_path: str | None = None, collection: str | None = None) -> None:
    """Set the store/collection this server serves by default."""
    _DEFAULTS["db_path"] = db_path
    _DEFAULTS["collection"] = collection


def default_collection() -> str:
    return _DEFAULTS["collection"] or settings.CHROMA_COLLECTION


def default_db_path() -> str:
    return _DEFAULTS["db_path"] or settings.VECTOR_DB_PATH


# --------------------------------------------------------------------------- #
#  Service registry                                                            #
# --------------------------------------------------------------------------- #

def get_service(
    collection: str | None = None,
    db_path: str | None = None,
    *,
    need_model: bool = True,
) -> RetrievalService:
    """One RetrievalService per (store, collection), built on first use.

    `need_model=False` (health, stats, collection listing) opens the store
    without loading the 2 GB E5 model, so those endpoints answer instantly.
    """
    store_path = db_path or default_db_path()
    name = collection or default_collection()
    key = f"{store_path}::{name}"
    with _SERVICE_LOCK:
        service = _SERVICES.get(key)
        if service is None:
            service = RetrievalService.build(
                settings,
                db_path=store_path,
                collection=name,
                with_model=False,
            )
            _SERVICES[key] = service
    if need_model and not service.model_loaded:
        with _MODEL_LOCK:
            service.ensure_embedder(
                device=settings.EMBEDDING_DEVICE,
                cache_path=settings.data_dir / "cache" / "embeddings.sqlite",
            )
    return service


def warm_model(collection: str | None = None) -> None:
    """Load the E5 model + open the store so the first search is instant."""
    _STATE["warming"] = True
    try:
        service = get_service(collection, need_model=True)
        with _MODEL_LOCK:
            service.embedder.embed_query("zagrevanje")
        _STATE["model_loaded"] = True
        _STATE["model_error"] = None
        logger.info("Model ready (%s), store has %d chunks",
                    service.model_name, service.store.count())
    except Exception as exc:                                  # pragma: no cover - env dependent
        _STATE["model_error"] = f"{type(exc).__name__}: {exc}"
        logger.error("Could not warm the model: %s", exc)
        logger.debug("%s", traceback.format_exc())
    finally:
        _STATE["warming"] = False


# --------------------------------------------------------------------------- #
#  Request handling                                                            #
# --------------------------------------------------------------------------- #

class Handler(BaseHTTPRequestHandler):
    server_version = "IMIRetrievalAPI/1.0"
    protocol_version = "HTTP/1.1"

    # ---- plumbing -------------------------------------------------------- #

    def log_message(self, fmt: str, *args) -> None:            # noqa: A003
        logger.info("%s - %s", self.address_string(), fmt % args)

    def handle_error(self, request, client_address) -> None:
        """A keep-alive client closing its socket is normal, not a crash."""
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionResetError, BrokenPipeError, TimeoutError)):
            logger.debug("client %s disconnected: %s", client_address, exc)
            return
        super().handle_error(request, client_address)

    def _send(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def _fail(self, message: str, status: int = 400, **extra) -> None:
        self._send({"ok": False, "error": message, **extra}, status=status)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8")) or {}
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"invalid JSON body: {exc}") from exc

    # ---- verbs ----------------------------------------------------------- #

    def do_OPTIONS(self) -> None:                              # noqa: N802
        self._send({"ok": True})

    def do_GET(self) -> None:                                  # noqa: N802
        path = urlparse(self.path).path.rstrip("/") or "/"
        try:
            if path == "/":
                self._send({"ok": True, "service": "imi retrieval api", "endpoints": [
                    "GET /api/health", "GET /api/stats", "GET /api/collections",
                    "POST /api/search", "GET /api/chunk/<id>",
                ]})
            elif path == "/api/health":
                self._send(self._health())
            elif path == "/api/stats":
                self._send({"ok": True, **get_service(self._collection_arg(), need_model=False).stats()})
            elif path == "/api/collections":
                self._send({"ok": True, "store": str(ROOT),
                            "collections": get_service(need_model=False).store.list_collections()})
            elif path.startswith("/api/chunk/"):
                self._chunk(unquote(path[len("/api/chunk/"):]))
            else:
                self._fail(f"unknown endpoint {path}", status=404)
        except Exception as exc:                               # pragma: no cover
            logger.error("%s", traceback.format_exc())
            self._fail(f"{type(exc).__name__}: {exc}", status=500)

    def do_POST(self) -> None:                                 # noqa: N802
        path = urlparse(self.path).path.rstrip("/")
        try:
            body = self._body()
        except ValueError as exc:
            return self._fail(str(exc))
        try:
            if path == "/api/search":
                self._search(body)
            elif path == "/api/warm":
                threading.Thread(target=warm_model, args=(body.get("collection"),),
                                 daemon=True).start()
                self._send({"ok": True, "warming": True})
            else:
                self._fail(f"unknown endpoint {path}", status=404)
        except Exception as exc:                               # pragma: no cover
            logger.error("%s", traceback.format_exc())
            self._fail(f"{type(exc).__name__}: {exc}", status=500)

    # ---- handlers -------------------------------------------------------- #

    def _collection_arg(self) -> str | None:
        query = urlparse(self.path).query
        for pair in query.split("&"):
            if pair.startswith("collection="):
                return unquote(pair.split("=", 1)[1])
        return None

    def _health(self) -> dict:
        service = get_service(need_model=False)
        size = service.store.db_size_bytes()
        return {
            "ok": True,
            "model_loaded": service.model_loaded or _STATE["model_loaded"],
            "model_error": _STATE["model_error"],
            "warming": _STATE["warming"],
            "uptime_s": round(time.time() - _STATE["started_at"], 1),
            "store": service.describe(),
            "size_bytes": size,
            "size_human": _human(size),
            "project_root": str(ROOT),
        }

    def _search(self, body: dict) -> None:
        query = (body.get("query") or "").strip()
        if not query:
            return self._fail("'query' must not be empty")
        service = get_service(body.get("collection"), body.get("db_path"))
        k = int(body.get("k") or settings.TOP_K_RESULTS)
        threshold = body.get("threshold")
        with _MODEL_LOCK:
            run = service.search(
                query,
                k=k,
                include_legacy=bool(body.get("include_legacy")),
                threshold=None if threshold in (None, "") else float(threshold),
                max_rejected=int(body.get("max_rejected") or 50),
            )
        _STATE["model_loaded"] = True
        self._send({"ok": True, **run})

    def _chunk(self, chunk_id: str) -> None:
        service = get_service(self._collection_arg(), need_model=False)
        got = service.store.get_chunks([chunk_id])
        if not got["ids"]:
            return self._fail(f"chunk {chunk_id} not found", status=404)
        self._send({
            "ok": True,
            "id": got["ids"][0],
            "document": got["documents"][0],
            "metadata": got["metadatas"][0] or {},
        })


def _human(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} GB"


# --------------------------------------------------------------------------- #
#  Entry point                                                                 #
# --------------------------------------------------------------------------- #

def make_server(host: str, port: int) -> ThreadingHTTPServer:
    """Bind the API socket (raises OSError when the port is busy)."""
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True
    return httpd


def serve(host: str, port: int, warm: bool = True, collection: str | None = None) -> None:
    httpd = make_server(host, port)
    logger.info("Retrieval API on http://%s:%d  (store %s)",
                host, port, settings.path(default_db_path()))
    if warm:
        threading.Thread(target=warm_model, args=(collection,), daemon=True).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("shutting down")
    finally:
        httpd.server_close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--host", default=settings.UI_HOST)
    parser.add_argument("--port", type=int, default=settings.UI_PORT)
    parser.add_argument("--collection", default=None,
                        help=f"default collection (default {settings.CHROMA_COLLECTION})")
    parser.add_argument("--db-path", default=None,
                        help=f"Chroma store folder (default {settings.VECTOR_DB_PATH})")
    parser.add_argument("--no-warm", action="store_true",
                        help="do not preload the embedding model at startup")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S",
    )
    for noisy in ("chromadb", "sentence_transformers", "transformers", "httpx",
                  "urllib3", "huggingface_hub", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    configure(args.db_path, args.collection)
    serve(args.host, args.port, warm=not args.no_warm, collection=args.collection)
    return 0


if __name__ == "__main__":
    sys.exit(main())
