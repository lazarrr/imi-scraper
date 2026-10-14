"""
settings.py
~~~~~~~~~~~
Scraper configuration (pydantic-settings, same style as the chatbot).

Every value can be overridden via environment variables or a `.env`
file in the project root.  List values are given as JSON, e.g.
    CRAWL_DOC_PATTERNS='["/pub/", "/knjiga_predmeta/2026/"]'

Settings that MUST match the chatbot (C:\\Projects\\imiChat):
    EMBEDDING_MODEL, VECTOR_DB_PATH, CHROMA_COLLECTION
"""

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # ── Input files (relative paths resolve against the project root) ── #
    LINKS_FILE: str = "links.txt"
    PROGRAMS_FILE: str = "programs.yaml"
    PROGRAMS_DRAFT_FILE: str = "programs.draft.yaml"
    EVAL_FILE: str = "eval_questions.yaml"
    DATA_DIR: str = "data"

    # ── DeepSeek (OpenAI-compatible API) ─────────────────────────────── #
    DEEPSEEK_API_KEY: Optional[str] = None
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    DEEPSEEK_MODEL: str = "deepseek-flash"      # = DeepSeek-V4.1-Flash
    DEEPSEEK_THINKING: bool = False             # non-thinking: faster, honours temperature
    DEEPSEEK_MAX_WORKERS: int = 8
    DEEPSEEK_TIMEOUT: float = 180.0
    DEEPSEEK_MAX_RETRIES: int = 4
    # USD per 1M tokens (peak rates) — only used for the cost estimate
    DEEPSEEK_PRICE_INPUT_HIT: float = 0.006
    DEEPSEEK_PRICE_INPUT_MISS: float = 0.30
    DEEPSEEK_PRICE_OUTPUT: float = 1.20

    # ── Embeddings (must match the chatbot) ──────────────────────────── #
    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-large"
    EMBEDDING_DEVICE: str = "cpu"               # or "cuda"
    EMBEDDING_BATCH_SIZE: int = 16
    EMBED_MAX_TOKENS: int = 512                 # E5 limit incl. "passage: " + special tokens
    EMBED_QUESTIONS: bool = True                # add DeepSeek questions to the embedded text

    # ── Vector DB (shared with the chatbot) ──────────────────────────── #
    VECTOR_DB_PATH: str = "../imiChat/data/vector_store"
    CHROMA_COLLECTION: str = "university_knowledge"

    # ── Crawling ─────────────────────────────────────────────────────── #
    CRAWL_DELAY: float = 1.0                    # seconds between requests to one host
    CRAWL_TIMEOUT: float = 30.0
    CRAWL_USER_AGENT: str = "IMIChatScraper/2.0 (university RAG indexer)"
    CRAWL_ALLOWED_DOMAINS: list[str] = ["imi.pmf.kg.ac.rs"]
    CRAWL_EXCLUDE_PATTERNS: list[str] = [
        "/moodle", "/administrator", "login", "logout", "back_url=",
        "/rss", "mailto:", "javascript:",
    ]
    CRAWL_DOC_PATTERNS: list[str] = ["/pub/"]   # docs=true follows only these
    CRAWL_MAX_DOCS_PER_PAGE: int = 25
    CRAWL_MAX_PAGES: int = 500
    RESPECT_ROBOTS: bool = True

    # ── Chunking ─────────────────────────────────────────────────────── #
    CHUNK_MAX_TOKENS: int = 300                 # prose
    CHUNK_TABLE_MAX_TOKENS: int = 120           # table rows: 1–2 per chunk → precise course hits
    CHUNK_OVERLAP_TOKENS: int = 40
    MIN_CHUNK_CHARS: int = 30
    LLM_BATCH_CHUNKS: int = 12                  # chunks per DeepSeek enrichment call

    # ── Retrieval (search / eval — mirrors the chatbot) ──────────────── #
    TOP_K_RESULTS: int = 5
    RETRIEVAL_FETCH_K: int = 20
    SIMILARITY_THRESHOLD: float = 0.7

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def path(self, value: str) -> Path:
        """Resolve a configured path relative to the project root."""
        p = Path(value)
        return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()

    @property
    def data_dir(self) -> Path:
        return self.path(self.DATA_DIR)


settings = Settings()
