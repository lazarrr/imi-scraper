"""
deepseek_client.py
~~~~~~~~~~~~~~~~~~
Thin DeepSeek client (official API, OpenAI-compatible).

  • model "deepseek-flash" (= DeepSeek-V4.1-Flash), non-thinking mode by
    default — faster, and the only mode that honours temperature=0
  • JSON output mode; the reply is validated against a pydantic schema,
    with one corrective retry
  • handles: empty content (retry), finish_reason "length" (raises
    OutputTruncated → caller splits the batch), "insufficient_system_
    resource" (backoff); 429 / 5xx / network errors are retried by the
    OpenAI SDK itself (max_retries)
  • every validated reply is cached in SQLite, keyed by model +
    PROMPT_VERSION + prompts → re-runs are free and deterministic
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from src.processing.prompts import PROMPT_VERSION
from src.utils.text import sha1

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    pass


class OutputTruncated(LLMError):
    pass


# --------------------------------------------------------------------------- #
#  Usage accounting                                                            #
# --------------------------------------------------------------------------- #

@dataclass
class Usage:
    calls: int = 0
    cached_calls: int = 0
    failed_calls: int = 0
    prompt_tokens: int = 0
    cache_hit_tokens: int = 0
    completion_tokens: int = 0
    models: set[str] = field(default_factory=set)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, response) -> None:
        usage = getattr(response, "usage", None)
        with self._lock:
            self.calls += 1
            if getattr(response, "model", None):
                self.models.add(response.model)
            if usage is None:
                return
            self.prompt_tokens += usage.prompt_tokens or 0
            self.completion_tokens += usage.completion_tokens or 0
            hit = getattr(usage, "prompt_cache_hit_tokens", None)
            if hit is None and getattr(usage, "prompt_tokens_details", None) is not None:
                hit = getattr(usage.prompt_tokens_details, "cached_tokens", 0)
            self.cache_hit_tokens += hit or 0

    def cost_usd(self, settings) -> float:
        miss = max(0, self.prompt_tokens - self.cache_hit_tokens)
        return (
            self.cache_hit_tokens * settings.DEEPSEEK_PRICE_INPUT_HIT
            + miss * settings.DEEPSEEK_PRICE_INPUT_MISS
            + self.completion_tokens * settings.DEEPSEEK_PRICE_OUTPUT
        ) / 1_000_000


# --------------------------------------------------------------------------- #
#  Response cache                                                              #
# --------------------------------------------------------------------------- #

class LLMCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, body TEXT, created REAL)"
        )
        self._conn.commit()
        self._lock = threading.Lock()

    def get(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT body FROM responses WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def put(self, key: str, body: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO responses (key, body, created) VALUES (?, ?, ?)",
                (key, body, time.time()),
            )
            self._conn.commit()


# --------------------------------------------------------------------------- #
#  Client                                                                      #
# --------------------------------------------------------------------------- #

def _strip_fences(content: str) -> str:
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1]
        if content.rstrip().endswith("```"):
            content = content.rstrip()[:-3]
    return content.strip()


class DeepSeekClient:
    def __init__(self, settings, cache_path: Path, use_cache: bool = True):
        if not settings.DEEPSEEK_API_KEY:
            raise LLMError(
                "DEEPSEEK_API_KEY is not set — put it in .env, or run with --no-llm"
            )
        from openai import OpenAI

        self._client = OpenAI(
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=settings.DEEPSEEK_BASE_URL,
            timeout=settings.DEEPSEEK_TIMEOUT,
            max_retries=settings.DEEPSEEK_MAX_RETRIES,
        )
        self.model = settings.DEEPSEEK_MODEL
        self.thinking = settings.DEEPSEEK_THINKING
        self.max_workers = settings.DEEPSEEK_MAX_WORKERS
        self.cache = LLMCache(cache_path) if use_cache else None
        self.usage = Usage()

    def complete_json(self, system: str, user: str, schema: type[T], max_tokens: int) -> T:
        key = sha1(json.dumps([self.model, PROMPT_VERSION, self.thinking, schema.__name__, system, user]))
        if self.cache is not None:
            cached = self.cache.get(key)
            if cached is not None:
                with self.usage._lock:
                    self.usage.cached_calls += 1
                return schema.model_validate_json(cached)

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        last_error = "no attempt"
        for attempt in range(3):
            kwargs = dict(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                max_tokens=max_tokens,
                extra_body={"thinking": {"type": "enabled" if self.thinking else "disabled"}},
            )
            if not self.thinking:
                kwargs["temperature"] = 0.0
            response = self._client.chat.completions.create(**kwargs)
            self.usage.record(response)

            choice = response.choices[0]
            content = _strip_fences(choice.message.content or "")
            if choice.finish_reason == "length":
                raise OutputTruncated(f"output hit max_tokens={max_tokens}")
            if choice.finish_reason == "insufficient_system_resource":
                last_error = "insufficient_system_resource"
                time.sleep(5 * (attempt + 1))
                continue
            if not content:
                last_error = "empty content"
                time.sleep(1 + attempt)
                continue
            try:
                result = schema.model_validate_json(content)
            except ValidationError as exc:
                last_error = str(exc)[:600]
                messages = messages[:2] + [
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": (
                        f"That json does not match the required shape: {last_error}\n"
                        "Return the corrected json object only."
                    )},
                ]
                continue
            if self.cache is not None:
                self.cache.put(key, result.model_dump_json())
            return result

        with self.usage._lock:
            self.usage.failed_calls += 1
        raise LLMError(f"no valid json from DeepSeek after retries: {last_error}")
