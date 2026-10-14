"""
Live DeepSeek integration test.

Everything else in `tests/` runs offline against the fake client in
`conftest.py`.  This module is the opposite: it talks to the **real**
DeepSeek API, so it only runs when a key is actually configured and the
operator opts in.

    export DEEPSEEK_API_KEY=sk-...      # or put it in .env
    RUN_LIVE_LLM=1 conda run -n imi python -m pytest tests/test_deepseek_live.py -v

It is skipped otherwise, which keeps `pytest` offline-safe on CI.

What it proves:
  * `DEEPSEEK_API_KEY` is picked up from `.env` by pydantic-settings;
  * the key authenticates against `DEEPSEEK_BASE_URL`;
  * `DEEPSEEK_MODEL` exists and accepts the *combined* request shape the
    client sends (json mode + temperature + `extra_body={"thinking": ...}`);
  * the reply validates against the project's real `EnrichmentBatch` schema;
  * the SQLite response cache returns the second call without a network hit.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from config.settings import settings
from src.core.deepseek_client import DeepSeekClient, LLMError
from src.processing.prompts import ENRICH_SYSTEM
from src.processing.schemas import EnrichmentBatch

pytestmark = pytest.mark.skipif(
    not os.getenv("RUN_LIVE_LLM"),
    reason="live DeepSeek call — set RUN_LIVE_LLM=1 to enable",
)

# A hand-built miniature of what `prompts.enrich_user` produces.  Keeping it
# literal means the test stays valid even if the real prompt builder moves on.
USER_PROMPT = json.dumps(
    {
        "page": {"url": "https://imi.pmf.kg.ac.rs/informatika-studije", "type": "studije"},
        "section": ["Informatika", "Osnovne (Upis 2026/27)"],
        "program_version": "OAS Informatika — akreditacija 2026, enrolling",
        "page_facts": ["Program OAS Informatika trenutno upisuje 2 modula."],
        "chunks": [
            {
                "id": "c0",
                "text": (
                    "Studijski program Informatika, osnovne akademske studije, "
                    "upis 2026/27. Program ima dva modula: Računarske nauke i "
                    "Informacione tehnologije."
                ),
            }
        ],
    },
    ensure_ascii=False,
)


@pytest.fixture
def live_client(tmp_path: Path) -> DeepSeekClient:
    if not settings.DEEPSEEK_API_KEY:
        pytest.skip("DEEPSEEK_API_KEY is not set — nothing to call")
    try:
        return DeepSeekClient(settings, tmp_path / "cache" / "llm.sqlite")
    except LLMError as exc:  # pragma: no cover - only on a bad configuration
        pytest.skip(str(exc))


def test_live_enrichment_roundtrip(live_client: DeepSeekClient) -> None:
    """Real call → validated `EnrichmentBatch` with sensible Serbian output."""
    result = live_client.complete_json(
        ENRICH_SYSTEM, USER_PROMPT, EnrichmentBatch, max_tokens=2048
    )

    assert isinstance(result, EnrichmentBatch)
    assert result.chunks, "model returned no chunks"

    chunk = result.chunks[0]
    assert chunk.id == "c0", "the model must echo the chunk id unchanged"
    assert chunk.context, "context sentence should not be empty"
    assert chunk.quality in {
        "ok",
        "boilerplate",
        "garbled",
        "low_value",
    }, f"unexpected quality enum value: {chunk.quality!r}"

    print(f"\ncontext   : {chunk.context}")
    print(f"questions : {chunk.questions}")
    print(f"keywords  : {chunk.keywords}")
    print(f"quality   : {chunk.quality}   outdated: {chunk.outdated}")
    print(f"usage     : {live_client.usage.calls} call(s), model(s)={live_client.usage.models}")
    print(f"cost      : ${live_client.usage.cost_usd(settings):.6f}")


def test_live_response_is_cached(tmp_path: Path) -> None:
    """The same prompt twice → one API call, second one served from SQLite."""
    if not settings.DEEPSEEK_API_KEY:
        pytest.skip("DEEPSEEK_API_KEY is not set — nothing to call")

    cache_path = tmp_path / "cache" / "llm.sqlite"
    client = DeepSeekClient(settings, cache_path)

    first = client.complete_json(ENRICH_SYSTEM, USER_PROMPT, EnrichmentBatch, 2048)
    calls_after_first = client.usage.calls
    assert calls_after_first == 1

    second = client.complete_json(ENRICH_SYSTEM, USER_PROMPT, EnrichmentBatch, 2048)
    assert client.usage.calls == calls_after_first, "second call should be a cache hit"
    assert client.usage.cached_calls == 1
    assert second.model_dump() == first.model_dump()
    assert cache_path.exists(), "cache database should have been written"
