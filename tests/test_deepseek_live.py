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
  * the reply validates against the project's real `CurationResponse` schema;
  * the SQLite response cache returns the second call without a network hit.

It sends the **real curation prompt** built by `prompts.curate_user`, so a
prompt/schema regression is caught here as well.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from config.settings import settings
from src.core.deepseek_client import DeepSeekClient, LLMError
from src.processing import prompts
from src.processing.schemas import Block, CurationResponse, Page, PageAnalysis, Section

pytestmark = pytest.mark.skipif(
    not os.getenv("RUN_LIVE_LLM"),
    reason="live DeepSeek call — set RUN_LIVE_LLM=1 to enable",
)

SOURCE = (
    "Programom Osnovnih akademskih studija INFORMATIKE predviđena su dva izborna "
    "modula, za koje se studenti opredeljuju pri upisu na 2. godinu studija: "
    "RAČUNARSKE NAUKE i SOFTVERSKO INŽENJERSTVO. Po završetku modula Računarske "
    "nauke student stiče zvanje Diplomirani informatičar – računarske nauke."
)
SECTION = Section(
    section_id="panel3inf_u_akreditaciji",
    dom_path="main/panel3inf_u_akreditaciji",
    label="Osnovne (Upis 2026/27)",
    breadcrumb=["Informatika", "Osnovne (Upis 2026/27)"],
    blocks=[Block(kind="text", text=SOURCE)],
)
PAGE = Page(
    url="https://imi.pmf.kg.ac.rs/informatika-studije",
    display_url="https://imi.pmf.kg.ac.rs/informatika-studije",
    doc_type="web",
    title="Informatika",
    sections=[SECTION],
)
ANALYSIS = PageAnalysis(
    page_type="studijski_program",
    page_summary="Stranica studija informatike prikazuje osnovne studije i njihove module.",
)
PAYLOAD = {
    "section_id": SECTION.section_id,
    "part_label": "1/1",
    "breadcrumb": "Informatika › Osnovne (Upis 2026/27)",
    "version_label": "OAS Informatika — akreditacija 2026 — aktuelni program (upis u toku) (upis od 2026/27)",
    "versioned": True,
    "block_types": "text",
    "text": SOURCE,
}
FACTS = [
    "Pregled verzija studijskog programa Informatika — osnovne akademske studije (OAS):\n"
    "• OAS Informatika — akreditacija 2026 — aktuelni program (upis u toku); upis od 2026/27: "
    "2 modula: Računarske nauke, Softversko inženjerstvo."
]
SYSTEM = prompts.curation_system(40, 220, 320)
USER_PROMPT = prompts.curate_user(PAGE, ANALYSIS, [PAYLOAD], FACTS)


@pytest.fixture
def live_client(tmp_path: Path) -> DeepSeekClient:
    if not settings.DEEPSEEK_API_KEY:
        pytest.skip("DEEPSEEK_API_KEY is not set — nothing to call")
    try:
        return DeepSeekClient(settings, tmp_path / "cache" / "llm.sqlite")
    except LLMError as exc:  # pragma: no cover - only on a bad configuration
        pytest.skip(str(exc))


def test_live_curation_roundtrip(live_client: DeepSeekClient) -> None:
    """Real call → validated `CurationResponse` with sensible Serbian output."""
    result = live_client.complete_json(SYSTEM, USER_PROMPT, CurationResponse, max_tokens=2048)

    assert isinstance(result, CurationResponse)
    assert result.sections, "model returned no sections"
    section = result.sections[0]
    assert section.source_section_id == SECTION.section_id
    assert section.chunks, "model returned no chunks"
    chunk = section.chunks[0]
    assert chunk.content.strip()
    assert chunk.source_evidence, "every chunk needs a source quote"
    assert chunk.source_evidence[0].source_section_id in {SECTION.section_id, ""}
    assert chunk.quality in {"ok", "boilerplate", "garbled", "low_value"}
    assert chunk.outdated is False, "versioned sections must never be flagged outdated"

    print(f"\ncontent  : {chunk.content[:300]}")
    print(f"context  : {chunk.context}")
    print(f"questions: {chunk.questions}")
    print(f"keywords : {chunk.keywords}")
    print(f"evidence : {chunk.source_evidence}")
    print(f"usage    : {live_client.usage.calls} call(s), model(s)={live_client.usage.models}")
    print(f"cost     : ${live_client.usage.cost_usd(settings):.6f}")


def test_live_response_is_cached(tmp_path: Path) -> None:
    """The same prompt twice → one API call, second one served from SQLite."""
    if not settings.DEEPSEEK_API_KEY:
        pytest.skip("DEEPSEEK_API_KEY is not set — nothing to call")

    cache_path = tmp_path / "cache" / "llm.sqlite"
    client = DeepSeekClient(settings, cache_path)

    first = client.complete_json(SYSTEM, USER_PROMPT, CurationResponse, 2048)
    calls_after_first = client.usage.calls
    assert calls_after_first == 1

    second = client.complete_json(SYSTEM, USER_PROMPT, CurationResponse, 2048)
    assert client.usage.calls == calls_after_first, "second call should be a cache hit"
    assert client.usage.cached_calls == 1
    assert second.model_dump() == first.model_dump()
    assert cache_path.exists(), "cache database should have been written"
