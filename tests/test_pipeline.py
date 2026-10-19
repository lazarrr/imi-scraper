"""
End-to-end ingest on the saved pages: real parser, registry (programs.yaml),
chunker, report and Chroma store; fake fetcher / DeepSeek / embedder.
"""

from datetime import date
from pathlib import Path

import yaml

from src.core.retrieval import apply_version_policy
from src.core.vector_store import ChromaStore
from src.processing.chunker import TokenCounter
from src.processing.ingest import IngestPipeline
from tests.conftest import INF_URL, MAT_URL, NEWS_URL, FakeClient, FakeEmbedder, FakeFetcher

ROOT = Path(__file__).resolve().parent.parent
CHATBOT_FIELDS = ("source", "filename", "type", "page", "url")


def _pipeline(test_settings, tmp_path, store, client, links=(INF_URL, MAT_URL, NEWS_URL), **kw):
    links_file = tmp_path / "links.txt"
    links_file.write_text("\n".join(links) + "\n", encoding="utf-8")
    test_settings.PROGRAMS_FILE = str(ROOT / "programs.yaml")
    return IngestPipeline(
        test_settings, links_file=links_file, collection="test", db_path=tmp_path / "store",
        today=date(2026, 10, 10), fetcher=FakeFetcher(), client=client,
        embedder=FakeEmbedder(), store=store, token_counter=TokenCounter(None), **kw,
    )


def test_full_ingest_then_unchanged_rerun(test_settings, tmp_path):
    store = ChromaStore(tmp_path / "store", "test")
    report, report_path = _pipeline(test_settings, tmp_path, store, FakeClient()).run()

    assert [p.outcome for p in report.pages] == ["ok", "ok", "ok"]
    got = store.collection.get(include=["metadatas", "documents"])
    metas, docs = got["metadatas"], got["documents"]
    assert len(metas) == store.count() > 100

    # every chunk is chatbot-compatible and scalar-only
    for meta in metas:
        assert all(field in meta for field in CHATBOT_FIELDS)
        assert all(isinstance(v, (str, int, float, bool)) for v in meta.values())
    assert len({m["source"] for m in metas if m["page_url"] == INF_URL}) > 10   # per-section sources

    # status is mandatory on every chunk: the registry status for versioned
    # text, "unknown" for unversioned content and generated version maps.
    statuses = {"enrolling", "phase_out", "upcoming", "legacy", "unknown"}
    assert all(m.get("status") in statuses for m in metas)
    assert all(m["status"] == "unknown" for m in metas if not m.get("version_label"))
    assert any(m["status"] == "enrolling" for m in metas)      # 2026 program
    assert any(m["status"] == "phase_out" for m in metas)      # 2019 program
    assert all(m["status"] == "unknown" for m in metas if m["page_url"] == NEWS_URL)

    # unreachable leftovers are not indexed
    assert not any(m["section_id"] == "panel2018infosn" for m in metas)

    # 2026 and 2019 bachelor intros carry their own version + status + tab link
    by_section: dict[str, list] = {}
    for m, d in zip(metas, docs):
        if m["page_url"] == INF_URL:
            by_section.setdefault(m["section_id"], []).append((m, d))
    new_meta, new_doc = by_section["panel3inf_u_akreditaciji"][0]
    old_meta, _ = by_section["panel3inf"][0]
    assert (new_meta["status"], new_meta["accreditation_year"]) == ("enrolling", 2026)
    assert (old_meta["status"], old_meta["accreditation_year"]) == ("phase_out", 2019)
    assert new_meta["url"].endswith("?osnovne_u_akreditaciji") and old_meta["url"].endswith("?osnovne")
    assert new_doc.startswith("[Informatika › Osnovne (Upis 2026/27) | OAS Informatika — akreditacija 2026")
    assert any("dva izborna modula" in d for _, d in by_section["panel3inf_u_akreditaciji"])
    assert any("tri izborna modula" in d for _, d in by_section["panel3inf"])

    # verified overview chunk
    vmap = [d for m, d in zip(metas, docs) if m.get("doc_kind") == "version_map" and m["level"] == "osnovne"
            and m["page_url"] == INF_URL]
    assert len(vmap) == 1 and "2 modula" in vmap[0] and "3 modula" in vmap[0]

    # DeepSeek curation produced the final documents
    assert all(m.get("curated") is True for m in metas if m.get("doc_kind") == "chunk")
    assert all(m.get("doc_kind") == "chunk" for m in metas if m.get("curated"))
    assert any(m.get("questions") for m in metas)
    assert any(m.get("source_evidence") for m in metas)
    assert report.pages[0].curation_status == "complete"
    cur = report.pages[0].curation
    assert cur["sections_failed"] == 0 and cur["chunks_curated"] > 0
    assert cur["chunks_rejected"] == 0                  # the default fake is honest
    assert all(m.get("curation_version") for m in metas if m.get("curated"))
    # context/questions improve the embedding but are never injected into the
    # chatbot's factual document (the header + curated content only)
    assert new_doc.split("\n", 1)[1].strip()
    assert "deo stranice." not in new_doc

    # the FAQ still saying "3 smera" is flagged against the registry facts → hidden from the chatbot
    faq = [(m, d) for m, d in zip(metas, docs) if m["section_id"] == "panel_faq" and "3 smera" in d]
    assert faq and all(m["outdated"] is True for m, _ in faq)
    assert all("moguće zastarelo" in d.split("\n", 1)[0] for _, d in faq)
    versioned = [m for m in metas if m.get("version_label")]
    assert versioned and not any(m["outdated"] for m in versioned)    # registry dates those, not the model

    # course overview chunks exist for listing questions
    assert any("predmeti: Osnovi programiranja" in d
               for m, d in zip(metas, docs) if m["section_id"] == "panel2026rac_nauke")

    # drafts: the real one is evidence-checked OK, the fabricated one is not used
    drafts = yaml.safe_load((tmp_path / "programs.draft.yaml").read_text(encoding="utf-8"))
    by_label = {d["version_label"]: d for d in drafts}
    assert by_label["DAS Informatika — opšte"]["evidence_ok"] is True
    assert by_label["Izmišljen program"]["evidence_ok"] is False
    assert not any(m.get("version_label") == "Izmišljen program" for m in metas)

    assert "Not indexed — unreachable" in report_path.read_text(encoding="utf-8")

    # curation diagnostics are written next to the report
    import json
    curation = json.loads((report_path.parent / "curation.json").read_text(encoding="utf-8"))
    assert curation and curation[0]["curation_status"] == "complete"
    assert curation[0]["stats"]["chunks_curated"] > 0
    assert curation[0]["stats"]["sections_failed"] == 0

    # retrieval regression: the final curated documents answer representative
    # questions (the hashing embedder is crude, so the queries use the same
    # vocabulary the page uses — this checks the knowledge made it into Chroma)
    emb = FakeEmbedder()
    checks = [
        ("Koliko modula ima OAS Informatika?", [["dva izborna modula"]]),
        ("izborni modula Računarske nauke Softversko inženjerstvo",
         [["Računarske nauke", "Softversko inženjerstvo"]]),
        ("predmeti prvi semestar ESPB Osnovi programiranja",
         [["Predmet: Osnovi programiranja", "ESPB: 6"]]),
        ("razlika stari program akreditacija 2016 novi program 2026",
         [["2 modula"], ["3 modula"]]),
    ]
    # k=100 because FakeEmbedder is a hashing bag-of-words smoke test, not E5;
    # the production ranking is measured by `run.py eval`.
    for query, expected in checks:
        docs = store.query(emb.embed_query(query), 100)["documents"][0]
        for needles in expected:
            assert any(all(needle in doc for needle in needles) for doc in docs), (query, needles)

    # second run: nothing changed → every page skipped, no LLM calls, same data
    count = store.count()
    client = FakeClient()
    report2, _ = _pipeline(test_settings, tmp_path, store, client).run()
    assert [p.outcome for p in report2.pages] == ["skipped"] * 3
    assert client.calls == [] and store.count() == count


def test_page_change_replaces_its_chunks(test_settings, tmp_path):
    store = ChromaStore(tmp_path / "store", "test")
    test_settings.CHUNK_MAX_TOKENS = 120
    _pipeline(test_settings, tmp_path, store, None, links=(NEWS_URL,), no_llm=True).run()
    before = set(store.page_ids(NEWS_URL))
    assert before

    # same page re-chunked with bigger chunks → fewer ids; the extra old ones must be deleted
    test_settings.CHUNK_MAX_TOKENS = 300
    report, _ = _pipeline(test_settings, tmp_path, store, None, links=(NEWS_URL,), no_llm=True, force=True).run()
    after = set(store.page_ids(NEWS_URL))
    assert report.pages[0].deleted == len(before - after) > 0
    assert store.count() == len(after)


def test_chatbot_style_retrieval_prefers_current_version(test_settings, tmp_path):
    store = ChromaStore(tmp_path / "store", "test")
    _pipeline(test_settings, tmp_path, store, None, links=(INF_URL,), no_llm=True).run()
    emb = FakeEmbedder()
    query = "Programom Osnovnih akademskih studija INFORMATIKE predviđena izborna modula"
    res = store.query(emb.embed_query(query), 20)
    docs, metas, _ = apply_version_policy(query, res["documents"][0], res["metadatas"][0], res["distances"][0], k=5)
    statuses = {m.get("status") for m in metas}
    assert "legacy" not in statuses
    assert any("dva izborna modula" in d for d in docs) and any("tri izborna modula" in d for d in docs)
