from src.core.retrieval import apply_version_policy, wants_legacy, where_filter
from src.core.vector_store import ChromaStore, sanitize_metadata
from tests.conftest import FakeEmbedder


def test_sanitize_metadata_only_scalars():
    meta = sanitize_metadata({
        "a": None, "b": [], "c": ["x", "y"], "d": True, "e": 3, "f": 1.5, "g": "s" * 3000, "_private": 1,
    })
    assert meta == {"c": "x; y", "d": True, "e": 3, "f": 1.5, "g": meta["g"]}
    assert len(meta["g"]) == 2000


def test_replace_page_upserts_then_deletes_stale(tmp_path):
    store = ChromaStore(tmp_path / "store", "test")
    emb = FakeEmbedder()
    url = "https://imi.pmf.kg.ac.rs/x"

    def write(ids):
        docs = [f"tekst {i}" for i in ids]
        metas = [{"page_url": url, "source": f"{url}#{i}", "status": None, "questions": []} for i in ids]
        return store.replace_page(url, ids, emb.embed_passages(docs), docs, metas)

    assert write(["a", "b", "c"]) == (3, 0)
    assert write(["b", "c", "d"]) == (3, 1)                 # "a" is stale
    assert sorted(store.page_ids(url)) == ["b", "c", "d"]
    other = "https://imi.pmf.kg.ac.rs/y"
    store.replace_page(other, ["z"], emb.embed_passages(["z"]), ["z"], [{"page_url": other}])
    assert store.remove_page(url) == 3
    assert store.count() == 1

    try:
        write(["e", "e"])
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate ids must be rejected")


def test_where_filter_excludes_legacy_and_outdated_inside_chroma(tmp_path):
    store = ChromaStore(tmp_path / "store", "test")
    emb = FakeEmbedder()
    url = "https://imi.pmf.kg.ac.rs/x"
    docs = ["predmeti informatika 2026", "predmeti informatika 2011", "predmeti informatika faq", "predmeti vest"]
    metas = [{"page_url": url, "status": "enrolling", "outdated": False},
             {"page_url": url, "status": "legacy", "outdated": False},
             {"page_url": url, "outdated": True},
             {"page_url": url}]                                   # old-style chunk without the keys
    store.replace_page(url, ["a", "b", "c", "d"], emb.embed_passages(docs), docs, metas)

    query = "predmeti informatika"
    got = store.query(emb.embed_query(query), 10, where=where_filter(query))
    assert sorted(got["ids"][0]) == ["a", "d"]
    old = "predmeti informatika iz 2011"
    assert where_filter(old) is None and len(store.query(emb.embed_query(old), 10, where_filter(old))["ids"][0]) == 4


def test_version_policy_drops_legacy_unless_asked():
    docs = ["2026", "2011", "2019", "2026 kopija"]
    metas = [
        {"status": "enrolling", "dedup_key": "k1"},
        {"status": "legacy", "dedup_key": "k2"},
        {"status": "phase_out", "dedup_key": "k3"},
        {"status": "enrolling", "dedup_key": "k1"},     # same text, same version
    ]
    dists = [0.1, 0.11, 0.12, 0.13]
    kept, _, _ = apply_version_policy("Koliko modula ima informatika?", docs, metas, dists, k=5)
    assert kept == ["2026", "2019"]
    kept, _, _ = apply_version_policy("Predmeti na starom programu informatike", docs, metas, dists, k=5)
    assert kept == ["2026", "2011", "2019"]
    faq = (["FAQ: 3 smera", "2026"], [{"outdated": True}, {"status": "enrolling"}], [0.1, 0.2])
    assert apply_version_policy("Koliko smerova ima informatika?", *faq, k=5)[0] == ["2026"]
    assert wants_legacy("akreditacija 2016") and wants_legacy("program iz 2011. godine")
    assert not wants_legacy("upis 2026/27") and not wants_legacy("studenti upisani 2019")
