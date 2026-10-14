from datetime import date
from pathlib import Path

from src.processing.chunker import TokenCounter, chunk_section, pack_overview, pack_units
from src.processing.ingest import course_questions
from src.processing.registry import Registry, check_draft, enrollment_year
from src.processing.schemas import Block, ProgramVersion, RegistryDraftEntry, Section
from src.utils.text import TextIndex
from tests.conftest import INF_URL, MAT_URL

ROOT = Path(__file__).resolve().parent.parent
TODAY = date(2026, 10, 10)


def _registry() -> Registry:
    return Registry.load(ROOT / "programs.yaml", TODAY, ["imi.pmf.kg.ac.rs"])


def _status(reg: Registry, label: str) -> str:
    entry = next(e for e in reg.entries if e.version_label == label)
    return reg.status(entry)


# --------------------------------------------------------------------------- #
#  registry                                                                    #
# --------------------------------------------------------------------------- #

def test_statuses_for_2026_10_10():
    reg = _registry()
    assert _status(reg, "OAS Informatika — akreditacija 2026") == "enrolling"
    assert _status(reg, "OAS Informatika — akreditacija 2019") == "phase_out"
    assert _status(reg, "OAS Informatika — akreditacija 2011") == "legacy"
    assert _status(reg, "OAS Matematika — program za upis od 2027/28") == "upcoming"
    assert _status(reg, "OAS Matematika — važeći program") == "enrolling"
    assert _status(reg, "DAS Informatika (računarske nauke) — akreditacija 2014") == "legacy"


def test_status_moves_with_time():
    entries = [
        ProgramVersion(page=MAT_URL, program="Matematika", level="osnovne", version_label="nova",
                       enrollment_from=2027),
        ProgramVersion(page=MAT_URL, program="Matematika", level="osnovne", version_label="stara",
                       enrollment_from=2019),
    ]
    now = Registry(entries, date(2026, 10, 10))
    later = Registry(entries, date(2027, 7, 1))
    assert [now.status(e) for e in now.entries] == ["upcoming", "enrolling"]
    assert [later.status(e) for e in later.entries] == ["enrolling", "phase_out"]
    assert enrollment_year(date(2027, 5, 31)) == 2026 and enrollment_year(date(2027, 6, 1)) == 2027


def test_nearest_mapped_ancestor_wins():
    reg = _registry()
    assert reg.resolve(INF_URL, ["panel3inf", "panel6inf"]).version_label == "OAS Informatika — akreditacija 2011"
    assert reg.resolve(INF_URL, ["panel3inf", "panel2019rac_nauke"]).version_label == "OAS Informatika — akreditacija 2019"
    assert reg.resolve(INF_URL, ["panel_faq"]) is None


def test_version_map_lists_versions_and_skips_unverified_modules():
    reg = _registry()
    index = TextIndex("Računarske nauke Softversko inženjerstvo Informaciono-komunikacione tehnologije "
                      "Veštačka inteligencija Nauka o podacima")
    maps = {m["level"]: m for m in reg.version_maps(INF_URL, index)}
    oas = maps["osnovne"]["text"]
    assert "akreditacija 2026 — aktuelni program (upis u toku); upis od 2026/27: 2 modula" in oas
    assert "akreditacija 2019 — prethodni program" in oas and "3 modula" in oas
    assert "Stari programi" in oas and "2011" in oas

    poor_index = TextIndex("Računarske nauke")
    missing = {m["level"]: m["missing_modules"] for m in reg.version_maps(INF_URL, poor_index)}
    assert "Softversko inženjerstvo" in missing["osnovne"]


def test_check_draft_requires_evidence_on_page():
    index = TextIndex("Doktorske akademske studije akreditovane 2014. godine")
    good = RegistryDraftEntry(program="Informatika", level="doktorske", version_label="DAS 2014",
                              panels=["p"], evidence=["akreditovane 2014"])
    bad = RegistryDraftEntry(program="Informatika", level="doktorske", version_label="DAS 2015",
                             panels=["p"], evidence=["akreditovane 2015"], modules=["Robotika"])
    assert check_draft(good, index) == (True, [])
    ok, problems = check_draft(bad, index)
    assert not ok and len(problems) == 2


# --------------------------------------------------------------------------- #
#  chunker                                                                     #
# --------------------------------------------------------------------------- #

def test_pack_units_respects_budget_and_overlap():
    counter = TokenCounter(None)
    units = [(f"Rečenica broj {i} ima nekoliko reči u sebi.", False) for i in range(60)]
    chunks = pack_units(units, counter, max_tokens=60, overlap_tokens=15)
    assert len(chunks) > 3
    assert all(counter.count(c) <= 60 for c in chunks)
    # overlap: the last sentence of a chunk starts the next one
    assert chunks[1].startswith(chunks[0].split(". ")[-1].rstrip(".")[:20])


def test_overview_lines_split_by_items_with_caption():
    counter = TokenCounter(None)
    line = "I godina — predmeti: " + "; ".join(f"Predmet broj {i}" for i in range(30))
    parts = pack_overview(line, counter, max_tokens=40)
    assert len(parts) > 1
    assert all(p.startswith("I godina — predmeti: ") and counter.count(p) <= 40 for p in parts)
    assert sum(p.count("Predmet broj") for p in parts) == 30


def test_course_questions_from_rows():
    rows = ("I godina — Semestar: 1. | Šifra: 25.IN1073 | Predmet: Osnovi programiranja | ESPB: 6\n"
            "I godina — Semestar: 1. | Predmet: Izborni predmet iz grupe G1\n"
            "Izborni predmeti grupe G1 — Šifra: 19.IN1066 | Izborni predmeti grupe G1: Engleski jezik A1 | ESPB: 5")
    assert course_questions(rows) == [
        "Koliko ESPB ima predmet Osnovi programiranja?", "U kom semestru se sluša Osnovi programiranja?",
        "Koliko ESPB ima predmet Engleski jezik A1?", "U kom semestru se sluša Engleski jezik A1?",
    ]


def test_tables_pack_by_row_and_heading_only_blocks_vanish():
    counter = TokenCounter(None)
    rows = [f"I godina — Predmet: Predmet {i} | ESPB: 6" for i in range(40)]
    overview = ["I godina — predmeti: " + "; ".join(f"Predmet {i} (6 ESPB)" for i in range(40))]
    section = Section(
        section_id="s", dom_path="s", label="s", breadcrumb=["x"],
        blocks=[
            Block(kind="text", text="Izborni modul: Nauka o podacima"),
            Block(kind="table", text="\n".join(rows), rows=rows, overview=overview,
                  heading_path=["Izborni modul: Nauka o podacima"]),
        ],
    )
    chunks = chunk_section(section, counter, max_tokens=300, overlap_tokens=10, table_max_tokens=30)
    table = [c for c in chunks if c.kind == "table"]
    assert {c.kind for c in chunks} == {"table", "table_overview"}       # heading-only block vanished
    assert all(counter.count(c.text) <= 30 for c in table)             # row-sized table chunks
    assert all(line in rows for c in table for line in c.text.split("\n"))
    assert sum(len(c.text.split("\n")) for c in table) == 40           # no row lost or repeated
    assert all(c.heading_path == ["Izborni modul: Nauka o podacima"] for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))
