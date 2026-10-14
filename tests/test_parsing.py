from bs4 import BeautifulSoup

from src.processing.html_parser import parse_html
from src.processing.site_profiles import IMI_PROFILE
from src.processing.tables import render_table
from tests.conftest import INF_URL, MAT_URL, NEWS_URL, fixture_html

DOMAINS = ["imi.pmf.kg.ac.rs"]


def _table(html: str):
    return BeautifulSoup(html, "lxml").find("table")


# --------------------------------------------------------------------------- #
#  tables                                                                      #
# --------------------------------------------------------------------------- #

COURSE_TABLE = """
<table><tbody>
<tr><td colspan="13">I godina</td></tr>
<tr><td rowspan="2" colspan="2">S.</td><td rowspan="2">R.br.</td><td rowspan="2">Š.</td>
    <td rowspan="2">Tip</td><td rowspan="2">Predmet</td><td colspan="5">Fond časova</td><td rowspan="2">ESPB</td></tr>
<tr><td colspan="2">p</td><td colspan="2">v</td><td>don</td></tr>
<tr><td rowspan="2" colspan="2">1.</td><td>1.</td><td>25.IN1073</td><td>TM</td>
    <td><a href="x.pdf">Osnovi programiranja</a></td><td colspan="2">2</td><td colspan="2">2</td><td>0</td><td>6</td></tr>
<tr><td>2.</td><td>19.FIZ060</td><td>TM</td><td>Teorijske osnove informatike</td>
    <td colspan="2">2</td><td colspan="2">2</td><td>0</td><td>5</td></tr>
<tr><td colspan="6">Zbir</td><td colspan="2">24</td><td colspan="2">24</td><td>1</td><td>60</td></tr>
</tbody></table>
"""


def test_course_table_rows_keep_column_meaning():
    rows = render_table(
        _table(COURSE_TABLE), IMI_PROFILE.table_header_glossary, IMI_PROFILE.table_drop_columns
    ).rows
    assert rows[0] == (
        "I godina — Semestar: 1. | Šifra: 25.IN1073 | Tip: TM | Predmet: Osnovi programiranja | "
        "Fond časova predavanja: 2 | Fond časova vežbe: 2 | Fond časova DON: 0 | ESPB: 6"
    )
    # semester inherited through rowspan, equal neighbours (2 | 2) kept
    assert rows[1].startswith("I godina — Semestar: 1. | Šifra: 19.FIZ060")
    # a colspan label spanning several columns is rendered once, without a column name
    assert rows[2] == "I godina — Zbir | Fond časova predavanja: 24 | Fond časova vežbe: 24 | Fond časova DON: 1 | ESPB: 60"


def test_course_table_gets_one_overview_line_per_caption():
    render = render_table(
        _table(COURSE_TABLE), IMI_PROFILE.table_header_glossary, IMI_PROFILE.table_drop_columns
    )
    # names only — ESPB-dense lists outranked the real rows for ESPB questions
    assert render.overview == ["I godina — predmeti: Osnovi programiranja; Teorijske osnove informatike"]


def test_two_column_table_keeps_row_association():
    html = "<table><tr><td>OAS Matematike</td><td>OAS Informatike</td></tr>" \
           "<tr><td>Profesor matematike</td><td>Računarske nauke</td></tr></table>"
    assert render_table(_table(html)).rows == [
        "OAS Matematike | OAS Informatike", "Profesor matematike | Računarske nauke",
    ]


# --------------------------------------------------------------------------- #
#  IMI program pages                                                           #
# --------------------------------------------------------------------------- #

def _page(url: str, fixture: str):
    return parse_html(fixture_html(fixture), url, url, DOMAINS)


def test_informatika_sections_follow_tabs_not_panel_ids():
    page = _page(INF_URL + "?osnovne", "informatika-studije.html")
    assert page.url == INF_URL                                      # ?osnovne collapsed
    new = page.section("panel2026rac_nauke")
    assert new.breadcrumb == ["Informatika", "Osnovne (Upis 2026/27)", "Računarske nauke"]
    assert new.panel_chain == ["panel3inf_u_akreditaciji", "panel2026rac_nauke"]
    assert new.top_trigger == "tabela_osn_u_akreditaciji_click_inf"
    assert "dva izborna modula" in page.section("panel3inf_u_akreditaciji").text
    assert "tri izborna modula" in page.section("panel3inf").text

    old = page.section("panel6inf")
    assert old.label == "Osnovne akademske studije akreditovane 2011"
    assert old.hidden_by_default and old.reachable
    assert old.breadcrumb[:2] == ["Informatika", "Osnovne"]        # follows the "Stari programi" link


def test_unreachable_panels_are_detected():
    page = _page(INF_URL, "informatika-studije.html")
    unreachable = {s.section_id for s in page.sections if not s.reachable}
    assert {"panel2018infosn", "panel2inf", "panel_stari"} <= unreachable
    assert page.section("panel1inf").reachable and not page.section("panel1inf").hidden_by_default


def test_math_page_copy_pasted_ids_and_leftovers():
    page = _page(MAT_URL, "matematika-studije.html")
    # same id as on the informatics page, different program — labelled by its tab
    assert page.section("panel2019rac_nauke").label == "Teorijska matematika i primene"
    # the informatics 2018 program pasted into the math page is unreachable
    assert not page.section("panel2018infosn").reachable
    assert page.section("panel3_u_akreditaciji").breadcrumb[-1] == "Osnovne (Upis 2027/28)"


def test_no_site_chrome_in_sections():
    page = _page(INF_URL, "informatika-studije.html")
    text = page.full_text
    for chrome in ("Pretraga", "Prirodno-matematički fakultet\nUniverzitet u Kragujevcu", "pogledaj program"):
        assert chrome not in text


def test_news_page_with_malformed_heading():
    page = _page(NEWS_URL, "vest-11836.html")
    assert page.title == "Smerovi na OAS Matematike i OAS Informatike"
    assert "akreditovan u februaru 2026" in page.full_text
    assert any(b.kind == "table" for s in page.sections for b in s.blocks)
