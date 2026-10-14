from src.processing.fetcher import canonicalize_url, display_url, with_param
from src.processing.links import parse_links_file
from src.utils.text import TextIndex, fold, join_units, normalize_text, split_units

DOMAINS = ["imi.pmf.kg.ac.rs"]


# --------------------------------------------------------------------------- #
#  text                                                                        #
# --------------------------------------------------------------------------- #

def test_normalize_fixes_cyrillic_homoglyphs_inside_latin_words():
    # "mаtemаtike" written with Cyrillic а (U+0430), as found on the math page
    assert normalize_text("Studije m\u0430tem\u0430tike") == "Studije matematike"


def test_normalize_transliterates_cyrillic_and_cleans_whitespace():
    text = "Информација  о\u00a0полагању\r\n\r\n\r\n\r\nиспита"
    assert normalize_text(text) == "Informacija o polaganju\n\nispita"


def test_fold_ignores_case_diacritics_quotes_and_dashes():
    assert fold("„Računarske NAUKE“ – Đak") == fold("racunarske nauke - djak")


def test_text_index_exact_and_fuzzy_quotes():
    index = TextIndex("Programom Osnovnih akademskih studija INFORMATIKE su predviđena dva izborna modula, "
                      "za koji se studenti opredeljuju pri upisu na 2. godinu studija.")
    assert index.contains("predviđena dva izborna modula")
    assert index.contains("ПРЕДВИЂЕНА ДВА ИЗБОРНА МОДУЛА")          # Cyrillic quote
    assert index.contains("su predvidjena dva izborna modula za koji se studenti opredeljuju")
    assert not index.contains("predviđena tri izborna modula")
    assert not index.contains("dva")                                   # too short to prove anything


def test_split_and_join_units_keep_line_structure():
    units = split_units("Prva rečenica. Druga rečenica.\nNovi red")
    assert units == [("Prva rečenica.", True), ("Druga rečenica.", False), ("Novi red", True)]
    assert join_units(units) == "Prva rečenica. Druga rečenica.\nNovi red"


# --------------------------------------------------------------------------- #
#  URLs                                                                        #
# --------------------------------------------------------------------------- #

def test_tab_view_params_collapse_to_one_identity():
    base = "https://imi.pmf.kg.ac.rs/informatika-studije"
    for variant in (
        base, base + "/", base + "?osnovne", base + "?master", base + "?doktorske",
        base + "?osnovne_u_akreditaciji", "http://IMI.pmf.kg.ac.rs/informatika-studije#x",
        base + "?utm_source=fb&fbclid=1",
    ):
        assert canonicalize_url(variant, DOMAINS) == base, variant


def test_real_query_pages_stay_distinct():
    a = canonicalize_url("https://imi.pmf.kg.ac.rs/index.php?id=783", DOMAINS)
    b = canonicalize_url("https://imi.pmf.kg.ac.rs/index.php?id=782", DOMAINS)
    assert a != b and a.endswith("id=783")


def test_display_url_keeps_served_form_and_with_param():
    assert display_url("https://imi.pmf.kg.ac.rs/mozesit/?osnovne") == "https://imi.pmf.kg.ac.rs/mozesit/"
    assert with_param("https://imi.pmf.kg.ac.rs/informatika-studije", "osnovne_u_akreditaciji") == \
        "https://imi.pmf.kg.ac.rs/informatika-studije?osnovne_u_akreditaciji"


# --------------------------------------------------------------------------- #
#  links file                                                                  #
# --------------------------------------------------------------------------- #

def test_links_file_parsing(tmp_path):
    path = tmp_path / "links.txt"
    path.write_text(
        "# comment\n\n"
        "https://imi.pmf.kg.ac.rs/informatika-studije\n"
        "https://imi.pmf.kg.ac.rs/upis-studije | depth=1 | docs=true\n"
        "https://imi.pmf.kg.ac.rs/pub/x/Информација о испиту.pdf\n"
        "C:/Projects/imiChat/data/raw/konkurs2026.pdf\n"
        "https://imi.pmf.kg.ac.rs/informatika-studije\n",       # duplicate
        encoding="utf-8",
    )
    specs = parse_links_file(path)
    assert [s.target for s in specs][:3] == [
        "https://imi.pmf.kg.ac.rs/informatika-studije",
        "https://imi.pmf.kg.ac.rs/upis-studije",
        "https://imi.pmf.kg.ac.rs/pub/x/Информација о испиту.pdf",
    ]
    assert len(specs) == 4
    assert (specs[1].depth, specs[1].docs) == (1, True)
    assert (specs[0].depth, specs[0].docs, specs[0].is_file) == (0, False, False)
    assert specs[3].is_file and specs[3].target.endswith("konkurs2026.pdf")


def test_links_file_rejects_unknown_option(tmp_path):
    path = tmp_path / "links.txt"
    path.write_text("https://imi.pmf.kg.ac.rs/x | deep=2\n", encoding="utf-8")
    try:
        parse_links_file(path)
    except ValueError as exc:
        assert "unknown option" in str(exc)
    else:
        raise AssertionError("expected ValueError")
