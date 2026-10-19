"""Unit tests for the deterministic fact guards around AI curation."""

from src.processing.factcheck import (
    is_rewritten,
    missing_registry_modules,
    numeric_facts,
    unsupported_facts,
)


def test_numeric_facts_cover_codes_dates_and_written_numbers():
    text = "Upis 2026/27, rok 15.09.2026, predmet 25.IN1073 nosi 6 ESPB (dva modula)."
    facts = numeric_facts(text)
    assert {"2026", "27", "15", "09", "6", "2"} <= facts
    assert any("in1073" in fact for fact in facts)


def test_unsupported_facts_catch_a_changed_credit_or_year():
    source = "Predmet 25.IN1073 nosi 6 ESPB. Upis traje do 15. septembra 2026."
    assert unsupported_facts("Predmet 25.IN1073 nosi 6 ESPB.", source) == []
    assert unsupported_facts("Predmet 25.IN1073 nosi 8 ESPB.", source) == ["8"]
    assert unsupported_facts("Upis traje do 15. septembra 2031.", source) == ["2031"]


def test_unsupported_code_is_caught_even_though_its_digits_exist():
    source = "Šifra predmeta je 25.IN1073 i nosi 6 ESPB."
    bad = unsupported_facts("Šifra predmeta je 25.IN9999.", source)
    assert any("in9999" in token for token in bad) and "9999" in bad


def test_date_formats_are_comparable():
    source = "Rok za prijavu je 15. septembar 2026."
    assert unsupported_facts("Rok za prijavu je 15.09.2026.", source) == []
    assert unsupported_facts("Rok za prijavu je 15. septembra 2026.", "Rok je 15.09.2026.") == []


def test_missing_registry_modules_reports_dropped_names():
    source = "Moduli su Računarske nauke i Softversko inženjerstvo."
    assert missing_registry_modules(
        ["Računarske nauke", "Softversko inženjerstvo"], source,
        ["Program ima modul Računarske nauke."],
    ) == ["Softversko inženjerstvo"]
    assert missing_registry_modules(["Nauka o podacima"], source, ["x"]) == []


def test_is_rewritten_is_folded_containment():
    source = "Upis traje do 15. septembra 2026."
    assert is_rewritten(source, source) is False
    assert is_rewritten("Prijava traje do 15. septembra 2026.", source) is True
