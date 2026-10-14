from src.processing.documents import PdfParser, title_from_location


def make_pdf(pages: list[list[tuple[int, str]]]) -> bytes:
    """Minimal PDF: each page is a list of (y, text) lines in Helvetica."""
    objects: list[bytes] = []
    n = len(pages)
    kids = " ".join(f"{4 + 2 * i} 0 R" for i in range(n))
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode())
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for i, lines in enumerate(pages):
        stream = "".join(f"BT /F1 12 Tf 72 {y} Td ({text}) Tj ET\n" for y, text in lines).encode()
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {5 + 2 * i} 0 R >>".encode()
        )
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"endstream")

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def test_pdf_keeps_pages_and_strips_headers_footers():
    pdf = make_pdf([
        [(815, "IMI PMF Kragujevac"), (600, "Prijemni ispit se polaze u junu."), (450, "Univerzitet u Kragujevcu"), (30, "1")],
        [(815, "IMI PMF Kragujevac"), (600, "Upis traje od 1. do 5. jula."), (450, "Univerzitet u Kragujevcu"), (30, "2")],
    ])
    page = PdfParser().parse(pdf, "https://imi.pmf.kg.ac.rs/pub/x/konkurs.pdf", "https://imi.pmf.kg.ac.rs/pub/x/konkurs.pdf")
    assert page.doc_type == "pdf" and page.title == "konkurs"
    assert [s.page for s in page.sections] == [1, 2]
    assert page.sections[0].text == "Prijemni ispit se polaze u junu."
    assert page.sections[1].breadcrumb == ["konkurs", "strana 2"]
    assert "Univerzitet u Kragujevcu" not in page.full_text     # repeated on every page
    assert "IMI PMF" not in page.full_text                      # header zone


def test_title_from_cyrillic_url():
    url = "https://imi.pmf.kg.ac.rs/pub/abc/%D0%98%D0%BD%D1%84%D0%BE%D1%80%D0%BC%D0%B0%D1%86%D0%B8%D1%98%D0%B0.pdf"
    assert title_from_location(url) == "Informacija"
