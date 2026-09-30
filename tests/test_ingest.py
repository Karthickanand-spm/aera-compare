import base64
import hashlib
import io
from pathlib import Path

import docx
import openpyxl
import pytest
from PIL import Image

from aera.ingest import IMAGE_MEDIA_TYPES, MAX_IMAGE_SIDE_PX, load_reply

REPLIES = Path(__file__).resolve().parents[1] / "data" / "sample" / "replies"
EXCEL = REPLIES / "A_Deccan_Corrupack_quotation.xlsx"
WORD = REPLIES / "C_Sahyadri_Boxes_offer.docx"


def test_excel_sample_keeps_cell_refs_and_all_sheets():
    p = load_reply(EXCEL)
    assert p.kind == "excel"
    assert p.sha256 == hashlib.sha256(EXCEL.read_bytes()).hexdigest()
    assert "=== Sheet: Quotation ===" in p.text
    assert "=== Sheet: Vendor Info ===" in p.text
    assert "Quotation!E8: 36.91" in p.text
    assert "Vendor Info!B2: 600 MT per month" in p.text
    # The price is shown in the same row line as its item, so the model can pair them.
    row8 = next(line for line in p.text.splitlines() if "Quotation!A8:" in line)
    assert "Quotation!B8: Rice cooker 1.8L" in row8


def test_excel_sample_notes_merged_headers():
    p = load_reply(EXCEL)
    assert "Merged cells" in p.text
    assert 'E6:G6: "Rate (Rs / Nos)"' in p.text


def test_excel_skips_empty_rows_and_keeps_formulas(tmp_path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Q"
    ws["A1"] = "Item"
    ws["A3"] = "Box"
    ws["B3"] = 10
    ws["C3"] = "=B3*2"  # openpyxl saves no calculated value, so the formula must show
    path = tmp_path / "q.xlsx"
    wb.save(path)

    lines = load_reply(path).text.splitlines()
    assert "Q!A1: Item" in lines
    assert not any(line.startswith("Q!A2") for line in lines)
    assert "Q!A3: Box | Q!B3: 10 | Q!C3: =B3*2 (formula, no saved value)" in lines


def test_word_sample_reads_paragraphs_in_order():
    p = load_reply(WORD)
    assert p.kind == "word"
    assert p.sha256 == hashlib.sha256(WORD.read_bytes()).hexdigest()
    assert p.text.startswith("SAHYADRI BOXES & CARTONS")
    assert "steam iron carton (310x150x140) at Rs 514" in p.text
    assert p.text.index("3-ply shipper") < p.text.index("5-ply range") < p.text.index("7-ply")


def test_word_keeps_tables_in_document_order(tmp_path):
    d = docx.Document()
    d.add_paragraph("Before table")
    t = d.add_table(rows=2, cols=3)
    t.cell(0, 0).merge(t.cell(0, 1)).text = "Rate"
    t.cell(0, 2).text = "Remarks"
    for i, v in enumerate(["Box", "12.50", "ok"]):
        t.cell(1, i).text = v
    d.add_paragraph("After table")
    path = tmp_path / "offer.docx"
    d.save(path)

    lines = load_reply(path).text.splitlines()
    assert lines == [
        "Before table",
        "[Table 1]",
        "Rate | Remarks",  # merged cell shown once
        "Box | 12.50 | ok",
        "[End of table 1]",
        "After table",
    ]


def test_text_payload_has_text_content_block():
    p = load_reply(WORD)
    assert len(p.content_blocks) == 1
    block = p.content_blocks[0]
    assert block["type"] == "text"
    assert WORD.name in block["text"] and p.text in block["text"]


def test_email_headers_and_body():
    p = load_reply(REPLIES / "E_Ganesh_Packaging_email.eml")
    assert p.kind == "email"
    assert p.text.startswith("From: ")
    assert "\nDate: Thu, 25 Sep 2026 18:42:10 +0530\n" in p.text  # exactly as the vendor sent it
    assert "\nSubject: " in p.text
    assert "₹42/kg" in p.text
    assert "Content-Type" not in p.text


def test_pdf_is_document_block():
    path = REPLIES / "B_Indus_Packaging_quotation.pdf"
    p = load_reply(path)
    assert p.kind == "pdf" and p.text is None
    block = p.content_blocks[0]
    assert block["type"] == "document"
    assert block["source"]["media_type"] == "application/pdf"
    assert base64.b64decode(block["source"]["data"]) == path.read_bytes()


def test_large_image_is_downscaled_but_hash_is_of_original(tmp_path):
    path = tmp_path / "photo.jpg"
    Image.new("RGB", (3000, 1500), "white").save(path)
    p = load_reply(path)
    assert p.kind == "image"
    assert p.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    sent = Image.open(io.BytesIO(base64.b64decode(p.content_blocks[0]["source"]["data"])))
    assert max(sent.size) == MAX_IMAGE_SIDE_PX
    assert sent.size == (2000, 1000)


def _small_sample_image() -> Path | None:
    """First image in the sample replies folder that is small enough to be sent as-is."""
    for path in sorted(REPLIES.iterdir()):
        if path.suffix.lower() in IMAGE_MEDIA_TYPES:
            with Image.open(path) as img:
                if max(img.size) <= MAX_IMAGE_SIDE_PX:
                    return path
    return None


def test_small_image_sent_unchanged():
    path = _small_sample_image()
    if path is None:
        pytest.skip(
            f"No .jpg/.jpeg/.png in {REPLIES} with a long side of {MAX_IMAGE_SIDE_PX}px or less; "
            "add a sample photo to run this test."
        )
    p = load_reply(path)
    assert p.content_blocks[0]["source"]["media_type"] == IMAGE_MEDIA_TYPES[path.suffix.lower()]
    assert base64.b64decode(p.content_blocks[0]["source"]["data"]) == path.read_bytes()


def test_unsupported_file_type(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("hello")
    with pytest.raises(ValueError, match="Unsupported"):
        load_reply(path)
