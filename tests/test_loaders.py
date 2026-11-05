"""Unit tests for the document loaders module (src/loaders.py).

These tests run completely offline using temporary files, requiring no external
services, model downloads, or API keys.
"""

import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pptx
import pypdf

from src.loaders import (
    DocumentSection,
    DocumentLoaderError,
    UnsupportedFormatError,
    EmptyDocumentError,
    ScannedDocumentError,
    load_pdf,
    load_excel,
    load_powerpoint,
    load_document,
    load_documents_from_directory,
)


def _create_sample_pdf_bytes(text: str = "Test PDF Document Content") -> bytes:
    """Create a valid in-memory single-page PDF with selectable text."""
    content = f"BT /F1 12 Tf 100 700 Td ({text}) Tj ET\n"
    stream_len = len(content.encode("latin1"))

    body = (
        "%PDF-1.4\n"
        "1 0 obj <</Type /Catalog /Pages 2 0 R>> endobj\n"
        "2 0 obj <</Type /Pages /Kids [3 0 R] /Count 1>> endobj\n"
        "3 0 obj <</Type /Page /Parent 2 0 R /Resources <</Font <</F1 4 0 R>>>> "
        "/MediaBox [0 0 612 792] /Contents 5 0 R>> endobj\n"
        "4 0 obj <</Type /Font /Subtype /Type1 /BaseFont /Helvetica>> endobj\n"
        f"5 0 obj <</Length {stream_len}>>\nstream\n"
        f"{content}endstream\nendobj\n"
    )

    xref_offset = len(body.encode("latin1"))
    trailer = (
        "xref\n0 6\n"
        "0000000000 65535 f \n"
        "0000000009 00000 n \n"
        "0000000058 00000 n \n"
        "0000000115 00000 n \n"
        "0000000226 00000 n \n"
        "0000000293 00000 n \n"
        "trailer <</Size 6 /Root 1 0 R>>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    )

    return (body + trailer).encode("latin1")


class TestDocumentLoaders(unittest.TestCase):
    """Test suite verifying robust multi-format document loading."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    # ------------------------------------------------------------------------
    # PDF Loader Tests
    # ------------------------------------------------------------------------

    def test_load_pdf_valid(self):
        """Test loading a standard PDF with selectable digital text."""
        pdf_path = self.test_dir / "sample.pdf"
        pdf_path.write_bytes(_create_sample_pdf_bytes("Company Travel Policy 2026"))

        sections = load_pdf(pdf_path)
        self.assertEqual(len(sections), 1)
        self.assertEqual(sections[0].source, "sample.pdf")
        self.assertEqual(sections[0].file_type, "pdf")
        self.assertEqual(sections[0].page, 1)
        self.assertIn("Company Travel Policy 2026", sections[0].text)

    def test_load_pdf_scanned_without_text(self):
        """Test that a blank/scanned PDF raises ScannedDocumentError with clear message."""
        pdf_path = self.test_dir / "scanned_doc.pdf"
        writer = pypdf.PdfWriter()
        writer.add_blank_page(width=200, height=200)
        with open(pdf_path, "wb") as f:
            writer.write(f)

        with self.assertRaises(ScannedDocumentError) as ctx:
            load_pdf(pdf_path)
        self.assertIn("scanned or image-based", str(ctx.exception))
        self.assertIn("OCR", str(ctx.exception))

    def test_load_pdf_zero_bytes(self):
        """Test that a 0-byte PDF raises EmptyDocumentError."""
        empty_pdf = self.test_dir / "zero.pdf"
        empty_pdf.write_bytes(b"")

        with self.assertRaises(EmptyDocumentError) as ctx:
            load_pdf(empty_pdf)
        self.assertIn("0 bytes", str(ctx.exception))

    def test_load_pdf_extraction_exception_failure(self):
        """Test that genuine extraction exceptions on all pages raise DocumentLoaderError (not ScannedDocumentError)."""
        pdf_path = self.test_dir / "corrupted_stream.pdf"
        pdf_path.write_bytes(_create_sample_pdf_bytes("Some content"))

        with patch("pypdf.PageObject.extract_text", side_effect=RuntimeError("Decompression stream error")):
            with self.assertRaises(DocumentLoaderError) as ctx:
                load_pdf(pdf_path)
            # Must be DocumentLoaderError, not falsely classified as ScannedDocumentError
            self.assertNotIsInstance(ctx.exception, ScannedDocumentError)
            self.assertIn("Extraction failed on all", str(ctx.exception))
            self.assertIn("Decompression stream error", str(ctx.exception))

    def test_load_pdf_partial_page_error(self):
        """Test that partial page extraction errors record warnings in metadata while preserving good pages."""
        pdf_path = self.test_dir / "multi_page.pdf"

        writer = pypdf.PdfWriter()
        writer.add_blank_page(width=200, height=200)
        writer.add_blank_page(width=200, height=200)
        with open(pdf_path, "wb") as f:
            writer.write(f)

        call_count = 0

        def mixed_extract(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "Valid Page 1 Text"
            raise RuntimeError("Corrupt font on page 2")

        with patch("pypdf.PageObject.extract_text", side_effect=mixed_extract):
            sections = load_pdf(pdf_path)
            self.assertEqual(len(sections), 1)
            self.assertEqual(sections[0].text, "Valid Page 1 Text")
            self.assertIn("extraction_warnings", sections[0].metadata)
            self.assertIn("Page 2 extraction error", sections[0].metadata["extraction_warnings"][0])

    # ------------------------------------------------------------------------
    # Excel Loader Tests
    # ------------------------------------------------------------------------

    def test_load_excel_valid(self):
        """Test loading tabular data from a multi-row Excel workbook."""
        excel_path = self.test_dir / "budgets.xlsx"
        df = pd.DataFrame({
            "Department": ["Engineering", "Marketing"],
            "Budget": [500000, 250000],
            "Lead": ["Alice", "Bob"]
        })
        df.to_excel(excel_path, sheet_name="Q4_Summary", index=False, engine="openpyxl")

        sections = load_excel(excel_path)
        self.assertEqual(len(sections), 2)

        # Row 1 check (spreadsheet row 2)
        row1 = sections[0]
        self.assertEqual(row1.source, "budgets.xlsx")
        self.assertEqual(row1.file_type, "xlsx")
        self.assertEqual(row1.sheet_name, "Q4_Summary")
        self.assertEqual(row1.row_index, 2)
        self.assertIn("Department: Engineering", row1.text)
        self.assertIn("Budget: 500000", row1.text)
        self.assertIn("Lead: Alice", row1.text)

        # Row 2 check (spreadsheet row 3)
        row2 = sections[1]
        self.assertEqual(row2.row_index, 3)
        self.assertIn("Department: Marketing", row2.text)

    def test_load_excel_empty_sheet(self):
        """Test that an Excel spreadsheet with only NaN values raises EmptyDocumentError."""
        excel_path = self.test_dir / "empty.xlsx"
        empty_df = pd.DataFrame({"A": [None, None], "B": [None, None]})
        empty_df.to_excel(excel_path, index=False, engine="openpyxl")

        with self.assertRaises(EmptyDocumentError) as ctx:
            load_excel(excel_path)
        self.assertIn("empty", str(ctx.exception).lower())

    def test_load_excel_worksheet_read_failure(self):
        """Test that an error reading any worksheet produces an actionable DocumentLoaderError."""
        excel_path = self.test_dir / "multi_sheet.xlsx"
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            pd.DataFrame({"A": [1, 2]}).to_excel(writer, sheet_name="GoodSheet", index=False)
            pd.DataFrame({"B": [3, 4]}).to_excel(writer, sheet_name="CorruptSheet", index=False)

        real_read_excel = pd.read_excel

        def faulty_read_excel(file_or_obj, sheet_name=None, **kwargs):
            if sheet_name == "CorruptSheet":
                raise RuntimeError("XML syntax error in sheet data stream")
            return real_read_excel(file_or_obj, sheet_name=sheet_name, **kwargs)

        with patch("pandas.read_excel", side_effect=faulty_read_excel):
            with self.assertRaises(DocumentLoaderError) as ctx:
                load_excel(excel_path)
            self.assertIn("CorruptSheet", str(ctx.exception))
            self.assertIn("XML syntax error in sheet data stream", str(ctx.exception))

    # ------------------------------------------------------------------------
    # PowerPoint Loader Tests
    # ------------------------------------------------------------------------

    def test_load_powerpoint_valid(self):
        """Test loading slides with titles, bullet items, and tables."""
        pptx_path = self.test_dir / "strategy.pptx"
        prs = pptx.Presentation()

        # Slide 1: Title and bullet text
        slide1 = prs.slides.add_slide(prs.slide_layouts[1])
        slide1.shapes.title.text = "Q4 Product Strategy"
        slide1.placeholders[1].text = "Focus on cloud deployment and automated indexing."

        # Slide 2: Table
        slide2 = prs.slides.add_slide(prs.slide_layouts[6])  # Blank layout
        table_shape = slide2.shapes.add_table(2, 2, pptx.util.Inches(1), pptx.util.Inches(1), pptx.util.Inches(4), pptx.util.Inches(2))
        table = table_shape.table
        table.cell(0, 0).text = "Region"
        table.cell(0, 1).text = "Target"
        table.cell(1, 0).text = "Europe"
        table.cell(1, 1).text = "100K Users"

        prs.save(str(pptx_path))

        sections = load_powerpoint(pptx_path)
        self.assertEqual(len(sections), 2)

        # Verify Slide 1
        self.assertEqual(sections[0].source, "strategy.pptx")
        self.assertEqual(sections[0].file_type, "pptx")
        self.assertEqual(sections[0].slide_number, 1)
        self.assertIn("Slide Title: Q4 Product Strategy", sections[0].text)
        self.assertIn("Focus on cloud deployment", sections[0].text)

        # Verify Slide 2 (Table)
        self.assertEqual(sections[1].slide_number, 2)
        self.assertIn("Table:", sections[1].text)
        self.assertIn("Europe | 100K Users", sections[1].text)

    def test_load_powerpoint_empty(self):
        """Test that a presentation with zero slides raises EmptyDocumentError."""
        pptx_path = self.test_dir / "empty_deck.pptx"
        prs = pptx.Presentation()
        prs.save(str(pptx_path))

        with self.assertRaises(EmptyDocumentError) as ctx:
            load_powerpoint(pptx_path)
        self.assertIn("0 slides", str(ctx.exception))

    # ------------------------------------------------------------------------
    # Universal Dispatcher and Edge Cases
    # ------------------------------------------------------------------------

    def test_load_document_dispatcher(self):
        """Test that load_document correctly routes to appropriate loader by extension."""
        pdf_path = self.test_dir / "test.pdf"
        pdf_path.write_bytes(_create_sample_pdf_bytes("Dispatcher Test"))

        sections = load_document(pdf_path)
        self.assertEqual(len(sections), 1)
        self.assertEqual(sections[0].file_type, "pdf")

    def test_unsupported_format(self):
        """Test that unsupported extensions raise UnsupportedFormatError."""
        txt_path = self.test_dir / "notes.txt"
        txt_path.write_text("Hello world", encoding="utf-8")

        with self.assertRaises(UnsupportedFormatError) as ctx:
            load_document(txt_path)
        self.assertIn("Unsupported file format '.txt'", str(ctx.exception))

    def test_legacy_format_warning(self):
        """Test that legacy .xls or .ppt formats give actionable conversion advice."""
        legacy_path = self.test_dir / "old_doc.xls"
        legacy_path.write_bytes(b"dummy binary data")

        with self.assertRaises(UnsupportedFormatError) as ctx:
            load_document(legacy_path)
        self.assertIn("Legacy Microsoft Office format '.xls' is not supported", str(ctx.exception))

    def test_file_not_found(self):
        """Test that missing files raise FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            load_document(self.test_dir / "non_existent.pdf")

    # ------------------------------------------------------------------------
    # Directory Batch Loading Tests
    # ------------------------------------------------------------------------

    def test_load_documents_from_directory(self):
        """Test batch loading from a directory with multiple mixed files."""
        # Create 1 valid PDF
        p1 = self.test_dir / "doc1.pdf"
        p1.write_bytes(_create_sample_pdf_bytes("Doc 1 Content"))

        # Create 1 valid Excel
        p2 = self.test_dir / "doc2.xlsx"
        df = pd.DataFrame({"Item": ["A"], "Price": [10]})
        df.to_excel(p2, index=False, engine="openpyxl")

        # Create 1 unreadable 0-byte PDF (to test graceful error capturing)
        p3 = self.test_dir / "bad.pdf"
        p3.write_bytes(b"")

        sections, errors = load_documents_from_directory(self.test_dir)
        self.assertGreaterEqual(len(sections), 2)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["file"], "bad.pdf")


if __name__ == "__main__":
    unittest.main()
