"""Unit tests for the document ingestion pipeline (src/ingestion.py).

Verifies multi-format file ingestion, directory batch scanning, unsupported file
handling, error isolation, repeated rebuild idempotence, and rollback-capable
atomic index replacement. All tests use deterministic fake embeddings and isolated
temporary directories with zero network or model downloads.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pptx

from src.ingestion import (
    IngestionResult,
    _atomic_replace_index,
    ingest_directory,
    ingest_file,
)
from src.vector_store import (
    DEFAULT_FAISS_INDEX_NAME,
    DEFAULT_METADATA_NAME,
    FAISSVectorStore,
)
from tests.test_vector_store import FakeEmbedder


def _create_sample_pdf_bytes(text: str = "Test PDF Document Content") -> bytes:
    """Create a minimal valid in-memory single-page PDF with selectable text."""
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


class TestIngestionPipeline(unittest.TestCase):
    """Test suite covering single-file and directory ingestion pipelines."""

    def setUp(self):
        """Set up isolated temporary directories for documents and persistent data."""
        self.temp_docs_dir = tempfile.TemporaryDirectory()
        self.docs_dir = Path(self.temp_docs_dir.name)

        self.temp_data_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_data_dir.name)

        self.embedder = FakeEmbedder(dimension=4)

    def tearDown(self):
        """Clean up temporary test directories."""
        self.temp_docs_dir.cleanup()
        self.temp_data_dir.cleanup()

    # ------------------------------------------------------------------------
    # Helper File Creators
    # ------------------------------------------------------------------------

    def _create_test_pdf(self, name: str = "report.pdf", text: str = "Quarterly Revenue Report 2026") -> Path:
        p = self.docs_dir / name
        p.write_bytes(_create_sample_pdf_bytes(text))
        return p

    def _create_test_excel(self, name: str = "budget.xlsx") -> Path:
        p = self.docs_dir / name
        df = pd.DataFrame({
            "Dept": ["Engineering", "Operations"],
            "Allocated": [450000, 320000],
        })
        df.to_excel(p, sheet_name="Allocations", index=False, engine="openpyxl")
        return p

    def _create_test_pptx(self, name: str = "presentation.pptx") -> Path:
        p = self.docs_dir / name
        prs = pptx.Presentation()
        slide = prs.slides.add_slide(prs.slide_layouts[1])
        slide.shapes.title.text = "Strategic Roadmap"
        slide.placeholders[1].text = "Expand global customer success operations."
        prs.save(str(p))
        return p

    # ------------------------------------------------------------------------
    # 1. Single File Ingestion Tests
    # ------------------------------------------------------------------------

    def test_ingest_single_pdf_without_save(self):
        """Ingesting a PDF without data_directory returns populated in-memory store."""
        pdf_path = self._create_test_pdf()
        res = ingest_file(pdf_path, embedder=self.embedder)

        self.assertTrue(res.success)
        self.assertEqual(res.total_files_found, 1)
        self.assertEqual(res.files_indexed, ["report.pdf"])
        self.assertGreater(res.total_chunks, 0)
        self.assertEqual(len(res.errors), 0)
        self.assertIsNotNone(res.vector_store)
        self.assertEqual(res.vector_store.count(), res.total_chunks)

    def test_ingest_single_excel_with_save(self):
        """Ingesting an Excel file with data_directory persists valid index files."""
        excel_path = self._create_test_excel()
        res = ingest_file(excel_path, data_directory=self.data_dir, embedder=self.embedder)

        self.assertTrue(res.success)
        self.assertEqual(res.files_indexed, ["budget.xlsx"])
        self.assertGreater(res.total_chunks, 0)

        # Verify disk persistence
        index_file = self.data_dir / DEFAULT_FAISS_INDEX_NAME
        meta_file = self.data_dir / DEFAULT_METADATA_NAME
        self.assertTrue(index_file.is_file())
        self.assertTrue(meta_file.is_file())

        # Verify reloaded store matches
        loaded = FAISSVectorStore.load_local(self.data_dir, embedder=self.embedder)
        self.assertEqual(loaded.count(), res.total_chunks)

    def test_ingest_single_pptx_with_metadata(self):
        """Ingesting a PowerPoint presentation preserves slide metadata."""
        pptx_path = self._create_test_pptx()
        res = ingest_file(pptx_path, embedder=self.embedder)

        self.assertTrue(res.success)
        self.assertGreater(res.total_chunks, 0)
        chunk = res.vector_store.chunks[0]
        self.assertEqual(chunk.source, "presentation.pptx")
        self.assertEqual(chunk.file_type, "pptx")
        self.assertEqual(chunk.slide_number, 1)

    def test_ingest_single_file_unsupported_format(self):
        """Ingesting a single unsupported file returns failure in skipped_files."""
        txt_path = self.docs_dir / "notes.txt"
        txt_path.write_text("Plain text content", encoding="utf-8")

        res = ingest_file(txt_path, embedder=self.embedder)
        self.assertFalse(res.success)
        self.assertEqual(res.total_chunks, 0)
        self.assertEqual(len(res.skipped_files), 1)
        self.assertIn("Unsupported format", res.skipped_files[0]["reason"])

    def test_ingest_single_file_not_found(self):
        """Ingesting a non-existent file returns failure in errors."""
        res = ingest_file(self.docs_dir / "missing.pdf", embedder=self.embedder)
        self.assertFalse(res.success)
        self.assertEqual(len(res.errors), 1)
        self.assertIn("File not found", res.errors[0]["error"])

    # ------------------------------------------------------------------------
    # 2. Directory Batch Ingestion Tests
    # ------------------------------------------------------------------------

    def test_ingest_directory_mixed_supported_and_unsupported(self):
        """Batch ingestion processes supported files and records unsupported in skipped_files."""
        self._create_test_pdf("doc1.pdf")
        self._create_test_excel("doc2.xlsx")
        self._create_test_pptx("doc3.pptx")

        # Unsupported files
        (self.docs_dir / "notes.txt").write_text("notes", encoding="utf-8")
        (self.docs_dir / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n")

        res = ingest_directory(
            documents_directory=self.docs_dir,
            data_directory=self.data_dir,
            embedder=self.embedder,
        )

        self.assertTrue(res.success)
        self.assertEqual(res.total_files_found, 5)
        self.assertEqual(len(res.files_indexed), 3)
        self.assertEqual(set(res.files_indexed), {"doc1.pdf", "doc2.xlsx", "doc3.pptx"})
        self.assertEqual(len(res.skipped_files), 2)
        skipped_names = {s["file"] for s in res.skipped_files}
        self.assertEqual(skipped_names, {"notes.txt", "image.png"})
        self.assertEqual(len(res.errors), 0)

        # Confirm saved index exists
        loaded = FAISSVectorStore.load_local(self.data_dir, embedder=self.embedder)
        self.assertEqual(loaded.count(), res.total_chunks)

    def test_ingest_directory_empty_and_scanned_documents(self):
        """Empty documents and scanned PDFs are reported in errors without halting valid files."""
        # Valid document
        self._create_test_pdf("valid.pdf", "Valid selectable text")

        # 0-byte file (EmptyDocumentError)
        (self.docs_dir / "empty.pdf").write_bytes(b"")

        # Scanned PDF without text layer (ScannedDocumentError)
        scanned_pdf_bytes = (
            b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
            b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
            b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]>>endobj\n"
            b"xref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n"
            b"0000000053 00000 n \n0000000109 00000 n \n"
            b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n168\n%%EOF\n"
        )
        (self.docs_dir / "scanned.pdf").write_bytes(scanned_pdf_bytes)

        res = ingest_directory(
            documents_directory=self.docs_dir,
            data_directory=self.data_dir,
            embedder=self.embedder,
        )

        # success must be False because errors occurred
        self.assertFalse(res.success)
        self.assertEqual(res.files_indexed, ["valid.pdf"])
        self.assertEqual(len(res.errors), 2)
        error_files = {e["file"] for e in res.errors}
        self.assertEqual(error_files, {"empty.pdf", "scanned.pdf"})

        # Valid document chunks should still be indexed and persisted
        self.assertGreater(res.total_chunks, 0)
        loaded = FAISSVectorStore.load_local(self.data_dir, embedder=self.embedder)
        self.assertEqual(loaded.count(), res.total_chunks)

    # ------------------------------------------------------------------------
    # 3. Repeated Rebuilds & Deduplication
    # ------------------------------------------------------------------------

    def test_repeated_rebuilds_prevent_duplicate_chunks(self):
        """Repeated ingestion rebuilds index from scratch, preventing duplicate chunks."""
        self._create_test_pdf("report.pdf", "Company financial report details.")
        self._create_test_excel("budget.xlsx")

        res_1 = ingest_directory(
            documents_directory=self.docs_dir,
            data_directory=self.data_dir,
            embedder=self.embedder,
        )
        self.assertTrue(res_1.success)
        count_1 = res_1.total_chunks
        self.assertGreater(count_1, 0)

        # Run ingestion a second time on the same documents
        res_2 = ingest_directory(
            documents_directory=self.docs_dir,
            data_directory=self.data_dir,
            embedder=self.embedder,
        )
        self.assertTrue(res_2.success)
        count_2 = res_2.total_chunks

        # Chunk count must be identical, NOT doubled
        self.assertEqual(count_1, count_2)

        # Confirm on-disk index count
        loaded = FAISSVectorStore.load_local(self.data_dir, embedder=self.embedder)
        self.assertEqual(loaded.count(), count_1)

    # ------------------------------------------------------------------------
    # 4. Protection of Existing Index on Missing or Failed Ingestion
    # ------------------------------------------------------------------------

    def test_empty_directory_preserves_existing_index(self):
        """An empty documents directory must not overwrite or wipe out an existing valid index."""
        # 1. First build a valid index
        self._create_test_pdf("initial.pdf", "Existing knowledge base data.")
        init_res = ingest_directory(
            documents_directory=self.docs_dir,
            data_directory=self.data_dir,
            embedder=self.embedder,
        )
        self.assertTrue(init_res.success)
        init_chunks = init_res.total_chunks

        # 2. Point to an empty directory
        empty_docs_dir = self.docs_dir / "empty_folder"
        empty_docs_dir.mkdir()

        res_empty = ingest_directory(
            documents_directory=empty_docs_dir,
            data_directory=self.data_dir,
            embedder=self.embedder,
        )
        self.assertFalse(res_empty.success)
        self.assertEqual(res_empty.total_chunks, 0)

        # 3. Verify existing index is preserved intact
        loaded = FAISSVectorStore.load_local(self.data_dir, embedder=self.embedder)
        self.assertEqual(loaded.count(), init_chunks)

    def test_all_files_failed_preserves_existing_index(self):
        """If all files in documents directory fail, the existing index is preserved."""
        # 1. Build initial index
        self._create_test_pdf("good.pdf", "Good content")
        init_res = ingest_directory(
            documents_directory=self.docs_dir,
            data_directory=self.data_dir,
            embedder=self.embedder,
        )
        self.assertTrue(init_res.success)
        init_chunks = init_res.total_chunks

        # 2. Replace good.pdf with a 0-byte corrupt file
        (self.docs_dir / "good.pdf").unlink()
        (self.docs_dir / "corrupt.pdf").write_bytes(b"")

        res_failed = ingest_directory(
            documents_directory=self.docs_dir,
            data_directory=self.data_dir,
            embedder=self.embedder,
        )
        self.assertFalse(res_failed.success)
        self.assertEqual(res_failed.total_chunks, 0)
        self.assertGreater(len(res_failed.errors), 0)

        # 3. Existing index in data_dir remains preserved
        loaded = FAISSVectorStore.load_local(self.data_dir, embedder=self.embedder)
        self.assertEqual(loaded.count(), init_chunks)

    # ------------------------------------------------------------------------
    # 5. Rollback on Failed Replacement or Verification
    # ------------------------------------------------------------------------

    def test_atomic_replacement_rollback_on_copy_error(self):
        """If file copying fails during staging-to-target copy, the previous index is rolled back.

        The failure is injected by inspecting the *source* path of each shutil.copy2 call:
        copies whose source is inside the staging directory are the staging-to-target transfer;
        copies whose source is inside the target directory are either backup creation or rollback
        restore. Only the first category is made to fail, so backup and rollback copies succeed.
        This avoids the fragile call-count heuristic.
        """
        # 1. Establish existing working index
        self._create_test_pdf("original.pdf", "Original index content")
        init_res = ingest_directory(
            documents_directory=self.docs_dir,
            data_directory=self.data_dir,
            embedder=self.embedder,
        )
        self.assertTrue(init_res.success)
        init_chunks = init_res.total_chunks

        # 2. Prepare staging directory with new (different) data
        with tempfile.TemporaryDirectory() as staging_tmp:
            staging_path = Path(staging_tmp)
            new_store = FAISSVectorStore(embedder=self.embedder)
            new_store.add_chunks(init_res.vector_store.chunks[:1])
            new_store.save(staging_path)

            real_copy2 = shutil.copy2
            staging_dir_str = str(staging_path)

            def fail_on_staging_source(src, dst):
                # Fail precisely when copying FROM the staging directory to the target.
                # Backup creation (target→backup) and rollback restore (backup→target)
                # both have source paths outside staging_dir, so they succeed normally.
                if str(src).startswith(staging_dir_str):
                    raise OSError("Simulated disk failure during staging-to-target copy")
                return real_copy2(src, dst)

            with patch("shutil.copy2", side_effect=fail_on_staging_source):
                with self.assertRaises(RuntimeError) as ctx:
                    _atomic_replace_index(staging_path, self.data_dir, 1, embedder=self.embedder)
                self.assertIn("Index replacement failed and was rolled back", str(ctx.exception))

        # 3. Verify original index was restored from backup
        restored = FAISSVectorStore.load_local(self.data_dir, embedder=self.embedder)
        self.assertEqual(restored.count(), init_chunks)

    # ------------------------------------------------------------------------
    # 6. Metadata Preservation and Retrieval from Saved Store
    # ------------------------------------------------------------------------

    def test_metadata_preservation_across_all_formats_in_saved_store(self):
        """Ingested multi-format documents retain all source metadata upon reload."""
        self._create_test_pdf("annual.pdf", "Annual revenue figures")
        self._create_test_excel("budget.xlsx")
        self._create_test_pptx("deck.pptx")

        res = ingest_directory(
            documents_directory=self.docs_dir,
            data_directory=self.data_dir,
            embedder=self.embedder,
        )
        self.assertTrue(res.success)

        # Reload store from disk
        store = FAISSVectorStore.load_local(self.data_dir, embedder=self.embedder)

        # Verify all 3 file formats are represented with proper attributes
        file_types = {c.file_type for c in store.chunks}
        self.assertEqual(file_types, {"pdf", "xlsx", "pptx"})

        pdf_chunk = next(c for c in store.chunks if c.file_type == "pdf")
        self.assertEqual(pdf_chunk.source, "annual.pdf")
        self.assertEqual(pdf_chunk.page, 1)

        xlsx_chunk = next(c for c in store.chunks if c.file_type == "xlsx")
        self.assertEqual(xlsx_chunk.source, "budget.xlsx")
        self.assertEqual(xlsx_chunk.sheet_name, "Allocations")
        self.assertEqual(xlsx_chunk.row_index, 2)

        pptx_chunk = next(c for c in store.chunks if c.file_type == "pptx")
        self.assertEqual(pptx_chunk.source, "deck.pptx")
        self.assertEqual(pptx_chunk.slide_number, 1)


class TestIngestionHardening(unittest.TestCase):
    """Targeted hardening tests for edge cases not covered by the main pipeline suite."""

    def setUp(self):
        """Isolated temporary directories and a deterministic FakeEmbedder."""
        self.temp_docs_dir = tempfile.TemporaryDirectory()
        self.docs_dir = Path(self.temp_docs_dir.name)

        self.temp_data_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_data_dir.name)

        self.embedder = FakeEmbedder(dimension=4)

    def tearDown(self):
        self.temp_docs_dir.cleanup()
        self.temp_data_dir.cleanup()

    def _create_test_pdf(self, dest_dir: Path, name: str = "doc.pdf", text: str = "Sample text") -> Path:
        p = dest_dir / name
        p.write_bytes(_create_sample_pdf_bytes(text))
        return p

    def _build_index_in_data_dir(self, text: str = "Initial content for existing index") -> int:
        """Helper: ingest a single PDF into self.data_dir; return chunk count."""
        self._create_test_pdf(self.docs_dir, "seed.pdf", text)
        res = ingest_directory(
            documents_directory=self.docs_dir,
            data_directory=self.data_dir,
            embedder=self.embedder,
        )
        self.assertTrue(res.success, f"Helper index build failed: {res.errors}")
        (self.docs_dir / "seed.pdf").unlink()  # clean up so later docs_dir use starts fresh
        return res.total_chunks

    # -------------------------------------------------------------------------
    # 1. Rollback when staged files copy successfully but validation fails
    # -------------------------------------------------------------------------

    def test_rollback_on_validation_failure_preserves_existing_index(self):
        """Rollback is triggered when files copy without error but post-copy validation raises.

        FAISSVectorStore.load_local is patched inside src.ingestion only, so the validation
        step inside _atomic_replace_index raises while the post-test verification call
        (made directly from the test, not via src.ingestion) succeeds normally.
        """
        # 1. Build a valid index in data_dir
        init_chunks = self._build_index_in_data_dir("Original knowledge base content")

        # 2. Prepare valid staging files from the existing index
        with tempfile.TemporaryDirectory() as staging_tmp:
            staging_path = Path(staging_tmp)
            loaded = FAISSVectorStore.load_local(self.data_dir, embedder=self.embedder)
            staging_store = FAISSVectorStore(embedder=self.embedder)
            staging_store.add_chunks(loaded.chunks)
            staging_store.save(staging_path)

            # 3. Patch load_local *only inside src.ingestion* so the validation call raises
            with patch(
                "src.ingestion.FAISSVectorStore.load_local",
                side_effect=ValueError("Simulated corrupt target after staging copy"),
            ):
                with self.assertRaises(RuntimeError) as ctx:
                    _atomic_replace_index(
                        staging_path, self.data_dir, loaded.count(), embedder=self.embedder
                    )
                self.assertIn("rolled back", str(ctx.exception))

        # 4. Existing index must be fully intact
        restored = FAISSVectorStore.load_local(self.data_dir, embedder=self.embedder)
        self.assertEqual(restored.count(), init_chunks)

    # -------------------------------------------------------------------------
    # 2. Missing staging files raise FileNotFoundError without touching existing index
    # -------------------------------------------------------------------------

    def test_missing_staging_files_raises_and_leaves_existing_index_intact(self):
        """_atomic_replace_index raises FileNotFoundError immediately when staging files are absent.

        The check occurs before any file is written to the target directory, so the
        existing index must be completely untouched.
        """
        # 1. Build a valid existing index
        init_chunks = self._build_index_in_data_dir("Protected existing index content")

        # 2. Empty staging directory — no index.faiss or metadata.json present
        with tempfile.TemporaryDirectory() as empty_staging:
            with self.assertRaises(FileNotFoundError) as ctx:
                _atomic_replace_index(
                    Path(empty_staging), self.data_dir, init_chunks, embedder=self.embedder
                )
            self.assertIn("Staged index files missing", str(ctx.exception))

        # 3. Existing index is completely untouched
        loaded = FAISSVectorStore.load_local(self.data_dir, embedder=self.embedder)
        self.assertEqual(loaded.count(), init_chunks)

    # -------------------------------------------------------------------------
    # 3. ingest_file() preserves a pre-existing index when replacement fails
    # -------------------------------------------------------------------------

    def test_ingest_file_preserves_pre_existing_index_on_replacement_failure(self):
        """ingest_file() with data_directory leaves the previous on-disk index intact if the
        atomic replacement step fails (validation raises after staging copy succeeds).

        This exercises the full ingest_file() → _atomic_replace_index() path rather than
        calling _atomic_replace_index() directly.
        """
        # 1. Build initial on-disk index via ingest_file
        pdf_path = self._create_test_pdf(self.docs_dir, "source.pdf", "Source document content")
        init_res = ingest_file(pdf_path, data_directory=self.data_dir, embedder=self.embedder)
        self.assertTrue(init_res.success)
        init_chunks = init_res.total_chunks
        self.assertGreater(init_chunks, 0)

        # 2. Attempt a second ingest of the same file; make validation fail so rollback fires
        with patch(
            "src.ingestion.FAISSVectorStore.load_local",
            side_effect=ValueError("Validation failure injected by test"),
        ):
            with self.assertRaises(RuntimeError):
                ingest_file(pdf_path, data_directory=self.data_dir, embedder=self.embedder)

        # 3. The pre-existing index must still be readable with the original chunk count
        loaded = FAISSVectorStore.load_local(self.data_dir, embedder=self.embedder)
        self.assertEqual(loaded.count(), init_chunks)

    # -------------------------------------------------------------------------
    # 4. recursive=True finds nested documents; recursive=False does not
    # -------------------------------------------------------------------------

    def test_recursive_scan_finds_nested_documents(self):
        """recursive=True discovers files in subdirectories; recursive=False stays at top level.

        Creates one PDF at the top level and one PDF two levels deep, then confirms:
        - recursive=False: total_files_found=1, only the top-level file is indexed.
        - recursive=True:  total_files_found=2, both files are indexed.
        """
        # Top-level document
        self._create_test_pdf(self.docs_dir, "top_level.pdf", "Top-level document content here")

        # Document nested two directories deep
        nested_dir = self.docs_dir / "reports" / "q3"
        nested_dir.mkdir(parents=True)
        nested_pdf = nested_dir / "nested_report.pdf"
        nested_pdf.write_bytes(_create_sample_pdf_bytes("Nested quarterly report content data"))

        # --- Non-recursive scan ---
        res_flat = ingest_directory(
            documents_directory=self.docs_dir,
            data_directory=self.data_dir,
            recursive=False,
            embedder=self.embedder,
        )
        self.assertTrue(res_flat.success)
        self.assertEqual(
            res_flat.total_files_found, 1,
            "Non-recursive should count only the single top-level file",
        )
        self.assertEqual(len(res_flat.files_indexed), 1)
        self.assertIn("top_level.pdf", res_flat.files_indexed)
        self.assertNotIn("nested_report.pdf", res_flat.files_indexed)

        # --- Recursive scan ---
        res_recursive = ingest_directory(
            documents_directory=self.docs_dir,
            data_directory=self.data_dir,
            recursive=True,
            embedder=self.embedder,
        )
        self.assertTrue(res_recursive.success)
        self.assertEqual(
            res_recursive.total_files_found, 2,
            "Recursive scan should find both the top-level and nested file",
        )
        self.assertEqual(len(res_recursive.files_indexed), 2)
        self.assertIn("top_level.pdf", res_recursive.files_indexed)
        self.assertIn("nested_report.pdf", res_recursive.files_indexed)
        self.assertGreater(res_recursive.total_chunks, res_flat.total_chunks)


if __name__ == "__main__":
    unittest.main()
