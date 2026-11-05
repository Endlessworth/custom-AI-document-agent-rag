"""Unit tests for the Gradio UI module (src/app.py).

All tests are 100% offline and deterministic. Tests exercise UI formatting and
handler functions directly with mocked backend pipelines and zero network requests,
model downloads, or Gradio web server launches.
"""

import unittest
from unittest.mock import MagicMock, patch

import gradio as gr

from src.app import (
    build_app,
    format_citations_for_display,
    handle_ingestion,
    handle_query,
)
from src.generator import (
    NO_EVIDENCE_ANSWER,
    AnswerResult,
    GeminiAuthError,
    GeminiGenerationError,
    GeminiQuotaError,
    GeminiUnavailableError,
)
from src.ingestion import IngestionResult
from src.retrieval import (
    VectorStoreCorruptedError,
    VectorStoreNotFoundError,
)
from src.vector_store import SentenceTransformerEmbedder


class TestAppUI(unittest.TestCase):
    """Test suite for Gradio UI handlers, citation formatting, and app structure."""

    # ------------------------------------------------------------------------
    # 1. Citation Formatting for Display Tests
    # ------------------------------------------------------------------------

    def test_format_citations_for_display_with_citations(self):
        """Structured citations are formatted into clean Markdown with locations and scores."""
        citations = [
            {
                "source": "annual_report.pdf",
                "location": "Page 3",
                "score": 0.9512,
                "snippet": "Quarterly fiscal report details revenue of 50 million.",
            },
            {
                "source": "budget.xlsx",
                "location": "Sheet 'Allocations', Row 12",
                "score": 0.8540,
                "snippet": "Dept: Engineering | Allocated: 450000",
            },
        ]

        md = format_citations_for_display(citations)

        self.assertIn("### 📚 Source Citations (2)", md)
        self.assertIn("**1. annual_report.pdf**", md)
        self.assertIn("`Page 3`", md)
        self.assertIn("0.9512", md)
        self.assertIn("Quarterly fiscal report details revenue", md)

        self.assertIn("**2. budget.xlsx**", md)
        self.assertIn("`Sheet 'Allocations', Row 12`", md)
        self.assertIn("Dept: Engineering", md)

    def test_format_citations_for_display_empty(self):
        """Empty citation list returns clear fallback markdown."""
        md = format_citations_for_display([])
        self.assertEqual(md, "*No source citations available.*")

    # ------------------------------------------------------------------------
    # 2. Ingestion Handler Tests
    # ------------------------------------------------------------------------

    @patch("src.ingestion.ingest_directory")
    def test_handle_ingestion_success(self, mock_ingest_dir):
        """Successful ingestion reports total chunks, indexed files, and sections."""
        mock_ingest_dir.return_value = IngestionResult(
            success=True,
            total_files_found=3,
            files_indexed=["doc1.pdf", "doc2.xlsx"],
            total_sections=5,
            total_chunks=15,
            errors=[],
            skipped_files=[],
            data_directory="data",
        )

        output = handle_ingestion(documents_dir="documents", data_dir="data")

        self.assertIn("### ✅ Ingestion Complete (15 chunks indexed)", output)
        self.assertIn("`doc1.pdf`", output)
        self.assertIn("`doc2.xlsx`", output)
        self.assertIn("Total Sections Extracted:** 5", output)
        mock_ingest_dir.assert_called_once_with(
            documents_directory="documents",
            data_directory="data",
            recursive=False,
        )

    @patch("src.ingestion.ingest_directory")
    def test_handle_ingestion_partial_and_errors(self, mock_ingest_dir):
        """Partial ingestion clearly surfaces skipped files and file errors."""
        mock_ingest_dir.return_value = IngestionResult(
            success=False,
            total_files_found=3,
            files_indexed=["good.pdf"],
            total_sections=2,
            total_chunks=6,
            errors=[{"file": "corrupt.pdf", "error": "EmptyDocumentError"}],
            skipped_files=[{"file": "notes.txt", "reason": "Unsupported format '.txt'"}],
            data_directory="data",
        )

        output = handle_ingestion(documents_dir="documents", data_dir="data")

        self.assertIn("### ⚠️ Partial Ingestion (6 chunks indexed with errors)", output)
        self.assertIn("`corrupt.pdf`: EmptyDocumentError", output)
        self.assertIn("`notes.txt`: Unsupported format '.txt'", output)

    def test_handle_ingestion_empty_documents_dir(self):
        """Empty documents directory argument returns clean validation error."""
        output = handle_ingestion(documents_dir="")
        self.assertIn("Documents directory path cannot be empty", output)

    @patch("src.ingestion.ingest_directory")
    def test_handle_ingestion_unexpected_exception(self, mock_ingest_dir):
        """Unexpected ingestion exception is reported safely without crashing."""
        mock_ingest_dir.side_effect = RuntimeError("Disk IO failure")
        output = handle_ingestion(documents_dir="documents")
        self.assertIn("Ingestion failed", output)
        self.assertIn("Disk IO failure", output)

    # ------------------------------------------------------------------------
    # 3. Query Handler Tests
    # ------------------------------------------------------------------------

    def test_handle_query_empty_question(self):
        """Empty question returns guidance and empty citations without querying backend."""
        ans, cit, status = handle_query(query="   ")
        self.assertIn("Please enter a question", ans)
        self.assertEqual(cit, "*No source citations available.*")
        self.assertIn("Empty question", status)

    def test_handle_query_invalid_top_k(self):
        """Invalid top_k returns clean parameter error."""
        ans, cit, status = handle_query(query="valid query", top_k=0)
        self.assertIn("Invalid top_k", ans)
        self.assertIn("Error", status)

    @patch("src.app.load_vector_store")
    def test_handle_query_missing_vector_store(self, mock_load_store):
        """Missing vector store index is handled gracefully with clear guidance."""
        mock_load_store.side_effect = VectorStoreNotFoundError(
            "Vector store directory not found: 'data'. Please run document ingestion first."
        )

        ans, cit, status = handle_query(query="What is revenue?")
        self.assertIn("No document index found", ans)
        self.assertIn("Please run document ingestion first", status)
        self.assertEqual(cit, "*No source citations available.*")

    @patch("src.app.load_vector_store")
    def test_handle_query_corrupted_vector_store(self, mock_load_store):
        """Corrupted index is reported clearly without crashing."""
        mock_load_store.side_effect = VectorStoreCorruptedError(
            "Incomplete vector store in 'data': missing FAISS index 'index.faiss'."
        )

        ans, cit, status = handle_query(query="What is revenue?")
        self.assertIn("Vector store index is corrupted", ans)
        self.assertIn("missing FAISS index", status)

    @patch("src.app.load_vector_store")
    def test_handle_query_empty_vector_store(self, mock_load_store):
        """Empty vector store (0 chunks) returns NO_EVIDENCE_ANSWER."""
        mock_store = MagicMock()
        mock_store.count.return_value = 0
        mock_load_store.return_value = mock_store

        ans, cit, status = handle_query(query="What is revenue?")
        self.assertEqual(ans, NO_EVIDENCE_ANSWER)
        self.assertIn("0 chunks", status)

    @patch("src.app.answer_query")
    @patch("src.app.load_vector_store")
    def test_handle_query_success(self, mock_load_store, mock_answer_query):
        """Successful answer query formats answer, citations, and model status."""
        mock_store = MagicMock()
        mock_store.count.return_value = 10
        mock_load_store.return_value = mock_store

        mock_answer_query.return_value = AnswerResult(
            answer="Revenue reached 50 million in 2026.",
            query="What is the revenue?",
            citations=[
                {
                    "source": "report.pdf",
                    "location": "Page 3",
                    "score": 0.95,
                    "snippet": "Revenue reached 50 million.",
                }
            ],
            context="Some context",
            model="gemini-2.5-flash",
            sources_used=["report.pdf"],
            grounded=True,
        )

        ans, cit, status = handle_query(query="What is the revenue?")

        self.assertEqual(ans, "Revenue reached 50 million in 2026.")
        self.assertIn("**1. report.pdf**", cit)
        self.assertIn("`Page 3`", cit)
        self.assertIn("gemini-2.5-flash", status)
        self.assertIn("1 sources retrieved", status)

    @patch("src.app.answer_query")
    @patch("src.app.load_vector_store")
    def test_handle_query_no_evidence_response(self, mock_load_store, mock_answer_query):
        """No-evidence response is clearly displayed with ungrounded status."""
        mock_store = MagicMock()
        mock_store.count.return_value = 10
        mock_load_store.return_value = mock_store

        mock_answer_query.return_value = AnswerResult(
            answer=NO_EVIDENCE_ANSWER,
            query="Unrelated query",
            citations=[],
            context="",
            model="none",
            sources_used=[],
            grounded=False,
        )

        ans, cit, status = handle_query(query="Unrelated query")

        self.assertEqual(ans, NO_EVIDENCE_ANSWER)
        self.assertEqual(cit, "*No source citations available.*")
        self.assertIn("No matching evidence found", status)

    @patch("src.app.answer_query")
    @patch("src.app.load_vector_store")
    def test_handle_query_gemini_auth_error(self, mock_load_store, mock_answer_query):
        """Authentication error is presented safely without crashing."""
        mock_store = MagicMock()
        mock_store.count.return_value = 5
        mock_load_store.return_value = mock_store

        mock_answer_query.side_effect = GeminiAuthError(
            "GEMINI_API_KEY environment variable is missing or empty."
        )

        ans, cit, status = handle_query(query="Query")
        self.assertIn("Authentication Error", ans)
        self.assertIn("GEMINI_API_KEY", status)

    @patch("src.app.answer_query")
    @patch("src.app.load_vector_store")
    def test_handle_query_gemini_quota_error(self, mock_load_store, mock_answer_query):
        """Quota exceeded error is presented with clean guidance."""
        mock_store = MagicMock()
        mock_store.count.return_value = 5
        mock_load_store.return_value = mock_store

        mock_answer_query.side_effect = GeminiQuotaError(
            "Gemini API quota or rate limit exceeded (HTTP 429)."
        )

        ans, cit, status = handle_query(query="Query")
        self.assertIn("API Quota Exceeded", ans)
        self.assertIn("HTTP 429", status)

    @patch("src.app.answer_query")
    @patch("src.app.load_vector_store")
    def test_handle_query_gemini_unavailable_error(self, mock_load_store, mock_answer_query):
        """Service unavailable error is reported gracefully."""
        mock_store = MagicMock()
        mock_store.count.return_value = 5
        mock_load_store.return_value = mock_store

        mock_answer_query.side_effect = GeminiUnavailableError(
            "Gemini service is temporarily unavailable or timed out."
        )

        ans, cit, status = handle_query(query="Query")
        self.assertIn("Gemini Service Unavailable", ans)
        self.assertIn("temporarily unavailable", status)

    # ------------------------------------------------------------------------
    # 4. App Building & Import Inertness Tests
    # ------------------------------------------------------------------------

    def test_build_app_returns_blocks_instance(self):
        """build_app creates a valid Gradio Blocks instance without launching a server."""
        app = build_app()
        self.assertIsInstance(app, gr.Blocks)

    def test_import_src_app_is_inert(self):
        """Importing src.app does not launch any server or start ingestion."""
        import src.app
        self.assertTrue(hasattr(src.app, "build_app"))
        self.assertTrue(hasattr(src.app, "main"))

    # ------------------------------------------------------------------------
    # 5. Embedding Model Reuse Tests
    #
    # These tests are written for the PLANNED feature (embedder forwarding in
    # handle_query).  Tests 1–3 are expected to FAIL against the current
    # production code and will pass only after src/app.py is updated.
    # Test 4 is expected to PASS today (lazy loading is already implemented).
    # ------------------------------------------------------------------------

    @patch("src.app.load_vector_store")
    def test_handle_query_passes_module_embedder_by_default(self, mock_load_store):
        """handle_query must forward _QUERY_EMBEDDER to load_vector_store by default.

        Expected state before feature implementation: FAIL
        (load_vector_store is currently called without an embedder argument).
        """
        from src.app import _QUERY_EMBEDDER

        # Return a mock store with 0 chunks so handle_query short-circuits
        # before answer_query — no Gemini call, no model download.
        mock_load_store.return_value.count.return_value = 0

        handle_query(query="What are the quarterly results?")

        mock_load_store.assert_called_once()
        call_kwargs = mock_load_store.call_args.kwargs
        self.assertIn(
            "embedder",
            call_kwargs,
            "load_vector_store must receive an 'embedder' keyword argument from handle_query",
        )
        self.assertIs(
            call_kwargs["embedder"],
            _QUERY_EMBEDDER,
            "The embedder passed to load_vector_store must be the module-level "
            "_QUERY_EMBEDDER instance, not a newly created one",
        )

    @patch("src.app.load_vector_store")
    def test_handle_query_reuses_same_embedder_instance_across_two_calls(self, mock_load_store):
        """Two separate handle_query calls must pass the identical embedder object.

        This proves weights are loaded once and the same SentenceTransformer
        instance is shared across requests.

        Expected state before feature implementation: FAIL
        (both calls currently pass no embedder, so None is None — the second
        assertIs against _QUERY_EMBEDDER exposes the gap).
        """
        from src.app import _QUERY_EMBEDDER

        mock_load_store.return_value.count.return_value = 0

        handle_query(query="First question about documents")
        handle_query(query="Second question about documents")

        self.assertEqual(
            mock_load_store.call_count,
            2,
            "load_vector_store must be called exactly once per handle_query invocation",
        )
        embedder_call_1 = mock_load_store.call_args_list[0].kwargs.get("embedder")
        embedder_call_2 = mock_load_store.call_args_list[1].kwargs.get("embedder")
        self.assertIs(
            embedder_call_1,
            embedder_call_2,
            "Both handle_query calls must pass the exact same embedder object (same id())",
        )
        self.assertIs(
            embedder_call_1,
            _QUERY_EMBEDDER,
            "The shared embedder must be the module-level _QUERY_EMBEDDER singleton",
        )

    @patch("src.app.load_vector_store")
    def test_handle_query_accepts_and_forwards_injected_embedder(self, mock_load_store):
        """An explicitly injected embedder must be forwarded unchanged to load_vector_store.

        This enables test isolation (FakeEmbedder injection) and future flexibility.

        Expected state before feature implementation: FAIL
        (TypeError: handle_query() got an unexpected keyword argument 'embedder').
        """
        from tests.test_vector_store import FakeEmbedder

        fake_embedder = FakeEmbedder(dimension=4)
        mock_load_store.return_value.count.return_value = 0

        # This call will raise TypeError until the embedder parameter is added
        # to handle_query in src/app.py.
        handle_query(query="What are the quarterly results?", embedder=fake_embedder)

        mock_load_store.assert_called_once()
        call_kwargs = mock_load_store.call_args.kwargs
        self.assertIs(
            call_kwargs.get("embedder"),
            fake_embedder,
            "The injected FakeEmbedder must be forwarded unchanged to load_vector_store",
        )

    def test_app_level_embedder_is_lazy_and_model_not_loaded(self):
        """_QUERY_EMBEDDER._model must be None — no transformer weights loaded at import.

        Verifies the lazy-loading contract of SentenceTransformerEmbedder holds at
        the application level.  Does not download any model or require network access.

        Expected state: PASS (lazy loading is already implemented).
        """
        from src.app import _QUERY_EMBEDDER

        self.assertIsInstance(
            _QUERY_EMBEDDER,
            SentenceTransformerEmbedder,
            "_QUERY_EMBEDDER must be a SentenceTransformerEmbedder instance",
        )
        self.assertIsNone(
            _QUERY_EMBEDDER._model,
            "_QUERY_EMBEDDER._model must be None before any embed call — "
            "transformer weights must not be loaded at import time",
        )


if __name__ == "__main__":
    unittest.main()
