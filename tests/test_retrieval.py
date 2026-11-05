"""Unit tests for the retrieval module (src/retrieval.py).

Verifies safe vector store loading, semantic retrieval, optional score thresholding,
error handling, and multi-format context/citation formatting. All tests use
deterministic fake embeddings and isolated temporary directories with zero network
calls or model downloads.
"""

import json
import math
import tempfile
import unittest
from pathlib import Path

from src.chunking import DocumentChunk
from src.retrieval import (
    VectorStoreCorruptedError,
    VectorStoreNotFoundError,
    format_citation_label,
    format_citations,
    format_retrieved_context,
    format_source_location,
    load_vector_store,
    retrieve,
)
from src.vector_store import (
    DEFAULT_FAISS_INDEX_NAME,
    DEFAULT_METADATA_NAME,
    FAISSVectorStore,
    SearchResult,
)
from tests.test_vector_store import FakeEmbedder


class TestRetrieval(unittest.TestCase):
    """Test suite covering vector store loading, retrieval, thresholding, and formatting."""

    def setUp(self):
        """Set up isolated temporary directory and sample multi-format chunks."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)

        # Deterministic 4D embedder with controlled vectors for ranking tests
        self.mapping = {
            # Query targets
            "q_finance": [1.0, 0.0, 0.0, 0.0],
            # Document chunks with controlled cosine similarities
            "c_high": [0.95, 0.05, 0.0, 0.0],      # Cosine similarity ~ 0.998
            "c_medium": [0.70, 0.70, 0.0, 0.0],    # Cosine similarity ~ 0.707
            "c_low": [0.10, 0.99, 0.0, 0.0],       # Cosine similarity ~ 0.100
        }
        self.embedder = FakeEmbedder(dimension=4, mapping=self.mapping)

        # Sample multi-format chunks
        self.chunk_pdf = DocumentChunk(
            text="Quarterly fiscal report details revenue of 50 million.",
            source="annual_report.pdf",
            file_type="pdf",
            chunk_index=0,
            total_chunks=1,
            chunk_id="annual_report.pdf:p3:c0",
            page=3,
        )

        self.chunk_xlsx = DocumentChunk(
            text="Dept: Engineering | Allocated: 450000 | Spent: 320000",
            source="department_budget.xlsx",
            file_type="xlsx",
            chunk_index=0,
            total_chunks=1,
            chunk_id="department_budget.xlsx:sAllocations:r12:c0",
            sheet_name="Allocations",
            row_index=12,
        )

        self.chunk_pptx = DocumentChunk(
            text="Global expansion roadmap: Europe, Asia, and North America.",
            source="strategy_deck.pptx",
            file_type="pptx",
            chunk_index=0,
            total_chunks=1,
            chunk_id="strategy_deck.pptx:s5:c0",
            slide_number=5,
        )

        # Populate a working store
        self.store = FAISSVectorStore(embedder=self.embedder)
        self.store.add_chunks([self.chunk_pdf, self.chunk_xlsx, self.chunk_pptx])

    def tearDown(self):
        """Clean up temporary test directory."""
        self.temp_dir.cleanup()

    # ------------------------------------------------------------------------
    # 1. Vector Store Loading Tests
    # ------------------------------------------------------------------------

    def test_load_vector_store_success(self):
        """Loading a valid saved vector store returns a populated FAISSVectorStore."""
        save_path = self.test_dir / "valid_store"
        self.store.save(save_path)

        loaded = load_vector_store(save_path, embedder=self.embedder)
        self.assertIsInstance(loaded, FAISSVectorStore)
        self.assertEqual(loaded.count(), 3)
        self.assertEqual(len(loaded.chunks), 3)

    def test_load_vector_store_missing_directory(self):
        """Loading from a non-existent directory raises VectorStoreNotFoundError."""
        non_existent = self.test_dir / "missing_dir"
        with self.assertRaises(VectorStoreNotFoundError) as ctx:
            load_vector_store(non_existent, embedder=self.embedder)
        self.assertIn("directory not found", str(ctx.exception))

    def test_load_vector_store_empty_directory_missing_files(self):
        """Loading from an empty directory raises VectorStoreNotFoundError."""
        empty_dir = self.test_dir / "empty_dir"
        empty_dir.mkdir()
        with self.assertRaises(VectorStoreNotFoundError) as ctx:
            load_vector_store(empty_dir, embedder=self.embedder)
        self.assertIn("missing in", str(ctx.exception))

    def test_load_vector_store_missing_index_faiss(self):
        """Missing index.faiss raises VectorStoreCorruptedError."""
        partial_dir = self.test_dir / "partial_dir"
        partial_dir.mkdir()
        # Create metadata.json only
        (partial_dir / DEFAULT_METADATA_NAME).write_text("{}", encoding="utf-8")

        with self.assertRaises(VectorStoreCorruptedError) as ctx:
            load_vector_store(partial_dir, embedder=self.embedder)
        self.assertIn("missing FAISS index", str(ctx.exception))

    def test_load_vector_store_missing_metadata_json(self):
        """Missing metadata.json raises VectorStoreCorruptedError."""
        partial_dir = self.test_dir / "partial_dir2"
        partial_dir.mkdir()
        # Create dummy index.faiss only
        (partial_dir / DEFAULT_FAISS_INDEX_NAME).write_bytes(b"dummy")

        with self.assertRaises(VectorStoreCorruptedError) as ctx:
            load_vector_store(partial_dir, embedder=self.embedder)
        self.assertIn("missing metadata file", str(ctx.exception))

    def test_load_vector_store_corrupted_metadata(self):
        """Corrupted metadata.json content raises VectorStoreCorruptedError."""
        corrupt_dir = self.test_dir / "corrupt_dir"
        self.store.save(corrupt_dir)
        # Overwrite metadata with invalid JSON
        (corrupt_dir / DEFAULT_METADATA_NAME).write_text("{invalid json", encoding="utf-8")

        with self.assertRaises(VectorStoreCorruptedError) as ctx:
            load_vector_store(corrupt_dir, embedder=self.embedder)
        self.assertIn("corrupted", str(ctx.exception))

    # ------------------------------------------------------------------------
    # 2. Retrieval Ordering & Top-K Tests
    # ------------------------------------------------------------------------

    def test_retrieve_ordering_and_top_k(self):
        """Retrieve returns results sorted in descending similarity order up to top_k."""
        # Create chunks using the controlled vectors
        c_high = DocumentChunk(
            text="c_high", source="doc1.pdf", file_type="pdf", chunk_index=0,
            total_chunks=1, chunk_id="d1:c0", page=1,
        )
        c_med = DocumentChunk(
            text="c_medium", source="doc2.pdf", file_type="pdf", chunk_index=0,
            total_chunks=1, chunk_id="d2:c0", page=2,
        )
        c_low = DocumentChunk(
            text="c_low", source="doc3.pdf", file_type="pdf", chunk_index=0,
            total_chunks=1, chunk_id="d3:c0", page=3,
        )

        ranked_store = FAISSVectorStore(embedder=self.embedder)
        ranked_store.add_chunks([c_low, c_high, c_med])

        # Query matches q_finance
        results = retrieve("q_finance", ranked_store, top_k=2)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].chunk.text, "c_high")
        self.assertEqual(results[1].chunk.text, "c_medium")
        self.assertGreater(results[0].score, results[1].score)

        # Retrieve with top_k exceeding total chunks returns all chunks without error
        all_results = retrieve("q_finance", ranked_store, top_k=10)
        self.assertEqual(len(all_results), 3)
        self.assertEqual(all_results[2].chunk.text, "c_low")

    def test_retrieve_preserves_chunk_and_search_result_objects(self):
        """Retrieved items are unaltered SearchResult and DocumentChunk instances."""
        single_store = FAISSVectorStore(embedder=self.embedder)
        single_store.add_chunks([self.chunk_pdf])

        results = retrieve("any search query", single_store, top_k=1)
        self.assertEqual(len(results), 1)
        res = results[0]

        self.assertIsInstance(res, SearchResult)
        self.assertIsInstance(res.chunk, DocumentChunk)
        self.assertIsInstance(res.score, float)
        # Check metadata preservation
        self.assertEqual(res.chunk.source, "annual_report.pdf")
        self.assertEqual(res.chunk.page, 3)
        self.assertEqual(res.chunk.chunk_id, "annual_report.pdf:p3:c0")
 
    def test_retrieve_all_case_register_records(self):
        """Comprehensive case queries retrieve every Case Register row only."""
        query = "Summarise all cases in the Case Register."

        case_chunks = [
            DocumentChunk(
                text=f"Suit Reference: LSH/TRN/2026/00{i} | Registry Status: Test status {i}",
                source="registry_tracker.xlsx",
                file_type="xlsx",
                chunk_index=i - 1,
                total_chunks=5,
                chunk_id=f"registry_tracker.xlsx:sCase Register:r{i + 1}:c0",
                sheet_name="Case Register",
                row_index=i + 1,
            )
            for i in range(1, 6)
        ]

        unrelated_sheet_chunks = [
            DocumentChunk(
                text=f"Unrelated administrative information {i}",
                source="registry_tracker.xlsx",
                file_type="xlsx",
                chunk_index=i,
                total_chunks=2,
                chunk_id=f"registry_tracker.xlsx:sOther Sheet:r{i + 1}:c0",
                sheet_name="Other Sheet",
                row_index=i + 1,
            )
            for i in range(2)
        ]

        unrelated_pdf = DocumentChunk(
            text="General training information unrelated to the case register.",
            source="training.pdf",
            file_type="pdf",
            chunk_index=0,
            total_chunks=1,
            chunk_id="training.pdf:p1:c0",
            page=1,
        )

        store = FAISSVectorStore(embedder=self.embedder)
        store.add_chunks(
            case_chunks + unrelated_sheet_chunks + [unrelated_pdf]
        )

        results = retrieve(query, store, top_k=4)

        self.assertEqual(len(results), 5)
        self.assertEqual(
            {result.chunk.chunk_id for result in results},
            {chunk.chunk_id for chunk in case_chunks},
        )
        self.assertTrue(
            all(isinstance(result, SearchResult) for result in results)
        )
        self.assertTrue(
            all(result.chunk.sheet_name == "Case Register" for result in results)
        )
        self.assertTrue(
            all(-1.0 <= result.score <= 1.0 for result in results)
        )

    def test_retrieve_prioritizes_case_register_for_case_specific_questions(self):
        """Case Register records outrank training questions for case-specific queries."""
        query = "Which case is awaiting payment-reference verification?"

        case_text = (
            "Suit Reference: LSH/TRN/2026/003 | "
            "Registry Status: Awaiting payment-reference verification"
        )
        training_text = (
            "Training question: Which case status should a registry officer verify?"
        )

        mapping = {
            query: [1.0, 0.0, 0.0, 0.0],
            case_text: [0.20, 0.98, 0.0, 0.0],
            training_text: [0.90, 0.10, 0.0, 0.0],
        }

        embedder = FakeEmbedder(dimension=4, mapping=mapping)

        case_chunk = DocumentChunk(
            text=case_text,
            source="registry_tracker.xlsx",
            file_type="xlsx",
            chunk_index=0,
            total_chunks=1,
            chunk_id="registry_tracker.xlsx:sCase Register:r4:c0",
            sheet_name="Case Register",
            row_index=4,
        )

        training_chunk = DocumentChunk(
            text=training_text,
            source="registry_tracker.xlsx",
            file_type="xlsx",
            chunk_index=0,
            total_chunks=1,
            chunk_id="registry_tracker.xlsx:sTest Questions:r2:c0",
            sheet_name="Test Questions",
            row_index=2,
        )

        store = FAISSVectorStore(embedder=embedder)
        store.add_chunks([training_chunk, case_chunk])

        results = retrieve(query, store, top_k=1)

        self.assertEqual(len(results), 1)
        self.assertIs(results[0].chunk, case_chunk)

    def test_retrieve_prioritizes_authoritative_sources_for_procedural_questions(self):
        """Registry guidance outranks training questions for procedural queries."""
        query = "What should registry staff do when required information is missing?"

        guide_text = (
            "Registry Guide: When required information is missing, "
            "flag the matter for human review."
        )
        admin_text = (
            "Admin Guidance: Staff should request missing information "
            "through the authorised registry process."
        )
        training_text = (
            "Training question: What happens when a case has missing information?"
        )

        mapping = {
            query: [1.0, 0.0, 0.0, 0.0],
            guide_text: [0.20, 0.98, 0.0, 0.0],
            admin_text: [0.30, 0.95, 0.0, 0.0],
            training_text: [0.90, 0.10, 0.0, 0.0],
        }

        embedder = FakeEmbedder(dimension=4, mapping=mapping)

        guide_chunk = DocumentChunk(
            text=guide_text,
            source="registry_guide.pdf",
            file_type="pdf",
            chunk_index=0,
            total_chunks=1,
            chunk_id="registry_guide.pdf:p1:c0",
            page=1,
        )

        admin_chunk = DocumentChunk(
            text=admin_text,
            source="registry_tracker.xlsx",
            file_type="xlsx",
            chunk_index=0,
            total_chunks=1,
            chunk_id="registry_tracker.xlsx:sAdmin Guidance:r2:c0",
            sheet_name="Admin Guidance",
            row_index=2,
        )

        training_chunk = DocumentChunk(
            text=training_text,
            source="registry_tracker.xlsx",
            file_type="xlsx",
            chunk_index=0,
            total_chunks=1,
            chunk_id="registry_tracker.xlsx:sTest Questions:r2:c0",
            sheet_name="Test Questions",
            row_index=2,
        )

        store = FAISSVectorStore(embedder=embedder)
        store.add_chunks([training_chunk, guide_chunk, admin_chunk])

        results = retrieve(query, store, top_k=1)

        self.assertEqual(len(results), 1)
        self.assertIn(results[0].chunk, [guide_chunk, admin_chunk])
       
    def test_retrieve_keeps_test_questions_available_for_training_queries(self):
        """Test Questions remain retrievable when the query concerns training."""
        query = "What are the sample test questions for registry training?"

        training_text = (
            "Sample test questions: What should a registry officer check "
            "when a case has missing information?"
        )

        guide_text = (
            "Registry Guide: Missing information should be flagged for review."
        )

        mapping = {
            query: [1.0, 0.0, 0.0, 0.0],
            training_text: [0.95, 0.10, 0.0, 0.0],
            guide_text: [0.20, 0.98, 0.0, 0.0],
        }

        embedder = FakeEmbedder(dimension=4, mapping=mapping)

        training_chunk = DocumentChunk(
            text=training_text,
            source="registry_tracker.xlsx",
            file_type="xlsx",
            chunk_index=0,
            total_chunks=1,
            chunk_id="registry_tracker.xlsx:sTest Questions:r2:c0",
            sheet_name="Test Questions",
            row_index=2,
        )

        guide_chunk = DocumentChunk(
            text=guide_text,
            source="registry_guide.pdf",
            file_type="pdf",
            chunk_index=0,
            total_chunks=1,
            chunk_id="registry_guide.pdf:p1:c0",
            page=1,
        )

        store = FAISSVectorStore(embedder=embedder)
        store.add_chunks([guide_chunk, training_chunk])

        results = retrieve(query, store, top_k=1)

        self.assertEqual(len(results), 1)
        self.assertIs(results[0].chunk, training_chunk)

    # ------------------------------------------------------------------------
    # 3. Similarity Threshold Safeguard Tests
    # ------------------------------------------------------------------------
    
    def test_retrieve_prioritizes_exact_case_register_match(self):
        """An exact Case Register match outranks semantically stronger distractors."""
        query = "What is the current status for case LSH/TRN/2026/005?"

        case_text = (
            "Suit Reference: LSH/TRN/2026/005 | "
            "Registry Status: Missing information flagged"
        )
        mention_text = (
            "Training question mentioning LSH/TRN/2026/005"
        )
        distractor_text = "unrelated high similarity"

        mapping = {
            query: [1.0, 0.0, 0.0, 0.0],
            case_text: [0.10, 0.99, 0.0, 0.0],
            mention_text: [0.20, 0.98, 0.0, 0.0],
            distractor_text: [0.99, 0.01, 0.0, 0.0],
        }
        embedder = FakeEmbedder(dimension=4, mapping=mapping)

        case_chunk = DocumentChunk(
            text=case_text,
            source="registry_tracker.xlsx",
            file_type="xlsx",
            chunk_index=0,
            total_chunks=1,
            chunk_id="registry_tracker.xlsx:sCase Register:r6:c0",
            sheet_name="Case Register",
            row_index=6,
        )

        mention_chunk = DocumentChunk(
            text=mention_text,
            source="test_questions.pdf",
            file_type="pdf",
            chunk_index=0,
            total_chunks=1,
            chunk_id="test_questions.pdf:p1:c0",
            page=1,
        )

        distractor_chunk = DocumentChunk(
            text=distractor_text,
            source="unrelated.pdf",
            file_type="pdf",
            chunk_index=0,
            total_chunks=1,
            chunk_id="unrelated.pdf:p1:c0",
            page=1,
        )

        store = FAISSVectorStore(embedder=embedder)
        store.add_chunks([mention_chunk, distractor_chunk, case_chunk])

        results = retrieve(query, store, top_k=1)

        self.assertEqual(len(results), 1)
        self.assertIs(results[0].chunk, case_chunk)
        self.assertIsInstance(results[0], SearchResult)
        self.assertGreaterEqual(results[0].score, -1.0)
        self.assertLessEqual(results[0].score, 1.0)


    def test_retrieve_threshold_filtering(self):
        """Supplying score_threshold filters out chunks below the threshold."""
        c_high = DocumentChunk(
            text="c_high", source="doc1.pdf", file_type="pdf", chunk_index=0,
            total_chunks=1, chunk_id="d1:c0",
        )
        c_med = DocumentChunk(
            text="c_medium", source="doc2.pdf", file_type="pdf", chunk_index=0,
            total_chunks=1, chunk_id="d2:c0",
        )
        c_low = DocumentChunk(
            text="c_low", source="doc3.pdf", file_type="pdf", chunk_index=0,
            total_chunks=1, chunk_id="d3:c0",
        )

        ranked_store = FAISSVectorStore(embedder=self.embedder)
        ranked_store.add_chunks([c_high, c_med, c_low])

        # Scores against q_finance: c_high ~ 0.998, c_medium ~ 0.707, c_low ~ 0.100
        # Threshold 0.8: should retain only c_high
        results_08 = retrieve("q_finance", ranked_store, top_k=4, score_threshold=0.8)
        self.assertEqual(len(results_08), 1)
        self.assertEqual(results_08[0].chunk.text, "c_high")

        # Threshold 0.5: should retain c_high and c_medium
        results_05 = retrieve("q_finance", ranked_store, top_k=4, score_threshold=0.5)
        self.assertEqual(len(results_05), 2)
        self.assertEqual([r.chunk.text for r in results_05], ["c_high", "c_medium"])

    def test_retrieve_threshold_filters_out_all_results(self):
        """When all chunks score below the threshold, retrieve safely returns an empty list."""
        c_low = DocumentChunk(
            text="c_low", source="doc3.pdf", file_type="pdf", chunk_index=0,
            total_chunks=1, chunk_id="d3:c0",
        )
        store = FAISSVectorStore(embedder=self.embedder)
        store.add_chunks([c_low])

        # High threshold eliminates the only chunk
        results = retrieve("q_finance", store, top_k=4, score_threshold=0.95)
        self.assertEqual(results, [])

    def test_retrieve_without_threshold_preserves_standard_top_k(self):
        """When score_threshold is None, standard top-k behavior is strictly preserved."""
        c_low = DocumentChunk(
            text="c_low", source="doc3.pdf", file_type="pdf", chunk_index=0,
            total_chunks=1, chunk_id="d3:c0",
        )
        store = FAISSVectorStore(embedder=self.embedder)
        store.add_chunks([c_low])

        results = retrieve("q_finance", store, top_k=4, score_threshold=None)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk.text, "c_low")

    # ------------------------------------------------------------------------
    # 4. Input Validation & Edge Case Tests
    # ------------------------------------------------------------------------

    def test_retrieve_blank_or_whitespace_query(self):
        """Blank or whitespace queries return empty list safely."""
        self.assertEqual(retrieve("", self.store), [])
        self.assertEqual(retrieve("   \t\n  ", self.store), [])

    def test_retrieve_empty_vector_store(self):
        """Querying an empty vector store returns empty list safely."""
        empty_store = FAISSVectorStore(embedder=self.embedder)
        self.assertEqual(retrieve("any query", empty_store), [])

    def test_retrieve_invalid_query_type(self):
        """Non-string query raises TypeError."""
        with self.assertRaises(TypeError):
            retrieve(123, self.store)
        with self.assertRaises(TypeError):
            retrieve(None, self.store)

    def test_retrieve_invalid_vector_store_type(self):
        """Non-FAISSVectorStore raises TypeError."""
        with self.assertRaises(TypeError):
            retrieve("query", "not a store")

    def test_retrieve_invalid_top_k(self):
        """Non-integer, boolean, or <= 0 top_k raises TypeError or ValueError."""
        with self.assertRaises(TypeError):
            retrieve("query", self.store, top_k="4")
        with self.assertRaises(TypeError):
            retrieve("query", self.store, top_k=True)  # bool is subclass of int
        with self.assertRaises(ValueError):
            retrieve("query", self.store, top_k=0)
        with self.assertRaises(ValueError):
            retrieve("query", self.store, top_k=-1)

    def test_retrieve_invalid_score_threshold(self):
        """Invalid threshold types or out-of-range values raise TypeError or ValueError."""
        # Non-numeric or boolean
        with self.assertRaises(TypeError):
            retrieve("query", self.store, score_threshold="0.5")
        with self.assertRaises(TypeError):
            retrieve("query", self.store, score_threshold=True)

        # Non-finite values
        with self.assertRaises(ValueError):
            retrieve("query", self.store, score_threshold=float("nan"))
        with self.assertRaises(ValueError):
            retrieve("query", self.store, score_threshold=float("inf"))

        # Out-of-range cosine similarity values
        with self.assertRaises(ValueError):
            retrieve("query", self.store, score_threshold=-1.5)
        with self.assertRaises(ValueError):
            retrieve("query", self.store, score_threshold=1.5)

    # ------------------------------------------------------------------------
    # 5. Multi-Format Source Formatting & Location Tests
    # ------------------------------------------------------------------------

    def test_format_source_location_pdf(self):
        """PDF chunk formats page location correctly."""
        loc = format_source_location(self.chunk_pdf)
        self.assertEqual(loc, "Page 3")

    def test_format_source_location_excel(self):
        """Excel chunk formats sheet and row correctly."""
        loc = format_source_location(self.chunk_xlsx)
        self.assertEqual(loc, "Sheet 'Allocations', Row 12")

    def test_format_source_location_pptx(self):
        """PowerPoint chunk formats slide number correctly."""
        loc = format_source_location(self.chunk_pptx)
        self.assertEqual(loc, "Slide 5")

    def test_format_source_location_missing_optional_fields(self):
        """Chunks with missing optional coordinates return empty string without fabricating."""
        # PDF without page
        bare_pdf = DocumentChunk(
            text="text", source="doc.pdf", file_type="pdf",
            chunk_index=0, total_chunks=1, chunk_id="d:c0",
        )
        self.assertEqual(format_source_location(bare_pdf), "")

        # Excel with only sheet name
        sheet_only = DocumentChunk(
            text="text", source="doc.xlsx", file_type="xlsx",
            chunk_index=0, total_chunks=1, chunk_id="d:c0",
            sheet_name="Summary",
        )
        self.assertEqual(format_source_location(sheet_only), "Sheet 'Summary'")

        # Excel with only row index
        row_only = DocumentChunk(
            text="text", source="doc.xlsx", file_type="xlsx",
            chunk_index=0, total_chunks=1, chunk_id="d:c0",
            row_index=5,
        )
        self.assertEqual(format_source_location(row_only), "Row 5")

    def test_format_citation_label(self):
        """Combined citation labels format accurately with and without location."""
        self.assertEqual(format_citation_label(self.chunk_pdf), "annual_report.pdf (Page 3)")
        self.assertEqual(
            format_citation_label(self.chunk_xlsx),
            "department_budget.xlsx (Sheet 'Allocations', Row 12)",
        )
        self.assertEqual(format_citation_label(self.chunk_pptx), "strategy_deck.pptx (Slide 5)")

        # Bare chunk without location
        bare_chunk = DocumentChunk(
            text="text", source="plain.pdf", file_type="pdf",
            chunk_index=0, total_chunks=1, chunk_id="p:c0",
        )
        self.assertEqual(format_citation_label(bare_chunk), "plain.pdf")

    # ------------------------------------------------------------------------
    # 6. Context Formatter for Prompts Tests
    # ------------------------------------------------------------------------

    def test_format_retrieved_context_multi_format(self):
        """Context formatter produces separated blocks with accurate source headers."""
        results = [
            SearchResult(chunk=self.chunk_pdf, score=0.95),
            SearchResult(chunk=self.chunk_xlsx, score=0.85),
        ]
        context = format_retrieved_context(results)

        self.assertIn("[Source: annual_report.pdf | Page 3]", context)
        self.assertIn("Quarterly fiscal report details revenue", context)
        self.assertIn("\n\n---\n\n", context)
        self.assertIn("[Source: department_budget.xlsx | Sheet 'Allocations', Row 12]", context)
        self.assertIn("Dept: Engineering", context)

    def test_format_retrieved_context_missing_location(self):
        """Context formatter omits location pipe if no coordinates exist."""
        bare_chunk = DocumentChunk(
            text="Bare content without page", source="notes.pdf", file_type="pdf",
            chunk_index=0, total_chunks=1, chunk_id="n:c0",
        )
        results = [SearchResult(chunk=bare_chunk, score=0.8)]
        context = format_retrieved_context(results)

        self.assertIn("[Source: notes.pdf]", context)
        self.assertNotIn("|", context)
        self.assertIn("Bare content without page", context)

    def test_format_retrieved_context_empty(self):
        """Empty results return specified fallback message."""
        self.assertEqual(
            format_retrieved_context([]),
            "No relevant document context found.",
        )
        custom_fallback = "No context available."
        self.assertEqual(
            format_retrieved_context([], empty_fallback=custom_fallback),
            custom_fallback,
        )

    # ------------------------------------------------------------------------
    # 7. Structured Citation Formatter for UI Tests
    # ------------------------------------------------------------------------

    def test_format_citations_structure_and_values(self):
        """format_citations returns structured dicts with exact metadata and rounded scores."""
        results = [
            SearchResult(chunk=self.chunk_pdf, score=0.912345),
            SearchResult(chunk=self.chunk_pptx, score=0.789123),
        ]
        citations = format_citations(results)

        self.assertEqual(len(citations), 2)

        # PDF citation check
        c_pdf = citations[0]
        self.assertEqual(c_pdf["source"], "annual_report.pdf")
        self.assertEqual(c_pdf["file_type"], "pdf")
        self.assertEqual(c_pdf["location"], "Page 3")
        self.assertEqual(c_pdf["label"], "annual_report.pdf (Page 3)")
        self.assertEqual(c_pdf["score"], 0.9123)
        self.assertEqual(c_pdf["page"], 3)
        self.assertIsNone(c_pdf["sheet_name"])
        self.assertIsNone(c_pdf["slide_number"])
        self.assertIn("Quarterly fiscal report", c_pdf["snippet"])

        # PPTX citation check
        c_pptx = citations[1]
        self.assertEqual(c_pptx["source"], "strategy_deck.pptx")
        self.assertEqual(c_pptx["file_type"], "pptx")
        self.assertEqual(c_pptx["location"], "Slide 5")
        self.assertEqual(c_pptx["label"], "strategy_deck.pptx (Slide 5)")
        self.assertEqual(c_pptx["score"], 0.7891)
        self.assertEqual(c_pptx["slide_number"], 5)
        self.assertIsNone(c_pptx["page"])

    def test_format_citations_snippet_truncation(self):
        """Long chunk text is truncated cleanly at snippet_length with ellipsis."""
        long_text = "Word " * 100  # 500 characters
        long_chunk = DocumentChunk(
            text=long_text, source="long.pdf", file_type="pdf",
            chunk_index=0, total_chunks=1, chunk_id="l:c0", page=1,
        )
        results = [SearchResult(chunk=long_chunk, score=0.9)]
        citations = format_citations(results, snippet_length=50)

        self.assertTrue(citations[0]["snippet"].endswith("..."))
        self.assertLessEqual(len(citations[0]["snippet"]), 53)  # 50 chars + "..."

    def test_format_citations_empty_results(self):
        """Empty results return an empty citations list."""
        self.assertEqual(format_citations([]), [])


if __name__ == "__main__":
    unittest.main()
