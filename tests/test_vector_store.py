"""Unit tests for the vector store module (src/vector_store.py).

Verifies native FAISS indexing, deterministic similarity search, cosine ranking,
metadata preservation, safe JSON persistence without pickle, and parameter validation.
All tests use deterministic fake embeddings with zero network calls or model downloads.
"""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import patch

import numpy as np

from src.chunking import DocumentChunk
from src.vector_store import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_FAISS_INDEX_NAME,
    DEFAULT_METADATA_NAME,
    FAISSVectorStore,
    SearchResult,
    SentenceTransformerEmbedder,
    chunk_from_dict,
    chunk_to_dict,
)


# ============================================================================
# Deterministic Fake Embedder for Tests (No network, zero downloads)
# ============================================================================

class FakeEmbedder:
    """Deterministic in-memory embedder for fast unit tests without network or GPU."""

    def __init__(
        self,
        dimension: int = 4,
        mapping: Optional[Dict[str, List[float]]] = None,
    ):
        self.dimension = dimension
        self.mapping = mapping or {}

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        vecs = []
        for t in texts:
            if t in self.mapping:
                raw = np.array(self.mapping[t], dtype=np.float32)
            else:
                # Deterministic pseudo-vector derived from string character values
                seed = sum(ord(c) * (i + 1) for i, c in enumerate(t)) % 10000
                rng = np.random.RandomState(seed)
                raw = rng.randn(self.dimension).astype(np.float32)

            norm = np.linalg.norm(raw)
            if norm > 0:
                raw = raw / norm
            vecs.append(raw)

        return np.array(vecs, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])


# ============================================================================
# Test Fixtures & Test Cases
# ============================================================================

class TestVectorStore(unittest.TestCase):
    """Complete test suite for FAISSVectorStore and related components."""

    def setUp(self):
        """Create sample document chunks representing multi-format inputs."""
        self.chunk_pdf = DocumentChunk(
            text="Annual revenue increased by 25% year over year.",
            source="annual_report.pdf",
            file_type="pdf",
            chunk_index=0,
            total_chunks=2,
            chunk_id="annual_report.pdf:p1:c0",
            page=1,
            metadata={"department": "Finance", "quarter": "Q4"},
        )
        self.chunk_excel = DocumentChunk(
            text="Sheet: Payroll | Row 5 | Employee: Alice | Salary: 95000",
            source="payroll.xlsx",
            file_type="xlsx",
            chunk_index=0,
            total_chunks=1,
            chunk_id="payroll.xlsx:sheet_Payroll:r5:c0",
            sheet_name="Payroll",
            row_index=5,
            metadata={"confidential": True},
        )
        self.chunk_pptx = DocumentChunk(
            text="Slide 3: Roadmap for Cloud Migration in Q2 2026.",
            source="strategy.pptx",
            file_type="pptx",
            chunk_index=0,
            total_chunks=1,
            chunk_id="strategy.pptx:s3:c0",
            slide_number=3,
            metadata={"presenter": "CTO"},
        )

    # ------------------------------------------------------------------------
    # 1. Lazy Loading & Architecture Constraints
    # ------------------------------------------------------------------------

    def test_lazy_embedder_does_not_download_or_instantiate_model(self):
        """SentenceTransformerEmbedder must initialize without loading model weights."""
        embedder = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")
        self.assertIsNone(embedder._model, "Model should be None upon initialization")
        self.assertEqual(embedder.model_name, "all-MiniLM-L6-v2")

    def test_default_vector_store_initialization_is_lazy(self):
        """FAISSVectorStore() with default parameters must not load SentenceTransformer."""
        store = FAISSVectorStore()
        self.assertEqual(store.model_name, DEFAULT_EMBEDDING_MODEL)
        self.assertIsInstance(store.embedder, SentenceTransformerEmbedder)
        self.assertIsNone(store.embedder._model, "SentenceTransformer must not be instantiated on store init")

    # ------------------------------------------------------------------------
    # 2. Document Chunk Indexing & Size Tracking
    # ------------------------------------------------------------------------

    def test_add_chunks_success_and_count(self):
        """Adding chunks should index vectors and update count accurately."""
        embedder = FakeEmbedder(dimension=4)
        store = FAISSVectorStore(embedder=embedder)

        added = store.add_chunks([self.chunk_pdf, self.chunk_excel, self.chunk_pptx])
        self.assertEqual(added, 3)
        self.assertEqual(store.count(), 3)
        self.assertEqual(len(store), 3)
        self.assertEqual(store.dimension, 4)

    def test_add_empty_chunks_list_is_safe(self):
        """Passing an empty list of chunks should safely return 0 without error."""
        embedder = FakeEmbedder(dimension=4)
        store = FAISSVectorStore(embedder=embedder)

        added = store.add_chunks([])
        self.assertEqual(added, 0)
        self.assertEqual(store.count(), 0)

    def test_clear_resets_store(self):
        """clear() should reset index and chunks list."""
        embedder = FakeEmbedder(dimension=4)
        store = FAISSVectorStore(embedder=embedder)
        store.add_chunks([self.chunk_pdf])
        self.assertEqual(store.count(), 1)

        store.clear()
        self.assertEqual(store.count(), 0)
        self.assertEqual(len(store), 0)

    # ------------------------------------------------------------------------
    # 3. Deterministic Cosine Similarity Search & Ranking
    # ------------------------------------------------------------------------

    def test_cosine_similarity_ranking_order(self):
        """Exact vector control: chunks with higher cosine similarity must rank first."""
        # 4-dimensional unit vectors
        mapping = {
            "query_target": [1.0, 0.0, 0.0, 0.0],
            # chunk A: highly aligned with query (cosine sim = 1.0)
            self.chunk_pdf.text: [1.0, 0.0, 0.0, 0.0],
            # chunk B: partially aligned with query (cosine sim = 0.5)
            self.chunk_excel.text: [0.5, 0.866025, 0.0, 0.0],
            # chunk C: orthogonal to query (cosine sim = 0.0)
            self.chunk_pptx.text: [0.0, 1.0, 0.0, 0.0],
        }

        embedder = FakeEmbedder(dimension=4, mapping=mapping)
        store = FAISSVectorStore(embedder=embedder)
        store.add_chunks([self.chunk_pptx, self.chunk_excel, self.chunk_pdf])

        results = store.search("query_target", top_k=3)
        self.assertEqual(len(results), 3)

        # 1st rank: PDF (score ~1.0)
        self.assertEqual(results[0].chunk.chunk_id, self.chunk_pdf.chunk_id)
        self.assertAlmostEqual(results[0].score, 1.0, places=4)

        # 2nd rank: Excel (score ~0.5)
        self.assertEqual(results[1].chunk.chunk_id, self.chunk_excel.chunk_id)
        self.assertAlmostEqual(results[1].score, 0.5, places=4)

        # 3rd rank: PowerPoint (score ~0.0)
        self.assertEqual(results[2].chunk.chunk_id, self.chunk_pptx.chunk_id)
        self.assertAlmostEqual(results[2].score, 0.0, places=4)

    def test_search_respects_top_k(self):
        """Search should return at most top_k results."""
        embedder = FakeEmbedder(dimension=4)
        store = FAISSVectorStore(embedder=embedder)
        store.add_chunks([self.chunk_pdf, self.chunk_excel, self.chunk_pptx])

        res_1 = store.search("revenue query", top_k=1)
        self.assertEqual(len(res_1), 1)

        res_2 = store.search("revenue query", top_k=2)
        self.assertEqual(len(res_2), 2)

        # Requesting more than total count should return all available without crashing
        res_all = store.search("revenue query", top_k=10)
        self.assertEqual(len(res_all), 3)

    def test_search_result_structure_and_unpacking(self):
        """SearchResult supports .chunk, .score, tuple unpacking, and .to_dict()."""
        embedder = FakeEmbedder(dimension=4)
        store = FAISSVectorStore(embedder=embedder)
        store.add_chunks([self.chunk_pdf])

        results = store.search("revenue", top_k=1)
        res = results[0]

        # Named attribute access
        self.assertIsInstance(res, SearchResult)
        self.assertEqual(res.chunk.source, "annual_report.pdf")
        self.assertIsInstance(res.score, float)

        # Tuple unpacking
        chunk, score = res
        self.assertEqual(chunk.chunk_id, self.chunk_pdf.chunk_id)
        self.assertIsInstance(score, float)

        # Dictionary serialization
        d = res.to_dict()
        self.assertIn("chunk", d)
        self.assertIn("score", d)
        self.assertEqual(d["chunk"]["page"], 1)

    # ------------------------------------------------------------------------
    # 4. Metadata Preservation Across All Document Types
    # ------------------------------------------------------------------------

    def test_metadata_preservation_pdf_excel_pptx(self):
        """All source metadata fields must be fully retained in search results."""
        embedder = FakeEmbedder(dimension=4)
        store = FAISSVectorStore(embedder=embedder)
        store.add_chunks([self.chunk_pdf, self.chunk_excel, self.chunk_pptx])

        # PDF chunk verification
        pdf_res = store.search(self.chunk_pdf.text, top_k=3)[0].chunk
        self.assertEqual(pdf_res.source, "annual_report.pdf")
        self.assertEqual(pdf_res.file_type, "pdf")
        self.assertEqual(pdf_res.page, 1)
        self.assertEqual(pdf_res.metadata["department"], "Finance")

        # Excel chunk verification
        excel_res = [r.chunk for r in store.search("Payroll", top_k=3) if r.chunk.file_type == "xlsx"][0]
        self.assertEqual(excel_res.source, "payroll.xlsx")
        self.assertEqual(excel_res.sheet_name, "Payroll")
        self.assertEqual(excel_res.row_index, 5)
        self.assertTrue(excel_res.metadata["confidential"])

        # PowerPoint chunk verification
        pptx_res = [r.chunk for r in store.search("Roadmap", top_k=3) if r.chunk.file_type == "pptx"][0]
        self.assertEqual(pptx_res.source, "strategy.pptx")
        self.assertEqual(pptx_res.slide_number, 3)
        self.assertEqual(pptx_res.metadata["presenter"], "CTO")

    # ------------------------------------------------------------------------
    # 5. Input Validation & Parameter Boundaries
    # ------------------------------------------------------------------------

    def test_invalid_chunks_argument_types(self):
        """Non-list chunks or lists with invalid items must raise TypeError."""
        store = FAISSVectorStore(embedder=FakeEmbedder(dimension=4))

        with self.assertRaises(TypeError):
            store.add_chunks("not a list")  # type: ignore

        with self.assertRaises(TypeError):
            store.add_chunks([self.chunk_pdf, "invalid chunk string"])  # type: ignore

    def test_invalid_query_and_top_k(self):
        """Invalid query type, non-int top_k, or top_k <= 0 must raise appropriate errors."""
        store = FAISSVectorStore(embedder=FakeEmbedder(dimension=4))
        store.add_chunks([self.chunk_pdf])

        with self.assertRaises(TypeError):
            store.search(12345)  # type: ignore

        with self.assertRaises(TypeError):
            store.search("query", top_k="2")  # type: ignore

        with self.assertRaises(ValueError):
            store.search("query", top_k=0)

        with self.assertRaises(ValueError):
            store.search("query", top_k=-5)

    def test_empty_or_whitespace_query_returns_safe_empty_list(self):
        """Empty or whitespace queries must safely return an empty result list."""
        store = FAISSVectorStore(embedder=FakeEmbedder(dimension=4))
        store.add_chunks([self.chunk_pdf])

        self.assertEqual(store.search(""), [])
        self.assertEqual(store.search("   \t\n  "), [])

    def test_search_empty_store_returns_safe_empty_list(self):
        """Searching an unpopulated vector store returns an empty result list."""
        store = FAISSVectorStore(embedder=FakeEmbedder(dimension=4))
        self.assertEqual(store.search("test query"), [])

    def test_dimension_mismatch_on_add_chunks(self):
        """Adding chunks with an incompatible embedding dimension must raise ValueError."""
        embedder_4d = FakeEmbedder(dimension=4)
        store = FAISSVectorStore(embedder=embedder_4d)
        store.add_chunks([self.chunk_pdf])

        # Attempt to add with 8D embedder
        store.embedder = FakeEmbedder(dimension=8)
        with self.assertRaises(ValueError) as ctx:
            store.add_chunks([self.chunk_excel])
        self.assertIn("Embedding dimension mismatch", str(ctx.exception))

    def test_dimension_mismatch_on_search(self):
        """Query embedding dimension mismatch against index must raise ValueError."""
        embedder_4d = FakeEmbedder(dimension=4)
        store = FAISSVectorStore(embedder=embedder_4d)
        store.add_chunks([self.chunk_pdf])

        store.embedder = FakeEmbedder(dimension=8)
        with self.assertRaises(ValueError) as ctx:
            store.search("test query")
        self.assertIn("Query embedding dimension mismatch", str(ctx.exception))

    # ------------------------------------------------------------------------
    # 6. Persistence: JSON Metadata & Native FAISS Index Round Trip
    # ------------------------------------------------------------------------

    def test_save_and_load_round_trip(self):
        """Vector store saves to disk as index.faiss + metadata.json and reloads cleanly."""
        embedder = FakeEmbedder(dimension=4)
        store = FAISSVectorStore(embedder=embedder)
        store.add_chunks([self.chunk_pdf, self.chunk_excel, self.chunk_pptx])

        with tempfile.TemporaryDirectory() as tmp_dir:
            store.save(tmp_dir)

            index_path = Path(tmp_dir) / DEFAULT_FAISS_INDEX_NAME
            metadata_path = Path(tmp_dir) / DEFAULT_METADATA_NAME

            # Verify files exist
            self.assertTrue(index_path.is_file(), "index.faiss must exist")
            self.assertTrue(metadata_path.is_file(), "metadata.json must exist")

            # Verify metadata.json is valid pure JSON (NO pickle)
            with open(metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.assertEqual(meta["dimension"], 4)
            self.assertEqual(meta["total_chunks"], 3)
            self.assertEqual(len(meta["chunks"]), 3)

            # Reload into a fresh instance using in-place load()
            reloaded_store = FAISSVectorStore(embedder=embedder)
            reloaded_store.load(tmp_dir)

            self.assertEqual(reloaded_store.count(), 3)
            self.assertEqual(reloaded_store.dimension, 4)
            self.assertEqual(reloaded_store.chunks[0].chunk_id, self.chunk_pdf.chunk_id)
            self.assertEqual(reloaded_store.chunks[1].chunk_id, self.chunk_excel.chunk_id)
            self.assertEqual(reloaded_store.chunks[2].chunk_id, self.chunk_pptx.chunk_id)

            # Also test classmethod load_local()
            factory_store = FAISSVectorStore.load_local(tmp_dir, embedder=embedder)
            self.assertEqual(factory_store.count(), 3)
            self.assertEqual(factory_store.dimension, 4)

            # Verify search results are identical between original and loaded store
            orig_res = store.search("revenue report", top_k=3)
            loaded_res = factory_store.search("revenue report", top_k=3)

            self.assertEqual(len(orig_res), len(loaded_res))
            for r_orig, r_load in zip(orig_res, loaded_res):
                self.assertEqual(r_orig.chunk.chunk_id, r_load.chunk.chunk_id)
                self.assertAlmostEqual(r_orig.score, r_load.score, places=5)

    def test_save_empty_directory_path_raises_error(self):
        """Saving with an empty directory path must raise ValueError."""
        store = FAISSVectorStore(embedder=FakeEmbedder(dimension=4))
        with self.assertRaises(ValueError):
            store.save("")

    # ------------------------------------------------------------------------
    # 7. Persistence Integrity & Missing/Corrupt File Handling
    # ------------------------------------------------------------------------

    def test_load_nonexistent_directory_raises_file_not_found(self):
        """Loading from a non-existent directory must raise FileNotFoundError."""
        store = FAISSVectorStore(embedder=FakeEmbedder(dimension=4))
        with self.assertRaises(FileNotFoundError):
            store.load("C:/path/does/not/exist/anywhere")

    def test_load_missing_index_file_raises_file_not_found(self):
        """Missing index.faiss in directory must raise FileNotFoundError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            metadata_file = Path(tmp_dir) / DEFAULT_METADATA_NAME
            metadata_file.write_text(json.dumps({"dimension": 4, "total_chunks": 0, "chunks": []}), encoding="utf-8")

            store = FAISSVectorStore(embedder=FakeEmbedder(dimension=4))
            with self.assertRaises(FileNotFoundError) as ctx:
                store.load(tmp_dir)
            self.assertIn("Missing FAISS index file", str(ctx.exception))

    def test_load_missing_metadata_file_raises_file_not_found(self):
        """Missing metadata.json in directory must raise FileNotFoundError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = FAISSVectorStore(embedder=FakeEmbedder(dimension=4))
            store.add_chunks([self.chunk_pdf])
            store.save(tmp_dir)

            # Remove metadata.json
            os.remove(Path(tmp_dir) / DEFAULT_METADATA_NAME)

            reloaded = FAISSVectorStore(embedder=FakeEmbedder(dimension=4))
            with self.assertRaises(FileNotFoundError) as ctx:
                reloaded.load(tmp_dir)
            self.assertIn("Missing metadata JSON file", str(ctx.exception))

    def test_load_corrupt_metadata_json_raises_value_error(self):
        """Corrupt JSON in metadata file must raise ValueError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = FAISSVectorStore(embedder=FakeEmbedder(dimension=4))
            store.add_chunks([self.chunk_pdf])
            store.save(tmp_dir)

            # Overwrite metadata with invalid JSON
            metadata_file = Path(tmp_dir) / DEFAULT_METADATA_NAME
            metadata_file.write_text("{corrupt json payload...", encoding="utf-8")

            reloaded = FAISSVectorStore(embedder=FakeEmbedder(dimension=4))
            with self.assertRaises(ValueError) as ctx:
                reloaded.load(tmp_dir)
            self.assertIn("Corrupt or unreadable metadata JSON", str(ctx.exception))

    def test_load_inconsistent_chunks_count_raises_value_error(self):
        """Mismatch between vector count in FAISS index and metadata chunks must raise ValueError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = FAISSVectorStore(embedder=FakeEmbedder(dimension=4))
            store.add_chunks([self.chunk_pdf, self.chunk_excel])
            store.save(tmp_dir)

            # Manually corrupt metadata to have only 1 chunk instead of 2
            metadata_file = Path(tmp_dir) / DEFAULT_METADATA_NAME
            with open(metadata_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
            meta["chunks"] = meta["chunks"][:1]  # 1 chunk, while index has 2 vectors
            meta["total_chunks"] = 1
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(meta, f)

            reloaded = FAISSVectorStore(embedder=FakeEmbedder(dimension=4))
            with self.assertRaises(ValueError) as ctx:
                reloaded.load(tmp_dir)
            self.assertIn("Inconsistent vector store", str(ctx.exception))
            self.assertIn("FAISS index has 2 vectors, but metadata JSON contains 1 chunks", str(ctx.exception))

    def test_load_inconsistent_dimension_raises_value_error(self):
        """Mismatch between FAISS index dimension and metadata dimension must raise ValueError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = FAISSVectorStore(embedder=FakeEmbedder(dimension=4))
            store.add_chunks([self.chunk_pdf])
            store.save(tmp_dir)

            # Alter dimension in metadata
            metadata_file = Path(tmp_dir) / DEFAULT_METADATA_NAME
            with open(metadata_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
            meta["dimension"] = 128  # Index is 4
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(meta, f)

            reloaded = FAISSVectorStore(embedder=FakeEmbedder(dimension=4))
            with self.assertRaises(ValueError) as ctx:
                reloaded.load(tmp_dir)
    def test_repeated_add_chunks_maintains_alignment(self):
        """Repeated calls to add_chunks() must maintain 1:1 vector-to-chunk alignment."""
        chunk_batch_1_a = DocumentChunk(
            text="Alpha document context",
            source="alpha.pdf",
            file_type="pdf",
            chunk_index=0,
            total_chunks=1,
            chunk_id="alpha.pdf:p1:c0",
        )
        chunk_batch_1_b = DocumentChunk(
            text="Beta document context",
            source="beta.pdf",
            file_type="pdf",
            chunk_index=0,
            total_chunks=1,
            chunk_id="beta.pdf:p1:c0",
        )
        chunk_batch_2_c = DocumentChunk(
            text="Gamma document context",
            source="gamma.xlsx",
            file_type="xlsx",
            chunk_index=0,
            total_chunks=1,
            chunk_id="gamma.xlsx:s1:c0",
        )
        chunk_batch_2_d = DocumentChunk(
            text="Delta document context",
            source="delta.pptx",
            file_type="pptx",
            chunk_index=0,
            total_chunks=1,
            chunk_id="delta.pptx:s1:c0",
        )

        mapping = {
            "query_alpha": [1.0, 0.0, 0.0, 0.0],
            "query_delta": [0.0, 0.0, 0.0, 1.0],
            chunk_batch_1_a.text: [1.0, 0.0, 0.0, 0.0],
            chunk_batch_1_b.text: [0.0, 1.0, 0.0, 0.0],
            chunk_batch_2_c.text: [0.0, 0.0, 1.0, 0.0],
            chunk_batch_2_d.text: [0.0, 0.0, 0.0, 1.0],
        }

        embedder = FakeEmbedder(dimension=4, mapping=mapping)
        store = FAISSVectorStore(embedder=embedder)

        # Batch 1 addition
        added_1 = store.add_chunks([chunk_batch_1_a, chunk_batch_1_b])
        self.assertEqual(added_1, 2)
        self.assertEqual(store.count(), 2)

        # Batch 2 addition
        added_2 = store.add_chunks([chunk_batch_2_c, chunk_batch_2_d])
        self.assertEqual(added_2, 2)
        self.assertEqual(store.count(), 4)

        # Verify retrieval from batch 1
        res_alpha = store.search("query_alpha", top_k=1)
        self.assertEqual(len(res_alpha), 1)
        self.assertEqual(res_alpha[0].chunk.chunk_id, chunk_batch_1_a.chunk_id)
        self.assertEqual(res_alpha[0].chunk.source, "alpha.pdf")
        self.assertAlmostEqual(res_alpha[0].score, 1.0, places=4)

        # Verify retrieval from batch 2
        res_delta = store.search("query_delta", top_k=1)
        self.assertEqual(len(res_delta), 1)
        self.assertEqual(res_delta[0].chunk.chunk_id, chunk_batch_2_d.chunk_id)
        self.assertEqual(res_delta[0].chunk.source, "delta.pptx")
        self.assertAlmostEqual(res_delta[0].score, 1.0, places=4)

        # Verify full retrieval across both batches
        all_results = store.search("query_alpha", top_k=4)
        self.assertEqual(len(all_results), 4)
        ids = [r.chunk.chunk_id for r in all_results]
        self.assertEqual(ids[0], chunk_batch_1_a.chunk_id)
        self.assertIn(chunk_batch_1_b.chunk_id, ids)
        self.assertIn(chunk_batch_2_c.chunk_id, ids)
        self.assertIn(chunk_batch_2_d.chunk_id, ids)

    def test_save_empty_store_preserves_custom_embedder_dimension(self):
        """Saving an empty store must infer dimension from the embedder and remain consistent."""
        embedder_8d = FakeEmbedder(dimension=8)
        store = FAISSVectorStore(embedder=embedder_8d)
        self.assertIsNone(store.dimension)

        with tempfile.TemporaryDirectory() as tmp_dir:
            store.save(tmp_dir)

            metadata_path = Path(tmp_dir) / DEFAULT_METADATA_NAME
            with open(metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            self.assertEqual(meta["dimension"], 8)
            self.assertEqual(meta["total_chunks"], 0)
            self.assertEqual(meta["chunks"], [])

            # Load into fresh store and verify adding 8D chunk succeeds
            reloaded = FAISSVectorStore.load_local(tmp_dir, embedder=embedder_8d)
            self.assertEqual(reloaded.dimension, 8)
            self.assertEqual(reloaded.count(), 0)

            chunk = DocumentChunk(
                text="Test 8D document",
                source="test.pdf",
                file_type="pdf",
                chunk_index=0,
                total_chunks=1,
                chunk_id="test.pdf:p1:c0",
            )
            added = reloaded.add_chunks([chunk])
            self.assertEqual(added, 1)
            self.assertEqual(reloaded.count(), 1)

    def test_add_chunks_with_nan_embedding_raises_value_error(self):
        """Embedding array containing NaN values must raise ValueError and not be indexed."""
        class NanEmbedder(FakeEmbedder):
            def embed_texts(self, texts):
                arr = super().embed_texts(texts)
                arr[0, 0] = np.nan
                return arr

        embedder = NanEmbedder(dimension=4)
        store = FAISSVectorStore(embedder=embedder)

        with self.assertRaises(ValueError) as ctx:
            store.add_chunks([self.chunk_pdf])
        self.assertIn("non-finite values (NaN or Inf)", str(ctx.exception))
        self.assertEqual(store.count(), 0)

    def test_add_chunks_with_inf_embedding_raises_value_error(self):
        """Embedding array containing Inf values must raise ValueError and not be indexed."""
        class InfEmbedder(FakeEmbedder):
            def embed_texts(self, texts):
                arr = super().embed_texts(texts)
                arr[0, 1] = np.inf
                return arr

        embedder = InfEmbedder(dimension=4)
        store = FAISSVectorStore(embedder=embedder)

        with self.assertRaises(ValueError) as ctx:
            store.add_chunks([self.chunk_pdf])
        self.assertIn("non-finite values (NaN or Inf)", str(ctx.exception))
        self.assertEqual(store.count(), 0)

    def test_search_with_nan_query_embedding_raises_value_error(self):
        """Query embedding containing NaN values must raise ValueError."""
        class NanQueryEmbedder(FakeEmbedder):
            def embed_query(self, query):
                arr = super().embed_query(query)
                arr[0, 0] = np.nan
                return arr

        embedder = NanQueryEmbedder(dimension=4)
        store = FAISSVectorStore(embedder=embedder)
        store.add_chunks([self.chunk_pdf])

        with self.assertRaises(ValueError) as ctx:
            store.search("query")
        self.assertIn("Query embedding contains non-finite values (NaN or Inf)", str(ctx.exception))

    def test_search_with_inf_query_embedding_raises_value_error(self):
        """Query embedding containing Inf values must raise ValueError."""
        class InfQueryEmbedder(FakeEmbedder):
            def embed_query(self, query):
                arr = super().embed_query(query)
                arr[0, 1] = np.inf
                return arr

        embedder = InfQueryEmbedder(dimension=4)
        store = FAISSVectorStore(embedder=embedder)
        store.add_chunks([self.chunk_pdf])

        with self.assertRaises(ValueError) as ctx:
            store.search("query")
        self.assertIn("Query embedding contains non-finite values (NaN or Inf)", str(ctx.exception))

    def test_load_malformed_chunk_missing_required_key_raises_value_error(self):
        """Malformed chunk dictionary missing required key must raise ValueError with original cause."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = FAISSVectorStore(embedder=FakeEmbedder(dimension=4))
            store.add_chunks([self.chunk_pdf])
            store.save(tmp_dir)

            # Corrupt the chunk dictionary in metadata.json by removing "text"
            metadata_file = Path(tmp_dir) / DEFAULT_METADATA_NAME
            with open(metadata_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
            del meta["chunks"][0]["text"]
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(meta, f)

            reloaded = FAISSVectorStore(embedder=FakeEmbedder(dimension=4))
            with self.assertRaises(ValueError) as ctx:
                reloaded.load(tmp_dir)
            self.assertIn("Corrupt or malformed chunk record in metadata JSON", str(ctx.exception))
            self.assertIsInstance(ctx.exception.__cause__, KeyError)

    def test_load_malformed_chunk_non_dict_record_raises_value_error(self):
        """Malformed chunk list containing a non-dict entry must raise ValueError with original cause."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = FAISSVectorStore(embedder=FakeEmbedder(dimension=4))
            store.add_chunks([self.chunk_pdf])
            store.save(tmp_dir)

            metadata_file = Path(tmp_dir) / DEFAULT_METADATA_NAME
            with open(metadata_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
            meta["chunks"][0] = "corrupt_string_entry"
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(meta, f)

            reloaded = FAISSVectorStore(embedder=FakeEmbedder(dimension=4))
            with self.assertRaises(ValueError) as ctx:
                reloaded.load(tmp_dir)
            self.assertIn("Corrupt or malformed chunk record in metadata JSON", str(ctx.exception))
            self.assertIsInstance(ctx.exception.__cause__, TypeError)


if __name__ == "__main__":
    unittest.main()
