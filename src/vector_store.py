"""Vector store module for the Custom AI Document Agent (RAG).

Provides local, persistent vector storage and semantic retrieval using native
FAISS (faiss-cpu) and Sentence Transformers embeddings without any LangChain
dependencies. Chunks are stored in a native FAISS index with metadata safely
persisted as standard JSON.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import faiss
import numpy as np

from src.chunking import DocumentChunk

# Default lightweight model for local Sentence Transformers embeddings
DEFAULT_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
DEFAULT_FAISS_INDEX_NAME: str = "index.faiss"
DEFAULT_METADATA_NAME: str = "metadata.json"


# ============================================================================
# Lazy Embedder Wrapper
# ============================================================================

class SentenceTransformerEmbedder:
    """Lazy wrapper around SentenceTransformer.

    Ensures the model is not instantiated or downloaded until embeddings
    are explicitly requested at runtime.
    """

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate normalized float32 embeddings for a list of text strings."""
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Generate a normalized float32 embedding vector for a single query."""
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty or whitespace.")
        embedding = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        return embedding.astype(np.float32)


# ============================================================================
# Search Result Data Structure
# ============================================================================

@dataclass
class SearchResult:
    """Represents a single retrieved DocumentChunk with its cosine similarity score."""
    chunk: DocumentChunk
    score: float

    def __iter__(self):
        """Enable tuple unpacking: chunk, score = search_result."""
        yield self.chunk
        yield self.score

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for inspection or JSON serialization."""
        return {
            "chunk": self.chunk.to_dict(),
            "score": float(self.score),
        }


# ============================================================================
# Helper Functions for DocumentChunk Serialization
# ============================================================================

def chunk_to_dict(chunk: DocumentChunk) -> Dict[str, Any]:
    """Convert a DocumentChunk to a clean JSON-serializable dictionary."""
    return chunk.to_dict()


def chunk_from_dict(data: Dict[str, Any]) -> DocumentChunk:
    """Reconstruct a DocumentChunk from a dictionary."""
    return DocumentChunk(
        text=data["text"],
        source=data["source"],
        file_type=data["file_type"],
        chunk_index=data["chunk_index"],
        total_chunks=data["total_chunks"],
        chunk_id=data["chunk_id"],
        page=data.get("page"),
        sheet_name=data.get("sheet_name"),
        row_index=data.get("row_index"),
        slide_number=data.get("slide_number"),
        metadata=data.get("metadata", {}),
    )


# ============================================================================
# Core FAISS Vector Store Class
# ============================================================================

class FAISSVectorStore:
    """Persistent local vector store backed by native FAISS and Sentence Transformers.

    Uses faiss.IndexFlatIP with L2-normalized embeddings for exact cosine
    similarity search. Metadata is persisted to disk alongside the index in
    pure JSON format, completely avoiding Python pickle security issues.
    """

    def __init__(
        self,
        embedder: Optional[Any] = None,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        dimension: Optional[int] = None,
    ):
        """Initialize the vector store.

        Args:
            embedder: Optional embedding object with `embed_texts` and `embed_query` methods.
                      If None, a lazy SentenceTransformerEmbedder is used.
            model_name: Name of the Sentence Transformers model to use if embedder is None.
            dimension: Optional vector dimension. If None, inferred upon first chunk addition.
        """
        self.model_name = model_name
        self.embedder = embedder if embedder is not None else SentenceTransformerEmbedder(model_name=model_name)
        self.dimension = dimension
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunks: List[DocumentChunk] = []

        if dimension is not None:
            if not isinstance(dimension, int) or dimension <= 0:
                raise ValueError(f"dimension must be a positive integer, got {dimension}")
            self.dimension = dimension
            self.index = faiss.IndexFlatIP(dimension)

    def count(self) -> int:
        """Return the total number of indexed vectors/chunks."""
        return len(self.chunks)

    def __len__(self) -> int:
        return self.count()

    def clear(self) -> None:
        """Clear all indexed vectors and stored chunks."""
        self.chunks = []
        if self.dimension is not None:
            self.index = faiss.IndexFlatIP(self.dimension)
        else:
            self.index = None

    def add_chunks(self, chunks: List[DocumentChunk]) -> int:
        """Index a list of DocumentChunk instances.

        Embeddings are generated, L2-normalized, and added to the FAISS index.

        Args:
            chunks: List of DocumentChunk instances.

        Returns:
            The number of chunks added.

        Raises:
            TypeError: If chunks is not a list or contains non-DocumentChunk items.
            ValueError: If embedding dimension does not match existing index.
        """
        if not isinstance(chunks, list):
            raise TypeError(f"chunks must be a list of DocumentChunk objects, got {type(chunks).__name__}")

        if not chunks:
            return 0

        for idx, item in enumerate(chunks):
            if not isinstance(item, DocumentChunk):
                raise TypeError(
                    f"Item at index {idx} is of type {type(item).__name__}, expected DocumentChunk"
                )

        texts = [c.text for c in chunks]
        raw_embeddings = self.embedder.embed_texts(texts)

        embeddings = np.ascontiguousarray(raw_embeddings, dtype=np.float32)
        if embeddings.ndim != 2:
            raise ValueError(f"Embeddings must be 2-dimensional (N, D), got shape {embeddings.shape}")

        if embeddings.shape[0] != len(chunks):
            raise ValueError(
                f"Embedding count ({embeddings.shape[0]}) does not match chunk count ({len(chunks)})"
            )

        if not np.all(np.isfinite(embeddings)):
            raise ValueError("Embeddings contain non-finite values (NaN or Inf).")

        # L2-normalize vectors for cosine similarity via IndexFlatIP
        faiss.normalize_L2(embeddings)
        emb_dim = embeddings.shape[1]

        if self.index is None:
            self.dimension = emb_dim
            self.index = faiss.IndexFlatIP(self.dimension)
        elif emb_dim != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dimension}, got {emb_dim}"
            )

        self.index.add(embeddings)
        self.chunks.extend(chunks)

        return len(chunks)

    def search(self, query: str, top_k: int = 4) -> List[SearchResult]:
        """Retrieve the top_k most similar document chunks for a query.

        Args:
            query: Natural language query string.
            top_k: Maximum number of relevant chunks to retrieve (default 4).

        Returns:
            List of SearchResult objects sorted by descending cosine similarity.

        Raises:
            TypeError: If query is not a string or top_k is not an int.
            ValueError: If top_k <= 0 or query embedding dimension mismatches index.
        """
        if not isinstance(query, str):
            raise TypeError(f"query must be a string, got {type(query).__name__}")
        if not isinstance(top_k, int):
            raise TypeError(f"top_k must be an integer, got {type(top_k).__name__}")
        if top_k <= 0:
            raise ValueError(f"top_k must be a positive integer greater than 0, got {top_k}")

        # Empty or whitespace query returns empty results safely
        if not query.strip():
            return []

        # Empty store returns empty results safely
        if self.index is None or self.count() == 0:
            return []

        query_emb = self.embedder.embed_query(query)
        query_vec = np.ascontiguousarray(query_emb, dtype=np.float32)

        if query_vec.ndim == 1:
            query_vec = np.expand_dims(query_vec, axis=0)
        elif query_vec.ndim != 2:
            raise ValueError(f"Query embedding must be 1D or 2D, got shape {query_vec.shape}")

        if not np.all(np.isfinite(query_vec)):
            raise ValueError("Query embedding contains non-finite values (NaN or Inf).")

        if query_vec.shape[1] != self.dimension:
            raise ValueError(
                f"Query embedding dimension mismatch: expected {self.dimension}, got {query_vec.shape[1]}"
            )

        # Normalize query vector for cosine similarity
        faiss.normalize_L2(query_vec)

        k = min(top_k, self.count())
        scores, indices = self.index.search(query_vec, k)

        results: List[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.chunks):
                results.append(SearchResult(chunk=self.chunks[idx], score=float(score)))

        return results

    def save(self, directory: Union[str, Path]) -> None:
        """Persist the FAISS index binary and chunk metadata JSON to disk.

        Args:
            directory: Target directory path where files will be saved.

        Raises:
            ValueError: If directory is empty.
        """
        if not directory:
            raise ValueError("Save directory path cannot be empty.")

        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        index_file = dir_path / DEFAULT_FAISS_INDEX_NAME
        metadata_file = dir_path / DEFAULT_METADATA_NAME

        # Determine dimension: explicitly configured dimension > embedder declared dimension > default (384)
        if self.dimension is not None:
            current_dim = self.dimension
        elif hasattr(self.embedder, "dimension") and isinstance(self.embedder.dimension, int) and self.embedder.dimension > 0:
            current_dim = self.embedder.dimension
            self.dimension = current_dim
        else:
            current_dim = 384
            self.dimension = current_dim

        if self.index is None:
            self.index = faiss.IndexFlatIP(current_dim)
        idx_to_save = self.index

        faiss.write_index(idx_to_save, str(index_file))

        metadata_payload = {
            "version": "1.0",
            "model_name": self.model_name,
            "dimension": current_dim,
            "total_chunks": len(self.chunks),
            "chunks": [chunk_to_dict(c) for c in self.chunks],
        }

        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata_payload, f, indent=2, ensure_ascii=False)

    def load(self, directory: Union[str, Path]) -> None:
        """Load an existing FAISS index and chunk metadata JSON from disk into this instance.

        Args:
            directory: Directory containing index.faiss and metadata.json.

        Raises:
            FileNotFoundError: If directory or required files are missing.
            ValueError: If metadata is corrupt, invalid, or inconsistent with the FAISS index.
        """
        dir_path = Path(directory)
        if not dir_path.is_dir():
            raise FileNotFoundError(f"Vector store directory not found: {directory}")

        index_file = dir_path / DEFAULT_FAISS_INDEX_NAME
        metadata_file = dir_path / DEFAULT_METADATA_NAME

        if not index_file.is_file():
            raise FileNotFoundError(f"Missing FAISS index file: {index_file}")
        if not metadata_file.is_file():
            raise FileNotFoundError(f"Missing metadata JSON file: {metadata_file}")

        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata_payload = json.load(f)
        except Exception as exc:
            raise ValueError(f"Corrupt or unreadable metadata JSON file: {exc}") from exc

        if not isinstance(metadata_payload, dict):
            raise ValueError("Metadata JSON root must be an object/dict")

        required_keys = ["dimension", "total_chunks", "chunks"]
        for key in required_keys:
            if key not in metadata_payload:
                raise ValueError(f"Metadata JSON missing required key: '{key}'")

        expected_dim = metadata_payload["dimension"]
        expected_chunks = metadata_payload["chunks"]

        if not isinstance(expected_chunks, list):
            raise ValueError("Metadata 'chunks' key must be a list")

        loaded_index = faiss.read_index(str(index_file))

        if loaded_index.d != expected_dim:
            raise ValueError(
                f"Inconsistent vector store: FAISS index dimension ({loaded_index.d}) "
                f"does not match metadata dimension ({expected_dim})"
            )

        if loaded_index.ntotal != len(expected_chunks):
            raise ValueError(
                f"Inconsistent vector store: FAISS index has {loaded_index.ntotal} vectors, "
                f"but metadata JSON contains {len(expected_chunks)} chunks"
            )

        # Reconstruct DocumentChunk instances
        try:
            reconstructed_chunks = [chunk_from_dict(c) for c in expected_chunks]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Corrupt or malformed chunk record in metadata JSON: {exc}") from exc

        self.dimension = expected_dim
        self.index = loaded_index
        self.chunks = reconstructed_chunks
        if "model_name" in metadata_payload:
            self.model_name = metadata_payload["model_name"]

    @classmethod
    def load_local(
        cls,
        directory: Union[str, Path],
        embedder: Optional[Any] = None,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
    ) -> "FAISSVectorStore":
        """Factory method to load a persistent vector store from disk.

        Args:
            directory: Directory containing index.faiss and metadata.json.
            embedder: Optional custom embedding object.
            model_name: Embedding model name if default embedder is used.

        Returns:
            A populated, ready-to-search FAISSVectorStore instance.
        """
        store = cls(embedder=embedder, model_name=model_name)
        store.load(directory)
        return store
