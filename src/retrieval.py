"""Retrieval and context assembly module for the Custom AI Document Agent (RAG).

Provides functions to:
1. Safely load an existing persistent vector store from disk without modifying it.
2. Retrieve top-k semantically relevant chunks for a user query with optional
   similarity threshold filtering.
3. Format retrieved chunks into clear, demarcated prompt context for LLM generation.
4. Format structured citations and human-readable source attributions for UI display.

Adheres strictly to the existing FAISSVectorStore, DocumentChunk, and SearchResult
data structures with zero external framework dependencies (no LangChain).
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.chunking import DocumentChunk
from src.vector_store import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_FAISS_INDEX_NAME,
    DEFAULT_METADATA_NAME,
    FAISSVectorStore,
    SearchResult,
)


# ============================================================================
# Exceptions
# ============================================================================

class VectorStoreNotFoundError(FileNotFoundError):
    """Raised when the vector store directory or index files do not exist."""
    pass


class VectorStoreCorruptedError(ValueError):
    """Raised when the vector store files are incomplete, corrupted, or mismatched."""
    pass


# ============================================================================
# Vector Store Loading
# ============================================================================

def load_vector_store(
    data_directory: Union[str, Path] = "data",
    embedder: Optional[Any] = None,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> FAISSVectorStore:
    """Load an existing persistent FAISS vector store from disk.

    This function is strictly read-only: it does not create, modify, or ingest
    documents. If the directory or required files (index.faiss, metadata.json)
    are missing or corrupt, it raises clear, descriptive exceptions.

    Args:
        data_directory: Path to directory containing index.faiss and metadata.json.
        embedder: Optional custom embedder (e.g., FakeEmbedder for tests).
                  Defaults to lazy SentenceTransformerEmbedder.
        model_name: Model name used if default lazy embedder is instantiated.

    Returns:
        A ready-to-query FAISSVectorStore instance.

    Raises:
        VectorStoreNotFoundError: If data_directory or index files are missing.
        VectorStoreCorruptedError: If index or metadata files are corrupt or mismatched.
    """
    path = Path(data_directory)
    if not path.is_dir():
        raise VectorStoreNotFoundError(
            f"Vector store directory not found: '{data_directory}'. "
            "Please run document ingestion first before attempting retrieval."
        )

    index_file = path / DEFAULT_FAISS_INDEX_NAME
    metadata_file = path / DEFAULT_METADATA_NAME

    if not index_file.is_file() and not metadata_file.is_file():
        raise VectorStoreNotFoundError(
            f"Vector store files ('{DEFAULT_FAISS_INDEX_NAME}' and '{DEFAULT_METADATA_NAME}') "
            f"are missing in '{data_directory}'. Please run document ingestion first."
        )
    if not index_file.is_file():
        raise VectorStoreCorruptedError(
            f"Incomplete vector store in '{data_directory}': missing FAISS index '{DEFAULT_FAISS_INDEX_NAME}'."
        )
    if not metadata_file.is_file():
        raise VectorStoreCorruptedError(
            f"Incomplete vector store in '{data_directory}': missing metadata file '{DEFAULT_METADATA_NAME}'."
        )

    try:
        return FAISSVectorStore.load_local(
            directory=path,
            embedder=embedder,
            model_name=model_name,
        )
    except FileNotFoundError as exc:
        raise VectorStoreNotFoundError(str(exc)) from exc
    except ValueError as exc:
        raise VectorStoreCorruptedError(f"Vector store in '{data_directory}' is corrupted: {exc}") from exc
    except Exception as exc:
        raise VectorStoreCorruptedError(
            f"Unexpected error loading vector store from '{data_directory}': {exc}"
        ) from exc


# ============================================================================
# Retrieval Function
# ============================================================================

def retrieve(
    query: str,
    vector_store: FAISSVectorStore,
    top_k: int = 4,
    score_threshold: Optional[float] = None,
) -> List[SearchResult]:
    """Retrieve the top-k most semantically similar document chunks for a query.

    Retrieval uses exact cosine similarity via FAISS IndexFlatIP with normalized
    embeddings. The returned SearchResult and DocumentChunk objects are preserved
    intact.

    Similarity Threshold Safeguard:
        - When `score_threshold` is None (the default), the vector store's standard
          top-k behaviour is strictly preserved.
        - When `score_threshold` is provided, chunks with cosine similarity score
          strictly less than `score_threshold` are filtered out.
        - Important: A similarity score alone does not guarantee semantic relevance
          or factual grounding. Threshold suitability varies by domain, embedding
          model, and question type, and must be empirically evaluated against real
          queries before fixing a default. Do not assume arbitrary thresholds
          (e.g., 0.7) are universally effective.

    Args:
        query: Natural language question or search string.
        vector_store: Initialized FAISSVectorStore instance to search within.
        top_k: Maximum number of top results to retrieve (default 4, must be > 0).
        score_threshold: Optional minimum cosine similarity score [-1.0, 1.0].
                         Chunks with score < score_threshold are excluded.

    Returns:
        List of SearchResult objects (chunk, score) sorted by descending similarity.
        Returns an empty list if query is blank, vector store is empty, or all
        results fall below the optional threshold.

    Raises:
        TypeError: If query is not a string, vector_store is invalid, top_k is not int,
                   or score_threshold is not a float/int when provided.
        ValueError: If top_k <= 0 or score_threshold is non-finite or outside [-1.0, 1.0].
    """
    if not isinstance(query, str):
        raise TypeError(f"query must be a string, got {type(query).__name__}")

    if not isinstance(vector_store, FAISSVectorStore):
        raise TypeError(
            f"vector_store must be a FAISSVectorStore instance, got {type(vector_store).__name__}"
        )

    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError(f"top_k must be an integer, got {type(top_k).__name__}")
    if top_k <= 0:
        raise ValueError(f"top_k must be a positive integer greater than 0, got {top_k}")

    if score_threshold is not None:
        if isinstance(score_threshold, bool) or not isinstance(score_threshold, (int, float)):
            raise TypeError(
                f"score_threshold must be a float or int, got {type(score_threshold).__name__}"
            )
        score_val = float(score_threshold)
        if not math.isfinite(score_val):
            raise ValueError("score_threshold must be a finite number.")
        if score_val < -1.0 or score_val > 1.0:
            raise ValueError(
                f"score_threshold must be within [-1.0, 1.0] for cosine similarity, got {score_threshold}"
            )

    # Empty or whitespace query returns empty results safely (consistent with vector_store.search)
    if not query.strip():
        return []

    # Safe return if store is unindexed or empty
    if vector_store.count() == 0:
        return []

        # Detect an exact case reference in the user's query.
    case_ref_pattern =  r"\b[A-Z0-9]+(?:/[A-Z0-9]+){2,}\b"
    case_refs = re.findall(case_ref_pattern, query, flags=re.IGNORECASE)

    if case_refs:
        # Retrieve all indexed chunks so an exact case match is not
        # hidden below the semantic Top K results.
        all_results = vector_store.search(
            query=query,
            top_k=vector_store.count(),
        )

        exact_ref = case_refs[0].casefold()

        # Identify chunks containing the exact case reference.
        exact_matches = [
            result for result in all_results
            if exact_ref in result.chunk.text.casefold()
        ]

        # Put the Case Register match first, while preserving the
        # original similarity order within each group.
        exact_matches.sort(
            key=lambda result: (
                0
                if str(result.chunk.sheet_name or "").casefold()
                == "case register"
                else 1
            )
        )

        # Keep exact matches first, followed by the remaining
        # semantic results in their original order.
        exact_ids = {id(result) for result in exact_matches}
        remaining_results = [
            result for result in all_results
            if id(result) not in exact_ids
        ]

        results = (exact_matches + remaining_results)[:top_k]
   
    elif (
        re.search(r"\b(all|every|complete|entire)\b", query, flags=re.IGNORECASE)
        and re.search(r"\bcase\s+register\b", query, flags=re.IGNORECASE)
    ):
        # Comprehensive Case Register queries must retrieve every row
        # from the identified worksheet, not just the semantic Top K.
        all_results = vector_store.search(
            query=query,
            top_k=vector_store.count(),
        )

        results = [
            result for result in all_results
            if str(result.chunk.sheet_name or "").casefold()
            == "case register"
        ]

    else:
        # Explicit training questions should keep ordinary semantic ranking.
        training_query = re.search(
            r"\b(training|test questions?|sample questions?|quiz|exercise)\b",
            query,
            flags=re.IGNORECASE,
        )

        # Identify queries asking for facts about a particular case,
        # even when the user has not supplied its exact reference.
        case_query = re.search(
            r"\b(which|what|identify|find|show|list)\b.*\bcases?\b"
            r"|\bcases?\b.*\b(status|awaiting|missing|pending|review|verification)\b",
            query,
            flags=re.IGNORECASE,
        )

        # Identify procedural questions about registry work.
        procedure_query = re.search(
            r"\b(what should|how do|how should|procedure|process|steps|"
            r"guidance|registry staff|registry officer)\b"
            r"|\bmissing information\b",
            query,
            flags=re.IGNORECASE,
        )

        if training_query:
            # Training material remains searchable when explicitly requested.
            results = vector_store.search(query=query, top_k=top_k)

        elif case_query or procedure_query:
            # Retrieve all candidates so a preferred source is not hidden
            # below the semantic Top K.
            all_results = vector_store.search(
                query=query,
                top_k=vector_store.count(),
            )

            if case_query:
                # Case Register is the preferred source for case-specific facts.
                all_results.sort(
                    key=lambda result: (
                        0
                        if str(result.chunk.sheet_name or "").casefold()
                        == "case register"
                        else 1,
                        -result.score,
                    )
                )

            else:
                # Registry Guide and Admin Guidance are preferred for procedures.
                # Test Questions remain available, but rank after other sources.
                all_results.sort(
                    key=lambda result: (
                        0
                        if (
                            "guide" in str(result.chunk.source or "").casefold()
                            or str(result.chunk.sheet_name or "").casefold()
                            == "admin guidance"
                        )
                        else (
                            2
                            if str(result.chunk.sheet_name or "").casefold()
                            == "test questions"
                            else 1
                        ),
                        -result.score,
                    )
                )

            results = all_results[:top_k]

        else:
            # Unclassified queries retain the original semantic Top K behaviour.
            results = vector_store.search(query=query, top_k=top_k)
   

    if score_threshold is not None:
        threshold_val = float(score_threshold)
        results = [r for r in results if r.score >= threshold_val]

    return results


# ============================================================================
# Source Citation & Location Extraction Helpers
# ============================================================================

def format_source_location(chunk: DocumentChunk) -> str:
    """Format an accurate, human-readable source location string for a chunk.

    Inspects actual chunk metadata fields without fabricating missing information:
    - PDF: Page number (e.g., 'Page 3').
    - Excel: Worksheet and row number (e.g., "Sheet 'Allocations', Row 12").
    - PowerPoint: Slide number (e.g., 'Slide 5').
    - Fallback: Empty string if no location coordinates are available.

    Args:
        chunk: DocumentChunk instance.

    Returns:
        Human-readable location string, or empty string if no location data exists.
    """
    parts: List[str] = []

    # PDF page attribution
    if chunk.page is not None:
        parts.append(f"Page {chunk.page}")

    # Excel sheet and row attribution
    if chunk.sheet_name is not None and chunk.sheet_name.strip():
        parts.append(f"Sheet '{chunk.sheet_name.strip()}'")
    if chunk.row_index is not None:
        parts.append(f"Row {chunk.row_index}")

    # PowerPoint slide attribution
    if chunk.slide_number is not None:
        parts.append(f"Slide {chunk.slide_number}")

    return ", ".join(parts)


def format_citation_label(chunk: DocumentChunk) -> str:
    """Generate a clean citation header label combining source name and location.

    Examples:
        - "report.pdf (Page 3)"
        - "budget.xlsx (Sheet 'Allocations', Row 12)"
        - "deck.pptx (Slide 2)"
        - "notes.pdf" (if page is missing)

    Args:
        chunk: DocumentChunk instance.

    Returns:
        Combined source and location label.
    """
    source_name = chunk.source if chunk.source else "Unknown source"
    location = format_source_location(chunk)
    if location:
        return f"{source_name} ({location})"
    return source_name


# ============================================================================
# Context Formatter for LLM Prompts
# ============================================================================

def format_retrieved_context(
    results: List[SearchResult],
    separator: str = "\n\n---\n\n",
    empty_fallback: str = "No relevant document context found.",
) -> str:
    """Format retrieved chunks into clearly demarcated context for LLM prompts.

    Each chunk is preceded by an unambiguous source attribution header:
    `[Source: <filename> | <location>]`
    Only available metadata is printed; missing coordinates are never fabricated.

    Args:
        results: List of SearchResult objects from retrieve().
        separator: Delimiter between consecutive chunk blocks (default '\\n\\n---\\n\\n').
        empty_fallback: String returned if results list is empty.

    Returns:
        Formatted context string ready for prompt injection.
    """
    if not results:
        return empty_fallback

    blocks: List[str] = []
    for r in results:
        chunk = r.chunk
        location = format_source_location(chunk)
        source_name = chunk.source if chunk.source else "Unknown source"

        if location:
            header = f"[Source: {source_name} | {location}]"
        else:
            header = f"[Source: {source_name}]"

        text_content = chunk.text.strip() if chunk.text else ""
        blocks.append(f"{header}\n{text_content}")

    return separator.join(blocks)


# ============================================================================
# Structured Citation Formatter for UI
# ============================================================================

def format_citations(
    results: List[SearchResult],
    snippet_length: int = 200,
) -> List[Dict[str, Any]]:
    """Format retrieved results into structured citation records for UI display.

    Returns clean dictionaries containing document source, coordinates, similarity
    score, and a preview snippet. Does not alter underlying DocumentChunk objects.

    Args:
        results: List of SearchResult objects.
        snippet_length: Maximum characters of text snippet for UI preview (default 200).

    Returns:
        List of citation dictionaries with keys:
            - source: Filename string
            - file_type: Extension string ('pdf', 'xlsx', 'pptx')
            - location: Human-readable location string (e.g. 'Page 3')
            - label: Combined citation label (e.g. 'report.pdf (Page 3)')
            - score: Cosine similarity score (float rounded to 4 decimals)
            - chunk_id: Stable chunk identifier
            - page: Optional[int]
            - sheet_name: Optional[str]
            - row_index: Optional[int]
            - slide_number: Optional[int]
            - snippet: Text preview string
    """
    citations: List[Dict[str, Any]] = []

    for r in results:
        chunk = r.chunk
        raw_text = chunk.text.strip() if chunk.text else ""
        if len(raw_text) > snippet_length:
            snippet = raw_text[:snippet_length].rstrip() + "..."
        else:
            snippet = raw_text

        citations.append({
            "source": chunk.source,
            "file_type": chunk.file_type,
            "location": format_source_location(chunk),
            "label": format_citation_label(chunk),
            "score": round(float(r.score), 4),
            "chunk_id": chunk.chunk_id,
            "page": chunk.page,
            "sheet_name": chunk.sheet_name,
            "row_index": chunk.row_index,
            "slide_number": chunk.slide_number,
            "snippet": snippet,
        })

    return citations
