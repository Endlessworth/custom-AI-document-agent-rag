"""Document ingestion pipeline for the Custom AI Document Agent (RAG).

Orchestrates loading multi-format documents (PDF, Excel, PowerPoint),
chunking them with semantic boundary preservation, indexing them in native
FAISS, and atomically persisting the vector database with rollback protection.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from src.chunking import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, DocumentChunk, chunk_sections
from src.loaders import (
    DocumentLoaderError,
    DocumentSection,
    EmptyDocumentError,
    ScannedDocumentError,
    UnsupportedFormatError,
    load_document,
)
from src.vector_store import (
    DEFAULT_FAISS_INDEX_NAME,
    DEFAULT_METADATA_NAME,
    FAISSVectorStore,
)

# Supported document formats matching loaders.py
SUPPORTED_EXTENSIONS: Set[str] = {
    ".pdf",
    ".xlsx",
    ".xlsm",
    ".xltx",
    ".xltm",
    ".pptx",
    ".pptm",
    ".potx",
}


# ============================================================================
# Ingestion Result Structure
# ============================================================================

@dataclass
class IngestionResult:
    """Detailed summary of a document ingestion run.

    Attributes:
        success: True only if at least one file was indexed and zero errors occurred.
        total_files_found: Count of files examined in the source location.
        files_indexed: List of filenames successfully parsed, chunked, and indexed.
        total_sections: Total DocumentSection count extracted across all files.
        total_chunks: Total DocumentChunk count generated and added to the index.
        errors: List of error dictionaries: [{"file": filename, "error": error_message}].
        skipped_files: List of skipped file dictionaries: [{"file": filename, "reason": reason}].
        data_directory: Target persistent directory path.
        vector_store: The populated FAISSVectorStore instance, or None if ingestion failed.
    """
    success: bool
    total_files_found: int
    files_indexed: List[str]
    total_sections: int
    total_chunks: int
    errors: List[Dict[str, str]]
    skipped_files: List[Dict[str, str]]
    data_directory: str
    vector_store: Optional[FAISSVectorStore] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result summary into a clean dictionary."""
        return {
            "success": self.success,
            "total_files_found": self.total_files_found,
            "files_indexed": self.files_indexed,
            "total_sections": self.total_sections,
            "total_chunks": self.total_chunks,
            "errors": self.errors,
            "skipped_files": self.skipped_files,
            "data_directory": str(self.data_directory),
        }


# ============================================================================
# Safe Rollback-Capable Index Replacement Procedure
# ============================================================================

def _atomic_replace_index(
    staging_dir: Path,
    target_dir: Path,
    expected_chunks: int,
    embedder: Optional[Any] = None,
) -> None:
    """Atomically replace target directory index files with staged index files.

    Uses a rollback-capable procedure suitable for Windows:
    1. Backs up existing index files if present.
    2. Copies staged files to the target directory.
    3. Validates that the target directory can be loaded and chunk counts agree.
    4. Rolls back to the backup if copying or validation fails.
    5. Cleans up backups upon verified success.

    Args:
        staging_dir: Temporary directory containing the validated new index files.
        target_dir: Destination data directory (e.g., 'data').
        expected_chunks: Number of chunks expected to be verified upon reload.
        embedder: Optional embedder instance used for validation reload.

    Raises:
        FileNotFoundError: If staging files are missing.
        RuntimeError: If replacement or validation fails (after restoring backup).
    """
    target_dir.mkdir(parents=True, exist_ok=True)

    staging_index = staging_dir / DEFAULT_FAISS_INDEX_NAME
    staging_meta = staging_dir / DEFAULT_METADATA_NAME

    if not staging_index.is_file() or not staging_meta.is_file():
        raise FileNotFoundError(f"Staged index files missing in {staging_dir}")

    target_index = target_dir / DEFAULT_FAISS_INDEX_NAME
    target_meta = target_dir / DEFAULT_METADATA_NAME

    backup_index = target_dir / (DEFAULT_FAISS_INDEX_NAME + ".backup")
    backup_meta = target_dir / (DEFAULT_METADATA_NAME + ".backup")

    has_existing = target_index.is_file() and target_meta.is_file()

    # Step 1: Backup existing index files before touching target
    if has_existing:
        shutil.copy2(target_index, backup_index)
        shutil.copy2(target_meta, backup_meta)

    # Step 2: Copy staged files to target and validate
    try:
        shutil.copy2(staging_index, target_index)
        shutil.copy2(staging_meta, target_meta)

        # Step 3: Validate that target index can be loaded and counts match
        verify_store = FAISSVectorStore.load_local(target_dir, embedder=embedder)
        if verify_store.count() != expected_chunks:
            raise ValueError(
                f"Target verification failed: expected {expected_chunks} chunks, got {verify_store.count()}"
            )
    except Exception as exc:
        # Step 4: Rollback
        if has_existing:
            if backup_index.is_file():
                shutil.copy2(backup_index, target_index)
            if backup_meta.is_file():
                shutil.copy2(backup_meta, target_meta)
        else:
            # Clean up partial target files if no previous index existed
            target_index.unlink(missing_ok=True)
            target_meta.unlink(missing_ok=True)
        raise RuntimeError(f"Index replacement failed and was rolled back: {exc}") from exc
    finally:
        # Step 5: Clean up backup files
        backup_index.unlink(missing_ok=True)
        backup_meta.unlink(missing_ok=True)


# ============================================================================
# Core Ingestion Functions
# ============================================================================

def ingest_file(
    file_path: Union[str, Path],
    data_directory: Optional[Union[str, Path]] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    embedder: Optional[Any] = None,
) -> IngestionResult:
    """Ingest a single document file into a clean vector store.

    Semantics:
        Processes the single file into a fresh vector store. If data_directory
        is provided, the replacement procedure safely updates the target directory
        without corrupting any previous index should processing fail. It never
        silently appends duplicates.

    Args:
        file_path: Path to the target document.
        data_directory: Optional target directory to persist the generated index.
        chunk_size: Maximum character count per chunk.
        chunk_overlap: Overlap characters between chunks.
        embedder: Optional custom embedding object (defaults to SentenceTransformer).

    Returns:
        IngestionResult summary of the ingestion run.
    """
    path = Path(file_path)
    data_dir_str = str(data_directory) if data_directory else ""

    if not path.is_file():
        return IngestionResult(
            success=False,
            total_files_found=0,
            files_indexed=[],
            total_sections=0,
            total_chunks=0,
            errors=[{"file": str(file_path), "error": "File not found"}],
            skipped_files=[],
            data_directory=data_dir_str,
            vector_store=None,
        )

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return IngestionResult(
            success=False,
            total_files_found=1,
            files_indexed=[],
            total_sections=0,
            total_chunks=0,
            errors=[],
            skipped_files=[{"file": path.name, "reason": f"Unsupported format '{ext}'"}],
            data_directory=data_dir_str,
            vector_store=None,
        )

    try:
        sections = load_document(path)
    except (DocumentLoaderError, FileNotFoundError) as exc:
        return IngestionResult(
            success=False,
            total_files_found=1,
            files_indexed=[],
            total_sections=0,
            total_chunks=0,
            errors=[{"file": path.name, "error": str(exc)}],
            skipped_files=[],
            data_directory=data_dir_str,
            vector_store=None,
        )
    except Exception as exc:
        return IngestionResult(
            success=False,
            total_files_found=1,
            files_indexed=[],
            total_sections=0,
            total_chunks=0,
            errors=[{"file": path.name, "error": f"Unexpected extraction error: {exc}"}],
            skipped_files=[],
            data_directory=data_dir_str,
            vector_store=None,
        )

    chunks = chunk_sections(sections, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        return IngestionResult(
            success=False,
            total_files_found=1,
            files_indexed=[],
            total_sections=len(sections),
            total_chunks=0,
            errors=[{"file": path.name, "error": "Document produced no text chunks"}],
            skipped_files=[],
            data_directory=data_dir_str,
            vector_store=None,
        )

    store = FAISSVectorStore(embedder=embedder)
    store.add_chunks(chunks)

    # Safe persistence if data_directory is specified
    if data_directory:
        target_path = Path(data_directory)
        with tempfile.TemporaryDirectory() as staging_dir:
            store.save(staging_dir)
            _atomic_replace_index(Path(staging_dir), target_path, len(chunks), embedder=embedder)

    return IngestionResult(
        success=True,
        total_files_found=1,
        files_indexed=[path.name],
        total_sections=len(sections),
        total_chunks=len(chunks),
        errors=[],
        skipped_files=[],
        data_directory=data_dir_str,
        vector_store=store,
    )


def ingest_directory(
    documents_directory: Union[str, Path] = "documents",
    data_directory: Union[str, Path] = "data",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    recursive: bool = False,
    embedder: Optional[Any] = None,
) -> IngestionResult:
    """Ingest all supported documents from a directory into a persistent vector store.

    Performs a safe, full rebuild:
    1. Scans documents_directory for supported and unsupported files.
    2. Parses documents with per-file exception isolation so valid files continue.
    3. Chunks all extracted sections.
    4. If no valid chunks are produced (or directory is empty/missing), preserves
       any existing index without overwriting.
    5. Builds the index in a temporary staging location.
    6. Validates the staged index before atomically updating data_directory with
       rollback protection.
    7. Prevents duplicate chunks on repeated runs.

    Args:
        documents_directory: Source directory containing files (default 'documents').
        data_directory: Persistent storage location (default 'data').
        chunk_size: Maximum character count per chunk.
        chunk_overlap: Overlap characters between chunks.
        recursive: Whether to scan subdirectories.
        embedder: Optional custom embedding object (defaults to SentenceTransformer).

    Returns:
        IngestionResult summary of the batch ingestion run.
    """
    docs_path = Path(documents_directory)
    data_path = Path(data_directory)
    data_dir_str = str(data_path)

    if not docs_path.is_dir():
        return IngestionResult(
            success=False,
            total_files_found=0,
            files_indexed=[],
            total_sections=0,
            total_chunks=0,
            errors=[{"file": str(documents_directory), "error": "Documents directory not found"}],
            skipped_files=[],
            data_directory=data_dir_str,
            vector_store=None,
        )

    pattern = "**/*" if recursive else "*"
    all_files = [p for p in docs_path.glob(pattern) if p.is_file()]
    total_files_found = len(all_files)

    if total_files_found == 0:
        return IngestionResult(
            success=False,
            total_files_found=0,
            files_indexed=[],
            total_sections=0,
            total_chunks=0,
            errors=[],
            skipped_files=[],
            data_directory=data_dir_str,
            vector_store=None,
        )

    supported_files: List[Path] = []
    skipped_files: List[Dict[str, str]] = []

    for file_path in all_files:
        ext = file_path.suffix.lower()
        if ext in SUPPORTED_EXTENSIONS:
            supported_files.append(file_path)
        else:
            skipped_files.append({"file": file_path.name, "reason": f"Unsupported format '{ext}'"})

    if not supported_files:
        return IngestionResult(
            success=False,
            total_files_found=total_files_found,
            files_indexed=[],
            total_sections=0,
            total_chunks=0,
            errors=[],
            skipped_files=skipped_files,
            data_directory=data_dir_str,
            vector_store=None,
        )

    all_sections: List[DocumentSection] = []
    files_indexed: List[str] = []
    errors: List[Dict[str, str]] = []

    for file_path in supported_files:
        try:
            sections = load_document(file_path)
            if sections:
                all_sections.extend(sections)
                files_indexed.append(file_path.name)
            else:
                errors.append({"file": file_path.name, "error": "Document produced 0 sections"})
        except (DocumentLoaderError, FileNotFoundError) as exc:
            errors.append({"file": file_path.name, "error": str(exc)})
        except Exception as exc:
            errors.append({"file": file_path.name, "error": f"Unexpected extraction error: {exc}"})

    if not all_sections:
        # All files failed or produced no extractable text; do NOT touch data_directory
        return IngestionResult(
            success=False,
            total_files_found=total_files_found,
            files_indexed=[],
            total_sections=0,
            total_chunks=0,
            errors=errors,
            skipped_files=skipped_files,
            data_directory=data_dir_str,
            vector_store=None,
        )

    chunks = chunk_sections(all_sections, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        return IngestionResult(
            success=False,
            total_files_found=total_files_found,
            files_indexed=files_indexed,
            total_sections=len(all_sections),
            total_chunks=0,
            errors=errors,
            skipped_files=skipped_files,
            data_directory=data_dir_str,
            vector_store=None,
        )

    # Build fresh vector store with new chunks (full rebuild guarantees no duplicates)
    staging_store = FAISSVectorStore(embedder=embedder)
    staging_store.add_chunks(chunks)

    # Staging and rollback-capable atomic replacement
    with tempfile.TemporaryDirectory() as staging_dir:
        staging_store.save(staging_dir)
        _atomic_replace_index(Path(staging_dir), data_path, len(chunks), embedder=embedder)

    is_success = (len(chunks) > 0 and len(errors) == 0)

    return IngestionResult(
        success=is_success,
        total_files_found=total_files_found,
        files_indexed=files_indexed,
        total_sections=len(all_sections),
        total_chunks=len(chunks),
        errors=errors,
        skipped_files=skipped_files,
        data_directory=data_dir_str,
        vector_store=staging_store,
    )
