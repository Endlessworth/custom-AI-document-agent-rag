"""Text chunking module for the Custom AI Document Agent.

This module splits extracted DocumentSection objects into smaller, overlapping
DocumentChunk objects while preserving source attribution and semantic boundaries.
It uses only standard library utilities with zero external dependencies.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.loaders import DocumentSection

# ============================================================================
# Sensible Default Parameters
# ============================================================================

DEFAULT_CHUNK_SIZE: int = 1000
DEFAULT_CHUNK_OVERLAP: int = 150

# Priority list of separators for natural splitting:
# Each entry is (separator_string, character_offset_to_advance)
# For sentence punctuation like ". ", we retain the period in the current chunk.
_SEPARATORS: List[Tuple[str, int]] = [
    ("\n\n", 2),   # Paragraph break
    ("\n", 1),     # Line break
    (". ", 1),     # Period + space
    ("? ", 1),     # Question mark + space
    ("! ", 1),     # Exclamation mark + space
    ("; ", 1),     # Semicolon + space
    (", ", 1),     # Comma + space
    (" ", 0),      # Word boundary (space)
]


# ============================================================================
# Parameter Validation
# ============================================================================

def validate_chunk_params(chunk_size: int, chunk_overlap: int) -> None:
    """Validate chunk size and overlap parameters.

    Args:
        chunk_size: Target character length per chunk.
        chunk_overlap: Number of characters shared between consecutive chunks.

    Raises:
        ValueError: If chunk_size <= 0, chunk_overlap < 0, or chunk_overlap >= chunk_size.
        TypeError: If arguments are not integers.
    """
    if not isinstance(chunk_size, int):
        raise TypeError(f"chunk_size must be an integer, got {type(chunk_size).__name__}")
    if not isinstance(chunk_overlap, int):
        raise TypeError(f"chunk_overlap must be an integer, got {type(chunk_overlap).__name__}")

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be a positive integer greater than 0, got {chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap must be a non-negative integer (>= 0), got {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})"
        )


# ============================================================================
# Core Chunk Data Structure
# ============================================================================

@dataclass
class DocumentChunk:
    """Represents a discrete text chunk derived from a DocumentSection.

    Attributes:
        text: The text content of this chunk.
        source: Filename of the source document.
        file_type: Extension or format ('pdf', 'xlsx', 'pptx').
        chunk_index: 0-indexed position within its parent section.
        total_chunks: Total number of chunks generated for this parent section.
        chunk_id: Stable identifier (e.g. 'doc.pdf:p1:c0').
        page: Page number for PDF documents (1-indexed).
        sheet_name: Worksheet name for Excel spreadsheets.
        row_index: Row number for Excel spreadsheets (1-indexed).
        slide_number: Slide number for PowerPoint presentations (1-indexed).
        metadata: Complete inherited and chunk-specific metadata.
    """
    text: str
    source: str
    file_type: str
    chunk_index: int
    total_chunks: int
    chunk_id: str
    page: Optional[int] = None
    sheet_name: Optional[str] = None
    row_index: Optional[int] = None
    slide_number: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk into a dictionary suitable for vector databases or JSON."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source": self.source,
            "file_type": self.file_type,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "page": self.page,
            "sheet_name": self.sheet_name,
            "row_index": self.row_index,
            "slide_number": self.slide_number,
            "metadata": self.metadata,
        }


# ============================================================================
# Identifier Generator
# ============================================================================

def make_chunk_id(
    source: str,
    file_type: str,
    chunk_index: int,
    page: Optional[int] = None,
    sheet_name: Optional[str] = None,
    row_index: Optional[int] = None,
    slide_number: Optional[int] = None,
) -> str:
    """Generate a clean, unambiguous identifier for a chunk.

    Examples:
        - report.pdf:p2:c0 (PDF page 2, chunk 0)
        - budget.xlsx:sheet_Q4:r5:c0 (Excel sheet Q4, row 5, chunk 0)
        - pitch.pptx:s3:c1 (PowerPoint slide 3, chunk 1)
    """
    locators: List[str] = []
    if page is not None:
        locators.append(f"p{page}")
    if sheet_name is not None:
        locators.append(f"sheet_{sheet_name}")
    if row_index is not None:
        locators.append(f"r{row_index}")
    if slide_number is not None:
        locators.append(f"s{slide_number}")

    locator_str = ":".join(locators) if locators else "sec"
    return f"{source}:{locator_str}:c{chunk_index}"


# ============================================================================
# Pure Text Splitter with Boundary Snapping
# ============================================================================

def split_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[str]:
    """Split a string into overlapping chunks honoring semantic boundaries.

    Paragraph breaks, newlines, sentence ends, and spaces are preferred
    over hard cutoffs. If a section contains no natural breaks, it safely
    falls back to a character cut at chunk_size.

    Args:
        text: The raw text string to split.
        chunk_size: Maximum character count per chunk.
        chunk_overlap: Overlapping character count between consecutive chunks.

    Returns:
        A list of clean, non-empty text chunks.
    """
    validate_chunk_params(chunk_size, chunk_overlap)

    if not text:
        return []

    # Normalize newlines, collapse excessive newline runs, and strip extraneous padding
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"\n{3,}", "\n\n", normalized).strip()
    if not normalized:
        return []

    # If the text already fits into a single chunk, preserve it intact
    if len(normalized) <= chunk_size:
        return [normalized]

    chunks: List[str] = []
    start: int = 0
    text_len: int = len(normalized)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        # If the remainder of the text fits within chunk_size, finish up
        if end == text_len:
            remaining = normalized[start:end].strip()
            if remaining:
                chunks.append(remaining)
            break

        # Search window for a natural split boundary:
        # Require a modest minimum advance beyond chunk_overlap to ensure each chunk
        # makes reasonable forward progress and does not create near-duplicate slivers.
        min_advance = max(1, (chunk_size - chunk_overlap) // 4)
        min_split = min(start + chunk_overlap + min_advance, end - 1)
        split_pos: Optional[int] = None

        if min_split < end:
            window = normalized[min_split:end]
            for sep, offset in _SEPARATORS:
                rpos = window.rfind(sep)
                if rpos != -1:
                    split_pos = min_split + rpos + offset
                    break

        # Fallback to hard cut if no separator was found
        if split_pos is None or split_pos <= start:
            split_pos = end

        chunk_candidate = normalized[start:split_pos].strip()
        if chunk_candidate:
            chunks.append(chunk_candidate)

        # Compute next starting point to maintain overlap
        target_start = split_pos - chunk_overlap
        next_start = target_start

        # Bounded word-boundary snapping:
        # Snap next_start to a word boundary without letting overlap vary by more than
        # half the configured overlap (capped at 20 characters). This prevents overlap
        # from collapsing to zero or expanding excessively.
        max_snap = min(chunk_overlap // 2, 20) if chunk_overlap > 0 else 0
        if max_snap > 0 and target_start > start and target_start < split_pos:
            # Prefer backward snap to slightly favor retaining context over starving it
            sp_back = normalized.rfind(" ", max(start + 1, target_start - max_snap), target_start)
            if sp_back != -1:
                next_start = sp_back + 1
            else:
                # Otherwise search forward, but never cross split_pos - 1
                sp_fwd = normalized.find(" ", target_start, min(target_start + max_snap, split_pos - 1))
                if sp_fwd != -1:
                    next_start = sp_fwd + 1

        # Prevent overlap collapse when chunk_overlap > 0
        if chunk_overlap > 0 and split_pos > start + 1 and next_start >= split_pos:
            next_start = split_pos - 1

        # Guarantee strict forward progress to prevent infinite loops
        if next_start <= start:
            next_start = start + 1

        start = next_start

    return chunks


# ============================================================================
# High-Level Section Chunking Functions
# ============================================================================

def chunk_section(
    section: DocumentSection,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[DocumentChunk]:
    """Chunk a single DocumentSection into one or more DocumentChunks.

    Short sections remain a single intact chunk. Source metadata and context
    are copied directly to all generated chunks.

    Args:
        section: An extracted DocumentSection instance.
        chunk_size: Maximum character count per chunk.
        chunk_overlap: Overlap character count.

    Returns:
        A list of DocumentChunk instances.
    """
    raw_chunks = split_text(section.text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not raw_chunks:
        return []

    total_chunks = len(raw_chunks)
    result: List[DocumentChunk] = []

    for idx, chunk_text in enumerate(raw_chunks):
        cid = make_chunk_id(
            source=section.source,
            file_type=section.file_type,
            chunk_index=idx,
            page=section.page,
            sheet_name=section.sheet_name,
            row_index=section.row_index,
            slide_number=section.slide_number,
        )

        chunk_meta = dict(section.metadata)
        chunk_meta["chunk_size"] = len(chunk_text)

        result.append(
            DocumentChunk(
                text=chunk_text,
                source=section.source,
                file_type=section.file_type,
                chunk_index=idx,
                total_chunks=total_chunks,
                chunk_id=cid,
                page=section.page,
                sheet_name=section.sheet_name,
                row_index=section.row_index,
                slide_number=section.slide_number,
                metadata=chunk_meta,
            )
        )

    return result


def chunk_sections(
    sections: List[DocumentSection],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[DocumentChunk]:
    """Chunk a sequence of DocumentSections into DocumentChunks.

    IMPORTANT: Sections are processed independently and are NEVER combined.
    This ensures that Excel rows, slide boundaries, and document pages are
    never mixed or polluted with adjacent sections.

    Args:
        sections: List of DocumentSection objects to chunk.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap characters.

    Returns:
        A flat list of all generated DocumentChunk objects.
    """
    validate_chunk_params(chunk_size, chunk_overlap)

    all_chunks: List[DocumentChunk] = []
    for section in sections:
        chunks = chunk_section(section, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        all_chunks.extend(chunks)

    return all_chunks
