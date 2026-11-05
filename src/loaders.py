"""Multi-format document loaders for PDF, Excel, and PowerPoint files.

This module provides clean, standalone extraction functions for user-supplied
documents without requiring external framework dependencies (such as LangChain).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import os
import pandas as pd
import pypdf
import pptx


# ============================================================================
# Custom Exceptions for Clear, Actionable Error Reporting
# ============================================================================

class DocumentLoaderError(Exception):
    """Base exception for all document loading and extraction errors."""
    pass


class UnsupportedFormatError(DocumentLoaderError):
    """Raised when an unsupported file extension is provided."""
    pass


class EmptyDocumentError(DocumentLoaderError):
    """Raised when a document has zero bytes or contains no extractable data."""
    pass


class ScannedDocumentError(DocumentLoaderError):
    """Raised when a PDF contains no digital text layer (scanned/image-only)."""
    pass


# ============================================================================
# Standard Extracted Content Structure
# ============================================================================

@dataclass
class DocumentSection:
    """Represents an extracted unit of document text with complete source metadata.

    Attributes:
        text: The extracted clean text.
        source: Filename of the source document.
        file_type: Extension or type of document ('pdf', 'xlsx', 'pptx').
        page: Page number for PDF documents (1-indexed).
        sheet_name: Worksheet name for Excel spreadsheets.
        row_index: Row number for Excel data (1-indexed spreadsheet row).
        slide_number: Slide number for PowerPoint presentations (1-indexed).
        metadata: Additional context information (e.g. total pages, slide titles).
    """
    text: str
    source: str
    file_type: str
    page: Optional[int] = None
    sheet_name: Optional[str] = None
    row_index: Optional[int] = None
    slide_number: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the DocumentSection into a standard dictionary."""
        return {
            "text": self.text,
            "source": self.source,
            "file_type": self.file_type,
            "page": self.page,
            "sheet_name": self.sheet_name,
            "row_index": self.row_index,
            "slide_number": self.slide_number,
            "metadata": self.metadata,
        }


# ============================================================================
# 1. PDF Loader (using pypdf)
# ============================================================================

def load_pdf(file_path: Union[str, Path]) -> List[DocumentSection]:
    """Extract text page-by-page from a PDF document using pypdf.

    Args:
        file_path: Path to the target PDF file.

    Returns:
        A list of DocumentSection instances, one for each page containing text.

    Raises:
        FileNotFoundError: If the file does not exist.
        EmptyDocumentError: If the file is 0 bytes or has 0 pages.
        ScannedDocumentError: If the PDF contains no extractable digital text.
        DocumentLoaderError: If opening or reading the PDF fails.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    if path.stat().st_size == 0:
        raise EmptyDocumentError(f"PDF file '{path.name}' is empty (0 bytes).")

    try:
        with open(str(path), "rb") as f:
            reader = pypdf.PdfReader(f)

            if reader.is_encrypted:
                try:
                    # Attempt to decrypt empty password / default read permission
                    reader.decrypt("")
                except Exception as exc:
                    raise DocumentLoaderError(
                        f"PDF file '{path.name}' is password-protected and cannot be decrypted."
                    ) from exc

            total_pages = len(reader.pages)
            if total_pages == 0:
                raise EmptyDocumentError(f"PDF file '{path.name}' contains 0 pages.")

            sections: List[DocumentSection] = []
            total_extracted_characters = 0
            page_errors: List[Tuple[int, str]] = []

            for page_num, page in enumerate(reader.pages, start=1):
                try:
                    page_text = page.extract_text() or ""
                except Exception as exc:
                    page_errors.append((page_num, str(exc)))
                    continue

                cleaned = page_text.strip()
                if cleaned:
                    total_extracted_characters += len(cleaned)
                    sections.append(
                        DocumentSection(
                            text=cleaned,
                            source=path.name,
                            file_type="pdf",
                            page=page_num,
                            metadata={"total_pages": total_pages}
                        )
                    )
    except DocumentLoaderError:
        raise
    except Exception as exc:
        raise DocumentLoaderError(f"Failed to read PDF file '{path.name}': {exc}") from exc

    # 1. If extraction failed on all pages due to genuine parsing/extraction exceptions
    if len(page_errors) == total_pages:
        first_err = page_errors[0][1]
        raise DocumentLoaderError(
            f"Extraction failed on all {total_pages} page(s) of PDF '{path.name}'. "
            f"Page {page_errors[0][0]} error: {first_err}"
        )

    # 2. If extraction completed without error on all pages, but produced zero text -> scanned/image-based
    if total_extracted_characters == 0 and not page_errors:
        raise ScannedDocumentError(
            f"No extractable text found in '{path.name}'. "
            f"The PDF appears to be scanned or image-based. Optical Character Recognition (OCR) is not yet enabled."
        )

    # 3. If zero text was extracted and there were page errors
    if total_extracted_characters == 0 and page_errors:
        failed_pages_str = ", ".join(f"page {p}: {err}" for p, err in page_errors)
        raise DocumentLoaderError(
            f"Failed to extract text from PDF '{path.name}'. "
            f"Extraction errors occurred on {len(page_errors)} page(s) ({failed_pages_str}) "
            f"and remaining pages contained no text."
        )

    # 4. If some pages succeeded, attach partial page failure warnings to metadata
    if page_errors:
        for section in sections:
            section.metadata["extraction_warnings"] = [
                f"Page {p} extraction error: {err}" for p, err in page_errors
            ]

    return sections


# ============================================================================
# 2. Excel Loader (using pandas and openpyxl)
# ============================================================================

def load_excel(file_path: Union[str, Path]) -> List[DocumentSection]:
    """Extract tabular data sheet-by-sheet from an Excel workbook (.xlsx).

    Args:
        file_path: Path to the target Excel spreadsheet.

    Returns:
        A list of DocumentSection instances, one for each non-empty row.

    Raises:
        FileNotFoundError: If the file does not exist.
        EmptyDocumentError: If the file is 0 bytes or contains no populated rows.
        DocumentLoaderError: If opening or reading the workbook fails, or if
            any individual worksheet encounters a reading/parsing failure.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Excel file not found: {file_path}")

    if path.stat().st_size == 0:
        raise EmptyDocumentError(f"Excel file '{path.name}' is empty (0 bytes).")

    try:
        with pd.ExcelFile(str(path), engine="openpyxl") as excel_file:
            sections: List[DocumentSection] = []
            sheet_errors: List[str] = []

            for sheet_name in excel_file.sheet_names:
                try:
                    df = pd.read_excel(excel_file, sheet_name=sheet_name, engine="openpyxl")
                except Exception as exc:
                    sheet_errors.append(f"Worksheet '{sheet_name}': {str(exc)}")
                    continue

                # Drop entirely empty rows and columns
                clean_df = df.dropna(how="all").dropna(axis=1, how="all")
                if clean_df.empty:
                    continue

                column_headers = [str(col).strip() for col in clean_df.columns]

                for row_idx, row in clean_df.iterrows():
                    field_statements: List[str] = []
                    for col_name, value in row.items():
                        if pd.notna(value):
                            val_str = str(value).strip()
                            if val_str:
                                field_statements.append(f"{str(col_name).strip()}: {val_str}")

                    if field_statements:
                        # 1-indexed Excel row (header is typically row 1, first data row is row 2)
                        excel_row_num = int(row_idx) + 2
                        row_text = f"Sheet: {sheet_name} | Row {excel_row_num} | " + " | ".join(field_statements)

                        sections.append(
                            DocumentSection(
                                text=row_text,
                                source=path.name,
                                file_type="xlsx",
                                sheet_name=sheet_name,
                                row_index=excel_row_num,
                                metadata={
                                    "sheet": sheet_name,
                                    "columns": column_headers
                                }
                            )
                        )
    except DocumentLoaderError:
        raise
    except Exception as exc:
        raise DocumentLoaderError(f"Failed to open Excel workbook '{path.name}': {exc}") from exc

    # If any worksheet threw an exception during loading, report it clearly
    if sheet_errors:
        details = "; ".join(sheet_errors)
        raise DocumentLoaderError(
            f"Failed to read one or more worksheets in Excel workbook '{path.name}': {details}"
        )

    if not sections:
        raise EmptyDocumentError(
            f"No extractable data found in Excel spreadsheet '{path.name}'. All sheets appear empty."
        )

    return sections


# ============================================================================
# 3. PowerPoint Loader (using python-pptx)
# ============================================================================

def load_powerpoint(file_path: Union[str, Path]) -> List[DocumentSection]:
    """Extract slide text, titles, bullet lists, and tables from a presentation (.pptx).

    Args:
        file_path: Path to the target PowerPoint presentation.

    Returns:
        A list of DocumentSection instances, one for each slide containing content.

    Raises:
        FileNotFoundError: If the file does not exist.
        EmptyDocumentError: If the file is 0 bytes or contains no text across slides.
        DocumentLoaderError: If opening or reading the presentation fails.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"PowerPoint file not found: {file_path}")

    if path.stat().st_size == 0:
        raise EmptyDocumentError(f"PowerPoint file '{path.name}' is empty (0 bytes).")

    try:
        prs = pptx.Presentation(str(path))
    except Exception as exc:
        raise DocumentLoaderError(f"Failed to open PowerPoint presentation '{path.name}': {exc}") from exc

    total_slides = len(prs.slides)
    if total_slides == 0:
        raise EmptyDocumentError(f"PowerPoint presentation '{path.name}' contains 0 slides.")

    sections: List[DocumentSection] = []

    for slide_num, slide in enumerate(prs.slides, start=1):
        slide_parts: List[str] = []
        slide_title = ""

        # Extract title if present
        if slide.shapes.title and slide.shapes.title.text:
            cleaned_title = slide.shapes.title.text.strip()
            if cleaned_title:
                slide_title = cleaned_title
                slide_parts.append(f"Slide Title: {slide_title}")

        # Iterate through shapes
        for shape in slide.shapes:
            # Skip repeating the title if already captured
            if shape == slide.shapes.title:
                continue

            # Text boxes and shape text frames
            if shape.has_text_frame:
                frame_text = shape.text_frame.text.strip()
                if frame_text:
                    slide_parts.append(frame_text)

            # Embedded tables
            elif shape.has_table:
                table_lines: List[str] = []
                for row in shape.table.rows:
                    row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                    if row_cells:
                        table_lines.append(" | ".join(row_cells))
                if table_lines:
                    slide_parts.append("Table:\n" + "\n".join(table_lines))

        # Check speaker notes
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
            notes_text = slide.notes_slide.notes_text_frame.text.strip()
            if notes_text:
                slide_parts.append(f"Speaker Notes: {notes_text}")

        combined_slide_text = "\n\n".join(slide_parts).strip()
        if combined_slide_text:
            sections.append(
                DocumentSection(
                    text=combined_slide_text,
                    source=path.name,
                    file_type="pptx",
                    slide_number=slide_num,
                    metadata={
                        "total_slides": total_slides,
                        "slide_title": slide_title or f"Slide {slide_num}"
                    }
                )
            )

    if not sections:
        raise EmptyDocumentError(
            f"No extractable text found in PowerPoint presentation '{path.name}'. "
            f"The presentation contains no text frames or tables."
        )

    return sections


# ============================================================================
# 4. Universal Document Loader Dispatcher
# ============================================================================

def load_document(file_path: Union[str, Path]) -> List[DocumentSection]:
    """Universal dispatcher: automatically inspects file extension and loads document.

    Args:
        file_path: Path to the target file (.pdf, .xlsx, or .pptx).

    Returns:
        A list of DocumentSection instances extracted from the document.

    Raises:
        FileNotFoundError: If the document does not exist.
        UnsupportedFormatError: If the extension is not supported or is legacy binary.
        EmptyDocumentError: If the file is empty.
        ScannedDocumentError: If a PDF is image-only without selectable text.
        DocumentLoaderError: If extraction fails.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Document file not found: {file_path}")

    ext = path.suffix.lower()

    if ext == ".pdf":
        return load_pdf(path)
    elif ext in (".xlsx", ".xlsm", ".xltx", ".xltm"):
        return load_excel(path)
    elif ext in (".pptx", ".pptm", ".potx"):
        return load_powerpoint(path)
    elif ext in (".xls", ".ppt"):
        raise UnsupportedFormatError(
            f"Legacy Microsoft Office format '{ext}' is not supported for '{path.name}'. "
            f"Please save the file as a modern XML-based format (.xlsx or .pptx)."
        )
    else:
        raise UnsupportedFormatError(
            f"Unsupported file format '{ext}' for '{path.name}'. "
            f"Supported formats are: .pdf, .xlsx, .pptx"
        )


# ============================================================================
# 5. Directory Ingestion Helper
# ============================================================================

def load_documents_from_directory(
    directory_path: Union[str, Path],
    recursive: bool = False
) -> Tuple[List[DocumentSection], List[Dict[str, str]]]:
    """Batch loads all supported documents found in a directory.

    Args:
        directory_path: Directory containing source files.
        recursive: Whether to scan subdirectories.

    Returns:
        A tuple of (loaded_sections, errors), where errors is a list of dicts:
        [{'file': filename, 'error': error_message}, ...]
    """
    dir_path = Path(directory_path)
    if not dir_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory_path}")

    all_sections: List[DocumentSection] = []
    errors: List[Dict[str, str]] = []

    pattern = "**/*" if recursive else "*"
    for item in dir_path.glob(pattern):
        if item.is_file() and item.suffix.lower() in (".pdf", ".xlsx", ".xlsm", ".pptx"):
            try:
                sections = load_document(item)
                all_sections.extend(sections)
            except (DocumentLoaderError, FileNotFoundError) as exc:
                errors.append({"file": item.name, "error": str(exc)})
            except Exception as exc:
                errors.append({"file": item.name, "error": f"Unexpected error: {str(exc)}"})

    return all_sections, errors
