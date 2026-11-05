"""Local Gradio web application for the Custom AI Document Agent (RAG).

Provides an interactive browser interface running strictly on 127.0.0.1 (local desktop)
to:
1. Ingest multi-format documents (PDF, Excel, PowerPoint) from a local directory.
2. Submit natural-language questions.
3. View synthesized, grounded answers with clearly delineated, verified citations.
4. Display clear, actionable status and error feedback.

Importing this module is 100% inert and does not launch the server or trigger ingestion.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

from src.generator import NO_EVIDENCE_ANSWER, AnswerResult, answer_query
from src.retrieval import (
    VectorStoreCorruptedError,
    VectorStoreNotFoundError,
    load_vector_store,
)
from src.vector_store import SentenceTransformerEmbedder

DEFAULT_DOCUMENTS_DIR: str = "documents"
DEFAULT_DATA_DIR: str = "data"

_QUERY_EMBEDDER = SentenceTransformerEmbedder()

# ============================================================================
# UI Formatting Helpers
# ============================================================================

def format_citations_for_display(citations: List[Dict[str, Any]]) -> str:
    """Format structured citation dictionaries into clean, human-readable Markdown.

    Args:
        citations: List of citation dictionaries from AnswerResult.citations.

    Returns:
        Formatted Markdown string suitable for display in the UI.
    """
    if not citations:
        return "*No source citations available.*"

    lines: List[str] = [f"### 📚 Source Citations ({len(citations)})", ""]
    for idx, c in enumerate(citations, 1):
        source = c.get("source", "Unknown file")
        location = c.get("location", "")
        score = c.get("score")
        snippet = c.get("snippet", "").strip()

        header_parts = [f"**{idx}. {source}**"]
        if location:
            header_parts.append(f"`{location}`")
        if score is not None:
            header_parts.append(f"*(Cosine score: {score:.4f})*")

        lines.append(" — ".join(header_parts))
        if snippet:
            lines.append(f"> \"{snippet}\"")
        lines.append("")

    return "\n".join(lines).strip()


# ============================================================================
# UI Handler Functions (Decoupled & Testable)
# ============================================================================

def handle_ingestion(
    documents_dir: str = DEFAULT_DOCUMENTS_DIR,
    data_dir: str = DEFAULT_DATA_DIR,
    recursive: bool = False,
) -> str:
    """Handle document ingestion request from the UI.

    Performs a safe, full rebuild of the vector database from the specified
    directory using src.ingestion.ingest_directory.

    Args:
        documents_dir: Path to directory containing source documents.
        data_dir: Destination path for persistent FAISS index and metadata.
        recursive: Whether to scan subdirectories.

    Returns:
        Markdown-formatted summary of ingestion results.
    """
    clean_docs_dir = str(documents_dir).strip() if documents_dir else ""
    clean_data_dir = str(data_dir).strip() if data_dir else DEFAULT_DATA_DIR

    if not clean_docs_dir:
        return "❌ **Error:** Documents directory path cannot be empty."

    try:
        from src.ingestion import ingest_directory
        result = ingest_directory(
            documents_directory=clean_docs_dir,
            data_directory=clean_data_dir,
            recursive=bool(recursive),
        )
    except Exception as exc:
        return f"❌ **Ingestion failed:** Unexpected error: {exc}"

    lines: List[str] = []
    if result.success:
        lines.append(f"### ✅ Ingestion Complete ({result.total_chunks} chunks indexed)")
    elif result.total_chunks > 0 and len(result.errors) > 0:
        lines.append(f"### ⚠️ Partial Ingestion ({result.total_chunks} chunks indexed with errors)")
    else:
        lines.append("### ❌ Ingestion Incomplete (0 chunks indexed)")

    lines.append(f"- **Files Examined:** {result.total_files_found}")
    lines.append(f"- **Files Successfully Indexed:** {len(result.files_indexed)}")
    for f in result.files_indexed:
        lines.append(f"  - `{f}`")
    lines.append(f"- **Total Sections Extracted:** {result.total_sections}")
    lines.append(f"- **Total Chunks Created:** {result.total_chunks}")
    lines.append(f"- **Index Directory:** `{result.data_directory}`")

    if result.skipped_files:
        lines.append("\n**Skipped Files (Unsupported Formats):**")
        for s in result.skipped_files:
            lines.append(f"- `{s.get('file', 'unknown')}`: {s.get('reason', 'unsupported')}")

    if result.errors:
        lines.append("\n**Errors / Extraction Failures:**")
        for e in result.errors:
            lines.append(f"- `{e.get('file', 'unknown')}`: {e.get('error', 'error')}")

    return "\n".join(lines)


def handle_query(
    query: str,
    top_k: int = 4,
    use_threshold: bool = False,
    score_threshold: float = 0.5,
    data_dir: str = DEFAULT_DATA_DIR,
    client: Optional[Any] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    embedder: Optional[Any] = None,
) -> Tuple[str, str, str]:
    """Execute end-to-end question answering pipeline for the UI.

    Args:
        query: User question string.
        top_k: Number of relevant chunks to retrieve (default 4).
        use_threshold: Whether to apply a similarity threshold filter.
        score_threshold: Float threshold value (only applied if use_threshold is True).
        data_dir: Directory where FAISS index and metadata are stored.
        client: Optional injected genai.Client instance (for testing).
        api_key: Optional explicit API key.
        model: Optional model identifier string.
        embedder: Optional embedder instance forwarded to load_vector_store.
                  When None (the default), the module-level _QUERY_EMBEDDER is
                  used, allowing the SentenceTransformer weights to be loaded
                  once and reused across requests. Pass a FakeEmbedder in tests
                  to avoid any model download or network access.

    Returns:
        Tuple of (answer_markdown, citations_markdown, status_message)
    """
    clean_query = str(query).strip() if query else ""
    if not clean_query:
        return (
            "Please enter a question to search your documents.",
            "*No source citations available.*",
            "⚠️ Empty question.",
        )

    try:
        k = int(top_k)
        if k <= 0:
            raise ValueError("top_k must be > 0")
    except (ValueError, TypeError):
        return (
            "Invalid top_k parameter.",
            "*No source citations available.*",
            "❌ Error: top_k must be a positive integer.",
        )

    threshold: Optional[float] = float(score_threshold) if use_threshold else None

    # 1. Load vector store safely
    clean_data_dir = str(data_dir).strip() if data_dir else DEFAULT_DATA_DIR
    query_embedder = embedder if embedder is not None else _QUERY_EMBEDDER
    try:
        vector_store = load_vector_store(data_directory=clean_data_dir, embedder=query_embedder)
    except VectorStoreNotFoundError as exc:
        return (
            "No document index found.",
            "*No source citations available.*",
            f"⚠️ {exc}",
        )
    except VectorStoreCorruptedError as exc:
        return (
            "Vector store index is corrupted.",
            "*No source citations available.*",
            f"❌ {exc}",
        )
    except Exception as exc:
        return (
            "Failed to load vector store.",
            "*No source citations available.*",
            f"❌ Error loading vector store: {exc}",
        )

    # 2. Check if store has any indexed chunks
    if vector_store.count() == 0:
        return (
            NO_EVIDENCE_ANSWER,
            "*No source citations available.*",
            "ℹ️ Document index contains 0 chunks. Please ingest documents first.",
        )

    # 3. Execute answer generation pipeline
    try:
        from src.generator import (
            GeminiAuthError,
            GeminiGenerationError,
            GeminiQuotaError,
            GeminiUnavailableError,
        )
        result = answer_query(
            query=clean_query,
            vector_store=vector_store,
            top_k=k,
            score_threshold=threshold,
            client=client,
            api_key=api_key,
            model=model,
        )
    except GeminiAuthError as exc:
        return (
            "Authentication Error: Unable to communicate with Gemini.",
            "*No source citations available.*",
            f"🔑 {exc}",
        )
    except GeminiQuotaError as exc:
        return (
            "API Quota Exceeded.",
            "*No source citations available.*",
            f"⏳ {exc}",
        )
    except GeminiUnavailableError as exc:
        return (
            "Gemini Service Unavailable.",
            "*No source citations available.*",
            f"📡 {exc}",
        )
    except GeminiGenerationError as exc:
        return (
            "Generation Error.",
            "*No source citations available.*",
            f"❌ {exc}",
        )
    except Exception as exc:
        return (
            "An unexpected error occurred during answer generation.",
            "*No source citations available.*",
            f"❌ Error: {exc}",
        )

    citations_display = format_citations_for_display(result.citations)

    if result.grounded:
        status = f"✅ Answer generated using model '{result.model}' ({len(result.citations)} sources retrieved)."
    else:
        status = "ℹ️ No matching evidence found in the document index."

    return (result.answer, citations_display, status)


# ============================================================================
# Gradio UI Construction
# ============================================================================

def build_app() -> gr.Blocks:
    """Build the Gradio interface without launching the local web server.

    Returns:
        Configured gr.Blocks application instance.
    """
    with gr.Blocks(title="Custom AI Document Agent (RAG)") as demo:
        gr.Markdown("# 📄 Custom AI Document Agent (RAG)")
        gr.Markdown(
            "Ask natural-language questions grounded strictly in your private documents "
            "(PDF, Excel spreadsheets, and PowerPoint presentations)."
        )

        with gr.Tabs():
            # ----------------------------------------------------------------
            # Tab 1: Question & Answer
            # ----------------------------------------------------------------
            with gr.Tab("💬 Question & Answer"):
                with gr.Row():
                    with gr.Column(scale=4):
                        query_input = gr.Textbox(
                            label="Your Question",
                            placeholder="e.g. What were the quarterly financial results for 2026?",
                            lines=2,
                        )

                        with gr.Accordion("⚙️ Retrieval Settings", open=False):
                            top_k_slider = gr.Slider(
                                minimum=1,
                                maximum=10,
                                value=4,
                                step=1,
                                label="Top K Chunks",
                                info="Number of document excerpts retrieved for context.",
                            )
                            use_threshold_cb = gr.Checkbox(
                                value=False,
                                label="Filter by Similarity Score Threshold",
                                info="Exclude chunks scoring below the minimum similarity score.",
                            )
                            threshold_slider = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                value=0.5,
                                step=0.05,
                                label="Minimum Cosine Similarity Score",
                            )

                        with gr.Row():
                            submit_btn = gr.Button("🔍 Submit Question", variant="primary")
                            clear_btn = gr.Button("Clear")

                    with gr.Column(scale=5):
                        status_box = gr.Markdown("ℹ️ Ready for questions.")
                        answer_box = gr.Markdown(
                            "### Answer\n\n*Answers will appear here once generated.*"
                        )
                        citations_box = gr.Markdown("*No source citations yet.*")

                # Wire up query actions
                def _ui_query_wrapper(q, k, use_t, t_val):
                    ans, cit, stat = handle_query(
                        query=q,
                        top_k=k,
                        use_threshold=use_t,
                        score_threshold=t_val,
                    )
                    ans_md = f"### Answer\n\n{ans}"
                    return ans_md, cit, stat

                submit_btn.click(
                    fn=_ui_query_wrapper,
                    inputs=[query_input, top_k_slider, use_threshold_cb, threshold_slider],
                    outputs=[answer_box, citations_box, status_box],
                )

                query_input.submit(
                    fn=_ui_query_wrapper,
                    inputs=[query_input, top_k_slider, use_threshold_cb, threshold_slider],
                    outputs=[answer_box, citations_box, status_box],
                )

                def _clear_inputs():
                    return (
                        "",
                        "### Answer\n\n*Answers will appear here once generated.*",
                        "*No source citations yet.*",
                        "ℹ️ Cleared.",
                    )

                clear_btn.click(
                    fn=_clear_inputs,
                    inputs=[],
                    outputs=[query_input, answer_box, citations_box, status_box],
                )

            # ----------------------------------------------------------------
            # Tab 2: Document Ingestion
            # ----------------------------------------------------------------
            with gr.Tab("📁 Ingest Documents"):
                gr.Markdown(
                    "### Build or Rebuild the Knowledge Base\n"
                    "Place your files (`.pdf`, `.xlsx`, `.pptx`) into the documents directory, "
                    "then click the button below to parse, chunk, and index them into the local FAISS database."
                )

                with gr.Row():
                    with gr.Column(scale=4):
                        docs_dir_input = gr.Textbox(
                            value=DEFAULT_DOCUMENTS_DIR,
                            label="Source Documents Directory",
                        )
                        recursive_cb = gr.Checkbox(
                            value=False,
                            label="Scan Subdirectories Recursively",
                        )
                        ingest_btn = gr.Button(
                            "🚀 Ingest / Rebuild Knowledge Base",
                            variant="primary",
                        )

                    with gr.Column(scale=5):
                        ingest_output = gr.Markdown(
                            "*Ingestion summary will appear here after clicking Ingest.*"
                        )

                ingest_btn.click(
                    fn=handle_ingestion,
                    inputs=[docs_dir_input, gr.State(DEFAULT_DATA_DIR), recursive_cb],
                    outputs=[ingest_output],
                )

    return demo


# ============================================================================
# Main Entry Point
# ============================================================================

def main() -> None:
    """Launch the local Gradio interface for desktop browser use.

    Runs strictly on localhost (127.0.0.1) without public sharing or network exposure.
    """
    demo = build_app()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
    )


if __name__ == "__main__":
    main()
