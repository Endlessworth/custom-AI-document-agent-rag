
# Custom AI Document Agent (RAG)

A local prototype of a document assistant powered by Retrieval-Augmented Generation (RAG). It lets users ask natural-language questions about their own documents and generates answers using retrieved document content, with source citations.

The project currently supports PDF, Excel (`.xlsx`), and PowerPoint (`.pptx`) files. It was developed as a learning and demonstration project using fictional Nigerian High Court Registry materials.

## Features

- Ingests PDF, Excel, and PowerPoint documents.
- Extracts and chunks document content while retaining source-location metadata.
- Creates local embeddings using Sentence Transformers.
- Stores and searches vectors using FAISS.
- Uses Google Gemini to generate answers grounded in retrieved context.
- Displays source citations and document locations.
- Provides a local browser interface using Gradio.
- Includes automated tests for the main application modules.

## Architecture

The application uses a modular Python architecture:

- **Document loading:** extracts text and structured content from supported files.
- **Chunking:** divides extracted content into smaller sections and retains source metadata.
- **Vector store:** uses Sentence Transformers embeddings and a FAISS index for local similarity search.
- **Retrieval:** searches indexed content and applies source-aware prioritisation for relevant query types.
- **Answer generation:** sends retrieved context to Google Gemini to generate a response.
- **User interface:** provides document ingestion and question-answering through Gradio.

Embeddings and vector search run locally. When a question is submitted, retrieved document text is sent to the configured Gemini API to generate the answer.

## Supported file formats

| Format | Extension | Content |
|---|---|---|
| PDF | `.pdf` | Extractable digital text |
| Excel | `.xlsx` | Workbook sheets and row-based content |
| PowerPoint | `.pptx` | Slide text and supported slide content |

Scanned PDFs that contain no extractable text may require OCR, which is not currently part of the documented workflow.

## MY RAG PROJECT

```text
MY RAG PROJECT/
├── data/                 # Persisted FAISS index and metadata
├── documents/            # Documents to ingest
├── src/
│   ├── app.py            # Gradio application
│   ├── chunking.py       # Document chunking
│   ├── generator.py      # Gemini answer generation
│   ├── ingestion.py      # Ingestion workflow
│   ├── loaders.py        # Supported document loaders
│   ├── retrieval.py      # Search and retrieval logic
│   ├── vector_store.py   # Embeddings and FAISS storage
│   └── __init__.py
├── tests/                # Automated tests
├── .env.example          # Environment-variable template
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup on Windows

The commands below use PowerShell and the project's existing virtual environment.

### 1. Open the project folder

```powershell
Set-Location "C:\Users\HomePC\Downloads\MY RAG PROJECT"
```

### 2. Install dependencies if needed

The project uses Python 3.12. If setting up a new environment, create and install dependencies with:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3. Configure the Gemini API key

Copy the example configuration:

```powershell
Copy-Item .env.example .env
notepad .env
```

Replace the placeholder value for `GEMINI_API_KEY` with your own key from [Google AI Studio](https://aistudio.google.com/). Save the file.

Keep the real `.env` file private. Do not commit it to version control or share its contents.

### 4. Launch the application

```powershell
.\.venv\Scripts\python.exe -m src.app
```

Open the local URL shown in the terminal, typically:

```text
http://127.0.0.1:7860
```

## Using the application

1. Open the application in your browser.
2. Use the document-ingestion interface to index supported files.
3. Ask a question about the indexed documents.
4. Review the answer and its source citations.

For the demonstration dataset, the project includes fictional High Court Registry materials. These documents are examples for testing and are not official court records or legal advice.

## Running the automated tests

From the project directory, run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The test suite covers the application's main modules, including loading, chunking, ingestion, vector storage, retrieval, generation, and the interface.

## Privacy and security

- Keep your Gemini API key in the local `.env` file.
- Local embedding and FAISS search are performed on the machine.
- Retrieved text is sent to the configured Gemini API for answer generation.
- Do not ingest confidential or sensitive documents unless you have confirmed that your use of the configured API and environment is appropriate for those documents.

## Current limitations

- This is a local learning and demonstration prototype, not a production or enterprise-ready system.
- Answer generation depends on Gemini API access, availability, and quota.
- Retrieval and generated answers can be imperfect; users should check the cited source material.
- The application has not been configured as a multi-user hosted service with production access controls.
- OCR for scanned documents is not included in the current documented workflow.

## Future work

Potential future improvements may include broader evaluation with real user workflows, improved handling of scanned documents, and deployment options. These are not required for the current prototype.
