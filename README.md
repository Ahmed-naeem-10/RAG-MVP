# RAG MVP — Document Q&A System

A local Retrieval-Augmented Generation (RAG) application that lets you ask natural-language questions about a corpus of PDF documents. Answers are grounded strictly in your documents — the system will never fabricate information.

---

## 1. Project Overview

| Component | Technology |
|---|---|
| Orchestration | Python + LangChain |
| Embeddings | OpenAI `text-embedding-3-small` |
| LLM | OpenAI `gpt-4o-mini` |
| Vector Store | ChromaDB (local, persistent) |
| Interface | Streamlit |
| Optional Hybrid Search | BM25 (rank-bm25) + Reciprocal Rank Fusion |

**Key features:**
- Zero-hallucination guarantee via strict system prompts + temperature=0.0 + similarity threshold filtering
- Source citations (file name + page number) alongside every answer
- Conversation history (up to 100 turns per session)
- Optional hybrid search combining dense vector + BM25 keyword retrieval

---

## 2. Prerequisites

- Python **3.10** or higher
- An **OpenAI API key** — get one at https://platform.openai.com/api-keys
- ~500 MB free disk space for ChromaDB and the Python environment
- Your PDF documents (10–20 recommended for the MVP)

---

## 3. Environment Setup

### Step 1 — Clone / download the project

```bash
# If using git:
git clone <your-repo-url>
cd rag_mvp

# Or simply open the project folder in your terminal.
```

### Step 2 — Create a virtual environment

```bash
python -m venv .venv

# Activate on Windows (CMD):
.venv\Scripts\activate.bat

# Activate on Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Activate on macOS / Linux:
source .venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Configure environment variables

```bash
# Copy the template
copy .env.template .env        # Windows CMD
# cp .env.template .env        # macOS / Linux

# Open .env in a text editor and fill in your OpenAI API key:
# OPENAI_API_KEY=sk-...
```

All other parameters (chunk size, top-K, similarity threshold, etc.) have sensible defaults defined in `.env.template`. Edit them if needed.

---

## 4. Document Ingestion

### Step 1 — Add your PDFs

Copy your PDF files into the `./data/` directory (create it if it doesn't exist):

```
rag_mvp/
└── data/
    ├── document1.pdf
    ├── document2.pdf
    └── ...
```

### Step 2 — Run ingestion (CLI)

```bash
python -m src.ingestion --docs-dir ./data
```

**Options:**

| Flag | Description |
|---|---|
| `--docs-dir PATH` | Directory containing PDFs (default: value of `DOCS_DIR` in `.env`) |
| `--force-reingest` | Delete and re-embed all existing documents |

**Or use the Streamlit UI button** — see Section 5.

---

## 5. Launching the Chat UI

```bash
streamlit run app.py
```

Your browser will open automatically at `http://localhost:8501`.

**First-time setup in the UI:**
1. In the left sidebar, click **"Ingest / Refresh Documents"** to parse and embed your PDFs.
2. Once ingestion completes, type your question in the chat input at the bottom of the page.
3. The system will retrieve relevant passages and generate a grounded answer with citations.

**Sidebar controls:**

| Control | Description |
|---|---|
| Indexed Documents | Count of unique PDFs currently in the vector store |
| Force re-ingest checkbox | Tick this before ingesting to overwrite existing embeddings |
| Ingest / Refresh Documents | Trigger ingestion from the UI |
| Clear History | Wipe the current session's conversation history |

---

## Project Structure

```
rag_mvp/
├── app.py                  # Streamlit chat interface
├── requirements.txt        # Pinned dependencies
├── .env.template           # Environment variable template
├── .env                    # Your local config (not committed)
├── README.md               # This file
├── data/                   # Place your PDF files here
├── chroma_db/              # ChromaDB persistence (auto-created)
└── src/
    ├── __init__.py
    ├── config.py           # Centralised config loader
    ├── ingestion.py        # PDF parsing, chunking, embedding, storage
    ├── retriever.py        # Semantic + optional hybrid retrieval
    └── generator.py        # Grounded answer generation + citations
```

---

## Troubleshooting

**`OPENAI_API_KEY` error on startup**
→ Make sure you copied `.env.template` to `.env` and added your actual API key.

**"No PDF files found"**
→ Check that your PDFs are in the `./data/` folder (or the path set in `DOCS_DIR`).

**Answers say "cannot find the answer in the provided documents"**
→ Try lowering `SIMILARITY_THRESHOLD` in `.env` (e.g., `0.20`) or check that the relevant PDFs were ingested.

**Slow ingestion**
→ OpenAI API rate limits may apply. Ingestion will proceed file-by-file and log progress.
