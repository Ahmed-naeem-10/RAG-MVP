"""
ingestion.py
------------
PDF ingestion pipeline for the RAG MVP system.

Uses local sentence-transformers for embeddings (FREE, no API key needed).
ChromaDB stores everything locally on disk.

CLI Usage:
  python -m src.ingestion --docs-dir ./data
  python -m src.ingestion --docs-dir ./data --force-reingest
"""

import argparse
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import List, Tuple

import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from src import config

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ChromaDB collection name (shared between ingestion + retriever)
COLLECTION_NAME = "rag_mvp_documents"


# ---------------------------------------------------------------------------
# Embedding model (local, free — sentence-transformers)
# ---------------------------------------------------------------------------

_embedding_function = None


def get_embedding_function():
    """
    Return a ChromaDB-compatible embedding function using sklearn HashingVectorizer.
    No downloads, no API key, no torch required.

    Parameters
    ----------
    None

    Returns
    -------
    callable
        ChromaDB-compatible embedding function.
    """
    global _embedding_function
    if _embedding_function is None:
        from sklearn.feature_extraction.text import HashingVectorizer

        class LocalHashEmbedding:
            """Fast local embeddings — no downloads, no API key needed."""

            # ChromaDB 1.x checks for this attribute
            def __init__(self):
                self._vec = HashingVectorizer(
                    n_features=384,
                    norm="l2",
                    alternate_sign=False,
                    ngram_range=(1, 2),
                )
                logger.info("Local hash embeddings ready (no downloads needed).")

            def __call__(self, input):
                matrix = self._vec.transform(input).toarray()
                return [row.tolist() for row in matrix]

        _embedding_function = LocalHashEmbedding()
    return _embedding_function


# ---------------------------------------------------------------------------
# ChromaDB helpers
# ---------------------------------------------------------------------------


def get_chroma_client() -> chromadb.ClientAPI:
    """
    Create and return a persistent ChromaDB client.

    Parameters
    ----------
    None

    Returns
    -------
    chromadb.ClientAPI
        ChromaDB client backed by the configured persistence directory.
    """
    return chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)


def get_or_create_collection(client: chromadb.ClientAPI) -> chromadb.Collection:
    """
    Get the existing ChromaDB collection or create it.
    Does NOT pass embedding_function to avoid version conflicts.

    Parameters
    ----------
    client : chromadb.ClientAPI

    Returns
    -------
    chromadb.Collection
    """
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def get_ingested_sources(collection: chromadb.Collection) -> set:
    """
    Return the set of source file names already stored in the collection.

    Parameters
    ----------
    collection : chromadb.Collection
        The ChromaDB collection to query.

    Returns
    -------
    set
        Set of source file name strings currently in the collection.
    """
    results = collection.get(include=["metadatas"])
    sources = set()
    for meta in results.get("metadatas", []):
        if meta and "source" in meta:
            sources.add(meta["source"])
    return sources


def delete_source_from_collection(collection: chromadb.Collection, source_name: str) -> None:
    """
    Delete all chunks belonging to a specific source file from the collection.

    Parameters
    ----------
    collection : chromadb.Collection
        The ChromaDB collection to modify.
    source_name : str
        The source file name whose chunks should be deleted.

    Returns
    -------
    None
    """
    results = collection.get(where={"source": source_name}, include=["metadatas"])
    ids_to_delete = results.get("ids", [])
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
        logger.info("Deleted %d existing chunks for '%s'.", len(ids_to_delete), source_name)


# ---------------------------------------------------------------------------
# PDF processing helpers
# ---------------------------------------------------------------------------


def extract_pages(pdf_path: Path) -> List[Tuple[str, int]]:
    """
    Extract text from every page of a PDF file.

    Skips image-only pages (no extractable text) and logs a warning for each.

    Parameters
    ----------
    pdf_path : Path
        Path to the PDF file.

    Returns
    -------
    List[Tuple[str, int]]
        List of (page_text, page_number) tuples. Page numbers are 1-indexed.

    Raises
    ------
    PdfReadError
        If the PDF cannot be opened or parsed.
    """
    reader = PdfReader(str(pdf_path))
    pages = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            logger.warning(
                "No extractable text on page %d of '%s' — skipping (may be image-only).",
                page_num, pdf_path.name,
            )
            continue
        pages.append((text, page_num))
    return pages


def chunk_pages(pages: List[Tuple[str, int]], source_name: str) -> List[dict]:
    """
    Split extracted page text into overlapping chunks with metadata.

    Parameters
    ----------
    pages : List[Tuple[str, int]]
        List of (page_text, page_number) tuples.
    source_name : str
        PDF file name used for the 'source' metadata field.

    Returns
    -------
    List[dict]
        List of dicts with keys 'text', 'source', and 'page'.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len,
    )
    all_chunks = []
    for page_text, page_num in pages:
        splits = splitter.split_text(page_text)
        for split in splits:
            all_chunks.append({
                "text": split,
                "source": source_name,
                "page": page_num,
            })
    return all_chunks


def embed_and_store(
    chunks: List[dict],
    collection: chromadb.Collection,
    source_name: str,
) -> None:
    """
    Embed chunks using local HashingVectorizer and store in ChromaDB.

    Parameters
    ----------
    chunks : List[dict]
    collection : chromadb.Collection
    source_name : str

    Returns
    -------
    None
    """
    texts = [c["text"] for c in chunks]
    ids = [f"{source_name}__p{c['page']}__c{i}" for i, c in enumerate(chunks)]
    metadatas = [{"source": c["source"], "page": c["page"]} for c in chunks]

    try:
        ef = get_embedding_function()
        embeddings = ef(texts)
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        logger.info("Stored %d chunks for '%s'.", len(chunks), source_name)
    except Exception as exc:
        logger.error("Failed to store chunks for '%s': %s — rolling back.", source_name, exc)
        delete_source_from_collection(collection, source_name)
        raise


# ---------------------------------------------------------------------------
# BM25 index (optional hybrid search)
# ---------------------------------------------------------------------------


def build_bm25_index(collection: chromadb.Collection) -> None:
    """
    Build a BM25 keyword index from all chunks and persist it to disk.

    Only runs when ENABLE_HYBRID_SEARCH is True.

    Parameters
    ----------
    collection : chromadb.Collection
        Source of documents for the BM25 index.

    Returns
    -------
    None
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        logger.error("rank-bm25 is not installed. Run: pip install rank-bm25")
        return

    results = collection.get(include=["documents", "metadatas"])
    documents = results.get("documents", [])
    metadatas = results.get("metadatas", [])
    ids = results.get("ids", [])

    if not documents:
        logger.warning("No documents found — BM25 index will be empty.")

    tokenized = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized)

    payload = {"bm25": bm25, "documents": documents, "metadatas": metadatas, "ids": ids}

    try:
        with open(config.BM25_INDEX_PATH, "wb") as f:
            pickle.dump(payload, f)
        logger.info("BM25 index saved to '%s'.", config.BM25_INDEX_PATH)
    except Exception as exc:
        logger.error(
            "Failed to persist BM25 index: %s — continuing without it.", exc
        )


# ---------------------------------------------------------------------------
# Main ingestion function
# ---------------------------------------------------------------------------


def run_ingestion(docs_dir: str, force_reingest: bool = False) -> None:
    """
    Execute the full ingestion pipeline for all PDFs in docs_dir.

    Parameters
    ----------
    docs_dir : str
        Path to directory containing PDF files.
    force_reingest : bool
        If True, delete existing embeddings and re-embed everything.

    Returns
    -------
    None
    """
    docs_path = Path(docs_dir)

    if not docs_path.exists() or not docs_path.is_dir():
        print(f"[ERROR] Directory not found: '{docs_dir}'", file=sys.stderr)
        sys.exit(1)

    pdf_files = sorted(docs_path.glob("*.pdf"))

    if not pdf_files:
        print(f"[ERROR] No PDF files found in '{docs_dir}'.", file=sys.stderr)
        sys.exit(1)

    logger.info("Found %d PDF file(s) in '%s'.", len(pdf_files), docs_dir)

    client = get_chroma_client()
    collection = get_or_create_collection(client)
    ingested_sources = get_ingested_sources(collection)

    for pdf_path in pdf_files:
        source_name = pdf_path.name

        if source_name in ingested_sources:
            if force_reingest:
                logger.info("Force re-ingesting '%s'...", source_name)
                delete_source_from_collection(collection, source_name)
            else:
                logger.info("Skipping '%s' — already ingested. Use --force-reingest to override.", source_name)
                continue

        logger.info("Processing '%s'...", source_name)

        try:
            pages = extract_pages(pdf_path)
        except PdfReadError as exc:
            logger.error("Cannot parse '%s': %s — skipping.", source_name, exc)
            continue
        except Exception as exc:
            logger.error("Unexpected error reading '%s': %s — skipping.", source_name, exc)
            continue

        if not pages:
            logger.warning("'%s' yielded no extractable text — skipping.", source_name)
            continue

        chunks = chunk_pages(pages, source_name)
        logger.info("'%s' → %d chunks from %d pages.", source_name, len(chunks), len(pages))

        try:
            embed_and_store(chunks, collection, source_name)
        except Exception:
            continue

    if config.ENABLE_HYBRID_SEARCH:
        logger.info("Building BM25 index...")
        build_bm25_index(collection)

    logger.info("Ingestion complete. Total chunks in store: %d", collection.count())


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI arguments and run the ingestion pipeline."""
    parser = argparse.ArgumentParser(
        description="Ingest PDF documents into the RAG MVP vector store."
    )
    parser.add_argument(
        "--docs-dir",
        default=config.DOCS_DIR,
        help="Path to the directory containing PDF files.",
    )
    parser.add_argument(
        "--force-reingest",
        action="store_true",
        help="Delete existing embeddings and re-ingest all files.",
    )
    args = parser.parse_args()
    run_ingestion(docs_dir=args.docs_dir, force_reingest=args.force_reingest)


if __name__ == "__main__":
    main()
