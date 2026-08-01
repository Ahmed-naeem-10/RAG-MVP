"""
retriever.py
------------
Semantic retrieval module for the RAG MVP system.

Uses local ChromaDB with sentence-transformers embeddings (completely free).
Optionally merges with BM25 keyword search via Reciprocal Rank Fusion.
"""

import logging
import pickle
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional

import chromadb

from src import config
from src.ingestion import (
    COLLECTION_NAME,
    get_chroma_client,
    get_or_create_collection,
    get_embedding_function,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


class ResultStatus(Enum):
    """Outcome of a retrieval operation."""
    SUCCESS = auto()     # ≥1 chunk passed the threshold
    NO_RESULTS = auto()  # query OK but no chunk met the threshold
    ERROR = auto()       # something failed


@dataclass
class RetrievedChunk:
    """A single retrieved document chunk with metadata and similarity score."""
    text: str
    source: str
    page: int
    score: float


@dataclass
class RetrievalResult:
    """
    The outcome of a retrieval operation.

    Attributes
    ----------
    status : ResultStatus
        SUCCESS, NO_RESULTS, or ERROR.
    chunks : List[RetrievedChunk]
        Non-empty only when status == SUCCESS.
    error_message : Optional[str]
        Non-None only when status == ERROR.
    """
    status: ResultStatus
    chunks: List[RetrievedChunk] = field(default_factory=list)
    error_message: Optional[str] = None


# ---------------------------------------------------------------------------
# BM25 helpers (optional hybrid search)
# ---------------------------------------------------------------------------


def _load_bm25_payload() -> Optional[dict]:
    """
    Load the persisted BM25 index from disk.

    Returns None with a warning if the file does not exist.

    Parameters
    ----------
    None

    Returns
    -------
    Optional[dict]
        BM25 payload dict or None if not found.
    """
    import os
    if not os.path.exists(config.BM25_INDEX_PATH):
        logger.warning(
            "BM25 index not found at '%s'. Falling back to dense-only retrieval.",
            config.BM25_INDEX_PATH,
        )
        return None
    try:
        with open(config.BM25_INDEX_PATH, "rb") as f:
            return pickle.load(f)
    except Exception as exc:
        logger.warning("Failed to load BM25 index: %s. Falling back to dense-only.", exc)
        return None


def _reciprocal_rank_fusion(
    dense_chunks: List[RetrievedChunk],
    bm25_docs: List[str],
    bm25_metadatas: List[dict],
    k_rrf: int = 60,
) -> List[RetrievedChunk]:
    """
    Merge dense vector results and BM25 results using Reciprocal Rank Fusion.

    RRF score = sum(1 / (k + rank)) across result lists. Uses k=60.

    Parameters
    ----------
    dense_chunks : List[RetrievedChunk]
        Ranked results from dense vector search.
    bm25_docs : List[str]
        Document texts ranked by BM25.
    bm25_metadatas : List[dict]
        Metadata for each BM25-ranked document.
    k_rrf : int
        RRF constant (default 60).

    Returns
    -------
    List[RetrievedChunk]
        Merged and re-ranked list of chunks.
    """
    scores: dict = {}

    for rank, chunk in enumerate(dense_chunks, start=1):
        key = f"{chunk.source}_{chunk.page}_{chunk.text[:30]}"
        scores.setdefault(key, {"chunk": chunk, "rrf": 0.0})
        scores[key]["rrf"] += 1.0 / (k_rrf + rank)

    for rank, (text, meta) in enumerate(zip(bm25_docs, bm25_metadatas), start=1):
        key = f"{meta.get('source', '')}_{meta.get('page', 0)}_{text[:30]}"
        if key not in scores:
            chunk = RetrievedChunk(
                text=text,
                source=meta.get("source", ""),
                page=meta.get("page", 0),
                score=0.0,
            )
            scores[key] = {"chunk": chunk, "rrf": 0.0}
        scores[key]["rrf"] += 1.0 / (k_rrf + rank)

    sorted_items = sorted(scores.values(), key=lambda x: x["rrf"], reverse=True)
    return [item["chunk"] for item in sorted_items]


# ---------------------------------------------------------------------------
# Public retrieval function
# ---------------------------------------------------------------------------


def retrieve(query: str) -> RetrievalResult:
    """
    Retrieve the most relevant document chunks for a given query.

    ChromaDB handles embedding using the local sentence-transformers model.
    Cosine distance [0, 2] is converted to similarity [0.0, 1.0].

    Parameters
    ----------
    query : str
        The natural-language question from the user.

    Returns
    -------
    RetrievalResult
        status=SUCCESS with chunks, status=NO_RESULTS, or status=ERROR.
    """
    if not query or not query.strip():
        return RetrievalResult(
            status=ResultStatus.ERROR,
            error_message="Query cannot be empty.",
        )

    try:
        client = get_chroma_client()
        collection = get_or_create_collection(client)

        total_docs = collection.count()
        if total_docs == 0:
            return RetrievalResult(status=ResultStatus.NO_RESULTS)

        n_results = min(config.TOP_K, total_docs)

        # Embed query using same local function
        ef = get_embedding_function()
        query_embedding = ef([query])[0]

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        logger.error("ChromaDB query failed: %s", exc)
        return RetrievalResult(
            status=ResultStatus.ERROR,
            error_message=f"Vector store query failed: {exc}",
        )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    # Convert cosine distance [0, 2] → similarity [0.0, 1.0]
    dense_chunks: List[RetrievedChunk] = []
    for text, meta, dist in zip(docs, metas, distances):
        similarity = round(1.0 - (dist / 2.0), 4)
        if similarity >= config.SIMILARITY_THRESHOLD:
            dense_chunks.append(
                RetrievedChunk(
                    text=text,
                    source=meta.get("source", ""),
                    page=meta.get("page", 0),
                    score=similarity,
                )
            )

    # Optional hybrid search (BM25 + RRF)
    if config.ENABLE_HYBRID_SEARCH:
        payload = _load_bm25_payload()
        if payload is not None:
            bm25 = payload["bm25"]
            bm25_documents = payload["documents"]
            bm25_metadatas = payload["metadatas"]

            tokenized_query = query.lower().split()
            bm25_scores = bm25.get_scores(tokenized_query)

            ranked_indices = sorted(
                range(len(bm25_scores)),
                key=lambda i: bm25_scores[i],
                reverse=True,
            )[: config.TOP_K]

            bm25_ranked_docs = [bm25_documents[i] for i in ranked_indices]
            bm25_ranked_metas = [bm25_metadatas[i] for i in ranked_indices]

            merged = _reciprocal_rank_fusion(dense_chunks, bm25_ranked_docs, bm25_ranked_metas)
            final_chunks = merged[: config.TOP_K]
        else:
            final_chunks = dense_chunks
    else:
        final_chunks = dense_chunks

    if not final_chunks:
        return RetrievalResult(status=ResultStatus.NO_RESULTS)

    return RetrievalResult(status=ResultStatus.SUCCESS, chunks=final_chunks)
