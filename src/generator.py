"""
generator.py
------------
Answer generation module for the RAG MVP system.

Uses Groq API (FREE) with LLaMA 3.1 70B for grounded answer generation.
Temperature is set to 0.0 for deterministic, factual output.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from groq import Groq, GroqError

from src import config
from src.retriever import RetrievalResult, ResultStatus, RetrievedChunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fixed messages
# ---------------------------------------------------------------------------

NO_RESULTS_MESSAGE = (
    "I am sorry, but I cannot find the answer in the provided documents. "
    "Please try rephrasing your question or ensure the relevant documents have been ingested."
)

SYSTEM_PROMPT = (
    "You are a precise, factual assistant. "
    "Answer the user's question strictly and only using the context passages provided. "
    "Do not use any knowledge, assumptions, or information outside the provided context. "
    "If the answer is not in the context, say: "
    "'I am sorry, but I cannot find the answer in the provided documents.'"
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class Citation:
    """A deduplicated source reference attached to a generated answer."""
    source: str
    page: int


@dataclass
class GenerationResult:
    """
    The outcome of an answer generation operation.

    Attributes
    ----------
    answer : str
        The generated answer text.
    citations : List[Citation]
        Deduplicated source citations. Empty for no-results/error cases.
    is_error : bool
        True when generation itself failed.
    error_message : Optional[str]
        Human-readable detail when is_error is True.
    """
    answer: str
    citations: List[Citation] = field(default_factory=list)
    is_error: bool = False
    error_message: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_context_block(chunks: List[RetrievedChunk]) -> str:
    """
    Format retrieved chunks into a numbered context block for the prompt.

    Parameters
    ----------
    chunks : List[RetrievedChunk]
        Ordered list of retrieved document chunks.

    Returns
    -------
    str
        Formatted context string.
    """
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        lines.append(f"[{i}] Source: {chunk.source}, Page: {chunk.page}\n{chunk.text}")
    return "\n\n---\n\n".join(lines)


def _deduplicate_citations(chunks: List[RetrievedChunk]) -> List[Citation]:
    """
    Build a deduplicated list of citations from retrieved chunks.

    Parameters
    ----------
    chunks : List[RetrievedChunk]
        Retrieved chunks whose metadata forms the citation list.

    Returns
    -------
    List[Citation]
        Ordered, deduplicated Citation objects.
    """
    seen: set = set()
    citations: List[Citation] = []
    for chunk in chunks:
        key: Tuple[str, int] = (chunk.source, chunk.page)
        if key not in seen:
            seen.add(key)
            citations.append(Citation(source=chunk.source, page=chunk.page))
    return citations


# ---------------------------------------------------------------------------
# Public generation function
# ---------------------------------------------------------------------------


def generate_answer(
    query: str,
    retrieval_result: RetrievalResult,
    conversation_history: Optional[List[dict]] = None,
) -> GenerationResult:
    """
    Generate a grounded answer using Groq (free LLaMA 3.1).

    Behaviour:
    - NO_RESULTS → return fixed "not found" message, no API call.
    - ERROR from retriever → return user-safe error message.
    - SUCCESS → call Groq with grounded context at temperature=0.0.

    Parameters
    ----------
    query : str
        The user's question.
    retrieval_result : RetrievalResult
        Output from retriever.retrieve().
    conversation_history : Optional[List[dict]]
        Prior turns as [{"role": "user"|"assistant", "content": "..."}].

    Returns
    -------
    GenerationResult
        Answer text, citations, and error info.
    """
    # Case 1: Retrieval error
    if retrieval_result.status == ResultStatus.ERROR:
        return GenerationResult(
            answer="An error occurred while searching the documents. Please try again.",
            is_error=True,
            error_message=retrieval_result.error_message,
        )

    # Case 2: No relevant chunks found
    if retrieval_result.status == ResultStatus.NO_RESULTS:
        return GenerationResult(answer=NO_RESULTS_MESSAGE, citations=[])

    # Case 3: Generate grounded answer with Groq
    chunks = retrieval_result.chunks
    context_block = _build_context_block(chunks)

    user_message = (
        f"Context passages:\n\n{context_block}\n\n"
        f"Question: {query}\n\n"
        "Answer based strictly on the context passages above:"
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if conversation_history:
        messages.extend(conversation_history)

    messages.append({"role": "user", "content": user_message})

    try:
        client = Groq(api_key=config.GROQ_API_KEY)
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=messages,
            temperature=0.0,
        )
        answer_text = response.choices[0].message.content or ""
    except GroqError as exc:
        logger.error("Groq API call failed: %s", exc)
        return GenerationResult(
            answer="The answer could not be generated. Please try again.",
            is_error=True,
            error_message=str(exc),
        )
    except Exception as exc:
        logger.error("Unexpected error during generation: %s", exc)
        return GenerationResult(
            answer="An unexpected error occurred. Please try again.",
            is_error=True,
            error_message=str(exc),
        )

    citations = _deduplicate_citations(chunks)
    return GenerationResult(answer=answer_text, citations=citations)
