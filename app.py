"""
app.py
------
Streamlit chat interface for the RAG MVP system.

Features:
  - Natural-language question input (≤1000 characters).
  - Loading spinner while the system processes a request.
  - Answer display with expandable Citations section (source + page).
  - Conversation history (max 100 turns, persisted in Streamlit session state).
  - "Clear History" button with auto-dismissing confirmation.
  - Sidebar showing the count of indexed source documents.
  - User-friendly error messages (no stack traces exposed).
  - "Ingest / Refresh Documents" button to trigger ingestion from the UI.
"""

import sys
import time
import logging

import streamlit as st

# ---------------------------------------------------------------------------
# Page config must be the very first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RAG MVP — Document Q&A",
    page_icon="📄",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Imports (after page config)
# ---------------------------------------------------------------------------
from src import config  # noqa: E402  (config validates GROQ_API_KEY on import)
from src.ingestion import (  # noqa: E402
    get_chroma_client,
    get_or_create_collection,
    run_ingestion,
    get_ingested_sources,
)
from src.retriever import retrieve, ResultStatus  # noqa: E402
from src.generator import generate_answer  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_QUERY_CHARS = 1000
MAX_HISTORY_TURNS = 100  # Req 5.1 / 5.6

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "conversation_history" not in st.session_state:
    # List of {"role": "user"|"assistant", "content": "..."}
    st.session_state.conversation_history = []

if "show_cleared_msg" not in st.session_state:
    st.session_state.show_cleared_msg = False

if "cleared_at" not in st.session_state:
    st.session_state.cleared_at = 0.0


# ---------------------------------------------------------------------------
# Helper: get document count from ChromaDB
# ---------------------------------------------------------------------------

def get_indexed_doc_count() -> int:
    """
    Return the number of unique source documents currently indexed.

    Parameters
    ----------
    None

    Returns
    -------
    int
        Count of unique source PDF file names in the vector store.
        Returns 0 if the collection is empty or an error occurs.
    """
    try:
        client = get_chroma_client()
        collection = get_or_create_collection(client)
        sources = get_ingested_sources(collection)
        return len(sources)
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("📄 RAG MVP")
    st.markdown("Ask questions about your ingested PDF documents.")
    st.divider()

    # Document count (Req 6.5 / 6.6)
    doc_count = get_indexed_doc_count()
    st.metric(label="📚 Indexed Documents", value=doc_count)

    st.divider()

    # Ingest button
    st.subheader("🔄 Ingest Documents")
    st.caption(f"PDFs are loaded from: `{config.DOCS_DIR}`")

    force_reingest = st.checkbox("Force re-ingest (overwrite existing)", value=False)

    if st.button("Ingest / Refresh Documents", use_container_width=True, type="primary"):
        with st.spinner("Ingesting documents… this may take a moment."):
            try:
                run_ingestion(docs_dir=config.DOCS_DIR, force_reingest=force_reingest)
                st.success("✅ Ingestion complete! Refresh the page to update the document count.")
            except SystemExit:
                st.error(
                    "Ingestion failed: no PDF files found in the configured directory, "
                    f"or the path `{config.DOCS_DIR}` does not exist. "
                    "Check your DOCS_DIR setting in .env."
                )
            except Exception as exc:
                logger.error("Ingestion error: %s", exc)
                st.error(
                    "Ingestion encountered an error. Please check the terminal logs and try again."
                )

    st.divider()

    # Clear history button (Req 5.4)
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.conversation_history = []
        st.session_state.show_cleared_msg = True
        st.session_state.cleared_at = time.time()
        st.rerun()

    st.divider()
    st.caption("Model: `{}`\nEmbeddings: `{}` (local)".format(config.LLM_MODEL, config.EMBEDDING_MODEL))
    st.caption(
        "Similarity threshold: `{}`\nTop-K: `{}`".format(
            config.SIMILARITY_THRESHOLD, config.TOP_K
        )
    )
    if config.ENABLE_HYBRID_SEARCH:
        st.caption("🔀 Hybrid search: **enabled**")

# ---------------------------------------------------------------------------
# Main area — title
# ---------------------------------------------------------------------------

st.title("📄 Document Q&A — RAG MVP")
st.caption("Answers are grounded strictly in your ingested documents.")

# ---------------------------------------------------------------------------
# Auto-dismiss "history cleared" message (Req 5.5)
# ---------------------------------------------------------------------------

if st.session_state.show_cleared_msg:
    elapsed = time.time() - st.session_state.cleared_at
    if elapsed < 3.0:
        st.success("✅ Conversation history cleared.")
        # Trigger a rerun after the remaining display time
        time.sleep(max(0, 3.0 - elapsed))
        st.session_state.show_cleared_msg = False
        st.rerun()
    else:
        st.session_state.show_cleared_msg = False

# ---------------------------------------------------------------------------
# Render conversation history (Req 5.3)
# ---------------------------------------------------------------------------

def render_history() -> None:
    """
    Render the full conversation history in the chat area.

    User messages are displayed with a "You" label (right-side avatar).
    System answers are displayed with a "System" label (left-side avatar).

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    for turn in st.session_state.conversation_history:
        if turn["role"] == "user":
            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(f"**You:** {turn['content']}")
        else:
            with st.chat_message("assistant", avatar="🤖"):
                # Answer text may include a trailing citations JSON; we store
                # answer and citations separately in history for clean rendering.
                st.markdown(f"**System:** {turn['content']}")
                if turn.get("citations"):
                    with st.expander("📎 Citations", expanded=False):
                        for cit in turn["citations"]:
                            st.markdown(f"- **{cit['source']}** — Page {cit['page']}")


render_history()

# ---------------------------------------------------------------------------
# Query input (Req 6.1)
# ---------------------------------------------------------------------------

query = st.chat_input(
    placeholder="Ask a question about your documents… (max 1000 characters)",
)

if query:
    # Validate length (Req 6.1)
    if len(query) > MAX_QUERY_CHARS:
        st.error(
            f"Your question exceeds the {MAX_QUERY_CHARS}-character limit "
            f"({len(query)} characters). Please shorten it and try again."
        )
        st.stop()

    # Display user message immediately
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(f"**You:** {query}")

    # Build conversation history for the LLM (only role/content pairs, no citations)
    llm_history = [
        {"role": turn["role"], "content": turn["content"]}
        for turn in st.session_state.conversation_history
    ]

    # Process with loading indicator (Req 6.2)
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Searching documents…"):
            # Retrieve
            retrieval_result = retrieve(query)

            # Generate
            gen_result = generate_answer(
                query=query,
                retrieval_result=retrieval_result,
                conversation_history=llm_history if llm_history else None,
            )

        # Display answer (Req 6.3 / 6.4)
        if gen_result.is_error:
            # User-safe error — no stack trace, includes retry suggestion (Req 6.4)
            st.error(
                f"⚠️ {gen_result.answer} "
                "Please try again or contact support if the issue persists."
            )
        else:
            st.markdown(f"**System:** {gen_result.answer}")

            # Show citations only when present (Req 6.3)
            if gen_result.citations:
                with st.expander("📎 Citations", expanded=False):
                    for cit in gen_result.citations:
                        st.markdown(f"- **{cit.source}** — Page {cit.page}")

    # Append turn to history AFTER answer is fully received (Req 5.2)
    if not gen_result.is_error:
        # Evict oldest turn if at capacity (Req 5.6)
        if len(st.session_state.conversation_history) >= MAX_HISTORY_TURNS * 2:
            # Each turn = 2 entries (user + assistant); evict the oldest pair
            st.session_state.conversation_history = (
                st.session_state.conversation_history[2:]
            )

        # Store user message
        st.session_state.conversation_history.append(
            {"role": "user", "content": query}
        )
        # Store assistant message with citations for re-rendering
        st.session_state.conversation_history.append(
            {
                "role": "assistant",
                "content": gen_result.answer,
                "citations": [
                    {"source": c.source, "page": c.page}
                    for c in gen_result.citations
                ],
            }
        )
