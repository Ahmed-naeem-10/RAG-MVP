"""
config.py
---------
Centralised configuration loader for the RAG MVP system.

Uses Groq (free LLM) + local sentence-transformers (free embeddings).
Reads all parameters from environment variables (.env file).
Exits with non-zero status if GROQ_API_KEY is missing.
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Load .env file from the project root (one level above src/)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

logger = logging.getLogger(__name__)


def _get_int(key: str, default: int) -> int:
    """Read an integer environment variable with a fallback default."""
    raw = os.environ.get(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid value for %s='%s'; using default %d", key, raw, default)
        return default


def _get_float(key: str, default: float) -> float:
    """Read a float environment variable with a fallback default."""
    raw = os.environ.get(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid value for %s='%s'; using default %.2f", key, raw, default)
        return default


def _get_bool(key: str, default: bool) -> bool:
    """Read a boolean environment variable ('true'/'false')."""
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.strip().lower() == "true"


# ---------------------------------------------------------------------------
# Validate required credentials
# ---------------------------------------------------------------------------

GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")

if not GROQ_API_KEY:
    print(
        "[ERROR] The environment variable 'GROQ_API_KEY' is not set.\n"
        "  1. Go to https://console.groq.com and sign up for FREE\n"
        "  2. Create an API key\n"
        "  3. Add it to your .env file: GROQ_API_KEY=gsk_...",
        file=sys.stderr,
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

# Groq LLM — free, fast
LLM_MODEL: str = os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")

# Local embeddings — runs on your machine, no API key needed
EMBEDDING_MODEL: str = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ---------------------------------------------------------------------------
# Chunking parameters
# ---------------------------------------------------------------------------

CHUNK_SIZE: int = _get_int("CHUNK_SIZE", 1000)
CHUNK_OVERLAP: int = _get_int("CHUNK_OVERLAP", 200)

# Clamp to safe ranges
CHUNK_SIZE = max(500, min(4000, CHUNK_SIZE))
CHUNK_OVERLAP = max(50, min(500, CHUNK_OVERLAP))

# ---------------------------------------------------------------------------
# Retrieval parameters
# ---------------------------------------------------------------------------

TOP_K: int = _get_int("TOP_K", 5)
TOP_K = max(1, min(20, TOP_K))

SIMILARITY_THRESHOLD: float = _get_float("SIMILARITY_THRESHOLD", 0.30)
SIMILARITY_THRESHOLD = max(0.0, min(1.0, SIMILARITY_THRESHOLD))

# ---------------------------------------------------------------------------
# Storage paths
# ---------------------------------------------------------------------------

CHROMA_PERSIST_DIR: str = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")
DOCS_DIR: str = os.environ.get("DOCS_DIR", "./data")
BM25_INDEX_PATH: str = os.environ.get("BM25_INDEX_PATH", "./bm25_index.pkl")

# ---------------------------------------------------------------------------
# Feature flags
# ---------------------------------------------------------------------------

ENABLE_HYBRID_SEARCH: bool = _get_bool("ENABLE_HYBRID_SEARCH", False)
