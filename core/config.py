"""
Central configuration for RAG_Agent.

Everything that can change between environments lives here as a constant
sourced from environment variables, so users can:

- Swap Ollama URL or model (`OLLAMA_URL`, `OLLAMA_MODEL`).
- Point at their own knowledge base (`MD_SOURCE_PATH`) without touching code.
- Reuse a pre-built FAISS index + chunks (`STORAGE_DIR` containing
  `faiss.index` and `docs.pkl`) — useful when shipping a packaged knowledge
  base instead of rebuilding from a markdown source.

Defaults are tuned so a fresh clone runs end-to-end on a tiny demo
knowledge file with no external services required.
"""
from __future__ import annotations

import os
from pathlib import Path

# --- LLM (Ollama / Gemma) -----------------------------------------------------
OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "gemma3:4b")
OLLAMA_TIMEOUT: int = int(os.getenv("OLLAMA_TIMEOUT", "90"))

# --- Embedding model ----------------------------------------------------------
# A small, fast model. Override with EMBEDDING_MODEL to point at a local path.
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

# --- Knowledge base sources ---------------------------------------------------
# A bundled tiny scikit-learn primer ships with the repo so retrieval works
# out of the box. Point MD_SOURCE_PATH at your `grounded_brain.md` to swap.
MD_SOURCE_PATH: str = os.getenv(
    "MD_SOURCE_PATH",
    "data/knowledge/sklearn_basics.md",
)

# Where the FAISS index, chunk pickle, and source-fingerprint live.
STORAGE_DIR: Path = Path(os.getenv("RAG_STORAGE_DIR", "storage"))
INDEX_PATH: Path = STORAGE_DIR / "faiss.index"
DOCS_PATH: Path = STORAGE_DIR / "docs.pkl"
HASH_PATH: Path = STORAGE_DIR / "index.hash"

# --- Chunking -----------------------------------------------------------------
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "64"))

# --- Behaviour ----------------------------------------------------------------
# When true, a failing/unreachable LLM returns a deterministic fallback
# instead of raising. Tests rely on this.
LLM_GRACEFUL_FALLBACK: bool = os.getenv("LLM_GRACEFUL_FALLBACK", "1") == "1"

STORAGE_DIR.mkdir(parents=True, exist_ok=True)
