"""
Knowledge base management.

Reads a markdown source, cleans + chunks it, embeds with the configured
sentence-transformer, and stores a FAISS index + chunk pickle + source
fingerprint under `STORAGE_DIR`.

If the user already has `storage/faiss.index` and `storage/docs.pkl` from
their own pipeline (e.g., the user's `grounded_brain.md` flow), this module
is a no-op — `ensure_knowledge_ready()` will skip the rebuild.
"""
from __future__ import annotations

import hashlib
import os
import pickle
from pathlib import Path
from typing import List

import numpy as np

from core.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCS_PATH,
    EMBEDDING_MODEL,
    HASH_PATH,
    INDEX_PATH,
    MD_SOURCE_PATH,
    STORAGE_DIR,
)
from core.exceptions import RetrievalError
from core.logger import get_logger
from extractor.chunker import generate_extraction_chunks
from extractor.sentence_rebuilder import rebuild_sentences
from extractor.text_cleaner import is_valid_chunk, normalize, remove_tables

logger = get_logger(__name__)


# -----------------------------------------------------------------------------
# Hashing
# -----------------------------------------------------------------------------
def compute_fingerprint(file_path: str | os.PathLike) -> str:
    """Stable fingerprint of source file using path + size + mtime."""
    try:
        stats = os.stat(file_path)
        combined = f"{file_path}_{stats.st_size}_{stats.st_mtime}"
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()
    except OSError as exc:
        logger.error("[KNOWLEDGE] Cannot fingerprint %s: %s", file_path, exc)
        return ""


def is_knowledge_synced() -> bool:
    """Return True if on-disk artifacts match the current source file."""
    if not (INDEX_PATH.exists() and DOCS_PATH.exists() and HASH_PATH.exists()):
        return False
    if not Path(MD_SOURCE_PATH).exists():
        # Source removed but artifacts present — treat as synced (user shipped
        # their own pre-built index without exposing the source markdown).
        return True
    return HASH_PATH.read_text().strip() == compute_fingerprint(MD_SOURCE_PATH)


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------
def load_chunks() -> List[str]:
    """Read the chunk pickle. Returns [] (and self-heals) on corruption."""
    if not DOCS_PATH.exists():
        return []
    if DOCS_PATH.stat().st_size == 0:
        logger.warning("[KNOWLEDGE] Empty docs.pkl — removing.")
        DOCS_PATH.unlink(missing_ok=True)
        return []
    try:
        with DOCS_PATH.open("rb") as fh:
            data = pickle.load(fh)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "chunks" in data:
            return list(data["chunks"])
        logger.warning("[KNOWLEDGE] Unexpected docs.pkl shape: %s", type(data))
        return []
    except (pickle.UnpicklingError, EOFError) as exc:
        logger.warning("[KNOWLEDGE] Corrupted docs.pkl (%s) — removing.", exc)
        DOCS_PATH.unlink(missing_ok=True)
        return []


# -----------------------------------------------------------------------------
# Build
# -----------------------------------------------------------------------------
def _split_blocks(raw_text: str) -> List[str]:
    """Split markdown by top-level headings to keep semantic locality."""
    blocks = raw_text.split("\n## ")
    return [b for b in blocks if b.strip()]


def build_knowledge_base() -> List[str]:
    """Read MD source -> clean -> chunk -> embed -> persist. Returns chunks."""
    src = Path(MD_SOURCE_PATH)
    if not src.exists():
        raise RetrievalError(
            f"Knowledge source not found: {MD_SOURCE_PATH}. Set MD_SOURCE_PATH "
            "to your grounded brain markdown file or drop a pre-built "
            "faiss.index + docs.pkl into the storage directory."
        )

    logger.info("[KNOWLEDGE] Building from %s", src)
    raw_text = src.read_text(encoding="utf-8")

    processed: List[str] = []
    for block in _split_blocks(raw_text):
        rebuilt = rebuild_sentences(block)
        cleaned = normalize(remove_tables(rebuilt))
        for chunk in generate_extraction_chunks(
            cleaned, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP
        ):
            if is_valid_chunk(chunk):
                processed.append(chunk)

    if not processed:
        raise RetrievalError("No valid chunks produced from knowledge source.")

    # Embedding — imported lazily so unit tests that stub retrieval don't pay
    # the cost of loading sentence-transformers.
    from sentence_transformers import SentenceTransformer  # noqa: WPS433
    import faiss  # noqa: WPS433

    logger.info("[KNOWLEDGE] Embedding %d chunks via %s",
                len(processed), EMBEDDING_MODEL)
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(
        [f"passage: {c}" for c in processed], show_progress_bar=False
    ).astype("float32")
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    with DOCS_PATH.open("wb") as fh:
        pickle.dump(processed, fh)
    HASH_PATH.write_text(compute_fingerprint(src))

    logger.info("[KNOWLEDGE] Built %d chunks, persisted to %s",
                len(processed), STORAGE_DIR)
    return processed


def ensure_knowledge_ready() -> None:
    """No-op if synced, otherwise rebuilds. Safe to call repeatedly."""
    if is_knowledge_synced() and DOCS_PATH.exists():
        logger.debug("[KNOWLEDGE] Index in sync — nothing to do.")
        return
    logger.info("[KNOWLEDGE] Index missing or stale — rebuilding.")
    build_knowledge_base()


if __name__ == "__main__":
    ensure_knowledge_ready()
    print(f"Loaded {len(load_chunks())} chunks from {STORAGE_DIR}")
