"""
Integration test for knowledge build / load (no embeddings).

This test runs the *cleaning + chunking + persistence* path of `rag.knowledge`
using a tiny in-memory markdown source, but stubs the embedding model so the
test stays fast (~1s) and runs offline.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pytest


@pytest.fixture
def tmp_kb(tmp_path, monkeypatch):
    """Point the config at a throwaway storage dir + tiny markdown source."""
    src = tmp_path / "tiny.md"
    src.write_text(
        "# Demo\n\n"
        "## A\n\n"
        "Use SimpleImputer with strategy median for numeric data with outliers. "
        "It is robust and handles skew well in scikit-learn.\n\n"
        "## B\n\n"
        "Apply OneHotEncoder for nominal categorical features. "
        "Set handle_unknown='ignore' so unseen labels do not raise.\n",
        encoding="utf-8",
    )

    storage = tmp_path / "storage"
    storage.mkdir()

    # Reload config + knowledge module against the new env so module-level
    # paths pick up the override.
    monkeypatch.setenv("MD_SOURCE_PATH", str(src))
    monkeypatch.setenv("RAG_STORAGE_DIR", str(storage))
    for mod in ("core.config", "rag.knowledge"):
        sys.modules.pop(mod, None)
    return tmp_path


def test_build_knowledge_base_produces_chunks_and_index(tmp_kb, monkeypatch):
    # Stub heavy embedding deps before knowledge.py imports them.
    class FakeST:
        def __init__(self, *a, **kw): pass
        def encode(self, items, show_progress_bar=False):
            import numpy as np
            return np.ones((len(items), 4), dtype="float32")

    class FakeFaiss:
        @staticmethod
        def normalize_L2(_x): pass

        class IndexFlatIP:
            def __init__(self, dim):
                self.dim = dim
                self.ntotal = 0
            def add(self, vec): self.ntotal += vec.shape[0]

        @staticmethod
        def write_index(_idx, path):
            Path(path).write_bytes(b"FAKE")

    monkeypatch.setitem(sys.modules, "sentence_transformers",
                        type(sys)("sentence_transformers"))
    sys.modules["sentence_transformers"].SentenceTransformer = FakeST
    monkeypatch.setitem(sys.modules, "faiss", FakeFaiss)

    from rag.knowledge import (DOCS_PATH, INDEX_PATH, build_knowledge_base,
                               is_knowledge_synced, load_chunks)

    chunks = build_knowledge_base()
    assert len(chunks) >= 2
    joined = " ".join(chunks).lower()
    # rebuild_sentences may split CamelCase, so we look at the lowercased prose
    assert "imputer" in joined
    assert "encoder" in joined

    # Persistence round-trip
    assert INDEX_PATH.exists() and DOCS_PATH.exists()
    with DOCS_PATH.open("rb") as fh:
        on_disk = pickle.load(fh)
    assert on_disk == chunks

    # Sync detection
    assert is_knowledge_synced() is True
    assert load_chunks() == chunks
