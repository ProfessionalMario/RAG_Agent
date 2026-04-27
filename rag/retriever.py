"""
FAISS retriever.

Loads a pre-built index from `STORAGE_DIR` (built by `rag.knowledge`) and
runs cosine-similarity search using a BGE sentence-transformer with the
required `query: ` / `passage: ` instruction prefixes.
"""
from __future__ import annotations

import pickle
from typing import List, Optional

from core.config import DOCS_PATH, EMBEDDING_MODEL, INDEX_PATH, STORAGE_DIR
from core.logger import get_logger

logger = get_logger(__name__)


class FaissRetriever:
    """Singleton-friendly cosine-similarity retriever."""

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        documents: Optional[List[str]] = None,
        score_threshold: float = 0.4,
    ) -> None:
        # Lazy heavy imports: the test-suite stubs the retriever and never
        # touches FAISS or sentence-transformers.
        import faiss  # noqa: WPS433
        from sentence_transformers import SentenceTransformer  # noqa: WPS433

        self._faiss = faiss
        self.score_threshold = score_threshold
        logger.info("[RETRIEVER] Loading embedder %s", model_name)
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents: List[str] = []

        if INDEX_PATH.exists() and DOCS_PATH.exists():
            self._load()
        elif documents:
            self._build(documents)
        else:
            logger.warning(
                "[RETRIEVER] No FAISS index on disk and no documents passed; "
                "call build_knowledge_base() or pass documents.",
            )

    # ------------------------------------------------------------------ build
    def _build(self, documents: List[str]) -> None:
        processed = list(
            dict.fromkeys(
                str(d).strip() for d in documents if len(str(d).split()) >= 3
            )
        )
        if not processed:
            raise ValueError("No valid documents to index.")

        passages = [f"passage: {t}" for t in processed]
        emb = self.model.encode(passages, show_progress_bar=False).astype("float32")
        self._faiss.normalize_L2(emb)

        self.index = self._faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb)
        self.documents = processed

        STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self.index, str(INDEX_PATH))
        with DOCS_PATH.open("wb") as fh:
            pickle.dump(self.documents, fh)
        logger.info("[RETRIEVER] Built fresh index: %d vectors", self.index.ntotal)

    def _load(self) -> None:
        self.index = self._faiss.read_index(str(INDEX_PATH))
        with DOCS_PATH.open("rb") as fh:
            self.documents = pickle.load(fh)
        logger.info("[RETRIEVER] Loaded %d docs from %s",
                    len(self.documents), STORAGE_DIR)

    # ----------------------------------------------------------------- search
    def retrieve(self, query: str, k: int = 3) -> List[str]:
        if not query or self.index is None:
            return []
        try:
            q_emb = self.model.encode([f"query: {query}"]).astype("float32")
            self._faiss.normalize_L2(q_emb)
            scores, indices = self.index.search(q_emb, k)
            return [
                self.documents[idx]
                for score, idx in zip(scores[0], indices[0])
                if 0 <= idx < len(self.documents) and score > self.score_threshold
            ]
        except Exception as exc:  # noqa: BLE001
            logger.error("[RETRIEVER] Search failed: %s", exc)
            return []


# -------------------------------------------------------------------- singleton
_instance: Optional[FaissRetriever] = None


def get_retriever(docs: Optional[List[str]] = None) -> FaissRetriever:
    """Return a process-wide retriever, creating one on first call."""
    global _instance  # noqa: PLW0603
    if _instance is None:
        _instance = FaissRetriever(EMBEDDING_MODEL, docs)
    return _instance


def reset_retriever() -> None:
    """Test helper — wipe the singleton so a fresh instance is built next call."""
    global _instance  # noqa: PLW0603
    _instance = None
