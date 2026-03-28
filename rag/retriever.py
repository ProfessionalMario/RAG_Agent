"""
FAISS Retriever 
"""

import os
import faiss
import pickle
import numpy as np
from typing import List

from sentence_transformers import SentenceTransformer
from core.logger import get_logger

logger = get_logger(__name__)

# -----------------------------
# CONFIG
# -----------------------------
FAISS_DIR = "storage/faiss"
INDEX_PATH = os.path.join(FAISS_DIR, "index.faiss")
DOCS_PATH = os.path.join(FAISS_DIR, "docs.pkl")

# -----------------------------
# MODEL CACHE
# -----------------------------
_model_cache = {}


def get_model(model_path: str):
    try:
        if model_path in _model_cache:
            return _model_cache[model_path]

        logger.info("[Retriever] Loading embedding model: %s", model_path)

        if model_path.startswith("./"):
            model = SentenceTransformer(model_path)
        else:
            model = SentenceTransformer("all-MiniLM-L6-v2")

        _model_cache[model_path] = model
        return model

    except Exception as e:
        logger.exception("[Retriever] Model load failed, using fallback")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        _model_cache[model_path] = model
        return model


# -----------------------------
# TEXT PIPELINE
# -----------------------------
def normalize_doc(doc):
    try:
        if isinstance(doc, str):
            return doc
        elif isinstance(doc, dict):
            return doc.get("text", "")
        elif hasattr(doc, "page_content"):
            return doc.page_content
        else:
            return str(doc)
    except Exception:
        logger.exception("[Retriever] normalize_doc failed")
        return ""


def clean_text(text: str) -> str:
    try:
        if not text:
            return ""

        text = text.encode("ascii", "ignore").decode()
        text = text.strip()
        text = " ".join(text.split())

        return text

    except Exception:
        logger.exception("[Retriever] clean_text failed")
        return ""


def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks


def is_valid(text):
    return len(text.split()) >= 5


# -----------------------------
# RETRIEVER
# -----------------------------
class FaissRetriever:
    def __init__(self, model_path: str, documents: List[str]):
        logger.info("[Retriever] Initializing FAISS retriever")

        self.model = get_model(model_path)

        # Try loading existing index
        if os.path.exists(INDEX_PATH) and os.path.exists(DOCS_PATH):
            self._load()
        else:
            self._build(documents)

    # -----------------------------
    # BUILD INDEX
    # -----------------------------
    def _build(self, documents: List[str]):
        logger.info("[Retriever] Building FAISS index")

        processed_docs = []

        for idx, raw_doc in enumerate(documents):
            try:
                doc = normalize_doc(raw_doc)
                doc = clean_text(doc)

                if not doc:
                    continue

                chunks = chunk_text(doc)

                for chunk in chunks:
                    if is_valid(chunk):
                        processed_docs.append(chunk)

            except Exception:
                logger.exception("[Retriever] Error processing doc %d", idx)

        if not processed_docs:
            raise ValueError("No valid documents after processing")

        processed_docs = list(set(processed_docs))
        logger.info("[Retriever] Final usable chunks: %d", len(processed_docs))

        embeddings = self.model.encode(
            processed_docs,
            show_progress_bar=True
        ).astype("float32")

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

        self.documents = processed_docs

        self._save()

        logger.info("[Retriever] FAISS index built and saved")

    # -----------------------------
    # SAVE / LOAD
    # -----------------------------
    def _save(self):
        os.makedirs(FAISS_DIR, exist_ok=True)

        faiss.write_index(self.index, INDEX_PATH)

        with open(DOCS_PATH, "wb") as f:
            pickle.dump(self.documents, f)

    def _load(self):
        logger.info("[Retriever] Loading FAISS index from disk")

        self.index = faiss.read_index(INDEX_PATH)

        with open(DOCS_PATH, "rb") as f:
            self.documents = pickle.load(f)

    # -----------------------------
    # RETRIEVE
    # -----------------------------
    def retrieve(self, query: str, k: int = 3) -> List[str]:
        try:
            if not query or not isinstance(query, str):
                raise ValueError("Invalid query")

            query = clean_text(query)

            q_emb = self.model.encode([query]).astype("float32")

            D, I = self.index.search(q_emb, k)

            results = [
                self.documents[i]
                for i in I[0]
                if i < len(self.documents)
            ]

            return results if results else ["No relevant knowledge found"]

        except Exception:
            logger.exception("[Retriever] Retrieval failed")
            return ["No relevant knowledge found"]
        
    def rebuild_index(self, documents: List[str]):
        """
        Rebuild the FAISS index from scratch with new or updated documents.
        """
        logger.info("[Retriever] Rebuilding FAISS index with updated documents")
        self._build(documents)


# -----------------------------
# SINGLETON
# -----------------------------
_retriever_instance = None


def get_retriever(model_path: str, docs: List[str]):
    global _retriever_instance

    if _retriever_instance is None:
        _retriever_instance = FaissRetriever(model_path, docs)
    else:
        # Detect if knowledge has changed
        current_doc_count = len(_retriever_instance.documents)
        if len(docs) > current_doc_count:
            logger.info("[Retriever] New docs detected, rebuilding index")
            _retriever_instance.rebuild_index(docs)

    return _retriever_instance