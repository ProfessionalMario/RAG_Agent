# import os
# import faiss
# import pickle
# from typing import List
# from sentence_transformers import SentenceTransformer
# from core.logger import get_logger
# import numpy as np
# logger = get_logger(__name__)

# # -----------------------------
# # CONFIG
# # -----------------------------
# FAISS_DIR = "storage"
# INDEX_PATH = os.path.join(FAISS_DIR, "faiss.index")
# DOCS_PATH = os.path.join(FAISS_DIR, "meta.pkl")

# # -----------------------------
# # MODEL CACHE
# # -----------------------------
# _model_cache = {}

# def get_model(model_path: str):
#     """Load SentenceTransformer model (with caching)"""
#     if model_path in _model_cache:
#         return _model_cache[model_path]
#     logger.info("[Retriever] Loading embedding model: %s", model_path)
#     # model = SentenceTransformer(model_path if model_path.startswith("./") else "all-MiniLM-L6-v2")
#     model = SentenceTransformer(model_path if model_path.startswith("./") else "BAAI/bge-small-en-v1.5")
#     _model_cache[model_path] = model
#     return model

# # -----------------------------
# # TEXT PIPELINE
# # -----------------------------
# def clean_text(text: str) -> str:
#     """Basic text cleaning: remove non-ascii, normalize spaces"""
#     if not text:
#         return ""
#     return " ".join(text.encode("ascii", "ignore").decode().split())

# def chunk_text(text: str, chunk_size=500, overlap=50) -> List[str]:
#     """Split text into overlapping chunks"""
#     chunks = []
#     start = 0
#     while start < len(text):
#         end = start + chunk_size
#         chunks.append(text[start:end])
#         start += chunk_size - overlap
#     return chunks

# def is_valid(text: str) -> bool:
#     """Check if chunk has enough words"""
#     return len(text.split()) >= 5

# def encode_in_batches(model, texts, batch_size=32):
#     embeddings = []

#     for i in range(0, len(texts), batch_size):
#         batch = texts[i:i + batch_size]
#         emb = model.encode(batch, show_progress_bar=False)
#         embeddings.append(emb)

#     return np.vstack(embeddings).astype("float32")

# # -----------------------------
# # RETRIEVER
# # -----------------------------
# class FaissRetriever:
#     def __init__(self, model_path: str, documents: List[str]):
#         logger.info("[Retriever] Initializing FAISS retriever")
#         self.model = get_model(model_path)
#         if os.path.exists(INDEX_PATH) and os.path.exists(DOCS_PATH):
#             self._load()
#         else:
#             self._build(documents)

#     def _build(self, documents: List[str]):
#         logger.info("[Retriever] Building FAISS index")
#         processed = []
#         for doc in documents:
#             doc_clean = clean_text(str(doc))

#             if is_valid(doc_clean):
#                 processed.append(doc_clean) 
#             if not processed:
#                 raise ValueError("No valid documents to index")
#         processed = list(set(processed))
#         # embeddings = self.model.encode(processed, show_progress_bar=True).astype("float32")
#         embeddings = encode_in_batches(
#         self.model,
#         [f"passage: {text}" for text in processed]
#         ).astype("float32")
#         embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
#         dim = embeddings.shape[1]
#         self.index = faiss.IndexFlatL2(dim)
#         self.index.add(embeddings)
#         self.documents = processed
#         # print("======================",processed[:5],"==========================")
#         self._save()
#         logger.info("[Retriever] FAISS index built")

#     def _save(self):
#         os.makedirs(FAISS_DIR, exist_ok=True)
#         faiss.write_index(self.index, INDEX_PATH)
#         with open(DOCS_PATH, "wb") as f:
#             pickle.dump(self.documents, f)

#     def _load(self):
#         logger.info("[Retriever] Loading FAISS index from disk")
#         self.index = faiss.read_index(INDEX_PATH)
#         with open(DOCS_PATH, "rb") as f:
#             self.documents = pickle.load(f)

#     def retrieve(self, query: str, k: int = 3, threshold: float = 2.0) -> List[str]:
#         logger.debug(f"[RETRIEVAL] Query: {query}")

#         if not query or not isinstance(query, str):
#             return []
        
#         q_emb = self.model.encode([f"query: {query}"]).astype("float32")
#         # q_emb = self.model.encode([query]).astype("float32")
#         D, I = self.index.search(q_emb, k)

#         results = []
#         for dist, idx in zip(D[0], I[0]):
#             if 0 <= idx < len(self.documents):  
#                 results.append(self.documents[idx]) 
#         logger.debug(f"[RETRIEVAL] Index ntotal: {self.index.ntotal}")
#         logger.debug(f"[RETRIEVAL] Top indices: {I}")
#         logger.debug(f"[RETRIEVAL] Distances: {D}")
#         logger.debug(f"[RETRIEVAL] Returning {len(results)} docs")
#         logger.debug(f"[RETRIEVAL] Retrieved docs: {results}")
#         return results
    

#     def rebuild_index(self, documents: List[str]):
#         """Rebuild FAISS index from new documents"""
#         logger.info("[Retriever] Rebuilding FAISS index")
#         self._build(documents)

# # -----------------------------
# # SINGLETON
# # -----------------------------
# _retriever_instance = None

# def get_retriever(model_path: str, docs: List[str]):
#     global _retriever_instance
#     if _retriever_instance is None:
#         _retriever_instance = FaissRetriever(model_path, docs)
#     elif len(docs) > len(_retriever_instance.documents):
#         _retriever_instance.rebuild_index(docs)
#     return _retriever_instance




















# import os
# import faiss
# import pickle
# import numpy as np
# from typing import List
# from sentence_transformers import SentenceTransformer
# from core.logger import get_logger
# logger = get_logger(__name__)
# # -----------------------------
# # CONFIG & MODEL
# # -----------------------------
# FAISS_DIR = "storage"
# INDEX_PATH = os.path.join(FAISS_DIR, "faiss.index")
# DOCS_PATH = os.path.join(FAISS_DIR, "meta.pkl")
# _retriever_instance = None


# def get_model(model_path: str):
#     # Using BGE-small as per your 2026 spec
#     return SentenceTransformer(model_path if model_path.startswith("./") else "BAAI/bge-small-en-v1.5")

# class FaissRetriever:
#     def __init__(self, model_path: str, documents: List[str]):
#         logger.info("[Retriever] Initializing FAISS retriever")
#         self.model = get_model(model_path)
#         self.index = None
#         self.documents = []

#         if os.path.exists(INDEX_PATH) and os.path.exists(DOCS_PATH):
#             self._load()
#         elif documents:
#             self._build(documents)
#         else:
#             logger.warning("[Retriever] No index found and no documents provided.")

#     def _build(self, documents: List[str]):
#         logger.info("[Retriever] Building FAISS index")
#         # Deduplicate and clean
#         processed = list(set([str(d).strip() for d in documents if len(str(d).split()) >= 5]))
        
#         if not processed:
#             raise ValueError("No valid documents to index")

#         # Encode with BGE Passage Prefix
#         embeddings = self.model.encode(
#             [f"passage: {text}" for text in processed], 
#             show_progress_bar=True
#         ).astype("float32")
        
#         # Normalize for Cosine Similarity (Inner Product)
#         faiss.normalize_L2(embeddings)
        
#         dim = embeddings.shape[1]
#         self.index = faiss.IndexFlatIP(dim) # IP + Normalized = Cosine Similarity
#         self.index.add(embeddings)
#         self.documents = processed
#         self._save()
#         logger.info(f"[Retriever] Index built with {self.index.ntotal} vectors")

#     def _save(self):
#         os.makedirs(FAISS_DIR, exist_ok=True)
#         faiss.write_index(self.index, INDEX_PATH)
#         with open(DOCS_PATH, "wb") as f:
#             pickle.dump(self.documents, f)

#     def _load(self):
#         logger.info("[Retriever] Loading FAISS index from disk")
#         self.index = faiss.read_index(INDEX_PATH)
#         with open(DOCS_PATH, "rb") as f:
#             self.documents = pickle.load(f)

#     def retrieve(self, query: str, k: int = 3) -> List[str]:
#         if not query or not self.index:
#             return []
        
#         # Encode with BGE Query Prefix
#         q_emb = self.model.encode([f"query: {query}"]).astype("float32")
#         faiss.normalize_L2(q_emb)
        
#         D, I = self.index.search(q_emb, k)

#         results = []
#         for dist, idx in zip(D[0], I[0]):
#             if 0 <= idx < len(self.documents):
#                 results.append(self.documents[idx])
        
#         return results

#     def rebuild_index(self, documents: List[str]):
#         self._build(documents)

# # -----------------------------
# # SINGLETON (Strict Return Type)
# # -----------------------------
# _retriever_instance = None

# def get_retriever(model_path: str, docs: List[str]):
#     global _retriever_instance
#     if _retriever_instance is None:
#         _retriever_instance = FaissRetriever(model_path, docs)
#     elif docs and len(docs) > len(_retriever_instance.documents):
#         _retriever_instance.rebuild_index(docs)
    
#     # Returning the object itself (NOT a tuple) to match original expectation
#     return _retriever_instance































import os
import faiss
import pickle
import numpy as np
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from core.logger import get_logger

logger = get_logger(__name__)

# -----------------------------
# CONFIG & PATHS
# -----------------------------
FAISS_DIR = "storage"
INDEX_PATH = os.path.join(FAISS_DIR, "faiss.index")
DOCS_PATH = os.path.join(FAISS_DIR, "docs.pkl") # Standardized with knowledge.py
MODEL_NAME = "BAAI/bge-small-en-v1.5"

class FaissRetriever:
    def __init__(self, model_name: str = MODEL_NAME, documents: Optional[List[str]] = None):
        """
        Retrieval Engine using BGE-small.
        IP (Inner Product) + L2 Normalization = Cosine Similarity.
        """
        logger.info(f"🔍 [RETRIEVER] Initializing with {model_name}")
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []

        if os.path.exists(INDEX_PATH) and os.path.exists(DOCS_PATH):
            self._load()
        elif documents:
            self._build(documents)
        else:
            logger.warning("[RETRIEVER] No index found on disk. Ready for build.")

    def _build(self, documents: List[str]):
        """Builds index with BGE 'passage: ' instruction."""
        try:
            logger.info(f"[RETRIEVER] Building index for {len(documents)} docs")
            # Filter and deduplicate
            processed = list(dict.fromkeys([str(d).strip() for d in documents if len(str(d).split()) >= 3]))
            
            # BGE Requirement: Passages must be prefixed
            passages = [f"passage: {text}" for text in processed]
            embeddings = self.model.encode(passages, show_progress_bar=True).astype("float32")
            
            # Normalize for Cosine Similarity
            faiss.normalize_L2(embeddings)
            
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim) 
            self.index.add(embeddings)
            self.documents = processed
            
            self._save()
            logger.info(f"✅ [RETRIEVER] Index built: {self.index.ntotal} vectors")
        except Exception as e:
            logger.error(f"❌ [RETRIEVER] Build failed: {e}")
            raise

    def _save(self):
        os.makedirs(FAISS_DIR, exist_ok=True)
        faiss.write_index(self.index, INDEX_PATH)
        with open(DOCS_PATH, "wb") as f:
            pickle.dump(self.documents, f)

    def _load(self):
        try:
            logger.info("[RETRIEVER] Loading assets from storage...")
            self.index = faiss.read_index(INDEX_PATH)
            with open(DOCS_PATH, "rb") as f:
                self.documents = pickle.load(f)
            logger.info(f"✅ [RETRIEVER] Loaded {len(self.documents)} documents")
        except Exception as e:
            logger.error(f"❌ [RETRIEVER] Load failed: {e}")

    def retrieve(self, query: str, k: int = 3) -> List[str]:
        """Search with BGE 'query: ' instruction."""
        if not query or self.index is None:
            return []
        
        try:
            # BGE Requirement: Queries must be prefixed
            q_emb = self.model.encode([f"query: {query}"]).astype("float32")
            faiss.normalize_L2(q_emb)
            
            scores, indices = self.index.search(q_emb, k)

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if 0 <= idx < len(self.documents):
                    # Only return results with a decent similarity score
                    if score > 0.4: 
                        results.append(self.documents[idx])
            
            return results
        except Exception as e:
            logger.error(f"[RETRIEVER] Search failed: {e}")
            return []

# -----------------------------
# SINGLETON ACCESS
# -----------------------------
_instance = None

def get_retriever(docs: Optional[List[str]] = None):
    """
    Ensures only one instance of the model and index is in memory.
    """
    global _instance
    if _instance is None:
        _instance = FaissRetriever(MODEL_NAME, docs)
    return _instance

# -----------------------------
# TEST BLOCK
# -----------------------------
if __name__ == "__main__":
    test_docs = [
        "Use SimpleImputer with strategy='median' for numeric data with outliers.",
        "OneHotEncoder is preferred for nominal categorical features.",
        "StandardScaler should not be used on sparse data."
    ]
    
    # Force rebuild for test
    r = FaissRetriever(MODEL_NAME, test_docs)
    res = r.retrieve("how to handle numeric outliers?")
    print(f"Query Result: {res}")