import faiss
import numpy as np
from typing import List

from sentence_transformers import SentenceTransformer
from core.logger import get_logger

logger = get_logger(__name__)


class FaissRetriever:
    def __init__(self, model_path: str, documents: List[str]):
        """
        model_path: local MiniLM path
        documents: knowledge base chunks
        """

        try:
            logger.info("Loading embedding model")
            self.model = SentenceTransformer(model_path)

            logger.info("Encoding knowledge base")
            self.documents = documents
            embeddings = self.model.encode(documents).astype("float32")

            logger.info("Building FAISS index")
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
            self.index.add(embeddings)

            logger.info("Retriever initialized successfully")

        except Exception as e:
            logger.exception(f"Retriever init failed: {str(e)}")
            raise

    def retrieve(self, query: str, k: int = 3) -> List[str]:
        try:
            logger.info(f"Retrieving for query: {query}")

            q_emb = self.model.encode([query]).astype("float32")

            D, I = self.index.search(q_emb, k)

            results = [self.documents[i] for i in I[0]]

            logger.info(f"Top {k} results retrieved")
            return results

        except Exception as e:
            logger.error(f"Retrieval failed: {str(e)}")
            return []