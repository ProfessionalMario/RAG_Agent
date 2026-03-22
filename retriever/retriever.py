from retriever.embedding import embed_texts
from retriever.vector_store import VectorStore

from core.logger import get_logger
from core.exceptions import RetrievalError
from core.tracking import log_params

logger = get_logger(__name__)


class Retriever:
    def __init__(self, knowledge_chunks):
        self.chunks = knowledge_chunks
        self.embeddings = embed_texts(self.chunks)

        self.store = VectorStore(dim=len(self.embeddings[0]))
        self.store.add(self.embeddings, self.chunks)

        logger.info("Retriever initialized with knowledge base")

    def retrieve(self, query: str, top_k=3):
        logger.info(f"Retrieval query: {query}")

        # Track experiment
        log_params({
            "top_k": top_k
        })

        query_embedding = embed_texts([query])[0]

        results = self.store.search(query_embedding, top_k)

        if not results:
            logger.error("No relevant knowledge found")
            raise RetrievalError("No relevant knowledge found")

        logger.info(f"Retrieved results: {results}")

        return results
    


