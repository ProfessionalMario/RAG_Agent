from pathlib import Path

from retriever.retriever import Retriever
from utils.text_chunking import chunk_text


def load_knowledge():
    kb_path = Path("storage/knowledge_base/eda_rules.txt")
    text = kb_path.read_text(encoding="utf-8")
    return chunk_text(text)


if __name__ == "__main__":
    chunks = load_knowledge()

    retriever = Retriever(chunks)

    query = "numeric column with high missing and skew"

    results = retriever.retrieve(query, top_k=3)

    print("\n🔍 Query:", query)
    print("\n📚 Retrieved Rules:\n")

    for i, r in enumerate(results, 1):
        print(f"{i}. {r}")


