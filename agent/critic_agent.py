from core.logger import get_logger
import requests

logger = get_logger(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma:2b"


def review_decision(column_data, retrieved_docs, decision):

    try:
        col = column_data.get("column")
        dtype = column_data.get("dtype")
        missing = round(column_data.get("missing_percent", 0))

        knowledge = "\n".join(f"- {doc}" for doc in retrieved_docs)

        prompt = f"""
You are a senior ML reviewer.

Column:
- {col}, {dtype}, {missing}% missing

Knowledge:
{knowledge}

Proposed Decision:
{decision}

Task:
Evaluate the decision.

Output format:

Final Decision: <refined decision>
Confidence: <High/Medium/Low>
Review: <short explanation>
"""

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )

        return response.json().get("response", "").strip()

    except Exception as e:
        logger.error(f"Critic failed: {str(e)}")
        return decision