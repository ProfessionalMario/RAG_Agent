import requests
from typing import List, Dict

from core.logger import get_logger

logger = get_logger(__name__)


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:4b-it-qat"  # or "gemma:2b" depending on your setup


def build_prompt(column_data: Dict, retrieved_docs: List[str]) -> str:
    """
    Build constrained prompt for reasoning
    """

    col = column_data.get("column")
    dtype = column_data.get("dtype")
    missing = round(column_data.get("missing_percent", 0))

    knowledge = "\n".join(f"- {doc}" for doc in retrieved_docs)

    prompt = f"""
    You are a senior ML engineer.

    Column:
    - Name: {col}
    - Type: {dtype}
    - Missing: {missing}%

    Knowledge:
    {knowledge}

    Task:
    Give the BEST preprocessing decision.

    Rules:
    - Be specific (mean, median, drop, mode)
    - Use only given knowledge
    - Be concise

    Output:

    Decision: <specific action>
    Reason: <clear justification>
    """

    return prompt.strip()


def call_llm(prompt: str) -> str:
    """
    Call Ollama (Gemma) for reasoning
    """

    try:
        logger.info("Calling Ollama LLM")

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )

        if response.status_code != 200:
            logger.error(f"Ollama error: {response.text}")
            return "LLM call failed"

        output = response.json().get("response", "")

        logger.info("LLM response received")
        return output.strip()

    except Exception as e:
        logger.exception(f"LLM call failed: {str(e)}")
        return "LLM call exception"
    


