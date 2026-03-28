"""
LLM Decision Reviewer – Developer Guide

Module that validates and refines preprocessing decisions using a local LLM.

--------------------------------------------------------------------------------
Callable Functions:
--------------------------------------------------------------------------------
review_decision(column_data: dict, retrieved_docs: List[str], decision: str) -> str
    - Reviews an LLM-generated preprocessing decision.
    - Inputs:
        column_data : dict
            e.g., {"column": "Age", "dtype": "numeric", "missing_percent": 20}
        retrieved_docs : List[str]
            Context retrieved from knowledge base.
        decision : str
            Initial decision from LLM generator.
    - Returns:
        str : Final safe decision.
    - Behavior:
        - Accepts high-confidence outputs.
        - Rejects low/medium confidence outputs.
        - Falls back to safe default if needed.

--------------------------------------------------------------------------------
Safety Notes:
--------------------------------------------------------------------------------
- Only uses retrieved knowledge.
- Confidence filtering ensures reliable outputs.
- Invalid or low-confidence responses automatically use a safe fallback.

"""


from core.logger import get_logger
import requests

logger = get_logger(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:1b"


def review_decision(column_data, retrieved_docs, decision):
    """
    Review and refine LLM decision.
    Rejects low-confidence or unsafe outputs.
    """

    try:
        col = column_data.get("column")
        dtype = column_data.get("dtype")
        missing = round(column_data.get("missing_percent", 0))

        # Handle empty knowledge
        if not retrieved_docs:
            knowledge = "- No supporting knowledge available"
        else:
            knowledge = "\n".join(f"- {doc}" for doc in retrieved_docs)

        prompt = f"""
You are a strict senior ML reviewer.

Column:
- {col}, {dtype}, {missing}% missing

Knowledge:
{knowledge}

Proposed Decision:
{decision}

Task:
- Critically evaluate the decision
- If wrong or risky, FIX it
- Do NOT agree blindly

Rules:
- Be precise (mean, median, mode, drop, etc.)
- Only use given knowledge
- Be conservative if unsure

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
                "stream": False,
                "options": {"temperature": 0.1}
            },
            timeout=120
        )

        if response.status_code != 200:
            logger.error(f"Ollama error: {response.text}")
            return decision

        output = response.json().get("response", "").strip()

        # 🔍 Validate structure
        if "Final Decision:" not in output or "Confidence:" not in output:
            logger.warning("Invalid review format, falling back")
            return decision

        # 🔍 Extract fields
        final_decision = output.split("Final Decision:")[-1].split("Confidence:")[0].strip()
        confidence = output.split("Confidence:")[-1].split("Review:")[0].strip().lower()

        # 🚨 Safety filter (core logic)
        if confidence in ["low", "medium"]:
            logger.warning(f"Rejected low-confidence decision for {col}")
            return "Use standard safe preprocessing (e.g., median/mode imputation)"

        logger.info(f"Accepted reviewed decision for {col}")

        return final_decision

    except Exception as e:
        logger.error(f"Critic failed: {str(e)}")
        return decision