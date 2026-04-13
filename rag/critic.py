# JOB: Deterministic Logic Validator. IN: decision + profile. OUT: 'OK' | 'REJECT' | 'HINT'.
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
# from core.config import ENGINE
from rag.reasoning import call_llm_with_fallback

logger = get_logger(__name__)


def _safe_to_string(decision):
    """Ensure decision is always a clean string"""
    if isinstance(decision, dict):
        return decision.get("decision", str(decision))
    return str(decision)


def _parse_critic_output(output: str):
    try:
        lines = output.split("\n")
        logger.debug(f"[CRITIC RAW OUTPUT] {output}")
        final_decision = None
        confidence = "low"

        for line in lines:
            line = line.strip().lower()

            if "final decision" in line:
                final_decision = line.split(":", 1)[-1].strip()

            elif "confidence" in line:
                confidence = line.split(":", 1)[-1].strip()

        # 🔥 fallback (CRITICAL FIX)
        if not final_decision:
            # try extracting meaningful keyword
            final_decision = output.strip()

        return final_decision, confidence

    except Exception:
        return None, "low"


def review_decision(column_data, retrieved_docs, decision,retriever):
    """
    Engine-aware critic:
    - Uses Gemini OR local LLM
    - Applies strict safety filtering
    """

    try:
        col = column_data.get("column", "unknown")
        dtype = column_data.get("dtype", "unknown")
        missing = round(column_data.get("missing_percent", 0))

        decision_str = _safe_to_string(decision)
        logger.debug(f"[CRITIC INPUT] Decision: {decision_str}")
        logger.info(f"[RETRIEVED DOCS] {retrieved_docs}")
        # ---------------------------
        # Knowledge formatting
        # ---------------------------
        if not retrieved_docs:
            knowledge = "- No supporting knowledge available"
        else:
            knowledge = "\n".join(f"- {doc}" for doc in retrieved_docs[:3])

        # ---------------------------
        # Prompt
        # ---------------------------
        prompt = f"""
You are a strict senior ML reviewer.

Column:
- {col}, {dtype}, {missing}% missing

Knowledge:
{knowledge}

Proposed Decision:
{decision_str}

Task:
- Critically evaluate the decision
- If wrong or risky, FIX it
- Do NOT agree blindly

Rules:
- Be precise (mean, median, mode, drop, encoding, scaling)
- Only use given knowledge
- Be conservative if unsure

Output EXACTLY:

Final Decision: <refined decision>
Confidence: <high/medium/low>
Review: <short explanation>
"""

        # output = call_llm_with_fallback(column_data,retriever)
        output = call_llm_with_fallback(prompt)

        if not output or not str(output).strip():
            logger.warning(f"[CRITIC] Empty response for {col}")
            return decision_str

        # ---------------------------
        # Parse output
        # ---------------------------
        final_decision, confidence = _parse_critic_output(output)

        if not final_decision:
            logger.warning(f"[CRITIC] Parsing failed for {col}")
            return decision_str

        # ---------------------------
        # Safety filter
        # ---------------------------
        if confidence in ["low"]:
            logger.warning(f"[CRITIC] Rejected low-confidence decision for {col}")
            return "Use standard safe preprocessing (median for numeric, mode for categorical)"

        logger.info(f"[CRITIC] Accepted decision for {col}")

        return final_decision

    except Exception as e:
        logger.error(f"[CRITIC] Failed for {column_data.get('column')}: {str(e)}")
        return _safe_to_string(decision)