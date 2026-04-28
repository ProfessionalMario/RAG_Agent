"""
LLM reasoning layer.

Talks to a local Ollama-served Gemma3 (or any compatible model) with a
strict, deterministic prompt. When Ollama is unreachable and
`LLM_GRACEFUL_FALLBACK` is enabled (default), returns a safe deterministic
string so the rest of the pipeline — and the test suite — keeps working.
"""
from __future__ import annotations

from typing import Dict, List

import requests

from core.config import (
    LLM_GRACEFUL_FALLBACK,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT,
    OLLAMA_URL,
)
from core.logger import get_logger

logger = get_logger(__name__)

_FALLBACK_DECISION = (
    "Decision: keep\n"
    "Reason: LLM unreachable — using safe fallback (no transformation)."
)


def build_prompt(column_data: Dict, retrieved_docs: List[str]) -> str:
    """Assemble the strict 'Senior ML engineer' prompt with RAG context."""
    col = column_data.get("column", "Unknown")
    dtype = column_data.get("dtype", "Unknown")
    missing = round(column_data.get("missing_percent", 0))
    skew = column_data.get("skew", 0)

    if retrieved_docs:
        knowledge = "\n".join(f"- [DOC]: {d}" for d in retrieved_docs)
    else:
        knowledge = "No specific scikit-learn documentation found."

    return (
        "You are a Senior ML Engineer. Pick a Scikit-Learn preprocessing strategy.\n\n"
        "RULES:\n"
        "- Use ONLY the provided KNOWLEDGE and COLUMN PROFILE.\n"
        "- Treat each request as independent — do not infer from past datasets.\n"
        "- If knowledge is weak, fall back to safe standard practice.\n\n"
        f"SCIKIT-LEARN KNOWLEDGE:\n{knowledge}\n\n"
        f"COLUMN PROFILE:\n"
        f"- Name: {col}\n"
        f"- Data Type: {dtype}\n"
        f"- Missing Values: {missing}%\n"
        f"- Skewness: {skew}\n\n"
        "TASK:\n"
        "Pick the right transformer. Focus on missing-value handling, encoding\n"
        "for categorical, transformation for skew, scaling, and small-sample risk.\n\n"
        "OUTPUT FORMAT (STRICT):\n"
        "Decision: <Scikit-Learn class and strategy>\n"
        "Reason: <Brief technical justification>\n"
    ).strip()


def call_llm_with_fallback(prompt: str, model_name: str = OLLAMA_MODEL) -> str:
    """POST the prompt to Ollama. On failure return a deterministic fallback."""
    try:
        logger.info("[REASONING] Calling Ollama model=%s", model_name)
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 512},
            },
            timeout=OLLAMA_TIMEOUT,
        )
        if response.status_code != 200:
            logger.error(f"[REASONING] Ollama HTTP {response.status_code}: {response.text[:200]}")
            if LLM_GRACEFUL_FALLBACK:
                return _FALLBACK_DECISION
            raise RuntimeError(f"Ollama HTTP {response.status_code}")
        return response.json().get("response", "").strip()

    except (requests.RequestException, ValueError) as exc:
        logger.error("[REASONING] Ollama call failed: %s", exc)
        if LLM_GRACEFUL_FALLBACK:
            return _FALLBACK_DECISION
        raise 


def parse_llm_output(output: str) -> Dict[str, str]:
    """Extract Decision / Reason keys; tolerant of formatting drift."""
    parsed = {"decision": "keep", "reason": "Fallback applied"}
    if not output:
        return parsed
    for line in output.splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        k = key.strip().lower()
        v = val.strip()
        if "decision" in k:
            parsed["decision"] = v or parsed["decision"]
        elif "reason" in k:
            parsed["reason"] = v or parsed["reason"]
        
    return parsed
