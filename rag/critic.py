"""
Critic / Validation layer.

`review_decision()` patches a draft preprocessing decision in place using a
ladder of validation rules:

  0. Safety guardrail   — blocks dangerous shell-ish patterns outright
  1. Model rejection    — preprocessing layer must not pick a *model*
  2. Type enforcement   — categorical can't be scaled, numeric can't be encoded
  3. Doc filtering      — re-rank retrieved docs by topical relevance
  4. LLM second opinion — ask the model to confirm/replace the decision
  5. Patch              — apply the corrected decision back into the record
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

from core.logger import get_logger
from rag.reasoning import call_llm_with_fallback

logger = get_logger(__name__)

# --- Vocabularies -------------------------------------------------------------
MODEL_KEYWORDS: Tuple[str, ...] = (
    "regressor", "classifier", "svm", "svc",
    "randomforest", "logisticregression", "linearregression",
    "xgboost", "boosting", "lightgbm", "knn", "naive bayes",
)

PREPROCESS_KEYWORDS: Tuple[str, ...] = (
    "encoder", "scaler", "imputer",
    "transform", "normalization", "encoding",
)

DANGEROUS_PATTERNS: Tuple[str, ...] = (
    r"\brm\s+-rf\b", r"\bsudo\b", r"\bchmod\s+777\b",
    r"\bshutdown\b", r"\breboot\b",
    r"\bos\.system\b", r"\bsubprocess\b",
    r"\b__import__\b", r"\beval\(", r"\bexec\(",
)


# --- Helpers ------------------------------------------------------------------
def is_dangerous(text: str) -> bool:
    """True if the decision string contains a shell/eval injection pattern."""
    text = (text or "").lower()
    return any(re.search(pat, text) for pat in DANGEROUS_PATTERNS)


def _safe_to_string(decision) -> str:
    if isinstance(decision, dict):
        return decision.get("decision", str(decision))
    return str(decision) if decision is not None else ""


def _parse_critic_output(output: str) -> str:
    """Pull the critic's first meaningful word, ignoring Decision/Final prefixes."""
    if not output:
        return ""
    first_line = output.strip().splitlines()[0].strip()
    # Strip 'Final Decision:' or 'Decision:' prefix the LLM may emit.
    first_line = re.sub(r"^(?:final\s+)?decision\s*:\s*",
                        "", first_line, flags=re.IGNORECASE)
    match = re.search(r"([A-Za-z][\w]*)", first_line)
    return match.group(1).strip() if match else first_line


def _score_and_filter_docs(docs: List[str], dtype: str) -> List[str]:
    """Deterministic re-rank — boost preprocessing context, penalise model docs."""
    scored: List[Tuple[int, str]] = []
    dtype = (dtype or "").lower()
    for doc in docs or []:
        text = doc.lower()
        score = 0
        if any(k in text for k in PREPROCESS_KEYWORDS):
            score += 3
        if "sklearn" in text or "scikit" in text:
            score += 1
        if any(n in dtype for n in ("numeric", "int", "float")) and "scal" in text:
            score += 2
        if any(c in dtype for c in ("object", "category", "string")) and "encod" in text:
            score += 2
        if any(k in text for k in MODEL_KEYWORDS):
            score -= 5
        if score > 0:
            scored.append((score, doc))

    scored.sort(reverse=True, key=lambda x: x[0])
    filtered = [d for _, d in scored[:3]]
    return filtered or list(docs or [])[:2]


# --- Public API ---------------------------------------------------------------
def review_decision(full_record: Dict, retrieved_docs: List[str]) -> None:
    """Patch `full_record["decision"]` and `["reason"]` in place."""
    col_name = full_record.get("column", "unknown")
    dtype = str(full_record.get("dtype", "unknown")).lower()
    full_record.setdefault("reason", "")

    current_decision = _safe_to_string(full_record.get("decision", "None"))
    logger.info("[CRITIC] Reviewing column=%s decision=%r", col_name, current_decision)

    # --- 0. Safety guardrail -------------------------------------------------
    if is_dangerous(current_decision):
        logger.critical("[CRITIC] Dangerous pattern in %s: %r",
                        col_name, current_decision)
        full_record["decision"] = "BLOCKED"
        full_record["reason"] = (
            (full_record["reason"] + " | ").lstrip(" |")
            + "Safety critic: dangerous command blocked."
        )
        return

    is_categorical = any(c in dtype for c in ("object", "category", "string"))
    is_numeric = "int" in dtype or "float" in dtype or "numeric" in dtype
    lower_decision = current_decision.lower()

    # --- 1. Hard model rejection --------------------------------------------
    if any(k in lower_decision for k in MODEL_KEYWORDS):
        logger.warning("[CRITIC] Model leaked into preprocessing for %s", col_name)
        full_record["decision"] = "None"
        full_record["reason"] = (
            (full_record["reason"] + " | ").lstrip(" |")
            + "Critic removed invalid model selection."
        )
        return

    # --- 2. Type enforcement -------------------------------------------------
    if is_categorical and "scaler" in lower_decision:
        full_record["decision"] = "OneHotEncoder"
        full_record["reason"] = (
            (full_record["reason"] + " | ").lstrip(" |")
            + "Critic: categorical features cannot be scaled."
        )
        return

    if is_numeric and "encoder" in lower_decision:
        full_record["decision"] = "StandardScaler"
        full_record["reason"] = (
            (full_record["reason"] + " | ").lstrip(" |")
            + "Critic: numeric features do not require encoding."
        )
        return

    # --- 3. Doc filter -------------------------------------------------------
    filtered_docs = _score_and_filter_docs(retrieved_docs, dtype)
    knowledge = (
        "\n".join(f"- {d}" for d in filtered_docs) if filtered_docs else "No context."
    )

    # --- 4. LLM validation ---------------------------------------------------
    prompt = (
        "ROLE: Strict ML preprocessing auditor.\n\n"
        f"COLUMN: {col_name} ({dtype})\n\n"
        f"DOCUMENTATION:\n{knowledge}\n\n"
        f"PROPOSED:\n{current_decision}\n\n"
        "RULES:\n"
        "- Only preprocessing classes allowed (no models).\n"
        "- Numeric -> scaling/imputation.\n"
        "- Categorical -> encoding/imputation.\n\n"
        "TASK: Return ONLY 'Keep' or the corrected sklearn class name.\n"
    )
    critic_raw = call_llm_with_fallback(prompt)
    if not critic_raw:
        return

    final_decision = _parse_critic_output(critic_raw)
    logger.debug("[CRITIC] LLM suggested: %r", final_decision)

    # --- 5. Patch ------------------------------------------------------------
    if final_decision and final_decision.lower() != "keep" and len(final_decision) > 2:
        full_record["decision"] = final_decision
        full_record["reason"] = (
            (full_record["reason"] + " | ").lstrip(" |")
            + f"Critic correction: {final_decision}"
        )
