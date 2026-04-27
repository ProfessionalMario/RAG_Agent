# # JOB: Deterministic Logic Validator. IN: decision + profile. OUT: 'OK' | 'REJECT' | 'HINT'.
# """
# LLM Decision Reviewer – Developer Guide

# Module that validates and refines preprocessing decisions using a local LLM.

# --------------------------------------------------------------------------------
# Callable Functions:
# --------------------------------------------------------------------------------
# review_decision(column_data: dict, retrieved_docs: List[str], decision: str) -> str
#     - Reviews an LLM-generated preprocessing decision.
#     - Inputs:
#         column_data : dict
#             e.g., {"column": "Age", "dtype": "numeric", "missing_percent": 20}
#         retrieved_docs : List[str]
#             Context retrieved from knowledge base.
#         decision : str
#             Initial decision from LLM generator.
#     - Returns:
#         str : Final safe decision.
#     - Behavior:
#         - Accepts high-confidence outputs.
#         - Rejects low/medium confidence outputs.
#         - Falls back to safe default if needed.

# --------------------------------------------------------------------------------
# Safety Notes:
# --------------------------------------------------------------------------------
# - Only uses retrieved knowledge.
# - Confidence filtering ensures reliable outputs.
# - Invalid or low-confidence responses automatically use a safe fallback.

# """

# from core.logger import get_logger
# # from core.config import ENGINE
# from rag.reasoning import call_llm_with_fallback

# logger = get_logger(__name__)


# def _safe_to_string(decision):
#     """Ensure decision is always a clean string"""
#     if isinstance(decision, dict):
#         return decision.get("decision", str(decision))
#     return str(decision)


# def _parse_critic_output(output: str):
#     try:
#         lines = output.split("\n")
#         logger.debug(f"[CRITIC RAW OUTPUT] {output}")
#         final_decision = None
#         confidence = "low"

#         for line in lines:
#             line = line.strip().lower()

#             if "final decision" in line:
#                 final_decision = line.split(":", 1)[-1].strip()

#             elif "confidence" in line:
#                 confidence = line.split(":", 1)[-1].strip()

#         # 🔥 fallback (CRITICAL FIX)
#         if not final_decision:
#             # try extracting meaningful keyword
#             final_decision = output.strip()

#         return final_decision, confidence

#     except Exception:
#         return None, "low"


# def review_decision(column_data, retrieved_docs, decision,retriever):
#     """
#     Engine-aware critic:
#     - Uses Gemini OR local LLM
#     - Applies strict safety filtering
#     """

#     try:
#         col = column_data.get("column", "unknown")
#         dtype = column_data.get("dtype", "unknown")
#         missing = round(column_data.get("missing_percent", 0))

#         decision_str = _safe_to_string(decision)
#         logger.debug(f"[CRITIC INPUT] Decision: {decision_str}")
#         logger.info(f"[RETRIEVED DOCS] {retrieved_docs}")
#         # ---------------------------
#         # Knowledge formatting
#         # ---------------------------
#         if not retrieved_docs:
#             knowledge = "- No supporting knowledge available"
#         else:
#             knowledge = "\n".join(f"- {doc}" for doc in retrieved_docs[:3])

#         # ---------------------------
#         # Prompt
#         # ---------------------------
#         prompt = f"""
# You are a strict senior ML reviewer.

# Column:
# - {col}, {dtype}, {missing}% missing

# Knowledge:
# {knowledge}

# Proposed Decision:
# {decision_str}

# Task:
# - Critically evaluate the decision
# - If wrong or risky, FIX it
# - Do NOT agree blindly

# Rules:
# - Be precise (mean, median, mode, drop, encoding, scaling)
# - Only use given knowledge
# - Be conservative if unsure

# Output EXACTLY:

# Final Decision: <refined decision>
# Confidence: <high/medium/low>
# Review: <short explanation>
# """

#         # output = call_llm_with_fallback(column_data,retriever)
#         output = call_llm_with_fallback(prompt)

#         if not output or not str(output).strip():
#             logger.warning(f"[CRITIC] Empty response for {col}")
#             return decision_str

#         # ---------------------------
#         # Parse output
#         # ---------------------------
#         final_decision, confidence = _parse_critic_output(output)

#         if not final_decision:
#             logger.warning(f"[CRITIC] Parsing failed for {col}")
#             return decision_str

#         # ---------------------------
#         # Safety filter
#         # ---------------------------
#         if confidence in ["low"]:
#             logger.warning(f"[CRITIC] Rejected low-confidence decision for {col}")
#             return "Use standard safe preprocessing (median for numeric, mode for categorical)"

#         logger.info(f"[CRITIC] Accepted decision for {col}")

#         return final_decision

#     except Exception as e:
#         logger.error(f"[CRITIC] Failed for {column_data.get('column')}: {str(e)}")
#         return _safe_to_string(decision)


















import re
from typing import List, Dict, Tuple
from core.logger import get_logger
from rag.reasoning import call_llm_with_fallback

logger = get_logger(__name__)

# --- GLOBAL KEYWORDS ---
MODEL_KEYWORDS = [
    "regressor", "classifier", "svm", "svc",
    "randomforest", "logisticregression",
    "linearregression", "xgboost", "boosting"
]

PREPROCESS_KEYWORDS = [
    "encoder", "scaler", "imputer",
    "transform", "normalization", "encoding"
]

DANGEROUS_PATTERNS = [
    "drop", "delete", "truncate",
    "rm ", "sudo", "chmod 777",
    "shutdown", "reboot",
    "os.system", "subprocess",
]

def is_dangerous(text: str) -> bool:
    text = text.lower()
    return any(pattern in text for pattern in DANGEROUS_PATTERNS)

def is_model_output(decision: str) -> bool:
    try:
        decision = decision.lower()

        model_keywords = [
            "regressor", "classifier", "svc", "svm",
            "forest", "boost", "xgboost", "lightgbm",
            "logistic", "linearregression", "knn",
            "naive bayes", "model"
        ]

        return any(k in decision for k in model_keywords)

    except Exception as e:
        logger.error(f"[MODEL CHECK] Failed: {e}")
        return False


def is_dangerous(command: str) -> bool:
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, command.lower()):
            return True
    return False

def should_skip_llm(col: dict, dataset_context: dict) -> bool:
    try:
        dtype = col.get("dtype", "").lower()
        missing = col.get("missing_percent", 0)
        skew = abs(col.get("skew", 0))
        is_target = col.get("is_target", False)
        high_corr = col.get("high_target_corr", False)

        # Hard stops (never skip)
        if is_target or high_corr:
            return False

        if dtype == "numeric" and missing == 0 and skew < 0.5:
            return True

        return False

    except Exception:
        return False

def _safe_to_string(decision) -> str:
    try:
        if isinstance(decision, dict):
            return decision.get("decision", str(decision))
        return str(decision)
    except Exception as e:
        logger.error(f"[CRITIC] _safe_to_string error: {e}")
        return ""


def _parse_critic_output(output: str) -> Tuple[str, str]:
    try:
        decision_match = re.search(r"(Final Decision:)?\s*([A-Za-z]+)", output)
        final_decision = decision_match.group(2) if decision_match else output.strip().split("\n")[0]
        return final_decision.strip(), "high"
    except Exception as e:
        logger.error(f"[CRITIC] Parse Error: {e}")
        return "", "low"


# --- NEW: Deterministic Doc Scoring ---
def _score_and_filter_docs(docs: List[str], dtype: str) -> List[str]:
    try:
        scored = []

        for doc in docs:
            text = doc.lower()
            score = 0

            # Positive signals
            if any(k in text for k in PREPROCESS_KEYWORDS):
                score += 3
            if "sklearn" in text:
                score += 1

            # Type alignment
            if "numeric" in dtype and "scale" in text:
                score += 2
            if "categorical" in dtype and "encode" in text:
                score += 2

            # Negative signals
            if any(k in text for k in MODEL_KEYWORDS):
                score -= 5

            if score > 0:
                scored.append((score, doc))

        scored.sort(reverse=True, key=lambda x: x[0])
        filtered = [d for _, d in scored[:3]]

        logger.info(f"[CRITIC] Filtered docs count: {len(filtered)} / {len(docs)}")
        return filtered if filtered else docs[:2]

    except Exception as e:
        logger.error(f"[CRITIC] Doc scoring failure: {e}")
        return docs[:2]


def review_decision(full_record: Dict, retrieved_docs: List[str], retriever=None) -> None:
    """
    Senior-level Reviewer. Patches 'full_record' in-place.
    """
    try:
        logger.info(f"\n[CRITIC] Reviewing column: {full_record.get('column')}")

        current_decision = _safe_to_string(full_record.get("decision", "None"))
        # --- 0. SAFETY GUARDRAIL (Deterministic) ---
        if is_dangerous(current_decision):
            logger.critical(f"🚨 [SAFETY] Dangerous decision detected for {col_name}: {current_decision}")
            
            full_record["decision"] = "BLOCKED"
            full_record["reason"] += " | Safety Critic: Dangerous command detected and blocked."
            
            return
        
        col_name = full_record.get("column", "unknown")
        dtype = str(full_record.get("dtype", "unknown")).lower()

        logger.debug(f"[CRITIC] Initial decision: {current_decision}")
        logger.debug(f"[CRITIC] dtype: {dtype}")

        is_categorical = any(c in dtype for c in ["object", "category", "string"])
        is_numeric = "int" in dtype or "float" in dtype

        # --- 1. HARD MODEL REJECTION ---
        if any(k in current_decision.lower() for k in MODEL_KEYWORDS):
            logger.warning(f"🚫 [CRITIC] Model detected in decision for {col_name}: {current_decision}")
            full_record["decision"] = "None"
            full_record["reason"] += " | Critic removed invalid model selection."
            return

        # --- 2. TYPE ENFORCEMENT ---
        if is_categorical and "scaler" in current_decision.lower():
            logger.warning(f"🛡️ [CRITIC] Fixing scaler on categorical: {col_name}")
            full_record["decision"] = "OneHotEncoder"
            full_record["reason"] += " | Critic: categorical cannot be scaled."
            return

        if is_numeric and "encoder" in current_decision.lower():
            logger.warning(f"🛡️ [CRITIC] Fixing encoder on numeric: {col_name}")
            full_record["decision"] = "StandardScaler"
            full_record["reason"] += " | Critic: numeric does not require encoding."
            return

        # --- 3. DOC FILTERING ---
        filtered_docs = _score_and_filter_docs(retrieved_docs, dtype)

        knowledge = "\n".join(
            f"- {doc}" for doc in filtered_docs
        ) if filtered_docs else "No context."

        # --- 4. LLM VALIDATION ---
        prompt = f"""
ROLE: Strict ML preprocessing auditor.

COLUMN: {col_name} ({dtype})

DOCUMENTATION:
{knowledge}

PROPOSED:
{current_decision}

RULES:
- Only preprocessing classes allowed
- No models (regression, classifier, SVM, etc.)
- Numeric → scaling/imputation
- Categorical → encoding/imputation

TASK:
Return ONLY:
- Keep
OR
- Correct sklearn class name

No explanation.
"""

        critic_raw = call_llm_with_fallback(prompt)

        if not critic_raw:
            logger.warning("[CRITIC] Empty LLM response")
            return

        final_decision, _ = _parse_critic_output(critic_raw)

        logger.debug(f"[CRITIC] LLM suggested: {final_decision}")

        # --- 5. PATCH ---
        if final_decision.lower() != "keep" and len(final_decision) > 2:
            logger.info(f"✅ [CRITIC] Patching {col_name}: {current_decision} -> {final_decision}")
            full_record["decision"] = final_decision
            full_record["reason"] += f" | Critic correction: {final_decision}"

    except Exception as e:
        logger.error(f"[CRITIC] Critical failure: {e}")
        # fail-safe: do nothing


# ------------------ TEST BLOCK ------------------

if __name__ == "__main__":
    print("\n" + "="*60)
    print("CRITIC TEST: Model Rejection + Type Enforcement")
    print("="*60)

    # Mock record
    record = {
        "column": "internet",
        "dtype": "object",
        "decision": "StandardScaler",
        "reason": "Initial guess"
    }

    # Bad docs (intentionally polluted)
    docs = [
        "StandardScaler scales numeric data.",
        "LogisticRegression is used for classification.",
        "RandomForestRegressor handles regression tasks."
    ]

    print(f"\nBefore:\n{record}")

    review_decision(record, docs)

    print(f"\nAfter:\n{record}")

    print("\nExpected:")
    print("- Should NOT contain model")
    print("- Should convert scaler → encoder")
    print("="*60 + "\n")