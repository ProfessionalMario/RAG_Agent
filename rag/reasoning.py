
import requests
from typing import List, Dict
from core.logger import get_logger

logger = get_logger(__name__)

# -----------------------------
# CONFIG
# -----------------------------
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL_NAME = "gemma3:4b"  # Optimized for speed/deterministic logic

def build_prompt(column_data: Dict, retrieved_docs: List[str]) -> str:
    """Build structured prompt for the LLM using RAG context."""
    col = column_data.get("column", "Unknown")
    dtype = column_data.get("dtype", "Unknown")
    missing = round(column_data.get("missing_percent", 0))
    skew = column_data.get("skew", 0)
    task = column_data.get("task", "unknown")  # regression / classification
    sample_size = column_data.get("sample_size", "unknown")  # small / large
    cardinality = column_data.get("cardinality", "unknown")  # low / high
    outliers = column_data.get("outliers", "unknown")  # present / none
    variance = column_data.get("variance", "unknown")
    clean_knowledge = []
    # Do it in one pass
    if retrieved_docs:
        knowledge = "\n".join(f"- [DOC REFERENCE]: {doc}" for doc in retrieved_docs)
    else:
        knowledge = "No specific scikit-learn documentation found."

    prompt = f"""
You are a Senior ML Engineer. Your goal is to choose a Scikit-Learn preprocessing strategy.

IMPORTANT:
- Treat EACH request as completely independent.
- Do NOT assume information from previous datasets.
- Use ONLY the provided KNOWLEDGE and COLUMN PROFILE.
- If knowledge is weak or missing, fall back to safe, standard preprocessing practices.

SCIKIT-LEARN KNOWLEDGE:
{knowledge}

COLUMN PROFILE:
- Name: {col}
- Data Type: {dtype}
- Missing Values: {missing}%
- Skewness: {skew}

ADDITIONAL CONTEXT (if available):
- Task Type: {task}
- Sample Size: {sample_size}
- Cardinality: {cardinality}
- Outliers: {outliers}
- Variance: {variance}

TASK:
Based ONLY on the COLUMN PROFILE ({col}), select the transformer.
    IGNORE column names mentioned in the [DOC REFERENCE] section; they are EXAMPLES only.
Focus on:
- Correct handling of missing values
- Proper encoding for categorical data
- Transformations for skewed data
- Whether scaling is required or unnecessary
- Avoiding overfitting for small datasets

OUTPUT FORMAT (STRICT):
Decision: <Scikit-Learn Class and strategy>
Reason: <Brief technical justification>
"""
    logger.debug(f"[PROMPT BUILDER] Built prompt for column: {col}")
    return prompt.strip()

def call_llm_with_fallback(prompt: str, model_name: str = MODEL_NAME) -> str:
    """Sends prompt to Ollama and handles connection failures."""
    try:
        logger.info(f"🧠 [REASONING] Calling Ollama ({model_name})...")
        
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,  # Zero randomness for statistical consistency
                    "num_predict": 120   # Enough for a clear reason
                }
            },
            timeout=45
        )
        
        if response.status_code != 200:
            logger.error(f"Ollama error {response.status_code}: {response.text}")
            return "Decision: SimpleImputer(strategy='median')\nReason: LLM service error fallback"

        result = response.json().get("response", "")
        return result.strip()

    except Exception as e:
        logger.error(f"Ollama connection failed: {e}")
        return "Decision: keep\nReason: LLM connection timeout"

def parse_llm_output(output: str) -> Dict[str, str]:
    """Extracts Decision and Reason from LLM text using strict splitting."""
    decision = "keep"
    reason = "Fallback applied"

    if not output:
        return {"decision": decision, "reason": reason}

    try:
        lines = output.split("\n")
        for line in lines:
            if ":" in line:
                key, val = line.split(":", 1)
                if "decision" in key.lower():
                    decision = val.strip()
                elif "reason" in key.lower():
                    reason = val.strip()
        
        return {"decision": decision, "reason": reason}
    except Exception as e:
        logger.warning(f"Failed to parse LLM output: {e}")
        return {"decision": decision, "reason": reason}

# -----------------------------
# TEST BLOCK
# -----------------------------
if __name__ == "__main__":
    test_col = {
        "column": "Age",
        "dtype": "numeric",
        "missing_percent": 15,
        "skew": 2.5
    }
    test_docs = ["For skewed data, use median imputation via SimpleImputer.", 
                 "Apply PowerTransformer for variables with skewness > 1."]
    
    print("--- Testing Reasoning Module ---")
    prompt = build_prompt(test_col, test_docs)
    raw = call_llm_with_fallback(prompt)
    parsed = parse_llm_output(raw)
    
    print(f"RAW OUTPUT:\n{raw}\n")
    print(f"PARSED RESULT: {parsed}")