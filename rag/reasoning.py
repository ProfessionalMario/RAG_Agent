"""
LLM Reasoning Module – Developer Guide

Generates preprocessing decisions using a local LLM (Ollama + RAG).

--------------------------------------------------------------------------------
Purpose:
--------------------------------------------------------------------------------
- Build structured prompts from column metadata and retrieved docs
- Send prompt to local LLM via Ollama API
- Return concise, structured decision
- Fallback: log errors and return message if LLM fails

--------------------------------------------------------------------------------
Flow:
--------------------------------------------------------------------------------
Column Data + Retrieved Docs
        ↓
   build_prompt()
        ↓
   call_llm()
        ↓
Decision + Reason

--------------------------------------------------------------------------------
Callable Functions:
--------------------------------------------------------------------------------
build_prompt(column_data: dict, retrieved_docs: list) -> str
    - Creates a structured LLM prompt from input data
    - Ensures rules, format, and safe fallback

call_llm(prompt: str) -> str
    - Sends prompt to Ollama LLM
    - Returns LLM-generated decision + reason
    - Handles errors and logs failures

--------------------------------------------------------------------------------
Usage Example:
--------------------------------------------------------------------------------
from rag.reasoning import build_prompt, call_llm

prompt = build_prompt(column_data, retrieved_docs)
decision = call_llm(prompt)

print(decision)  
# "Decision: Use median imputation
#  Reason: Numeric column with moderate missing values and likely skewed distribution"
"""

import requests
from typing import List, Dict
from core.logger import get_logger

logger = get_logger(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:1b"  

def build_prompt(column_data: Dict, retrieved_docs: List[str]) -> str:
    """Build structured prompt for the LLM using RAG context."""
    col = column_data.get("column", "Unknown")
    dtype = column_data.get("dtype", "Unknown")
    missing = round(column_data.get("missing_percent", 0))
    
    # Format knowledge into a bulleted list
    knowledge = "\n".join(f"- {doc}" for doc in retrieved_docs) if retrieved_docs else "No specific domain knowledge found."

    prompt = f"""
You are a senior ML engineer specializing in data preprocessing.

CONTEXT FROM DOCUMENTATION:
{knowledge}

TARGET COLUMN:
- Name: {col}
- Data Type: {dtype}
- Missing Values: {missing}%

TASK:
Determine the best preprocessing action (e.g., mean imputation, median imputation, mode imputation, drop, or keep).
If the provided context suggests a specific business rule for this column, follow it. 
Otherwise, use standard ML best practices.

OUTPUT FORMAT (Strict):
Decision: <action>
Reason: <brief justification>
"""
    return prompt.strip()

def call_llm_with_fallback(prompt: str, model_name: str = MODEL_NAME) -> str:
    """
    Pure LLM call. Does one thing: Talks to Ollama.
    """
    try:
        logger.info(f"Sending prompt to Ollama ({model_name})")
        
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Keep it deterministic
                    "num_predict": 100   # Keep it concise
                }
            },
            timeout=60
        )
        
        if response.status_code != 200:
            logger.error(f"Ollama error {response.status_code}: {response.text}")
            return "Decision: keep\nReason: LLM service error"

        result = response.json().get("response", "")
        return result.strip()

    except Exception as e:
        logger.error(f"Ollama connection failed: {e}")
        return "Decision: keep\nReason: LLM connection timeout"

def parse_llm_output(output: str) -> Dict[str, str]:
    """Extracts Decision and Reason from LLM text."""
    decision = "keep"
    reason = "Fallback applied"

    if not output:
        return {"decision": decision, "reason": reason}

    lines = output.split("\n")
    for line in lines:
        if line.lower().startswith("decision:"):
            decision = line.split(":", 1)[1].strip()
        elif line.lower().startswith("reason:"):
            reason = line.split(":", 1)[1].strip()

    return {
        "decision": decision,
        "reason": reason
    }