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
MODEL_NAME = "gemma3:1b"  # or "gemma:2b" depending on your setup


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
    - If knowledge is insufficient, make the safest general ML decision
    - Be concise

    Output:

    Decision: <specific action>
    Reason: <clear justification>
    """

    return prompt.strip()


def call_llm_with_fallback(column_data: Dict, retriever, model_name: str = MODEL_NAME) -> str:
    """
    Call Ollama LLM with FAISS retrieval fallback.
    If Ollama fails or returns empty, fallback to FAISS retrieval.
    """
    # Retrieve docs once
    retrieved_docs = retriever.retrieve(column_data.get("column", ""))
    prompt = build_prompt(column_data, retrieved_docs)

    try:
        logger.info(f"Querying Ollama for column '{column_data.get('column')}'")

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )

        if response.status_code != 200:
            logger.warning(f"Ollama returned {response.status_code}. Falling back to FAISS.")
            return "\n".join(retrieved_docs) or "No relevant knowledge found."

        output = response.json().get("response", "")
        if not output.strip():
            logger.warning("Ollama returned empty response. Using FAISS fallback.")
            return "\n".join(retrieved_docs) or "No relevant knowledge found."

        logger.info("LLM response received successfully")
        return output.strip()

    except Exception as e:
        logger.warning(f"Ollama call failed: {e}. Using FAISS retrieval fallback.")
        return "\n".join(retrieved_docs) or "No relevant knowledge found."