"""
RAG Preprocessing Pipeline – Developer Guide

This module runs a full preprocessing pipeline using EDA reports, LLMs,
knowledge retrieval, and a critic module to produce safe column-level decisions.

--------------------------------------------------------------------------------
Pipeline Flow:
--------------------------------------------------------------------------------
1. Load report text file
2. Parse report → column metadata
3. Load knowledge base documents
4. Initialize FAISS retriever
5. For each column:
    - Apply guardrails (skip/no-action/drop)
    - Build retrieval query
    - Retrieve knowledge
    - Build LLM prompt
    - Call LLM → initial decision
    - Validate/refine decision with critic
    - Fallback to LLM output if critic empty
6. Aggregate results per column

--------------------------------------------------------------------------------
Key Functions:
--------------------------------------------------------------------------------
run_pipeline(report_path: str) -> List[Dict]
    - Main pipeline execution
    - Returns list of dicts: column, query, decision (or error info)

safe_run_pipeline(report_path: str) -> List[Dict]
    - Wrapper for run_pipeline with exception handling
    - Returns structured error info if pipeline fails

load_text_file(path: str) -> str
    - Reads text file content

apply_guardrails(column_data: dict) -> Optional[str]
    - Checks missing % to skip or drop columns
    - Returns guardrail decision string or None

load_knowledge_by_route(route: str) -> List[str]
    - Loads knowledge base documents by route
    - TXT and PDF support

run_query(query: str) -> str
    - Routes query, retrieves docs, calls LLM
    - Returns text answer

--------------------------------------------------------------------------------
Usage Notes:
--------------------------------------------------------------------------------
from rag.pipeline import run_pipeline

results = run_pipeline("reports.txt")

for r in results:
    print(f"{r['column']}: {r['decision']}")
"""

from rag.parser import parse_report,normalize_decision
from rag.query import build_query
from rag.retriever import FaissRetriever
from rag.reasoning import build_prompt, call_llm_with_fallback,parse_llm_output
from rag.critic import review_decision
from rag.router import route_query
from rag.knowledge import load_txt, load_pdf
from core.logger import get_logger
import os
from core.exceptions import ParserError
from rag.knowledge import load_chunks
from rag.retriever import get_retriever
from core.config import ENGINE
from rag.gemini_client import call_gemini

docs = load_chunks()
logger = get_logger(__name__)
from rag.knowledge import ensure_knowledge_ready
ollama_model = None  # because your call_llm function handles the requests itself
ensure_knowledge_ready()
docs = load_chunks()

# -----------------------------
# LOADERS
# -----------------------------

def load_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()



# -----------------------------
# GUARDRAILS
# -----------------------------

def apply_guardrails(column_data):
    missing = column_data.get("missing_percent", 0)

    if missing == 0:
        return  {
    "hint": "no_missing_values",
    "reason": "Column has no missing values"
}

    if missing > 70:
        return "Decision: Drop column\nReason: Too many missing values (>70%)"

    return None


def load_report(report_path: str) -> str:
    if not os.path.exists(report_path):
        raise ParserError(f"Report file not found: {report_path}")

    try:
        report_text = load_report(report_path)

    except Exception as e:
        raise ParserError(f"Failed to read report: {str(e)}")
    
def process_column(col: dict) -> bool:
    """
    Decide whether a column needs LLM processing
    """

    missing = col.get("missing_percent", 0)
    dtype = col.get("dtype")

    # 🚫 Skip clean numeric columns
    if missing == 0 and dtype == "numeric":
        return False

    # 🚫 Skip low-value categoricals (optional tweak later)
    if missing == 0 and dtype == "categorical":
        return False

    return True


# -----------------------------
# MAIN PIPELINE
# -----------------------------
def run_pipeline(report_path: str):
    try:
        logger.info("Starting pipeline")

        # 1. Load report
        report_text = load_text_file(report_path)

        # 2. Parse
        parsed_columns = parse_report(report_text)

        # 3. Load knowledge
        docs = load_chunks()

        # 4. Retriever
        retriever = get_retriever("./models/minilm", docs)

        final_results = []

        # 5. Process each column
        for column_data in parsed_columns:
            col_name = column_data.get("column", "unknown")
            logger.info(f"[PIPELINE] Processing column: {col_name}")

            try:
                # 0. Pre-filter
                missing = column_data.get("missing_percent", 0)

                # Only skip truly useless columns (optional)
                if missing == 0 and column_data.get("dtype") == "numeric":
                    logger.debug(f"[LIGHT PASS] {col_name} has no missing values")

                # 1. Guardrails
                guardrail = apply_guardrails(column_data)

                if guardrail:
                    logger.info(f"[GUARDRAIL] Hint for {col_name}: {guardrail}")
                    final_results.append({
                        "column": col_name,
                        "decision": guardrail
                    })
                    # continue

                # 2. Query
                query = build_query(column_data)
                logger.debug(f"[QUERY] {col_name}: {query}")

                # 3. Retrieve
                retrieved_docs = retriever.retrieve(query, k=2)

                if not retrieved_docs:
                    logger.warning(f"[RETRIEVAL] No docs found for {col_name}")

                # 4. Prompt
                prompt = build_prompt(column_data, retrieved_docs)

                # 5. LLM
                if ENGINE == "gemini":
                    raw_output = call_gemini(column_data, report_text)
                else:
                    raw_output = call_llm_with_fallback(...)


                if not decision_raw or not decision_raw.strip():
                    logger.warning(f"[GEMINI] Empty response for {col_name}")
                    decision_raw = "Decision: Unable to determine\nReason: Empty LLM output"

                # 6. Parse
                parsed_output = parse_llm_output(decision_raw)

                if not parsed_output.get("decision"):
                    logger.warning(f"[PARSE] Failed for {col_name}")
                    parsed_output = {
                        "decision": decision_raw,
                        "reason": "Fallback: parsing failed"
                    }

                parsed_output["decision"] = normalize_decision(parsed_output["decision"])

                # 7. Critic
                final_decision = review_decision(
                    column_data,
                    retrieved_docs,
                    parsed_output
                )

                if not final_decision or not final_decision.strip():
                    logger.warning(f"[CRITIC] Empty decision for {col_name}")
                    final_decision = parsed_output["decision"]

                # 8. Save result
                final_results.append({
                    "column": col_name,
                    "query": query,
                    "decision": final_decision
                })

            except Exception as inner_e:
                logger.exception(f"[ERROR] Failed processing {col_name}: {str(inner_e)}")

                final_results.append({
                    "column": col_name,
                    "decision": "Error",
                    "error": str(inner_e)
                })
        
        # 9. Dataset-level fallback (AFTER LOOP)
        if not final_results:
            logger.info("[PIPELINE] No actionable columns found")
            final_results.append({
                "column": "Dataset",
                "decision": "No preprocessing required\nReason: No missing values or issues detected"
            })

        logger.info("[PIPELINE] Completed successfully")
        return final_results or []

    except Exception as e:
        logger.exception(f"Pipeline failed: {str(e)}")
        raise RuntimeError("Pipeline execution failed")

def load_knowledge_by_route(route: str):

    docs = []

    if route == "eda":
        docs += load_txt("data/eda.txt")

    elif route == "ml":
        docs += load_txt("data/general.txt")

    else:
        docs += load_txt("data/general.txt")

    # 🔥 Load ALL PDFs
    pdf_dir = "data/pdfs"

    if os.path.exists(pdf_dir):
        for file in os.listdir(pdf_dir):
            if file.endswith(".pdf"):
                docs += load_pdf(os.path.join(pdf_dir, file))

    return docs


def run_query(query: str) -> str:
    try:
        logger.info(f"Query received: {query}")

        from rag.knowledge import ensure_knowledge_ready, load_chunks
        ensure_knowledge_ready()
        docs = load_chunks()

        from rag.retriever import get_retriever
        retriever = get_retriever("./models/minilm", docs)

        results = retriever.retrieve(query, k=3)

        if not results:
            return "No relevant knowledge found."

        return "\n".join(results)

    except Exception as e:
        logger.exception("Query failed")

        return "System temporarily failed to process query."

def safe_run_pipeline(report_path: str):
    try:
        # print("\n[DEBUG] Pipeline started")
        return run_pipeline(report_path)

    except FileNotFoundError:
        return [{
            "column": "N/A",
            "decision": "Error: Report file not found\nReason: Please provide valid path"
        }]

    except Exception as e:
        return [{
            "column": "N/A",
            "decision": f"Error: Pipeline failed\nReason: {str(e)}"
        }]

# -----------------------------
# 🖥️ CLI ENTRYPOINT
# -----------------------------

if __name__ == "__main__":

    results = run_pipeline("reports.txt")

    for r in results:
        print("\n==========================")
        print(f"Column: {r['column']}")
        print("--------------------------")

        for line in r["decision"].split("\n"):
            print(line.strip())