# """
# RAG Preprocessing Pipeline – Developer Guide

# This module runs a full preprocessing pipeline using EDA reports, LLMs,
# knowledge retrieval, and a critic module to produce safe column-level decisions.

# --------------------------------------------------------------------------------
# Pipeline Flow:
# --------------------------------------------------------------------------------
# 1. Load report text file
# 2. Parse report → column metadata
# 3. Load knowledge base documents
# 4. Initialize FAISS retriever
# 5. For each column:
#     - Apply guardrails (skip/no-action/drop)
#     - Build retrieval query
#     - Retrieve knowledge
#     - Build LLM prompt
#     - Call LLM → initial decision
#     - Validate/refine decision with critic
#     - Fallback to LLM output if critic empty
# 6. Aggregate results per column

# --------------------------------------------------------------------------------
# Key Functions:
# --------------------------------------------------------------------------------
# run_pipeline(report_path: str) -> List[Dict]
#     - Main pipeline execution
#     - Returns list of dicts: column, query, decision (or error info)

# safe_run_pipeline(report_path: str) -> List[Dict]
#     - Wrapper for run_pipeline with exception handling
#     - Returns structured error info if pipeline fails

# load_text_file(path: str) -> str
#     - Reads text file content

# apply_guardrails(column_data: dict) -> Optional[str]
#     - Checks missing % to skip or drop columns
#     - Returns guardrail decision string or None

# load_knowledge_by_route(route: str) -> List[str]
#     - Loads knowledge base documents by route
#     - TXT and PDF support

# run_query(query: str) -> str
#     - Routes query, retrieves docs, calls LLM
#     - Returns text answer

# --------------------------------------------------------------------------------
# Usage Notes:
# --------------------------------------------------------------------------------
# from rag.pipeline import run_pipeline

# results = run_pipeline("reports.txt")

# for r in results:
#     print(f"{r['column']}: {r['decision']}")
# """

# from rag.parser import parse_report,normalize_decision
# from rag.query import build_query
# from rag.retriever import FaissRetriever
# from rag.reasoning import build_prompt, parse_llm_output
# from rag.critic import review_decision
# from rag.router import route_query
# from rag.knowledge import load_txt
# from core.logger import get_logger
# import os
# from core.exceptions import ParserError
# from rag.knowledge import load_chunks
# from rag.retriever import get_retriever
# from sentence_transformers import SentenceTransformer

# # model = SentenceTransformer("all-MiniLM-L6-v2")

# docs = load_chunks()
# logger = get_logger(__name__)
# from rag.knowledge import ensure_knowledge_ready
# ollama_model = None  # because your call_llm function handles the requests itself
# ensure_knowledge_ready()
# retriever = get_retriever("BAAI/bge-small-en-v1.5", docs)
# # -----------------------------
# # LOADERS
# # -----------------------------

# def load_text_file(path: str) -> str:
#     with open(path, "r", encoding="utf-8", errors="ignore") as f:
#         return f.read()



# # -----------------------------
# # GUARDRAILS
# # -----------------------------

# def apply_guardrails(column_data):
#     missing = column_data.get("missing_percent", 0)

#     if missing == 0:
#         return  {
#     "hint": "no_missing_values",
#     "reason": "Column has no missing values"}

#     if missing > 70:
#         return "Decision: Drop column\nReason: Too many missing values (>70%)"

#     return None


# # def load_report(report_path: str) -> str:
#     if not os.path.exists(report_path):
#         raise ParserError(f"Report file not found: {report_path}")

#     try:
#         report_text = load_report(report_path)

#     except Exception as e:
#         raise ParserError(f"Failed to read report: {str(e)}")


# def process_column(col: dict) -> bool:
#     """
#     Decide whether a column needs LLM processing
#     """

#     missing = col.get("missing_percent", 0)
#     dtype = col.get("dtype")

#     # 🚫 Skip clean numeric columns
#     if missing == 0 and dtype == "numeric":
#         return False

#     # 🚫 Skip low-value categoricals (optional tweak later)
#     if missing == 0 and dtype == "categorical":
#         return False

#     return True


# # -----------------------------
# # MAIN PIPELINE
# # -----------------------------
# def run_pipeline(report_path: str):
#     try:
#         logger.info("Starting pipeline")

#         # 1. Load report
#         report_text = load_text_file(report_path)

#         # 2. Parse
#         parsed_columns = parse_report(report_text)

#         # 3. Load knowledge
#         docs = load_chunks()
#         final_results = []

#         # 4. Process each column
#         for column_data in parsed_columns:
#             col_name = column_data.get("column", "unknown")
#             logger.info(f"[PIPELINE] Processing column: {col_name}")

#             try:
#                 # 0. Pre-filter
#                 missing = column_data.get("missing_percent", 0)

#                 # Only skip truly useless columns (optional)
#                 if missing == 0 and column_data.get("dtype") == "numeric":
#                     logger.debug(f"[LIGHT PASS] {col_name} has no missing values")

#                 # 1. Guardrails
#                 guardrail = apply_guardrails(column_data)

#                 if guardrail:
#                     logger.info(f"[GUARDRAIL] Hint for {col_name}: {guardrail}")

#                     hint = guardrail.get("hint", "")

#                     # ONLY hard stop for real actions
#                     if "drop" in hint:
#                         decision_raw = normalize_decision(hint)

#                         final_results.append({
#                             "column": col_name,
#                             "decision": decision_raw,
#                             "reason": guardrail.get("reason", "")
#                         })

#                         logger.debug(f"[PIPELINE] Hard guardrail applied for {col_name}")
#                         continue

#                     # otherwise just log and continue pipeline
#                     logger.debug(f"[PIPELINE] Guardrail is informational, continuing...")

#                 # 2. Query
#                 query = build_query(column_data)
#                 logger.debug(f"[QUERY] {col_name}: {query}")

#                 # 3. Retrieve
#                 # retriever = get_retriever("./models/minilm", docs)
#                 retrieved_docs = retriever.retrieve(query, k=2)
#                 # print("=====================",query,"===================")

#                 if not retrieved_docs:
#                     logger.warning(f"[RETRIEVAL] No docs found for {col_name}")

#                 # 4. Prompt
#                 prompt = build_prompt(column_data, retrieved_docs)

#                 # 5. LLM
#                 decision_raw = "Decision: UNKNOWN\nReason: No rule applied" # deterministic, local only

#                 if not decision_raw.strip():
#                     logger.warning(f"[LLM] Empty response for {col_name}")
#                     decision_raw = "Decision: UNKNOWN\nReason: Empty LLM output"

#                 # 6. Parse
#                 parsed_output = parse_llm_output(decision_raw)

#                 if not parsed_output.get("decision"):
#                     logger.warning(f"[PARSE] Failed for {col_name}")
#                     parsed_output = {
#                         "decision": decision_raw,
#                         "reason": "Fallback: parsing failed"
#                     }

#                 parsed_output["decision"] = normalize_decision(parsed_output["decision"])
#                 # logger.debug(f"[TYPE CHECK BEFORE CRITIC] {col_name}: {type(parsed_output)} → {parsed_output}")
#                 # 7. Critic
#                 print(type(parsed_output["decision"]))  # MUST be str
#                 final_decision = review_decision(
#                     column_data,
#                     retrieved_docs,
#                     parsed_output["decision"],
#                     retriever
#                 )

#                 if isinstance(final_decision, dict):
#                     final_decision = final_decision.get("decision", str(final_decision))

#                 if not final_decision or not str(final_decision).strip():
#                     logger.warning(f"[CRITIC] Empty decision for {col_name}")
#                     final_decision = parsed_output.get("decision", "UNKNOWN")

#                 # 8. Save result
#                 final_results.append({
#                     "column": col_name,
#                     "query": query,
#                     "decision": final_decision
#                 })

#             except Exception as inner_e:
#                 logger.exception(f"[ERROR] Failed processing {col_name}: {str(inner_e)}")

#                 final_results.append({
#                     "column": col_name,
#                     "decision": "Error",
#                     "error": str(inner_e)
#                 })
        
#         # 9. Dataset-level fallback (AFTER LOOP)
#         if not final_results:
#             logger.info("[PIPELINE] No actionable columns found")
#             final_results.append({
#                 "column": "Dataset",
#                 "decision": "No preprocessing required\nReason: No missing values or issues detected"
#             })

#         logger.info("[PIPELINE] Completed successfully")
#         return final_results or []

#     except Exception as e:
#         logger.exception(f"Pipeline failed: {str(e)}")
#         raise RuntimeError("Pipeline execution failed")

# def load_knowledge_by_route(route: str):

#     docs = []

#     if route == "eda":
#         docs += load_txt("data/eda.txt")

#     elif route == "ml":
#         docs += load_txt("data/general.txt")

#     else:
#         docs += load_txt("data/general.txt")

#     # 🔥 Load ALL PDFs
#     pdf_dir = "data/pdfs"

#     # if os.path.exists(pdf_dir):
#     #     for file in os.listdir(pdf_dir):
#     #         if file.endswith(".pdf"):
#     #             docs += load_pdf(os.path.join(pdf_dir, file))

#     return docs


# def run_query(query: str) -> str:
#     try:
#         logger.info(f"Query received: {query}")
#         ensure_knowledge_ready()
#         docs = load_chunks()


#         results = retriever.retrieve(query, k=3)

#         if not results:
#             return "No relevant knowledge found."

#         return "\n".join(results)

#     except Exception as e:
#         logger.exception("Query failed")

#         return "System temporarily failed to process query."

# def safe_run_pipeline(report_path: str):
#     try:
#         # print("\n[DEBUG] Pipeline started")
#         return run_pipeline(report_path)

#     except FileNotFoundError:
#         return [{
#             "column": "N/A",
#             "decision": "Error: Report file not found\nReason: Please provide valid path"
#         }]

#     except Exception as e:
#         return [{
#             "column": "N/A",
#             "decision": f"Error: Pipeline failed\nReason: {str(e)}"
#         }]

# # -----------------------------
# # 🖥️ CLI ENTRYPOINT
# # -----------------------------

# if __name__ == "__main__":

#     results = run_pipeline("reports.txt")

#     for r in results:
#         print("\n==========================")
#         print(f"Column: {r['column']}")
#         print("--------------------------")

#         for line in r["decision"].split("\n"):
#             print(line.strip())














































# import os
# from core.logger import get_logger
# from core.exceptions import ParserError

# # RAG components
# from rag.parser import parse_report, normalize_decision
# from rag.query import build_query
# from rag.retriever import get_retriever
# from rag.reasoning import build_prompt, parse_llm_output
# from rag.critic import review_decision
# from rag.knowledge import ensure_knowledge_ready, load_chunks
# from rag.reasoning import call_llm_with_fallback

# logger = get_logger(__name__)

# # -----------------------------
# # GLOBAL INITIALIZATION
# # -----------------------------
# # 1. First, check if PDF/Text hashes changed and rebuild if necessary
# ensure_knowledge_ready()

# # 2. Load the actual chunks (from meta.pkl)
# docs = load_chunks()

# # 3. Get the retriever singleton (Automatically loads faiss.index)
# retriever = get_retriever("BAAI/bge-small-en-v1.5", docs)

# # -----------------------------
# # HELPERS
# # -----------------------------

# def load_text_file(path: str) -> str:
#     if not os.path.exists(path):
#         raise ParserError(f"Report file not found: {path}")
#     with open(path, "r", encoding="utf-8", errors="ignore") as f:
#         return f.read()

# def apply_guardrails(column_data):
#     missing = column_data.get("missing_percent", 0)
#     if missing == 0:
#         return {"hint": "no_missing_values", "reason": "Column has no missing values"}
#     if missing > 70:
#         return {"hint": "drop_column", "reason": "Too many missing values (>70%)"}
#     return None

# def should_process_column(col: dict) -> bool:
#     """Decide if a column actually needs LLM reasoning (Your original logic)"""
#     missing = col.get("missing_percent", 0)
#     dtype = col.get("dtype")
#     # Skip clean numeric/categorical columns to save time/compute
#     if missing == 0:
#         return False
#     return True

# # -----------------------------
# # MAIN PIPELINE LOGIC
# # -----------------------------
# def run_pipeline(report_path: str):
#     try:
#         logger.info("[PIPELINE] Starting orchestration for: %s", report_path)

#         # 1. Load & Parse Report
#         report_text = load_text_file(report_path)
#         parsed_columns = parse_report(report_text)
        
#         final_results = []

#         # 2. Process every column (No more should_process pre-filter)
#         for column_data in parsed_columns:
#             col_name = column_data.get("column", "unknown")
#             logger.info(f"[PIPELINE] Processing column: {col_name}")
#             column_data.pop('retriever', None)
#             try:
#                 # A. Guardrail Check (Hard-stop for drops/logic stops)
#                 # This now acts as your primary filter
#                 guardrail = apply_guardrails(column_data)
#                 if guardrail:
#                     hint = guardrail.get("hint", "")
#                     if "drop" in hint:
#                         final_results.append({
#                             "column": col_name,
#                             "decision": normalize_decision(hint),
#                             "reason": guardrail.get("reason", "Guardrail trigger")
#                         })
#                         logger.info(f"[GUARDRAIL] Dropping {col_name}")
#                         continue
                    
#                     # If guardrail says 'no_missing_values', we still continue 
#                     # because your "better functions" will handle the nuance.
#                     logger.debug(f"[GUARDRAIL] Informational hint: {hint}")

#                 # B. Query & Retrieve
#                 query = build_query(column_data)
#                 retrieved_docs = retriever.retrieve(query, k=2)

#                 # 3. FIX: Send only the prompt (string) to the LLM
#                 prompt = build_prompt(column_data, retrieved_docs)
#                 decision_raw = call_llm_with_fallback(prompt) # Inside call_llm, json is now just {model, prompt, stream}

#                 # D. Parse & Normalize
#                 parsed_output = parse_llm_output(decision_raw)
#                 norm_decision = normalize_decision(parsed_output.get("decision", "keep"))

#                 # E. Critic Review
#                 final_decision = review_decision(
#                     column_data,
#                     retrieved_docs,
#                     norm_decision,
#                     retriever
#                 )

#                 if isinstance(final_decision, dict):
#                     final_decision = final_decision.get("decision", norm_decision)

#                 # F. Finalizing Result
#                 final_results.append({
#                     "column": col_name,
#                     "query": query,
#                     "decision": final_decision,
#                     "reason": parsed_output.get("reason", "Decision derived from RAG context")
#                 })

#             except Exception as col_err:
#                 logger.error(f"[PIPELINE] Column {col_name} failed: {col_err}")
#                 final_results.append({
#                     "column": col_name, 
#                     "decision": "Error", 
#                     "reason": str(col_err)
#                 })

#         # 3. Safety Fallback
#         if not final_results:
#             logger.warning("[PIPELINE] No results generated after processing all columns.")
#             final_results.append({
#                 "column": "System",
#                 "decision": "None",
#                 "reason": "Input report was empty or parsing failed."
#             })

#         return final_results

#     except Exception as e:
#         logger.exception(f"Pipeline Fatal Error: {e}")
#         raise

# # -----------------------------
# # CLI ENTRYPOINT
# # -----------------------------
# if __name__ == "__main__":
#     # This runs when you call 'python pipeline.py'
#     report_to_process = "reports.txt"
#     if os.path.exists(report_to_process):
#         results = run_pipeline(report_to_process)
#         for r in results:
#             print(f"\nCOLUMN: {r['column']}\nDECISION: {r['decision']}")
#     else:
#         print(f"Error: {report_to_process} not found.")




























































import json
from tabulate import tabulate # Optional: pip install tabulate for pretty printing
import os
from typing import List, Dict
from core.logger import get_logger
from core.exceptions import ParserError

# RAG Component Imports
from rag.parser import parse_report, normalize_decision
from rag.query import build_query
from rag.retriever import get_retriever
from rag.reasoning import build_prompt, call_llm_with_fallback, parse_llm_output
from rag.critic import review_decision
from rag.knowledge import ensure_knowledge_ready, load_chunks

logger = get_logger(__name__)

# -----------------------------
# GLOBAL INIT
# -----------------------------
# Ensure index is ready and singleton retriever is active before any calls
ensure_knowledge_ready()
retriever = get_retriever(load_chunks())

# -----------------------------
# CORE LOGIC UNITS
# -----------------------------

def load_text_file(path: str) -> str:
    """Read the senior analyst report."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Report file missing: {path}")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

# def should_process_column(col_data: Dict) -> bool:
#     dtype = str(col_data.get("dtype", "unknown")).lower()
#     missing = col_data.get("missing_percent", 0)
#     skew = abs(col_data.get("skew", 0))
#     unique_count = col_data.get("unique_count", 999) # Add this to your parser!

#     if missing > 0: return True
#     if dtype in ["object", "category", "string"]: return True
    
#     # NEW: If it's numeric but has very few unique values, 
#     # it's likely a categorical encoded as an int. PROCESS IT.
#     if unique_count < 10: 
#         return True

#     if dtype in ["numeric", "int", "float"] and skew > 1.0: return True

#     return False


def is_clean_numeric(col_data: Dict) -> bool:
    """
    Returns True ONLY if the column is numeric, complete, and balanced.
    If it fails any check, it returns False and triggers the RAG flow.
    """
    dtype = str(col_data.get("dtype", "unknown")).lower()
    missing = col_data.get("missing_percent", 0)
    # Use 0 as default for skew if missing
    skew = abs(col_data.get("skew", 0))
    # If your parser doesn't provide unique_count yet, default to 999
    unique_count = col_data.get("unique_count", 999)

    # 1. Numeric check
    is_num = any(n in dtype for n in ['numeric', 'int', 'float'])
    
    # 2. Skip criteria: Numeric AND no missing AND low skew AND high cardinality
    if is_num and missing == 0 and skew < 1.0 and unique_count >= 10:
        return True
        
    return False


def apply_guardrails(column_data: Dict) -> Dict:
    """
    Hard-coded statistical safety checks.
    Returns a hint dict if a guardrail triggers, else None.
    """
    missing = column_data.get("missing_percent", 0)
    
    if missing > 70:
        return {
            "hint": "drop_column", 
            "reason": f"Missingness ({missing}%) exceeds 70% threshold."
        }
    
    if missing == 0:
        # We don't stop here, but we pass a hint to the LLM via the pipeline
        return {"hint": "no_missing_values", "reason": "Data is complete."}
        
    return None



def run_query(query: str) -> str:
    try:
        logger.info(f"Query received: {query}")
        ensure_knowledge_ready()
        docs = load_chunks()


        results = retriever.retrieve(query, k=3)

        if not results:
            return "No relevant knowledge found."

        return "\n".join(results)

    except Exception as e:
        logger.exception("Query failed")

        return "System temporarily failed to process query."

def safe_run_pipeline(report_path: str):
    try:
        print("\n[DEBUG] Pipeline started")
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
def run_pipeline(report_path: str) -> List[Dict]:
    """
    Refactored Orchestrator: 
    No more 'guessing'—if it's not perfect, it goes to RAG.
    """
    logger.info(f"🚀 [PIPELINE] Starting orchestration: {report_path}")
    
    report_text = load_text_file(report_path)
    parsed_columns = parse_report(report_text)
    logger.info(f"[PARSER] Total columns parsed: {len(parsed_columns)}")
    logger.debug(f"[PARSER] Columns: {[c['column'] for c in parsed_columns]}")
    
    final_results = []

    for col_data in parsed_columns:
        name = col_data.get("column", "unknown")
        
        try:
            # 1. Check if we can skip
            if is_clean_numeric(col_data) and not col_data.get("high_target_corr", False):
                final_results.append({
                    "column": name,
                    "decision": "Keep (No Action)",
                    "reason": "Meets baseline: Numeric, no missing values, low skew."
                })
                continue 

            # 2. Generator Logic
            query = build_query(col_data)
            context = retriever.retrieve(query, k=3)
            prompt = build_prompt(col_data, context)
            
            raw_output = call_llm_with_fallback(prompt)
            parsed_gen = parse_llm_output(raw_output)

            # Create the initial record
            record = {
                "column": name,
                "decision": parsed_gen.get("decision", "Keep"),
                "reason": parsed_gen.get("reason", "RAG reasoning.")
            }

            # 3. Critic Logic (The Git Patch)
            # This edits 'record' directly
            review_decision(record, context)

            # 4. Save the single, merged result
            final_results.append(record)

        except Exception as e:
            logger.error(f"Failed {name}: {e}")
    return final_results


