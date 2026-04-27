"""
Pipeline orchestrator.

Connects parser -> query builder -> retriever -> reasoner -> critic to turn
a parsed EDA report into a list of column-level preprocessing decisions.

Initialization is fully lazy: importing this module is cheap and never
touches the embedding model, the FAISS index, or the LLM. The first call to
`run_pipeline()` / `run_query()` triggers the heavy machinery.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from core.exceptions import ParserError, RetrievalError
from core.logger import get_logger
from rag.critic import review_decision
from rag.knowledge import ensure_knowledge_ready, load_chunks
from rag.parser import parse_report
from rag.query import build_query
from rag.reasoning import build_prompt, call_llm_with_fallback, parse_llm_output

logger = get_logger(__name__)

_retriever = None  # set on first call


# -----------------------------------------------------------------------------
def _get_retriever():
    """Lazy retriever bootstrap. Importing FAISS only when needed."""
    global _retriever  # noqa: PLW0603
    if _retriever is None:
        ensure_knowledge_ready()
        from rag.retriever import get_retriever  # noqa: WPS433
        _retriever = get_retriever(load_chunks())
    return _retriever


def reset_pipeline() -> None:
    """Test helper — drop cached retriever so the next call rebuilds."""
    global _retriever  # noqa: PLW0603
    _retriever = None


# --- Helpers -----------------------------------------------------------------
def load_text_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Report file missing: {path}")
    return p.read_text(encoding="utf-8", errors="ignore")


def is_clean_numeric(col_data: Dict) -> bool:
    """A column we can safely 'Keep' without invoking the LLM."""
    dtype = str(col_data.get("dtype", "unknown")).lower()
    missing = col_data.get("missing_percent", 0) or 0
    skew = abs(col_data.get("skew", 0) or 0)
    unique_count = col_data.get("unique_count", 999)
    is_num = any(n in dtype for n in ("numeric", "int", "float"))
    return bool(is_num and missing == 0 and skew < 1.0 and unique_count >= 10)


def apply_guardrails(col_data: Dict) -> Optional[Dict[str, str]]:
    """Hard-coded statistical short-circuits (drop / keep hint)."""
    missing = col_data.get("missing_percent", 0) or 0
    if missing > 70:
        return {"hint": "drop_column",
                "reason": f"Missingness ({missing}%) exceeds 70% threshold."}
    if missing == 0:
        return {"hint": "no_missing_values", "reason": "Data is complete."}
    return None


# --- Public API --------------------------------------------------------------
def run_pipeline(report_path: str, retriever=None) -> List[Dict]:
    """Process a parsed EDA report into per-column decisions.

    `retriever` may be passed (e.g. from tests) to avoid touching FAISS.
    """
    logger.info("[PIPELINE] Starting orchestration: %s", report_path)
    report_text = load_text_file(report_path)
    parsed_columns = parse_report(report_text)
    logger.info("[PIPELINE] Parsed %d columns", len(parsed_columns))

    if not parsed_columns:
        raise ParserError(f"No columns parsed from {report_path}")

    retr = retriever or _get_retriever()
    final_results: List[Dict] = []

    for col_data in parsed_columns:
        name = col_data.get("column", "unknown")
        try:
            # Short-circuit clean numeric columns.
            if is_clean_numeric(col_data) and not col_data.get("high_target_corr"):
                final_results.append({
                    "column": name,
                    "decision": "Keep (No Action)",
                    "reason": "Baseline met: numeric, no missing, low skew.",
                })
                continue

            guardrail = apply_guardrails(col_data)
            if guardrail and guardrail["hint"] == "drop_column":
                final_results.append({
                    "column": name,
                    "decision": "drop_column",
                    "reason": guardrail["reason"],
                })
                continue

            query = build_query(col_data)
            context = retr.retrieve(query, k=3) if retr else []
            prompt = build_prompt(col_data, context)
            raw_output = call_llm_with_fallback(prompt)
            parsed = parse_llm_output(raw_output)

            record = {
                "column": name,
                "dtype": col_data.get("dtype"),
                "decision": parsed.get("decision", "keep"),
                "reason": parsed.get("reason", "RAG reasoning."),
            }
            review_decision(record, context)
            final_results.append(record)
        except Exception as exc:  # noqa: BLE001
            logger.error("[PIPELINE] Column %s failed: %s", name, exc)
            final_results.append({
                "column": name, "decision": "Error", "reason": str(exc),
            })

    return final_results


def safe_run_pipeline(report_path: str) -> List[Dict]:
    """Wrapper that always returns a list (never raises)."""
    try:
        return run_pipeline(report_path)
    except FileNotFoundError as exc:
        return [{"column": "N/A", "decision": "Error",
                 "reason": f"Report not found: {exc}"}]
    except (ParserError, RetrievalError) as exc:
        return [{"column": "N/A", "decision": "Error", "reason": str(exc)}]
    except Exception as exc:  # noqa: BLE001
        return [{"column": "N/A", "decision": "Error",
                 "reason": f"Pipeline failure: {exc}"}]


def run_query(question: str, retriever=None) -> str:
    """Free-form QA against the knowledge base."""
    if not question or not question.strip():
        return "Empty question."
    try:
        retr = retriever or _get_retriever()
        results = retr.retrieve(question, k=3) if retr else []
        if not results:
            return "No relevant knowledge found in the index."
        return "\n\n---\n\n".join(results)
    except RetrievalError as exc:
        return f"Knowledge base unavailable: {exc}"
    except Exception as exc:  # noqa: BLE001
        logger.exception("[PIPELINE] Query failed")
        return f"System error while answering: {exc}"
