from parser.eda_parser import parse_report
from retriever.query_builder import build_query
from retriever.faiss_retriever import FaissRetriever
from agent.reasoning_agent import build_prompt, call_llm
from agent.critic_agent import review_decision

from core.logger import get_logger

logger = get_logger(__name__)


def apply_guardrails(column_data):
    """
    Deterministic rules before LLM (robust to bad parser values)
    """

    raw_missing = column_data.get("missing_percent")

    # 🔥 SAFE conversion (handles None, "", "0", etc.)
    try:
        missing = float(raw_missing)
    except (TypeError, ValueError):
        logger.warning(f"Invalid missing_percent '{raw_missing}', defaulting to 0")
        missing = 0.0

    # ✅ Rule 1: No missing → no action
    if missing <= 0:
        return "Decision: No action needed\nReason: Column has no missing values"

    # ✅ Rule 2: Extremely high missing → drop
    if missing > 70:
        return "Decision: Drop column\nReason: Too many missing values (>70%)"

    return None


def run_pipeline(report_path: str):
    """
    Full RAG pipeline:
    Report → Parse → Query → Retrieve → Reason → Critic → Output
    """

    try:
        logger.info("Starting pipeline")

        # 1. Load report
        with open(report_path, "r", encoding="utf-8", errors="ignore") as f:
            report_text = f.read()

        # 2. Parse report
        parsed_columns = parse_report(report_text)

        # 3. Load knowledge base
        with open("data/knowledge.txt", "r", encoding="utf-8", errors="ignore") as f:
            docs = [line.strip() for line in f.readlines() if line.strip()]

        # 4. Initialize retriever (ONLY once)
        retriever = FaissRetriever("./models/minilm", docs)

        final_results = []

        # 5. Process each column
        for column_data in parsed_columns:

            col_name = column_data.get("column", "unknown")
            logger.info(f"Processing column: {col_name}")

            # 🔥 Step 1: Guardrails FIRST
            guardrail_decision = apply_guardrails(column_data)

            if guardrail_decision:
                final_results.append({
                    "column": col_name,
                    "decision": guardrail_decision
                })
                continue  # 🔥 DO NOT go to LLM

            # 🔹 Step 2: Build query
            query = build_query(column_data)

            # 🔹 Step 3: Retrieve knowledge
            retrieved_docs = retriever.retrieve(query, k=2)

            # 🔹 Step 4: Build prompt
            prompt = build_prompt(column_data, retrieved_docs)

            # 🔹 Step 5: LLM reasoning
            decision = call_llm(prompt)

            # 🔹 Step 6: Critic validation
            final_decision = review_decision(column_data, retrieved_docs, decision)

            # 🔥 Safety fallback
            if not final_decision or not final_decision.strip():
                logger.warning(f"Empty critic output for {col_name}, using fallback")
                final_decision = decision or "Decision: Unable to determine\nReason: LLM returned empty response"

            final_results.append({
                "column": col_name,
                "query": query,
                "decision": final_decision
            })

        logger.info("Pipeline completed successfully")
        return final_results

    except Exception as e:
        logger.exception(f"Pipeline failed: {str(e)}")
        raise


# -----------------------------
# 🖥️ CLI ENTRYPOINT
# -----------------------------
if __name__ == "__main__":

    results = run_pipeline("reports.txt")

    for r in results:
        print("\n==========================")
        print(f"Column: {r['column']}")
        print("--------------------------")

        decision_text = r.get("decision", "")

        for line in decision_text.split("\n"):
            print(line.strip())


