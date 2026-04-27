"""
Integration test for the full pipeline.

Uses a stub retriever (so FAISS / sentence-transformers are never loaded)
and the graceful-fallback LLM path (so Ollama isn't required either). This
mirrors how the user can swap in their real Gemma3 + grounded MD/PKL files
without changing any other code.
"""
from __future__ import annotations

from rag.pipeline import run_pipeline, run_query, safe_run_pipeline


def test_run_pipeline_against_sample_report(stub_retriever, sample_report_path):
    results = run_pipeline(sample_report_path, retriever=stub_retriever)

    assert len(results) >= 5
    columns = {r["column"] for r in results}
    assert {"age", "G3", "absences", "sex"} <= columns

    # Every row has the contract fields.
    for r in results:
        assert "decision" in r
        assert "reason" in r
        # No row should leak a model name into the preprocessing decision.
        assert "regressor" not in str(r["decision"]).lower()
        assert "classifier" not in str(r["decision"]).lower()


def test_high_corr_target_is_not_skipped(stub_retriever, sample_report_path):
    results = run_pipeline(sample_report_path, retriever=stub_retriever)
    g3 = next(r for r in results if r["column"] == "G3")
    # G3 has corr=1.0 -> high_target_corr -> should NOT short-circuit to Keep
    assert g3["decision"] != "Keep (No Action)"


def test_categorical_decision_not_a_scaler(stub_retriever, sample_report_path):
    results = run_pipeline(sample_report_path, retriever=stub_retriever)
    sex = next(r for r in results if r["column"] == "sex")
    assert "scaler" not in str(sex["decision"]).lower()


def test_safe_run_pipeline_handles_missing_file():
    out = safe_run_pipeline("does/not/exist.txt")
    assert isinstance(out, list) and len(out) == 1
    assert out[0]["decision"] == "Error"
    assert "not found" in out[0]["reason"].lower()


def test_run_query_returns_string(stub_retriever):
    answer = run_query("how to handle missing numeric values?",
                       retriever=stub_retriever)
    assert isinstance(answer, str)
    assert "SimpleImputer" in answer or "imputation" in answer.lower()


def test_run_query_handles_empty_question():
    assert "Empty" in run_query("")
