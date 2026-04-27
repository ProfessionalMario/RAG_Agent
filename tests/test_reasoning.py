"""Unit tests for rag.reasoning (LLM call layer)."""
from __future__ import annotations

import pytest
import requests

from rag import reasoning
from rag.reasoning import build_prompt, call_llm_with_fallback, parse_llm_output


# --- Prompt builder ----------------------------------------------------------
def test_build_prompt_contains_column_profile_and_docs():
    prompt = build_prompt(
        {"column": "age", "dtype": "numeric", "missing_percent": 15, "skew": 2.1},
        ["doc one", "doc two"],
    )
    assert "age" in prompt
    assert "numeric" in prompt
    assert "15" in prompt
    assert "doc one" in prompt
    assert "Decision:" in prompt
    assert "Reason:" in prompt


def test_build_prompt_handles_no_docs():
    prompt = build_prompt({"column": "x", "dtype": "numeric",
                           "missing_percent": 0, "skew": 0}, [])
    assert "No specific scikit-learn documentation" in prompt


# --- Output parser -----------------------------------------------------------
def test_parse_llm_output_extracts_decision_and_reason():
    raw = "Decision: SimpleImputer\nReason: missing values"
    parsed = parse_llm_output(raw)
    assert parsed == {"decision": "SimpleImputer", "reason": "missing values"}


def test_parse_llm_output_falls_back_on_garbage():
    parsed = parse_llm_output("")
    assert parsed["decision"] == "keep"


def test_parse_llm_output_ignores_extra_lines():
    raw = "Header\nDecision: OneHotEncoder\nFooter\nReason: nominal"
    parsed = parse_llm_output(raw)
    assert parsed["decision"] == "OneHotEncoder"
    assert parsed["reason"] == "nominal"


# --- LLM call ----------------------------------------------------------------
def test_call_llm_uses_fallback_on_connection_error(monkeypatch):
    def boom(*_a, **_kw):
        raise requests.ConnectionError("nope")
    monkeypatch.setattr(reasoning.requests, "post", boom)
    monkeypatch.setattr(reasoning, "LLM_GRACEFUL_FALLBACK", True)
    out = call_llm_with_fallback("anything")
    assert "Decision:" in out and "fallback" in out.lower()


def test_call_llm_raises_when_fallback_disabled(monkeypatch):
    def boom(*_a, **_kw):
        raise requests.ConnectionError("nope")
    monkeypatch.setattr(reasoning.requests, "post", boom)
    monkeypatch.setattr(reasoning, "LLM_GRACEFUL_FALLBACK", False)
    with pytest.raises(requests.ConnectionError):
        call_llm_with_fallback("anything")


def test_call_llm_returns_response_field_on_success(monkeypatch):
    class FakeResp:
        status_code = 200
        def json(self):  # noqa: D401
            return {"response": "Decision: Keep\nReason: ok"}
    monkeypatch.setattr(reasoning.requests, "post",
                        lambda *_a, **_kw: FakeResp())
    out = call_llm_with_fallback("p")
    assert "Decision: Keep" in out
