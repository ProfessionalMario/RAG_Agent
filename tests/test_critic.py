"""Unit tests for the critic / validation layer."""
from __future__ import annotations

import pytest

from rag import critic
from rag.critic import is_dangerous, review_decision


# --- Safety guardrail --------------------------------------------------------
class TestDangerous:
    @pytest.mark.parametrize("text", [
        "rm -rf /",
        "sudo shutdown -h now",
        "os.system('ls')",
        "subprocess.call(...)",
        "eval('2+2')",
        "__import__('os').system('rm')",
    ])
    def test_flags_dangerous(self, text):
        assert is_dangerous(text) is True

    @pytest.mark.parametrize("text", [
        "StandardScaler",
        "OneHotEncoder",
        "Decision: keep",
        "",
    ])
    def test_safe_text(self, text):
        assert is_dangerous(text) is False


def test_blocks_dangerous_decision(monkeypatch):
    monkeypatch.setattr(critic, "call_llm_with_fallback",
                        lambda _p: "Keep")
    record = {"column": "x", "dtype": "numeric",
              "decision": "rm -rf /", "reason": "init"}
    review_decision(record, [])
    assert record["decision"] == "BLOCKED"
    assert "Safety critic" in record["reason"]


# --- Model rejection ---------------------------------------------------------
def test_rejects_model_in_decision(monkeypatch):
    monkeypatch.setattr(critic, "call_llm_with_fallback",
                        lambda _p: "Keep")
    record = {"column": "G3", "dtype": "numeric",
              "decision": "RandomForestRegressor", "reason": "init"}
    review_decision(record, [])
    assert record["decision"] == "None"
    assert "model selection" in record["reason"]


# --- Type enforcement --------------------------------------------------------
def test_categorical_cannot_be_scaled(monkeypatch):
    monkeypatch.setattr(critic, "call_llm_with_fallback",
                        lambda _p: "Keep")
    record = {"column": "sex", "dtype": "object",
              "decision": "StandardScaler", "reason": "init"}
    review_decision(record, [])
    assert record["decision"] == "OneHotEncoder"


def test_numeric_cannot_be_encoded(monkeypatch):
    monkeypatch.setattr(critic, "call_llm_with_fallback",
                        lambda _p: "Keep")
    record = {"column": "age", "dtype": "int64",
              "decision": "OneHotEncoder", "reason": "init"}
    review_decision(record, [])
    assert record["decision"] == "StandardScaler"


# --- LLM patch path ----------------------------------------------------------
def test_llm_can_replace_decision(monkeypatch):
    monkeypatch.setattr(critic, "call_llm_with_fallback",
                        lambda _p: "RobustScaler")
    record = {"column": "income", "dtype": "float64",
              "decision": "MinMaxScaler", "reason": "init"}
    review_decision(record, ["RobustScaler handles outliers."])
    assert record["decision"] == "RobustScaler"
    assert "Critic correction" in record["reason"]


def test_llm_keeps_decision_when_keep(monkeypatch):
    monkeypatch.setattr(critic, "call_llm_with_fallback",
                        lambda _p: "Keep")
    record = {"column": "age", "dtype": "float64",
              "decision": "StandardScaler", "reason": "init"}
    review_decision(record, [])
    assert record["decision"] == "StandardScaler"


# --- Output parser ----------------------------------------------------------
def test_parse_critic_strips_decision_prefix():
    assert critic._parse_critic_output("Decision: keep") == "keep"
    assert critic._parse_critic_output("Final Decision: RobustScaler") == "RobustScaler"
    assert critic._parse_critic_output("StandardScaler") == "StandardScaler"
    assert critic._parse_critic_output("") == ""


# --- Doc filter --------------------------------------------------------------
def test_doc_filter_prefers_preprocessing_over_models():
    docs = [
        "RandomForestRegressor handles regression tasks.",
        "StandardScaler scales numeric features.",
        "LogisticRegression is a classifier.",
    ]
    out = critic._score_and_filter_docs(docs, "numeric")
    assert "StandardScaler scales numeric features." in out
    # Model docs should be deprioritised away
    assert "RandomForestRegressor handles regression tasks." not in out
