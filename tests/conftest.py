"""
Pytest fixtures shared by every test module.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Make sure the project root is importable.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Force the LLM into graceful fallback mode for the entire test session so
# tests never touch the network.
os.environ.setdefault("LLM_GRACEFUL_FALLBACK", "1")
os.environ.setdefault("OLLAMA_URL", "http://127.0.0.1:1/never")  # invalid

import pytest  # noqa: E402


class StubRetriever:
    """Deterministic retriever used in pipeline integration tests."""

    def __init__(self, docs):
        self.docs = list(docs)
        self.calls = []

    def retrieve(self, query, k=3):
        self.calls.append((query, k))
        return self.docs[:k]


@pytest.fixture
def stub_retriever():
    return StubRetriever([
        "Use SimpleImputer(strategy='median') for skewed numeric data.",
        "Apply OneHotEncoder for nominal categorical features.",
        "RobustScaler is preferred when outliers are present.",
    ])


@pytest.fixture
def sample_report_path():
    return str(ROOT / "data" / "reports" / "sample_report.txt")


@pytest.fixture
def reset_pipeline_singleton():
    """Wipe the pipeline retriever cache before/after each test that needs it."""
    from rag import pipeline
    pipeline.reset_pipeline()
    yield
    pipeline.reset_pipeline()
