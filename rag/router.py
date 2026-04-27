"""
Query router.

Routes a free-form user question into one of three knowledge domains so the
right context is retrieved.
"""
from __future__ import annotations

EDA_KEYWORDS = (
    "missing", "null", "column", "dataset",
    "eda", "feature", "imputation", "duplicate",
)
ML_KEYWORDS = (
    "model", "train", "algorithm", "classification",
    "regression", "pipeline", "fit",
)


def route_query(query: str) -> str:
    """Return one of 'eda', 'ml', 'general'."""
    if not query:
        return "general"
    q = query.lower()
    if any(k in q for k in EDA_KEYWORDS):
        return "eda"
    if any(k in q for k in ML_KEYWORDS):
        return "ml"
    return "general"
