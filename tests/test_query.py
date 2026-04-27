"""Unit tests for rag.query.build_query."""
from rag.query import build_query


def test_numeric_clean_uses_standard_scaler_terms():
    q = build_query({"column": "age", "dtype": "numeric",
                     "missing_percent": 0, "skew": 0.2})
    assert "StandardScaler" in q
    assert "sklearn" in q


def test_numeric_with_outliers_uses_robust_terms():
    q = build_query({"column": "income", "dtype": "numeric",
                     "missing_percent": 0, "skew": 2.5,
                     "outliers": True})
    assert "RobustScaler" in q
    assert "PowerTransformer" in q


def test_numeric_with_missing_includes_imputer():
    q = build_query({"column": "age", "dtype": "numeric",
                     "missing_percent": 20, "skew": 0.1})
    assert "SimpleImputer" in q


def test_categorical_uses_encoder_terms():
    q = build_query({"column": "sex", "dtype": "object",
                     "missing_percent": 0})
    assert "OneHotEncoder" in q
    assert "OrdinalEncoder" in q


def test_categorical_with_missing_adds_strategy():
    q = build_query({"column": "city", "dtype": "category",
                     "missing_percent": 5})
    assert "missing values strategy" in q


def test_failure_returns_safe_default(monkeypatch):
    # Force exception inside try/except by passing something that breaks .lower()
    bad = {"column": object(), "dtype": object()}
    out = build_query(bad)
    assert isinstance(out, str)
    assert "sklearn" in out
