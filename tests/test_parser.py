"""Unit tests for rag.parser."""
from __future__ import annotations

import pytest

from rag.parser import (
    compute_missing_percent,
    is_valid_column,
    normalize_decision,
    parse_report,
)


class TestIsValidColumn:
    def test_real_column_is_kept(self):
        assert is_valid_column("age") is True

    def test_structural_token_rejected(self):
        assert is_valid_column("DATASET") is False

    def test_empty_rejected(self):
        assert is_valid_column("") is False

    def test_too_long_rejected(self):
        assert is_valid_column("a" * 30) is False

    def test_pure_digits_rejected(self):
        assert is_valid_column("1234") is False


class TestParseReport:
    def test_parses_skew_corr_missing(self):
        text = "age | skew=0.4 | corr=0.12 | missing=15"
        cols = parse_report(text)
        assert len(cols) == 1
        col = cols[0]
        assert col["column"] == "age"
        assert col["dtype"] == "numeric"  # default fallback
        assert col["skew"] == pytest.approx(0.4)
        assert col["correlation"] == pytest.approx(0.12)
        assert col["missing_percent"] == pytest.approx(15.0)

    def test_binary_marks_categorical(self):
        text = "sex | binary"
        col = parse_report(text)[0]
        assert col["dtype"] == "categorical"

    def test_high_target_corr_flag(self):
        text = "G3 | corr=1.0"
        col = parse_report(text)[0]
        assert col["high_target_corr"] is True

    def test_ignores_comment_and_blank_lines(self):
        text = "# comment\n\nage | skew=0.1\n"
        cols = parse_report(text)
        assert len(cols) == 1

    def test_empty_report_returns_empty_list(self):
        assert parse_report("") == []

    def test_demo_report_parses(self, sample_report_path):
        cols = parse_report(open(sample_report_path).read())
        names = {c["column"] for c in cols}
        assert {"age", "G3", "absences", "sex", "Mjob"} <= names


class TestNormalizeDecision:
    @pytest.mark.parametrize("raw,expected", [
        ("Drop the column", "drop_column"),
        ("SimpleImputer(strategy='median')", "impute_median"),
        ("Use mean imputation", "impute_mean"),
        ("most_frequent / mode imputation", "impute_mode"),
        ("OneHotEncoder", "encode_onehot"),
        ("StandardScaler", "standardize"),
        ("Keep as-is", "keep"),
        ("Some weird unknown", "keep"),
    ])
    def test_mapping(self, raw, expected):
        assert normalize_decision(raw) == expected

    def test_non_string_falls_back_to_keep(self):
        assert normalize_decision(None) == "keep"
        assert normalize_decision(42) == "keep"


class TestComputeMissingPercent:
    def test_basic(self):
        assert compute_missing_percent(20, 100) == 20.0

    def test_zero_rows(self):
        assert compute_missing_percent(5, 0) == 0.0
