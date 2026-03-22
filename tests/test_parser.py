import pytest
from parser.eda_parser import parse_report
from core.exceptions import ParserError


# 🧪 Test: Valid report parsing
def test_parse_valid_report():
    # hint: simulate a well-formed EDA report input
    report = """
    Column: age
    Type: numeric
    Missing: 32%
    Skew: high
    """

    # hint: parse the report into structured output
    result = parse_report(report)

    # hint: ensure exactly one column is parsed
    assert len(result) == 1

    # hint: validate extracted fields
    assert result[0]["column"] == "age"
    assert result[0]["missing_percent"] == 32.0
    assert result[0]["dtype"] == "numeric"
    assert result[0]["skew"] == "high"


# 🧪 Test: Missing optional fields should fallback to defaults
def test_parse_missing_values_default():
    # hint: report without Missing and Skew info
    report = """
    Column: salary
    Type: numeric
    """

    # hint: parse incomplete report
    result = parse_report(report)

    # hint: default values should be applied
    assert result[0]["missing_percent"] == 0.0
    assert result[0]["skew"] == "unknown"


# 🧪 Test: Empty input should raise error
def test_empty_report():
    # hint: empty report is invalid input
    with pytest.raises(ParserError):
        parse_report("")