from parser.eda_parser import parse_report


def test_full_parse_flow():
    report = """
    Column: age
    Type: numeric
    Missing: 20%
    Skew: high
    """

    result = parse_report(report)

    assert isinstance(result, list)
    assert result[0]["column"] == "age"



def test_weird_format():
    report = "age Missing: 30% Type numeric"

    result = parse_report(report)

    assert len(result) >= 0  # should not crash