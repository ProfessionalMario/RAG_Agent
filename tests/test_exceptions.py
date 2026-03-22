from core.exceptions import ParserError


def test_parser_error():
    try:
        raise ParserError("Test error")
    except ParserError as e:
        assert str(e) == "Test error"