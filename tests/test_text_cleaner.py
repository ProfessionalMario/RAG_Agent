"""Unit tests for extractor.text_cleaner."""
from extractor.text_cleaner import is_valid_chunk, normalize, remove_tables


def test_normalize_strips_replacement_chars_and_whitespace():
    raw = "hello\ufffd  world\nfoo  bar"
    out = normalize(raw)
    assert "\ufffd" not in out
    assert "  " not in out
    assert out == "hello world foo bar"


def test_normalize_handles_empty_string():
    assert normalize("") == ""


def test_remove_tables_drops_separator_lines():
    raw = "intro\n|---|---|\nrow a | row b\nfooter"
    out = remove_tables(raw)
    assert "---" not in out
    assert "row a row b" in out
    assert "footer" in out


def test_is_valid_chunk_rejects_short():
    assert is_valid_chunk("short") is False
    assert is_valid_chunk("a b c d e") is False


def test_is_valid_chunk_accepts_long():
    big = "This is a sufficiently long chunk with enough words to be valid."
    assert is_valid_chunk(big) is True
