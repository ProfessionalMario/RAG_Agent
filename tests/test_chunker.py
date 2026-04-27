"""Unit tests for extractor.chunker."""
from extractor.chunker import chunk_text, generate_extraction_chunks


def test_chunk_text_empty_returns_empty():
    assert chunk_text("") == []


def test_chunk_text_respects_size_and_overlap():
    text = ("word " * 200).strip()
    chunks = chunk_text(text, chunk_size=80, chunk_overlap=20)
    assert len(chunks) >= 2
    assert all(len(c) <= 100 for c in chunks)


def test_chunks_cover_input_text():
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"
    chunks = chunk_text(text, chunk_size=20, chunk_overlap=4)
    joined = " ".join(chunks)
    assert "alpha" in joined and "mu" in joined


def test_generate_extraction_chunks_filters_blank():
    assert generate_extraction_chunks("") == []
    assert generate_extraction_chunks("   \n\n  ") == []
