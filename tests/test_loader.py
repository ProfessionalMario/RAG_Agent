import pytest
from ingestion.loader import load_report


def test_load_report(tmp_path):
    file = tmp_path / "sample.txt"
    file.write_text("Column: age")

    content = load_report(str(file))

    assert "age" in content