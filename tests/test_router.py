"""Unit tests for rag.router.route_query."""
import pytest

from rag.router import route_query


@pytest.mark.parametrize("q,expected", [
    ("how do I handle missing values in age?", "eda"),
    ("explain the imputation strategy",        "eda"),
    ("train a regression model on this data",  "ml"),
    ("which classification algorithm fits?",   "ml"),
    ("how is the weather today?",              "general"),
    ("",                                       "general"),
])
def test_route(q, expected):
    assert route_query(q) == expected
