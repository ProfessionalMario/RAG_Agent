"""
Query Builder – Developer Guide

Converts structured column metadata into human-readable queries for retrieval.

--------------------------------------------------------------------------------
Purpose:
--------------------------------------------------------------------------------
- Input: column-level EDA dict (name, dtype, missing %)
- Output: short descriptive query
- Focuses on:
    • data type (numeric / categorical)
    • missing value percentage
- Fallback: generic query if any failure occurs

--------------------------------------------------------------------------------
Callable Functions:
--------------------------------------------------------------------------------
build_query(column_data: dict) -> str
    - Converts column metadata to a retrieval query
    - Returns string like:
      "numeric column with 20 percent missing values"
    - On failure, returns:
      "data preprocessing best practices"

--------------------------------------------------------------------------------
Usage Example:
--------------------------------------------------------------------------------
from rag.query import build_query

query = build_query({
    "column": "Age",
    "missing_percent": 19.8,
    "dtype": "numeric"
})

print(query)  # "numeric column with 20 percent missing values"
"""

from typing import Dict
from core.logger import get_logger

logger = get_logger(__name__)


def build_query(column_data: Dict) -> str:
    """
    Convert structured column data into a retrieval query
    """

    try:
        col = column_data.get("column", "unknown")
        dtype = column_data.get("dtype", "unknown")
        missing = round(column_data.get("missing_percent", 0))

        query_parts = []

        # Type
        if dtype != "unknown":
            query_parts.append(f"{dtype} column")

        # Missing
        if missing > 0:
            query_parts.append(f"with {missing} percent missing values")

        # Combine
        query = " ".join(query_parts)

        logger.info(f"Built query for {col}: {query}")

        return query

    except Exception as e:
        logger.error(f"Query building failed: {str(e)}")
        return "data preprocessing best practices"