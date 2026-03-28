"""
Query Router Module – Developer Guide

Routes a user query to the appropriate knowledge context.

--------------------------------------------------------------------------------
Purpose:
--------------------------------------------------------------------------------
- Categorize a natural language query for retrieval
- Supports multiple knowledge domains:
    • EDA (exploratory data analysis)
    • ML (modeling / training / pipeline)
    • General fallback

--------------------------------------------------------------------------------
Core Function:
--------------------------------------------------------------------------------
route_query(query: str) -> str
    - Input: a user query string
    - Output: route key as string: "eda", "ml", or "general"
    - Behavior:
        • Matches keywords in lowercase
        • Returns first matching route
        • Falls back to "general" if no match

--------------------------------------------------------------------------------
Usage Example:
--------------------------------------------------------------------------------
route = route_query("How to handle missing values in Age column?")
print(route)
# "eda"

route = route_query("Train a regression model")
print(route)
# "ml"
"""

def route_query(query: str) -> str:
    q = query.lower()

    eda_keywords = [
        "missing", "null", "column", "dataset",
        "eda", "feature", "imputation", "duplicate"
    ]

    ml_keywords = [
        "model", "train", "algorithm", "classification",
        "regression", "pipeline", "fit"
    ]

    if any(k in q for k in eda_keywords):
        return "eda"

    if any(k in q for k in ml_keywords):
        return "ml"

    return "general"