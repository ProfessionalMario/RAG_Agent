"""
Query builder.

Converts a parsed column profile into a dense, technical search string aimed
at Scikit-Learn documentation chunks.
"""
from __future__ import annotations

from typing import Dict

from core.logger import get_logger

logger = get_logger(__name__)


def build_query(column_data: Dict) -> str:
    """Build a sharp, technical retrieval query for one column."""
    try:
        col = column_data.get("column", "unknown")
        dtype = str(column_data.get("dtype", "unknown")).lower()
        missing = column_data.get("missing_percent", 0) or 0
        skew = abs(column_data.get("skew", 0) or 0)
        has_outliers = bool(column_data.get("outliers", False))

        parts = ["sklearn"]

        if any(n in dtype for n in ("numeric", "int", "float")):
            parts.append("preprocessing")
            if missing > 0:
                parts.append("impute SimpleImputer IterativeImputer")
            if has_outliers or skew > 1.0:
                parts.append("RobustScaler PowerTransformer QuantileTransformer")
            else:
                parts.append("StandardScaler MinMaxScaler")
        elif any(c in dtype for c in ("object", "string", "category", "nominal")):
            parts.append("preprocessing OneHotEncoder OrdinalEncoder")
            if missing > 0:
                parts.append("missing values strategy")

        parts.append(f"{dtype} feature {col}")
        query = " ".join(parts)
        logger.info("[QUERY] %s -> %s", col, query)
        return query
    except Exception as exc:  # noqa: BLE001
        logger.error("[QUERY] Failed to build query: %s", exc)
        return "sklearn preprocessing documentation"
