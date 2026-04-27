# """
# Query Builder – Developer Guide

# Converts structured column metadata into human-readable queries for retrieval.

# --------------------------------------------------------------------------------
# Purpose:
# --------------------------------------------------------------------------------
# - Input: column-level EDA dict (name, dtype, missing %)
# - Output: short descriptive query
# - Focuses on:
#     • data type (numeric / categorical)
#     • missing value percentage
# - Fallback: generic query if any failure occurs

# --------------------------------------------------------------------------------
# Callable Functions:
# --------------------------------------------------------------------------------
# build_query(column_data: dict) -> str
#     - Converts column metadata to a retrieval query
#     - Returns string like:
#       "numeric column with 20 percent missing values"
#     - On failure, returns:
#       "data preprocessing best practices"

# --------------------------------------------------------------------------------
# Usage Example:
# --------------------------------------------------------------------------------
# from rag.query import build_query

# query = build_query({
#     "column": "Age",
#     "missing_percent": 19.8,
#     "dtype": "numeric"
# })

# print(query)  # "numeric column with 20 percent missing values"
# """

# from typing import Dict
# from core.logger import get_logger

# logger = get_logger(__name__)



# def build_query(column_data: Dict) -> str:
#     """
#     UPGRADED: Maps structured data conditions to statistical 'Search Intents'
#     """
#     try:
#         col = column_data.get("column", "unknown")
#         dtype = column_data.get("dtype", "unknown")
#         skew = column_data.get("skew", 0)
#         cardinality = column_data.get("cardinality", "low")

#         query_parts = ["scikit-learn preprocessing"] # Add a 'Domain Anchor'

#         # 1. Handle Skewness (The most important signal)
#         if abs(skew) > 1.0:
#             query_parts.append("power transformation and handling high skewness")
        
#         # 2. Handle Categorical Encoding
#         elif dtype in ["nominal", "ordinal"]:
#             query_parts.append(f"encoding {dtype} categorical features")

#         # 3. Handle Numerical Scaling
#         elif dtype == "numeric":
#             query_parts.append("feature scaling and normalization for linear models")

#         # Combine into a 'Theoretical Question'
#         query = " ".join(query_parts)
#         logger.info(f"🚀 REFINED QUERY for {col}: {query}")
#         return query

#     except Exception as e:
#         logger.error(f"Query building failed: {str(e)}")
#         return "scikit-learn machine learning best practices"








# from typing import Dict
# from core.logger import get_logger

# logger = get_logger(__name__)

# def build_query(column_data: Dict) -> str:
#     """
#     Maps column stats to Scikit-Learn documentation search terms.
#     """
#     try:
#         col = column_data.get("column", "unknown")
#         dtype = column_data.get("dtype", "unknown")
#         missing = column_data.get("missing_percent", 0)
#         skew = column_data.get("skew", 0)

#         # Domain Anchor
#         query_parts = ["scikit-learn"]

#         # 1. Intent: Imputation (Based on missing %)
#         if missing > 0:
#             if missing > 40:
#                 query_parts.append("dropping columns with high missing values")
#             else:
#                 query_parts.append(f"imputation strategy for {dtype} data")

#         # 2. Intent: Transformation (Based on skew)
#         if dtype == "numeric" and abs(skew) > 1.0:
#             query_parts.append("power transformation Yeo-Johnson for skewed data")

#         # 3. Intent: Encoding
#         if dtype in ["nominal", "categorical", "string"]:
#             query_parts.append("onehotencoder vs ordinalencoder for categorical features")

#         # 4. Intent: Scaling
#         if dtype == "numeric" and missing < 5: # Scale only if relatively clean
#             query_parts.append("standardscaler vs robustscaler for feature scaling")

#         query = " ".join(query_parts)
#         logger.info(f"🔎 GENERATED QUERY [{col}]: {query}")
#         return query

#     except Exception as e:
#         logger.error(f"Query building failed: {str(e)}")
#         return "scikit-learn preprocessing best practices"







































from typing import Dict, List
from pathlib import Path
from core.logger import get_logger
from rag.parser import parse_report
logger = get_logger(__name__)


# -----------------------------
# LOAD REPORT
# -----------------------------
def load_report(path: str) -> str:
    try:
        report = Path(path).read_text(encoding="utf-8")
        logger.info(f"[REPORT] Loaded report from {path}")
        return report
    except Exception as e:
        logger.error(f"[REPORT LOAD FAILED] {e}")
        return ""


# -----------------------------
# QUERY BUILDER
# -----------------------------
import logging

logger = logging.getLogger(__name__)
def build_query(column_data: Dict) -> str:
    """
    Builds a focused, module-specific query for Scikit-Learn documentation.
    We remove the 'noise' words to keep the vector search on target.
    """
    try:
        col = column_data.get("column", "unknown")
        dtype = str(column_data.get("dtype", "unknown")).lower()
        missing = column_data.get("missing_percent", 0)
        skew = abs(column_data.get("skew", 0))
        has_outliers = column_data.get("outliers", False)

        # 🔹 Start with the Core Library (The Anchor)
        # Using 'sklearn' often works better than 'scikit-learn' for technical docs
        query_parts = ["sklearn"]

        # 🔹 Module-Specific Anchors (The "Junior" Secret Sauce)
        # Instead of 'machine learning problem', use the module name
        if dtype in ["numeric", "int", "float"]:
            query_parts.append("preprocessing")
            
            if missing > 0:
                query_parts.append("impute SimpleImputer IterativeImputer")
            
            if has_outliers or skew > 1.0:
                query_parts.append("RobustScaler PowerTransformer QuantileTransformer")
            else:
                query_parts.append("StandardScaler MinMaxScaler")

        elif dtype in ["object", "string", "category", "nominal"]:
            query_parts.append("preprocessing OneHotEncoder OrdinalEncoder")
            if missing > 0:
                query_parts.append("missing values strategy")

        # 🔹 Add the specific column context sparingly
        query_parts.append(f"{dtype} feature {col}")

        # Final assembly: Keep it short and dense.
        # Vector DBs love technical density over prose.
        query = " ".join(query_parts)

        logger.info(f"🔎 SHARPENED QUERY [{col}]: {query}")
        return query

    except Exception as e:
        logger.error(f"Query building failed: {str(e)}")
        return "sklearn preprocessing documentation"
    
    
# -----------------------------
# TEST BLOCK
# -----------------------------
def test_query_builder():
    report_path = "data/pdfs/reports.txt"

    report = load_report(report_path)
    dataset_context = parse_report(report)

    # Example column (simulate analyzer output)
    feature_report = {
    
    "column": "failures",
    "dtype": "numeric",
    "missing_percent": 0,
    "skew": 2.3,
    "task": "classification",
    "sample_size": "small",
    "outliers": True
}

    dataset_context = {
        "task": "classification",
        "dataset_size": "small"
    }

    combined = {**dataset_context, **feature_report}
    query = build_query(combined)
    print(query)


# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    test_query_builder()