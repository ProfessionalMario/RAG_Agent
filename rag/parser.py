"""
EDA Report Parser – Developer Guide

This module converts structured EDA reports into machine-readable
column-level data (list of dictionaries).

--------------------------------------------------------------------------------
Callable Functions:
--------------------------------------------------------------------------------

parse_report(report_text: str) -> List[Dict]
    - Parses structured EDA text into a list of column dictionaries.
    - Output fields per column: column name, missing count, missing percent, dtype.
    - Raises ParserError on empty/malformed reports.

extract_rows(text: str) -> int
    - Extracts total number of rows from report text.

extract_missing_by_column(text: str) -> Dict[str, int]
    - Extracts missing counts per column.
    - Returns empty dict if section missing.

extract_column_types(text: str) -> Tuple[List[str], List[str]]
    - Returns numeric and categorical columns.

resolve_dtype(column: str, numeric_cols: List[str], categorical_cols: List[str]) -> str
    - Returns column datatype: "numeric", "categorical", or "unknown".

compute_missing_percent(missing_count: int, total_rows: int) -> float
    - Computes missing percentage safely.

--------------------------------------------------------------------------------
Usage Notes:
--------------------------------------------------------------------------------
from parser import parse_report

report_text = open("eda_report.txt").read()
parsed_columns = parse_report(report_text)

for col in parsed_columns:
    print(f"{col['column']}: {col['missing_percent']}% missing")
"""

import re
from typing import List, Dict

from core.logger import get_logger
from core.exceptions import ParserError

logger = get_logger(__name__)

INVALID_COLUMNS = {
    "DATASET", "TASK", "DISTRIBUTION", "FEATURE", 
    "TARGET", "CARDINALITY", "VARIANCE", "END",
    "GLOBAL", "FEATURES", "SUMMARY"
}
VALID_DECISIONS = {
    "drop_column",
    "impute_mean",
    "impute_median",
    "impute_mode",
    "encode_onehot",
    "normalize",
    "standardize",
    "keep"
}

def is_valid_column(col: str) -> bool:
    """
    Balanced filter: removes structural noise without killing real columns
    """
    if not col:
        return False

    col_clean = col.strip()

    # Block known structural tokens only
    if col_clean.upper() in INVALID_COLUMNS:
        return False
    
    if len(col_clean) > 25:
        return False

    # Optional: block weird tokens (numbers-only, etc.)
    if col_clean.isdigit():
        return False

    return True
def parse_report(report_text: str, missing_map: Dict[str, int] = None, total_rows: int = None):
    columns = {}
    lines = report_text.splitlines()

    logger.info(f"[PARSER] Total lines: {len(lines)}")

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        if "|" not in line:
            continue

        logger.debug(f"[PARSER] Processing line {i}: {line}")

        parts = [p.strip() for p in line.split("|")]

        # -----------------------------------
        # 🧠 Extract column name (FIRST TOKEN)
        # -----------------------------------
        col_candidate = parts[0]

        # Clean patterns like: "TASK: regression"
        col_candidate = col_candidate.split(":")[0].strip()

        if not is_valid_column(col_candidate):
            logger.debug(f"[PARSER] ❌ Invalid column skipped: {col_candidate}")
            continue

        col = col_candidate
        logger.debug(f"[PARSER] ✅ Column detected: {col}")

        columns.setdefault(col, {})
        columns[col]["column"] = col

        # -----------------------------------
        # 📊 Parse attributes from rest
        # -----------------------------------
        for part in parts[1:]:
            part = part.lower()

            # Skew
            if "skew=" in part:
                try:
                    value = float(part.split("skew=")[1])
                    columns[col]["skew"] = value
                    logger.debug(f"[PARSER] {col} skew={value}")
                except:
                    logger.warning(f"[PARSER] Failed skew parse: {part}")

            # Correlation
            if "corr=" in part:
                try:
                    raw = part.split("corr=")[1]
                    value = float(re.findall(r"-?\d+\.\d+", raw)[0])
                    columns[col]["correlation"] = value
                    columns[col]["high_target_corr"] = abs(value) > 0.7
                    logger.debug(f"[PARSER] {col} corr={value}")
                except:
                    logger.warning(f"[PARSER] Failed corr parse: {part}")

            # Binary
            if "binary" in part:
                columns[col]["dtype"] = "categorical"

            # Ordinal
            if "ordinal" in part:
                columns[col]["dtype"] = "categorical"

            # Cardinality hint
            if "cardinality" in part:
                columns[col]["dtype"] = "categorical"

        # -----------------------------------
        # 🧠 Default dtype fallback
        # -----------------------------------
        if "dtype" not in columns[col]:
            columns[col]["dtype"] = "numeric"  # safe fallback

    # -----------------------------------
    # 📦 Final normalization
    # -----------------------------------
    parsed_columns = []

    for col, data in columns.items():
        missing_count = 0
        missing_percent = 0.0

        if missing_map and col in missing_map:
            missing_count = missing_map[col]

        if total_rows:
            missing_percent = compute_missing_percent(missing_count, total_rows)

        parsed_columns.append({
            "column": col,
            "dtype": data.get("dtype", "unknown"),
            "missing_percent": missing_percent,
            "skew": data.get("skew", 0),
            "correlation": data.get("correlation", 0),
            "high_target_corr": data.get("high_target_corr", False),
        })

    # -----------------------------------
    # 📊 Logging
    # -----------------------------------
    logger.info(f"[PARSER] Final column count: {len(parsed_columns)}")

    if len(parsed_columns) == 0:
        logger.error("[PARSER] ❌ No columns parsed — FORMAT mismatch")
    else:
        logger.debug(f"[PARSER] Columns parsed: {[c['column'] for c in parsed_columns]}")

    return parsed_columns

# -----------------------------
# 🔍 HELPER FUNCTIONS
# -----------------------------


def extract_rows(text: str) -> int:
    match = re.search(r"Rows:\s*(\d+)", text)
    if match:
        return int(match.group(1))

    logger.error("Rows not found in report")
    raise ParserError("Rows not found")


def extract_missing_by_column(text: str) -> Dict[str, int]:
    """
    Extract missing values per column
    """

    logger.info("Extracting missing values by column")

    missing_map = {}

    section = re.search(
        r"Missing by Column(.*?)(\n\n|\Z)", text, re.DOTALL
    )

    if not section:
        logger.warning("Missing by column section not found")
        return missing_map

    lines = section.group(1).strip().split("\n")

    for line in lines:
        match = re.match(r"(\w+):\s*(\d+)", line.strip())
        if match:
            col = match.group(1)
            val = int(match.group(2))
            missing_map[col] = val

    return missing_map


def extract_column_types(text: str):
    """
    Extract numeric and categorical columns
    """

    logger.info("Extracting column types")

    numeric_match = re.search(r"Numeric\s*:\s*(.*)", text)
    categorical_match = re.search(r"Categorical\s*:\s*(.*)", text)

    if not numeric_match or not categorical_match:
        logger.error("Column types not found")
        raise ParserError("Column types section missing")

    numeric_cols = [c.strip() for c in numeric_match.group(1).split(",")]
    categorical_cols = [c.strip() for c in categorical_match.group(1).split(",")]

    return numeric_cols, categorical_cols


def resolve_dtype(column: str, numeric_cols: List[str], categorical_cols: List[str]) -> str:
    """
    Resolve datatype for a column
    """

    if column in numeric_cols:
        return "numeric"
    elif column in categorical_cols:
        return "categorical"

    logger.warning(f"Datatype unknown for column: {column}")
    return "unknown"


def compute_missing_percent(missing_count: int, total_rows: int) -> float:
    """
    Compute missing percentage safely
    """

    try:
        if total_rows == 0:
            return 0.0

        return round((missing_count / total_rows) * 100, 2)

    except Exception as e:
        logger.error(f"Error computing missing %: {str(e)}")
        return 0.0

def normalize_decision(decision: str) -> str:
    """
    Normalize LLM output into strict preprocessing actions.
    Enforces ENUM safety.
    """

    if not isinstance(decision, str):
        logger.error(f"[NORMALIZE] Invalid type: {type(decision)} | value={decision}")
        return "keep"

    original = decision
    decision = decision.lower().strip()

    logger.debug(f"[NORMALIZE] Raw: '{original}' → '{decision}'")

    # 🔒 Strict mapping
    if any(k in decision for k in ["drop", "remove"]):
        result = "drop_column"
    elif "mean" in decision:
        result = "impute_mean"
    elif "median" in decision:
        result = "impute_median"
    elif "mode" in decision:
        result = "impute_mode"
    elif "encode" in decision or "onehot" in decision:
        result = "encode_onehot"
    elif "normalize" in decision:
        result = "normalize"
    elif "standardize" in decision or "scale" in decision:
        result = "standardize"
    else:
        result = "keep"

    # 🛑 Final guard
    if result not in VALID_DECISIONS:
        logger.warning(f"[NORMALIZE] Invalid mapped decision: {result}, fallback to keep")
        return "keep"

    logger.info(f"[NORMALIZE] Final decision: '{result}' (from '{original}')")

    return result