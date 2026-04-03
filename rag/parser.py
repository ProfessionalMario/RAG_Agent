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

import re

def parse_report(report_text: str):
    """
    Parse dataset report into column metadata.

    Returns:
        List[dict] with:
        - column
        - dtype
        - missing_percent
    """

    columns = []

    # ---------------------------
    # 1. Extract missing section
    # ---------------------------
    missing_section = re.search(
        r"Missing by Column\n[-]+\n(.*?)\n\n",
        report_text,
        re.DOTALL
    )

    missing_data = {}

    if missing_section:
        content = missing_section.group(1).strip()

        if content.lower() != "none":
            for line in content.split("\n"):
                match = re.match(r"(.+?):\s*(\d+)%", line)
                if match:
                    col = match.group(1).strip()
                    pct = int(match.group(2))
                    missing_data[col] = pct

    # ---------------------------
    # 2. Extract column types
    # ---------------------------
    type_section = re.search(
        r"Column Types\n[-]+\n(.*?)\n\n",
        report_text,
        re.DOTALL
    )

    if not type_section:
        return []

    type_content = type_section.group(1)

    numeric_cols = []
    categorical_cols = []

    for line in type_content.split("\n"):
        if "Numeric" in line:
            numeric_cols = [c.strip() for c in line.split(":")[1].split(",")]
        elif "Categorical" in line:
            categorical_cols = [c.strip() for c in line.split(":")[1].split(",")]

    # ---------------------------
    # 3. Build column objects
    # ---------------------------
    all_columns = []

    for col in numeric_cols:
        all_columns.append({
            "column": col,
            "dtype": "numeric",
            "missing_percent": missing_data.get(col, 0)
        })

    for col in categorical_cols:
        all_columns.append({
            "column": col,
            "dtype": "categorical",
            "missing_percent": missing_data.get(col, 0)
        })

    return all_columns


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
    decision = decision.lower()

    if "drop" in decision:
        return "drop_column"
    elif "mean" in decision:
        return "impute_mean"
    elif "median" in decision:
        return "impute_median"
    elif "mode" in decision:
        return "impute_mode"
    elif "encode" in decision:
        return "encode_onehot"
    elif "normalize" in decision:
        return "normalize"
    elif "standardize" in decision:
        return "standardize"

    return "keep"