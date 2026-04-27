"""
EDA report parser.

Reads a structured pipe-separated EDA report and produces a list of
column-level dictionaries the rest of the pipeline understands.

A line looks like:

    age | skew=0.4 | corr=0.12 | missing=15

Recognised attributes per column:
    skew=<float>      -> profile["skew"]
    corr=<float>      -> profile["correlation"]; sets high_target_corr if |x|>0.7
    missing=<float>   -> profile["missing_percent"]
    binary            -> categorical
    ordinal           -> categorical
    cardinality       -> categorical
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

from core.exceptions import ParserError
from core.logger import get_logger

logger = get_logger(__name__)

INVALID_COLUMNS = {
    "DATASET", "TASK", "DISTRIBUTION", "FEATURE",
    "TARGET", "CARDINALITY", "VARIANCE", "END",
    "GLOBAL", "FEATURES", "SUMMARY",
}

VALID_DECISIONS = {
    "drop_column",
    "impute_mean",
    "impute_median",
    "impute_mode",
    "encode_onehot",
    "normalize",
    "standardize",
    "keep",
}


def is_valid_column(col: str) -> bool:
    """Filter out structural noise without killing real column names."""
    if not col:
        return False
    col_clean = col.strip()
    if col_clean.upper() in INVALID_COLUMNS:
        return False
    if len(col_clean) > 25:
        return False
    if col_clean.isdigit():
        return False
    return True


def parse_report(
    report_text: str,
    missing_map: Dict[str, int] | None = None,
    total_rows: int | None = None,
) -> List[Dict]:
    """Convert a pipe-separated EDA report into column profile dicts."""
    columns: Dict[str, Dict] = {}
    for i, line in enumerate(report_text.splitlines()):
        line = line.strip()
        if not line or line.startswith("#") or "|" not in line:
            continue

        parts = [p.strip() for p in line.split("|")]
        col_candidate = parts[0].split(":")[0].strip()
        if not is_valid_column(col_candidate):
            logger.debug("[PARSER] line %d skipped: %s", i, col_candidate)
            continue

        col = col_candidate
        columns.setdefault(col, {"column": col})

        for part in parts[1:]:
            part_l = part.lower()

            if "skew=" in part_l:
                try:
                    columns[col]["skew"] = float(part_l.split("skew=")[1].split()[0])
                except (ValueError, IndexError):
                    logger.warning("[PARSER] bad skew in %r", part)

            if "corr=" in part_l:
                try:
                    raw = part_l.split("corr=")[1]
                    val = float(re.findall(r"-?\d+(?:\.\d+)?", raw)[0])
                    columns[col]["correlation"] = val
                    columns[col]["high_target_corr"] = abs(val) > 0.7
                except (ValueError, IndexError):
                    logger.warning("[PARSER] bad corr in %r", part)

            if "missing=" in part_l:
                try:
                    val = float(part_l.split("missing=")[1].split()[0])
                    columns[col]["missing_percent"] = val
                except (ValueError, IndexError):
                    logger.warning("[PARSER] bad missing in %r", part)

            if "binary" in part_l or "ordinal" in part_l or "cardinality" in part_l:
                columns[col]["dtype"] = "categorical"

        columns[col].setdefault("dtype", "numeric")

    parsed_columns: List[Dict] = []
    for col, data in columns.items():
        missing_count = (missing_map or {}).get(col, 0)
        if total_rows and "missing_percent" not in data:
            data["missing_percent"] = compute_missing_percent(missing_count, total_rows)
        parsed_columns.append({
            "column": col,
            "dtype": data.get("dtype", "unknown"),
            "missing_percent": data.get("missing_percent", 0.0),
            "skew": data.get("skew", 0),
            "correlation": data.get("correlation", 0),
            "high_target_corr": data.get("high_target_corr", False),
        })

    logger.info("[PARSER] Parsed %d columns", len(parsed_columns))
    return parsed_columns


# ---------------------------------------------------------------------------
def extract_rows(text: str) -> int:
    match = re.search(r"Rows:\s*(\d+)", text)
    if not match:
        raise ParserError("Rows not found")
    return int(match.group(1))


def extract_missing_by_column(text: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    section = re.search(r"Missing by Column(.*?)(\n\n|\Z)", text, re.DOTALL)
    if not section:
        return out
    for line in section.group(1).strip().split("\n"):
        m = re.match(r"(\w+):\s*(\d+)", line.strip())
        if m:
            out[m.group(1)] = int(m.group(2))
    return out


def extract_column_types(text: str) -> Tuple[List[str], List[str]]:
    numeric_match = re.search(r"Numeric\s*:\s*(.*)", text)
    categorical_match = re.search(r"Categorical\s*:\s*(.*)", text)
    if not numeric_match or not categorical_match:
        raise ParserError("Column types section missing")
    numeric_cols = [c.strip() for c in numeric_match.group(1).split(",")]
    categorical_cols = [c.strip() for c in categorical_match.group(1).split(",")]
    return numeric_cols, categorical_cols


def resolve_dtype(column: str, numeric_cols: List[str], categorical_cols: List[str]) -> str:
    if column in numeric_cols:
        return "numeric"
    if column in categorical_cols:
        return "categorical"
    return "unknown"


def compute_missing_percent(missing_count: int, total_rows: int) -> float:
    if not total_rows:
        return 0.0
    return round((missing_count / total_rows) * 100, 2)


def normalize_decision(decision: str) -> str:
    """Map a free-form LLM decision to one of `VALID_DECISIONS`."""
    if not isinstance(decision, str):
        return "keep"
    d = decision.lower().strip()
    if any(k in d for k in ("drop", "remove")):
        result = "drop_column"
    elif "median" in d:
        result = "impute_median"
    elif "mean" in d:
        result = "impute_mean"
    elif "mode" in d:
        result = "impute_mode"
    elif "encode" in d or "onehot" in d:
        result = "encode_onehot"
    elif "normalize" in d:
        result = "normalize"
    elif "standardize" in d or "scale" in d:
        result = "standardize"
    else:
        result = "keep"
    return result if result in VALID_DECISIONS else "keep"
