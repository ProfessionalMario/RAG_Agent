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