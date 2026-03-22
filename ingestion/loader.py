from core.logger import get_logger
from core.exceptions import ValidationError

logger = get_logger(__name__)


def load_report(file_path: str) -> str:
    """
    Load EDA report from a text file
    """

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        if not content.strip():
            logger.error("File is empty")
            raise ValidationError("Report file is empty")

        logger.info(f"Loaded report from {file_path}")
        return content

    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise ValidationError(f"File not found: {file_path}")

    except Exception as e:
        logger.error(f"Error loading file: {str(e)}")
        raise ValidationError(f"Error loading file: {str(e)}")