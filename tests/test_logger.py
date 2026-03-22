from core.logger import get_logger


def test_logger_creation():
    logger = get_logger("test_logger")
    assert logger is not None

    logger.info("Test log message")