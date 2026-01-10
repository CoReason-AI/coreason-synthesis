from src.coreason_synthesis.utils.logger import logger


def test_logger_exists():
    assert logger is not None
    logger.info("Test log message")
