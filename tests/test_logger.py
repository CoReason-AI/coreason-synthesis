from src.coreason_synthesis.utils.logger import logger


def test_logger_exists() -> None:
    assert logger is not None
    logger.info("Test log message")
