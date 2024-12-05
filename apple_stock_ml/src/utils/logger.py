import logging
import sys
from pathlib import Path


def setup_logger(name: str, log_file: str = None, level=logging.INFO) -> logging.Logger:
    """Set up and return a logger instance with both file and console handlers.

    Args:
        name (str): Name of the logger
        log_file (str, optional): Path to log file. If None, only console logging is enabled.
        level (int, optional): Logging level. Defaults to logging.INFO.

    Returns:
        logging.Logger: Configured logger instance
    """

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create formatters
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Create file handler if log_file is specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
