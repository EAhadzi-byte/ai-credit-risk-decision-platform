"""
=========================================================
Credit Risk AI Platform
Centralised Logger
=========================================================
"""

import logging
from pathlib import Path

from src.config.pipeline_config import LOG_DIR


def get_logger(name: str):

    """
    Creates and returns a reusable logger.
    """

    logger = logging.getLogger(name)

    if logger.handlers:

        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(

        "%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    log_file = LOG_DIR / "credit_risk_platform.log"

    file_handler = logging.FileHandler(log_file)

    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()

    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    logger.addHandler(stream_handler)

    return logger


if __name__ == "__main__":

    logger = get_logger(__name__)

    logger.info("Logger working successfully.")

