"""
Logger – Developer Guide

Provides a unified logging system writing to both file and console.

--------------------------------------------------------------------------------
Callable Functions:
--------------------------------------------------------------------------------
get_logger(name: str) -> logging.Logger
    - Creates or retrieves a logger by name.
    - Logs to console and storage/logs/app.log.
    - Prevents duplicate handlers.
    - Usage: logger = get_logger("module_name"); logger.info("Message")
"""
import logging
import os

LOG_FILE = "storage/logs/app.log"

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # prevent duplicate handlers

    logger.setLevel(logging.DEBUG)  # capture everything

    # Ensure log directory exists
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    # 🧾 File handler (FULL logs)
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)

    # 🖥️ Console handler (MINIMAL logs)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)  # 👈 KEY LINE

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger