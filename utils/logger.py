"""
Module Header
Project: Web-Grounded LLM Content Generation for Engineering Education
Author: Antigravity AI
Purpose: Centralized logger utility providing dual logging (file and color-coded console).
Dependencies: logging, sys, pathlib, config.config
"""

import logging
import sys
from pathlib import Path
from config.config import LOG_LEVEL, LOGS_DIR

def setup_logger(name: str) -> logging.Logger:
    """
    Sets up a logger with the given name, outputting to console and a rotating log file.
    
    Args:
        name (str): Name of the logger (usually __name__).
        
    Returns:
        logging.Logger: The configured Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    
    # Avoid duplicate handlers if already configured
    if logger.handlers:
        return logger
        
    # Formatting
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(log_format)
    
    # File handler
    log_file = LOGS_DIR / "app.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Do not propagate to parent loggers
    logger.propagate = False
    
    return logger

if __name__ == "__main__":
    logger = setup_logger("test_logger")
    logger.debug("This is a DEBUG message")
    logger.info("This is an INFO message")
    logger.warning("This is a WARNING message")
    logger.error("This is an ERROR message")
    print("Success: Logger module executed.")
