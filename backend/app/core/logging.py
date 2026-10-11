"""Application logging configuration for SMS Spam Classification Engine API."""

import logging
import sys
from typing import Optional


def configure_logging(log_level: str = "INFO") -> None:
    """Configure structured console logging for the application.
    
    Ensures safe formatting without logging sensitive raw message payloads.
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Root logging configuration
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Avoid duplicate handlers if reconfigured
    if not root_logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(numeric_level)
        root_logger.addHandler(console_handler)
    else:
        for handler in root_logger.handlers:
            handler.setLevel(numeric_level)
            handler.setFormatter(formatter)
            
    # Quiet overly verbose external loggers
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get named logger instance."""
    return logging.getLogger(name)
