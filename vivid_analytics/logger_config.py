import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str = __name__,
    level: int = logging.INFO,
    log_to_file: bool = False,
    log_to_console: bool = True,
    log_dir: str = "logs",
) -> logging.Logger:
    """
    Set up a logger with both file and console handlers.

    Args:
        name: Logger name (usually __name__)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to log to file
        log_to_console: Whether to log to console
        log_dir: Directory to store log files

    Returns:
        Configured logger instance
    """
    # * Create logger
    logger = logging.getLogger(name)

    # * Prevent duplicate handlers if logger already exists
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # * Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-2s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # * Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # * File handler
    if log_to_file:
        # * Create logs directory if it doesn't exist
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)

        # * Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = log_path / f"{name.replace('.', '_')}_{timestamp}.log"

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = __name__) -> logging.Logger:
    """
    Get or create a logger instance.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance
    """
    return setup_logger(name)


# * Pre-configured loggers for common modules
def get_analytics_logger() -> logging.Logger:
    """Get logger for analytics module."""
    return setup_logger("analytics", level=logging.INFO)


def get_telegram_logger() -> logging.Logger:
    """Get logger for telegram bot module."""
    return setup_logger("telegram_bot", level=logging.INFO)


def get_langgraph_logger() -> logging.Logger:
    """Get logger for langgraph agent module."""
    return setup_logger("langgraph_agent", level=logging.INFO)


def get_tools_logger() -> logging.Logger:
    """Get logger for tools module."""
    return setup_logger("tools", level=logging.INFO)


def get_data_logger() -> logging.Logger:
    """Get logger for data generation module."""
    return setup_logger("data_generation", level=logging.INFO)


# * Utility functions for common logging patterns
def log_function_entry(logger: logging.Logger, func_name: str, **kwargs):
    """Log function entry with parameters."""
    params = ", ".join(f"{k}={v}" for k, v in kwargs.items())
    logger.debug(f"Entering {func_name}({params})")


def log_function_exit(
    logger: logging.Logger, func_name: str, result: Optional[str] = None
):
    """Log function exit with optional result."""
    if result:
        logger.debug(f"Exiting {func_name} -> {result}")
    else:
        logger.debug(f"Exiting {func_name}")


def log_error(logger: logging.Logger, error: Exception, context: str = ""):
    """Log error with context information."""
    if context:
        logger.error(f"{context}: {str(error)}", exc_info=True)
    else:
        logger.error(f"Error: {str(error)}", exc_info=True)


def log_success(logger: logging.Logger, message: str):
    """Log success message."""
    logger.info(message)


def log_warning(logger: logging.Logger, message: str):
    """Log warning message."""
    logger.warning(message)


def log_debug(logger: logging.Logger, message: str):
    """Log debug message."""
    logger.debug(message)


def log_info(logger: logging.Logger, message: str):
    """Log info message."""
    logger.info(message)


# * Configure root logger to suppress noisy third-party logs
def configure_third_party_loggers():
    """Configure third-party library loggers to reduce noise."""
    # * Reduce noise from HTTP libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    # * Reduce noise from OpenAI
    logging.getLogger("openai").setLevel(logging.WARNING)

    # * Reduce noise from aiogram
    logging.getLogger("aiogram").setLevel(logging.WARNING)


# * Initialize third-party logger configuration
configure_third_party_loggers()
