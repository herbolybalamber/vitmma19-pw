from typing import Optional, Union, Literal, Dict
from datetime import datetime
from pathlib import Path
import logging
import sys
import os

from .config_manager import settings

LogLevel = Literal["INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL"]

# Load logging configuration. No defaults, fail if config not found
try:
    if hasattr(settings, "get_logging_config"):
        log_config = settings.get_logging_config()
    else:
        log_config = settings.get("logging")

    # Validate that all required logging config keys are present
    required_keys = ["log_dir", "console_level", "file_level", "log_to_file", "log_to_console", "log_format", "date_format"]
    missing_keys = [key for key in required_keys if key not in log_config]

    if missing_keys:
        raise ValueError(f"Missing required logging configuration keys: {missing_keys}")

    DEFAULT_LOG_DIR = log_config["log_dir"]
    DEFAULT_CONSOLE_LEVEL = log_config["console_level"]
    DEFAULT_FILE_LEVEL = log_config["file_level"]
    DEFAULT_LOG_TO_FILE = log_config["log_to_file"]
    DEFAULT_LOG_TO_CONSOLE = log_config["log_to_console"]
    LOG_FORMAT = log_config["log_format"]
    DATE_FORMAT = log_config["date_format"]
    CONSOLE_DATE_FORMAT = log_config.get("console_date_format", DATE_FORMAT)  # This one can default to DATE_FORMAT

except Exception as e:
    raise RuntimeError(f"FATAL: Could not load logging configuration. Logger cannot be initialized. Error: {str(e)}")


class Logger:
    """
    This class provides a simple interface to log messages at different levels:
    - DEBUG: Detailed information typically only of interest to a developer trying to diagnose a problem
    - INFO: General information about system operation, confirmation that things are working as expected, such as processing progress
    - WARNING: Indication of a potential problem, something unexpected happened. The software is still working as expected, possibly the 'error condition' could be fixed without user intervention.
    - ERROR: A serious problem occurred during execution, the software has not been able to perform some function.
    - CRITICAL: A serious error, indicating that the entire program itself may be unable to continue running, or will be terminated immediately.


    Usage:
        logger = Logger("component_name")
        logger.info("Processing file: example.pdf")
        logger.error("Failed to process file", exc_info=exception)
    """

    # Class variables to track file handlers
    _file_handlers: Dict[str, logging.FileHandler] = {}
    _log_dir: Optional[Path] = None

    @classmethod
    def _get_file_handler(cls, log_dir: Path, file_level: str) -> logging.FileHandler:
        """Get or create a file handler for the current day"""
        timestamp = datetime.now().strftime("%Y%m%d")
        handler_key = f"{timestamp}_{file_level}"

        # If we already have this handler, return it
        if handler_key in cls._file_handlers:
            return cls._file_handlers[handler_key]

        # Otherwise, create a new handler
        os.makedirs(log_dir, exist_ok=True)
        log_file = log_dir / f"{timestamp}.log"

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(getattr(logging, file_level))

        # Store for reuse
        cls._file_handlers[handler_key] = file_handler
        cls._log_dir = log_dir

        return file_handler

    def __init__(
        self,
        name: str,
        log_dir: Optional[Union[str, Path]] = None,
        console_level: LogLevel = DEFAULT_CONSOLE_LEVEL,
        file_level: LogLevel = DEFAULT_FILE_LEVEL,
        log_to_file: bool = DEFAULT_LOG_TO_FILE,
        log_to_console: bool = DEFAULT_LOG_TO_CONSOLE,
    ):
        """
        Initialize the logger.

        Args:
            name: The name of the logger (usually component or module name)
            log_dir: Directory to store log files (default: from config or ./logs)
            console_level: Minimum log level for console output (default: from config)
            file_level: Minimum log level for file output (default: from config)
            log_to_file: Whether to log to file (default: from config)
            log_to_console: Whether to log to console (default: from config)
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)  # Capture all levels

        # Only remove handlers created by this class, not all handlers
        for handler in list(self.logger.handlers):
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                self.logger.removeHandler(handler)

        # Add console handler if requested
        if log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_formatter = logging.Formatter(LOG_FORMAT, datefmt=CONSOLE_DATE_FORMAT)
            console_handler.setFormatter(console_formatter)
            console_handler.setLevel(getattr(logging, console_level))
            self.logger.addHandler(console_handler)

        # Add file handler if requested
        if log_to_file:
            # Determine log directory
            if log_dir is None:
                log_dir = Path(DEFAULT_LOG_DIR)
            else:
                log_dir = Path(log_dir)

            # Get or create the appropriate file handler
            file_handler = self._get_file_handler(log_dir, file_level)

            # Check if this logger already has this handler
            has_handler = False
            for handler in self.logger.handlers:
                if isinstance(handler, logging.FileHandler) and handler.baseFilename == file_handler.baseFilename:
                    has_handler = True
                    break

            # Only add if the logger doesn't already have it
            if not has_handler:
                self.logger.addHandler(file_handler)

    def debug(self, message: str, **kwargs):
        """Log a debug message."""
        self.logger.debug(message, **kwargs)

    def info(self, message: str, **kwargs):
        """Log an info message."""
        self.logger.info(message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log a warning message."""
        self.logger.warning(message, **kwargs)

    def error(self, message: str, exc_info=None, **kwargs):
        """
        Log an error message.

        Args:
            message: The error message
            exc_info: Exception information to include in the log
            **kwargs: Additional logging parameters
        """
        if exc_info is True or isinstance(exc_info, Exception):
            self.logger.error(message, exc_info=exc_info, **kwargs)
        else:
            self.logger.error(message, **kwargs)

    def critical(self, message: str, exc_info=None, **kwargs):
        """
        Log a critical error.

        Args:
            message: The error message
            exc_info: Exception information to include in the log
            **kwargs: Additional logging parameters
        """
        if exc_info is True or isinstance(exc_info, Exception):
            self.logger.critical(message, exc_info=exc_info, **kwargs)
        else:
            self.logger.critical(message, **kwargs)

    @classmethod
    def get_log_dir(cls) -> Optional[Path]:
        """Get the current log directory"""
        return cls._log_dir
