"""
Structured Logging Module

Provides structured logging with JSON format support for Chan-ZKP protocol.
"""

import json
import logging
import sys
from datetime import datetime
from typing import Any, Dict, Optional
from enum import Enum

from .config import get_config


class LogLevel(Enum):
    """Log level enumeration."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class StructuredLogger:
    """
    Structured logger with JSON format support.
    
    Logs are structured as JSON objects with consistent fields:
    {
        "timestamp": "ISO8601",
        "level": "INFO|DEBUG|WARNING|ERROR|CRITICAL",
        "component": "PROVER|VERIFIER|CORE|MAIN",
        "message": "Human-readable message",
        "data": { ... }  # Optional structured data
    }
    """
    
    def __init__(self, name: str, config: Optional[Any] = None):
        """
        Initialize structured logger.
        
        Args:
            name: Logger name (typically component name like "PROVER", "VERIFIER")
            config: Optional config instance (defaults to global config)
        """
        self.name = name
        self.config = config or get_config()
        self._logger = logging.getLogger(f"chan_zkp.{name}")
        self._setup_logger()
    
    def _setup_logger(self) -> None:
        """Setup logger with appropriate handler and formatter."""
        self._logger.setLevel(getattr(logging, self.config.log_level, logging.INFO))
        
        # Remove existing handlers
        self._logger.handlers.clear()
        
        # Create console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(self._logger.level)
        
        # Set formatter based on config
        if self.config.log_format == "json":
            handler.setFormatter(JSONFormatter())
        else:
            handler.setFormatter(logging.Formatter(
                '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ))
        
        self._logger.addHandler(handler)
    
    def _log(self, level: str, message: str, data: Optional[Dict[str, Any]] = None) -> None:
        """
        Internal logging method.
        
        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            message: Log message
            data: Optional structured data dictionary
        """
        if self.config.log_format == "json":
            log_entry = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "level": level.upper(),
                "component": self.name,
                "message": message
            }
            if data:
                log_entry["data"] = data
            print(json.dumps(log_entry))
        else:
            # Use standard logging for text format
            log_method = getattr(self._logger, level.lower(), self._logger.info)
            if data:
                message = f"{message} | data={json.dumps(data)}"
            log_method(message)
    
    def debug(self, message: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Log debug message."""
        if self.config.verbose or self._logger.isEnabledFor(logging.DEBUG):
            self._log("DEBUG", message, data)
    
    def info(self, message: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Log info message."""
        self._log("INFO", message, data)
    
    def warning(self, message: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Log warning message."""
        self._log("WARNING", message, data)
    
    def error(self, message: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Log error message."""
        self._log("ERROR", message, data)
    
    def critical(self, message: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Log critical message."""
        self._log("CRITICAL", message, data)


class JSONFormatter(logging.Formatter):
    """Custom formatter for JSON log output."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "component": record.name.split(".")[-1] if "." in record.name else record.name,
            "message": record.getMessage(),
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, "data"):
            log_entry["data"] = record.data
        
        return json.dumps(log_entry)


def get_logger(name: str) -> StructuredLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name (component name)
        
    Returns:
        StructuredLogger instance
    """
    return StructuredLogger(name)

