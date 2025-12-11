"""
Configuration Management Module

Centralized configuration for Chan-ZKP, handling environment variables,
defaults, and configuration validation.
"""

import os
from typing import Optional, Union
from dataclasses import dataclass


@dataclass
class ChanZKPConfig:
    """Configuration for Chan-ZKP protocol."""
    
    # Cryptographic keys (can be overridden via environment variables)
    color_key: bytes
    commit_key: bytes
    session_key: bytes
    
    # Protocol parameters
    default_dimension: int = 3
    default_modulus: int = 97
    
    # Security parameters
    max_secret_attempts: int = 10000
    nonce_length: int = 8
    session_id_length: int = 8
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "json"  # "json" or "text"
    verbose: bool = False
    
    @classmethod
    def from_env(cls) -> "ChanZKPConfig":
        """
        Create configuration from environment variables.
        
        Environment Variables:
            CHAN_ZKP_COLOR_KEY: Key for vector coloring (default: "chan-zkp-color-key")
            CHAN_ZKP_COMMIT_KEY: Key for commitment generation (default: "chan-zkp-commit-key")
            CHAN_ZKP_SESSION_KEY: Key for session binding/transcript MAC (default: "chan-zkp-session-key")
            CHAN_ZKP_LOG_LEVEL: Logging level (default: "INFO")
            CHAN_ZKP_LOG_FORMAT: Log format - "json" or "text" (default: "json")
            CHAN_ZKP_VERBOSE: Enable verbose logging (default: "false")
        """
        # Helper to convert env var to bytes
        def get_key_bytes(env_var: str, default: str) -> bytes:
            value = os.getenv(env_var, default)
            if isinstance(value, bytes):
                return value
            return value.encode() if isinstance(value, str) else str(value).encode()
        
        color_key = get_key_bytes("CHAN_ZKP_COLOR_KEY", "chan-zkp-color-key")
        commit_key = get_key_bytes("CHAN_ZKP_COMMIT_KEY", "chan-zkp-commit-key")
        session_key = get_key_bytes("CHAN_ZKP_SESSION_KEY", "chan-zkp-session-key")
        
        log_level = os.getenv("CHAN_ZKP_LOG_LEVEL", "INFO").upper()
        log_format = os.getenv("CHAN_ZKP_LOG_FORMAT", "json").lower()
        verbose = os.getenv("CHAN_ZKP_VERBOSE", "false").lower() in ("true", "1", "yes")
        
        return cls(
            color_key=color_key,
            commit_key=commit_key,
            session_key=session_key,
            log_level=log_level,
            log_format=log_format,
            verbose=verbose
        )
    
    def get_color_key(self) -> bytes:
        """Get color key for vector coloring."""
        return self.color_key
    
    def get_commit_key(self) -> bytes:
        """Get commitment key for commitment generation."""
        return self.commit_key
    
    def get_session_key(self) -> bytes:
        """Get session key for transcript MAC."""
        return self.session_key


# Global configuration instance (lazy initialization)
_config: Optional[ChanZKPConfig] = None


def get_config() -> ChanZKPConfig:
    """Get global configuration instance (singleton pattern)."""
    global _config
    if _config is None:
        _config = ChanZKPConfig.from_env()
    return _config


def set_config(config: ChanZKPConfig) -> None:
    """Set global configuration instance."""
    global _config
    _config = config


def reset_config() -> None:
    """Reset global configuration (useful for testing)."""
    global _config
    _config = None

