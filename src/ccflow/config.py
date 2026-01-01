"""
Configuration Management - Environment-based settings.

Provides centralized configuration with environment variable support
and sensible defaults for ccflow middleware.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class CCFlowSettings(BaseSettings):
    """Environment-based configuration for ccflow.

    All settings can be overridden via environment variables
    with CCFLOW_ prefix.

    Example:
        >>> export CCFLOW_DEFAULT_MODEL=opus
        >>> export CCFLOW_TOON_ENABLED=true
        >>> settings = get_settings()
        >>> print(settings.default_model)  # "opus"
    """

    model_config = SettingsConfigDict(
        env_prefix="CCFLOW_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # CLI Configuration
    claude_cli_path: str | None = Field(
        default=None,
        description="Path to claude CLI. Auto-detected if not set.",
    )

    # Model defaults
    default_model: str = Field(
        default="sonnet",
        description="Default model to use (sonnet, opus, haiku, or full name)",
    )

    fallback_model: str | None = Field(
        default=None,
        description="Fallback model when primary is overloaded",
    )

    # Execution limits
    default_timeout: float = Field(
        default=300.0,
        ge=1.0,
        le=3600.0,
        description="Default execution timeout in seconds",
    )

    default_max_turns: int | None = Field(
        default=None,
        ge=1,
        le=100,
        description="Default maximum agentic turns",
    )

    # Session storage
    session_storage_path: str | None = Field(
        default=None,
        description="Path for session storage (uses CLI default if not set)",
    )

    # TOON defaults
    toon_enabled: bool = Field(
        default=True,
        description="Enable TOON encoding by default",
    )

    toon_delimiter: Literal[",", "\t", "|"] = Field(
        default=",",
        description="Default TOON field delimiter",
    )

    toon_track_savings: bool = Field(
        default=True,
        description="Track TOON compression metrics",
    )

    # Observability
    enable_metrics: bool = Field(
        default=True,
        description="Enable Prometheus metrics",
    )

    log_level: str = Field(
        default="INFO",
        description="Logging level",
    )

    log_format: Literal["console", "json"] = Field(
        default="console",
        description="Log output format",
    )

    # Rate limiting
    max_concurrent_requests: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum concurrent CLI requests",
    )

    rate_limit_per_minute: int = Field(
        default=60,
        ge=1,
        le=1000,
        description="Maximum requests per minute",
    )

    # Security
    allow_dangerous_permissions: bool = Field(
        default=False,
        description="Allow --dangerously-skip-permissions flag",
    )

    # Paths
    temp_dir: str | None = Field(
        default=None,
        description="Temporary directory for MCP configs",
    )

    @property
    def temp_path(self) -> Path:
        """Get temporary directory as Path."""
        if self.temp_dir:
            return Path(self.temp_dir)
        return Path(os.environ.get("TMPDIR", "/tmp"))


@lru_cache
def get_settings() -> CCFlowSettings:
    """Get cached settings instance.

    Returns:
        CCFlowSettings instance (cached)
    """
    return CCFlowSettings()


def configure_logging(settings: CCFlowSettings | None = None) -> None:
    """Configure structlog based on settings.

    Args:
        settings: Settings to use (uses global settings if None)
    """
    import structlog

    settings = settings or get_settings()

    if settings.log_format == "json":
        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ]
    else:
        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_log_level,
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
