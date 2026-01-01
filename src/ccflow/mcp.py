"""
MCP Config Manager - Generate and manage MCP server configurations.

Handles programmatic MCP server definition, temporary config file
generation, and server health checking.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import structlog

from ccflow.exceptions import MCPConfigError
from ccflow.types import MCPServerConfig

logger = structlog.get_logger(__name__)


class MCPConfigManager:
    """Manages MCP server configuration files.

    Creates temporary JSON configuration files for the --mcp-config flag,
    allowing programmatic MCP server definition.

    Example:
        >>> manager = MCPConfigManager()
        >>> servers = {
        ...     "github": MCPServerConfig(command="npx", args=["@mcp/github"]),
        ...     "postgres": MCPServerConfig(command="python", args=["-m", "mcp_postgres"]),
        ... }
        >>> config_path = manager.create_config_file(servers)
        >>> # Use with: --mcp-config {config_path}
    """

    def __init__(self, temp_dir: Path | str | None = None) -> None:
        """Initialize config manager.

        Args:
            temp_dir: Directory for temporary config files
        """
        self._temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir())
        self._config_files: list[Path] = []

    def create_config_file(
        self,
        servers: dict[str, MCPServerConfig],
    ) -> Path:
        """Create temporary MCP config JSON file.

        Args:
            servers: Dict mapping server names to configurations

        Returns:
            Path to generated config file

        Raises:
            MCPConfigError: If config generation fails
        """
        if not servers:
            raise MCPConfigError("No MCP servers provided")

        try:
            config = self._build_config(servers)
            config_path = self._write_config(config)

            self._config_files.append(config_path)
            logger.info(
                "mcp_config_created",
                path=str(config_path),
                servers=list(servers.keys()),
            )

            return config_path

        except Exception as e:
            raise MCPConfigError(f"Failed to create MCP config: {e}") from e

    def _build_config(self, servers: dict[str, MCPServerConfig]) -> dict[str, Any]:
        """Build MCP configuration dictionary.

        Args:
            servers: Server configurations

        Returns:
            Config dict matching Claude CLI expected format
        """
        mcp_servers: dict[str, Any] = {}

        for name, config in servers.items():
            server_config: dict[str, Any] = {}

            if config.transport == "stdio":
                server_config["command"] = config.command
                if config.args:
                    server_config["args"] = config.args
                if config.env:
                    server_config["env"] = config.env
            else:
                # SSE or HTTP transport
                server_config["type"] = config.transport
                server_config["url"] = config.url
                if config.env:
                    server_config["headers"] = config.env  # Env as headers for remote

            mcp_servers[name] = server_config

        return {"mcpServers": mcp_servers}

    def _write_config(self, config: dict[str, Any]) -> Path:
        """Write config to temporary file.

        Args:
            config: Configuration dictionary

        Returns:
            Path to written file
        """
        # Create temp file with .json extension
        fd, path_str = tempfile.mkstemp(
            suffix=".json",
            prefix="ccflow_mcp_",
            dir=self._temp_dir,
        )

        path = Path(path_str)

        try:
            with open(fd, "w") as f:
                json.dump(config, f, indent=2)
        except Exception:
            path.unlink(missing_ok=True)
            raise

        return path

    def cleanup(self) -> None:
        """Remove all temporary config files."""
        for path in self._config_files:
            try:
                path.unlink(missing_ok=True)
                logger.debug("mcp_config_cleaned", path=str(path))
            except Exception as e:
                logger.warning("mcp_config_cleanup_failed", path=str(path), error=str(e))

        self._config_files.clear()

    def __enter__(self) -> MCPConfigManager:
        return self

    def __exit__(self, *args: Any) -> None:
        self.cleanup()


# Convenience functions for common MCP servers


def github_server(token_env: str = "GITHUB_TOKEN") -> MCPServerConfig:
    """Create GitHub MCP server config.

    Args:
        token_env: Environment variable containing GitHub token

    Returns:
        MCPServerConfig for GitHub MCP server
    """
    return MCPServerConfig(
        command="npx",
        args=["@anthropic-ai/mcp-server-github"],
        env={token_env: f"${{{token_env}}}"},
    )


def postgres_server(
    connection_string_env: str = "DATABASE_URL",
) -> MCPServerConfig:
    """Create PostgreSQL MCP server config.

    Args:
        connection_string_env: Environment variable with connection string

    Returns:
        MCPServerConfig for PostgreSQL MCP server
    """
    return MCPServerConfig(
        command="python",
        args=["-m", "mcp_postgres"],
        env={connection_string_env: f"${{{connection_string_env}}}"},
    )


def playwright_server() -> MCPServerConfig:
    """Create Playwright MCP server config.

    Returns:
        MCPServerConfig for Playwright browser automation
    """
    return MCPServerConfig(
        command="npx",
        args=["@playwright/mcp@latest"],
    )


def filesystem_server(
    allowed_paths: list[str] | None = None,
) -> MCPServerConfig:
    """Create filesystem MCP server config.

    Args:
        allowed_paths: Paths the server can access

    Returns:
        MCPServerConfig for filesystem access
    """
    args = ["@anthropic-ai/mcp-server-filesystem"]
    if allowed_paths:
        args.extend(allowed_paths)

    return MCPServerConfig(
        command="npx",
        args=args,
    )


def custom_http_server(
    url: str,
    headers: dict[str, str] | None = None,
) -> MCPServerConfig:
    """Create custom HTTP MCP server config.

    Args:
        url: Server URL
        headers: Optional headers (e.g., for authentication)

    Returns:
        MCPServerConfig for HTTP MCP server
    """
    return MCPServerConfig(
        command="",  # Not used for HTTP
        transport="http",
        url=url,
        env=headers or {},
    )
