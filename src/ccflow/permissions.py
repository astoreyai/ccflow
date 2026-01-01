"""
Permission Translator - Convert SDK permissions to CLI flags.

Handles translation of tool permissions, permission modes,
and MCP tool access patterns.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from ccflow.types import PermissionMode

logger = structlog.get_logger(__name__)


class PermissionTranslator:
    """Translates SDK permissions to CLI flags.

    Handles the conversion between SDK-style permission configurations
    and CLI flag formats.

    Example:
        >>> translator = PermissionTranslator()
        >>> flags = translator.translate_allowed_tools(["Read", "Bash(git:*)"])
        >>> # Returns: ["--allowedTools", "Read", "Bash(git:*)"]
    """

    # Pattern for validating tool specifications
    TOOL_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\([^)]*\))?$")

    # Pattern for MCP tool format
    MCP_TOOL_PATTERN = re.compile(r"^mcp__([a-z0-9_-]+)__([a-z0-9_]+)$", re.IGNORECASE)

    @staticmethod
    def translate_allowed_tools(tools: list[str]) -> list[str]:
        """Convert tool list to --allowedTools format.

        Args:
            tools: List of tool specifications

        Returns:
            CLI flags for allowed tools

        Example:
            >>> translate_allowed_tools(["Read", "Edit", "Bash(git:*)"])
            ["--allowedTools", "Read", "Edit", "Bash(git:*)"]
        """
        if not tools:
            return []

        # Validate tool patterns
        for tool in tools:
            if not PermissionTranslator._validate_tool_spec(tool):
                logger.warning("invalid_tool_spec", tool=tool)

        return ["--allowedTools", *tools]

    @staticmethod
    def translate_disallowed_tools(tools: list[str]) -> list[str]:
        """Convert tool list to --disallowedTools format.

        Args:
            tools: List of tool specifications to deny

        Returns:
            CLI flags for disallowed tools
        """
        if not tools:
            return []

        return ["--disallowedTools", *tools]

    @staticmethod
    def translate_permission_mode(mode: PermissionMode) -> list[str]:
        """Convert permission mode to CLI flag.

        Args:
            mode: Permission mode enum value

        Returns:
            CLI flags for permission mode
        """
        return ["--permission-mode", mode.value]

    @staticmethod
    def translate_mcp_tools(mcp_tools: dict[str, list[str]]) -> list[str]:
        """Format MCP tool permissions for --allowedTools.

        Converts a dictionary of server -> tool mappings to
        mcp__server__tool format strings.

        Args:
            mcp_tools: Dict mapping server names to tool lists

        Returns:
            List of MCP tool permission strings

        Example:
            >>> translate_mcp_tools({"github": ["get_issue", "list_prs"]})
            ["mcp__github__get_issue", "mcp__github__list_prs"]
        """
        result: list[str] = []

        for server, tools in mcp_tools.items():
            # Normalize server name
            server_normalized = server.lower().replace("-", "_")

            for tool in tools:
                tool_normalized = tool.lower().replace("-", "_")
                mcp_tool = f"mcp__{server_normalized}__{tool_normalized}"
                result.append(mcp_tool)

        return result

    @staticmethod
    def parse_mcp_tool(tool_spec: str) -> tuple[str, str] | None:
        """Parse MCP tool specification into server and tool name.

        Args:
            tool_spec: MCP tool string (e.g., "mcp__github__get_issue")

        Returns:
            Tuple of (server_name, tool_name) or None if not MCP format
        """
        match = PermissionTranslator.MCP_TOOL_PATTERN.match(tool_spec)
        if match:
            return (match.group(1), match.group(2))
        return None

    @staticmethod
    def _validate_tool_spec(tool: str) -> bool:
        """Validate tool specification format.

        Args:
            tool: Tool specification string

        Returns:
            True if valid format
        """
        # Check for MCP format
        if tool.startswith("mcp__"):
            return PermissionTranslator.MCP_TOOL_PATTERN.match(tool) is not None

        # Check standard tool format
        return PermissionTranslator.TOOL_PATTERN.match(tool) is not None

    @staticmethod
    def build_permission_flags(
        mode: PermissionMode | None = None,
        allowed_tools: list[str] | None = None,
        disallowed_tools: list[str] | None = None,
        mcp_tools: dict[str, list[str]] | None = None,
        dangerous_skip: bool = False,
    ) -> list[str]:
        """Build complete permission flags from configuration.

        Combines all permission settings into a single list of CLI flags.

        Args:
            mode: Permission mode
            allowed_tools: Tools to allow
            disallowed_tools: Tools to deny
            mcp_tools: MCP server tool permissions
            dangerous_skip: Whether to skip all permissions

        Returns:
            Combined list of CLI flags
        """
        flags: list[str] = []

        # Permission mode
        if mode:
            flags.extend(PermissionTranslator.translate_permission_mode(mode))

        # Allowed tools (including MCP tools)
        all_allowed: list[str] = list(allowed_tools or [])
        if mcp_tools:
            all_allowed.extend(PermissionTranslator.translate_mcp_tools(mcp_tools))

        if all_allowed:
            flags.extend(PermissionTranslator.translate_allowed_tools(all_allowed))

        # Disallowed tools
        if disallowed_tools:
            flags.extend(PermissionTranslator.translate_disallowed_tools(disallowed_tools))

        # Dangerous skip (should be used with extreme caution)
        if dangerous_skip:
            flags.append("--dangerously-skip-permissions")
            logger.warning("dangerous_skip_enabled", message="All permission checks bypassed")

        return flags


# Convenience functions


def validate_tool_list(tools: list[str]) -> list[str]:
    """Validate and filter tool list.

    Args:
        tools: List of tool specifications

    Returns:
        List of valid tool specifications (invalid ones filtered out)
    """
    valid: list[str] = []
    translator = PermissionTranslator()

    for tool in tools:
        if translator._validate_tool_spec(tool):
            valid.append(tool)
        else:
            logger.warning("invalid_tool_filtered", tool=tool)

    return valid


def format_bash_pattern(command: str, allow_args: bool = False) -> str:
    """Format Bash tool permission pattern.

    Args:
        command: Base command (e.g., "git", "npm run")
        allow_args: Whether to allow any arguments

    Returns:
        Formatted Bash permission pattern

    Example:
        >>> format_bash_pattern("git", allow_args=True)
        "Bash(git:*)"
        >>> format_bash_pattern("npm run test")
        "Bash(npm run test:*)"
    """
    if allow_args:
        return f"Bash({command}:*)"
    return f"Bash({command})"
