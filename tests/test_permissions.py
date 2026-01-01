"""Tests for permission translator."""


from ccflow.permissions import (
    PermissionTranslator,
    format_bash_pattern,
    validate_tool_list,
)
from ccflow.types import PermissionMode


class TestPermissionTranslator:
    """Tests for PermissionTranslator class."""

    def test_translate_allowed_tools(self):
        """Test translating allowed tools."""
        tools = ["Read", "Edit", "Bash(git:*)"]
        flags = PermissionTranslator.translate_allowed_tools(tools)

        assert flags == ["--allowedTools", "Read", "Edit", "Bash(git:*)"]

    def test_translate_allowed_tools_empty(self):
        """Test empty tools list returns empty."""
        flags = PermissionTranslator.translate_allowed_tools([])
        assert flags == []

    def test_translate_disallowed_tools(self):
        """Test translating disallowed tools."""
        tools = ["Bash(rm:*)", "Bash(sudo:*)"]
        flags = PermissionTranslator.translate_disallowed_tools(tools)

        assert flags == ["--disallowedTools", "Bash(rm:*)", "Bash(sudo:*)"]

    def test_translate_permission_mode(self):
        """Test translating permission modes."""
        cases = [
            (PermissionMode.DEFAULT, ["--permission-mode", "default"]),
            (PermissionMode.PLAN, ["--permission-mode", "plan"]),
            (PermissionMode.DONT_ASK, ["--permission-mode", "dontAsk"]),
            (PermissionMode.ACCEPT_EDITS, ["--permission-mode", "acceptEdits"]),
            (PermissionMode.DELEGATE, ["--permission-mode", "delegate"]),
            (PermissionMode.BYPASS, ["--permission-mode", "bypassPermissions"]),
        ]

        for mode, expected in cases:
            flags = PermissionTranslator.translate_permission_mode(mode)
            assert flags == expected

    def test_translate_mcp_tools(self):
        """Test translating MCP tool permissions."""
        mcp_tools = {
            "github": ["get_issue", "list_prs"],
            "postgres": ["query"],
        }
        result = PermissionTranslator.translate_mcp_tools(mcp_tools)

        assert "mcp__github__get_issue" in result
        assert "mcp__github__list_prs" in result
        assert "mcp__postgres__query" in result

    def test_parse_mcp_tool(self):
        """Test parsing MCP tool specification."""
        result = PermissionTranslator.parse_mcp_tool("mcp__github__get_issue")
        assert result == ("github", "get_issue")

    def test_parse_mcp_tool_invalid(self):
        """Test parsing invalid MCP tool returns None."""
        result = PermissionTranslator.parse_mcp_tool("Read")
        assert result is None

    def test_build_permission_flags(self):
        """Test building complete permission flags."""
        flags = PermissionTranslator.build_permission_flags(
            mode=PermissionMode.ACCEPT_EDITS,
            allowed_tools=["Read", "Edit"],
            disallowed_tools=["Bash(rm:*)"],
            mcp_tools={"github": ["get_issue"]},
        )

        assert "--permission-mode" in flags
        assert "acceptEdits" in flags
        assert "--allowedTools" in flags
        assert "Read" in flags
        assert "mcp__github__get_issue" in flags
        assert "--disallowedTools" in flags
        assert "Bash(rm:*)" in flags


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_validate_tool_list(self):
        """Test validating tool list."""
        tools = ["Read", "Edit", "Invalid Tool!", "Bash(git:*)"]
        valid = validate_tool_list(tools)

        assert "Read" in valid
        assert "Edit" in valid
        assert "Bash(git:*)" in valid
        assert "Invalid Tool!" not in valid

    def test_format_bash_pattern_with_args(self):
        """Test formatting Bash pattern with wildcard args."""
        result = format_bash_pattern("git", allow_args=True)
        assert result == "Bash(git:*)"

    def test_format_bash_pattern_exact(self):
        """Test formatting exact Bash pattern."""
        result = format_bash_pattern("npm run test")
        assert result == "Bash(npm run test)"
