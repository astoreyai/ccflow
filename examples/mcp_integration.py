#!/usr/bin/env python3
"""
MCP Integration Example

Demonstrates programmatic MCP server configuration with ccflow.
"""

import asyncio
from pathlib import Path

from ccflow import CLIAgentOptions, TextMessage, query
from ccflow.mcp import (
    MCPConfigManager,
    MCPServerConfig,
    filesystem_server,
    github_server,
)


async def main():
    """Demonstrate MCP server configuration."""

    print("=" * 60)
    print("MCP Integration Demo")
    print("=" * 60)

    # Configure MCP servers
    servers = {
        # Custom server definition
        "my_api": MCPServerConfig(
            command="python",
            args=["-m", "my_custom_mcp_server"],
            env={"API_KEY": "${API_KEY}"},
        ),

        # Use convenience function for common servers
        "github": github_server("GITHUB_TOKEN"),

        # Filesystem access
        "files": filesystem_server([str(Path.cwd())]),
    }

    print("\nConfigured MCP servers:")
    for name, config in servers.items():
        print(f"  - {name}: {config.command} {' '.join(config.args)}")

    # Create config manager (handles temp file creation/cleanup)
    with MCPConfigManager() as manager:
        # Create config file
        config_path = manager.create_config_file(servers)
        print(f"\nMCP config file: {config_path}")

        # Configure options with MCP
        options = CLIAgentOptions(
            model="sonnet",
            mcp_servers=servers,
            strict_mcp=False,  # Allow other MCP servers too
            # MCP tools are auto-allowed via translate_mcp_tools
            allowed_tools=[
                "mcp__github__get_issue",
                "mcp__files__read_file",
            ],
        )

        print("\n" + "-" * 40)
        print("Querying with MCP tools...")
        print("-" * 40)

        # Note: This will only work if the MCP servers are actually running
        # This is a demonstration of the configuration pattern
        try:
            async for msg in query(
                "List the files in the current directory using filesystem MCP",
                options,
            ):
                if isinstance(msg, TextMessage):
                    print(msg.content, end="", flush=True)
            print()
        except Exception as e:
            print(f"Note: MCP query failed (expected if servers not running): {e}")

    print("\n" + "=" * 60)
    print("MCP config file cleaned up automatically")


if __name__ == "__main__":
    asyncio.run(main())
