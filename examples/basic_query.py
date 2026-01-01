#!/usr/bin/env python3
"""
Basic Query Example

Demonstrates the simplest usage of ccflow - single query with streaming response.
"""

import asyncio

from ccflow import query, CLIAgentOptions, TextMessage, AssistantMessage


async def main():
    """Execute a basic query and stream the response."""

    # Configure options
    options = CLIAgentOptions(
        model="sonnet",
        max_turns=5,
        allowed_tools=["Read", "Grep"],
    )

    print("Sending query to Claude...")
    print("-" * 40)

    # Stream the response
    async for msg in query("What files are in the current directory?", options):
        if isinstance(msg, TextMessage):
            print(msg.content, end="", flush=True)
        elif isinstance(msg, AssistantMessage):
            # Full assistant message - extract text content
            print(msg.text_content, end="", flush=True)

    print()
    print("-" * 40)
    print("Query complete!")


if __name__ == "__main__":
    asyncio.run(main())
