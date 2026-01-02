#!/usr/bin/env python3
"""
Multi-Turn Session Example

Demonstrates session management for multi-turn conversations,
with session statistics tracking.
"""

import asyncio

from ccflow import CLIAgentOptions, Session, TextMessage
from ccflow.types import PermissionMode


async def print_response(messages):
    """Print text messages from async iterator."""
    async for msg in messages:
        if isinstance(msg, TextMessage):
            print(msg.content, end="", flush=True)
    print()


async def main():
    """Run a multi-turn session."""

    # Create session with options
    session = Session(
        options=CLIAgentOptions(
            model="sonnet",
            permission_mode=PermissionMode.ACCEPT_EDITS,
        )
    )

    print(f"Session started: {session.session_id}")
    print("=" * 50)

    # First turn
    print("\n[Turn 1] Asking about the codebase...")
    await print_response(session.send_message(
        "Give me a brief overview of this project structure"
    ))

    # Second turn (continues the conversation)
    print("\n[Turn 2] Follow-up question...")
    await print_response(session.send_message(
        "What are the main entry points?"
    ))

    # Third turn
    print("\n[Turn 3] Another follow-up...")
    await print_response(session.send_message(
        "Summarize what we discussed in 2 sentences"
    ))

    # Close session and get stats
    stats = await session.close()

    print("\n" + "=" * 50)
    print("Session Statistics:")
    print(f"  Session ID: {stats.session_id}")
    print(f"  Total turns: {stats.total_turns}")
    print(f"  Input tokens: {stats.total_input_tokens}")
    print(f"  Output tokens: {stats.total_output_tokens}")
    print(f"  Total tokens: {stats.total_tokens}")
    print(f"  Duration: {stats.duration_seconds:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
