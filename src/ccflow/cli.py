"""
CLI Entry Point - Command-line interface for ccflow.

Provides a simple CLI for testing and one-off queries.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import NoReturn

from ccflow import __version__
from ccflow.api import query, query_simple
from ccflow.config import configure_logging, get_settings
from ccflow.types import CLIAgentOptions, PermissionMode, ToonConfig


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        prog="ccflow",
        description="Claude Code CLI Middleware - SDK-like interface for CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ccflow "Explain this code"
  ccflow -m opus "Review for security issues"
  ccflow --stream "Analyze the codebase"
  echo "code here" | ccflow "Explain this"
        """,
    )

    parser.add_argument(
        "prompt",
        nargs="?",
        help="Prompt to send to Claude (reads from stdin if not provided)",
    )

    parser.add_argument(
        "-m", "--model",
        default=None,
        help="Model to use (sonnet, opus, haiku)",
    )

    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream response in real-time",
    )

    parser.add_argument(
        "--max-budget",
        type=float,
        default=None,
        help="Maximum budget in USD",
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Timeout in seconds",
    )

    parser.add_argument(
        "--permission-mode",
        choices=["default", "plan", "dontAsk", "acceptEdits", "delegate", "bypass"],
        default="default",
        help="Permission mode for tool execution",
    )

    parser.add_argument(
        "--allowed-tools",
        nargs="+",
        help="Tools to allow without prompting",
    )

    parser.add_argument(
        "--no-toon",
        action="store_true",
        help="Disable TOON encoding",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"ccflow {__version__}",
    )

    return parser


def get_permission_mode(mode_str: str) -> PermissionMode:
    """Convert string to PermissionMode enum."""
    mapping = {
        "default": PermissionMode.DEFAULT,
        "plan": PermissionMode.PLAN,
        "dontAsk": PermissionMode.DONT_ASK,
        "acceptEdits": PermissionMode.ACCEPT_EDITS,
        "delegate": PermissionMode.DELEGATE,
        "bypass": PermissionMode.BYPASS,
    }
    return mapping.get(mode_str, PermissionMode.DEFAULT)


async def run_query(args: argparse.Namespace) -> int:
    """Execute query based on arguments."""
    # Get prompt
    prompt = args.prompt
    if not prompt:
        if sys.stdin.isatty():
            print("Error: No prompt provided", file=sys.stderr)
            return 1
        prompt = sys.stdin.read().strip()
        if not prompt:
            print("Error: Empty prompt", file=sys.stderr)
            return 1

    # Build options
    settings = get_settings()

    options = CLIAgentOptions(
        model=args.model or settings.default_model,
        max_budget_usd=args.max_budget,
        timeout=args.timeout or settings.default_timeout,
        permission_mode=get_permission_mode(args.permission_mode),
        allowed_tools=args.allowed_tools,
        verbose=args.verbose,
        toon=ToonConfig(enabled=not args.no_toon),
    )

    try:
        if args.stream:
            # Streaming mode
            async for msg in query(prompt, options):
                if hasattr(msg, "content"):
                    print(msg.content, end="", flush=True)
            print()  # Final newline
        else:
            # Simple mode
            result = await query_simple(prompt, options)
            print(result)

        return 0

    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def main() -> NoReturn:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Configure logging
    configure_logging()

    # Run async
    exit_code = asyncio.run(run_query(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
