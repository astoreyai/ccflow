"""
CLI Entry Point - Command-line interface for ccflow.

Provides a full-featured CLI with subcommands for querying,
session management, server control, and statistics.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime
from typing import NoReturn

from ccflow import __version__
from ccflow.api import query, query_simple
from ccflow.config import configure_logging, get_settings
from ccflow.types import CLIAgentOptions, PermissionMode, ToonConfig


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="ccflow",
        description="Claude Code CLI Middleware - SDK-like interface for CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"ccflow {__version__}",
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Query command (default behavior)
    _add_query_parser(subparsers)

    # Sessions command
    _add_sessions_parser(subparsers)

    # Server command
    _add_server_parser(subparsers)

    # Stats command
    _add_stats_parser(subparsers)

    return parser


def _add_query_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add query subcommand parser."""
    query_parser = subparsers.add_parser(
        "query",
        aliases=["q"],
        help="Execute a query",
        description="Send a prompt to Claude and get a response",
        epilog="""
Examples:
  ccflow query "Explain this code"
  ccflow q -m opus "Review for security issues"
  ccflow query --stream "Analyze the codebase"
  echo "code here" | ccflow query "Explain this"
        """,
    )

    query_parser.add_argument(
        "prompt",
        nargs="?",
        help="Prompt to send to Claude (reads from stdin if not provided)",
    )

    query_parser.add_argument(
        "-m",
        "--model",
        default=None,
        help="Model to use (sonnet, opus, haiku)",
    )

    query_parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream response in real-time",
    )

    query_parser.add_argument(
        "--max-budget",
        type=float,
        default=None,
        help="Maximum budget in USD",
    )

    query_parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Timeout in seconds",
    )

    query_parser.add_argument(
        "--permission-mode",
        choices=["default", "plan", "dontAsk", "acceptEdits", "delegate", "bypass"],
        default="default",
        help="Permission mode for tool execution",
    )

    query_parser.add_argument(
        "--allowed-tools",
        nargs="+",
        help="Tools to allow without prompting",
    )

    query_parser.add_argument(
        "--session-id",
        help="Session ID for multi-turn conversation",
    )

    query_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume previous session",
    )

    query_parser.add_argument(
        "--no-toon",
        action="store_true",
        help="Disable TOON encoding",
    )


def _add_sessions_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add sessions subcommand parser."""
    sessions_parser = subparsers.add_parser(
        "sessions",
        aliases=["s"],
        help="Manage sessions",
        description="List, view, and manage conversation sessions",
    )

    sessions_subparsers = sessions_parser.add_subparsers(
        dest="sessions_action",
        help="Session actions",
    )

    # List sessions
    list_parser = sessions_subparsers.add_parser(
        "list",
        aliases=["ls"],
        help="List sessions",
    )
    list_parser.add_argument(
        "--status",
        choices=["active", "closed", "error"],
        help="Filter by status",
    )
    list_parser.add_argument(
        "--model",
        help="Filter by model",
    )
    list_parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of sessions to show",
    )
    list_parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )

    # Get session
    get_parser = sessions_subparsers.add_parser(
        "get",
        help="Get session details",
    )
    get_parser.add_argument(
        "session_id",
        help="Session ID to get",
    )
    get_parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )

    # Delete session
    delete_parser = sessions_subparsers.add_parser(
        "delete",
        aliases=["rm"],
        help="Delete a session",
    )
    delete_parser.add_argument(
        "session_id",
        help="Session ID to delete",
    )
    delete_parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force delete without confirmation",
    )

    # Cleanup sessions
    cleanup_parser = sessions_subparsers.add_parser(
        "cleanup",
        help="Clean up old sessions",
    )
    cleanup_parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Delete sessions older than N days",
    )
    cleanup_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without deleting",
    )


def _add_server_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add server subcommand parser."""
    server_parser = subparsers.add_parser(
        "server",
        aliases=["serve"],
        help="Start HTTP server",
        description="Start the FastAPI HTTP/WebSocket server",
    )

    server_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )

    server_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on (default: 8000)",
    )

    server_parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )

    server_parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes",
    )

    server_parser.add_argument(
        "--cors-origins",
        nargs="+",
        default=["*"],
        help="Allowed CORS origins",
    )

    server_parser.add_argument(
        "--metrics-port",
        type=int,
        default=None,
        help="Separate port for Prometheus metrics",
    )


def _add_stats_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add stats subcommand parser."""
    stats_parser = subparsers.add_parser(
        "stats",
        help="Show statistics",
        description="Display session and usage statistics",
    )

    stats_parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )

    stats_parser.add_argument(
        "--rate-limiter",
        action="store_true",
        help="Show rate limiter statistics",
    )


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
    """Execute query command."""
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
        session_id=getattr(args, "session_id", None),
        resume=getattr(args, "resume", False),
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


async def run_sessions(args: argparse.Namespace) -> int:
    """Execute sessions command."""
    from ccflow.manager import SessionManager
    from ccflow.store import SessionStatus

    action = args.sessions_action

    if not action:
        print("Error: No action specified. Use 'ccflow sessions --help'", file=sys.stderr)
        return 1

    async with SessionManager() as manager:
        if action in ("list", "ls"):
            # List sessions
            status_filter = None
            if args.status:
                status_filter = SessionStatus(args.status)

            sessions = await manager.list_sessions(
                status=status_filter,
                model=args.model,
                limit=args.limit,
            )

            if args.json:
                import json

                data = [
                    {
                        "session_id": s.session_id,
                        "model": s.model,
                        "status": s.status.value,
                        "turn_count": s.turn_count,
                        "created_at": s.created_at.isoformat(),
                        "updated_at": s.updated_at.isoformat(),
                    }
                    for s in sessions
                ]
                print(json.dumps(data, indent=2))
            else:
                if not sessions:
                    print("No sessions found.")
                    return 0

                print(f"{'SESSION ID':<40} {'MODEL':<10} {'STATUS':<10} {'TURNS':<6} {'UPDATED'}")
                print("-" * 90)
                for s in sessions:
                    updated = s.updated_at.strftime("%Y-%m-%d %H:%M")
                    print(
                        f"{s.session_id:<40} {s.model or 'unknown':<10} {s.status.value:<10} {s.turn_count:<6} {updated}"
                    )

        elif action == "get":
            # Get session details
            session = await manager.get_session(args.session_id)
            if session is None:
                print(f"Session not found: {args.session_id}", file=sys.stderr)
                return 1

            if args.json:
                import json

                session_data = {
                    "session_id": session.session_id,
                    "model": session.options.model if session.options else None,
                    "turn_count": session.turn_count,
                    "is_closed": session.is_closed,
                    "created_at": session.created_at.isoformat(),
                    "updated_at": session.updated_at.isoformat(),
                    "tags": list(session.tags),
                }
                print(json.dumps(session_data, indent=2))
            else:
                print(f"Session ID:  {session.session_id}")
                print(f"Model:       {session.options.model if session.options else 'unknown'}")
                print(f"Turn Count:  {session.turn_count}")
                print(f"Status:      {'closed' if session.is_closed else 'active'}")
                print(f"Created:     {session.created_at}")
                print(f"Updated:     {session.updated_at}")
                if session.tags:
                    print(f"Tags:        {', '.join(session.tags)}")

        elif action in ("delete", "rm"):
            # Delete session
            if not args.force:
                confirm = input(f"Delete session {args.session_id}? [y/N] ")
                if confirm.lower() != "y":
                    print("Cancelled.")
                    return 0

            deleted = await manager.delete_session(args.session_id)
            if deleted:
                print(f"Deleted session: {args.session_id}")
            else:
                print(f"Session not found: {args.session_id}", file=sys.stderr)
                return 1

        elif action == "cleanup":
            # Cleanup old sessions
            from datetime import timedelta

            threshold = timedelta(days=args.days)

            if args.dry_run:
                cutoff = datetime.now() - threshold
                sessions = await manager.list_sessions(
                    created_before=cutoff,
                    limit=1000,
                )
                print(f"Would delete {len(sessions)} sessions older than {args.days} days:")
                for s in sessions[:10]:
                    print(f"  - {s.session_id} (updated {s.updated_at})")
                if len(sessions) > 10:
                    print(f"  ... and {len(sessions) - 10} more")
            else:
                count = await manager.cleanup_expired()
                print(f"Cleaned up {count} expired sessions")

    return 0


async def run_server(args: argparse.Namespace) -> int:
    """Execute server command."""
    try:
        import uvicorn
    except ImportError:
        print(
            "Error: uvicorn not installed. Install with: pip install ccflow[server]",
            file=sys.stderr,
        )
        return 1

    try:
        from ccflow.server import CCFlowServer
    except ImportError:
        print(
            "Error: FastAPI not installed. Install with: pip install ccflow[server]",
            file=sys.stderr,
        )
        return 1

    print(f"Starting ccflow server on {args.host}:{args.port}")

    # Start metrics server if requested
    if args.metrics_port:
        from ccflow.metrics_handlers import start_metrics_server

        if start_metrics_server(args.metrics_port):
            print(f"Metrics available at http://{args.host}:{args.metrics_port}/metrics")

    # Create and run server
    server = CCFlowServer(cors_origins=args.cors_origins)

    config = uvicorn.Config(
        server.app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        log_level="info",
    )

    server_instance = uvicorn.Server(config)
    await server_instance.serve()

    return 0


async def run_stats(args: argparse.Namespace) -> int:
    """Execute stats command."""
    from ccflow.manager import SessionManager
    from ccflow.rate_limiting import get_limiter

    stats: dict = {}

    # Session stats
    async with SessionManager() as manager:
        session_stats = await manager.get_stats()
        stats["sessions"] = session_stats

    # Rate limiter stats
    if args.rate_limiter:
        limiter = get_limiter()
        stats["rate_limiter"] = limiter.stats

    if args.json:
        import json

        print(json.dumps(stats, indent=2, default=str))
    else:
        print("Session Statistics:")
        print(f"  Total sessions:     {stats['sessions'].get('total_sessions', 0)}")
        print(f"  Active sessions:    {stats['sessions'].get('active_sessions', 0)}")
        print(f"  Closed sessions:    {stats['sessions'].get('closed_sessions', 0)}")
        print(f"  In-memory sessions: {stats['sessions'].get('in_memory_sessions', 0)}")

        if args.rate_limiter and "rate_limiter" in stats:
            print("\nRate Limiter Statistics:")
            rl = stats["rate_limiter"]
            if "rate_limiter" in rl:
                rls = rl["rate_limiter"]
                print(f"  Total requests:   {rls.get('total_requests', 0)}")
                print(f"  Total waits:      {rls.get('total_waits', 0)}")
                print(f"  Avg wait time:    {rls.get('average_wait_time', 0):.2f}s")
                print(f"  Rejected:         {rls.get('rejected_requests', 0)}")
            if "concurrency_limiter" in rl:
                cls = rl["concurrency_limiter"]
                print(f"  Peak concurrent:  {cls.get('peak_concurrent', 0)}")
                print(f"  Current:          {cls.get('current_concurrent', 0)}")

    return 0


def main() -> NoReturn:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Configure logging
    configure_logging()

    # Route to appropriate handler
    if args.command in ("query", "q") or (hasattr(args, "prompt") and args.prompt):
        exit_code = asyncio.run(run_query(args))
    elif args.command in ("sessions", "s"):
        exit_code = asyncio.run(run_sessions(args))
    elif args.command in ("server", "serve"):
        exit_code = asyncio.run(run_server(args))
    elif args.command == "stats":
        exit_code = asyncio.run(run_stats(args))
    elif args.command is None:
        parser.print_help()
        exit_code = 0
    else:
        parser.print_help()
        exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
