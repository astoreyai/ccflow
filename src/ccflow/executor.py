"""
CLI Executor - Low-level subprocess management for claude CLI.

Handles async subprocess spawning, streaming stdout parsing,
and process lifecycle management.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, AsyncIterator

import structlog

from ccflow.exceptions import (
    CLIExecutionError,
    CLINotFoundError,
    CLITimeoutError,
    ParseError,
)

if TYPE_CHECKING:
    from ccflow.types import CLIAgentOptions

logger = structlog.get_logger(__name__)


class CLIExecutor:
    """Manages claude CLI subprocess execution with streaming.

    This is the lowest layer of the middleware stack, responsible for:
    - Spawning claude CLI as async subprocess
    - Streaming NDJSON output line-by-line
    - Process lifecycle and timeout management
    - Error detection and propagation

    Example:
        >>> executor = CLIExecutor()
        >>> async for event in executor.execute("Explain this", ["--output-format", "stream-json"]):
        ...     print(event)
    """

    def __init__(self, cli_path: str | None = None) -> None:
        """Initialize executor.

        Args:
            cli_path: Path to claude CLI. Auto-detected if None.
        """
        self._cli_path = cli_path or self._find_cli()
        self._active_processes: dict[str, asyncio.subprocess.Process] = {}

    def _find_cli(self) -> str:
        """Find claude CLI in PATH."""
        cli = shutil.which("claude")
        if cli is None:
            raise CLINotFoundError(
                "Claude CLI not found. Install with: npm install -g @anthropic-ai/claude-code"
            )
        return cli

    def build_flags(self, options: CLIAgentOptions) -> list[str]:
        """Convert CLIAgentOptions to CLI flags.

        Args:
            options: Configuration options

        Returns:
            List of CLI flags
        """
        flags: list[str] = []

        # Output format (always stream-json for async iteration)
        # Note: stream-json requires --verbose in print mode
        flags.extend(["--output-format", "stream-json", "--verbose"])

        # Model
        if options.model:
            flags.extend(["--model", options.model])

        if options.fallback_model:
            flags.extend(["--fallback-model", options.fallback_model])

        # System prompt
        if options.system_prompt:
            flags.extend(["--system-prompt", options.system_prompt])

        if options.append_system_prompt:
            flags.extend(["--append-system-prompt", options.append_system_prompt])

        # Permissions
        flags.extend(["--permission-mode", options.permission_mode.value])

        if options.allowed_tools:
            flags.append("--allowedTools")
            flags.extend(options.allowed_tools)

        if options.disallowed_tools:
            flags.append("--disallowedTools")
            flags.extend(options.disallowed_tools)

        # Session management
        if options.session_id and options.resume:
            flags.extend(["--resume", options.session_id])
            if options.fork_session:
                flags.append("--fork-session")
        elif options.session_id:
            flags.extend(["--session-id", options.session_id])

        # Execution limits
        if options.max_turns is not None:
            flags.extend(["--max-turns", str(options.max_turns)])

        # Working directory
        if options.add_dirs:
            for dir_path in options.add_dirs:
                flags.extend(["--add-dir", dir_path])

        # Verbose
        if options.verbose:
            flags.append("--verbose")

        if options.include_partial:
            flags.append("--include-partial-messages")

        return flags

    async def execute(
        self,
        prompt: str,
        flags: list[str],
        timeout: float = 300.0,
        cwd: Path | str | None = None,
    ) -> AsyncIterator[dict]:
        """Execute claude CLI and stream NDJSON responses.

        Args:
            prompt: The prompt to send to Claude
            flags: CLI flags (e.g., ["--output-format", "stream-json"])
            timeout: Maximum execution time in seconds
            cwd: Working directory for CLI execution

        Yields:
            Parsed NDJSON events as dictionaries

        Raises:
            CLIExecutionError: If subprocess fails
            CLITimeoutError: If execution exceeds timeout
            ParseError: If NDJSON parsing fails
        """
        cmd = [self._cli_path, "-p", prompt, *flags]

        logger.debug(
            "executing_cli",
            command=cmd[:5],  # Log first 5 elements to avoid logging full prompt
            cwd=str(cwd) if cwd else None,
            timeout=timeout,
        )

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )

            # Track active process for potential cancellation
            process_id = str(id(process))
            self._active_processes[process_id] = process

            try:
                async for event in self._stream_output(process, timeout):
                    yield event
            finally:
                self._active_processes.pop(process_id, None)

            # Check exit code
            await process.wait()
            if process.returncode != 0:
                stderr = ""
                if process.stderr:
                    stderr_bytes = await process.stderr.read()
                    stderr = stderr_bytes.decode("utf-8", errors="replace")

                raise CLIExecutionError(
                    f"CLI exited with code {process.returncode}",
                    stderr=stderr,
                    exit_code=process.returncode,
                )

        except asyncio.TimeoutError as e:
            raise CLITimeoutError(timeout) from e

    async def _stream_output(
        self,
        process: asyncio.subprocess.Process,
        timeout: float,
    ) -> AsyncIterator[dict]:
        """Stream and parse NDJSON output from process.

        Args:
            process: Running subprocess
            timeout: Timeout in seconds

        Yields:
            Parsed JSON objects from each line
        """
        if process.stdout is None:
            return

        line_number = 0
        deadline = asyncio.get_event_loop().time() + timeout

        async for line_bytes in process.stdout:
            # Check timeout
            if asyncio.get_event_loop().time() > deadline:
                process.terminate()
                raise CLITimeoutError(timeout)

            line_number += 1
            line = line_bytes.decode("utf-8", errors="replace").strip()

            if not line:
                continue

            try:
                event = json.loads(line)
                logger.debug("parsed_event", event_type=event.get("type"), line=line_number)
                yield event
            except json.JSONDecodeError as e:
                logger.warning(
                    "json_parse_error",
                    line=line[:100],
                    line_number=line_number,
                    error=str(e),
                )
                raise ParseError(
                    f"Invalid JSON: {e}",
                    line=line,
                    line_number=line_number,
                ) from e

    async def cancel_all(self) -> None:
        """Cancel all active processes."""
        for process_id, process in list(self._active_processes.items()):
            try:
                process.terminate()
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                process.kill()
            finally:
                self._active_processes.pop(process_id, None)

    async def check_cli_available(self) -> bool:
        """Check if CLI is available and authenticated.

        Returns:
            True if CLI is ready for use
        """
        try:
            process = await asyncio.create_subprocess_exec(
                self._cli_path,
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await process.wait()
            return process.returncode == 0
        except Exception:
            return False


# Module-level default executor
_default_executor: CLIExecutor | None = None


def get_executor() -> CLIExecutor:
    """Get or create the default executor instance."""
    global _default_executor
    if _default_executor is None:
        _default_executor = CLIExecutor()
    return _default_executor
