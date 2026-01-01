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
from ccflow.mcp import MCPConfigManager

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
        self._mcp_manager = MCPConfigManager()

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
        perm_mode = options.permission_mode
        if hasattr(perm_mode, 'value'):
            perm_mode = perm_mode.value
        flags.extend(["--permission-mode", perm_mode])

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

        # Budget limit
        if options.max_budget_usd is not None:
            flags.extend(["--max-budget-usd", str(options.max_budget_usd)])

        # Working directory
        if options.add_dirs:
            for dir_path in options.add_dirs:
                flags.extend(["--add-dir", dir_path])

        # Verbose
        if options.verbose:
            flags.append("--verbose")

        if options.include_partial:
            flags.append("--include-partial-messages")

        # MCP server configuration
        if options.mcp_servers:
            config_path = self._mcp_manager.create_config_file(options.mcp_servers)
            flags.extend(["--mcp-config", str(config_path)])

            if options.strict_mcp:
                # Only use specified MCP servers, ignore other configs
                flags.append("--strict-mcp-config")

        # Debug mode
        if options.debug:
            if isinstance(options.debug, bool):
                flags.append("--debug")
            else:
                flags.extend(["--debug", options.debug])

        # Structured output
        if options.json_schema:
            schema_str = (
                options.json_schema
                if isinstance(options.json_schema, str)
                else json.dumps(options.json_schema)
            )
            flags.extend(["--json-schema", schema_str])

        # Input format
        if options.input_format:
            flags.extend(["--input-format", options.input_format])

        # Permission bypass
        if options.dangerously_skip_permissions:
            flags.append("--dangerously-skip-permissions")

        # Tool specification
        if options.tools is not None:
            if options.tools:
                flags.append("--tools")
                flags.extend(options.tools)
            else:
                # Empty list means disable all tools
                flags.extend(["--tools", ""])

        # Continue most recent session
        if options.continue_session:
            flags.append("--continue")

        # Session persistence
        if options.no_session_persistence:
            flags.append("--no-session-persistence")

        # Agent configuration
        if options.agent:
            flags.extend(["--agent", options.agent])

        if options.agents:
            flags.extend(["--agents", json.dumps(options.agents)])

        # Beta features
        if options.betas:
            flags.append("--betas")
            flags.extend(options.betas)

        # Settings
        if options.settings:
            flags.extend(["--settings", options.settings])

        # Plugin directories
        if options.plugin_dirs:
            for plugin_dir in options.plugin_dirs:
                flags.extend(["--plugin-dir", plugin_dir])

        # Slash commands
        if options.disable_slash_commands:
            flags.append("--disable-slash-commands")

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

    def cleanup(self) -> None:
        """Clean up resources including MCP config files."""
        self._mcp_manager.cleanup()

    def __del__(self) -> None:
        """Clean up on garbage collection."""
        try:
            self.cleanup()
        except Exception:
            pass  # Ignore errors during cleanup


# Module-level default executor
_default_executor: CLIExecutor | None = None


def get_executor() -> CLIExecutor:
    """Get or create the default executor instance."""
    global _default_executor
    if _default_executor is None:
        _default_executor = CLIExecutor()
    return _default_executor
