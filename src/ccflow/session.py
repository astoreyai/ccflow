"""
Session Manager - Stateful multi-turn conversation management.

Maps SDK Session interface to CLI --resume functionality,
tracking conversation state and usage statistics.
"""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING, AsyncIterator

import structlog

from ccflow.executor import CLIExecutor, get_executor
from ccflow.parser import StreamParser
from ccflow.toon_integration import ToonSerializer
from ccflow.types import (
    CLIAgentOptions,
    InitMessage,
    Message,
    SessionStats,
    StopMessage,
    TextMessage,
)

if TYPE_CHECKING:
    pass

logger = structlog.get_logger(__name__)


class Session:
    """Stateful session matching Agent SDK Session interface.

    Manages multi-turn conversations via CLI --resume flag,
    tracking usage statistics and supporting session forking.

    Example:
        >>> session = Session(options=CLIAgentOptions(model="opus"))
        >>> async for msg in session.send_message("Review this code"):
        ...     print(msg.content, end="")
        >>> async for msg in session.send_message("Focus on security"):
        ...     print(msg.content, end="")
        >>> stats = await session.close()
        >>> print(f"Total tokens: {stats.total_tokens}")
    """

    def __init__(
        self,
        session_id: str | None = None,
        options: CLIAgentOptions | None = None,
        executor: CLIExecutor | None = None,
    ) -> None:
        """Initialize session.

        Args:
            session_id: Specific session UUID (auto-generated if None)
            options: CLI agent options
            executor: CLI executor instance (uses default if None)
        """
        self._session_id = session_id or str(uuid.uuid4())
        self._options = options or CLIAgentOptions()
        self._executor = executor or get_executor()
        self._parser = StreamParser()
        self._toon = ToonSerializer(self._options.toon)

        # Statistics
        self._turn_count = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._start_time = time.monotonic()
        self._is_closed = False

        # Track if this is a fresh session or resumed
        self._is_first_message = True

        logger.info("session_created", session_id=self._session_id)

    @property
    def session_id(self) -> str:
        """Get session UUID."""
        return self._session_id

    @property
    def turn_count(self) -> int:
        """Get number of conversation turns."""
        return self._turn_count

    @property
    def is_closed(self) -> bool:
        """Check if session is closed."""
        return self._is_closed

    async def send_message(
        self,
        content: str,
        context: dict | None = None,
    ) -> AsyncIterator[Message]:
        """Send message and stream response.

        Args:
            content: Message content to send
            context: Optional additional context (TOON-encoded if enabled)

        Yields:
            Message objects from Claude's response

        Raises:
            RuntimeError: If session is closed
        """
        if self._is_closed:
            raise RuntimeError("Cannot send message on closed session")

        self._turn_count += 1

        # Build options for this turn
        turn_options = CLIAgentOptions(
            model=self._options.model,
            fallback_model=self._options.fallback_model,
            system_prompt=self._options.system_prompt,
            append_system_prompt=self._options.append_system_prompt,
            permission_mode=self._options.permission_mode,
            allowed_tools=self._options.allowed_tools,
            disallowed_tools=self._options.disallowed_tools,
            session_id=self._session_id,
            resume=not self._is_first_message,  # Resume after first message
            max_turns=self._options.max_turns,
            timeout=self._options.timeout,
            cwd=self._options.cwd,
            add_dirs=self._options.add_dirs,
            mcp_servers=self._options.mcp_servers,
            strict_mcp=self._options.strict_mcp,
            toon=self._options.toon,
            verbose=self._options.verbose,
            include_partial=self._options.include_partial,
        )

        # Inject context if provided
        if context and self._options.toon.encode_context:
            context_str = self._toon.format_for_prompt(context, label="Turn Context")
            existing = turn_options.append_system_prompt or ""
            turn_options.append_system_prompt = existing + context_str

        # Build CLI flags
        flags = self._executor.build_flags(turn_options)

        # Execute and stream
        async for event in self._executor.execute(
            content,
            flags,
            timeout=self._options.timeout,
            cwd=self._options.cwd,
        ):
            msg = self._parser.parse_event(event)

            # Update session ID from init message if needed
            if isinstance(msg, InitMessage):
                if self._is_first_message:
                    self._session_id = msg.session_id
                    logger.debug("session_id_updated", session_id=self._session_id)

            # Track usage from stop message
            if isinstance(msg, StopMessage):
                self._total_input_tokens += msg.usage.get("input_tokens", 0)
                self._total_output_tokens += msg.usage.get("output_tokens", 0)
                logger.debug(
                    "turn_complete",
                    turn=self._turn_count,
                    input_tokens=msg.usage.get("input_tokens", 0),
                    output_tokens=msg.usage.get("output_tokens", 0),
                )

            yield msg

        self._is_first_message = False

    async def fork(self) -> Session:
        """Create new session branched from current state.

        The forked session starts with the conversation history
        from this session but gets a new session ID.

        Returns:
            New Session instance
        """
        fork_options = CLIAgentOptions(
            model=self._options.model,
            fallback_model=self._options.fallback_model,
            system_prompt=self._options.system_prompt,
            append_system_prompt=self._options.append_system_prompt,
            permission_mode=self._options.permission_mode,
            allowed_tools=self._options.allowed_tools,
            disallowed_tools=self._options.disallowed_tools,
            session_id=self._session_id,
            resume=True,
            fork_session=True,  # This creates new ID from current state
            max_turns=self._options.max_turns,
            timeout=self._options.timeout,
            cwd=self._options.cwd,
            add_dirs=self._options.add_dirs,
            toon=self._options.toon,
        )

        forked = Session(
            session_id=None,  # Will get new ID from CLI
            options=fork_options,
            executor=self._executor,
        )

        logger.info(
            "session_forked",
            parent_session=self._session_id,
            child_session=forked.session_id,
        )

        return forked

    async def close(self) -> SessionStats:
        """Close session and return statistics.

        Returns:
            SessionStats with usage information
        """
        if self._is_closed:
            logger.warning("session_already_closed", session_id=self._session_id)

        self._is_closed = True
        duration = time.monotonic() - self._start_time

        stats = SessionStats(
            session_id=self._session_id,
            total_turns=self._turn_count,
            total_input_tokens=self._total_input_tokens,
            total_output_tokens=self._total_output_tokens,
            duration_seconds=duration,
            toon_savings_ratio=self._options.toon.last_compression_ratio,
        )

        logger.info(
            "session_closed",
            session_id=self._session_id,
            turns=stats.total_turns,
            total_tokens=stats.total_tokens,
            duration_seconds=f"{duration:.2f}",
        )

        return stats


async def resume_session(
    session_id: str,
    options: CLIAgentOptions | None = None,
) -> Session:
    """Resume an existing session by ID.

    Args:
        session_id: Session UUID to resume
        options: Optional options override

    Returns:
        Session instance configured to resume
    """
    opts = options or CLIAgentOptions()
    opts.session_id = session_id
    opts.resume = True

    session = Session(session_id=session_id, options=opts)
    session._is_first_message = False  # Mark as resumed

    logger.info("session_resumed", session_id=session_id)

    return session
