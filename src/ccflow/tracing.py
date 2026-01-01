"""
TracingSession - Session subclass that auto-records full traces.

Extends Session to capture complete prompt/response cycles including
thinking content, tool calls, and metrics for replay and analysis.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict
from datetime import datetime
from typing import TYPE_CHECKING, Any

import structlog

from ccflow.session import Session
from ccflow.types import (
    CLIAgentOptions,
    Message,
    StopMessage,
    TextMessage,
    ThinkingMessage,
    ToolResultMessage,
    ToolUseMessage,
    TraceData,
    TraceStatus,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from ccflow.events import EventEmitter
    from ccflow.executor import CLIExecutor
    from ccflow.store import SessionStore
    from ccflow.trace_store import TraceStore

logger = structlog.get_logger(__name__)


class TracingSession(Session):
    """Session subclass that auto-records full traces.

    Captures complete prompt/response cycles including thinking content,
    tool calls, and metrics. Supports both turn-level and optional
    message-level detail capture.

    Example:
        >>> session = TracingSession(
        ...     options=CLIAgentOptions(model="sonnet", ultrathink=True),
        ...     trace_store=SQLiteTraceStore(),
        ...     detailed=True,  # Capture message-level stream
        ... )
        >>> async for msg in session.send_message("Analyze this code"):
        ...     print(msg.content, end="")
        >>> trace = session.last_trace
        >>> print(f"Thinking tokens: {trace.thinking_tokens}")
        >>> print(f"Tool calls: {len(trace.tool_calls)}")
    """

    def __init__(
        self,
        session_id: str | None = None,
        options: CLIAgentOptions | None = None,
        executor: CLIExecutor | None = None,
        store: SessionStore | None = None,
        emitter: EventEmitter | None = None,
        *,
        project_id: str | None = None,
        trace_store: TraceStore | None = None,
        detailed: bool = False,
    ) -> None:
        """Initialize tracing session.

        Args:
            session_id: Specific session UUID (auto-generated if None)
            options: CLI agent options
            executor: CLI executor instance (uses default if None)
            store: Optional session store for persistence
            emitter: Optional event emitter (uses global if None)
            project_id: Parent project ID for organizing traces
            trace_store: Store for persisting traces
            detailed: Capture message-level stream detail
        """
        super().__init__(
            session_id=session_id,
            options=options,
            executor=executor,
            store=store,
            emitter=emitter,
        )

        self._project_id = project_id
        self._trace_store = trace_store
        self._detailed = detailed
        self._current_trace: TraceData | None = None
        self._sequence_number = 0

        logger.debug(
            "tracing_session_created",
            session_id=self._session_id,
            project_id=project_id,
            detailed=detailed,
        )

    @property
    def project_id(self) -> str | None:
        """Get parent project ID."""
        return self._project_id

    @property
    def last_trace(self) -> TraceData | None:
        """Get the most recent trace."""
        return self._current_trace

    @property
    def trace_count(self) -> int:
        """Get number of traces recorded."""
        return self._sequence_number

    async def send_message(
        self,
        content: str,
        context: dict | None = None,
        *,
        detailed: bool | None = None,
    ) -> AsyncIterator[Message]:
        """Send message with full trace recording.

        Args:
            content: Message content to send
            context: Optional additional context (TOON-encoded if enabled)
            detailed: Override session-level detail capture for this call

        Yields:
            Message objects from Claude's response

        Raises:
            RuntimeError: If session is closed
        """
        if self._is_closed:
            raise RuntimeError("Cannot send message on closed session")

        capture_detail = detailed if detailed is not None else self._detailed
        start_time = time.monotonic()

        # Create trace record
        trace_id = str(uuid.uuid4())
        options_dict = self._options_to_dict(self._options)

        self._current_trace = TraceData(
            trace_id=trace_id,
            session_id=self._session_id,
            project_id=self._project_id,
            sequence_number=self._sequence_number,
            prompt=content,
            options_snapshot=options_dict,
            status=TraceStatus.PENDING,
            created_at=datetime.now().isoformat(),
        )

        # Collectors
        response_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        message_stream: list[dict[str, Any]] | None = [] if capture_detail else None

        try:
            # Use parent's send_message and intercept messages
            async for msg in super().send_message(content, context):
                # Capture message-level detail if enabled
                if capture_detail and message_stream is not None:
                    message_stream.append({
                        "type": type(msg).__name__,
                        "timestamp_ms": int((time.monotonic() - start_time) * 1000),
                        "data": self._message_to_dict(msg),
                    })

                # Collect content by type
                if isinstance(msg, TextMessage):
                    response_parts.append(msg.content)
                elif isinstance(msg, ThinkingMessage):
                    thinking_parts.append(msg.content)
                    self._current_trace.thinking_tokens += msg.thinking_tokens
                elif isinstance(msg, ToolUseMessage):
                    tool_calls.append({
                        "name": msg.tool,
                        "args": msg.args,
                    })
                elif isinstance(msg, ToolResultMessage):
                    if tool_calls:
                        tool_calls[-1]["result"] = msg.result
                elif isinstance(msg, StopMessage):
                    self._current_trace.input_tokens = msg.usage.get("input_tokens", 0)
                    self._current_trace.output_tokens = msg.usage.get("output_tokens", 0)

                yield msg

            # Finalize trace on success
            self._current_trace.response = "".join(response_parts)
            self._current_trace.thinking = "".join(thinking_parts)
            self._current_trace.tool_calls = tool_calls
            self._current_trace.message_stream = message_stream
            self._current_trace.status = TraceStatus.SUCCESS
            self._current_trace.duration_ms = int((time.monotonic() - start_time) * 1000)
            self._current_trace.updated_at = datetime.now().isoformat()

        except Exception as e:
            # Record error in trace
            self._current_trace.status = TraceStatus.ERROR
            self._current_trace.error_message = str(e)
            self._current_trace.duration_ms = int((time.monotonic() - start_time) * 1000)
            self._current_trace.updated_at = datetime.now().isoformat()

            # Persist error trace
            if self._trace_store:
                await self._trace_store.save(self._current_trace)

            raise

        # Persist successful trace
        if self._trace_store:
            await self._trace_store.save(self._current_trace)

        self._sequence_number += 1

        logger.debug(
            "trace_recorded",
            trace_id=trace_id,
            session_id=self._session_id,
            duration_ms=self._current_trace.duration_ms,
            tool_count=len(tool_calls),
            thinking_tokens=self._current_trace.thinking_tokens,
        )

    def _options_to_dict(self, options: CLIAgentOptions) -> dict[str, Any]:
        """Convert CLIAgentOptions to dict for storage."""
        return {
            "model": options.model,
            "fallback_model": options.fallback_model,
            "system_prompt": options.system_prompt,
            "append_system_prompt": options.append_system_prompt,
            "permission_mode": options.permission_mode.value if options.permission_mode else None,
            "allowed_tools": options.allowed_tools,
            "disallowed_tools": options.disallowed_tools,
            "max_budget_usd": options.max_budget_usd,
            "timeout": options.timeout,
            "ultrathink": options.ultrathink,
        }

    def _message_to_dict(self, msg: Message) -> dict[str, Any]:
        """Convert message to dict for storage."""
        if hasattr(msg, "__dataclass_fields__"):
            return asdict(msg)  # type: ignore[arg-type]
        return {"raw": str(msg)}

    async def get_traces(self, limit: int = 100) -> list[TraceData]:
        """Get all traces for this session.

        Args:
            limit: Maximum traces to return

        Returns:
            List of traces ordered by sequence number
        """
        if not self._trace_store:
            return []
        return await self._trace_store.get_session_traces(self._session_id, limit)


async def create_tracing_session(
    project_id: str | None = None,
    options: CLIAgentOptions | None = None,
    trace_store: TraceStore | None = None,
    session_store: SessionStore | None = None,
    detailed: bool = False,
) -> TracingSession:
    """Create a new tracing session.

    Convenience function for creating TracingSession with common configuration.

    Args:
        project_id: Parent project ID
        options: CLI agent options
        trace_store: Store for traces
        session_store: Store for session state
        detailed: Capture message-level detail

    Returns:
        Configured TracingSession
    """
    return TracingSession(
        options=options,
        store=session_store,
        trace_store=trace_store,
        project_id=project_id,
        detailed=detailed,
    )
