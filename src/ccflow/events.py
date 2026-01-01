"""
Event System - Typed events and hooks for session lifecycle.

Provides an extensible event system for monitoring and reacting to
session events, message flow, and usage statistics.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Awaitable, Callable, TypeVar

import structlog

logger = structlog.get_logger(__name__)


class EventType(str, Enum):
    """Types of events that can be emitted."""

    # Session lifecycle
    SESSION_CREATED = "session.created"
    SESSION_RESUMED = "session.resumed"
    SESSION_CLOSED = "session.closed"
    SESSION_ERROR = "session.error"

    # Message flow
    TURN_STARTED = "turn.started"
    TURN_COMPLETED = "turn.completed"
    MESSAGE_RECEIVED = "message.received"

    # Tool events
    TOOL_CALLED = "tool.called"
    TOOL_RESULT = "tool.result"

    # Usage tracking
    TOKENS_USED = "tokens.used"
    COST_INCURRED = "cost.incurred"

    # Store events
    SESSION_PERSISTED = "session.persisted"
    SESSION_LOADED = "session.loaded"
    SESSION_DELETED = "session.deleted"


@dataclass
class Event:
    """Base event class with common fields."""

    type: EventType
    timestamp: datetime = field(default_factory=datetime.now)
    session_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionEvent(Event):
    """Event related to session lifecycle."""

    model: str | None = None
    turn_count: int = 0
    total_tokens: int = 0


@dataclass
class SessionCreatedEvent(SessionEvent):
    """Emitted when a new session is created."""

    type: EventType = field(default=EventType.SESSION_CREATED)
    tags: list[str] = field(default_factory=list)


@dataclass
class SessionResumedEvent(SessionEvent):
    """Emitted when a session is resumed."""

    type: EventType = field(default=EventType.SESSION_RESUMED)
    previous_turn_count: int = 0


@dataclass
class SessionClosedEvent(SessionEvent):
    """Emitted when a session is closed."""

    type: EventType = field(default=EventType.SESSION_CLOSED)
    duration_seconds: float = 0.0
    total_cost_usd: float = 0.0


@dataclass
class SessionErrorEvent(SessionEvent):
    """Emitted when a session encounters an error."""

    type: EventType = field(default=EventType.SESSION_ERROR)
    error_type: str = ""
    error_message: str = ""


@dataclass
class TurnEvent(Event):
    """Event related to conversation turns."""

    turn_number: int = 0
    prompt: str = ""


@dataclass
class TurnStartedEvent(TurnEvent):
    """Emitted when a new turn starts."""

    type: EventType = field(default=EventType.TURN_STARTED)
    context_size: int = 0


@dataclass
class TurnCompletedEvent(TurnEvent):
    """Emitted when a turn completes."""

    type: EventType = field(default=EventType.TURN_COMPLETED)
    response: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    duration_seconds: float = 0.0


@dataclass
class MessageEvent(Event):
    """Event for individual messages."""

    type: EventType = field(default=EventType.MESSAGE_RECEIVED)
    message_type: str = ""
    content: str = ""


@dataclass
class ToolEvent(Event):
    """Event for tool usage."""

    tool_name: str = ""
    tool_id: str = ""


@dataclass
class ToolCalledEvent(ToolEvent):
    """Emitted when a tool is called."""

    type: EventType = field(default=EventType.TOOL_CALLED)
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResultEvent(ToolEvent):
    """Emitted when a tool returns a result."""

    type: EventType = field(default=EventType.TOOL_RESULT)
    result: str = ""
    success: bool = True
    error: str | None = None


@dataclass
class TokensUsedEvent(Event):
    """Emitted when tokens are consumed."""

    type: EventType = field(default=EventType.TOKENS_USED)
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""


@dataclass
class CostIncurredEvent(Event):
    """Emitted when cost is incurred."""

    type: EventType = field(default=EventType.COST_INCURRED)
    amount_usd: float = 0.0
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class SessionPersistedEvent(Event):
    """Emitted when session state is persisted."""

    type: EventType = field(default=EventType.SESSION_PERSISTED)


@dataclass
class SessionLoadedEvent(Event):
    """Emitted when session is loaded from store."""

    type: EventType = field(default=EventType.SESSION_LOADED)
    turn_count: int = 0
    total_tokens: int = 0


@dataclass
class SessionDeletedEvent(Event):
    """Emitted when session is deleted."""

    type: EventType = field(default=EventType.SESSION_DELETED)


# Type for event handlers
E = TypeVar("E", bound=Event)
EventHandler = Callable[[E], Awaitable[None] | None]
SyncEventHandler = Callable[[E], None]
AsyncEventHandler = Callable[[E], Awaitable[None]]


class EventEmitter:
    """Event emitter with typed event support.

    Supports both sync and async handlers. Async handlers are
    awaited while sync handlers are called directly.

    Example:
        >>> emitter = EventEmitter()
        >>> @emitter.on(EventType.SESSION_CREATED)
        ... async def on_session_created(event: SessionCreatedEvent):
        ...     print(f"Session {event.session_id} created")
        >>> await emitter.emit(SessionCreatedEvent(session_id="abc"))
    """

    def __init__(self) -> None:
        """Initialize event emitter."""
        self._handlers: dict[EventType, list[EventHandler]] = {}
        self._global_handlers: list[EventHandler] = []

    def on(
        self,
        event_type: EventType | None = None,
    ) -> Callable[[EventHandler[E]], EventHandler[E]]:
        """Decorator to register an event handler.

        Args:
            event_type: Event type to handle, or None for all events.

        Returns:
            Decorator function.
        """

        def decorator(handler: EventHandler[E]) -> EventHandler[E]:
            if event_type is None:
                self._global_handlers.append(handler)
            else:
                if event_type not in self._handlers:
                    self._handlers[event_type] = []
                self._handlers[event_type].append(handler)
            return handler

        return decorator

    def add_handler(
        self,
        handler: EventHandler,
        event_type: EventType | None = None,
    ) -> None:
        """Add an event handler.

        Args:
            handler: Handler function.
            event_type: Event type to handle, or None for all events.
        """
        if event_type is None:
            self._global_handlers.append(handler)
        else:
            if event_type not in self._handlers:
                self._handlers[event_type] = []
            self._handlers[event_type].append(handler)

    def remove_handler(
        self,
        handler: EventHandler,
        event_type: EventType | None = None,
    ) -> bool:
        """Remove an event handler.

        Args:
            handler: Handler to remove.
            event_type: Event type, or None to remove from global.

        Returns:
            True if handler was found and removed.
        """
        if event_type is None:
            if handler in self._global_handlers:
                self._global_handlers.remove(handler)
                return True
        else:
            if event_type in self._handlers and handler in self._handlers[event_type]:
                self._handlers[event_type].remove(handler)
                return True
        return False

    async def emit(self, event: Event) -> None:
        """Emit an event to all registered handlers.

        Args:
            event: Event to emit.
        """
        handlers = list(self._global_handlers)
        if event.type in self._handlers:
            handlers.extend(self._handlers[event.type])

        for handler in handlers:
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.warning(
                    "event_handler_error",
                    event_type=event.type.value,
                    error=str(e),
                )

    def emit_sync(self, event: Event) -> None:
        """Emit an event synchronously (for sync-only handlers).

        Note: Async handlers will be skipped with a warning.

        Args:
            event: Event to emit.
        """
        handlers = list(self._global_handlers)
        if event.type in self._handlers:
            handlers.extend(self._handlers[event.type])

        for handler in handlers:
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    logger.warning(
                        "async_handler_in_sync_emit",
                        event_type=event.type.value,
                    )
                    result.close()  # Prevent warning about unawaited coroutine
            except Exception as e:
                logger.warning(
                    "event_handler_error",
                    event_type=event.type.value,
                    error=str(e),
                )

    def clear(self, event_type: EventType | None = None) -> None:
        """Clear handlers.

        Args:
            event_type: Event type to clear, or None to clear all.
        """
        if event_type is None:
            self._handlers.clear()
            self._global_handlers.clear()
        elif event_type in self._handlers:
            self._handlers[event_type].clear()

    @property
    def handler_count(self) -> int:
        """Get total number of registered handlers."""
        count = len(self._global_handlers)
        for handlers in self._handlers.values():
            count += len(handlers)
        return count


# Pre-built event handlers


class LoggingHandler:
    """Event handler that logs all events.

    Example:
        >>> emitter.add_handler(LoggingHandler())
    """

    def __init__(self, level: str = "debug") -> None:
        """Initialize logging handler.

        Args:
            level: Log level (debug, info, warning, error).
        """
        self._level = level
        self._logger = structlog.get_logger("ccflow.events")

    def __call__(self, event: Event) -> None:
        """Handle event by logging it."""
        log_fn = getattr(self._logger, self._level, self._logger.debug)
        log_fn(
            event.type.value,
            session_id=event.session_id,
            **event.metadata,
        )


class MetricsHandler:
    """Event handler that collects metrics.

    Tracks counts, token usage, costs, and timing.

    Example:
        >>> handler = MetricsHandler()
        >>> emitter.add_handler(handler)
        >>> # Later:
        >>> print(handler.metrics)
    """

    def __init__(self) -> None:
        """Initialize metrics handler."""
        self._metrics: dict[str, Any] = {
            "sessions_created": 0,
            "sessions_closed": 0,
            "turns_completed": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cost_usd": 0.0,
            "tools_called": 0,
            "errors": 0,
        }

    @property
    def metrics(self) -> dict[str, Any]:
        """Get current metrics."""
        return self._metrics.copy()

    def reset(self) -> None:
        """Reset all metrics to zero."""
        for key in self._metrics:
            if isinstance(self._metrics[key], int):
                self._metrics[key] = 0
            else:
                self._metrics[key] = 0.0

    def __call__(self, event: Event) -> None:
        """Handle event by updating metrics."""
        if event.type == EventType.SESSION_CREATED:
            self._metrics["sessions_created"] += 1
        elif event.type == EventType.SESSION_CLOSED:
            self._metrics["sessions_closed"] += 1
            if isinstance(event, SessionClosedEvent):
                self._metrics["total_cost_usd"] += event.total_cost_usd
        elif event.type == EventType.TURN_COMPLETED:
            self._metrics["turns_completed"] += 1
            if isinstance(event, TurnCompletedEvent):
                self._metrics["total_input_tokens"] += event.input_tokens
                self._metrics["total_output_tokens"] += event.output_tokens
        elif event.type == EventType.TOKENS_USED:
            if isinstance(event, TokensUsedEvent):
                self._metrics["total_input_tokens"] += event.input_tokens
                self._metrics["total_output_tokens"] += event.output_tokens
        elif event.type == EventType.COST_INCURRED:
            if isinstance(event, CostIncurredEvent):
                self._metrics["total_cost_usd"] += event.amount_usd
        elif event.type == EventType.TOOL_CALLED:
            self._metrics["tools_called"] += 1
        elif event.type == EventType.SESSION_ERROR:
            self._metrics["errors"] += 1


class CostTracker:
    """Event handler that tracks costs per session.

    Example:
        >>> tracker = CostTracker()
        >>> emitter.add_handler(tracker)
        >>> # Later:
        >>> print(tracker.get_session_cost("session-123"))
    """

    # Approximate costs per 1M tokens (as of 2024)
    COSTS_PER_MILLION = {
        "sonnet": {"input": 3.0, "output": 15.0},
        "opus": {"input": 15.0, "output": 75.0},
        "haiku": {"input": 0.25, "output": 1.25},
        "claude-3-5-sonnet-20241022": {"input": 3.0, "output": 15.0},
        "claude-3-opus-20240229": {"input": 15.0, "output": 75.0},
        "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    }

    def __init__(self) -> None:
        """Initialize cost tracker."""
        self._session_costs: dict[str, float] = {}
        self._session_tokens: dict[str, dict[str, int]] = {}

    def get_session_cost(self, session_id: str) -> float:
        """Get total cost for a session."""
        return self._session_costs.get(session_id, 0.0)

    def get_session_tokens(self, session_id: str) -> dict[str, int]:
        """Get token usage for a session."""
        return self._session_tokens.get(session_id, {"input": 0, "output": 0}).copy()

    def get_total_cost(self) -> float:
        """Get total cost across all sessions."""
        return sum(self._session_costs.values())

    def clear(self) -> None:
        """Clear all tracked costs."""
        self._session_costs.clear()
        self._session_tokens.clear()

    def _calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Calculate cost for token usage."""
        # Normalize model name
        model_key = model.lower()
        for key in self.COSTS_PER_MILLION:
            if key in model_key or model_key in key:
                costs = self.COSTS_PER_MILLION[key]
                return (
                    (input_tokens / 1_000_000) * costs["input"]
                    + (output_tokens / 1_000_000) * costs["output"]
                )

        # Default to sonnet pricing if unknown
        costs = self.COSTS_PER_MILLION["sonnet"]
        return (
            (input_tokens / 1_000_000) * costs["input"]
            + (output_tokens / 1_000_000) * costs["output"]
        )

    def __call__(self, event: Event) -> None:
        """Handle event by tracking costs."""
        if event.session_id is None:
            return

        if event.type == EventType.TURN_COMPLETED and isinstance(
            event, TurnCompletedEvent
        ):
            # Initialize if needed
            if event.session_id not in self._session_costs:
                self._session_costs[event.session_id] = 0.0
                self._session_tokens[event.session_id] = {"input": 0, "output": 0}

            # Track tokens
            self._session_tokens[event.session_id]["input"] += event.input_tokens
            self._session_tokens[event.session_id]["output"] += event.output_tokens

            # Calculate and track cost
            model = event.metadata.get("model", "sonnet")
            cost = self._calculate_cost(model, event.input_tokens, event.output_tokens)
            self._session_costs[event.session_id] += cost

        elif event.type == EventType.TOKENS_USED and isinstance(event, TokensUsedEvent):
            if event.session_id not in self._session_costs:
                self._session_costs[event.session_id] = 0.0
                self._session_tokens[event.session_id] = {"input": 0, "output": 0}

            self._session_tokens[event.session_id]["input"] += event.input_tokens
            self._session_tokens[event.session_id]["output"] += event.output_tokens

            cost = self._calculate_cost(
                event.model, event.input_tokens, event.output_tokens
            )
            self._session_costs[event.session_id] += cost


# Global event emitter instance
_global_emitter: EventEmitter | None = None


def get_emitter() -> EventEmitter:
    """Get or create the global event emitter."""
    global _global_emitter
    if _global_emitter is None:
        _global_emitter = EventEmitter()
    return _global_emitter


def reset_emitter() -> None:
    """Reset the global event emitter."""
    global _global_emitter
    if _global_emitter is not None:
        _global_emitter.clear()
    _global_emitter = None
