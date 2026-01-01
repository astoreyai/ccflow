"""
ccflow - Claude Code CLI ↔ SDK Middleware

Production middleware bridging Claude Code CLI with SDK-like Python interfaces.
Enables subscription-based usage with TOON token optimization.

Example:
    >>> from ccflow import query, CLIAgentOptions
    >>> async for msg in query("Analyze this code", CLIAgentOptions(model="sonnet")):
    ...     print(msg.content, end="")
"""

from ccflow.api import batch_query, query
from ccflow.events import (
    CostIncurredEvent,
    CostTracker,
    Event,
    EventEmitter,
    EventType,
    LoggingHandler,
    MetricsHandler,
    SessionClosedEvent,
    SessionCreatedEvent,
    SessionDeletedEvent,
    SessionErrorEvent,
    SessionEvent,
    SessionLoadedEvent,
    SessionPersistedEvent,
    SessionResumedEvent,
    TokensUsedEvent,
    ToolCalledEvent,
    ToolResultEvent,
    TurnCompletedEvent,
    TurnStartedEvent,
    get_emitter,
    reset_emitter,
)
from ccflow.manager import SessionManager, get_manager, init_manager
from ccflow.metrics_handlers import (
    PrometheusEventHandler,
    setup_metrics,
    start_metrics_server,
)
from ccflow.rate_limiting import (
    CombinedLimiter,
    ConcurrencyLimiter,
    ConcurrencyLimitExceededError,
    RateLimitExceededError,
    RateLimiterStats,
    RetryConfig,
    RetryHandler,
    SlidingWindowRateLimiter,
    TokenBucketRateLimiter,
    get_limiter,
    reset_limiter,
)
from ccflow.exceptions import (
    CCFlowError,
    CLIAuthenticationError,
    CLIExecutionError,
    CLINotFoundError,
    CLITimeoutError,
    ParseError,
    PermissionDeniedError,
    SessionNotFoundError,
    SessionStoreError,
    ToonEncodingError,
)
from ccflow.session import Session, load_session, resume_session
from ccflow.store import (
    BaseSessionStore,
    SessionFilter,
    SessionMetadata,
    SessionState,
    SessionStatus,
    SessionStore,
)
from ccflow.stores import MemorySessionStore, SQLiteSessionStore
from ccflow.types import (
    AssistantMessage,
    CLIAgentOptions,
    HookMessage,
    InitMessage,
    MCPServerConfig,
    Message,
    PermissionMode,
    QueryResult,
    ResultMessage,
    SessionStats,
    StopMessage,
    TextMessage,
    ToonConfig,
    ToolResultMessage,
    ToolUseMessage,
    UnknownMessage,
    UnknownSystemMessage,
    UserMessage,
)

__version__ = "0.1.0"
__all__ = [
    # Core API
    "query",
    "batch_query",
    "Session",
    "load_session",
    "resume_session",
    # Session Management
    "SessionManager",
    "get_manager",
    "init_manager",
    # Session Persistence
    "SessionStore",
    "BaseSessionStore",
    "SQLiteSessionStore",
    "MemorySessionStore",
    "SessionState",
    "SessionMetadata",
    "SessionFilter",
    "SessionStatus",
    # Events
    "Event",
    "EventType",
    "EventEmitter",
    "get_emitter",
    "reset_emitter",
    "SessionEvent",
    "SessionCreatedEvent",
    "SessionResumedEvent",
    "SessionClosedEvent",
    "SessionErrorEvent",
    "SessionPersistedEvent",
    "SessionLoadedEvent",
    "SessionDeletedEvent",
    "TurnStartedEvent",
    "TurnCompletedEvent",
    "ToolCalledEvent",
    "ToolResultEvent",
    "TokensUsedEvent",
    "CostIncurredEvent",
    "LoggingHandler",
    "MetricsHandler",
    "CostTracker",
    # Prometheus Metrics
    "PrometheusEventHandler",
    "setup_metrics",
    "start_metrics_server",
    # Rate Limiting
    "TokenBucketRateLimiter",
    "SlidingWindowRateLimiter",
    "ConcurrencyLimiter",
    "CombinedLimiter",
    "RetryHandler",
    "RetryConfig",
    "RateLimiterStats",
    "RateLimitExceededError",
    "ConcurrencyLimitExceededError",
    "get_limiter",
    "reset_limiter",
    # Configuration
    "CLIAgentOptions",
    "ToonConfig",
    "MCPServerConfig",
    "PermissionMode",
    # Message types
    "Message",
    "InitMessage",
    "TextMessage",
    "ToolUseMessage",
    "ToolResultMessage",
    "StopMessage",
    "HookMessage",
    "AssistantMessage",
    "UserMessage",
    "ResultMessage",
    "UnknownSystemMessage",
    "UnknownMessage",
    # Results
    "QueryResult",
    "SessionStats",
    # Exceptions
    "CCFlowError",
    "CLINotFoundError",
    "CLIAuthenticationError",
    "CLIExecutionError",
    "CLITimeoutError",
    "SessionNotFoundError",
    "SessionStoreError",
    "ParseError",
    "ToonEncodingError",
    "PermissionDeniedError",
]
