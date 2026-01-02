"""
ccflow - Claude Code CLI ↔ SDK Middleware

Production middleware bridging Claude Code CLI with SDK-like Python interfaces.
Enables subscription-based usage with TOON token optimization.

Example:
    >>> from ccflow import query, CLIAgentOptions
    >>> async for msg in query("Analyze this code", CLIAgentOptions(model="sonnet")):
    ...     print(msg.content, end="")

Agent System:
    >>> from ccflow import Agent, AgentDefinition, AgentRegistry
    >>> agent = Agent(AgentDefinition(name="reviewer", description="Reviews code", prompt="..."))
    >>> async for msg in agent.execute("Review main.py"):
    ...     print(msg.content, end="")

Hooks:
    >>> from ccflow import HookEvent, HookRegistry, get_hook_registry
    >>> hooks = get_hook_registry()
    >>> @hooks.on(HookEvent.PRE_TOOL_USE, pattern="Bash.*")
    ... async def audit_bash(ctx):
    ...     print(f"Bash: {ctx.tool_input}")
    ...     return ctx

Custom Tools:
    >>> from ccflow import tool, create_sdk_mcp_server
    >>> @tool(description="Add numbers")
    ... async def add(a: int, b: int) -> int:
    ...     return a + b
"""

from ccflow.agent import (
    Agent,
    AgentRegistry,
    get_agent_registry,
    parse_yaml_frontmatter,
    reset_agent_registry,
)
from ccflow.api import batch_query, query
from ccflow.api_client import (
    APIClient,
    APIClientConfig,
    APIClientError,
    APINotAvailableError,
    APIRateLimitError,
    APIResponse,
    FallbackConfig,
    FallbackExecutor,
    get_api_client,
    get_fallback_executor,
    reset_api_client,
    reset_fallback_executor,
)
from ccflow.commands import (
    CommandDefinition,
    CommandRegistry,
    get_command_registry,
    handle_command,
    parse_command,
    reset_command_registry,
)
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
    ThinkingReceivedEvent,
    TokensUsedEvent,
    ToolCalledEvent,
    ToolResultEvent,
    TurnCompletedEvent,
    TurnStartedEvent,
    get_emitter,
    reset_emitter,
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
from ccflow.hooks import (
    HookCallback,
    HookContext,
    HookEvent,
    HookMatcher,
    HookRegistry,
    get_hook_registry,
    reset_hook_registry,
)
from ccflow.manager import SessionManager, get_manager, init_manager
from ccflow.metrics_handlers import (
    PrometheusEventHandler,
    setup_metrics,
    start_metrics_server,
)
from ccflow.parser import (
    collect_text,
    collect_thinking,
    extract_thinking_from_assistant,
    extract_thinking_tokens,
)
from ccflow.pool import (
    PoolConfig,
    PoolStats,
    PoolTask,
    ProcessPool,
    StreamingPool,
    StreamingTask,
    TaskStatus,
    get_pool,
    get_streaming_pool,
    reset_pools,
)
from ccflow.pricing import (
    ModelPricing,
    ModelTier,
    UsageStats,
    calculate_cost,
    extract_model_from_events,
    extract_usage_from_events,
    get_pricing,
    get_usage_tracker,
    reset_usage_tracker,
    track_usage,
)
from ccflow.project import Project
from ccflow.rate_limiting import (
    CombinedLimiter,
    ConcurrencyLimiter,
    ConcurrencyLimitExceededError,
    RateLimiterStats,
    RateLimitExceededError,
    RetryHandler,
    SlidingWindowRateLimiter,
    TokenBucketRateLimiter,
    get_limiter,
    reset_limiter,
)
from ccflow.reliability import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
    CircuitBreakerStats,
    CircuitState,
    HealthChecker,
    HealthStatus,
    RetryConfig,
    RetryExhaustedError,
    RetryStats,
    bind_correlation_id,
    calculate_delay,
    get_cli_circuit_breaker,
    get_correlation_id,
    get_health_checker,
    reset_cli_circuit_breaker,
    reset_health_checker,
    retry_with_backoff,
    set_correlation_id,
    with_retry,
)
from ccflow.session import Session, load_session, resume_session
from ccflow.skills import (
    SkillLoader,
    get_skill_loader,
    reset_skill_loader,
)
from ccflow.store import (
    BaseSessionStore,
    SessionFilter,
    SessionMetadata,
    SessionState,
    SessionStatus,
    SessionStore,
)
from ccflow.stores import (
    MemorySessionStore,
    SQLiteProjectStore,
    SQLiteSessionStore,
    SQLiteTraceStore,
)
from ccflow.subagent import (
    SubagentCoordinator,
    SubagentTask,
    get_subagent_coordinator,
    reset_subagent_coordinator,
)
from ccflow.tools import (
    McpSdkServerConfig,
    SdkMcpServer,
    SdkMcpTool,
    TaskToolUse,
    create_sdk_mcp_server,
    parse_task_tool,
    tool,
)
from ccflow.trace_store import (
    BaseProjectStore,
    BaseTraceStore,
    ProjectStore,
    TraceStore,
)
from ccflow.harness import (
    AgentHarness,
    Feature,
    FeatureList,
    FeatureStatus,
    ProgressEntry,
    ProgressTracker,
    get_harness,
    init_harness,
    reset_harness,
)
from ccflow.tracing import TracingSession, create_tracing_session
from ccflow.types import (
    AgentDefinition,
    AssistantMessage,
    CanUseTool,
    CLIAgentOptions,
    HookMessage,
    InitMessage,
    MCPServerConfig,
    Message,
    PermissionMode,
    PermissionResult,
    ProjectData,
    ProjectFilter,
    QueryResult,
    ResultMessage,
    SessionStats,
    SkillDefinition,
    StopMessage,
    TextMessage,
    ThinkingMessage,
    ToolResultMessage,
    ToolUseMessage,
    ToonConfig,
    TraceData,
    TraceFilter,
    TraceStatus,
    UnknownMessage,
    UnknownSystemMessage,
    UserMessage,
)

__version__ = "0.2.0"
__all__ = [
    # API Fallback
    "APIClient",
    "APIClientConfig",
    "APIClientError",
    "APINotAvailableError",
    "APIRateLimitError",
    "APIResponse",
    # Agent System
    "Agent",
    "AgentDefinition",
    "AgentHarness",
    "AgentRegistry",
    "AssistantMessage",
    # Store base classes
    "BaseProjectStore",
    "BaseSessionStore",
    "BaseTraceStore",
    # Exceptions
    "CCFlowError",
    # Configuration
    "CLIAgentOptions",
    "CLIAuthenticationError",
    "CLIExecutionError",
    "CLINotFoundError",
    "CLITimeoutError",
    # Permission callback
    "CanUseTool",
    # Reliability (Circuit Breaker, Retry, Health)
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerError",
    "CircuitBreakerStats",
    "CircuitState",
    "CombinedLimiter",
    # Commands System
    "CommandDefinition",
    "CommandRegistry",
    "ConcurrencyLimitExceededError",
    "ConcurrencyLimiter",
    "CostIncurredEvent",
    "CostTracker",
    # Events
    "Event",
    "EventEmitter",
    "EventType",
    "FallbackConfig",
    "FallbackExecutor",
    # Harness (Agentic Workflows)
    "Feature",
    "FeatureList",
    "FeatureStatus",
    "HealthChecker",
    "HealthStatus",
    # Hook System
    "HookCallback",
    "HookContext",
    "HookEvent",
    "HookMatcher",
    "HookMessage",
    "HookRegistry",
    "InitMessage",
    "LoggingHandler",
    "MCPServerConfig",
    # MCP SDK Tools
    "McpSdkServerConfig",
    "MemorySessionStore",
    # Message types
    "Message",
    "MetricsHandler",
    # Pricing and Usage
    "ModelPricing",
    "ModelTier",
    "ParseError",
    "PermissionDeniedError",
    "PermissionMode",
    "PermissionResult",
    "PoolConfig",
    "PoolStats",
    "PoolTask",
    # Process Pool
    "ProcessPool",
    "ProgressEntry",
    "ProgressTracker",
    # Project and Trace system
    "Project",
    "ProjectData",
    "ProjectFilter",
    "ProjectStore",
    # Prometheus Metrics
    "PrometheusEventHandler",
    # Results
    "QueryResult",
    "RateLimitExceededError",
    "RateLimiterStats",
    "ResultMessage",
    "RetryConfig",
    "RetryExhaustedError",
    "RetryHandler",
    "RetryStats",
    "SQLiteProjectStore",
    "SQLiteSessionStore",
    "SQLiteTraceStore",
    # SDK MCP Server
    "SdkMcpServer",
    "SdkMcpTool",
    "Session",
    "SessionClosedEvent",
    "SessionCreatedEvent",
    "SessionDeletedEvent",
    "SessionErrorEvent",
    "SessionEvent",
    "SessionFilter",
    "SessionLoadedEvent",
    # Session Management
    "SessionManager",
    "SessionMetadata",
    "SessionNotFoundError",
    "SessionPersistedEvent",
    "SessionResumedEvent",
    "SessionState",
    "SessionStats",
    "SessionStatus",
    # Session Persistence
    "SessionStore",
    "SessionStoreError",
    # Skill System
    "SkillDefinition",
    "SkillLoader",
    "SlidingWindowRateLimiter",
    "StopMessage",
    "StreamingPool",
    "StreamingTask",
    # Subagent System
    "SubagentCoordinator",
    "SubagentTask",
    "TaskStatus",
    "TaskToolUse",
    "TextMessage",
    "ThinkingMessage",
    "ThinkingReceivedEvent",
    # Rate Limiting
    "TokenBucketRateLimiter",
    "TokensUsedEvent",
    "ToolCalledEvent",
    "ToolResultEvent",
    "ToolResultMessage",
    "ToolUseMessage",
    "ToonConfig",
    "ToonEncodingError",
    # Trace system
    "TraceData",
    "TraceFilter",
    "TraceStatus",
    "TraceStore",
    "TracingSession",
    "TurnCompletedEvent",
    "TurnStartedEvent",
    "UnknownMessage",
    "UnknownSystemMessage",
    "UsageStats",
    "UserMessage",
    "batch_query",
    "bind_correlation_id",
    "calculate_cost",
    "calculate_delay",
    # Parser helpers
    "collect_text",
    "collect_thinking",
    "create_sdk_mcp_server",
    "create_tracing_session",
    "extract_model_from_events",
    "extract_thinking_from_assistant",
    "extract_thinking_tokens",
    "extract_usage_from_events",
    "get_agent_registry",
    "get_api_client",
    "get_cli_circuit_breaker",
    "get_command_registry",
    # Correlation IDs
    "get_correlation_id",
    "get_emitter",
    "get_fallback_executor",
    "get_harness",
    "get_health_checker",
    "get_hook_registry",
    "get_limiter",
    "get_manager",
    "get_pool",
    "get_pricing",
    "get_skill_loader",
    "get_streaming_pool",
    "get_subagent_coordinator",
    "get_usage_tracker",
    "handle_command",
    "init_harness",
    "init_manager",
    "load_session",
    "parse_command",
    "parse_task_tool",
    "parse_yaml_frontmatter",
    # Core API
    "query",
    "reset_agent_registry",
    "reset_api_client",
    "reset_cli_circuit_breaker",
    "reset_command_registry",
    "reset_emitter",
    "reset_fallback_executor",
    "reset_harness",
    "reset_health_checker",
    "reset_hook_registry",
    "reset_limiter",
    "reset_pools",
    "reset_skill_loader",
    "reset_subagent_coordinator",
    "reset_usage_tracker",
    "resume_session",
    "retry_with_backoff",
    "set_correlation_id",
    "setup_metrics",
    "start_metrics_server",
    "tool",
    "track_usage",
    "with_retry",
]
