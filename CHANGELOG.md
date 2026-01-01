# Changelog

All notable changes to ccflow will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-01-01

### Added

#### Core API
- `query()` - Async streaming query with typed messages
- `query_simple()` - Simple string response
- `batch_query()` - Concurrent batch processing with configurable parallelism
- `stream_to_callback()` - Callback-based streaming

#### Session Management
- `Session` class for multi-turn conversations
- `SessionManager` for lifecycle management
- `load_session()` and `resume_session()` helpers
- Auto-persistence with configurable stores

#### Project & Trace System
- `Project` class for hierarchical organization of sessions
- `TracingSession` - Session subclass that auto-records full traces
- `TraceData` - Complete capture of prompt/response/thinking/tools per turn
- `TraceStatus` enum (pending, success, error, cancelled)
- `TraceStore` and `ProjectStore` protocols
- `SQLiteTraceStore` and `SQLiteProjectStore` implementations
- Nested sub-projects via `parent_project_id`
- Replay capability: `replay_as_new()` and `replay_fork()` methods
- Message-level detail capture with `detailed=True` option
- Aggregate analysis via `get_trace_summary()`

#### Extended Thinking (Ultrathink)
- `ultrathink` option in `CLIAgentOptions` to enable deep reasoning
- `ThinkingMessage` type for thinking content blocks
- `ThinkingReceivedEvent` for event-driven thinking tracking
- `thinking_tokens` field in usage events
- Parser helpers: `collect_thinking()`, `extract_thinking_from_assistant()`

#### Storage
- `SQLiteSessionStore` - Async SQLite persistence
- `MemorySessionStore` - In-memory storage for testing
- `SessionStore` protocol for custom implementations
- Session filtering, cleanup, and expiration

#### Message Types
- 11 typed message variants (InitMessage, TextMessage, ToolUseMessage, etc.)
- Forward-compatible `UnknownMessage` for new event types
- Full type hints and dataclass-based definitions

#### Configuration
- `CLIAgentOptions` - 40+ configuration fields
- `ToonConfig` - TOON serialization settings
- `MCPServerConfig` - MCP server configuration
- `PermissionMode` enum for tool permissions
- Environment variable support (CCFLOW_* prefix)

#### TOON Integration
- `ToonSerializer` for token-optimized encoding (30-60% savings)
- Automatic context injection into system prompts
- Token savings tracking and metrics
- Fallback to JSON when TOON unavailable

#### Reliability
- `CircuitBreaker` with configurable thresholds
- `TokenBucketRateLimiter` and `SlidingWindowRateLimiter`
- `ConcurrencyLimiter` for parallel request control
- `CombinedLimiter` for rate + concurrency
- `retry_with_backoff()` with exponential backoff
- `HealthChecker` for CLI availability

#### Events & Observability
- `EventEmitter` with pub/sub pattern
- 15 event types (SESSION_CREATED, TURN_COMPLETED, THINKING_RECEIVED, etc.)
- `CostTracker` for usage monitoring
- Correlation ID support via context variables
- Prometheus metrics integration

#### Pricing & Cost Tracking
- `ModelPricing` for haiku/sonnet/opus tiers
- `calculate_cost()` function
- `UsageStats` for aggregate tracking
- Per-model usage breakdown

#### Process Pool
- `ProcessPool` for concurrent CLI execution
- `StreamingPool` for streaming results
- Configurable workers and queue size

#### API Fallback (Optional)
- `APIClient` for direct Anthropic SDK
- `FallbackExecutor` for automatic CLI→API fallback
- Requires `pip install ccflow[api]`

#### CLI
- `ccflow` command-line interface
- Model selection, streaming, session management
- HTTP server mode with `ccflow server`

#### Docker Support
- Multi-stage Dockerfile with Claude CLI
- docker-compose.yml with dev/prod profiles
- Prometheus and Grafana integration

#### Documentation
- Comprehensive README with examples
- API reference (docs/api.md)
- Architecture documentation (docs/architecture.md)
- LLM-friendly LLMS.txt
- TOON format research (docs/TOON_RESEARCH.md)

### Dependencies

**Required:**
- pydantic>=2.0
- pydantic-settings>=2.0
- structlog>=23.0
- aiosqlite>=0.19
- tiktoken>=0.5
- prometheus-client>=0.17
- uvicorn[standard]>=0.23

**Optional:**
- `[toon]`: toon-format>=0.1.0
- `[tracing]`: opentelemetry-api, opentelemetry-sdk
- `[server]`: fastapi, websockets
- `[api]`: anthropic>=0.25.0
- `[dev]`: pytest, pytest-asyncio, pytest-cov, ruff, mypy, pre-commit

### Testing

- 879 test functions
- 86% code coverage
- pytest-asyncio for async tests
- Comprehensive mocking and fixtures

---

[0.1.0]: https://github.com/astoreyai/ccflow/releases/tag/v0.1.0
