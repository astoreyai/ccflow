# Changelog

All notable changes to ccflow will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Extended Thinking (Ultrathink)** - Full support for Claude's extended thinking mode
  - `ultrathink` option in `CLIAgentOptions` to enable deep reasoning
  - `ThinkingMessage` type for thinking content blocks
  - `ThinkingReceivedEvent` for event-driven thinking tracking
  - `thinking_tokens` field in `TurnCompletedEvent`, `TokensUsedEvent`, `CostIncurredEvent`
  - Parser helpers: `collect_thinking()`, `extract_thinking_from_assistant()`, `extract_thinking_tokens()`
  - Auto-prepends "ultrathink" prefix to prompts when enabled

## [0.1.0] - 2026-01-01

### Added

#### Core API
- `query()` - Async streaming query with typed messages
- `query_simple()` - Simple string response
- `batch_query()` - Concurrent batch processing
- `stream_to_callback()` - Callback-based streaming

#### Session Management
- `Session` class for multi-turn conversations
- `SessionManager` for lifecycle management
- `load_session()` and `resume_session()` helpers
- Auto-persistence with configurable stores

#### Storage
- `SQLiteSessionStore` - Async SQLite persistence
- `MemorySessionStore` - In-memory storage for testing
- `SessionStore` protocol for custom implementations
- Session filtering, cleanup, and expiration

#### Message Types
- 11 typed message variants (InitMessage, TextMessage, ToolUseMessage, etc.)
- Forward-compatible UnknownMessage for new event types
- Full type hints and dataclass-based definitions

#### Configuration
- `CLIAgentOptions` - 40+ configuration fields
- `ToonConfig` - TOON serialization settings
- `MCPServerConfig` - MCP server configuration
- `PermissionMode` enum for tool permissions
- Environment variable support (CCFLOW_* prefix)

#### Reliability
- `CircuitBreaker` with configurable thresholds
- `TokenBucketRateLimiter` and `SlidingWindowRateLimiter`
- `ConcurrencyLimiter` for parallel request control
- `CombinedLimiter` for rate + concurrency
- `retry_with_backoff()` with exponential backoff
- `HealthChecker` for CLI availability

#### Events & Observability
- `EventEmitter` with pub/sub pattern
- 13 event types (SESSION_CREATED, TURN_COMPLETED, etc.)
- `CostTracker` for usage monitoring
- `LoggingHandler` and `MetricsHandler`
- Correlation ID support via context variables

#### Pricing & Cost Tracking
- `ModelPricing` for haiku/sonnet/opus tiers
- `calculate_cost()` function
- `UsageStats` for aggregate tracking
- Per-model usage breakdown
- Batch discount support

#### Process Pool
- `ProcessPool` for concurrent CLI execution
- `StreamingPool` for streaming results
- Configurable workers and queue size
- Task management (submit, cancel, gather)

#### API Fallback (Optional)
- `APIClient` for direct Anthropic SDK
- `FallbackExecutor` for automatic CLI→API fallback
- Configurable fallback triggers (circuit open, health fail)
- Requires `pip install ccflow[api]`

#### Docker Support
- Multi-stage Dockerfile with Claude CLI
- docker-compose.yml with dev/prod profiles
- Prometheus and Grafana integration
- Health checks and resource limits

#### Documentation
- Comprehensive API reference (docs/api.md)
- Architecture documentation (docs/architecture.md)
- LLM-friendly LLMS.txt
- Inline docstrings throughout

### Dependencies
- Required: pydantic, pydantic-settings, structlog, aiosqlite
- Optional: tiktoken, prometheus-client, opentelemetry, fastapi, uvicorn, anthropic

### Testing
- 807+ test functions
- 80%+ coverage target
- pytest-asyncio for async tests
- Comprehensive mocking and fixtures

---

## Version History

- **0.1.0** (2026-01-01) - Initial release

[Unreleased]: https://github.com/astoreyai/ccflow/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/astoreyai/ccflow/releases/tag/v0.1.0
