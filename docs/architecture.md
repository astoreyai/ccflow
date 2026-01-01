# ccflow Architecture

## Overview

ccflow is a production middleware that bridges the Claude Code CLI with SDK-like Python interfaces. It enables subscription-based Claude usage through the CLI while providing a clean async Python API.

## Design Principles

1. **Subscription-First**: Uses Claude CLI OAuth, not API keys
2. **Async Throughout**: All I/O operations are non-blocking
3. **Graceful Degradation**: Optional dependencies fail gracefully
4. **Minimal Core**: 4 required dependencies
5. **Observable**: Built-in events, metrics, structured logging

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  query()    │  │ Session     │  │ SessionManager      │  │
│  │  batch_*()  │  │ send_msg()  │  │ create/list/cleanup │  │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘  │
└─────────┼────────────────┼─────────────────────┼────────────┘
          │                │                     │
┌─────────▼────────────────▼─────────────────────▼────────────┐
│                    Middleware Layer                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ StreamParser│  │ EventEmitter│  │ ToonSerializer      │  │
│  │ NDJSON→Msgs │  │ Pub/Sub     │  │ Token Optimization  │  │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘  │
└─────────┼────────────────┼─────────────────────┼────────────┘
          │                │                     │
┌─────────▼────────────────▼─────────────────────▼────────────┐
│                    Executor Layer                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ CLIExecutor │  │ RateLimiter │  │ CircuitBreaker      │  │
│  │ Subprocess  │  │ Token/Window│  │ Fault Tolerance     │  │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘  │
│         │                │                     │             │
│  ┌──────▼────────────────▼─────────────────────▼──────────┐ │
│  │                  RetryHandler                           │ │
│  │              Exponential Backoff                        │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                    Persistence Layer                         │
│  ┌─────────────────────┐  ┌─────────────────────────────┐   │
│  │ SQLiteSessionStore  │  │ MemorySessionStore          │   │
│  │ Async aiosqlite     │  │ Testing/Ephemeral           │   │
│  └─────────────────────┘  └─────────────────────────────┘   │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                    Infrastructure                            │
│  ┌─────────────────────────────────────────────────────┐    │
│  │           Claude Code CLI (subprocess)               │    │
│  │           OAuth Authentication                       │    │
│  │           NDJSON Streaming Output                    │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## Component Details

### Application Layer

#### `api.py` - High-Level Query Interface

```python
# Streaming query
async for msg in query(prompt, options):
    yield msg

# Simple string response
result = await query_simple(prompt)

# Batch processing with concurrency
results = await batch_query(prompts, concurrency=5)
```

**Responsibilities:**
- Orchestrate execution pipeline
- Collect usage metrics
- Handle TOON encoding
- Emit events

#### `session.py` - Stateful Conversations

```python
session = Session(session_id="abc")
async for msg in session.send_message("Hello"):
    process(msg)
# Subsequent messages use CLI --resume flag
```

**Responsibilities:**
- Track turn count and tokens
- Manage session state
- Persist to storage
- Resume conversations via CLI

#### `manager.py` - Session Lifecycle

```python
manager = await init_manager(store=SQLiteSessionStore())
session = await manager.create_session(tags=["research"])
sessions = await manager.list_sessions(status=SessionStatus.ACTIVE)
await manager.cleanup_expired()
```

**Responsibilities:**
- Create/list/cleanup sessions
- Auto-persistence
- Background cleanup task
- Event emission

### Middleware Layer

#### `parser.py` - Stream Parsing

Converts raw NDJSON from CLI to typed Message objects:

```
{"type": "system", "subtype": "init", "session_id": "..."} → InitMessage
{"type": "message", "content": "Hello"} → TextMessage
{"type": "tool_use", "tool": "Bash", "args": {...}} → ToolUseMessage
```

**Forward Compatibility:**
- Unknown event types → UnknownMessage
- Unknown system types → UnknownSystemMessage

#### `events.py` - Event System

Pub/sub for lifecycle observability:

```python
emitter = get_emitter()
emitter.on(EventType.TOKENS_USED, handle_tokens)
emitter.on(EventType.SESSION_CLOSED, handle_close)
await emitter.emit(TokensUsedEvent(...))
```

**Event Types:**
- `SESSION_CREATED`, `SESSION_CLOSED`, `SESSION_ERROR`
- `TURN_STARTED`, `TURN_COMPLETED`
- `TOOL_CALLED`, `TOOL_RESULT`
- `TOKENS_USED`, `COST_INCURRED`

#### `toon_integration.py` - Token Optimization

TOON (Token-Oriented Object Notation) reduces token consumption:

```python
# JSON: 150 tokens
{"positions": [{"symbol": "AAPL", "qty": 100}, ...]}

# TOON: 90 tokens (40% savings)
positions[2]{symbol,qty}: AAPL,100|GOOGL,50
```

### Executor Layer

#### `executor.py` - CLI Subprocess

Core engine spawning Claude CLI:

```python
executor = CLIExecutor()
flags = executor.build_flags(options)  # Translate options to CLI flags
async for event in executor.execute(prompt, flags):
    yield event
```

**Integration:**
- Rate limiting (acquire before execute)
- Circuit breaker (wrap execution)
- Retry (on transient failures)
- Health check (verify CLI available)

#### `rate_limiting.py` - Request Throttling

Two algorithms available:

**Token Bucket** (burst-friendly):
```python
limiter = TokenBucketRateLimiter(rate=10.0, burst=20)
async with limiter.acquire():
    # Rate-limited operation
```

**Sliding Window** (smooth rate):
```python
limiter = SlidingWindowRateLimiter(rate=60.0, window=60.0)
# Exactly 60 requests per 60 seconds
```

**Combined Limiter** (rate + concurrency):
```python
limiter = CombinedLimiter(rate=10.0, max_concurrent=5)
```

#### `reliability.py` - Fault Tolerance

**Circuit Breaker:**
```
CLOSED ─[5 failures]→ OPEN ─[30s timeout]→ HALF_OPEN ─[success]→ CLOSED
                                              │
                                              └─[failure]→ OPEN
```

**Retry with Backoff:**
```python
await retry_with_backoff(func, RetryConfig(
    max_retries=3,
    base_delay=1.0,
    exponential_base=2.0,  # 1s, 2s, 4s
))
```

**Health Checker:**
```python
checker = get_health_checker()
status = await checker.check()
if not status.healthy:
    # Trigger fallback
```

### Persistence Layer

#### `store.py` - Storage Protocol

```python
class SessionStore(Protocol):
    async def save(self, state: SessionState) -> None
    async def load(self, session_id: str) -> SessionState | None
    async def list(self, filter: SessionFilter) -> list[SessionMetadata]
    async def cleanup(self, older_than: timedelta) -> int
```

#### `stores/sqlite.py` - SQLite Backend

```python
store = SQLiteSessionStore("sessions.db")
# Uses aiosqlite for async I/O
# WAL mode for concurrent reads
# Automatic migrations
```

#### `stores/memory.py` - In-Memory Backend

```python
store = MemorySessionStore()
# Fast, ephemeral, testing-focused
```

## Data Flow

### Query Execution

```
User Code                        ccflow                           Claude CLI
    │                               │                                  │
    ├─query(prompt)─────────────────▶│                                  │
    │                               ├─ToonEncoder.encode(context)──────▶│
    │                               ├─executor.build_flags()───────────▶│
    │                               ├─limiter.acquire()────────────────▶│
    │                               ├─circuit_breaker.call()───────────▶│
    │                               ├─create_subprocess_exec()─────────▶│
    │                               │                                  ├─OAuth─▶Anthropic API
    │                               │                                  │◀──stream──│
    │                               │◀───NDJSON line───────────────────│
    │                               ├─parser.parse_event()─────────────▶│
    │◀──Message────────────────────│                                  │
    │                               ├─emitter.emit(TokensUsed)─────────▶│
    │                               │                                  │
```

### Session Resumption

```
Turn 1: Session.send_message("Hello")
        └─ CLI: claude --print "Hello"
        └─ Response includes session_id

Turn 2: Session.send_message("Continue")
        └─ CLI: claude --print --resume <session_id> "Continue"
        └─ Claude remembers conversation
```

## Configuration Hierarchy

```
1. Code defaults (CLIAgentOptions dataclass)
        ↓
2. Environment variables (CCFLOW_* prefix)
        ↓
3. .env file (loaded by pydantic-settings)
        ↓
4. Runtime arguments (passed to functions)
        ↓
5. Per-call options (CLIAgentOptions instance)
```

## Error Handling Strategy

```
             ┌──────────────────────────────────────┐
             │         User Application             │
             └──────────────────┬───────────────────┘
                                │
    ┌───────────────────────────▼───────────────────────────┐
    │                    CCFlowError                         │
    │  ┌─────────────┬─────────────┬─────────────────────┐  │
    │  │ Retryable   │ Non-Retry   │ Fallback-able       │  │
    │  │ ─────────── │ ─────────── │ ───────────────     │  │
    │  │ Timeout     │ AuthError   │ CLIUnavailable      │  │
    │  │ RateLimit   │ ParseError  │ CircuitOpen         │  │
    │  │ Connection  │ Permission  │ HealthCheckFail     │  │
    │  └──────┬──────┴──────┬──────┴──────────┬──────────┘  │
    └─────────┼─────────────┼─────────────────┼─────────────┘
              │             │                 │
    ┌─────────▼─────┐ ┌─────▼─────┐ ┌─────────▼─────────┐
    │ RetryHandler  │ │ Propagate │ │ FallbackExecutor  │
    │ Exp. Backoff  │ │ to User   │ │ CLI → API Client  │
    └───────────────┘ └───────────┘ └───────────────────┘
```

## Global Singletons

Factory pattern with lazy initialization:

```python
# Module-level
_default_executor: CLIExecutor | None = None

def get_executor() -> CLIExecutor:
    global _default_executor
    if _default_executor is None:
        _default_executor = CLIExecutor()
    return _default_executor

def reset_executor() -> None:
    """For testing - clears singleton."""
    global _default_executor
    _default_executor = None
```

**Singletons:**
- `get_executor()` - CLI executor
- `get_manager()` - Session manager
- `get_limiter()` - Rate limiter
- `get_cli_circuit_breaker()` - Circuit breaker
- `get_health_checker()` - Health checker
- `get_emitter()` - Event emitter
- `get_api_client()` - API client (fallback)
- `get_usage_tracker()` - Usage statistics

All have corresponding `reset_*()` functions for testing.

## Optional Dependencies

Graceful degradation when optional packages unavailable:

```python
# metrics.py
try:
    from prometheus_client import Counter, Histogram
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

def record_tokens(count: int) -> None:
    if PROMETHEUS_AVAILABLE:
        tokens_counter.inc(count)
```

**Optional Packages:**
- `prometheus-client` - Metrics export
- `opentelemetry-*` - Distributed tracing
- `anthropic` - API fallback
- `fastapi`, `uvicorn` - HTTP server
- `tiktoken` - Token counting
- `toon-format` - TOON encoding

## Security Considerations

1. **No Hardcoded Secrets**: All credentials via environment or CLI
2. **Session IDs**: UUID v4 (cryptographically random)
3. **SQL Injection**: Parameterized queries only
4. **CLI Injection**: Arguments as list, not shell string
5. **Permission Modes**: Respected from CLI configuration
6. **Credential Handling**: Delegated to Claude CLI OAuth

## Performance Characteristics

| Component | Complexity | Latency |
|-----------|------------|---------|
| Rate Limiter (Token Bucket) | O(1) | <1ms |
| Rate Limiter (Sliding Window) | O(n) on window | <1ms |
| Circuit Breaker | O(1) | <1ms |
| Stream Parser | O(n) per line | <1ms |
| SQLite Store | O(log n) | 10-50ms |
| TOON Encoding | O(n) data size | <10ms |

## Extension Points

1. **Custom Store**: Implement `SessionStore` protocol
2. **Custom Events**: Add handlers via `emitter.on()`
3. **Custom Metrics**: Subscribe to events
4. **API Fallback**: Configure `FallbackExecutor`
5. **MCP Servers**: Configure via `MCPServerConfig`
