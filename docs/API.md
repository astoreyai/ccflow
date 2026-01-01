# ccflow API Reference

**Version**: 0.1.0

Production middleware bridging Claude Code CLI with SDK-like Python interfaces.

---

## Quick Start

```python
import asyncio
from ccflow import query, CLIAgentOptions, TextMessage

async def main():
    async for msg in query("What files are here?", CLIAgentOptions(model="sonnet")):
        if isinstance(msg, TextMessage):
            print(msg.content, end="")

asyncio.run(main())
```

---

## Core API

### `query()`

Execute a single query with streaming response.

```python
async def query(
    prompt: str,
    options: CLIAgentOptions | None = None,
) -> AsyncIterator[Message]
```

**Parameters:**
- `prompt` - The prompt to send to Claude
- `options` - Configuration options (optional)

**Yields:** `Message` objects (text, tool_use, tool_result, etc.)

**Example:**
```python
from ccflow import query, CLIAgentOptions, TextMessage

options = CLIAgentOptions(
    model="sonnet",
    allowed_tools=["Read", "Grep"],
)

async for msg in query("Analyze this codebase", options):
    if isinstance(msg, TextMessage):
        print(msg.content, end="")
```

---

### `query_simple()`

Execute a query and return the complete text response.

```python
async def query_simple(
    prompt: str,
    options: CLIAgentOptions | None = None,
) -> str
```

**Parameters:**
- `prompt` - The prompt to send to Claude
- `options` - Configuration options (optional)

**Returns:** Complete text response as string

**Example:**
```python
from ccflow import query_simple, CLIAgentOptions

response = await query_simple(
    "Summarize the README.md",
    CLIAgentOptions(model="haiku")
)
print(response)
```

---

### `batch_query()`

Execute multiple independent queries concurrently.

```python
async def batch_query(
    prompts: list[str],
    options: CLIAgentOptions | None = None,
    concurrency: int = 5,
) -> list[QueryResult]
```

**Parameters:**
- `prompts` - List of prompts to process
- `options` - Shared configuration options
- `concurrency` - Maximum concurrent executions (default: 5)

**Returns:** List of `QueryResult` objects

**Example:**
```python
from ccflow import batch_query, CLIAgentOptions

prompts = [
    "Review auth.py for security issues",
    "Check database.py for SQL injection",
    "Analyze api.py for rate limiting",
]

results = await batch_query(prompts, concurrency=3)
for result in results:
    if result.success:
        print(f"{result.session_id}: {result.result[:100]}...")
```

---

## Session Management

### `Session`

Stateful session for multi-turn conversations.

```python
class Session:
    def __init__(
        self,
        session_id: str | None = None,
        options: CLIAgentOptions | None = None,
    ) -> None

    async def send_message(self, content: str) -> AsyncIterator[Message]
    async def fork(self) -> Session
    async def close(self) -> SessionStats
```

**Properties:**
- `session_id: str` - Unique session identifier
- `turn_count: int` - Number of turns in session
- `is_closed: bool` - Whether session is closed
- `created_at: datetime` - Session creation time
- `updated_at: datetime` - Last activity time

**Example:**
```python
from ccflow import Session, CLIAgentOptions, TextMessage
from ccflow.types import PermissionMode

session = Session(options=CLIAgentOptions(
    model="opus",
    permission_mode=PermissionMode.ACCEPT_EDITS,
))

# First turn
async for msg in session.send_message("Review this codebase"):
    if isinstance(msg, TextMessage):
        print(msg.content, end="")

# Follow-up (maintains context)
async for msg in session.send_message("Focus on security issues"):
    if isinstance(msg, TextMessage):
        print(msg.content, end="")

# Close and get stats
stats = await session.close()
print(f"Total tokens: {stats.total_tokens}")
```

---

### `SessionManager`

Manages session lifecycle with persistence and pooling.

```python
class SessionManager:
    def __init__(
        self,
        store: SessionStore | None = None,
        max_sessions: int = 100,
        session_ttl: float = 3600.0,
    ) -> None

    async def create_session(options: CLIAgentOptions) -> Session
    async def get_session(session_id: str) -> Session | None
    async def list_sessions(**filters) -> list[SessionMetadata]
    async def delete_session(session_id: str) -> bool
```

**Example:**
```python
from ccflow import SessionManager, CLIAgentOptions

async with SessionManager() as manager:
    # Create session
    session = await manager.create_session(CLIAgentOptions(model="sonnet"))

    # Use session
    async for msg in session.send_message("Hello"):
        print(msg)

    # List all sessions
    sessions = await manager.list_sessions()
    print(f"Active sessions: {len(sessions)}")
```

---

## Configuration

### `CLIAgentOptions`

Configuration options for CLI middleware.

```python
@dataclass
class CLIAgentOptions:
    # Model selection
    model: str = "sonnet"
    fallback_model: str | None = None

    # System prompt
    system_prompt: str | None = None
    append_system_prompt: str | None = None

    # Permissions
    permission_mode: PermissionMode = PermissionMode.DEFAULT
    allowed_tools: list[str] | None = None
    disallowed_tools: list[str] | None = None

    # Session management
    session_id: str | None = None
    resume: bool = False
    fork_session: bool = False

    # Execution limits
    max_budget_usd: float | None = None
    timeout: float = 300.0

    # Working directory
    cwd: str | None = None
    add_dirs: list[str] | None = None

    # MCP configuration
    mcp_servers: dict[str, MCPServerConfig] | None = None
    strict_mcp: bool = False

    # TOON serialization
    toon: ToonConfig = field(default_factory=ToonConfig)
    context: dict[str, Any] | None = None

    # Output options
    verbose: bool = False
    include_partial: bool = False
```

---

### `PermissionMode`

Tool permission modes for execution.

```python
class PermissionMode(str, Enum):
    DEFAULT = "default"           # Ask for permission
    PLAN = "plan"                 # Show plan before execution
    DONT_ASK = "dontAsk"          # Don't ask, but show results
    ACCEPT_EDITS = "acceptEdits"  # Auto-accept file edits
    DELEGATE = "delegate"         # Delegate to LLM judgment
    BYPASS = "bypassPermissions"  # Bypass all permissions (sandboxed only)
```

---

### `ToonConfig`

Configuration for TOON token optimization.

```python
@dataclass
class ToonConfig:
    enabled: bool = True
    delimiter: Literal[",", "\t", "|"] = ","
    indent: int = 2
    length_marker: bool = False
    encode_context: bool = True
    encode_tool_results: bool = True
    track_savings: bool = True

    @property
    def last_compression_ratio(self) -> float
```

**Example:**
```python
from ccflow import CLIAgentOptions
from ccflow.types import ToonConfig

options = CLIAgentOptions(
    context={"data": [...]},  # Auto-TOON encoded
    toon=ToonConfig(
        enabled=True,
        delimiter="\t",  # Tab for extra savings
        track_savings=True,
    ),
)
```

---

## Message Types

All messages inherit from the `Message` union type:

### `TextMessage`
```python
@dataclass
class TextMessage:
    content: str
    type: Literal["message"] = "message"
```

### `ToolUseMessage`
```python
@dataclass
class ToolUseMessage:
    tool: str
    args: dict[str, Any]
    type: Literal["tool_use"] = "tool_use"
```

### `ToolResultMessage`
```python
@dataclass
class ToolResultMessage:
    tool: str
    result: str
    type: Literal["tool_result"] = "tool_result"
```

### `AssistantMessage`
```python
@dataclass
class AssistantMessage:
    content: list[dict[str, Any]]
    model: str
    message_id: str
    session_id: str
    usage: dict[str, Any]
    stop_reason: str | None = None

    @property
    def text_content(self) -> str
```

### `InitMessage`
```python
@dataclass
class InitMessage:
    session_id: str
    type: Literal["system"] = "system"
    subtype: Literal["init"] = "init"
```

### `StopMessage`
```python
@dataclass
class StopMessage:
    session_id: str
    usage: dict[str, int]  # {input_tokens, output_tokens}
    type: Literal["system"] = "system"
    subtype: Literal["stop"] = "stop"
```

---

## Results

### `QueryResult`

Result from batch_query().

```python
@dataclass
class QueryResult:
    session_id: str
    result: str
    input_tokens: int
    output_tokens: int
    duration_seconds: float
    toon_savings_ratio: float = 0.0
    error: str | None = None

    @property
    def total_tokens(self) -> int

    @property
    def success(self) -> bool
```

### `SessionStats`

Statistics from Session.close().

```python
@dataclass
class SessionStats:
    session_id: str
    total_turns: int
    total_input_tokens: int
    total_output_tokens: int
    duration_seconds: float
    toon_savings_ratio: float = 0.0

    @property
    def total_tokens(self) -> int
```

---

## Rate Limiting

### Global Rate Limiter

```python
from ccflow import get_limiter, reset_limiter

# Get global limiter
limiter = get_limiter()

# Check stats
print(limiter.stats)

# Reset (for testing)
reset_limiter()
```

### Custom Limiter

```python
from ccflow import CombinedLimiter, TokenBucketRateLimiter, ConcurrencyLimiter

limiter = CombinedLimiter(
    rate_limiter=TokenBucketRateLimiter(rate=60, burst=10),
    concurrency_limiter=ConcurrencyLimiter(max_concurrent=5),
)

async with limiter.acquire() as wait_time:
    if wait_time > 0:
        print(f"Waited {wait_time:.2f}s for rate limit")
    # Execute request
```

---

## Events

### Event Types

```python
from ccflow import (
    EventType,
    SessionCreatedEvent,
    SessionClosedEvent,
    TurnStartedEvent,
    TurnCompletedEvent,
    ToolCalledEvent,
    ToolResultEvent,
    TokensUsedEvent,
    CostIncurredEvent,
)
```

### Event Handlers

```python
from ccflow import get_emitter, LoggingHandler, MetricsHandler

emitter = get_emitter()

# Add handlers
emitter.add_handler(EventType.SESSION_CREATED, LoggingHandler())
emitter.add_handler(EventType.TOKENS_USED, MetricsHandler())

# Custom handler
async def my_handler(event):
    print(f"Event: {event.event_type}")

emitter.add_handler(EventType.TURN_COMPLETED, my_handler)
```

---

## MCP Configuration

### MCPServerConfig

```python
from ccflow import MCPServerConfig

# Custom MCP server
server = MCPServerConfig(
    command="python",
    args=["-m", "my_mcp_server"],
    env={"API_KEY": "${API_KEY}"},
    transport="stdio",
)
```

### Convenience Functions

```python
from ccflow.mcp import (
    github_server,
    postgres_server,
    playwright_server,
    filesystem_server,
)

servers = {
    "github": github_server("GITHUB_TOKEN"),
    "files": filesystem_server(["/home/user/project"]),
    "postgres": postgres_server("DATABASE_URL"),
}

options = CLIAgentOptions(
    mcp_servers=servers,
    strict_mcp=False,
)
```

---

## Exceptions

```python
from ccflow import (
    CCFlowError,              # Base exception
    CLINotFoundError,         # Claude CLI not in PATH
    CLIAuthenticationError,   # CLI not authenticated
    CLIExecutionError,        # Subprocess failed
    CLITimeoutError,          # Execution timeout
    SessionNotFoundError,     # Session not found
    SessionStoreError,        # Storage error
    ParseError,               # NDJSON parse error
    ToonEncodingError,        # TOON encoding failed
    PermissionDeniedError,    # Tool permission denied
    RateLimitExceededError,   # Rate limit exceeded
    ConcurrencyLimitExceededError,  # Concurrency limit exceeded
)
```

---

## CLI Commands

```bash
# Query
ccflow query "What files are here?" --model sonnet --stream

# Sessions
ccflow sessions list [--status active] [--json]
ccflow sessions get <session-id>
ccflow sessions delete <session-id> [-f]
ccflow sessions cleanup [--days 7] [--dry-run]

# Server
ccflow server [--host 0.0.0.0] [--port 8000] [--reload]

# Stats
ccflow stats [--json] [--rate-limiter]
```

---

## HTTP API

When running `ccflow server`:

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/ready` | Readiness probe |
| POST | `/query` | Execute query |
| POST | `/query/stream` | Streaming query (SSE) |
| GET | `/sessions` | List sessions |
| GET | `/sessions/{id}` | Get session |
| DELETE | `/sessions/{id}` | Delete session |
| GET | `/metrics` | Prometheus metrics |
| GET | `/stats/sessions` | Session statistics |
| GET | `/stats/rate-limiter` | Rate limiter stats |
| WS | `/ws` | WebSocket interface |

### Query Request

```json
POST /query
{
    "prompt": "Analyze this code",
    "model": "sonnet",
    "allowed_tools": ["Read", "Grep"],
    "timeout": 60.0
}
```

### Query Response

```json
{
    "session_id": "abc123",
    "content": "...",
    "input_tokens": 150,
    "output_tokens": 500,
    "duration_seconds": 2.5
}
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CCFLOW_DEFAULT_MODEL` | `sonnet` | Default model |
| `CCFLOW_DEFAULT_TIMEOUT` | `300.0` | Default timeout (seconds) |
| `CCFLOW_SESSION_TTL` | `3600.0` | Session TTL (seconds) |
| `CCFLOW_MAX_CONCURRENT` | `10` | Max concurrent requests |
| `CCFLOW_RATE_LIMIT` | `60` | Requests per minute |
| `CCFLOW_TOON_ENABLED` | `true` | Enable TOON by default |
| `CCFLOW_LOG_LEVEL` | `INFO` | Logging level |
| `CCFLOW_METRICS_ENABLED` | `true` | Enable metrics |
