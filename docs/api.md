# ccflow API Reference

Complete API documentation for ccflow - Claude Code CLI Middleware.

## Table of Contents

- [Core Query Functions](#core-query-functions)
- [Session Management](#session-management)
- [Session Manager](#session-manager)
- [Configuration](#configuration)
- [Message Types](#message-types)
- [Parser Helpers](#parser-helpers)
- [Storage & Persistence](#storage--persistence)
- [Project & Tracing](#project--tracing)
- [Events & Observability](#events--observability)
- [Rate Limiting](#rate-limiting)
- [Reliability & Fault Tolerance](#reliability--fault-tolerance)
- [Pricing & Cost Tracking](#pricing--cost-tracking)
- [Process Pool](#process-pool)
- [API Fallback](#api-fallback)
- [Exceptions](#exceptions)

---

## Core Query Functions

### `query()`

Stream responses from Claude using the CLI.

```python
async def query(
    prompt: str,
    options: CLIAgentOptions | None = None,
) -> AsyncIterator[Message]
```

**Parameters:**
- `prompt` (str): The prompt to send to Claude
- `options` (CLIAgentOptions, optional): Configuration options

**Yields:**
- `Message`: Typed message objects (TextMessage, ToolUseMessage, etc.)

**Example:**
```python
from ccflow import query, CLIAgentOptions

async for msg in query("Explain quantum computing", CLIAgentOptions(model="sonnet")):
    if hasattr(msg, 'content'):
        print(msg.content, end="")
```

---

### `query_simple()`

Get a simple string response (non-streaming).

```python
async def query_simple(
    prompt: str,
    options: CLIAgentOptions | None = None,
) -> str
```

**Parameters:**
- `prompt` (str): The prompt to send
- `options` (CLIAgentOptions, optional): Configuration options

**Returns:**
- `str`: The complete text response

**Example:**
```python
from ccflow import query_simple

response = await query_simple("What is 2+2?")
print(response)  # "4"
```

---

### `batch_query()`

Execute multiple queries concurrently.

```python
async def batch_query(
    prompts: list[str],
    options: CLIAgentOptions | None = None,
    concurrency: int = 5,
) -> list[QueryResult]
```

**Parameters:**
- `prompts` (list[str]): List of prompts to process
- `options` (CLIAgentOptions, optional): Shared configuration
- `concurrency` (int): Maximum concurrent requests (default: 5)

**Returns:**
- `list[QueryResult]`: Results for each prompt

**Example:**
```python
from ccflow import batch_query

prompts = ["Explain AI", "Explain ML", "Explain DL"]
results = await batch_query(prompts, concurrency=3)

for result in results:
    print(f"Tokens: {result.total_tokens}, Cost: ${result.cost_usd:.4f}")
```

---

### `stream_to_callback()`

Stream responses to a callback function.

```python
async def stream_to_callback(
    prompt: str,
    callback: Callable[[Message], None],
    options: CLIAgentOptions | None = None,
) -> QueryResult
```

**Parameters:**
- `prompt` (str): The prompt to send
- `callback` (Callable): Function called for each message
- `options` (CLIAgentOptions, optional): Configuration options

**Returns:**
- `QueryResult`: Final result with usage statistics

**Example:**
```python
from ccflow import stream_to_callback

def on_message(msg):
    if hasattr(msg, 'content'):
        print(msg.content, end="", flush=True)

result = await stream_to_callback("Tell me a story", on_message)
```

---

## Session Management

### `Session`

Stateful multi-turn conversation management.

```python
class Session:
    def __init__(
        self,
        session_id: str | None = None,
        options: CLIAgentOptions | None = None,
        executor: CLIExecutor | None = None,
        store: SessionStore | None = None,
        emitter: EventEmitter | None = None,
    ) -> None
```

**Properties:**
- `session_id` (str): Unique session identifier
- `turn_count` (int): Number of turns completed
- `is_closed` (bool): Whether session is closed
- `total_input_tokens` (int): Cumulative input tokens
- `total_output_tokens` (int): Cumulative output tokens

**Methods:**

#### `send_message()`

```python
async def send_message(
    self,
    content: str,
    context: dict | None = None,
) -> AsyncIterator[Message]
```

Send a message and stream the response.

**Parameters:**
- `content` (str): Message content
- `context` (dict, optional): Additional context (TOON-encoded if enabled)

**Yields:**
- `Message`: Response messages

**Example:**
```python
from ccflow import Session

session = Session()
async for msg in session.send_message("Hello!"):
    print(msg)

async for msg in session.send_message("What did I just say?"):
    print(msg)  # Claude remembers the conversation

await session.close()
```

#### `fork()`

```python
async def fork(self) -> Session
```

Create a new session branching from current state.

**Returns:**
- `Session`: New session with shared history

#### `close()`

```python
async def close(self) -> SessionStats
```

Close the session and return statistics.

**Returns:**
- `SessionStats`: Final usage statistics

---

### `load_session()`

Load an existing session from storage.

```python
async def load_session(
    session_id: str,
    store: SessionStore,
    options: CLIAgentOptions | None = None,
    emitter: EventEmitter | None = None,
) -> Session | None
```

---

### `resume_session()`

Resume a previous session.

```python
async def resume_session(
    session_id: str,
    options: CLIAgentOptions | None = None,
    store: SessionStore | None = None,
    emitter: EventEmitter | None = None,
) -> Session
```

---

## Session Manager

### `SessionManager`

High-level session lifecycle management.

```python
class SessionManager:
    def __init__(
        self,
        store: SessionStore | None = None,
        executor: CLIExecutor | None = None,
        emitter: EventEmitter | None = None,
        default_options: CLIAgentOptions | None = None,
        auto_cleanup: bool = True,
        cleanup_interval: float = 300.0,
        session_ttl: float = 3600.0,
    ) -> None
```

**Methods:**

#### `create_session()`

```python
async def create_session(
    self,
    session_id: str | None = None,
    options: CLIAgentOptions | None = None,
    tags: list[str] | None = None,
) -> Session
```

#### `get_session()`

```python
async def get_session(self, session_id: str) -> Session | None
```

#### `list_sessions()`

```python
async def list_sessions(
    self,
    status: SessionStatus | None = None,
    model: str | None = None,
    tags: list[str] | None = None,
    created_after: datetime | None = None,
    created_before: datetime | None = None,
    limit: int | None = None,
    offset: int = 0,
) -> list[SessionMetadata]
```

#### `cleanup_expired()`

```python
async def cleanup_expired(self) -> int
```

Returns count of sessions cleaned up.

**Example:**
```python
from ccflow import get_manager, init_manager

# Initialize manager
manager = await init_manager()

# Create session
session = await manager.create_session(tags=["research"])

# Send messages
async for msg in session.send_message("Hello"):
    print(msg)

# List all sessions
sessions = await manager.list_sessions(status=SessionStatus.ACTIVE)

# Cleanup
await manager.stop()
```

---

## Configuration

### `CLIAgentOptions`

Complete configuration for CLI execution.

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
    max_tokens: int | None = None
    timeout: float = 300.0

    # Working directory
    cwd: str | None = None
    add_dirs: list[str] | None = None

    # MCP configuration
    mcp_servers: dict[str, MCPServerConfig] | None = None
    strict_mcp: bool = False

    # TOON serialization
    toon: ToonConfig = field(default_factory=ToonConfig)

    # Context data
    context: dict[str, Any] | None = None

    # Output options
    verbose: bool = False
    include_partial: bool = False
    debug: str | bool | None = None

    # Structured output
    json_schema: str | dict | None = None

    # Permission bypass (sandboxed only)
    dangerously_skip_permissions: bool = False

    # Tool specification
    tools: list[str] | None = None

    # Session options
    continue_session: bool = False
    no_session_persistence: bool = False

    # Agent configuration
    agent: str | None = None
    agents: dict | None = None

    # Beta features
    betas: list[str] | None = None

    # Extended thinking
    ultrathink: bool = False  # Enable deep reasoning mode
```

### `PermissionMode`

```python
class PermissionMode(str, Enum):
    DEFAULT = "default"
    PLAN = "plan"
    DONT_ASK = "dontAsk"
    ACCEPT_EDITS = "acceptEdits"
    DELEGATE = "delegate"
    BYPASS = "bypassPermissions"
```

### `ToonConfig`

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

### `MCPServerConfig`

```python
@dataclass
class MCPServerConfig:
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    transport: Literal["stdio", "sse", "http"] = "stdio"
    url: str | None = None

    def to_dict(self) -> dict[str, Any]
```

---

## Message Types

### Union Type

```python
Message = (
    InitMessage |
    TextMessage |
    ThinkingMessage |
    ToolUseMessage |
    ToolResultMessage |
    ErrorMessage |
    StopMessage |
    HookMessage |
    AssistantMessage |
    UserMessage |
    ResultMessage |
    UnknownSystemMessage |
    UnknownMessage
)
```

### Individual Types

```python
@dataclass
class InitMessage:
    session_id: str
    type: Literal["system"] = "system"
    subtype: Literal["init"] = "init"

@dataclass
class TextMessage:
    content: str
    type: Literal["message"] = "message"

@dataclass
class ThinkingMessage:
    """Extended thinking content from ultrathink mode."""
    content: str
    thinking_tokens: int = 0
    type: Literal["thinking"] = "thinking"

@dataclass
class ToolUseMessage:
    tool: str
    args: dict[str, Any]
    type: Literal["tool_use"] = "tool_use"

@dataclass
class ToolResultMessage:
    tool: str
    result: str
    type: Literal["tool_result"] = "tool_result"

@dataclass
class StopMessage:
    session_id: str
    usage: dict[str, int]
    type: Literal["system"] = "system"
    subtype: Literal["stop"] = "stop"

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

@dataclass
class ResultMessage:
    result: str
    session_id: str
    duration_ms: int
    num_turns: int
    total_cost_usd: float
    usage: dict[str, Any]
    is_error: bool = False
```

---

## Parser Helpers

### `collect_text()`

Collect all text content from messages.

```python
def collect_text(messages: list[Message]) -> str
```

**Example:**
```python
from ccflow import collect_text

messages = [...]  # from query()
full_text = collect_text(messages)
```

---

### `collect_thinking()`

Collect all thinking content from messages (ultrathink mode).

```python
def collect_thinking(messages: list[Message]) -> str
```

**Example:**
```python
from ccflow import collect_thinking, ThinkingMessage

messages = [...]  # from query() with ultrathink=True
thinking = collect_thinking(messages)
print(f"Model reasoning: {thinking}")
```

---

### `extract_thinking_from_assistant()`

Extract thinking content from AssistantMessage content blocks.

```python
def extract_thinking_from_assistant(msg: AssistantMessage) -> str
```

**Example:**
```python
from ccflow import extract_thinking_from_assistant, AssistantMessage

# When AssistantMessage has thinking blocks embedded
thinking = extract_thinking_from_assistant(assistant_msg)
```

---

### `extract_thinking_tokens()`

Extract total thinking token count from raw events.

```python
def extract_thinking_tokens(events: list[dict[str, Any]]) -> int
```

**Example:**
```python
from ccflow import extract_thinking_tokens

events = [...]  # raw NDJSON events
total_thinking = extract_thinking_tokens(events)
print(f"Thinking tokens used: {total_thinking}")
```

---

## Storage & Persistence

### `SessionStore` Protocol

```python
class SessionStore(Protocol):
    async def save(self, state: SessionState) -> None
    async def load(self, session_id: str) -> SessionState | None
    async def delete(self, session_id: str) -> bool
    async def list(self, filter: SessionFilter | None = None) -> list[SessionMetadata]
    async def exists(self, session_id: str) -> bool
    async def count(self, filter: SessionFilter | None = None) -> int
    async def update_status(self, session_id: str, status: SessionStatus) -> bool
    async def cleanup(self, older_than: timedelta) -> int
    async def close(self) -> None
```

### `SQLiteSessionStore`

```python
class SQLiteSessionStore(BaseSessionStore):
    def __init__(
        self,
        db_path: str | Path | None = None,
        timeout: float = 30.0,
        auto_vacuum: bool = True,
    ) -> None
```

### `MemorySessionStore`

```python
class MemorySessionStore(BaseSessionStore):
    def __init__(self) -> None
```

### `SessionState`

```python
@dataclass
class SessionState:
    session_id: str
    created_at: datetime
    updated_at: datetime
    status: SessionStatus
    model: str
    system_prompt: str | None = None
    turn_count: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int
```

### `SessionStatus`

```python
class SessionStatus(str, Enum):
    ACTIVE = "active"
    CLOSED = "closed"
    ERROR = "error"
    EXPIRED = "expired"
```

---

## Project & Tracing

Hierarchical project organization with full trace recording for replay and analysis.

### `Project`

Organize related sessions under a project with nested sub-projects.

```python
class Project:
    def __init__(
        self,
        project_id: str | None = None,
        name: str = "",
        description: str = "",
        parent_project_id: str | None = None,
        *,
        store: ProjectStore | None = None,
        trace_store: TraceStore | None = None,
        session_store: SessionStore | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None
```

**Properties:**
- `project_id` (str): Unique project identifier
- `name` (str): Project name
- `description` (str): Project description
- `parent_project_id` (str | None): Parent for nested projects

**Methods:**

#### `create_session()`

```python
def create_session(
    self,
    options: CLIAgentOptions | None = None,
    *,
    detailed: bool = False,
) -> TracingSession
```

Create a tracing session within this project.

#### `create_subproject()`

```python
def create_subproject(
    self,
    name: str,
    description: str = "",
    *,
    metadata: dict[str, Any] | None = None,
) -> Project
```

Create a nested sub-project.

#### `get_traces()`

```python
async def get_traces(
    self,
    *,
    include_subprojects: bool = False,
    limit: int = 100,
) -> list[TraceData]
```

Get all traces in this project.

#### `replay_as_new()`

```python
async def replay_as_new(
    self,
    trace_id: str,
    *,
    options_override: CLIAgentOptions | None = None,
    detailed: bool = False,
) -> TracingSession
```

Replay a trace as a new session with the original prompt.

#### `replay_fork()`

```python
async def replay_fork(
    self,
    trace_id: str,
    *,
    options_override: CLIAgentOptions | None = None,
) -> TracingSession
```

Fork from original session state using CLI `--fork`.

#### `get_trace_summary()`

```python
async def get_trace_summary(self) -> dict[str, Any]
```

Get aggregate statistics: total_traces, input_tokens, output_tokens, thinking_tokens, total_cost_usd, etc.

#### `save()` / `load()`

```python
async def save(self) -> None

@classmethod
async def load(
    cls,
    project_id: str,
    store: ProjectStore,
    *,
    trace_store: TraceStore | None = None,
    session_store: SessionStore | None = None,
) -> Project | None
```

**Example:**
```python
from ccflow import Project, CLIAgentOptions
from ccflow.stores import SQLiteProjectStore, SQLiteTraceStore, SQLiteSessionStore

# Initialize stores
db_path = "ccflow.db"
project_store = SQLiteProjectStore(db_path)
trace_store = SQLiteTraceStore(db_path)
session_store = SQLiteSessionStore(db_path)

# Create project
project = Project(
    name="Code Review",
    store=project_store,
    trace_store=trace_store,
    session_store=session_store,
)
await project.save()

# Create session with tracing
session = project.create_session(
    options=CLIAgentOptions(model="sonnet", ultrathink=True),
    detailed=True,
)

async for msg in session.send_message("Review this code"):
    print(msg.content, end="")

# Get trace
trace = session.last_trace
print(f"Duration: {trace.duration_ms}ms")

# Replay with different model
new_session = await project.replay_as_new(
    trace.trace_id,
    options_override=CLIAgentOptions(model="opus"),
)
```

---

### `TracingSession`

Session subclass that auto-records full traces.

```python
class TracingSession(Session):
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
    ) -> None
```

**Properties:**
- `project_id` (str | None): Parent project ID
- `last_trace` (TraceData | None): Most recent trace
- `trace_count` (int): Number of traces recorded

**Methods:**

#### `send_message()`

```python
async def send_message(
    self,
    content: str,
    context: dict | None = None,
    *,
    detailed: bool | None = None,
) -> AsyncIterator[Message]
```

Send message with full trace recording.

#### `get_traces()`

```python
async def get_traces(self, limit: int = 100) -> list[TraceData]
```

Get all traces for this session.

---

### `TraceData`

Complete trace of a single prompt/response cycle.

```python
@dataclass
class TraceData:
    trace_id: str
    session_id: str
    project_id: str | None = None
    parent_trace_id: str | None = None
    sequence_number: int = 0

    # Content
    prompt: str = ""
    response: str = ""
    thinking: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    message_stream: list[dict[str, Any]] | None = None  # detailed mode

    # Configuration snapshot
    options_snapshot: dict[str, Any] = field(default_factory=dict)

    # Metrics
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    cost_usd: float = 0.0
    duration_ms: int = 0

    # Status
    status: TraceStatus = TraceStatus.PENDING
    error_message: str | None = None

    # Timestamps
    created_at: str = ""
    updated_at: str = ""

    # Extensibility
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_detail(self) -> bool

    @property
    def total_tokens(self) -> int
```

---

### `TraceStatus`

```python
class TraceStatus(str, Enum):
    PENDING = "pending"
    SUCCESS = "success"
    ERROR = "error"
    CANCELLED = "cancelled"
```

---

### `ProjectData`

```python
@dataclass
class ProjectData:
    project_id: str
    name: str
    description: str = ""
    parent_project_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""
```

---

### `TraceFilter`

```python
@dataclass
class TraceFilter:
    session_id: str | None = None
    project_id: str | None = None
    status: TraceStatus | None = None
    parent_trace_id: str | None = None
    after: str | None = None  # ISO timestamp
    before: str | None = None
    limit: int = 100
    offset: int = 0
```

---

### `ProjectFilter`

```python
@dataclass
class ProjectFilter:
    name_contains: str | None = None
    parent_project_id: str | None = None
    after: str | None = None  # ISO timestamp
    before: str | None = None
    limit: int = 100
    offset: int = 0
```

---

### `TraceStore` Protocol

```python
class TraceStore(Protocol):
    async def save(self, trace: TraceData) -> None
    async def load(self, trace_id: str) -> TraceData | None
    async def delete(self, trace_id: str) -> bool
    async def list(self, filter: TraceFilter | None = None) -> list[TraceData]
    async def get_session_traces(self, session_id: str, limit: int = 100) -> list[TraceData]
    async def get_project_traces(self, project_id: str, limit: int = 100) -> list[TraceData]
    async def count(self, filter: TraceFilter | None = None) -> int
    async def cleanup(self, older_than: timedelta) -> int
    async def close(self) -> None
```

---

### `ProjectStore` Protocol

```python
class ProjectStore(Protocol):
    async def save(self, project: ProjectData) -> None
    async def load(self, project_id: str) -> ProjectData | None
    async def delete(self, project_id: str) -> bool
    async def list(self, filter: ProjectFilter | None = None) -> list[ProjectData]
    async def get_project_sessions(self, project_id: str, limit: int = 100) -> list[SessionMetadata]
    async def get_subprojects(self, project_id: str) -> list[ProjectData]
    async def count(self, filter: ProjectFilter | None = None) -> int
    async def close(self) -> None
```

---

### `SQLiteTraceStore`

```python
class SQLiteTraceStore(BaseTraceStore):
    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        timeout: float = 30.0,
    ) -> None
```

---

### `SQLiteProjectStore`

```python
class SQLiteProjectStore(BaseProjectStore):
    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        timeout: float = 30.0,
    ) -> None
```

---

### `create_tracing_session()`

```python
async def create_tracing_session(
    project_id: str | None = None,
    options: CLIAgentOptions | None = None,
    trace_store: TraceStore | None = None,
    session_store: SessionStore | None = None,
    detailed: bool = False,
) -> TracingSession
```

Convenience function for creating TracingSession.

---

## Events & Observability

### `EventEmitter`

```python
class EventEmitter:
    def emit_sync(self, event: Event) -> None
    async def emit(self, event: Event) -> None
    def on(self, event_type: EventType | str, handler: Callable) -> None
    def off(self, event_type: EventType | str, handler: Callable) -> None
```

### `EventType`

```python
class EventType(str, Enum):
    SESSION_CREATED = "session.created"
    SESSION_RESUMED = "session.resumed"
    SESSION_CLOSED = "session.closed"
    SESSION_ERROR = "session.error"
    TURN_STARTED = "turn.started"
    TURN_COMPLETED = "turn.completed"
    THINKING_RECEIVED = "thinking.received"
    TOOL_CALLED = "tool.called"
    TOOL_RESULT = "tool.result"
    TOKENS_USED = "tokens.used"
    COST_INCURRED = "cost.incurred"
    SESSION_PERSISTED = "session.persisted"
    SESSION_LOADED = "session.loaded"
    SESSION_DELETED = "session.deleted"
```

### Event Classes

```python
@dataclass
class SessionCreatedEvent(SessionEvent):
    session_id: str
    model: str
    tags: list[str]

@dataclass
class TurnCompletedEvent(SessionEvent):
    session_id: str
    turn_number: int
    input_tokens: int
    output_tokens: int
    thinking_tokens: int = 0  # Extended thinking tokens
    duration_ms: int

@dataclass
class ThinkingReceivedEvent(Event):
    """Emitted when extended thinking content is received."""
    content: str
    thinking_tokens: int = 0

@dataclass
class TokensUsedEvent(Event):
    input_tokens: int
    output_tokens: int
    thinking_tokens: int = 0  # Extended thinking tokens
    model: str

@dataclass
class CostIncurredEvent(Event):
    cost_usd: float
    model: str
    input_tokens: int
    output_tokens: int
    thinking_tokens: int = 0  # Extended thinking tokens
```

**Example:**
```python
from ccflow import get_emitter, EventType

emitter = get_emitter()

def on_tokens(event):
    print(f"Tokens: {event.input_tokens} in, {event.output_tokens} out")

emitter.on(EventType.TOKENS_USED, on_tokens)
```

---

## Rate Limiting

### `TokenBucketRateLimiter`

```python
class TokenBucketRateLimiter:
    def __init__(
        self,
        rate: float,
        burst: int | None = None,
        wait_timeout: float | None = None,
    ) -> None

    async def acquire(self) -> AsyncContextManager[float]
    def get_stats(self) -> RateLimiterStats
```

### `SlidingWindowRateLimiter`

```python
class SlidingWindowRateLimiter:
    def __init__(
        self,
        rate: float,
        window: float = 60.0,
        wait_timeout: float | None = None,
    ) -> None
```

### `ConcurrencyLimiter`

```python
class ConcurrencyLimiter:
    def __init__(self, max_concurrent: int) -> None
    async def acquire(self) -> AsyncContextManager
```

### `CombinedLimiter`

```python
class CombinedLimiter:
    def __init__(
        self,
        rate: float | None = None,
        max_concurrent: int | None = None,
        rate_type: str = "sliding_window",
    ) -> None
```

**Example:**
```python
from ccflow import get_limiter, CombinedLimiter

# Get default limiter
limiter = get_limiter()

# Or create custom
limiter = CombinedLimiter(rate=10.0, max_concurrent=5)

async with limiter.acquire():
    # Rate-limited operation
    pass
```

---

## Reliability & Fault Tolerance

### `CircuitBreaker`

```python
class CircuitBreaker:
    def __init__(
        self,
        config: CircuitBreakerConfig | None = None,
        name: str = "default",
    ) -> None

    async def call(self) -> AsyncContextManager
    def get_stats(self) -> CircuitBreakerStats
    def reset(self) -> None

    @property
    def state(self) -> CircuitState
```

### `CircuitBreakerConfig`

```python
@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    success_threshold: int = 2
    reset_timeout: float = 30.0
    half_open_max_calls: int = 1
```

### `CircuitState`

```python
class CircuitState(str, Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing recovery
```

### `retry_with_backoff()`

```python
async def retry_with_backoff(
    func: Callable[..., Awaitable[T]],
    config: RetryConfig | None = None,
    *args,
    **kwargs,
) -> T
```

### `RetryConfig`

```python
@dataclass
class RetryConfig:
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    exponential_base: float = 2.0
    retryable_errors: tuple = (Exception,)
```

### `HealthChecker`

```python
class HealthChecker:
    async def check(self) -> HealthStatus
    async def wait_healthy(self, timeout: float = 30.0) -> bool
```

---

## Pricing & Cost Tracking

### `ModelPricing`

```python
@dataclass(frozen=True)
class ModelPricing:
    input_per_million: float
    output_per_million: float
    cache_write_per_million: float | None = None
    cache_read_per_million: float | None = None
    batch_discount: float = 0.5

    def input_cost(self, tokens: int, *, batch: bool = False) -> float
    def output_cost(self, tokens: int, *, batch: bool = False) -> float
    def total_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        *,
        batch: bool = False,
    ) -> float
```

### Pricing Constants

```python
HAIKU_PRICING = ModelPricing(
    input_per_million=0.80,
    output_per_million=4.00,
)

SONNET_PRICING = ModelPricing(
    input_per_million=3.00,
    output_per_million=15.00,
)

OPUS_PRICING = ModelPricing(
    input_per_million=15.00,
    output_per_million=75.00,
)
```

### `calculate_cost()`

```python
def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    *,
    batch: bool = False,
) -> float
```

### `UsageStats`

```python
@dataclass
class UsageStats:
    input_tokens: int = 0
    output_tokens: int = 0
    total_cost_usd: float = 0.0
    request_count: int = 0
    model_usage: dict[str, dict] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int

    def add(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float

    def merge(self, other: UsageStats) -> None
    def reset(self) -> None
    def to_dict(self) -> dict
```

---

## Process Pool

### `ProcessPool`

```python
class ProcessPool:
    def __init__(
        self,
        config: PoolConfig | None = None,
        executor: CLIExecutor | None = None,
    ) -> None

    async def submit(
        self,
        prompt: str,
        options: CLIAgentOptions | None = None,
    ) -> PoolTask

    async def map(
        self,
        prompts: list[str],
        options: CLIAgentOptions | None = None,
    ) -> list[PoolTask]

    async def gather(self, *task_ids: str) -> list[PoolTask | Exception]
    async def get_task(self, task_id: str) -> PoolTask | None
    async def cancel(self, task_id: str) -> bool
    async def shutdown(self, wait: bool = True) -> None
    def get_stats(self) -> PoolStats
```

### `StreamingPool`

```python
class StreamingPool:
    def __init__(
        self,
        config: PoolConfig | None = None,
        executor: CLIExecutor | None = None,
    ) -> None

    async def submit(
        self,
        prompt: str,
        options: CLIAgentOptions | None = None,
    ) -> StreamingTask

    async def shutdown(self, wait: bool = True) -> None
```

### `PoolConfig`

```python
@dataclass
class PoolConfig:
    max_workers: int = 4
    queue_size: int = 100
    task_timeout: float = 300.0
```

---

## API Fallback

### `APIClient`

Direct Anthropic SDK integration for fallback when CLI is unavailable.

```python
class APIClient:
    def __init__(self, config: APIClientConfig | None = None) -> None

    @property
    def is_available(self) -> bool

    async def execute(
        self,
        prompt: str,
        options: CLIAgentOptions | None = None,
        *,
        correlation_id: str | None = None,
    ) -> AsyncIterator[dict]

    async def close(self) -> None
```

### `APIClientConfig`

```python
@dataclass
class APIClientConfig:
    api_key: str | None = None  # Uses ANTHROPIC_API_KEY env var
    base_url: str | None = None
    timeout: float = 300.0
    max_retries: int = 2
    default_model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 4096
    stream: bool = True
```

### `FallbackExecutor`

```python
class FallbackExecutor:
    def __init__(
        self,
        config: FallbackConfig | None = None,
        cli_executor: Any | None = None,
        api_client: APIClient | None = None,
    ) -> None

    async def execute(
        self,
        prompt: str,
        options: CLIAgentOptions | None = None,
        timeout: float = 300.0,
        *,
        force_api: bool = False,
        force_cli: bool = False,
    ) -> AsyncIterator[dict]
```

### `FallbackConfig`

```python
@dataclass
class FallbackConfig:
    fallback_on_cli_unavailable: bool = True
    fallback_on_circuit_open: bool = True
    fallback_on_timeout: bool = False
    api_config: APIClientConfig = field(default_factory=APIClientConfig)
```

---

## Exceptions

### Hierarchy

```
CCFlowError (base)
├── CLINotFoundError
├── CLIAuthenticationError
├── CLIExecutionError
├── CLITimeoutError
├── SessionNotFoundError
├── SessionStoreError
├── ParseError
├── ToonEncodingError
├── PermissionDeniedError
├── RateLimitExceededError
├── ConcurrencyLimitExceededError
├── CircuitBreakerError
├── RetryExhaustedError
└── APIClientError
    ├── APINotAvailableError
    └── APIRateLimitError
```

### Exception Details

```python
class CCFlowError(Exception):
    """Base exception for all ccflow errors."""

class CLINotFoundError(CCFlowError):
    """Claude CLI binary not found in PATH."""

class CLIAuthenticationError(CCFlowError):
    """Claude CLI authentication failed."""

class CLIExecutionError(CCFlowError):
    """Error during CLI execution."""
    def __init__(self, message: str, exit_code: int | None = None)

class CLITimeoutError(CCFlowError):
    """CLI execution timed out."""
    def __init__(self, message: str, timeout: float)

class SessionNotFoundError(CCFlowError):
    """Session not found in storage."""
    def __init__(self, session_id: str)

class CircuitBreakerError(CCFlowError):
    """Circuit breaker is open, requests rejected."""

class APIRateLimitError(APIClientError):
    """API rate limit exceeded."""
    def __init__(self, message: str, retry_after: float | None = None)
```

---

## Global Functions

### Singletons

```python
# Executor
def get_executor() -> CLIExecutor
def reset_executor() -> None

# Session Manager
def get_manager() -> SessionManager
async def init_manager(store: SessionStore | None = None) -> SessionManager

# Rate Limiter
def get_limiter() -> CombinedLimiter
def reset_limiter() -> None

# Circuit Breaker
def get_cli_circuit_breaker() -> CircuitBreaker
def reset_cli_circuit_breaker() -> None

# Health Checker
def get_health_checker() -> HealthChecker
def reset_health_checker() -> None

# Event Emitter
def get_emitter() -> EventEmitter
def reset_emitter() -> None

# API Client
def get_api_client() -> APIClient
async def reset_api_client() -> None

# Fallback Executor
def get_fallback_executor() -> FallbackExecutor
async def reset_fallback_executor() -> None

# Usage Tracker
def get_usage_tracker() -> UsageStats
def reset_usage_tracker() -> None
def track_usage(model: str, input_tokens: int, output_tokens: int) -> float

# Process Pools
def get_pool() -> ProcessPool
def get_streaming_pool() -> StreamingPool
async def reset_pools() -> None
```

### Correlation IDs

```python
def get_correlation_id() -> str
def set_correlation_id(cid: str | None = None) -> str
def bind_correlation_id(log: Any = None) -> structlog.BoundLogger
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CCFLOW_CLAUDE_CLI_PATH` | Auto-detect | Path to claude CLI |
| `CCFLOW_DEFAULT_MODEL` | `sonnet` | Default model |
| `CCFLOW_DEFAULT_TIMEOUT` | `300.0` | Request timeout (seconds) |
| `CCFLOW_SESSION_STORAGE_PATH` | None | SQLite database path |
| `CCFLOW_TOON_ENABLED` | `true` | Enable TOON encoding |
| `CCFLOW_ENABLE_METRICS` | `true` | Enable Prometheus metrics |
| `CCFLOW_LOG_LEVEL` | `INFO` | Logging level |
| `CCFLOW_LOG_FORMAT` | `console` | Log format (console/json) |
| `CCFLOW_MAX_CONCURRENT_REQUESTS` | `10` | Concurrency limit |
| `CCFLOW_RATE_LIMIT_PER_MINUTE` | `60` | Rate limit |

---

## Version

```python
__version__ = "0.1.0"
```
