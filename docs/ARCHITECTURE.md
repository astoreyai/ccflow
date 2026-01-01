# ccflow Architecture Design

**Project**: Claude Code CLI ↔ SDK Middleware
**Version**: 1.0.0-draft
**Date**: 2025-12-31

---

## 1. Overview

ccflow is a production middleware that bridges the Claude Code CLI with SDK-like Python interfaces, enabling:

1. **Subscription Billing** - Route through CLI to use Pro/Max subscription instead of API tokens
2. **Token Efficiency** - TOON serialization reduces context tokens by 40-60%
3. **SDK Ergonomics** - Async Python interface matching Claude Agent SDK patterns
4. **Session Continuity** - Multi-turn conversations via CLI `--resume`

**Target Cost Reduction**: 60-80% vs direct API usage

---

## 2. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           User Application                                   │
│                                                                              │
│   async for msg in query("Analyze portfolio", options):                     │
│       print(msg.content)                                                     │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ccflow Middleware Stack                               │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ Layer 4: SDK-Compatible API                                              ││
│  │ • query() - async generator matching Agent SDK                          ││
│  │ • Session - stateful multi-turn interface                               ││
│  │ • batch_query() - concurrent processing                                 ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                      │                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ Layer 3: TOON Serialization                                              ││
│  │ • Auto-encode context objects                                           ││
│  │ • Track compression metrics                                             ││
│  │ • Format for system prompt injection                                    ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                      │                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ Layer 2: Permission & Config Translation                                 ││
│  │ • SDK options → CLI flags                                               ││
│  │ • Tool permissions formatting                                           ││
│  │ • MCP config file generation                                            ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                      │                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ Layer 1: CLI Executor                                                    ││
│  │ • asyncio subprocess management                                         ││
│  │ • NDJSON stream parsing                                                 ││
│  │ • Session ID extraction & management                                    ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  claude -p "prompt" --output-format stream-json --resume <session-id>        │
│          --allowedTools "Read,Edit,Bash" --append-system-prompt "[TOON]..."  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                              Claude API (via CLI)
```

---

## 3. Package Structure

```
ccflow/
├── pyproject.toml
├── README.md
├── docs/
│   ├── ARCHITECTURE.md          # This file
│   ├── TOON_RESEARCH.md         # TOON format research
│   └── API.md                   # API documentation
├── src/
│   └── ccflow/
│       ├── __init__.py          # Public API exports
│       ├── api.py               # query(), Session, batch_query()
│       ├── executor.py          # CLI subprocess management
│       ├── parser.py            # NDJSON stream parser
│       ├── session.py           # Session state management
│       ├── permissions.py       # Tool permission translation
│       ├── mcp.py               # MCP configuration management
│       ├── toon_integration.py  # TOON wrapper
│       ├── types.py             # Pydantic models
│       ├── config.py            # Configuration management
│       ├── metrics.py           # Observability
│       └── exceptions.py        # Custom exceptions
├── tests/
│   ├── conftest.py
│   ├── test_executor.py
│   ├── test_parser.py
│   ├── test_session.py
│   ├── test_toon.py
│   └── fixtures/
│       └── cli_responses/       # Mocked CLI NDJSON
└── examples/
    ├── basic_query.py
    ├── multi_turn_session.py
    ├── batch_processing.py
    └── toon_context.py
```

---

## 4. Component Specifications

### 4.1 CLI Executor (`executor.py`)

**Responsibility**: Low-level async subprocess management for `claude -p` invocation.

```python
class CLIExecutor:
    """Manages claude CLI subprocess execution with streaming."""

    async def execute(
        self,
        prompt: str,
        flags: list[str],
        timeout: float = 300.0,
        cwd: Path | None = None,
    ) -> AsyncIterator[dict]:
        """
        Execute claude CLI and stream NDJSON responses.

        Args:
            prompt: The prompt to send to Claude
            flags: CLI flags (e.g., ["--output-format", "stream-json"])
            timeout: Maximum execution time in seconds
            cwd: Working directory for CLI execution

        Yields:
            Parsed NDJSON events as dictionaries

        Raises:
            CLIExecutionError: If subprocess fails
            CLITimeoutError: If execution exceeds timeout
        """
```

**Key Implementation Details**:
- Use `asyncio.create_subprocess_exec()` for non-blocking execution
- Parse stdout line-by-line as NDJSON
- Capture stderr for error messages
- Handle process termination gracefully

### 4.2 Stream Parser (`parser.py`)

**Responsibility**: Parse NDJSON stream into typed Message objects.

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
    delta_type: Literal["text_delta"] = "text_delta"

@dataclass
class ToolUseMessage:
    tool: str
    args: dict
    type: Literal["tool_use"] = "tool_use"

@dataclass
class ToolResultMessage:
    tool: str
    result: str
    type: Literal["tool_result"] = "tool_result"

@dataclass
class StopMessage:
    session_id: str
    usage: dict  # {input_tokens, output_tokens}
    type: Literal["system"] = "system"
    subtype: Literal["stop"] = "stop"

Message = InitMessage | TextMessage | ToolUseMessage | ToolResultMessage | StopMessage

class StreamParser:
    """Parses NDJSON stream into typed Message objects."""

    def parse_line(self, line: bytes) -> Message:
        """Parse single NDJSON line into Message."""

    async def parse_stream(
        self,
        stream: asyncio.StreamReader,
    ) -> AsyncIterator[Message]:
        """Parse streaming NDJSON into Messages."""
```

### 4.3 Session Manager (`session.py`)

**Responsibility**: Map SDK sessions to CLI `--resume` flags.

```python
@dataclass
class SessionStats:
    session_id: str
    total_turns: int
    total_input_tokens: int
    total_output_tokens: int
    duration_seconds: float
    toon_savings_ratio: float

class Session:
    """Stateful session matching Agent SDK Session interface."""

    def __init__(
        self,
        session_id: str | None = None,
        options: CLIAgentOptions | None = None,
    ):
        self._session_id = session_id or str(uuid.uuid4())
        self._options = options or CLIAgentOptions()
        self._turn_count = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0

    @property
    def session_id(self) -> str:
        return self._session_id

    async def send_message(
        self,
        content: str,
    ) -> AsyncIterator[Message]:
        """Send message and stream response."""

    async def fork(self) -> "Session":
        """Create new session branched from current state."""

    async def close(self) -> SessionStats:
        """Close session and return statistics."""
```

### 4.4 Permission Translator (`permissions.py`)

**Responsibility**: Convert SDK permission configs to CLI flags.

```python
class PermissionTranslator:
    """Translates SDK permissions to CLI flags."""

    @staticmethod
    def translate_allowed_tools(tools: list[str]) -> list[str]:
        """
        Convert tool list to --allowedTools format.

        Input: ["Read", "Edit", "Bash(git:*)"]
        Output: ["--allowedTools", "Read", "Edit", "Bash(git:*)"]
        """

    @staticmethod
    def translate_permission_mode(mode: PermissionMode) -> list[str]:
        """
        Convert permission mode to CLI flag.

        Input: PermissionMode.ACCEPT_EDITS
        Output: ["--permission-mode", "acceptEdits"]
        """

    @staticmethod
    def translate_mcp_tools(mcp_tools: dict[str, list[str]]) -> list[str]:
        """
        Format MCP tool permissions.

        Input: {"github": ["get_issue", "list_prs"]}
        Output: ["mcp__github__get_issue", "mcp__github__list_prs"]
        """
```

### 4.5 MCP Config Manager (`mcp.py`)

**Responsibility**: Generate temporary MCP configuration files.

```python
@dataclass
class MCPServerConfig:
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    transport: Literal["stdio", "sse", "http"] = "stdio"
    url: str | None = None  # For SSE/HTTP

class MCPConfigManager:
    """Manages MCP server configuration files."""

    def __init__(self, temp_dir: Path | None = None):
        self._temp_dir = temp_dir or Path(tempfile.gettempdir())

    def create_config_file(
        self,
        servers: dict[str, MCPServerConfig],
    ) -> Path:
        """
        Create temporary MCP config JSON file.

        Returns path to config file for --mcp-config flag.
        """

    def cleanup(self) -> None:
        """Remove temporary config files."""
```

### 4.6 TOON Integration (`toon_integration.py`)

See [TOON_RESEARCH.md](./TOON_RESEARCH.md) for complete specification.

```python
class ToonSerializer:
    """Wrapper around toon-format library."""

    def __init__(self, config: ToonConfig):
        self.config = config

    def encode(self, data: Any) -> str:
        """Encode Python object to TOON format."""

    def decode(self, toon_str: str) -> Any:
        """Decode TOON format to Python object."""

    def format_for_prompt(self, data: Any, label: str) -> str:
        """Format data for system prompt injection."""
```

### 4.7 SDK-Compatible API (`api.py`)

**Responsibility**: Public interface matching Agent SDK patterns.

```python
async def query(
    prompt: str,
    options: CLIAgentOptions | None = None,
) -> AsyncIterator[Message]:
    """
    Execute single query with streaming response.

    Drop-in replacement for claude_agent_sdk.query().

    Args:
        prompt: The prompt to send to Claude
        options: Configuration options

    Yields:
        Message objects (text, tool_use, tool_result, etc.)

    Example:
        async for msg in query("Explain this code", options):
            if isinstance(msg, TextMessage):
                print(msg.content, end="")
    """

async def batch_query(
    prompts: list[str],
    options: CLIAgentOptions | None = None,
    concurrency: int = 5,
) -> list[QueryResult]:
    """
    Execute multiple independent queries concurrently.

    Args:
        prompts: List of prompts to process
        options: Shared configuration options
        concurrency: Maximum concurrent executions

    Returns:
        List of QueryResult objects with responses
    """
```

---

## 5. Type Definitions (`types.py`)

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Any

class PermissionMode(str, Enum):
    PLAN = "plan"
    ASK = "askPermissions"
    ACCEPT_EDITS = "acceptEdits"
    BYPASS = "bypassPermissions"

@dataclass
class ToonConfig:
    """TOON serialization configuration."""
    enabled: bool = True
    delimiter: Literal[",", "\t", "|"] = ","
    indent: int = 2
    length_marker: bool = False
    encode_context: bool = True
    encode_tool_results: bool = True
    track_savings: bool = True
    _last_json_tokens: int = field(default=0, repr=False)
    _last_toon_tokens: int = field(default=0, repr=False)

    @property
    def last_compression_ratio(self) -> float:
        if self._last_json_tokens == 0:
            return 0.0
        return 1.0 - (self._last_toon_tokens / self._last_json_tokens)

@dataclass
class CLIAgentOptions:
    """Configuration options for CLI middleware."""

    # Model selection
    model: str = "sonnet"
    fallback_model: str | None = None

    # System prompt
    system_prompt: str | None = None
    append_system_prompt: str | None = None

    # Permissions
    permission_mode: PermissionMode = PermissionMode.ASK
    allowed_tools: list[str] | None = None
    disallowed_tools: list[str] | None = None

    # Session
    session_id: str | None = None
    resume: bool = False
    fork_session: bool = False

    # Execution limits
    max_turns: int | None = None
    timeout: float = 300.0

    # Working directory
    cwd: str | None = None
    add_dirs: list[str] | None = None

    # MCP
    mcp_servers: dict[str, "MCPServerConfig"] | None = None
    strict_mcp: bool = False

    # TOON
    toon: ToonConfig = field(default_factory=ToonConfig)

    # Context (will be TOON-encoded if enabled)
    context: dict[str, Any] | None = None

    # Output
    verbose: bool = False
    include_partial: bool = False

@dataclass
class QueryResult:
    """Result from a completed query."""
    session_id: str
    result: str
    input_tokens: int
    output_tokens: int
    duration_seconds: float
    toon_savings_ratio: float
    error: str | None = None
```

---

## 6. Error Handling (`exceptions.py`)

```python
class CCFlowError(Exception):
    """Base exception for ccflow middleware."""

class CLINotFoundError(CCFlowError):
    """Claude CLI not found in PATH."""

class CLIAuthenticationError(CCFlowError):
    """Claude CLI not authenticated."""

class CLIExecutionError(CCFlowError):
    """CLI subprocess execution failed."""

    def __init__(self, message: str, stderr: str, exit_code: int):
        super().__init__(message)
        self.stderr = stderr
        self.exit_code = exit_code

class CLITimeoutError(CCFlowError):
    """CLI execution exceeded timeout."""

class SessionNotFoundError(CCFlowError):
    """Requested session ID not found."""

class ParseError(CCFlowError):
    """Failed to parse CLI output."""

class ToonEncodingError(CCFlowError):
    """TOON encoding failed."""

class PermissionDeniedError(CCFlowError):
    """Tool permission was denied."""
```

---

## 7. Configuration (`config.py`)

```python
from pydantic_settings import BaseSettings

class CCFlowSettings(BaseSettings):
    """Environment-based configuration."""

    # CLI path (auto-detect if not set)
    claude_cli_path: str | None = None

    # Default model
    default_model: str = "sonnet"

    # Default timeout (seconds)
    default_timeout: float = 300.0

    # Session storage
    session_storage_path: str | None = None

    # TOON defaults
    toon_enabled: bool = True
    toon_delimiter: str = ","

    # Observability
    enable_metrics: bool = True
    log_level: str = "INFO"

    # Rate limiting
    max_concurrent_requests: int = 10
    rate_limit_per_minute: int = 60

    class Config:
        env_prefix = "CCFLOW_"
```

---

## 8. Metrics & Observability (`metrics.py`)

```python
from prometheus_client import Counter, Histogram, Gauge

# Request metrics
REQUESTS_TOTAL = Counter(
    "ccflow_requests_total",
    "Total CLI requests",
    ["model", "status"]
)

REQUEST_DURATION = Histogram(
    "ccflow_request_duration_seconds",
    "Request duration in seconds",
    ["model"]
)

# Token metrics
TOKENS_INPUT = Counter(
    "ccflow_tokens_input_total",
    "Total input tokens consumed",
    ["model"]
)

TOKENS_OUTPUT = Counter(
    "ccflow_tokens_output_total",
    "Total output tokens generated",
    ["model"]
)

# TOON metrics
TOON_SAVINGS_RATIO = Histogram(
    "ccflow_toon_savings_ratio",
    "TOON compression ratio (0-1)",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
)

# Session metrics
ACTIVE_SESSIONS = Gauge(
    "ccflow_active_sessions",
    "Currently active sessions"
)
```

---

## 9. Usage Examples

### 9.1 Basic Query

```python
from ccflow import query, CLIAgentOptions

async def main():
    options = CLIAgentOptions(
        model="sonnet",
        allowed_tools=["Read", "Grep"],
        max_turns=10,
    )

    async for msg in query("Find all TODO comments in src/", options):
        if hasattr(msg, "content"):
            print(msg.content, end="")

asyncio.run(main())
```

### 9.2 Multi-Turn Session

```python
from ccflow import Session, CLIAgentOptions

async def main():
    session = Session(options=CLIAgentOptions(
        model="opus",
        permission_mode="acceptEdits",
    ))

    # First message
    async for msg in session.send_message("Review this codebase"):
        print_message(msg)

    # Follow-up
    async for msg in session.send_message("Focus on security issues"):
        print_message(msg)

    # Get stats
    stats = await session.close()
    print(f"Total tokens: {stats.total_input_tokens + stats.total_output_tokens}")

asyncio.run(main())
```

### 9.3 TOON Context Injection

```python
from ccflow import query, CLIAgentOptions, ToonConfig

async def main():
    # Structured data to inject
    portfolio = {
        "positions": [
            {"symbol": "AAPL", "qty": 100, "pnl": 1500.00},
            {"symbol": "GOOGL", "qty": 50, "pnl": -200.00},
        ],
        "cash": 50000.00,
        "margin_used": 0.35,
    }

    options = CLIAgentOptions(
        context=portfolio,  # Auto-TOON encoded
        toon=ToonConfig(
            enabled=True,
            delimiter="\t",  # Tab delimiter for extra savings
            track_savings=True,
        ),
    )

    async for msg in query("Analyze risk exposure", options):
        print_message(msg)

    # Check savings
    print(f"TOON saved {options.toon.last_compression_ratio:.1%} tokens")

asyncio.run(main())
```

### 9.4 Batch Processing

```python
from ccflow import batch_query, CLIAgentOptions

async def main():
    prompts = [
        "Review src/auth.py for security issues",
        "Check src/database.py for SQL injection",
        "Analyze src/api.py for rate limiting",
    ]

    options = CLIAgentOptions(
        model="haiku",  # Fast model for batch
        allowed_tools=["Read", "Grep"],
        max_turns=5,
    )

    results = await batch_query(prompts, options, concurrency=3)

    for result in results:
        print(f"Session: {result.session_id}")
        print(f"Result: {result.result[:200]}...")
        print(f"Tokens: {result.input_tokens + result.output_tokens}")
        print()

asyncio.run(main())
```

---

## 10. Testing Strategy

### Unit Tests
- Mock subprocess calls with fixture NDJSON responses
- Test parser with various message types
- Test permission translation logic
- Test TOON encoding edge cases

### Integration Tests
- Run actual CLI in isolated environment
- Test session resume functionality
- Test MCP server configuration
- Verify TOON round-trip encoding

### Load Tests
- Concurrent session handling
- Rate limiting behavior
- Memory usage under load

---

## 11. Dependencies

```toml
[project]
dependencies = [
    "toon-format @ git+https://github.com/toon-format/toon-python.git",
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "prometheus-client>=0.17",
    "structlog>=23.0",
]

[project.optional-dependencies]
server = [
    "fastapi>=0.100",
    "uvicorn>=0.23",
    "websockets>=11.0",
]
dev = [
    "pytest>=7.0",
    "pytest-asyncio>=0.21",
    "pytest-cov>=4.0",
    "ruff>=0.1",
    "mypy>=1.0",
]
```

---

## 12. Next Steps

1. **Scaffold project structure** with pyproject.toml
2. **Implement Layer 1** (CLI Executor) with basic streaming
3. **Implement Layer 2** (Parser) with message types
4. **Add TOON integration** using toon-format library
5. **Implement Session** management
6. **Add SDK-compatible API** layer
7. **Write comprehensive tests**
8. **Add observability** (metrics, logging)
9. **Optional: FastAPI server** for remote access
