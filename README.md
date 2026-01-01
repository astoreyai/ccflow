# ccflow

Production middleware bridging Claude Code CLI with SDK-like Python interfaces.

**ccflow** enables subscription-based usage (Pro/Max) instead of API token billing, with integrated TOON serialization for 30-60% token reduction on structured data.

## Features

- **SDK-Compatible API** - `query()` and `Session` interfaces matching Claude Agent SDK patterns
- **Subscription Billing** - Route through CLI to use Pro/Max subscription credits
- **TOON Integration** - Automatic token-optimized encoding for structured context data
- **Streaming Support** - Real-time async iteration over CLI NDJSON output
- **Session Management** - Multi-turn conversations via CLI `--resume`
- **Batch Processing** - Concurrent query execution with configurable parallelism
- **Extended Thinking** - Ultrathink mode for deeper reasoning with thinking token tracking
- **MCP Support** - Programmatic MCP server configuration
- **Observability** - Prometheus metrics, structured logging

## Installation

```bash
# Basic installation
pip install git+https://github.com/astoreyai/ccflow.git

# With TOON support (recommended)
pip install "ccflow[toon] @ git+https://github.com/astoreyai/ccflow.git"

# With all extras
pip install "ccflow[all] @ git+https://github.com/astoreyai/ccflow.git"
```

**Prerequisites:**
- Python 3.11+
- Claude Code CLI installed and authenticated (`npm install -g @anthropic-ai/claude-code`)

## Quick Start

### Basic Query

```python
import asyncio
from ccflow import query, CLIAgentOptions, TextMessage

async def main():
    options = CLIAgentOptions(
        model="sonnet",
        max_turns=5,
        allowed_tools=["Read", "Grep"],
    )

    async for msg in query("Explain this codebase", options):
        if isinstance(msg, TextMessage):
            print(msg.content, end="")

asyncio.run(main())
```

### Multi-Turn Session

```python
from ccflow import Session, CLIAgentOptions

async def main():
    session = Session(options=CLIAgentOptions(model="opus"))

    # First turn
    async for msg in session.send_message("Review this code"):
        print(msg.content, end="")

    # Follow-up (continues conversation)
    async for msg in session.send_message("Focus on security issues"):
        print(msg.content, end="")

    # Get session stats
    stats = await session.close()
    print(f"Total tokens: {stats.total_tokens}")

asyncio.run(main())
```

### TOON Context Injection

```python
from ccflow import query, CLIAgentOptions
from ccflow.types import ToonConfig

portfolio = {
    "positions": [
        {"symbol": "AAPL", "qty": 100, "pnl": 1500.00},
        {"symbol": "GOOGL", "qty": 50, "pnl": -200.00},
    ],
    "cash": 50000.00,
}

options = CLIAgentOptions(
    context=portfolio,  # Auto-TOON encoded
    toon=ToonConfig(enabled=True, track_savings=True),
)

async for msg in query("Analyze portfolio risk", options):
    print(msg.content, end="")

# Check savings
print(f"TOON saved {options.toon.last_compression_ratio:.1%} tokens")
```

### Batch Processing

```python
from ccflow import batch_query, CLIAgentOptions

prompts = [
    "Review file A for bugs",
    "Review file B for bugs",
    "Review file C for bugs",
]

results = await batch_query(prompts, CLIAgentOptions(model="haiku"), concurrency=3)

for result in results:
    print(f"{result.session_id}: {result.result[:100]}...")
```

### Extended Thinking (Ultrathink)

```python
from ccflow import query, CLIAgentOptions, ThinkingMessage, TextMessage

# Enable extended thinking for complex reasoning tasks
options = CLIAgentOptions(
    model="sonnet",
    ultrathink=True,  # Enables deep reasoning mode
)

async for msg in query("Analyze this complex algorithm and find edge cases", options):
    if isinstance(msg, ThinkingMessage):
        print(f"[Thinking] {msg.content[:100]}...")
        print(f"  Thinking tokens: {msg.thinking_tokens}")
    elif isinstance(msg, TextMessage):
        print(msg.content, end="")
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Your Python Application                   │
└─────────────────────────────────┬───────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────┐
│                      ccflow Middleware                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ query()     │  │ Session     │  │ ToonSerializer      │  │
│  │ batch_query │  │ .send()     │  │ .encode()           │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│                          │                                   │
│  ┌───────────────────────▼──────────────────────────────┐   │
│  │ CLI Executor (asyncio subprocess + NDJSON streaming) │   │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────┬───────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────┐
│  claude -p "prompt" --output-format stream-json --resume    │
└─────────────────────────────────────────────────────────────┘
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CCFLOW_DEFAULT_MODEL` | Default model | `sonnet` |
| `CCFLOW_DEFAULT_TIMEOUT` | Query timeout (seconds) | `300` |
| `CCFLOW_TOON_ENABLED` | Enable TOON encoding | `true` |
| `CCFLOW_LOG_LEVEL` | Logging level | `INFO` |
| `CCFLOW_ENABLE_METRICS` | Enable Prometheus metrics | `true` |

### CLIAgentOptions

```python
CLIAgentOptions(
    # Model
    model="sonnet",              # Model alias or full name
    fallback_model="haiku",      # Fallback on overload

    # System prompt
    system_prompt=None,          # Replace system prompt
    append_system_prompt=None,   # Append to system prompt

    # Permissions
    permission_mode=PermissionMode.ASK,
    allowed_tools=["Read", "Edit"],
    disallowed_tools=["Bash(rm:*)"],

    # Session
    session_id=None,             # Specific session UUID
    resume=False,                # Resume previous session

    # Limits
    max_turns=10,
    timeout=300.0,

    # Extended thinking
    ultrathink=False,            # Enable deep reasoning mode

    # TOON
    toon=ToonConfig(enabled=True),
    context={"data": "to inject"},  # Auto-encoded

    # MCP
    mcp_servers={"github": MCPServerConfig(...)},
)
```

## TOON Serialization

TOON (Token-Oriented Object Notation) reduces token consumption by 30-60% for structured data:

```
JSON (47 tokens):                    TOON (20 tokens):
{"positions": [                      positions[2]{symbol,qty,price}:
  {"symbol":"AAPL","qty":100,...},     AAPL,100,150.25
  {"symbol":"GOOGL","qty":50,...}      GOOGL,50,2800
]}
```

See [docs/TOON_RESEARCH.md](docs/TOON_RESEARCH.md) for the complete specification.

## CLI Usage

```bash
# Basic query
ccflow "Explain this code"

# With model selection
ccflow -m opus "Review for security"

# Streaming mode
ccflow --stream "Analyze the codebase"

# Pipe input
cat file.py | ccflow "Explain this"
```

## Development

```bash
# Clone and install
git clone https://github.com/astoreyai/ccflow.git
cd ccflow
pip install -e ".[dev]"

# Run tests
pytest

# Type checking
mypy src/

# Lint
ruff check src/ tests/
```

## License

MIT

## Related

- [Claude Code CLI](https://code.claude.com/)
- [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview)
- [TOON Format](https://toonformat.dev/)
