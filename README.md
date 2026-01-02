# ccflow

[![PyPI version](https://img.shields.io/pypi/v/ccflow.svg)](https://pypi.org/project/ccflow/)
[![Python versions](https://img.shields.io/pypi/pyversions/ccflow.svg)](https://pypi.org/project/ccflow/)
[![License](https://img.shields.io/pypi/l/ccflow.svg)](https://github.com/astoreyai/ccflow/blob/main/LICENSE)
[![Tests](https://github.com/astoreyai/ccflow/actions/workflows/ci.yml/badge.svg)](https://github.com/astoreyai/ccflow/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-86%25-brightgreen.svg)](https://github.com/astoreyai/ccflow)

**Production middleware bridging Claude Code CLI with SDK-like Python interfaces.**

ccflow enables **subscription-based usage** (Pro/Max) instead of API token billing, with integrated TOON serialization for **30-60% token reduction** on structured data.

**v0.2.0**: Now with **Agent System**, **Hooks**, **Skills**, **Subagent Coordination**, and **CLI Commands** - full SDK parity!

---

## Why ccflow?

| Feature | Claude API | ccflow |
|---------|------------|--------|
| **Billing** | Per-token API costs | Pro/Max subscription |
| **Auth** | API keys | CLI OAuth (already configured) |
| **Tools** | Manual implementation | CLI's 15+ built-in tools |
| **MCP** | Not available | Full MCP server support |
| **Sessions** | Manual state management | Automatic persistence |

---

## Installation

```bash
# From PyPI
pip install ccflow

# With TOON support (recommended for 30-60% token savings)
pip install "ccflow[toon]"

# With all extras (server, tracing, API fallback)
pip install "ccflow[all]"
```

**Prerequisites:**
- Python 3.11+
- Claude Code CLI installed and authenticated:
  ```bash
  npm install -g @anthropic-ai/claude-code
  claude auth login
  ```

---

## Quick Start

### Simple Query

```python
from ccflow import query_simple

response = await query_simple("What is 2+2?")
print(response)  # "4"
```

### Streaming Query

```python
from ccflow import query, CLIAgentOptions, TextMessage

options = CLIAgentOptions(
    model="sonnet",
    max_turns=5,
    allowed_tools=["Read", "Grep"],
)

async for msg in query("Explain this codebase", options):
    if isinstance(msg, TextMessage):
        print(msg.content, end="")
```

### Multi-Turn Session

```python
from ccflow import Session, CLIAgentOptions

session = Session(options=CLIAgentOptions(model="opus"))

# First turn
async for msg in session.send_message("Review this code"):
    print(msg.content, end="")

# Follow-up (conversation continues)
async for msg in session.send_message("Focus on security issues"):
    print(msg.content, end="")

stats = await session.close()
print(f"Total tokens: {stats.total_tokens}")
```

### Extended Thinking (Ultrathink)

```python
from ccflow import query, CLIAgentOptions, ThinkingMessage, TextMessage

options = CLIAgentOptions(
    model="sonnet",
    ultrathink=True,  # Enable deep reasoning
)

async for msg in query("Analyze this algorithm for edge cases", options):
    if isinstance(msg, ThinkingMessage):
        print(f"[Thinking: {msg.thinking_tokens} tokens]")
    elif isinstance(msg, TextMessage):
        print(msg.content, end="")
```

### Batch Processing

```python
from ccflow import batch_query, CLIAgentOptions

prompts = ["Review file A", "Review file B", "Review file C"]
results = await batch_query(prompts, CLIAgentOptions(), concurrency=3)

for result in results:
    print(f"{result.session_id}: {result.result[:100]}...")
```

---

## Agent System (v0.2.0)

Define specialized agents with isolated tools, models, and system prompts:

### Programmatic Definition

```python
from ccflow import Agent, AgentDefinition, CLIAgentOptions

# Define agent
reviewer = Agent(AgentDefinition(
    name="code-reviewer",
    description="Expert code reviewer for quality and security",
    system_prompt="You are a senior code reviewer focused on security...",
    tools=["Read", "Grep", "Glob"],
    model="sonnet",
))

# Execute task
async for msg in reviewer.execute("Review auth.py for security issues"):
    print(msg.content, end="")
```

### Filesystem Agents (YAML Frontmatter)

Create `~/.claude/agents/code-reviewer.md`:

```markdown
---
name: code-reviewer
description: Expert code reviewer for quality and security
model: sonnet
tools:
  - Read
  - Grep
  - Glob
permission_mode: default
timeout: 300
---

# Code Reviewer

You are a senior code reviewer...
```

```python
from ccflow import get_agent_registry

# Auto-discovered from ~/.claude/agents/
registry = get_agent_registry()
agent = registry.get("code-reviewer")
```

---

## Hook System (v0.2.0)

SDK-compatible lifecycle hooks for tool execution and session events:

```python
from ccflow import get_hook_registry, HookEvent, HookContext

hooks = get_hook_registry()

# Audit all Bash commands
@hooks.on(HookEvent.PRE_TOOL_USE, pattern="Bash.*")
async def audit_bash(ctx: HookContext) -> HookContext:
    print(f"Bash command: {ctx.tool_input}")
    return ctx

# Log all tool results
@hooks.on(HookEvent.POST_TOOL_USE)
async def log_results(ctx: HookContext) -> HookContext:
    print(f"Tool {ctx.tool_name} returned: {ctx.tool_result[:100]}...")
    return ctx

# Session end cleanup
@hooks.on(HookEvent.STOP)
async def cleanup(ctx: HookContext) -> None:
    print(f"Session {ctx.session_id} ended: {ctx.stop_reason}")
```

### Hook Events

| Event | Description |
|-------|-------------|
| `PRE_TOOL_USE` | Before tool execution (can modify input) |
| `POST_TOOL_USE` | After tool execution (can modify result) |
| `USER_PROMPT_SUBMIT` | User submits prompt |
| `STOP` | Agent stops execution |
| `SUBAGENT_STOP` | Subagent finishes |
| `PRE_COMPACT` | Before context compaction |

---

## Skill System (v0.2.0)

Domain knowledge with semantic matching (SKILL.md format):

### Define Skill

Create `~/.claude/skills/security-audit/SKILL.md`:

```markdown
---
name: security-audit
description: Security vulnerability analysis and OWASP compliance
keywords:
  - security
  - vulnerability
  - OWASP
---

# Security Audit Skill

When analyzing code for security issues:
1. Check for injection vulnerabilities
2. Verify authentication flows
3. Audit authorization controls
...
```

### Use Skills

```python
from ccflow import get_skill_loader

loader = get_skill_loader()

# Semantic matching
skills = loader.match("check for vulnerabilities")
for skill in skills:
    print(f"Matched: {skill.name}")

# Load full content
skill = loader.load_full("security-audit")
print(skill.instructions)
```

---

## Subagent Coordination (v0.2.0)

Run multiple agents in parallel with concurrency control:

```python
from ccflow import get_subagent_coordinator

coordinator = get_subagent_coordinator()

# Parallel execution with semaphore-based concurrency
results = await coordinator.parallel([
    ("code-reviewer", "Review auth.py"),
    ("security-auditor", "Check for vulnerabilities"),
    ("test-runner", "Run test suite"),
])

for result in results:
    print(result)
```

### Background Tasks

```python
# Spawn background task
task_id = await coordinator.spawn_background(
    "code-reviewer",
    "Review the entire codebase"
)

# Do other work...

# Get result when ready
task = await coordinator.get_task_result(task_id, wait=True)
print(f"Status: {task.status}, Result: {task.result}")

# Cancel if needed
coordinator.cancel(task_id)
```

---

## CLI Commands (v0.2.0)

Built-in `/command` system:

```bash
# List available agents
ccflow /agents

# List available skills
ccflow /skills

# Spawn subagent
ccflow /spawn code-reviewer "Review main.py"

# List registered hooks
ccflow /hooks

# Show help
ccflow /help
```

### Register Custom Commands

```python
from ccflow import get_command_registry

registry = get_command_registry()

@registry.command("status", "Show project status")
async def cmd_status() -> str:
    return "All systems operational"
```

---

## Agentic Harness (v0.2.0)

Tools for long-running agent workflows based on [Anthropic's best practices](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents):

### Feature Tracking (JSON-based)

```python
from ccflow import FeatureList, Feature, FeatureStatus

features = FeatureList("features.json", auto_save=True)

# Add features with priority
features.add(Feature(id="auth", description="OAuth2 login", priority=1))
features.add(Feature(id="api", description="REST endpoints", priority=2))

# Work through features
feature = features.next_feature()  # Gets highest priority pending/failing
features.mark_in_progress(feature.id)
features.mark_passing(feature.id)  # or mark_failing(id, error="...")

print(f"Progress: {features.progress_percent():.0f}%")
```

### Progress Tracking (Cross-Session)

```python
from ccflow import ProgressTracker

tracker = ProgressTracker("progress.json", auto_save=True)
tracker.set_context(session_id="abc123", agent_name="builder")

tracker.log_feature_started("auth", "Implementing OAuth2")
tracker.log("checkpoint", feature_id="auth", details="JWT working")
tracker.log_feature_completed("auth", git_commit="abc123")

# Summary for session restart
print(tracker.recent_summary(10))
```

### Combined Harness

```python
from ccflow import AgentHarness, Feature

harness = AgentHarness(project_dir="my_project", auto_save=True)
harness.features.add(Feature(id="f1", description="First task"))

# Initialize session (returns context for agent)
context = harness.init_session("coding-agent", "session-001")

# Feature lifecycle
harness.start_feature("f1")
harness.complete_feature(git_commit="abc123")

# Status report for agent restarts
print(harness.status_report())
```

---

## TOON Token Optimization

TOON (Token-Oriented Object Notation) reduces token consumption by 30-60% for structured data:

```
JSON (47 tokens):                    TOON (20 tokens):
{"positions": [                      positions[2]{symbol,qty,price}:
  {"symbol":"AAPL","qty":100,...},     AAPL,100,150.25
  {"symbol":"GOOGL","qty":50,...}      GOOGL,50,2800
]}
```

```python
from ccflow import query, CLIAgentOptions
from ccflow.types import ToonConfig

options = CLIAgentOptions(
    context={"positions": [...], "cash": 50000},  # Auto-TOON encoded
    toon=ToonConfig(enabled=True, track_savings=True),
)

async for msg in query("Analyze portfolio risk", options):
    print(msg.content, end="")

print(f"Saved {options.toon.last_compression_ratio:.1%} tokens")
```

---

## Project & Trace Recording

Organize sessions hierarchically and capture full traces for analysis and replay:

```python
from ccflow import Project, CLIAgentOptions
from ccflow.stores import SQLiteProjectStore, SQLiteTraceStore, SQLiteSessionStore

# Initialize stores
db_path = "ccflow.db"
project = Project(
    name="Code Review",
    store=SQLiteProjectStore(db_path),
    trace_store=SQLiteTraceStore(db_path),
    session_store=SQLiteSessionStore(db_path),
)
await project.save()

# Create traced session
session = project.create_session(
    options=CLIAgentOptions(model="sonnet", ultrathink=True),
    detailed=True,  # Capture message-level stream
)

async for msg in session.send_message("Analyze this function"):
    print(msg.content, end="")

# Access trace
trace = session.last_trace
print(f"Thinking tokens: {trace.thinking_tokens}")
print(f"Tool calls: {len(trace.tool_calls)}")
print(f"Cost: ${trace.cost_usd:.4f}")

# Replay with different model
new_session = await project.replay_as_new(
    trace.trace_id,
    options_override=CLIAgentOptions(model="opus"),
)
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Your Python Application                     │
│   agent = Agent("code-reviewer", tools=["Read", "Grep"])        │
│   async for msg in agent.execute("Review auth.py"):             │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                     Agent Layer (v0.2.0)                         │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│  │ AgentRegistry│  │ SkillLoader  │  │ SubagentCoordinator   │  │
│  │ .register()  │  │ .match()     │  │ .spawn() .parallel()  │  │
│  └──────────────┘  └──────────────┘  └───────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                     Hook Layer (v0.2.0)                          │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ HookRegistry: pre_tool, post_tool, stop, subagent_stop   │   │
│  │ HookMatcher: tool pattern matching, async callbacks      │   │
│  └──────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                      ccflow Core Layer                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐      │
│  │ query()     │  │ Session     │  │ ToonSerializer      │      │
│  │ batch_query │  │ .send()     │  │ .encode()           │      │
│  └─────────────┘  └─────────────┘  └─────────────────────┘      │
│                          │                                       │
│  ┌───────────────────────▼──────────────────────────────┐       │
│  │ CLI Executor (asyncio subprocess + NDJSON streaming) │       │
│  └───────────────────────────────────────────────────────┘      │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────┐
│  claude -p "prompt" --output-format stream-json --resume        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Configuration

### CLIAgentOptions

```python
CLIAgentOptions(
    # Model
    model="sonnet",              # haiku, sonnet, opus
    fallback_model="haiku",      # On overload

    # System prompt
    system_prompt=None,          # Replace system prompt
    append_system_prompt=None,   # Append to system prompt

    # Permissions
    permission_mode=PermissionMode.DEFAULT,
    allowed_tools=["Read", "Edit"],
    disallowed_tools=["Bash(rm:*)"],

    # Session
    session_id=None,             # Specific session UUID
    resume=False,                # Resume previous session

    # Limits
    max_turns=10,
    timeout=300.0,

    # Extended thinking
    ultrathink=False,            # Enable deep reasoning

    # Context
    context={"data": "value"},   # Auto-TOON encoded
    toon=ToonConfig(enabled=True),

    # MCP
    mcp_servers={"github": MCPServerConfig(...)},
)
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CCFLOW_DEFAULT_MODEL` | Default model | `sonnet` |
| `CCFLOW_DEFAULT_TIMEOUT` | Timeout (seconds) | `300` |
| `CCFLOW_TOON_ENABLED` | Enable TOON encoding | `true` |
| `CCFLOW_LOG_LEVEL` | Logging level | `INFO` |
| `CCFLOW_ENABLE_METRICS` | Enable Prometheus | `true` |

---

## Message Types

```python
Message = (
    InitMessage |           # Session initialization
    TextMessage |           # Text content from Claude
    ThinkingMessage |       # Extended thinking content
    ToolUseMessage |        # Tool invocation
    ToolResultMessage |     # Tool execution result
    ErrorMessage |          # Error from CLI
    StopMessage |           # Turn completion with usage
    ResultMessage |         # Final result
    UnknownMessage          # Forward-compatible fallback
)
```

---

## Reliability Features

### Rate Limiting

```python
from ccflow import CombinedLimiter

limiter = CombinedLimiter(rate=10.0, max_concurrent=5)
async with limiter.acquire():
    # rate-limited operation
```

### Circuit Breaker

```python
from ccflow import CircuitBreaker, CircuitBreakerConfig

breaker = CircuitBreaker(CircuitBreakerConfig(
    failure_threshold=5,
    recovery_timeout=30.0,
))
async with breaker.call():
    # protected operation
```

### Retry with Backoff

```python
from ccflow import retry_with_backoff, RetryConfig

result = await retry_with_backoff(
    async_func,
    RetryConfig(max_retries=3, base_delay=1.0, exponential=True),
)
```

---

## CLI Usage

```bash
# Basic query
ccflow "Explain this code"

# With model selection
ccflow -m opus "Review for security"

# Streaming mode
ccflow --stream "Analyze the codebase"

# Session management
ccflow sessions --list
ccflow sessions --resume <session-id>

# HTTP server
ccflow server --port 8080
```

---

## Docker

```bash
# Build
docker build -t ccflow .

# Run (mount Claude credentials)
docker run -v ~/.claude:/home/ccflow/.claude ccflow

# Docker Compose (with Prometheus/Grafana)
docker compose up
```

---

## Development

```bash
# Clone and install
git clone https://github.com/astoreyai/ccflow.git
cd ccflow
pip install -e ".[dev]"

# Run tests
pytest

# Type checking
mypy src/ccflow/

# Lint
ruff check src/ tests/
```

---

## API Reference

See [docs/api.md](docs/api.md) for complete API documentation.

For LLM-friendly documentation, see [LLMS.txt](LLMS.txt).

---

## License

[MIT](LICENSE)

---

## Links

- [Documentation](https://github.com/astoreyai/ccflow#readme)
- [PyPI](https://pypi.org/project/ccflow/)
- [Changelog](CHANGELOG.md)
- [Claude Code CLI](https://code.claude.com/)
- [TOON Format](https://toonformat.dev/)
