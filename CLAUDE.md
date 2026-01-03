# ccflow Development Guide

## Project Overview

**ccflow v0.2.0** - Production middleware bridging Claude Code CLI with SDK-like Python interfaces.

- **Purpose**: Enable subscription-based Claude usage (Pro/Max) instead of API tokens
- **Key Feature**: TOON serialization for 30-60% token reduction
- **Python**: 3.11+ (async-first architecture)

## Quick Start

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=src/ccflow --cov-report=term-missing

# Type checking
mypy src/ccflow/

# Lint
ruff check src/ tests/
```

## Architecture

```
Application   →  query(), Session, SessionManager
Agent Layer   →  Agent, Hooks, Skills, SubagentCoordinator
Middleware    →  Parser, Events, CircuitBreaker, RateLimiter, TOON
Executor      →  CLIExecutor (subprocess + NDJSON streaming)
     ↓
Claude Code CLI (--resume, stream-json)
```

## Key Modules

| Module | Purpose |
|--------|---------|
| `api.py` | High-level query interface (query, batch_query) |
| `session.py` | Multi-turn conversation management |
| `executor.py` | CLI subprocess + NDJSON streaming |
| `parser.py` | NDJSON → typed Message objects |
| `reliability.py` | Circuit breaker, retry, health checks |
| `rate_limiting.py` | Token bucket, sliding window limiters |
| `harness.py` | Feature tracking, progress management |
| `agent.py` | Specialized agents with isolated tools |
| `hooks.py` | Lifecycle hooks (pre/post tool, stop) |
| `skills.py` | Domain knowledge with semantic matching |
| `subagent.py` | Parallel agent coordination |
| `tracing.py` | Full trace capture & replay |
| `stores/` | Persistence backends (SQLite, memory) |

## Design Patterns

1. **Singleton Factories**: `get_executor()`, `get_limiter()`, `get_emitter()` with `reset_*()` for testing
2. **Protocol-Based Storage**: `SessionStore`, `TraceStore`, `ProjectStore` protocols
3. **Event-Driven Observability**: Pub/sub via `EventEmitter`
4. **Graceful Degradation**: Optional deps fail gracefully

## Code Style

- **Formatter**: ruff (line-length: 100)
- **Type Hints**: Required for public APIs
- **Async**: All I/O operations must be async
- **Tests**: Maintain 80%+ coverage (target 86%)

## Testing

```bash
# Unit tests only (fast)
pytest -m "not slow and not integration"

# Integration tests (requires CLI)
pytest -m integration

# Specific module
pytest tests/test_harness.py -v
```

## Important Files

- `src/ccflow/__init__.py` - Public API exports
- `src/ccflow/types.py` - CLIAgentOptions, Message types
- `examples/` - 18 working examples
- `docs/api.md` - Complete API reference

## Security Rules

- **No API keys in code**: Auth delegated to Claude CLI OAuth
- **No hardcoded secrets**: Use environment variables
- **Parameterized queries**: SQLite uses `?` placeholders exclusively
- **CLI argument safety**: Args passed as list, never shell strings

## Common Tasks

### Add a new message type
1. Add dataclass to `types.py`
2. Update `Message` union type
3. Handle in `parser.py` `parse_event()`
4. Add tests in `test_parser.py`

### Add a new hook event
1. Add to `HookEvent` enum in `hooks.py`
2. Create `HookContext` fields if needed
3. Emit from appropriate location
4. Document in README.md

### Add a new store backend
1. Implement `SessionStore` protocol from `store.py`
2. Add to `stores/` directory
3. Export from `__init__.py`
4. Add integration tests

## Dependencies

**Core** (always installed):
- pydantic, pydantic-settings
- structlog, aiosqlite
- tiktoken, prometheus-client, uvicorn

**Optional** (extras):
- `[toon]`: toon-format
- `[server]`: fastapi, websockets
- `[api]`: anthropic (CLI fallback)
- `[tracing]`: opentelemetry

## Release Process

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Run full test suite: `pytest`
4. Build: `python -m build`
5. Upload: `twine upload dist/*`
