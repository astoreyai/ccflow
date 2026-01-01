"""
Pytest configuration and fixtures for ccflow tests.

Provides mocked CLI responses, async fixtures, and test utilities.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccflow.types import CLIAgentOptions, ToonConfig

# Fixture directory
FIXTURES_DIR = Path(__file__).parent / "fixtures"
CLI_RESPONSES_DIR = FIXTURES_DIR / "cli_responses"


@pytest.fixture
def mock_cli_path() -> str:
    """Return mock CLI path."""
    return "/usr/bin/claude"


@pytest.fixture
def default_options() -> CLIAgentOptions:
    """Return default CLI agent options for tests."""
    return CLIAgentOptions(
        model="sonnet",
        max_budget_usd=5.0,
        timeout=30.0,
        toon=ToonConfig(enabled=True, track_savings=True),
    )


@pytest.fixture
def simple_ndjson_response() -> list[dict]:
    """Return simple NDJSON response events."""
    return [
        {"type": "system", "subtype": "init", "session_id": "test-session-123"},
        {"type": "message", "content": "Hello, ", "delta_type": "text_delta"},
        {"type": "message", "content": "world!", "delta_type": "text_delta"},
        {
            "type": "system",
            "subtype": "stop",
            "session_id": "test-session-123",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        },
    ]


@pytest.fixture
def tool_use_ndjson_response() -> list[dict]:
    """Return NDJSON response with tool use."""
    return [
        {"type": "system", "subtype": "init", "session_id": "test-session-456"},
        {"type": "message", "content": "Let me read that file.", "delta_type": "text_delta"},
        {"type": "tool_use", "tool": "Read", "args": {"file_path": "/path/to/file.py"}},
        {"type": "tool_result", "tool": "Read", "content": "def hello():\n    pass"},
        {"type": "message", "content": "The file contains a simple function.", "delta_type": "text_delta"},
        {
            "type": "system",
            "subtype": "stop",
            "session_id": "test-session-456",
            "usage": {"input_tokens": 50, "output_tokens": 30},
        },
    ]


@pytest.fixture
def error_ndjson_response() -> list[dict]:
    """Return NDJSON response with error."""
    return [
        {"type": "system", "subtype": "init", "session_id": "test-session-789"},
        {"type": "error", "message": "Rate limit exceeded", "code": "rate_limit"},
    ]


def ndjson_bytes(events: list[dict]) -> bytes:
    """Convert events to NDJSON bytes."""
    lines = [json.dumps(event) for event in events]
    return ("\n".join(lines) + "\n").encode("utf-8")


@pytest.fixture
def mock_subprocess(simple_ndjson_response: list[dict]):
    """Create mock subprocess for CLI execution."""

    async def mock_create_subprocess_exec(*args, **kwargs):
        process = AsyncMock()
        process.returncode = 0
        process.pid = 12345

        # Create async iterator for stdout
        output = ndjson_bytes(simple_ndjson_response)

        async def stdout_iterator():
            for line in output.decode().strip().split("\n"):
                yield (line + "\n").encode()

        process.stdout = MagicMock()
        process.stdout.__aiter__ = lambda self: stdout_iterator()
        process.stderr = AsyncMock()
        process.stderr.read = AsyncMock(return_value=b"")
        process.wait = AsyncMock(return_value=0)

        return process

    with patch("asyncio.create_subprocess_exec", side_effect=mock_create_subprocess_exec):
        yield


@pytest.fixture
def mock_subprocess_tool_use(tool_use_ndjson_response: list[dict]):
    """Create mock subprocess with tool use response."""

    async def mock_create_subprocess_exec(*args, **kwargs):
        process = AsyncMock()
        process.returncode = 0

        output = ndjson_bytes(tool_use_ndjson_response)

        async def stdout_iterator():
            for line in output.decode().strip().split("\n"):
                yield (line + "\n").encode()

        process.stdout = MagicMock()
        process.stdout.__aiter__ = lambda self: stdout_iterator()
        process.stderr = AsyncMock()
        process.stderr.read = AsyncMock(return_value=b"")
        process.wait = AsyncMock(return_value=0)

        return process

    with patch("asyncio.create_subprocess_exec", side_effect=mock_create_subprocess_exec):
        yield


@pytest.fixture
def mock_subprocess_error(error_ndjson_response: list[dict]):
    """Create mock subprocess with error response."""

    async def mock_create_subprocess_exec(*args, **kwargs):
        process = AsyncMock()
        process.returncode = 1

        output = ndjson_bytes(error_ndjson_response)

        async def stdout_iterator():
            for line in output.decode().strip().split("\n"):
                yield (line + "\n").encode()

        process.stdout = MagicMock()
        process.stdout.__aiter__ = lambda self: stdout_iterator()
        process.stderr = AsyncMock()
        process.stderr.read = AsyncMock(return_value=b"CLI error occurred")
        process.wait = AsyncMock(return_value=1)

        return process

    with patch("asyncio.create_subprocess_exec", side_effect=mock_create_subprocess_exec):
        yield


@pytest.fixture
def sample_portfolio_data() -> dict:
    """Sample portfolio data for TOON testing."""
    return {
        "account": "U1234567",
        "positions": [
            {"symbol": "AAPL", "qty": 100, "avgCost": 150.25, "pnl": 1500.00},
            {"symbol": "GOOGL", "qty": 50, "avgCost": 2800.00, "pnl": -200.00},
            {"symbol": "MSFT", "qty": 75, "avgCost": 380.50, "pnl": 850.00},
        ],
        "cash": 50000.00,
        "margin_used": 0.35,
    }


@pytest.fixture
def sample_orders_data() -> list[dict]:
    """Sample orders data (uniform array - ideal for TOON)."""
    return [
        {"orderId": "ORD001", "symbol": "AAPL", "side": "BUY", "qty": 10, "status": "FILLED"},
        {"orderId": "ORD002", "symbol": "GOOGL", "side": "SELL", "qty": 5, "status": "FILLED"},
        {"orderId": "ORD003", "symbol": "MSFT", "side": "BUY", "qty": 20, "status": "PENDING"},
        {"orderId": "ORD004", "symbol": "TSLA", "side": "BUY", "qty": 15, "status": "CANCELLED"},
    ]


# Create fixtures directory if it doesn't exist
@pytest.fixture(scope="session", autouse=True)
def ensure_fixtures_dir():
    """Ensure fixtures directory exists."""
    CLI_RESPONSES_DIR.mkdir(parents=True, exist_ok=True)
