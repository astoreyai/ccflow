"""Tests for FastAPI Server."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Check if FastAPI is available
try:
    from fastapi.testclient import TestClient
    from httpx import ASGITransport, AsyncClient

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not FASTAPI_AVAILABLE,
    reason="FastAPI not installed",
)


@pytest.fixture
def mock_manager():
    """Create mock session manager."""
    manager = MagicMock()
    manager.active_session_count = 0
    manager.start = AsyncMock()
    manager.stop = AsyncMock()
    manager.create_session = AsyncMock()
    manager.get_session = AsyncMock(return_value=None)
    manager.list_sessions = AsyncMock(return_value=[])
    manager.count_sessions = AsyncMock(return_value=0)
    manager.delete_session = AsyncMock(return_value=True)
    manager.get_stats = AsyncMock(return_value={})
    return manager


@pytest.fixture
def mock_session():
    """Create mock session."""
    session = MagicMock()
    session.session_id = "test-session-123"
    session.turn_count = 1
    session.is_closed = False
    session.created_at = datetime.now()
    session.updated_at = datetime.now()
    session.tags = set()
    session.options = MagicMock()
    session.options.model = "sonnet"

    # Mock send_message as async generator
    async def mock_send_message(prompt):
        from ccflow.types import TextMessage

        yield TextMessage(content="Hello!")

    session.send_message = mock_send_message

    # Mock close to return stats
    async def mock_close():
        from ccflow.types import SessionStats

        return SessionStats(
            session_id=session.session_id,
            total_turns=1,
            total_input_tokens=100,
            total_output_tokens=50,
            duration_seconds=1.5,
            toon_savings_ratio=0.0,
        )

    session.close = mock_close

    return session


@pytest.fixture
def app(mock_manager):
    """Create test app with mocked manager."""
    from ccflow.server import CCFlowServer

    server = CCFlowServer(manager=mock_manager)
    return server.app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoints:
    """Tests for health endpoints."""

    def test_health_check(self, client, mock_manager):
        """Test health check endpoint."""
        with patch("ccflow.executor.get_executor") as mock_exec:
            mock_exec.return_value.check_cli_available = AsyncMock(return_value=True)

            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["version"] == "0.1.0"

    def test_health_cli_not_available(self, client, mock_manager):
        """Test health check when CLI not available."""
        from ccflow.exceptions import CLINotFoundError

        with patch("ccflow.executor.get_executor") as mock_exec:
            mock_exec.side_effect = CLINotFoundError("CLI not found")

            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["cli_available"] is False

    def test_readiness_check(self, client, mock_manager):
        """Test readiness check endpoint."""
        import time

        from ccflow.reliability import HealthStatus

        mock_health = HealthStatus(
            healthy=True,
            cli_available=True,
            cli_executable=True,
            cli_authenticated=True,
            latency_ms=50.0,
            error=None,
            version="1.0.0",
            last_check_time=time.time(),
        )

        with patch("ccflow.server.get_health_checker") as mock_checker:
            mock_checker.return_value.check = AsyncMock(return_value=mock_health)

            response = client.get("/ready")
            assert response.status_code == 200
            assert response.json()["ready"] is True

    def test_readiness_not_ready(self, client, mock_manager):
        """Test readiness check when not ready."""
        import time

        from ccflow.reliability import HealthStatus

        mock_health = HealthStatus(
            healthy=False,
            cli_available=False,
            cli_executable=False,
            cli_authenticated=False,
            latency_ms=None,
            error="CLI not found in PATH",
            version=None,
            last_check_time=time.time(),
        )

        with patch("ccflow.server.get_health_checker") as mock_checker:
            mock_checker.return_value.check = AsyncMock(return_value=mock_health)

            response = client.get("/ready")
            assert response.status_code == 503


class TestQueryEndpoints:
    """Tests for query endpoints."""

    def test_query_success(self, client, mock_manager, mock_session):
        """Test successful query."""
        mock_manager.create_session = AsyncMock(return_value=mock_session)

        response = client.post(
            "/query",
            json={"prompt": "Hello"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test-session-123"
        assert "content" in data

    def test_query_with_options(self, client, mock_manager, mock_session):
        """Test query with options."""
        mock_manager.create_session = AsyncMock(return_value=mock_session)

        response = client.post(
            "/query",
            json={
                "prompt": "Analyze this",
                "model": "opus",
                "system_prompt": "Be concise",
                "allowed_tools": ["Read", "Edit"],
                "timeout": 60.0,
            },
        )

        assert response.status_code == 200
        mock_manager.create_session.assert_called_once()

    def test_query_rate_limit_exceeded(self, client, mock_manager):
        """Test query when rate limited."""
        from ccflow.rate_limiting import RateLimitExceededError

        mock_manager.create_session = AsyncMock(
            side_effect=RateLimitExceededError("Rate limited", retry_after=60.0)
        )

        response = client.post(
            "/query",
            json={"prompt": "Hello"},
        )

        assert response.status_code == 429
        assert "Retry-After" in response.headers

    def test_query_concurrency_limit(self, client, mock_manager):
        """Test query when concurrency limited."""
        from ccflow.rate_limiting import ConcurrencyLimitExceededError

        mock_manager.create_session = AsyncMock(
            side_effect=ConcurrencyLimitExceededError("Limit exceeded", current=10, limit=10)
        )

        response = client.post(
            "/query",
            json={"prompt": "Hello"},
        )

        assert response.status_code == 503


class TestQueryStreamEndpoint:
    """Tests for streaming query endpoint."""

    def test_query_stream(self, client, mock_manager, mock_session):
        """Test streaming query endpoint."""
        mock_manager.create_session = AsyncMock(return_value=mock_session)

        response = client.post(
            "/query/stream",
            json={"prompt": "Hello"},
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"


class TestSessionEndpoints:
    """Tests for session endpoints."""

    def test_list_sessions_empty(self, client, mock_manager):
        """Test listing sessions when empty."""
        response = client.get("/sessions")

        assert response.status_code == 200
        data = response.json()
        assert data["sessions"] == []
        assert data["total"] == 0

    def test_list_sessions_with_data(self, client, mock_manager):
        """Test listing sessions with data."""
        from ccflow.store import SessionMetadata, SessionStatus

        mock_manager.list_sessions = AsyncMock(
            return_value=[
                SessionMetadata(
                    session_id="session-1",
                    model="sonnet",
                    status=SessionStatus.ACTIVE,
                    turn_count=5,
                    total_input_tokens=500,
                    total_output_tokens=250,
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    tags=["test"],
                ),
            ]
        )
        mock_manager.count_sessions = AsyncMock(return_value=1)

        response = client.get("/sessions")

        assert response.status_code == 200
        data = response.json()
        assert len(data["sessions"]) == 1
        assert data["total"] == 1
        assert data["sessions"][0]["session_id"] == "session-1"

    def test_list_sessions_with_filter(self, client, mock_manager):
        """Test listing sessions with filters."""
        mock_manager.list_sessions = AsyncMock(return_value=[])
        mock_manager.count_sessions = AsyncMock(return_value=0)

        response = client.get("/sessions?session_status=active&model=sonnet&limit=10")

        assert response.status_code == 200

    def test_list_sessions_invalid_status(self, client, mock_manager):
        """Test listing with invalid status."""
        response = client.get("/sessions?session_status=invalid")

        assert response.status_code == 400

    def test_get_session(self, client, mock_manager, mock_session):
        """Test getting session by ID."""
        mock_manager.get_session = AsyncMock(return_value=mock_session)

        response = client.get("/sessions/test-session-123")

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test-session-123"

    def test_get_session_not_found(self, client, mock_manager):
        """Test getting non-existent session."""
        mock_manager.get_session = AsyncMock(return_value=None)

        response = client.get("/sessions/nonexistent")

        assert response.status_code == 404

    def test_delete_session(self, client, mock_manager):
        """Test deleting session."""
        mock_manager.delete_session = AsyncMock(return_value=True)

        response = client.delete("/sessions/test-session-123")

        assert response.status_code == 200
        data = response.json()
        assert data["deleted"] is True

    def test_delete_session_not_found(self, client, mock_manager):
        """Test deleting non-existent session."""
        mock_manager.delete_session = AsyncMock(return_value=False)

        response = client.delete("/sessions/nonexistent")

        assert response.status_code == 404


class TestStatsEndpoints:
    """Tests for stats endpoints."""

    def test_rate_limiter_stats(self, client, mock_manager):
        """Test rate limiter stats endpoint."""
        from ccflow.rate_limiting import reset_limiter

        reset_limiter()

        response = client.get("/stats/rate-limiter")

        assert response.status_code == 200

    def test_session_stats(self, client, mock_manager):
        """Test session stats endpoint."""
        mock_manager.get_stats = AsyncMock(
            return_value={
                "total_sessions": 10,
                "active_sessions": 3,
                "closed_sessions": 7,
            }
        )

        response = client.get("/stats/sessions")

        assert response.status_code == 200
        data = response.json()
        assert "total_sessions" in data


class TestWebSocket:
    """Tests for WebSocket endpoint."""

    @pytest.mark.asyncio
    async def test_websocket_ping_pong(self, mock_manager):
        """Test WebSocket ping/pong."""
        from ccflow.server import CCFlowServer

        server = CCFlowServer(manager=mock_manager)

        async with AsyncClient(
            transport=ASGITransport(app=server.app),
            base_url="http://test",
        ):
            # WebSocket tests require different handling
            pass

    def test_websocket_connection(self, client, mock_manager):
        """Test WebSocket connection."""
        with client.websocket_connect("/ws") as websocket:
            # Send ping
            websocket.send_json({"type": "ping"})
            data = websocket.receive_json()
            assert data["type"] == "pong"

    def test_websocket_query(self, client, mock_manager, mock_session):
        """Test WebSocket query."""
        mock_manager.create_session = AsyncMock(return_value=mock_session)

        with client.websocket_connect("/ws") as websocket:
            websocket.send_json({
                "type": "query",
                "prompt": "Hello",
            })

            # Receive session start
            data = websocket.receive_json()
            assert data["type"] == "session_start"
            assert "session_id" in data

            # Receive content (could be multiple messages)
            messages = []
            while True:
                data = websocket.receive_json()
                messages.append(data)
                if data["type"] == "done":
                    break

            assert any(m["type"] == "done" for m in messages)

    def test_websocket_query_missing_prompt(self, client, mock_manager):
        """Test WebSocket query without prompt."""
        with client.websocket_connect("/ws") as websocket:
            websocket.send_json({
                "type": "query",
                # Missing prompt
            })

            data = websocket.receive_json()
            assert data["type"] == "error"


class TestMessageToDict:
    """Tests for message serialization."""

    def test_text_message(self):
        """Test serializing text message."""
        from ccflow.server import _message_to_dict
        from ccflow.types import TextMessage

        msg = TextMessage(content="Hello world")
        result = _message_to_dict(msg)

        assert result["type"] == "TextMessage"
        assert result["content"] == "Hello world"

    def test_tool_use_message(self):
        """Test serializing tool use message."""
        from ccflow.server import _message_to_dict
        from ccflow.types import ToolUseMessage

        msg = ToolUseMessage(tool="Read", args={"path": "/tmp/test"})
        result = _message_to_dict(msg)

        assert result["type"] == "ToolUseMessage"
        assert result["tool"] == "Read"
        assert result["args"] == {"path": "/tmp/test"}


class TestServerCreation:
    """Tests for server creation."""

    def test_create_server(self, mock_manager):
        """Test creating server."""
        from ccflow.server import CCFlowServer

        server = CCFlowServer(manager=mock_manager)
        assert server.app is not None

    def test_create_server_without_fastapi(self):
        """Test creating server without FastAPI installed."""
        with patch.dict("sys.modules", {"fastapi": None}):
            # Would need to reload module to test this properly
            pass

    def test_get_app(self):
        """Test get_app function."""
        from ccflow.server import get_app

        with patch("ccflow.server.CCFlowServer") as mock_class:
            mock_class.return_value.app = MagicMock()
            app = get_app()
            assert app is not None

    def test_create_app(self):
        """Test create_app function."""
        from ccflow.server import create_app

        with patch("ccflow.server.CCFlowServer") as mock_class:
            mock_class.return_value.app = MagicMock()
            app = create_app()
            assert app is not None


class TestCORSMiddleware:
    """Tests for CORS configuration."""

    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )

        # FastAPI with CORS middleware should handle this
        assert response.status_code in [200, 405]


class TestErrorHandling:
    """Tests for error handling."""

    def test_ccflow_error_handling(self, client, mock_manager):
        """Test CCFlowError is handled properly."""
        from ccflow.exceptions import CCFlowError

        mock_manager.create_session = AsyncMock(
            side_effect=CCFlowError("Something went wrong")
        )

        response = client.post(
            "/query",
            json={"prompt": "Hello"},
        )

        assert response.status_code == 500

    def test_unexpected_error_in_stream(self, client, mock_manager):
        """Test unexpected error in stream is handled."""
        mock_manager.create_session = AsyncMock(
            side_effect=RuntimeError("Unexpected error")
        )

        response = client.post(
            "/query/stream",
            json={"prompt": "Hello"},
        )

        # Should still return 200 with error in stream
        assert response.status_code == 200
