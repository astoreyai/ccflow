"""Tests for API client module - direct Anthropic SDK integration and fallback."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccflow.api_client import (
    APIClient,
    APIClientConfig,
    APIClientError,
    APINotAvailableError,
    APIRateLimitError,
    APIResponse,
    FallbackConfig,
    FallbackExecutor,
    get_api_client,
    get_fallback_executor,
    reset_api_client,
    reset_fallback_executor,
)
from ccflow.types import CLIAgentOptions


# =============================================================================
# APIClientConfig Tests
# =============================================================================


class TestAPIClientConfig:
    """Tests for API client configuration."""

    def test_default_config(self):
        """Default configuration should have sensible values."""
        config = APIClientConfig()
        assert config.timeout == 300.0
        assert config.max_retries == 2
        assert config.max_tokens == 4096
        assert config.stream is True

    def test_custom_config(self):
        """Should accept custom configuration."""
        config = APIClientConfig(
            api_key="test-key",
            timeout=60.0,
            max_tokens=8192,
            stream=False,
        )
        assert config.api_key == "test-key"
        assert config.timeout == 60.0
        assert config.max_tokens == 8192
        assert config.stream is False


# =============================================================================
# APIResponse Tests
# =============================================================================


class TestAPIResponse:
    """Tests for API response dataclass."""

    def test_total_tokens(self):
        """Should calculate total tokens correctly."""
        response = APIResponse(
            content="Hello",
            model="sonnet",
            input_tokens=100,
            output_tokens=50,
        )
        assert response.total_tokens == 150


# =============================================================================
# APIClient Tests
# =============================================================================


class TestAPIClient:
    """Tests for the API client."""

    def test_is_available_no_sdk(self):
        """Should return False if SDK not available."""
        with patch("ccflow.api_client.ANTHROPIC_AVAILABLE", False):
            client = APIClient()
            assert not client.is_available

    def test_is_available_no_api_key(self):
        """Should return False if no API key."""
        with patch("ccflow.api_client.ANTHROPIC_AVAILABLE", True):
            with patch.dict("os.environ", {}, clear=True):
                config = APIClientConfig(api_key=None)
                client = APIClient(config)
                # Need to also clear ANTHROPIC_API_KEY if it exists
                import os

                original = os.environ.pop("ANTHROPIC_API_KEY", None)
                try:
                    assert not client.is_available
                finally:
                    if original:
                        os.environ["ANTHROPIC_API_KEY"] = original

    def test_is_available_with_api_key(self):
        """Should return True if API key is set."""
        with patch("ccflow.api_client.ANTHROPIC_AVAILABLE", True):
            config = APIClientConfig(api_key="test-key")
            client = APIClient(config)
            assert client.is_available

    def test_map_model_aliases(self):
        """Should map model aliases to full names."""
        client = APIClient()
        assert "haiku" in client._map_model("haiku")
        assert "sonnet" in client._map_model("sonnet")
        assert "opus" in client._map_model("opus")

    def test_map_model_full_name(self):
        """Should return full model names unchanged."""
        client = APIClient()
        model = "claude-3-5-sonnet-20241022"
        assert client._map_model(model) == model

    def test_map_model_none_uses_default(self):
        """Should use default model when None passed."""
        config = APIClientConfig(default_model="test-model")
        client = APIClient(config)
        assert client._map_model(None) == "test-model"

    @pytest.mark.asyncio
    async def test_execute_not_available(self):
        """Should raise error if SDK not available."""
        with patch("ccflow.api_client.ANTHROPIC_AVAILABLE", False):
            client = APIClient()
            with pytest.raises(APINotAvailableError):
                async for _ in client.execute("Hello"):
                    pass

    @pytest.mark.asyncio
    async def test_execute_streaming(self):
        """Should stream responses correctly."""
        mock_client = MagicMock()

        # Create mock stream context manager
        mock_stream = AsyncMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream.__aexit__ = AsyncMock(return_value=None)

        # Create mock events
        mock_message_start = MagicMock()
        mock_message_start.type = "message_start"
        mock_message_start.message = MagicMock()
        mock_message_start.message.usage = MagicMock()
        mock_message_start.message.usage.input_tokens = 100

        mock_content_delta = MagicMock()
        mock_content_delta.type = "content_block_delta"
        mock_content_delta.delta = MagicMock()
        mock_content_delta.delta.text = "Hello"

        mock_message_delta = MagicMock()
        mock_message_delta.type = "message_delta"
        mock_message_delta.usage = MagicMock()
        mock_message_delta.usage.output_tokens = 50

        async def mock_aiter():
            yield mock_message_start
            yield mock_content_delta
            yield mock_message_delta

        mock_stream.__aiter__ = lambda self: mock_aiter()
        mock_client.messages.stream = MagicMock(return_value=mock_stream)

        with patch("ccflow.api_client.ANTHROPIC_AVAILABLE", True):
            with patch("ccflow.api_client.AsyncAnthropic", return_value=mock_client):
                config = APIClientConfig(api_key="test-key")
                client = APIClient(config)

                events = []
                async for event in client.execute("Test prompt"):
                    events.append(event)

                # Should have init, text, stop, result events
                event_types = [e["type"] for e in events]
                assert "init" in event_types
                assert "text" in event_types
                assert "stop" in event_types
                assert "result" in event_types

    @pytest.mark.asyncio
    async def test_execute_non_streaming(self):
        """Should handle non-streaming responses."""
        mock_client = MagicMock()

        # Create mock response
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Hello World")]
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50

        mock_client.messages.create = AsyncMock(return_value=mock_response)

        with patch("ccflow.api_client.ANTHROPIC_AVAILABLE", True):
            with patch("ccflow.api_client.AsyncAnthropic", return_value=mock_client):
                config = APIClientConfig(api_key="test-key", stream=False)
                client = APIClient(config)

                events = []
                async for event in client.execute("Test prompt"):
                    events.append(event)

                # Should have init, text, stop, result events
                assert len(events) >= 3
                assert events[-1]["type"] == "result"
                assert events[-1]["content"] == "Hello World"

    @pytest.mark.asyncio
    async def test_close(self):
        """Should close client properly."""
        mock_client = MagicMock()
        mock_client.close = AsyncMock()

        with patch("ccflow.api_client.ANTHROPIC_AVAILABLE", True):
            with patch("ccflow.api_client.AsyncAnthropic", return_value=mock_client):
                config = APIClientConfig(api_key="test-key")
                client = APIClient(config)
                client._client = mock_client

                await client.close()

                mock_client.close.assert_called_once()
                assert client._client is None


# =============================================================================
# FallbackConfig Tests
# =============================================================================


class TestFallbackConfig:
    """Tests for fallback configuration."""

    def test_default_config(self):
        """Default fallback config should enable CLI fallbacks."""
        config = FallbackConfig()
        assert config.fallback_on_cli_unavailable is True
        assert config.fallback_on_circuit_open is True
        assert config.fallback_on_timeout is False


# =============================================================================
# FallbackExecutor Tests
# =============================================================================


class TestFallbackExecutor:
    """Tests for the fallback executor."""

    @pytest.fixture
    def mock_cli_executor(self):
        """Create mock CLI executor."""
        executor = MagicMock()
        executor.build_flags = MagicMock(return_value=["--flag"])

        async def mock_execute(*args, **kwargs):
            yield {"type": "init", "session_id": "cli-123"}
            yield {"type": "text", "content": "Hello from CLI"}
            yield {"type": "stop", "usage": {"input_tokens": 100, "output_tokens": 50}}

        executor.execute = mock_execute
        return executor

    @pytest.fixture
    def mock_api_client(self):
        """Create mock API client."""
        client = MagicMock()
        client.is_available = True

        async def mock_execute(*args, **kwargs):
            yield {"type": "init", "session_id": "api-123"}
            yield {"type": "text", "content": "Hello from API"}
            yield {"type": "stop", "usage": {"input_tokens": 100, "output_tokens": 50}}

        client.execute = mock_execute
        return client

    @pytest.mark.asyncio
    async def test_uses_cli_when_healthy(self, mock_cli_executor, mock_api_client):
        """Should use CLI when healthy."""
        with patch("ccflow.reliability.get_health_checker") as mock_checker:
            mock_status = MagicMock()
            mock_status.healthy = True
            mock_checker.return_value.check = AsyncMock(return_value=mock_status)

            with patch("ccflow.reliability.get_cli_circuit_breaker") as mock_breaker:
                from ccflow.reliability import CircuitState

                mock_breaker.return_value.state = CircuitState.CLOSED

                executor = FallbackExecutor(
                    cli_executor=mock_cli_executor,
                    api_client=mock_api_client,
                )

                events = []
                async for event in executor.execute("Test"):
                    events.append(event)

                # Should NOT have fallback indicator
                assert not any(e.get("_fallback") for e in events)
                assert any("CLI" in e.get("content", "") for e in events)

    @pytest.mark.asyncio
    async def test_uses_api_when_circuit_open(self, mock_cli_executor, mock_api_client):
        """Should use API when circuit breaker is open."""
        with patch("ccflow.reliability.get_health_checker") as mock_checker:
            mock_status = MagicMock()
            mock_status.healthy = True
            mock_checker.return_value.check = AsyncMock(return_value=mock_status)

            with patch("ccflow.reliability.get_cli_circuit_breaker") as mock_breaker:
                from ccflow.reliability import CircuitState

                mock_breaker.return_value.state = CircuitState.OPEN

                executor = FallbackExecutor(
                    cli_executor=mock_cli_executor,
                    api_client=mock_api_client,
                )

                events = []
                async for event in executor.execute("Test"):
                    events.append(event)

                # Should have fallback indicator
                assert any(e.get("_fallback") for e in events)
                assert any("API" in e.get("content", "") for e in events)

    @pytest.mark.asyncio
    async def test_uses_api_when_cli_unhealthy(self, mock_cli_executor, mock_api_client):
        """Should use API when CLI is unhealthy."""
        with patch("ccflow.reliability.get_health_checker") as mock_checker:
            mock_status = MagicMock()
            mock_status.healthy = False
            mock_status.error = "CLI not found"
            mock_checker.return_value.check = AsyncMock(return_value=mock_status)

            with patch("ccflow.reliability.get_cli_circuit_breaker") as mock_breaker:
                from ccflow.reliability import CircuitState

                mock_breaker.return_value.state = CircuitState.CLOSED

                executor = FallbackExecutor(
                    cli_executor=mock_cli_executor,
                    api_client=mock_api_client,
                )

                events = []
                async for event in executor.execute("Test"):
                    events.append(event)

                # Should have fallback indicator
                assert any(e.get("_fallback") for e in events)

    @pytest.mark.asyncio
    async def test_force_api(self, mock_cli_executor, mock_api_client):
        """Should use API when force_api=True."""
        executor = FallbackExecutor(
            cli_executor=mock_cli_executor,
            api_client=mock_api_client,
        )

        events = []
        async for event in executor.execute("Test", force_api=True):
            events.append(event)

        assert any(e.get("_fallback") for e in events)
        assert any("API" in e.get("content", "") for e in events)

    @pytest.mark.asyncio
    async def test_force_cli_no_fallback(self, mock_cli_executor, mock_api_client):
        """Should not fallback when force_cli=True."""
        with patch("ccflow.reliability.get_health_checker") as mock_checker:
            mock_status = MagicMock()
            mock_status.healthy = True
            mock_checker.return_value.check = AsyncMock(return_value=mock_status)

            with patch("ccflow.reliability.get_cli_circuit_breaker") as mock_breaker:
                from ccflow.reliability import CircuitState

                mock_breaker.return_value.state = CircuitState.CLOSED

                executor = FallbackExecutor(
                    cli_executor=mock_cli_executor,
                    api_client=mock_api_client,
                )

                events = []
                async for event in executor.execute("Test", force_cli=True):
                    events.append(event)

                assert not any(e.get("_fallback") for e in events)

    @pytest.mark.asyncio
    async def test_force_api_not_available(self, mock_cli_executor):
        """Should raise error when force_api but API not available."""
        mock_api = MagicMock()
        mock_api.is_available = False

        executor = FallbackExecutor(
            cli_executor=mock_cli_executor,
            api_client=mock_api,
        )

        with pytest.raises(APINotAvailableError):
            async for _ in executor.execute("Test", force_api=True):
                pass


# =============================================================================
# Global Management Tests
# =============================================================================


class TestGlobalManagement:
    """Tests for global API client and fallback executor management."""

    @pytest.mark.asyncio
    async def test_get_api_client_singleton(self):
        """get_api_client should return singleton."""
        await reset_api_client()

        client1 = get_api_client()
        client2 = get_api_client()
        assert client1 is client2

        await reset_api_client()

    @pytest.mark.asyncio
    async def test_get_fallback_executor_singleton(self):
        """get_fallback_executor should return singleton."""
        await reset_fallback_executor()

        executor1 = get_fallback_executor()
        executor2 = get_fallback_executor()
        assert executor1 is executor2

        await reset_fallback_executor()

    @pytest.mark.asyncio
    async def test_reset_api_client(self):
        """reset_api_client should clear singleton."""
        client1 = get_api_client()
        await reset_api_client()
        client2 = get_api_client()

        assert client1 is not client2

        await reset_api_client()

    @pytest.mark.asyncio
    async def test_reset_fallback_executor(self):
        """reset_fallback_executor should clear singleton."""
        executor1 = get_fallback_executor()
        await reset_fallback_executor()
        executor2 = get_fallback_executor()

        assert executor1 is not executor2

        await reset_fallback_executor()
