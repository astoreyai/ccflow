"""
API Client - Direct Anthropic SDK integration for API fallback.

Provides a unified interface matching CLI executor but using direct API calls.
Used as fallback when CLI is unavailable (circuit breaker open, health check fails).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

from ccflow.exceptions import CCFlowError
from ccflow.reliability import bind_correlation_id, get_correlation_id, set_correlation_id

if TYPE_CHECKING:
    from ccflow.types import CLIAgentOptions

logger = structlog.get_logger(__name__)

# Check if anthropic SDK is available
try:
    import anthropic
    from anthropic import AsyncAnthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    anthropic = None  # type: ignore
    AsyncAnthropic = None  # type: ignore


class APIClientError(CCFlowError):
    """Error from API client."""

    pass


class APINotAvailableError(APIClientError):
    """API client not available (SDK not installed or no API key)."""

    pass


class APIRateLimitError(APIClientError):
    """API rate limit exceeded."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


@dataclass
class APIClientConfig:
    """Configuration for API client."""

    # API settings
    api_key: str | None = None  # Uses ANTHROPIC_API_KEY env var if None
    base_url: str | None = None
    timeout: float = 300.0
    max_retries: int = 2

    # Model defaults
    default_model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 4096

    # Streaming
    stream: bool = True


@dataclass
class APIResponse:
    """Response from API call."""

    content: str
    model: str
    input_tokens: int
    output_tokens: int
    stop_reason: str | None = None
    id: str | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class APIClient:
    """Direct Anthropic API client.

    Provides API access as fallback when CLI is unavailable.
    Matches the streaming interface of CLIExecutor for compatibility.

    Example:
        >>> client = APIClient()
        >>> async for event in client.execute("Hello", options):
        ...     print(event)

    With explicit config:
        >>> config = APIClientConfig(api_key="sk-...", max_tokens=8192)
        >>> client = APIClient(config)
    """

    def __init__(self, config: APIClientConfig | None = None) -> None:
        """Initialize API client.

        Args:
            config: API client configuration
        """
        self.config = config or APIClientConfig()
        self._client: AsyncAnthropic | None = None
        self._initialized = False

    @property
    def is_available(self) -> bool:
        """Check if API client is available."""
        if not ANTHROPIC_AVAILABLE:
            return False

        # Check for API key
        import os

        if self.config.api_key:
            return True
        return bool(os.environ.get("ANTHROPIC_API_KEY"))

    def _get_client(self) -> AsyncAnthropic:
        """Get or create the async client."""
        if not ANTHROPIC_AVAILABLE:
            raise APINotAvailableError(
                "anthropic SDK not installed. Install with: pip install anthropic"
            )

        if self._client is None:
            self._client = AsyncAnthropic(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
            )
        return self._client

    def _map_model(self, model: str | None) -> str:
        """Map model alias to full model name."""
        if model is None:
            return self.config.default_model

        model_lower = model.lower()

        # Map simple names to latest models
        model_map = {
            "haiku": "claude-haiku-4-20250514",
            "sonnet": "claude-sonnet-4-20250514",
            "opus": "claude-opus-4-20250514",
        }

        if model_lower in model_map:
            return model_map[model_lower]

        # Return as-is if it looks like a full model name
        if "claude" in model_lower:
            return model

        # Default to sonnet
        return self.config.default_model

    def _build_messages(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Build messages array for API call.

        Returns:
            Tuple of (messages, system_prompt)
        """
        messages = [{"role": "user", "content": prompt}]
        return messages, system_prompt

    async def execute(
        self,
        prompt: str,
        options: CLIAgentOptions | None = None,
        *,
        correlation_id: str | None = None,
    ) -> AsyncIterator[dict]:
        """Execute API call and stream responses.

        Matches CLIExecutor.execute() interface for compatibility.

        Args:
            prompt: The prompt to send
            options: CLI agent options (subset supported)
            correlation_id: Request correlation ID

        Yields:
            Event dictionaries matching CLI output format
        """
        from ccflow.types import CLIAgentOptions

        options = options or CLIAgentOptions()
        cid = set_correlation_id(correlation_id)
        log = bind_correlation_id()

        client = self._get_client()
        model = self._map_model(options.model)
        messages, system = self._build_messages(prompt, options.system_prompt)

        log.debug(
            "api_execute_start",
            model=model,
            max_tokens=options.max_tokens or self.config.max_tokens,
        )

        start_time = time.time()

        # Emit init event
        yield {
            "type": "init",
            "session_id": f"api-{cid}",
            "model": model,
        }

        try:
            if self.config.stream:
                async for event in self._stream_response(
                    client, model, messages, system, options, log
                ):
                    yield event
            else:
                async for event in self._sync_response(
                    client, model, messages, system, options, log
                ):
                    yield event

        except anthropic.RateLimitError as e:
            log.warning("api_rate_limit", error=str(e))
            raise APIRateLimitError(
                str(e),
                retry_after=getattr(e, "retry_after", None),
            ) from e

        except anthropic.APIError as e:
            log.error("api_error", error=str(e))
            raise APIClientError(str(e)) from e

        finally:
            elapsed = time.time() - start_time
            log.debug("api_execute_complete", elapsed=f"{elapsed:.2f}s")

    async def _stream_response(
        self,
        client: AsyncAnthropic,
        model: str,
        messages: list[dict],
        system: str | None,
        options: CLIAgentOptions,
        log: structlog.BoundLogger,
    ) -> AsyncIterator[dict]:
        """Stream response from API."""
        from ccflow.types import CLIAgentOptions

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": options.max_tokens or self.config.max_tokens,
        }

        if system:
            kwargs["system"] = system

        input_tokens = 0
        output_tokens = 0
        content_parts: list[str] = []

        async with client.messages.stream(**kwargs) as stream:
            async for event in stream:
                if event.type == "message_start":
                    if hasattr(event, "message") and hasattr(event.message, "usage"):
                        input_tokens = event.message.usage.input_tokens

                elif event.type == "content_block_delta":
                    if hasattr(event, "delta") and hasattr(event.delta, "text"):
                        text = event.delta.text
                        content_parts.append(text)
                        yield {
                            "type": "text",
                            "content": text,
                        }

                elif event.type == "message_delta":
                    if hasattr(event, "usage"):
                        output_tokens = event.usage.output_tokens

        # Emit stop event
        yield {
            "type": "stop",
            "session_id": f"api-{get_correlation_id()}",
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
        }

        # Emit result event
        yield {
            "type": "result",
            "content": "".join(content_parts),
            "model": model,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
        }

    async def _sync_response(
        self,
        client: AsyncAnthropic,
        model: str,
        messages: list[dict],
        system: str | None,
        options: CLIAgentOptions,
        log: structlog.BoundLogger,
    ) -> AsyncIterator[dict]:
        """Get non-streaming response from API."""
        from ccflow.types import CLIAgentOptions

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": options.max_tokens or self.config.max_tokens,
        }

        if system:
            kwargs["system"] = system

        response = await client.messages.create(**kwargs)

        content = ""
        if response.content:
            for block in response.content:
                if hasattr(block, "text"):
                    content += block.text

        # Emit text
        yield {
            "type": "text",
            "content": content,
        }

        # Emit stop
        yield {
            "type": "stop",
            "session_id": f"api-{get_correlation_id()}",
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        }

        # Emit result
        yield {
            "type": "result",
            "content": content,
            "model": model,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        }

    async def check_available(self) -> bool:
        """Check if API is available and authenticated.

        Returns:
            True if API is ready for use
        """
        if not self.is_available:
            return False

        try:
            client = self._get_client()
            # Make a minimal API call to verify authentication
            await client.messages.create(
                model="claude-haiku-4-20250514",
                max_tokens=1,
                messages=[{"role": "user", "content": "hi"}],
            )
            return True
        except Exception as e:
            logger.debug("api_check_failed", error=str(e))
            return False

    async def close(self) -> None:
        """Close the API client."""
        if self._client is not None:
            await self._client.close()
            self._client = None


# =============================================================================
# Fallback Executor
# =============================================================================


@dataclass
class FallbackConfig:
    """Configuration for fallback behavior."""

    # When to use API fallback
    fallback_on_cli_unavailable: bool = True
    fallback_on_circuit_open: bool = True
    fallback_on_timeout: bool = False

    # API client config
    api_config: APIClientConfig = field(default_factory=APIClientConfig)


class FallbackExecutor:
    """Executor with automatic API fallback.

    Tries CLI first, falls back to API when:
    - CLI is not available (not in PATH)
    - Circuit breaker is open
    - CLI times out (optional)

    Example:
        >>> executor = FallbackExecutor()
        >>> async for event in executor.execute("Hello", options):
        ...     print(event)  # Uses CLI or API automatically
    """

    def __init__(
        self,
        config: FallbackConfig | None = None,
        cli_executor: Any | None = None,
        api_client: APIClient | None = None,
    ) -> None:
        """Initialize fallback executor.

        Args:
            config: Fallback configuration
            cli_executor: Custom CLI executor (uses default if None)
            api_client: Custom API client (creates one if None)
        """
        self.config = config or FallbackConfig()
        self._cli_executor = cli_executor
        self._api_client = api_client

    @property
    def cli_executor(self) -> Any:
        """Get or create CLI executor."""
        if self._cli_executor is None:
            from ccflow.executor import get_executor

            self._cli_executor = get_executor()
        return self._cli_executor

    @property
    def api_client(self) -> APIClient:
        """Get or create API client."""
        if self._api_client is None:
            self._api_client = APIClient(self.config.api_config)
        return self._api_client

    async def _should_use_api(self) -> tuple[bool, str]:
        """Determine if API should be used.

        Returns:
            Tuple of (should_use_api, reason)
        """
        from ccflow.reliability import get_cli_circuit_breaker, get_health_checker

        # Check circuit breaker
        if self.config.fallback_on_circuit_open:
            breaker = get_cli_circuit_breaker()
            from ccflow.reliability import CircuitState

            if breaker.state == CircuitState.OPEN:
                return True, "circuit_breaker_open"

        # Check CLI health
        if self.config.fallback_on_cli_unavailable:
            checker = get_health_checker()
            status = await checker.check()
            if not status.healthy:
                return True, f"cli_unhealthy: {status.error}"

        return False, ""

    async def execute(
        self,
        prompt: str,
        options: CLIAgentOptions | None = None,
        timeout: float = 300.0,
        cwd: Any = None,
        *,
        correlation_id: str | None = None,
        force_api: bool = False,
        force_cli: bool = False,
    ) -> AsyncIterator[dict]:
        """Execute with automatic fallback.

        Args:
            prompt: The prompt to send
            options: CLI agent options
            timeout: Execution timeout
            cwd: Working directory (CLI only)
            correlation_id: Request correlation ID
            force_api: Force API usage
            force_cli: Force CLI usage (no fallback)

        Yields:
            Event dictionaries
        """
        from ccflow.types import CLIAgentOptions

        options = options or CLIAgentOptions()
        cid = set_correlation_id(correlation_id)
        log = bind_correlation_id()

        # Determine execution path
        use_api = force_api
        reason = "forced" if force_api else ""

        if not force_api and not force_cli:
            use_api, reason = await self._should_use_api()

        if use_api:
            if not self.api_client.is_available:
                raise APINotAvailableError(
                    "API fallback requested but API client not available"
                )

            log.info("using_api_fallback", reason=reason)

            async for event in self.api_client.execute(
                prompt, options, correlation_id=cid
            ):
                # Add fallback indicator
                event["_fallback"] = True
                event["_fallback_reason"] = reason
                yield event

        else:
            # Use CLI
            log.debug("using_cli")
            flags = self.cli_executor.build_flags(options)

            try:
                async for event in self.cli_executor.execute(
                    prompt=prompt,
                    flags=flags,
                    timeout=timeout,
                    cwd=cwd,
                    correlation_id=cid,
                ):
                    yield event

            except Exception as e:
                # Check if we should fallback on this error
                from ccflow.exceptions import CLITimeoutError
                from ccflow.reliability import CircuitBreakerError

                should_fallback = False

                if isinstance(e, CircuitBreakerError):
                    should_fallback = self.config.fallback_on_circuit_open
                    reason = "circuit_breaker_error"
                elif isinstance(e, CLITimeoutError):
                    should_fallback = self.config.fallback_on_timeout
                    reason = "cli_timeout"

                if should_fallback and self.api_client.is_available:
                    log.info("cli_failed_using_api_fallback", reason=reason, error=str(e))

                    async for event in self.api_client.execute(
                        prompt, options, correlation_id=cid
                    ):
                        event["_fallback"] = True
                        event["_fallback_reason"] = reason
                        yield event
                else:
                    raise

    async def close(self) -> None:
        """Close resources."""
        if self._api_client is not None:
            await self._api_client.close()


# =============================================================================
# Module-level management
# =============================================================================

_default_api_client: APIClient | None = None
_default_fallback_executor: FallbackExecutor | None = None


def get_api_client(config: APIClientConfig | None = None) -> APIClient:
    """Get or create the default API client."""
    global _default_api_client
    if _default_api_client is None:
        _default_api_client = APIClient(config)
    return _default_api_client


def get_fallback_executor(config: FallbackConfig | None = None) -> FallbackExecutor:
    """Get or create the default fallback executor."""
    global _default_fallback_executor
    if _default_fallback_executor is None:
        _default_fallback_executor = FallbackExecutor(config)
    return _default_fallback_executor


async def reset_api_client() -> None:
    """Reset the global API client."""
    global _default_api_client
    if _default_api_client is not None:
        await _default_api_client.close()
        _default_api_client = None


async def reset_fallback_executor() -> None:
    """Reset the global fallback executor."""
    global _default_fallback_executor
    if _default_fallback_executor is not None:
        await _default_fallback_executor.close()
        _default_fallback_executor = None
