"""
FastAPI Server - HTTP/WebSocket interface for ccflow.

Provides REST endpoints and WebSocket streaming for remote access
to ccflow functionality.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager, suppress
from datetime import datetime
from typing import TYPE_CHECKING, Any

import structlog

from ccflow.config import get_settings
from ccflow.exceptions import (
    CCFlowError,
    CLINotFoundError,
)
from ccflow.manager import SessionManager
from ccflow.rate_limiting import (
    ConcurrencyLimitExceededError,
    RateLimitExceededError,
    get_limiter,
)
from ccflow.reliability import (
    CircuitBreakerError,
    get_correlation_id,
    get_health_checker,
    set_correlation_id,
)
from ccflow.types import CLIAgentOptions, Message

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from starlette.requests import Request
    from starlette.responses import Response

logger = structlog.get_logger(__name__)

# Try to import FastAPI
try:
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel, Field

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

    # Stub classes for type checking when FastAPI not installed
    class BaseModel:  # type: ignore
        pass


# Request/Response Models (only defined if FastAPI available)
if FASTAPI_AVAILABLE:

    class QueryRequest(BaseModel):
        """Request body for query endpoint."""

        prompt: str = Field(..., description="The prompt to send to Claude")
        model: str | None = Field(None, description="Model to use (sonnet, opus, haiku)")
        system_prompt: str | None = Field(None, description="System prompt override")
        session_id: str | None = Field(None, description="Session ID for multi-turn")
        resume: bool = Field(False, description="Resume existing session")
        timeout: float = Field(300.0, description="Timeout in seconds")
        max_budget_usd: float | None = Field(None, description="Maximum budget in USD")
        allowed_tools: list[str] | None = Field(None, description="Allowed tool names")
        context: dict[str, Any] | None = Field(None, description="Context data (TOON encoded)")
        stream: bool = Field(True, description="Stream response")

    class QueryResponse(BaseModel):
        """Response from query endpoint."""

        session_id: str
        content: str
        input_tokens: int
        output_tokens: int
        duration_seconds: float
        turn_count: int

    class SessionInfo(BaseModel):
        """Session information."""

        session_id: str
        model: str | None
        status: str
        turn_count: int
        created_at: datetime
        updated_at: datetime
        tags: list[str]

    class SessionListResponse(BaseModel):
        """Response for session list."""

        sessions: list[SessionInfo]
        total: int

    class HealthResponse(BaseModel):
        """Health check response."""

        status: str
        version: str
        cli_available: bool
        active_sessions: int
        uptime_seconds: float

    class DeepHealthResponse(BaseModel):
        """Deep health check response with CLI verification."""

        healthy: bool
        cli_available: bool
        cli_executable: bool
        cli_authenticated: bool
        cli_version: str | None
        latency_ms: float | None
        error: str | None
        active_sessions: int
        uptime_seconds: float
        circuit_breaker_state: str | None

    class ErrorResponse(BaseModel):
        """Error response."""

        error: str
        error_type: str
        detail: str | None = None


class CCFlowServer:
    """FastAPI server for ccflow middleware.

    Provides REST and WebSocket interfaces for remote access.

    Example:
        >>> server = CCFlowServer()
        >>> app = server.app
        >>> # Run with: uvicorn ccflow.server:app
    """

    def __init__(
        self,
        manager: SessionManager | None = None,
        *,
        cors_origins: list[str] | None = None,
        enable_metrics: bool = True,
    ) -> None:
        """Initialize server.

        Args:
            manager: Session manager instance. Creates default if None.
            cors_origins: Allowed CORS origins. Defaults to ["*"].
            enable_metrics: Enable Prometheus metrics endpoint.
        """
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI not installed. Install with: pip install ccflow[server]")

        self._manager = manager
        self._cors_origins = cors_origins or ["*"]
        self._enable_metrics = enable_metrics
        self._start_time = datetime.now()

        # Active WebSocket connections
        self._websocket_connections: dict[str, WebSocket] = {}

        # Create FastAPI app
        self._app = FastAPI(
            title="ccflow API",
            description="REST and WebSocket API for Claude Code CLI middleware",
            version="0.1.0",
            lifespan=self._lifespan,
        )

        # Add middleware
        self._setup_middleware()

        # Register routes
        self._setup_routes()

        logger.info("ccflow_server_initialized")

    @property
    def app(self) -> FastAPI:
        """Get FastAPI application."""
        return self._app

    @asynccontextmanager
    async def _lifespan(self, _app: FastAPI) -> AsyncIterator[None]:
        """Application lifespan handler."""
        # Startup
        if self._manager is None:
            self._manager = SessionManager(enable_metrics=self._enable_metrics)
        await self._manager.start()
        logger.info("ccflow_server_started")

        yield

        # Shutdown
        await self._manager.stop()

        # Close WebSocket connections
        for ws in list(self._websocket_connections.values()):
            with suppress(Exception):
                await ws.close()

        logger.info("ccflow_server_stopped")

    def _setup_middleware(self) -> None:
        """Configure middleware."""
        self._app.add_middleware(
            CORSMiddleware,
            allow_origins=self._cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Add correlation ID middleware
        from starlette.middleware.base import BaseHTTPMiddleware

        class CorrelationIDMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request: Request, call_next: Any) -> Response:
                # Get or generate correlation ID
                correlation_id = request.headers.get("X-Correlation-ID")
                cid = set_correlation_id(correlation_id)

                # Process request
                response = await call_next(request)

                # Add correlation ID to response headers
                response.headers["X-Correlation-ID"] = cid
                return response

        self._app.add_middleware(CorrelationIDMiddleware)

    def _setup_routes(self) -> None:
        """Register API routes."""
        app = self._app

        # Health endpoints
        @app.get("/health", response_model=HealthResponse, tags=["Health"])
        async def health_check() -> HealthResponse:
            """Check server health."""
            from ccflow.executor import get_executor

            cli_available = False
            try:
                executor = get_executor()
                cli_available = await executor.check_cli_available()
            except CLINotFoundError:
                pass

            uptime = (datetime.now() - self._start_time).total_seconds()

            return HealthResponse(
                status="healthy",
                version="0.1.0",
                cli_available=cli_available,
                active_sessions=self._manager.active_session_count if self._manager else 0,
                uptime_seconds=uptime,
            )

        @app.get("/ready", tags=["Health"])
        async def readiness_check() -> dict:
            """Check if server is ready to accept requests."""
            health_checker = get_health_checker()
            health_status = await health_checker.check()

            if not health_status.healthy:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=health_status.error or "CLI not available",
                )

            return {"ready": True, "correlation_id": get_correlation_id()}

        @app.get("/health/deep", response_model=DeepHealthResponse, tags=["Health"])
        async def deep_health_check(force: bool = False) -> DeepHealthResponse:
            """Deep health check that verifies CLI can execute queries.

            Args:
                force: Bypass cache and perform fresh check

            This endpoint:
            - Verifies CLI is in PATH
            - Checks CLI is executable (version check)
            - Executes a minimal query to verify authentication
            - Reports circuit breaker state
            """
            from ccflow.reliability import get_cli_circuit_breaker

            health_checker = get_health_checker()
            health_status = await health_checker.check(force=force)
            uptime = (datetime.now() - self._start_time).total_seconds()

            # Get circuit breaker state
            breaker = get_cli_circuit_breaker()
            circuit_state = breaker.state.value if breaker else None

            return DeepHealthResponse(
                healthy=health_status.healthy,
                cli_available=health_status.cli_available,
                cli_executable=health_status.cli_executable,
                cli_authenticated=health_status.cli_authenticated,
                cli_version=health_status.version,
                latency_ms=health_status.latency_ms,
                error=health_status.error,
                active_sessions=self._manager.active_session_count if self._manager else 0,
                uptime_seconds=uptime,
                circuit_breaker_state=circuit_state,
            )

        # Query endpoints
        @app.post("/query", response_model=QueryResponse, tags=["Query"])
        async def query(request: QueryRequest) -> QueryResponse:
            """Execute a query and return complete response."""
            if self._manager is None:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Server not initialized",
                )

            options = CLIAgentOptions(
                model=request.model or get_settings().default_model,
                system_prompt=request.system_prompt,
                session_id=request.session_id,
                resume=request.resume,
                timeout=request.timeout,
                max_budget_usd=request.max_budget_usd,
                allowed_tools=request.allowed_tools,
                context=request.context,
            )

            try:
                session = await self._manager.create_session(options=options)
                content_parts: list[str] = []
                input_tokens = 0
                output_tokens = 0

                async for msg in session.send_message(request.prompt):
                    if hasattr(msg, "content") and msg.content:
                        content_parts.append(str(msg.content))
                    # Aggregate tokens from stop message
                    if hasattr(msg, "usage"):
                        input_tokens = msg.usage.get("input_tokens", 0)
                        output_tokens = msg.usage.get("output_tokens", 0)

                stats = await session.close()

                return QueryResponse(
                    session_id=session.session_id,
                    content="".join(content_parts),
                    input_tokens=stats.total_input_tokens or input_tokens,
                    output_tokens=stats.total_output_tokens or output_tokens,
                    duration_seconds=stats.duration_seconds,
                    turn_count=stats.total_turns,
                )

            except RateLimitExceededError as e:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=str(e),
                    headers={"Retry-After": str(int(e.retry_after))},
                ) from e
            except ConcurrencyLimitExceededError as e:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=str(e),
                ) from e
            except CircuitBreakerError as e:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"Service temporarily unavailable: {e}",
                    headers={"Retry-After": str(int(e.retry_after))} if e.retry_after else None,
                ) from e
            except CCFlowError as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=str(e),
                ) from e

        @app.post("/query/stream", tags=["Query"])
        async def query_stream(request: QueryRequest) -> StreamingResponse:
            """Execute a query with streaming response."""
            if self._manager is None:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Server not initialized",
                )

            async def generate() -> AsyncIterator[str]:
                manager = self._manager
                assert manager is not None  # Checked above

                options = CLIAgentOptions(
                    model=request.model or get_settings().default_model,
                    system_prompt=request.system_prompt,
                    session_id=request.session_id,
                    resume=request.resume,
                    timeout=request.timeout,
                    max_budget_usd=request.max_budget_usd,
                    allowed_tools=request.allowed_tools,
                    context=request.context,
                )

                try:
                    session = await manager.create_session(options=options)

                    async for msg in session.send_message(request.prompt):
                        # Yield as Server-Sent Events
                        import json

                        event_data = _message_to_dict(msg)
                        yield f"data: {json.dumps(event_data)}\n\n"

                    # Send done event
                    stats = await session.close()
                    yield f"data: {json.dumps({'type': 'done', 'session_id': session.session_id, 'stats': {'turns': stats.total_turns, 'duration': stats.duration_seconds}})}\n\n"

                except Exception as e:
                    import json

                    yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )

        # Session endpoints
        @app.get("/sessions", response_model=SessionListResponse, tags=["Sessions"])
        async def list_sessions(
            session_status: str | None = None,
            model: str | None = None,
            limit: int = 100,
            offset: int = 0,
        ) -> SessionListResponse:
            """List sessions with optional filtering."""
            if self._manager is None:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Server not initialized",
                )

            from ccflow.store import SessionStatus

            status_filter = None
            if session_status:
                try:
                    status_filter = SessionStatus(session_status)
                except ValueError as e:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Invalid status: {session_status}",
                    ) from e

            sessions = await self._manager.list_sessions(
                status=status_filter,
                model=model,
                limit=limit,
                offset=offset,
            )

            total = await self._manager.count_sessions(
                status=status_filter,
                model=model,
            )

            return SessionListResponse(
                sessions=[
                    SessionInfo(
                        session_id=s.session_id,
                        model=s.model,
                        status=s.status.value,
                        turn_count=s.turn_count,
                        created_at=s.created_at,
                        updated_at=s.updated_at,
                        tags=s.tags,
                    )
                    for s in sessions
                ],
                total=total,
            )

        @app.get("/sessions/{session_id}", response_model=SessionInfo, tags=["Sessions"])
        async def get_session(session_id: str) -> SessionInfo:
            """Get session information."""
            if self._manager is None:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Server not initialized",
                )

            session = await self._manager.get_session(session_id)
            if session is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Session not found: {session_id}",
                )

            return SessionInfo(
                session_id=session.session_id,
                model=session.options.model if session.options else None,
                status="active" if not session.is_closed else "closed",
                turn_count=session.turn_count,
                created_at=session.created_at,
                updated_at=session.updated_at,
                tags=list(session.tags),
            )

        @app.delete("/sessions/{session_id}", tags=["Sessions"])
        async def delete_session(session_id: str) -> dict:
            """Delete a session."""
            if self._manager is None:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Server not initialized",
                )

            deleted = await self._manager.delete_session(session_id)
            if not deleted:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Session not found: {session_id}",
                )

            return {"deleted": True, "session_id": session_id}

        # WebSocket endpoint
        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket) -> None:
            """WebSocket endpoint for real-time streaming."""
            await websocket.accept()

            connection_id = str(uuid.uuid4())
            self._websocket_connections[connection_id] = websocket

            try:
                while True:
                    # Receive query request
                    data = await websocket.receive_json()

                    if data.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})
                        continue

                    if data.get("type") == "query":
                        await self._handle_ws_query(websocket, data)

            except WebSocketDisconnect:
                logger.debug("websocket_disconnected", connection_id=connection_id)
            except Exception as e:
                logger.error("websocket_error", error=str(e))
                with suppress(Exception):
                    await websocket.send_json(
                        {
                            "type": "error",
                            "error": str(e),
                        }
                    )
            finally:
                self._websocket_connections.pop(connection_id, None)

        # Metrics endpoint (if Prometheus available)
        if self._enable_metrics:
            try:
                from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

                @app.get("/metrics", tags=["Metrics"])
                async def metrics() -> StreamingResponse:
                    """Prometheus metrics endpoint."""
                    return StreamingResponse(
                        iter([generate_latest()]),
                        media_type=CONTENT_TYPE_LATEST,
                    )

            except ImportError:
                pass

        # Rate limiter stats
        @app.get("/stats/rate-limiter", tags=["Stats"])
        async def rate_limiter_stats() -> dict:
            """Get rate limiter statistics."""
            limiter = get_limiter()
            return limiter.stats

        @app.get("/stats/sessions", tags=["Stats"])
        async def session_stats() -> dict:
            """Get session statistics."""
            if self._manager is None:
                return {}
            return await self._manager.get_stats()

    async def _handle_ws_query(self, websocket: WebSocket, data: dict) -> None:
        """Handle WebSocket query request."""
        if self._manager is None:
            await websocket.send_json(
                {
                    "type": "error",
                    "error": "Server not initialized",
                }
            )
            return

        prompt = data.get("prompt", "")
        if not prompt:
            await websocket.send_json(
                {
                    "type": "error",
                    "error": "Missing prompt",
                }
            )
            return

        options = CLIAgentOptions(
            model=data.get("model") or get_settings().default_model,
            system_prompt=data.get("system_prompt"),
            session_id=data.get("session_id"),
            resume=data.get("resume", False),
            timeout=data.get("timeout", 300.0),
            max_budget_usd=data.get("max_budget_usd"),
            allowed_tools=data.get("allowed_tools"),
            context=data.get("context"),
        )

        try:
            session = await self._manager.create_session(options=options)

            # Send session start
            await websocket.send_json(
                {
                    "type": "session_start",
                    "session_id": session.session_id,
                }
            )

            async for msg in session.send_message(prompt):
                event_data = _message_to_dict(msg)
                await websocket.send_json(event_data)

            # Send done
            stats = await session.close()
            await websocket.send_json(
                {
                    "type": "done",
                    "session_id": session.session_id,
                    "stats": {
                        "turns": stats.total_turns,
                        "input_tokens": stats.total_input_tokens,
                        "output_tokens": stats.total_output_tokens,
                        "duration": stats.duration_seconds,
                    },
                }
            )

        except RateLimitExceededError as e:
            await websocket.send_json(
                {
                    "type": "error",
                    "error": "rate_limit_exceeded",
                    "retry_after": e.retry_after,
                }
            )
        except ConcurrencyLimitExceededError as e:
            await websocket.send_json(
                {
                    "type": "error",
                    "error": "concurrency_limit_exceeded",
                    "current": e.current,
                    "limit": e.limit,
                }
            )
        except CircuitBreakerError as e:
            await websocket.send_json(
                {
                    "type": "error",
                    "error": "circuit_breaker_open",
                    "retry_after": e.retry_after,
                }
            )
        except Exception as e:
            await websocket.send_json(
                {
                    "type": "error",
                    "error": str(e),
                }
            )


def _message_to_dict(msg: Message) -> dict:
    """Convert Message to dictionary for JSON serialization."""
    result: dict[str, Any] = {"type": type(msg).__name__}

    if hasattr(msg, "content"):
        result["content"] = msg.content
    if hasattr(msg, "session_id"):
        result["session_id"] = msg.session_id
    if hasattr(msg, "tool"):
        result["tool"] = msg.tool
    if hasattr(msg, "args"):
        result["args"] = msg.args
    if hasattr(msg, "result"):
        result["result"] = msg.result
    if hasattr(msg, "usage"):
        result["usage"] = msg.usage

    return result


# Module-level app instance for uvicorn
_server: CCFlowServer | None = None


def get_app() -> FastAPI:
    """Get or create the FastAPI application.

    Usage with uvicorn:
        uvicorn ccflow.server:get_app --factory
    """
    global _server
    if _server is None:
        _server = CCFlowServer()
    return _server.app


# Direct app reference for simpler uvicorn usage
# uvicorn ccflow.server:app
def create_app() -> FastAPI:
    """Create a new FastAPI application."""
    return CCFlowServer().app
