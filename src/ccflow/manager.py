"""
Session Manager - High-level session lifecycle management.

Provides a unified interface for creating, discovering, and managing
multiple sessions with automatic persistence and cleanup.
"""

from __future__ import annotations

import asyncio
import contextlib
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

import structlog

from ccflow.events import (
    EventEmitter,
    SessionDeletedEvent,
    get_emitter,
)
from ccflow.executor import CLIExecutor, get_executor
from ccflow.metrics_handlers import PrometheusEventHandler, setup_metrics
from ccflow.session import Session, load_session
from ccflow.store import (
    SessionFilter,
    SessionMetadata,
    SessionStatus,
    SessionStore,
)
from ccflow.stores import SQLiteSessionStore
from ccflow.types import CLIAgentOptions, Message

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = structlog.get_logger(__name__)


class SessionManager:
    """High-level session lifecycle manager.

    Provides a unified interface for:
    - Creating new sessions with automatic persistence
    - Loading and resuming existing sessions
    - Discovering sessions with filtering
    - Automatic cleanup of expired sessions
    - Tracking active sessions

    Example:
        >>> manager = SessionManager()  # Uses default SQLite store
        >>> async with manager:
        ...     session = await manager.create_session(model="opus")
        ...     async for msg in session.send_message("Hello"):
        ...         print(msg.content, end="")
        ...     await session.close()

    With custom store:
        >>> store = SQLiteSessionStore("my_sessions.db")
        >>> manager = SessionManager(store=store)
    """

    def __init__(
        self,
        store: SessionStore | None = None,
        executor: CLIExecutor | None = None,
        emitter: EventEmitter | None = None,
        *,
        auto_cleanup: bool = True,
        cleanup_interval: timedelta = timedelta(hours=1),
        session_ttl: timedelta = timedelta(days=7),
        enable_metrics: bool = False,
    ) -> None:
        """Initialize session manager.

        Args:
            store: Session store for persistence. Uses SQLiteSessionStore if None.
            executor: CLI executor instance. Uses default if None.
            emitter: Event emitter instance. Uses global if None.
            auto_cleanup: Enable automatic cleanup of expired sessions.
            cleanup_interval: How often to run cleanup (if auto_cleanup=True).
            session_ttl: Time-to-live for inactive sessions.
            enable_metrics: Enable Prometheus metrics collection.
        """
        self._store = store or SQLiteSessionStore()
        self._executor = executor or get_executor()
        self._emitter = emitter or get_emitter()
        self._auto_cleanup = auto_cleanup
        self._cleanup_interval = cleanup_interval
        self._session_ttl = session_ttl
        self._metrics_handler: PrometheusEventHandler | None = None

        # Track active sessions
        self._active_sessions: dict[str, Session] = {}
        self._cleanup_task: asyncio.Task | None = None
        self._running = False

        # Set up metrics if enabled
        if enable_metrics:
            self._metrics_handler = setup_metrics(self._emitter, start_server=False)

        logger.debug(
            "session_manager_initialized",
            auto_cleanup=auto_cleanup,
            session_ttl=str(session_ttl),
            metrics_enabled=enable_metrics,
        )

    @property
    def store(self) -> SessionStore:
        """Get the underlying session store."""
        return self._store

    @property
    def active_session_count(self) -> int:
        """Get count of currently active (in-memory) sessions."""
        return len(self._active_sessions)

    @property
    def active_sessions(self) -> list[str]:
        """Get list of active session IDs."""
        return list(self._active_sessions.keys())

    async def start(self) -> None:
        """Start the session manager.

        Initializes the store and starts background cleanup if enabled.
        """
        if self._running:
            return

        self._running = True

        if self._auto_cleanup:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.debug("cleanup_task_started", interval=str(self._cleanup_interval))

        logger.info("session_manager_started")

    async def stop(self) -> None:
        """Stop the session manager.

        Closes all active sessions and stops background tasks.
        """
        if not self._running:
            return

        self._running = False

        # Cancel cleanup task
        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._cleanup_task
            self._cleanup_task = None

        # Close all active sessions
        for session_id in list(self._active_sessions.keys()):
            try:
                session = self._active_sessions.pop(session_id)
                if not session.is_closed:
                    await session.close()
            except Exception as e:
                logger.warning(
                    "session_close_failed",
                    session_id=session_id,
                    error=str(e),
                )

        # Close store
        await self._store.close()

        logger.info("session_manager_stopped")

    async def __aenter__(self) -> SessionManager:
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Async context manager exit."""
        await self.stop()

    # Session Factory Methods

    async def create_session(
        self,
        session_id: str | None = None,
        options: CLIAgentOptions | None = None,
        tags: list[str] | None = None,
    ) -> Session:
        """Create a new session with automatic persistence.

        Args:
            session_id: Optional specific session ID (auto-generated if None).
            options: CLI agent options.
            tags: Optional tags for organization.

        Returns:
            New Session instance.
        """
        session = Session(
            session_id=session_id,
            options=options,
            executor=self._executor,
            store=self._store,
            emitter=self._emitter,
        )

        if tags:
            for tag in tags:
                session.add_tag(tag)

        # Track active session
        self._active_sessions[session.session_id] = session

        # Initial persist
        await session._persist()

        logger.info(
            "session_created",
            session_id=session.session_id,
            tags=tags,
        )

        return session

    async def get_session(self, session_id: str) -> Session | None:
        """Get an active session by ID.

        Returns the in-memory session if active, otherwise loads from store.

        Args:
            session_id: Session UUID.

        Returns:
            Session if found, None otherwise.
        """
        # Check active sessions first
        if session_id in self._active_sessions:
            return self._active_sessions[session_id]

        # Load from store
        session = await load_session(session_id, self._store, emitter=self._emitter)
        if session is not None:
            self._active_sessions[session_id] = session

        return session

    async def load_session(
        self,
        session_id: str,
        options: CLIAgentOptions | None = None,
    ) -> Session | None:
        """Load and resume a session from the store.

        Args:
            session_id: Session UUID to load.
            options: Optional options override.

        Returns:
            Session if found, None otherwise.
        """
        session = await load_session(session_id, self._store, options, emitter=self._emitter)
        if session is not None:
            self._active_sessions[session_id] = session
        return session

    async def close_session(self, session_id: str) -> bool:
        """Close a specific session.

        Args:
            session_id: Session UUID to close.

        Returns:
            True if closed, False if not found.
        """
        session = self._active_sessions.pop(session_id, None)
        if session is not None:
            if not session.is_closed:
                await session.close()
            return True

        # Try to update status in store directly
        return await self._store.update_status(session_id, SessionStatus.CLOSED)

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session from the store.

        Args:
            session_id: Session UUID to delete.

        Returns:
            True if deleted, False if not found.
        """
        # Remove from active sessions
        session = self._active_sessions.pop(session_id, None)
        if session is not None and not session.is_closed:
            session._is_closed = True  # Mark closed without persisting

        deleted = await self._store.delete(session_id)
        if deleted:
            await self._emitter.emit(SessionDeletedEvent(session_id=session_id))
        return deleted

    # Session Discovery

    async def list_sessions(
        self,
        status: SessionStatus | None = None,
        model: str | None = None,
        tags: list[str] | None = None,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
        min_turns: int | None = None,
        max_turns: int | None = None,
        limit: int = 100,
        offset: int = 0,
        order_by: str = "updated_at",
        order_desc: bool = True,
    ) -> list[SessionMetadata]:
        """List sessions matching filter criteria.

        Args:
            status: Filter by session status.
            model: Filter by model name.
            tags: Filter by tags (matches any).
            created_after: Filter by creation date.
            created_before: Filter by creation date.
            min_turns: Minimum turn count.
            max_turns: Maximum turn count.
            limit: Maximum results.
            offset: Pagination offset.
            order_by: Sort field (created_at, updated_at, turn_count).
            order_desc: Sort descending.

        Returns:
            List of session metadata.
        """
        filter = SessionFilter(
            status=status,
            model=model,
            tags=tags,
            created_after=created_after,
            created_before=created_before,
            min_turns=min_turns,
            max_turns=max_turns,
            limit=limit,
            offset=offset,
            order_by=order_by,
            order_desc=order_desc,
        )
        return await self._store.list(filter)

    async def list_active_sessions(self) -> list[SessionMetadata]:
        """List all active sessions."""
        return await self.list_sessions(status=SessionStatus.ACTIVE)

    async def list_recent_sessions(
        self,
        hours: int = 24,
        limit: int = 10,
    ) -> list[SessionMetadata]:
        """List recently updated sessions.

        Args:
            hours: Look back period in hours.
            limit: Maximum results.

        Returns:
            List of recent session metadata.
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        return await self.list_sessions(
            created_after=cutoff,
            limit=limit,
            order_by="updated_at",
            order_desc=True,
        )

    async def search_by_tags(self, tags: list[str]) -> list[SessionMetadata]:
        """Search sessions by tags.

        Args:
            tags: Tags to search for (matches any).

        Returns:
            Matching session metadata.
        """
        return await self.list_sessions(tags=tags)

    async def count_sessions(
        self,
        status: SessionStatus | None = None,
        model: str | None = None,
    ) -> int:
        """Count sessions matching criteria.

        Args:
            status: Filter by status.
            model: Filter by model.

        Returns:
            Session count.
        """
        filter = SessionFilter(status=status, model=model)
        return await self._store.count(filter)

    async def session_exists(self, session_id: str) -> bool:
        """Check if a session exists.

        Args:
            session_id: Session UUID.

        Returns:
            True if exists.
        """
        if session_id in self._active_sessions:
            return True
        return await self._store.exists(session_id)

    # Cleanup and Maintenance

    async def cleanup_expired(self) -> int:
        """Clean up expired sessions.

        Deletes sessions that haven't been updated within the TTL.

        Returns:
            Number of sessions deleted.
        """
        deleted = await self._store.cleanup(older_than=self._session_ttl)
        if deleted > 0:
            logger.info("expired_sessions_cleaned", count=deleted)
        return deleted

    async def cleanup_closed(self, older_than: timedelta | None = None) -> int:
        """Clean up closed sessions.

        Args:
            older_than: Only delete sessions closed before this duration.
                       Defaults to session_ttl.

        Returns:
            Number of sessions deleted.
        """
        threshold = older_than or self._session_ttl
        cutoff = datetime.now() - threshold

        # Get closed sessions older than cutoff
        closed = await self.list_sessions(
            status=SessionStatus.CLOSED,
            created_before=cutoff,
            limit=1000,
        )

        count = 0
        for meta in closed:
            if await self._store.delete(meta.session_id):
                count += 1

        if count > 0:
            logger.info("closed_sessions_cleaned", count=count)

        return count

    async def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while self._running:
            try:
                await asyncio.sleep(self._cleanup_interval.total_seconds())
                if self._running:
                    await self.cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("cleanup_error", error=str(e))

    # Statistics

    async def get_stats(self) -> dict:
        """Get session statistics.

        Returns:
            Dictionary with session statistics.
        """
        total = await self.count_sessions()
        active = await self.count_sessions(status=SessionStatus.ACTIVE)
        closed = await self.count_sessions(status=SessionStatus.CLOSED)

        return {
            "total_sessions": total,
            "active_sessions": active,
            "closed_sessions": closed,
            "in_memory_sessions": self.active_session_count,
        }

    # Convenience Methods

    async def quick_query(
        self,
        prompt: str,
        model: str | None = None,
        system_prompt: str | None = None,
    ) -> AsyncIterator[Message]:
        """Execute a quick one-off query without session management.

        Creates a temporary session, executes the query, and closes it.

        Args:
            prompt: The prompt to send.
            model: Model to use.
            system_prompt: Optional system prompt.

        Yields:
            Response messages.
        """
        options = CLIAgentOptions(
            model=model or "sonnet",
            system_prompt=system_prompt,
        )

        session = await self.create_session(options=options)
        try:
            async for msg in session.send_message(prompt):
                yield msg
        finally:
            await session.close()
            self._active_sessions.pop(session.session_id, None)


# Module-level default manager
_default_manager: SessionManager | None = None


def get_manager() -> SessionManager:
    """Get or create the default session manager."""
    global _default_manager
    if _default_manager is None:
        _default_manager = SessionManager()
    return _default_manager


async def init_manager(
    store: SessionStore | None = None,
    **kwargs: Any,
) -> SessionManager:
    """Initialize and start the default session manager.

    Args:
        store: Optional custom store.
        **kwargs: Additional SessionManager arguments.

    Returns:
        Started SessionManager instance.
    """
    global _default_manager
    _default_manager = SessionManager(store=store, **kwargs)
    await _default_manager.start()
    return _default_manager
