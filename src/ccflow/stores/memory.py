"""
Memory Session Store - In-memory session storage for testing.

Provides a fast, ephemeral storage backend that doesn't persist
between process restarts. Ideal for testing and development.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta

import structlog

from ccflow.store import (
    BaseSessionStore,
    SessionFilter,
    SessionMetadata,
    SessionState,
    SessionStatus,
)

logger = structlog.get_logger(__name__)


class MemorySessionStore(BaseSessionStore):
    """In-memory session storage for testing.

    All sessions are stored in a dictionary and lost when the
    process exits. Thread-safe for async operations.

    Example:
        >>> store = MemorySessionStore()
        >>> await store.save(session_state)
        >>> loaded = await store.load(session_id)
        >>> assert loaded.session_id == session_state.session_id
    """

    def __init__(self) -> None:
        """Initialize empty in-memory store."""
        self._sessions: dict[str, SessionState] = {}
        logger.debug("memory_store_initialized")

    async def save(self, state: SessionState) -> None:
        """Save or update session state."""
        # Deep copy to prevent external mutations
        self._sessions[state.session_id] = deepcopy(state)
        logger.debug("session_saved", session_id=state.session_id)

    async def load(self, session_id: str) -> SessionState | None:
        """Load session state by ID."""
        state = self._sessions.get(session_id)
        if state is not None:
            # Return a copy to prevent external mutations
            return deepcopy(state)
        return None

    async def delete(self, session_id: str) -> bool:
        """Delete session by ID."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.debug("session_deleted", session_id=session_id)
            return True
        return False

    async def list(self, filter: SessionFilter | None = None) -> list[SessionMetadata]:
        """List sessions matching filter criteria."""
        filter = filter or SessionFilter()
        results: list[SessionMetadata] = []

        for state in self._sessions.values():
            # Apply filters
            if filter.status is not None and state.status != filter.status:
                continue

            if filter.model is not None and state.model != filter.model:
                continue

            if filter.created_after is not None and state.created_at < filter.created_after:
                continue

            if filter.created_before is not None and state.created_at > filter.created_before:
                continue

            if filter.updated_after is not None and state.updated_at < filter.updated_after:
                continue

            if filter.min_turns is not None and state.turn_count < filter.min_turns:
                continue

            if filter.max_turns is not None and state.turn_count > filter.max_turns:
                continue

            if filter.tags and not any(tag in state.tags for tag in filter.tags):
                # Match any tag
                continue

            results.append(state.to_metadata())

        # Sort results
        reverse = filter.order_desc
        if filter.order_by == "created_at":
            results.sort(key=lambda m: m.created_at, reverse=reverse)
        elif filter.order_by == "updated_at":
            results.sort(key=lambda m: m.updated_at, reverse=reverse)
        elif filter.order_by == "turn_count":
            results.sort(key=lambda m: m.turn_count, reverse=reverse)
        elif filter.order_by == "total_tokens":
            results.sort(key=lambda m: m.total_tokens, reverse=reverse)

        # Apply pagination
        start = filter.offset
        end = start + filter.limit
        return results[start:end]

    async def cleanup(self, older_than: timedelta) -> int:
        """Delete sessions older than specified duration."""
        cutoff = datetime.now() - older_than
        to_delete = [sid for sid, state in self._sessions.items() if state.updated_at < cutoff]

        for sid in to_delete:
            del self._sessions[sid]

        if to_delete:
            logger.info("sessions_cleaned_up", count=len(to_delete))

        return len(to_delete)

    async def exists(self, session_id: str) -> bool:
        """Check if session exists."""
        return session_id in self._sessions

    async def count(self, filter: SessionFilter | None = None) -> int:
        """Count sessions matching filter."""
        # Use list and count for simplicity (efficient enough for in-memory)
        sessions = await self.list(filter)
        return len(sessions)

    async def update_status(self, session_id: str, status: SessionStatus) -> bool:
        """Update session status."""
        if session_id not in self._sessions:
            return False

        self._sessions[session_id].status = status
        self._sessions[session_id].updated_at = datetime.now()
        logger.debug("session_status_updated", session_id=session_id, status=status.value)
        return True

    async def close(self) -> None:
        """Clear all sessions (for cleanup)."""
        self._sessions.clear()
        logger.debug("memory_store_closed")

    def clear(self) -> None:
        """Synchronous clear for testing convenience."""
        self._sessions.clear()

    @property
    def session_count(self) -> int:
        """Get current session count (for testing)."""
        return len(self._sessions)
