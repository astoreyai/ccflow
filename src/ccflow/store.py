"""
Session Store - Persistence layer for session state management.

Provides a protocol-based interface for storing and retrieving session
state, with implementations for SQLite and in-memory storage.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Protocol, runtime_checkable

import structlog

logger = structlog.get_logger(__name__)


class SessionStatus(str, Enum):
    """Session lifecycle status."""

    ACTIVE = "active"
    CLOSED = "closed"
    EXPIRED = "expired"
    ERROR = "error"


@dataclass
class SessionMetadata:
    """Lightweight session information for listing.

    Contains only essential fields for session discovery
    without loading full conversation history.
    """

    session_id: str
    created_at: datetime
    updated_at: datetime
    status: SessionStatus
    model: str
    turn_count: int
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float = 0.0
    tags: list[str] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        """Total tokens consumed."""
        return self.total_input_tokens + self.total_output_tokens

    @property
    def duration(self) -> timedelta:
        """Time since session creation."""
        return self.updated_at - self.created_at


@dataclass
class SessionState:
    """Complete session state for persistence.

    Contains full session data including conversation history
    checksum for integrity verification.
    """

    # Identity
    session_id: str
    created_at: datetime
    updated_at: datetime
    status: SessionStatus

    # Configuration
    model: str
    system_prompt: str | None = None
    append_system_prompt: str | None = None

    # Statistics
    turn_count: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0

    # Conversation tracking
    messages_hash: str = ""  # SHA256 of message history for integrity
    last_prompt: str = ""
    last_response: str = ""

    # Metadata
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # TOON tracking
    toon_savings_ratio: float = 0.0

    @property
    def total_tokens(self) -> int:
        """Total tokens consumed."""
        return self.total_input_tokens + self.total_output_tokens

    def to_metadata(self) -> SessionMetadata:
        """Convert to lightweight metadata."""
        return SessionMetadata(
            session_id=self.session_id,
            created_at=self.created_at,
            updated_at=self.updated_at,
            status=self.status,
            model=self.model,
            turn_count=self.turn_count,
            total_input_tokens=self.total_input_tokens,
            total_output_tokens=self.total_output_tokens,
            total_cost_usd=self.total_cost_usd,
            tags=self.tags.copy(),
        )

    def compute_hash(self, content: str) -> str:
        """Compute SHA256 hash of content."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def update_hash(self, prompt: str, response: str) -> None:
        """Update messages hash with new turn."""
        combined = f"{self.messages_hash}:{prompt}:{response}"
        self.messages_hash = self.compute_hash(combined)
        self.last_prompt = prompt
        self.last_response = response

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for storage."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "status": self.status.value,
            "model": self.model,
            "system_prompt": self.system_prompt,
            "append_system_prompt": self.append_system_prompt,
            "turn_count": self.turn_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": self.total_cost_usd,
            "messages_hash": self.messages_hash,
            "last_prompt": self.last_prompt,
            "last_response": self.last_response,
            "tags": self.tags,
            "metadata": self.metadata,
            "toon_savings_ratio": self.toon_savings_ratio,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionState:
        """Deserialize from dictionary."""
        return cls(
            session_id=data["session_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            status=SessionStatus(data["status"]),
            model=data["model"],
            system_prompt=data.get("system_prompt"),
            append_system_prompt=data.get("append_system_prompt"),
            turn_count=data.get("turn_count", 0),
            total_input_tokens=data.get("total_input_tokens", 0),
            total_output_tokens=data.get("total_output_tokens", 0),
            total_cost_usd=data.get("total_cost_usd", 0.0),
            messages_hash=data.get("messages_hash", ""),
            last_prompt=data.get("last_prompt", ""),
            last_response=data.get("last_response", ""),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
            toon_savings_ratio=data.get("toon_savings_ratio", 0.0),
        )


@dataclass
class SessionFilter:
    """Filter criteria for session queries."""

    status: SessionStatus | None = None
    model: str | None = None
    tags: list[str] | None = None
    created_after: datetime | None = None
    created_before: datetime | None = None
    updated_after: datetime | None = None
    min_turns: int | None = None
    max_turns: int | None = None
    limit: int = 100
    offset: int = 0
    order_by: str = "updated_at"
    order_desc: bool = True


@runtime_checkable
class SessionStore(Protocol):
    """Protocol for session persistence backends.

    Implementations must be async-compatible and handle
    concurrent access appropriately.

    Example:
        >>> store = SQLiteSessionStore("sessions.db")
        >>> await store.save(session_state)
        >>> loaded = await store.load(session_id)
        >>> sessions = await store.list(SessionFilter(status=SessionStatus.ACTIVE))
    """

    async def save(self, state: SessionState) -> None:
        """Save or update session state.

        Args:
            state: Session state to persist

        Raises:
            SessionStoreError: If save fails
        """
        ...

    async def load(self, session_id: str) -> SessionState | None:
        """Load session state by ID.

        Args:
            session_id: Session UUID to load

        Returns:
            SessionState if found, None otherwise

        Raises:
            SessionStoreError: If load fails
        """
        ...

    async def delete(self, session_id: str) -> bool:
        """Delete session by ID.

        Args:
            session_id: Session UUID to delete

        Returns:
            True if deleted, False if not found

        Raises:
            SessionStoreError: If delete fails
        """
        ...

    async def list(self, filter: SessionFilter | None = None) -> list[SessionMetadata]:
        """List sessions matching filter criteria.

        Args:
            filter: Optional filter criteria

        Returns:
            List of session metadata (lightweight)

        Raises:
            SessionStoreError: If query fails
        """
        ...

    async def cleanup(self, older_than: timedelta) -> int:
        """Delete sessions older than specified duration.

        Args:
            older_than: Age threshold for deletion

        Returns:
            Number of sessions deleted

        Raises:
            SessionStoreError: If cleanup fails
        """
        ...

    async def exists(self, session_id: str) -> bool:
        """Check if session exists.

        Args:
            session_id: Session UUID to check

        Returns:
            True if exists, False otherwise
        """
        ...

    async def count(self, filter: SessionFilter | None = None) -> int:
        """Count sessions matching filter.

        Args:
            filter: Optional filter criteria

        Returns:
            Number of matching sessions
        """
        ...

    async def update_status(self, session_id: str, status: SessionStatus) -> bool:
        """Update session status.

        Args:
            session_id: Session UUID
            status: New status

        Returns:
            True if updated, False if not found
        """
        ...

    async def close(self) -> None:
        """Close store and release resources."""
        ...


class BaseSessionStore(ABC):
    """Abstract base class for session stores.

    Provides common functionality and enforces the SessionStore protocol.
    """

    @abstractmethod
    async def save(self, state: SessionState) -> None:
        """Save or update session state."""
        pass

    @abstractmethod
    async def load(self, session_id: str) -> SessionState | None:
        """Load session state by ID."""
        pass

    @abstractmethod
    async def delete(self, session_id: str) -> bool:
        """Delete session by ID."""
        pass

    @abstractmethod
    async def list(self, filter: SessionFilter | None = None) -> list[SessionMetadata]:
        """List sessions matching filter criteria."""
        pass

    @abstractmethod
    async def cleanup(self, older_than: timedelta) -> int:
        """Delete sessions older than specified duration."""
        pass

    async def exists(self, session_id: str) -> bool:
        """Check if session exists."""
        state = await self.load(session_id)
        return state is not None

    async def count(self, filter: SessionFilter | None = None) -> int:
        """Count sessions matching filter."""
        sessions = await self.list(filter)
        return len(sessions)

    async def update_status(self, session_id: str, status: SessionStatus) -> bool:
        """Update session status."""
        state = await self.load(session_id)
        if state is None:
            return False
        state.status = status
        state.updated_at = datetime.now()
        await self.save(state)
        return True

    async def close(self) -> None:
        """Close store and release resources."""
        pass

    async def __aenter__(self) -> BaseSessionStore:
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
