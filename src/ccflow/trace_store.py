"""
Trace and Project Store - Persistence protocols for trace recording and project organization.

Provides protocol-based interfaces for storing traces (full prompt/response cycles)
and projects (hierarchical organization containers).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import structlog

from ccflow.types import ProjectData, ProjectFilter, TraceData, TraceFilter, TraceStatus

if TYPE_CHECKING:
    from datetime import timedelta

    from ccflow.store import SessionMetadata

logger = structlog.get_logger(__name__)


@runtime_checkable
class TraceStore(Protocol):
    """Protocol for trace persistence backends.

    Traces capture complete prompt/response cycles including
    thinking content, tool calls, and metrics.

    Example:
        >>> store = SQLiteTraceStore("ccflow.db")
        >>> await store.save(trace)
        >>> loaded = await store.load(trace_id)
        >>> traces = await store.list(TraceFilter(session_id="..."))
    """

    async def save(self, trace: TraceData) -> None:
        """Save or update a trace.

        Args:
            trace: Trace data to persist

        Raises:
            SessionStoreError: If save fails
        """
        ...

    async def load(self, trace_id: str) -> TraceData | None:
        """Load trace by ID.

        Args:
            trace_id: Trace UUID to load

        Returns:
            TraceData if found, None otherwise

        Raises:
            SessionStoreError: If load fails
        """
        ...

    async def delete(self, trace_id: str) -> bool:
        """Delete trace by ID.

        Args:
            trace_id: Trace UUID to delete

        Returns:
            True if deleted, False if not found

        Raises:
            SessionStoreError: If delete fails
        """
        ...

    async def list(self, filter: TraceFilter | None = None) -> list[TraceData]:
        """List traces matching filter criteria.

        Args:
            filter: Optional filter criteria

        Returns:
            List of traces matching criteria

        Raises:
            SessionStoreError: If query fails
        """
        ...

    async def get_session_traces(
        self, session_id: str, limit: int = 100
    ) -> list[TraceData]:
        """Get all traces for a session.

        Args:
            session_id: Session UUID
            limit: Maximum traces to return

        Returns:
            Traces ordered by sequence_number
        """
        ...

    async def get_project_traces(
        self, project_id: str, limit: int = 100
    ) -> list[TraceData]:
        """Get all traces for a project.

        Args:
            project_id: Project UUID
            limit: Maximum traces to return

        Returns:
            Traces ordered by created_at descending
        """
        ...

    async def count(self, filter: TraceFilter | None = None) -> int:
        """Count traces matching filter.

        Args:
            filter: Optional filter criteria

        Returns:
            Number of matching traces
        """
        ...

    async def cleanup(self, older_than: timedelta) -> int:
        """Delete traces older than specified duration.

        Args:
            older_than: Age threshold for deletion

        Returns:
            Number of traces deleted

        Raises:
            SessionStoreError: If cleanup fails
        """
        ...

    async def close(self) -> None:
        """Close store and release resources."""
        ...


@runtime_checkable
class ProjectStore(Protocol):
    """Protocol for project persistence backends.

    Projects organize sessions and traces into hierarchical groups.

    Example:
        >>> store = SQLiteProjectStore("ccflow.db")
        >>> await store.save(project)
        >>> loaded = await store.load(project_id)
        >>> projects = await store.list(ProjectFilter(name_contains="PhD"))
    """

    async def save(self, project: ProjectData) -> None:
        """Save or update a project.

        Args:
            project: Project data to persist

        Raises:
            SessionStoreError: If save fails
        """
        ...

    async def load(self, project_id: str) -> ProjectData | None:
        """Load project by ID.

        Args:
            project_id: Project UUID to load

        Returns:
            ProjectData if found, None otherwise

        Raises:
            SessionStoreError: If load fails
        """
        ...

    async def delete(self, project_id: str) -> bool:
        """Delete project by ID.

        Args:
            project_id: Project UUID to delete

        Returns:
            True if deleted, False if not found

        Raises:
            SessionStoreError: If delete fails
        """
        ...

    async def list(self, filter: ProjectFilter | None = None) -> list[ProjectData]:
        """List projects matching filter criteria.

        Args:
            filter: Optional filter criteria

        Returns:
            List of projects matching criteria

        Raises:
            SessionStoreError: If query fails
        """
        ...

    async def get_project_sessions(
        self, project_id: str, limit: int = 100
    ) -> list[SessionMetadata]:
        """Get all sessions for a project.

        Args:
            project_id: Project UUID
            limit: Maximum sessions to return

        Returns:
            Sessions ordered by updated_at descending
        """
        ...

    async def get_subprojects(self, project_id: str) -> list[ProjectData]:
        """Get child projects of a parent.

        Args:
            project_id: Parent project UUID

        Returns:
            Child projects
        """
        ...

    async def count(self, filter: ProjectFilter | None = None) -> int:
        """Count projects matching filter.

        Args:
            filter: Optional filter criteria

        Returns:
            Number of matching projects
        """
        ...

    async def close(self) -> None:
        """Close store and release resources."""
        ...


class BaseTraceStore(ABC):
    """Abstract base class for trace stores.

    Provides common functionality and enforces the TraceStore protocol.
    """

    @abstractmethod
    async def save(self, trace: TraceData) -> None:
        """Save or update a trace."""
        pass

    @abstractmethod
    async def load(self, trace_id: str) -> TraceData | None:
        """Load trace by ID."""
        pass

    @abstractmethod
    async def delete(self, trace_id: str) -> bool:
        """Delete trace by ID."""
        pass

    @abstractmethod
    async def list(self, filter: TraceFilter | None = None) -> list[TraceData]:
        """List traces matching filter criteria."""
        pass

    async def get_session_traces(
        self, session_id: str, limit: int = 100
    ) -> list[TraceData]:
        """Get all traces for a session."""
        return await self.list(TraceFilter(session_id=session_id, limit=limit))

    async def get_project_traces(
        self, project_id: str, limit: int = 100
    ) -> list[TraceData]:
        """Get all traces for a project."""
        return await self.list(TraceFilter(project_id=project_id, limit=limit))

    async def count(self, filter: TraceFilter | None = None) -> int:
        """Count traces matching filter."""
        traces = await self.list(filter)
        return len(traces)

    async def update_status(self, trace_id: str, status: TraceStatus) -> bool:
        """Update trace status."""
        trace = await self.load(trace_id)
        if trace is None:
            return False
        trace.status = status
        await self.save(trace)
        return True

    @abstractmethod
    async def cleanup(self, older_than: timedelta) -> int:
        """Delete traces older than specified duration."""
        pass

    async def close(self) -> None:  # noqa: B027
        """Close store and release resources.

        Default implementation does nothing. Override in subclasses
        that need cleanup (e.g., database connections).
        """

    async def __aenter__(self) -> BaseTraceStore:
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Async context manager exit."""
        await self.close()


class BaseProjectStore(ABC):
    """Abstract base class for project stores.

    Provides common functionality and enforces the ProjectStore protocol.
    """

    @abstractmethod
    async def save(self, project: ProjectData) -> None:
        """Save or update a project."""
        pass

    @abstractmethod
    async def load(self, project_id: str) -> ProjectData | None:
        """Load project by ID."""
        pass

    @abstractmethod
    async def delete(self, project_id: str) -> bool:
        """Delete project by ID."""
        pass

    @abstractmethod
    async def list(self, filter: ProjectFilter | None = None) -> list[ProjectData]:
        """List projects matching filter criteria."""
        pass

    @abstractmethod
    async def get_project_sessions(
        self, project_id: str, limit: int = 100
    ) -> list[SessionMetadata]:
        """Get all sessions for a project."""
        pass

    async def get_subprojects(self, project_id: str) -> list[ProjectData]:
        """Get child projects of a parent."""
        return await self.list(ProjectFilter(parent_project_id=project_id))

    async def count(self, filter: ProjectFilter | None = None) -> int:
        """Count projects matching filter."""
        projects = await self.list(filter)
        return len(projects)

    async def exists(self, project_id: str) -> bool:
        """Check if project exists."""
        return await self.load(project_id) is not None

    async def close(self) -> None:  # noqa: B027
        """Close store and release resources.

        Default implementation does nothing. Override in subclasses
        that need cleanup (e.g., database connections).
        """

    async def __aenter__(self) -> BaseProjectStore:
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Async context manager exit."""
        await self.close()
