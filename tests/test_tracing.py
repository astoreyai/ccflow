"""Tests for TracingSession and Project functionality."""

from __future__ import annotations

import asyncio
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccflow.project import Project
from ccflow.stores.traces import SQLiteProjectStore, SQLiteTraceStore
from ccflow.stores import SQLiteSessionStore
from ccflow.tracing import TracingSession, create_tracing_session
from ccflow.types import (
    CLIAgentOptions,
    ProjectData,
    ProjectFilter,
    TraceData,
    TraceFilter,
    TraceStatus,
)


class TestTraceData:
    """Tests for TraceData dataclass."""

    def test_create_trace_data(self) -> None:
        """Test creating TraceData with defaults."""
        trace = TraceData(
            trace_id="test-trace-id",
            session_id="test-session-id",
        )

        assert trace.trace_id == "test-trace-id"
        assert trace.session_id == "test-session-id"
        assert trace.project_id is None
        assert trace.sequence_number == 0
        assert trace.prompt == ""
        assert trace.response == ""
        assert trace.thinking == ""
        assert trace.tool_calls == []
        assert trace.message_stream is None
        assert trace.status == TraceStatus.PENDING
        assert trace.input_tokens == 0
        assert trace.output_tokens == 0
        assert trace.thinking_tokens == 0

    def test_trace_data_has_detail(self) -> None:
        """Test has_detail property."""
        trace_no_detail = TraceData(trace_id="t1", session_id="s1")
        assert not trace_no_detail.has_detail

        trace_with_detail = TraceData(
            trace_id="t2",
            session_id="s2",
            message_stream=[{"type": "TextMessage"}],
        )
        assert trace_with_detail.has_detail

    def test_trace_data_total_tokens(self) -> None:
        """Test total_tokens property."""
        trace = TraceData(
            trace_id="t1",
            session_id="s1",
            input_tokens=100,
            output_tokens=200,
            thinking_tokens=50,
        )
        assert trace.total_tokens == 350


class TestProjectData:
    """Tests for ProjectData dataclass."""

    def test_create_project_data(self) -> None:
        """Test creating ProjectData with defaults."""
        project = ProjectData(
            project_id="test-project-id",
            name="Test Project",
        )

        assert project.project_id == "test-project-id"
        assert project.name == "Test Project"
        assert project.description == ""
        assert project.parent_project_id is None
        assert project.metadata == {}


class TestTraceStatus:
    """Tests for TraceStatus enum."""

    def test_status_values(self) -> None:
        """Test TraceStatus enum values."""
        assert TraceStatus.PENDING.value == "pending"
        assert TraceStatus.SUCCESS.value == "success"
        assert TraceStatus.ERROR.value == "error"
        assert TraceStatus.CANCELLED.value == "cancelled"


class TestTraceFilter:
    """Tests for TraceFilter dataclass."""

    def test_default_filter(self) -> None:
        """Test default filter values."""
        filter = TraceFilter()
        assert filter.session_id is None
        assert filter.project_id is None
        assert filter.status is None
        assert filter.limit == 100
        assert filter.offset == 0


class TestProjectFilter:
    """Tests for ProjectFilter dataclass."""

    def test_default_filter(self) -> None:
        """Test default filter values."""
        filter = ProjectFilter()
        assert filter.name_contains is None
        assert filter.parent_project_id is None
        assert filter.limit == 100
        assert filter.offset == 0


class TestSQLiteTraceStore:
    """Tests for SQLiteTraceStore."""

    @pytest.fixture
    def db_path(self) -> Path:
        """Create temporary database path."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            return Path(f.name)

    @pytest.fixture
    async def store(self, db_path: Path) -> SQLiteTraceStore:
        """Create SQLiteTraceStore instance."""
        store = SQLiteTraceStore(db_path)
        yield store
        await store.close()
        db_path.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_save_and_load_trace(self, store: SQLiteTraceStore) -> None:
        """Test saving and loading a trace."""
        trace = TraceData(
            trace_id="trace-1",
            session_id="session-1",
            project_id="project-1",
            prompt="Hello",
            response="Hi there",
            status=TraceStatus.SUCCESS,
        )

        await store.save(trace)
        loaded = await store.load("trace-1")

        assert loaded is not None
        assert loaded.trace_id == "trace-1"
        assert loaded.session_id == "session-1"
        assert loaded.project_id == "project-1"
        assert loaded.prompt == "Hello"
        assert loaded.response == "Hi there"
        assert loaded.status == TraceStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_load_nonexistent_trace(self, store: SQLiteTraceStore) -> None:
        """Test loading a trace that doesn't exist."""
        loaded = await store.load("nonexistent")
        assert loaded is None

    @pytest.mark.asyncio
    async def test_delete_trace(self, store: SQLiteTraceStore) -> None:
        """Test deleting a trace."""
        trace = TraceData(trace_id="trace-1", session_id="session-1")
        await store.save(trace)

        deleted = await store.delete("trace-1")
        assert deleted is True

        loaded = await store.load("trace-1")
        assert loaded is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_trace(self, store: SQLiteTraceStore) -> None:
        """Test deleting a trace that doesn't exist."""
        deleted = await store.delete("nonexistent")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_list_traces_with_filter(self, store: SQLiteTraceStore) -> None:
        """Test listing traces with filter."""
        # Create multiple traces
        for i in range(5):
            trace = TraceData(
                trace_id=f"trace-{i}",
                session_id="session-1" if i < 3 else "session-2",
            )
            await store.save(trace)

        # Filter by session_id
        filter = TraceFilter(session_id="session-1")
        traces = await store.list(filter)
        assert len(traces) == 3

    @pytest.mark.asyncio
    async def test_get_session_traces(self, store: SQLiteTraceStore) -> None:
        """Test getting traces for a session."""
        for i in range(3):
            trace = TraceData(
                trace_id=f"trace-{i}",
                session_id="session-1",
                sequence_number=i,
            )
            await store.save(trace)

        traces = await store.get_session_traces("session-1")
        assert len(traces) == 3
        # Should be ordered by sequence_number
        assert traces[0].sequence_number == 0
        assert traces[1].sequence_number == 1
        assert traces[2].sequence_number == 2

    @pytest.mark.asyncio
    async def test_get_project_traces(self, store: SQLiteTraceStore) -> None:
        """Test getting traces for a project."""
        for i in range(3):
            trace = TraceData(
                trace_id=f"trace-{i}",
                session_id=f"session-{i}",
                project_id="project-1",
            )
            await store.save(trace)

        traces = await store.get_project_traces("project-1")
        assert len(traces) == 3

    @pytest.mark.asyncio
    async def test_save_trace_with_tool_calls(self, store: SQLiteTraceStore) -> None:
        """Test saving trace with tool calls."""
        trace = TraceData(
            trace_id="trace-1",
            session_id="session-1",
            tool_calls=[
                {"name": "Read", "args": {"file": "test.py"}, "result": "content"},
                {"name": "Write", "args": {"file": "out.py"}},
            ],
        )

        await store.save(trace)
        loaded = await store.load("trace-1")

        assert loaded is not None
        assert len(loaded.tool_calls) == 2
        assert loaded.tool_calls[0]["name"] == "Read"
        assert loaded.tool_calls[1]["name"] == "Write"

    @pytest.mark.asyncio
    async def test_save_trace_with_message_stream(self, store: SQLiteTraceStore) -> None:
        """Test saving trace with message stream detail."""
        trace = TraceData(
            trace_id="trace-1",
            session_id="session-1",
            message_stream=[
                {"type": "TextMessage", "timestamp_ms": 100, "data": {"content": "Hi"}},
                {"type": "StopMessage", "timestamp_ms": 200, "data": {}},
            ],
        )

        await store.save(trace)
        loaded = await store.load("trace-1")

        assert loaded is not None
        assert loaded.has_detail
        assert len(loaded.message_stream) == 2


class TestSQLiteProjectStore:
    """Tests for SQLiteProjectStore."""

    @pytest.fixture
    def db_path(self) -> Path:
        """Create temporary database path."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            return Path(f.name)

    @pytest.fixture
    async def store(self, db_path: Path) -> SQLiteProjectStore:
        """Create SQLiteProjectStore instance."""
        store = SQLiteProjectStore(db_path)
        yield store
        await store.close()
        db_path.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_save_and_load_project(self, store: SQLiteProjectStore) -> None:
        """Test saving and loading a project."""
        project = ProjectData(
            project_id="project-1",
            name="Test Project",
            description="A test project",
        )

        await store.save(project)
        loaded = await store.load("project-1")

        assert loaded is not None
        assert loaded.project_id == "project-1"
        assert loaded.name == "Test Project"
        assert loaded.description == "A test project"

    @pytest.mark.asyncio
    async def test_load_nonexistent_project(self, store: SQLiteProjectStore) -> None:
        """Test loading a project that doesn't exist."""
        loaded = await store.load("nonexistent")
        assert loaded is None

    @pytest.mark.asyncio
    async def test_delete_project(self, store: SQLiteProjectStore) -> None:
        """Test deleting a project."""
        project = ProjectData(project_id="project-1", name="Test")
        await store.save(project)

        deleted = await store.delete("project-1")
        assert deleted is True

        loaded = await store.load("project-1")
        assert loaded is None

    @pytest.mark.asyncio
    async def test_list_projects(self, store: SQLiteProjectStore) -> None:
        """Test listing projects."""
        for i in range(3):
            project = ProjectData(
                project_id=f"project-{i}",
                name=f"Project {i}",
            )
            await store.save(project)

        projects = await store.list()
        assert len(projects) == 3

    @pytest.mark.asyncio
    async def test_list_projects_with_name_filter(
        self, store: SQLiteProjectStore
    ) -> None:
        """Test listing projects with name filter."""
        await store.save(ProjectData(project_id="p1", name="Alpha Project"))
        await store.save(ProjectData(project_id="p2", name="Beta Project"))
        await store.save(ProjectData(project_id="p3", name="Gamma Test"))

        filter = ProjectFilter(name_contains="Project")
        projects = await store.list(filter)
        assert len(projects) == 2

    @pytest.mark.asyncio
    async def test_get_subprojects(self, store: SQLiteProjectStore) -> None:
        """Test getting child projects."""
        parent = ProjectData(project_id="parent", name="Parent")
        child1 = ProjectData(project_id="child1", name="Child 1", parent_project_id="parent")
        child2 = ProjectData(project_id="child2", name="Child 2", parent_project_id="parent")
        unrelated = ProjectData(project_id="other", name="Other")

        await store.save(parent)
        await store.save(child1)
        await store.save(child2)
        await store.save(unrelated)

        subprojects = await store.get_subprojects("parent")
        assert len(subprojects) == 2


class TestProject:
    """Tests for Project class."""

    @pytest.fixture
    def db_path(self) -> Path:
        """Create temporary database path."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            return Path(f.name)

    @pytest.fixture
    async def stores(self, db_path: Path):
        """Create all stores."""
        project_store = SQLiteProjectStore(db_path)
        trace_store = SQLiteTraceStore(db_path)
        session_store = SQLiteSessionStore(db_path)
        yield project_store, trace_store, session_store
        await project_store.close()
        await trace_store.close()
        await session_store.close()
        db_path.unlink(missing_ok=True)

    def test_create_project(self) -> None:
        """Test creating a project."""
        project = Project(name="Test Project", description="Test description")

        assert project.name == "Test Project"
        assert project.description == "Test description"
        assert project.project_id is not None
        assert project.parent_project_id is None

    def test_create_project_with_id(self) -> None:
        """Test creating a project with specific ID."""
        project = Project(project_id="custom-id", name="Test")
        assert project.project_id == "custom-id"

    def test_create_subproject(self) -> None:
        """Test creating a sub-project."""
        parent = Project(name="Parent")
        child = parent.create_subproject("Child", "Child description")

        assert child.name == "Child"
        assert child.description == "Child description"
        assert child.parent_project_id == parent.project_id

    def test_create_session(self) -> None:
        """Test creating a tracing session from project."""
        project = Project(name="Test")
        session = project.create_session(
            options=CLIAgentOptions(model="sonnet"),
            detailed=True,
        )

        assert isinstance(session, TracingSession)
        assert session.project_id == project.project_id

    @pytest.mark.asyncio
    async def test_save_and_load_project(self, stores) -> None:
        """Test saving and loading a project."""
        project_store, trace_store, session_store = stores

        project = Project(
            name="Test Project",
            description="Test",
            store=project_store,
            trace_store=trace_store,
            session_store=session_store,
        )

        await project.save()

        loaded = await Project.load(
            project.project_id,
            project_store,
            trace_store=trace_store,
            session_store=session_store,
        )

        assert loaded is not None
        assert loaded.name == "Test Project"
        assert loaded.project_id == project.project_id

    def test_to_data(self) -> None:
        """Test converting project to ProjectData."""
        project = Project(
            project_id="test-id",
            name="Test",
            description="Test desc",
            metadata={"key": "value"},
        )

        data = project.to_data()

        assert isinstance(data, ProjectData)
        assert data.project_id == "test-id"
        assert data.name == "Test"
        assert data.description == "Test desc"
        assert data.metadata == {"key": "value"}

    def test_repr(self) -> None:
        """Test project repr."""
        project = Project(project_id="12345678-1234-1234-1234-123456789012", name="Test")
        repr_str = repr(project)
        assert "Project" in repr_str
        assert "Test" in repr_str


class TestTracingSession:
    """Tests for TracingSession."""

    def test_create_tracing_session(self) -> None:
        """Test creating a tracing session."""
        session = TracingSession(
            options=CLIAgentOptions(model="sonnet"),
            project_id="test-project",
            detailed=True,
        )

        assert session.project_id == "test-project"
        assert session._detailed is True
        assert session.last_trace is None
        assert session.trace_count == 0

    @pytest.mark.asyncio
    async def test_create_tracing_session_helper(self) -> None:
        """Test create_tracing_session helper function."""
        session = await create_tracing_session(
            project_id="test-project",
            options=CLIAgentOptions(model="sonnet"),
            detailed=True,
        )

        assert isinstance(session, TracingSession)
        assert session.project_id == "test-project"


class TestIntegration:
    """Integration tests for the tracing system."""

    @pytest.fixture
    def db_path(self) -> Path:
        """Create temporary database path."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            return Path(f.name)

    @pytest.fixture
    async def project(self, db_path: Path):
        """Create project with all stores."""
        project_store = SQLiteProjectStore(db_path)
        trace_store = SQLiteTraceStore(db_path)
        session_store = SQLiteSessionStore(db_path)

        project = Project(
            name="Integration Test",
            store=project_store,
            trace_store=trace_store,
            session_store=session_store,
        )
        await project.save()

        yield project

        await project_store.close()
        await trace_store.close()
        await session_store.close()
        db_path.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_project_hierarchy(self, project: Project) -> None:
        """Test creating project hierarchy."""
        # Create sub-projects
        chapter1 = project.create_subproject("Chapter 1", "Introduction")
        chapter2 = project.create_subproject("Chapter 2", "Methods")

        await chapter1.save()
        await chapter2.save()

        # Get subprojects
        subprojects = await project.get_subprojects()
        assert len(subprojects) == 2
        names = {sp.name for sp in subprojects}
        assert names == {"Chapter 1", "Chapter 2"}

    @pytest.mark.asyncio
    async def test_trace_summary(self, project: Project) -> None:
        """Test getting trace summary for a project."""
        # Save some traces directly
        trace_store = project._trace_store
        for i in range(3):
            trace = TraceData(
                trace_id=f"trace-{i}",
                session_id=f"session-{i}",
                project_id=project.project_id,
                input_tokens=100,
                output_tokens=200,
                thinking_tokens=50,
                cost_usd=0.01,
                duration_ms=1000,
                status=TraceStatus.SUCCESS,
            )
            await trace_store.save(trace)

        summary = await project.get_trace_summary()

        assert summary["total_traces"] == 3
        assert summary["input_tokens"] == 300
        assert summary["output_tokens"] == 600
        assert summary["thinking_tokens"] == 150
        assert summary["total_tokens"] == 1050
        assert summary["total_cost_usd"] == 0.03
        assert summary["success_count"] == 3
        assert summary["error_count"] == 0
