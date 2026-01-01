"""Tests for SessionManager."""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from ccflow.manager import SessionManager, get_manager, init_manager
from ccflow.store import SessionState, SessionStatus
from ccflow.stores import MemorySessionStore
from ccflow.types import CLIAgentOptions


@pytest.fixture
def memory_store() -> MemorySessionStore:
    """Create memory store for testing."""
    return MemorySessionStore()


@pytest.fixture
async def manager(memory_store: MemorySessionStore) -> SessionManager:
    """Create session manager with memory store."""
    mgr = SessionManager(store=memory_store, auto_cleanup=False)
    yield mgr
    await mgr.stop()


@pytest.fixture
async def started_manager(memory_store: MemorySessionStore) -> SessionManager:
    """Create and start session manager."""
    mgr = SessionManager(store=memory_store, auto_cleanup=False)
    await mgr.start()
    yield mgr
    await mgr.stop()


class TestSessionManagerInit:
    """Tests for SessionManager initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default values."""
        manager = SessionManager(auto_cleanup=False)
        assert manager._auto_cleanup is False
        assert manager.active_session_count == 0

    def test_init_with_custom_store(self, memory_store: MemorySessionStore):
        """Test initialization with custom store."""
        manager = SessionManager(store=memory_store, auto_cleanup=False)
        assert manager.store is memory_store

    def test_init_with_custom_ttl(self, memory_store: MemorySessionStore):
        """Test initialization with custom TTL."""
        ttl = timedelta(hours=12)
        manager = SessionManager(
            store=memory_store,
            session_ttl=ttl,
            auto_cleanup=False,
        )
        assert manager._session_ttl == ttl

    async def test_start_stop(self, memory_store: MemorySessionStore):
        """Test start and stop lifecycle."""
        manager = SessionManager(store=memory_store, auto_cleanup=False)

        await manager.start()
        assert manager._running is True

        await manager.stop()
        assert manager._running is False

    async def test_context_manager(self, memory_store: MemorySessionStore):
        """Test async context manager."""
        async with SessionManager(store=memory_store, auto_cleanup=False) as manager:
            assert manager._running is True
        assert manager._running is False


class TestSessionCreation:
    """Tests for session creation."""

    async def test_create_session(self, started_manager: SessionManager):
        """Test creating a new session."""
        session = await started_manager.create_session()

        assert session is not None
        assert session.session_id in started_manager.active_sessions
        assert started_manager.active_session_count == 1

    async def test_create_session_with_id(self, started_manager: SessionManager):
        """Test creating session with specific ID."""
        session = await started_manager.create_session(session_id="custom-id-123")
        assert session.session_id == "custom-id-123"

    async def test_create_session_with_options(self, started_manager: SessionManager):
        """Test creating session with options."""
        opts = CLIAgentOptions(model="opus", system_prompt="Be concise")
        session = await started_manager.create_session(options=opts)

        assert session._options.model == "opus"
        assert session._options.system_prompt == "Be concise"

    async def test_create_session_with_tags(self, started_manager: SessionManager):
        """Test creating session with tags."""
        session = await started_manager.create_session(tags=["test", "important"])
        assert session.tags == ["test", "important"]

    async def test_create_session_persisted(
        self, started_manager: SessionManager, memory_store: MemorySessionStore
    ):
        """Test that created session is persisted."""
        session = await started_manager.create_session()

        # Verify persisted to store
        assert await memory_store.exists(session.session_id)


class TestSessionRetrieval:
    """Tests for session retrieval."""

    async def test_get_session_active(self, started_manager: SessionManager):
        """Test getting an active session."""
        session = await started_manager.create_session()
        retrieved = await started_manager.get_session(session.session_id)

        assert retrieved is session  # Same instance

    async def test_get_session_from_store(
        self, started_manager: SessionManager, memory_store: MemorySessionStore
    ):
        """Test getting session from store when not active."""
        # Create and save a session state directly
        state = SessionState(
            session_id="stored-session",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            status=SessionStatus.ACTIVE,
            model="sonnet",
            turn_count=5,
            total_input_tokens=100,
            total_output_tokens=150,
        )
        await memory_store.save(state)

        # Get session (not in active dict)
        session = await started_manager.get_session("stored-session")

        assert session is not None
        assert session.session_id == "stored-session"
        assert session._turn_count == 5

    async def test_get_session_not_found(self, started_manager: SessionManager):
        """Test getting non-existent session."""
        session = await started_manager.get_session("nonexistent")
        assert session is None

    async def test_load_session(
        self, started_manager: SessionManager, memory_store: MemorySessionStore
    ):
        """Test loading session from store."""
        state = SessionState(
            session_id="load-test",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            status=SessionStatus.ACTIVE,
            model="opus",
            turn_count=3,
            total_input_tokens=50,
            total_output_tokens=75,
        )
        await memory_store.save(state)

        session = await started_manager.load_session("load-test")

        assert session is not None
        assert session.session_id == "load-test"
        assert "load-test" in started_manager.active_sessions


class TestSessionClosing:
    """Tests for session closing and deletion."""

    async def test_close_session(self, started_manager: SessionManager):
        """Test closing a session."""
        session = await started_manager.create_session()
        session_id = session.session_id

        result = await started_manager.close_session(session_id)

        assert result is True
        assert session_id not in started_manager.active_sessions
        assert session.is_closed

    async def test_close_session_not_active(
        self, started_manager: SessionManager, memory_store: MemorySessionStore
    ):
        """Test closing session not in active dict."""
        state = SessionState(
            session_id="inactive-session",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            status=SessionStatus.ACTIVE,
            model="sonnet",
            turn_count=1,
            total_input_tokens=10,
            total_output_tokens=15,
        )
        await memory_store.save(state)

        result = await started_manager.close_session("inactive-session")
        assert result is True

        # Verify status updated in store
        loaded = await memory_store.load("inactive-session")
        assert loaded is not None
        assert loaded.status == SessionStatus.CLOSED

    async def test_close_session_not_found(self, started_manager: SessionManager):
        """Test closing non-existent session."""
        result = await started_manager.close_session("nonexistent")
        assert result is False

    async def test_delete_session(self, started_manager: SessionManager):
        """Test deleting a session."""
        session = await started_manager.create_session()
        session_id = session.session_id

        result = await started_manager.delete_session(session_id)

        assert result is True
        assert session_id not in started_manager.active_sessions
        assert not await started_manager.session_exists(session_id)

    async def test_delete_session_not_found(self, started_manager: SessionManager):
        """Test deleting non-existent session."""
        result = await started_manager.delete_session("nonexistent")
        assert result is False


class TestSessionDiscovery:
    """Tests for session discovery and listing."""

    async def test_list_sessions(
        self, started_manager: SessionManager, memory_store: MemorySessionStore
    ):
        """Test listing sessions."""
        # Create multiple sessions
        for i in range(5):
            state = SessionState(
                session_id=f"session-{i}",
                created_at=datetime.now() - timedelta(hours=i),
                updated_at=datetime.now() - timedelta(minutes=i),
                status=SessionStatus.ACTIVE if i % 2 == 0 else SessionStatus.CLOSED,
                model="sonnet" if i < 3 else "opus",
                turn_count=i + 1,
                total_input_tokens=100 * i,
                total_output_tokens=150 * i,
            )
            await memory_store.save(state)

        sessions = await started_manager.list_sessions()
        assert len(sessions) == 5

    async def test_list_sessions_with_status(
        self, started_manager: SessionManager, memory_store: MemorySessionStore
    ):
        """Test listing sessions by status."""
        for i in range(4):
            state = SessionState(
                session_id=f"status-{i}",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                status=SessionStatus.ACTIVE if i % 2 == 0 else SessionStatus.CLOSED,
                model="sonnet",
                turn_count=1,
                total_input_tokens=10,
                total_output_tokens=15,
            )
            await memory_store.save(state)

        active = await started_manager.list_sessions(status=SessionStatus.ACTIVE)
        assert len(active) == 2

    async def test_list_active_sessions(
        self, started_manager: SessionManager, memory_store: MemorySessionStore
    ):
        """Test listing active sessions convenience method."""
        for i in range(3):
            state = SessionState(
                session_id=f"active-{i}",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                status=SessionStatus.ACTIVE,
                model="sonnet",
                turn_count=1,
                total_input_tokens=10,
                total_output_tokens=15,
            )
            await memory_store.save(state)

        active = await started_manager.list_active_sessions()
        assert len(active) == 3

    async def test_list_recent_sessions(
        self, started_manager: SessionManager, memory_store: MemorySessionStore
    ):
        """Test listing recent sessions."""
        # Create sessions at different times
        for i in range(5):
            state = SessionState(
                session_id=f"recent-{i}",
                created_at=datetime.now() - timedelta(hours=i * 6),
                updated_at=datetime.now() - timedelta(hours=i * 6),
                status=SessionStatus.ACTIVE,
                model="sonnet",
                turn_count=1,
                total_input_tokens=10,
                total_output_tokens=15,
            )
            await memory_store.save(state)

        recent = await started_manager.list_recent_sessions(hours=12, limit=10)
        assert len(recent) >= 2  # At least sessions from last 12 hours

    async def test_search_by_tags(
        self, started_manager: SessionManager, memory_store: MemorySessionStore
    ):
        """Test searching by tags."""
        for i, tags in enumerate([["test"], ["prod"], ["test", "important"]]):
            state = SessionState(
                session_id=f"tagged-{i}",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                status=SessionStatus.ACTIVE,
                model="sonnet",
                turn_count=1,
                total_input_tokens=10,
                total_output_tokens=15,
                tags=tags,
            )
            await memory_store.save(state)

        results = await started_manager.search_by_tags(["test"])
        assert len(results) == 2

    async def test_count_sessions(
        self, started_manager: SessionManager, memory_store: MemorySessionStore
    ):
        """Test counting sessions."""
        for i in range(5):
            state = SessionState(
                session_id=f"count-{i}",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                status=SessionStatus.ACTIVE,
                model="sonnet",
                turn_count=1,
                total_input_tokens=10,
                total_output_tokens=15,
            )
            await memory_store.save(state)

        count = await started_manager.count_sessions()
        assert count == 5

    async def test_session_exists(self, started_manager: SessionManager):
        """Test checking session existence."""
        session = await started_manager.create_session()

        assert await started_manager.session_exists(session.session_id)
        assert not await started_manager.session_exists("nonexistent")


class TestCleanup:
    """Tests for cleanup functionality."""

    async def test_cleanup_expired(
        self, memory_store: MemorySessionStore
    ):
        """Test cleaning up expired sessions."""
        # Create sessions with different ages
        for i in range(3):
            state = SessionState(
                session_id=f"old-{i}",
                created_at=datetime.now() - timedelta(days=10),
                updated_at=datetime.now() - timedelta(days=10),
                status=SessionStatus.ACTIVE,
                model="sonnet",
                turn_count=1,
                total_input_tokens=10,
                total_output_tokens=15,
            )
            await memory_store.save(state)

        manager = SessionManager(
            store=memory_store,
            session_ttl=timedelta(days=7),
            auto_cleanup=False,
        )

        deleted = await manager.cleanup_expired()
        assert deleted == 3

    async def test_cleanup_closed(
        self, memory_store: MemorySessionStore
    ):
        """Test cleaning up closed sessions."""
        for i in range(3):
            state = SessionState(
                session_id=f"closed-{i}",
                created_at=datetime.now() - timedelta(days=10),
                updated_at=datetime.now() - timedelta(days=10),
                status=SessionStatus.CLOSED,
                model="sonnet",
                turn_count=1,
                total_input_tokens=10,
                total_output_tokens=15,
            )
            await memory_store.save(state)

        manager = SessionManager(
            store=memory_store,
            session_ttl=timedelta(days=7),
            auto_cleanup=False,
        )

        deleted = await manager.cleanup_closed()
        assert deleted == 3


class TestAutoCleanup:
    """Tests for automatic cleanup."""

    async def test_auto_cleanup_starts(self, memory_store: MemorySessionStore):
        """Test that auto cleanup task starts."""
        manager = SessionManager(
            store=memory_store,
            auto_cleanup=True,
            cleanup_interval=timedelta(seconds=1),
        )

        await manager.start()
        assert manager._cleanup_task is not None

        await manager.stop()
        assert manager._cleanup_task is None


class TestStatistics:
    """Tests for statistics."""

    async def test_get_stats(
        self, started_manager: SessionManager, memory_store: MemorySessionStore
    ):
        """Test getting statistics."""
        # Create some sessions
        for i in range(5):
            state = SessionState(
                session_id=f"stat-{i}",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                status=SessionStatus.ACTIVE if i < 3 else SessionStatus.CLOSED,
                model="sonnet",
                turn_count=1,
                total_input_tokens=10,
                total_output_tokens=15,
            )
            await memory_store.save(state)

        stats = await started_manager.get_stats()

        assert stats["total_sessions"] == 5
        assert stats["active_sessions"] == 3
        assert stats["closed_sessions"] == 2


class TestModuleLevelFunctions:
    """Tests for module-level functions."""

    def test_get_manager(self):
        """Test get_manager returns same instance."""
        # Reset global
        import ccflow.manager as mgr_module
        mgr_module._default_manager = None

        manager1 = get_manager()
        manager2 = get_manager()

        assert manager1 is manager2

        # Cleanup
        mgr_module._default_manager = None

    async def test_init_manager(self, memory_store: MemorySessionStore):
        """Test init_manager creates and starts manager."""
        import ccflow.manager as mgr_module
        mgr_module._default_manager = None

        manager = await init_manager(store=memory_store, auto_cleanup=False)

        assert manager._running is True
        assert manager.store is memory_store

        await manager.stop()
        mgr_module._default_manager = None


class TestStopBehavior:
    """Tests for stop behavior."""

    async def test_stop_closes_active_sessions(self, memory_store: MemorySessionStore):
        """Test that stop closes all active sessions."""
        manager = SessionManager(store=memory_store, auto_cleanup=False)
        await manager.start()

        session1 = await manager.create_session()
        session2 = await manager.create_session()

        await manager.stop()

        assert session1.is_closed
        assert session2.is_closed
        assert manager.active_session_count == 0

    async def test_stop_handles_close_error(self, memory_store: MemorySessionStore):
        """Test that stop handles session close errors gracefully."""
        manager = SessionManager(store=memory_store, auto_cleanup=False)
        await manager.start()

        session = await manager.create_session()

        # Mock close to raise
        with patch.object(session, "close", side_effect=Exception("Close error")):
            # Should not raise
            await manager.stop()
