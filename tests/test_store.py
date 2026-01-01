"""Tests for session store protocol and implementations."""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from ccflow.exceptions import SessionStoreError
from ccflow.store import (
    BaseSessionStore,
    SessionFilter,
    SessionMetadata,
    SessionState,
    SessionStatus,
)
from ccflow.stores import MemorySessionStore, SQLiteSessionStore


# Fixtures


@pytest.fixture
def sample_session_state() -> SessionState:
    """Create sample session state for testing."""
    now = datetime.now()
    return SessionState(
        session_id="test-session-001",
        created_at=now,
        updated_at=now,
        status=SessionStatus.ACTIVE,
        model="sonnet",
        system_prompt="You are a helpful assistant.",
        turn_count=3,
        total_input_tokens=150,
        total_output_tokens=200,
        total_cost_usd=0.005,
        messages_hash="abc123",
        last_prompt="What is Python?",
        last_response="Python is a programming language...",
        tags=["test", "development"],
        metadata={"source": "test"},
    )


@pytest.fixture
def sample_sessions() -> list[SessionState]:
    """Create multiple sample sessions for testing."""
    now = datetime.now()
    sessions = []

    for i in range(5):
        session = SessionState(
            session_id=f"test-session-{i:03d}",
            created_at=now - timedelta(hours=i),
            updated_at=now - timedelta(minutes=i * 10),
            status=SessionStatus.ACTIVE if i % 2 == 0 else SessionStatus.CLOSED,
            model="sonnet" if i < 3 else "opus",
            turn_count=i + 1,
            total_input_tokens=100 * (i + 1),
            total_output_tokens=150 * (i + 1),
            tags=[f"tag-{i}"] if i % 2 == 0 else [],
        )
        sessions.append(session)

    return sessions


@pytest.fixture
def memory_store() -> MemorySessionStore:
    """Create memory store for testing."""
    return MemorySessionStore()


@pytest.fixture
async def sqlite_store(tmp_path: Path) -> SQLiteSessionStore:
    """Create SQLite store with temp database."""
    db_path = tmp_path / "test_sessions.db"
    store = SQLiteSessionStore(db_path=db_path)
    yield store
    await store.close()


# SessionState tests


class TestSessionState:
    """Tests for SessionState dataclass."""

    def test_total_tokens(self, sample_session_state: SessionState):
        """Test total_tokens property."""
        assert sample_session_state.total_tokens == 350

    def test_to_metadata(self, sample_session_state: SessionState):
        """Test conversion to metadata."""
        metadata = sample_session_state.to_metadata()

        assert isinstance(metadata, SessionMetadata)
        assert metadata.session_id == sample_session_state.session_id
        assert metadata.status == sample_session_state.status
        assert metadata.model == sample_session_state.model
        assert metadata.turn_count == sample_session_state.turn_count
        assert metadata.tags == sample_session_state.tags

    def test_to_dict(self, sample_session_state: SessionState):
        """Test serialization to dict."""
        data = sample_session_state.to_dict()

        assert data["session_id"] == "test-session-001"
        assert data["status"] == "active"
        assert data["model"] == "sonnet"
        assert isinstance(data["created_at"], str)

    def test_from_dict(self, sample_session_state: SessionState):
        """Test deserialization from dict."""
        data = sample_session_state.to_dict()
        restored = SessionState.from_dict(data)

        assert restored.session_id == sample_session_state.session_id
        assert restored.status == sample_session_state.status
        assert restored.turn_count == sample_session_state.turn_count
        assert restored.tags == sample_session_state.tags

    def test_compute_hash(self, sample_session_state: SessionState):
        """Test hash computation."""
        hash1 = sample_session_state.compute_hash("test content")
        hash2 = sample_session_state.compute_hash("test content")
        hash3 = sample_session_state.compute_hash("different content")

        assert hash1 == hash2  # Same content = same hash
        assert hash1 != hash3  # Different content = different hash
        assert len(hash1) == 16  # Truncated to 16 chars

    def test_update_hash(self, sample_session_state: SessionState):
        """Test hash update with new turn."""
        original_hash = sample_session_state.messages_hash

        sample_session_state.update_hash("New prompt", "New response")

        assert sample_session_state.messages_hash != original_hash
        assert sample_session_state.last_prompt == "New prompt"
        assert sample_session_state.last_response == "New response"


class TestSessionMetadata:
    """Tests for SessionMetadata dataclass."""

    def test_total_tokens(self):
        """Test total_tokens property."""
        metadata = SessionMetadata(
            session_id="test",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            status=SessionStatus.ACTIVE,
            model="sonnet",
            turn_count=1,
            total_input_tokens=100,
            total_output_tokens=200,
        )
        assert metadata.total_tokens == 300

    def test_duration(self):
        """Test duration property."""
        now = datetime.now()
        created = now - timedelta(hours=2)

        metadata = SessionMetadata(
            session_id="test",
            created_at=created,
            updated_at=now,
            status=SessionStatus.ACTIVE,
            model="sonnet",
            turn_count=1,
            total_input_tokens=100,
            total_output_tokens=200,
        )

        assert metadata.duration >= timedelta(hours=1, minutes=59)


class TestSessionFilter:
    """Tests for SessionFilter dataclass."""

    def test_defaults(self):
        """Test default filter values."""
        f = SessionFilter()

        assert f.status is None
        assert f.model is None
        assert f.limit == 100
        assert f.offset == 0
        assert f.order_by == "updated_at"
        assert f.order_desc is True


# MemorySessionStore tests


class TestMemorySessionStore:
    """Tests for MemorySessionStore."""

    async def test_save_and_load(
        self, memory_store: MemorySessionStore, sample_session_state: SessionState
    ):
        """Test saving and loading session."""
        await memory_store.save(sample_session_state)
        loaded = await memory_store.load(sample_session_state.session_id)

        assert loaded is not None
        assert loaded.session_id == sample_session_state.session_id
        assert loaded.model == sample_session_state.model

    async def test_load_nonexistent(self, memory_store: MemorySessionStore):
        """Test loading nonexistent session returns None."""
        loaded = await memory_store.load("nonexistent-id")
        assert loaded is None

    async def test_delete(
        self, memory_store: MemorySessionStore, sample_session_state: SessionState
    ):
        """Test deleting session."""
        await memory_store.save(sample_session_state)
        assert await memory_store.exists(sample_session_state.session_id)

        deleted = await memory_store.delete(sample_session_state.session_id)
        assert deleted is True
        assert not await memory_store.exists(sample_session_state.session_id)

    async def test_delete_nonexistent(self, memory_store: MemorySessionStore):
        """Test deleting nonexistent session."""
        deleted = await memory_store.delete("nonexistent-id")
        assert deleted is False

    async def test_list_all(
        self, memory_store: MemorySessionStore, sample_sessions: list[SessionState]
    ):
        """Test listing all sessions."""
        for session in sample_sessions:
            await memory_store.save(session)

        result = await memory_store.list()
        assert len(result) == 5

    async def test_list_with_status_filter(
        self, memory_store: MemorySessionStore, sample_sessions: list[SessionState]
    ):
        """Test listing with status filter."""
        for session in sample_sessions:
            await memory_store.save(session)

        active = await memory_store.list(SessionFilter(status=SessionStatus.ACTIVE))
        closed = await memory_store.list(SessionFilter(status=SessionStatus.CLOSED))

        assert len(active) == 3  # Sessions 0, 2, 4
        assert len(closed) == 2  # Sessions 1, 3

    async def test_list_with_model_filter(
        self, memory_store: MemorySessionStore, sample_sessions: list[SessionState]
    ):
        """Test listing with model filter."""
        for session in sample_sessions:
            await memory_store.save(session)

        sonnet = await memory_store.list(SessionFilter(model="sonnet"))
        opus = await memory_store.list(SessionFilter(model="opus"))

        assert len(sonnet) == 3  # Sessions 0, 1, 2
        assert len(opus) == 2  # Sessions 3, 4

    async def test_list_with_pagination(
        self, memory_store: MemorySessionStore, sample_sessions: list[SessionState]
    ):
        """Test listing with pagination."""
        for session in sample_sessions:
            await memory_store.save(session)

        page1 = await memory_store.list(SessionFilter(limit=2, offset=0))
        page2 = await memory_store.list(SessionFilter(limit=2, offset=2))
        page3 = await memory_store.list(SessionFilter(limit=2, offset=4))

        assert len(page1) == 2
        assert len(page2) == 2
        assert len(page3) == 1

    async def test_list_with_date_filter(
        self, memory_store: MemorySessionStore, sample_sessions: list[SessionState]
    ):
        """Test listing with date filters."""
        for session in sample_sessions:
            await memory_store.save(session)

        now = datetime.now()
        recent = await memory_store.list(
            SessionFilter(created_after=now - timedelta(hours=2))
        )

        assert len(recent) >= 2

    async def test_list_with_turn_filter(
        self, memory_store: MemorySessionStore, sample_sessions: list[SessionState]
    ):
        """Test listing with turn count filters."""
        for session in sample_sessions:
            await memory_store.save(session)

        high_turns = await memory_store.list(SessionFilter(min_turns=3))
        low_turns = await memory_store.list(SessionFilter(max_turns=2))

        assert len(high_turns) == 3  # Sessions with 3, 4, 5 turns
        assert len(low_turns) == 2  # Sessions with 1, 2 turns

    async def test_list_with_tags_filter(
        self, memory_store: MemorySessionStore, sample_sessions: list[SessionState]
    ):
        """Test listing with tags filter."""
        for session in sample_sessions:
            await memory_store.save(session)

        tagged = await memory_store.list(SessionFilter(tags=["tag-0", "tag-2"]))
        assert len(tagged) == 2

    async def test_cleanup(
        self, memory_store: MemorySessionStore, sample_sessions: list[SessionState]
    ):
        """Test cleanup of old sessions."""
        for session in sample_sessions:
            await memory_store.save(session)

        # Clean up sessions older than 30 minutes
        deleted = await memory_store.cleanup(older_than=timedelta(minutes=30))
        assert deleted >= 1

    async def test_exists(
        self, memory_store: MemorySessionStore, sample_session_state: SessionState
    ):
        """Test exists check."""
        assert not await memory_store.exists(sample_session_state.session_id)

        await memory_store.save(sample_session_state)
        assert await memory_store.exists(sample_session_state.session_id)

    async def test_count(
        self, memory_store: MemorySessionStore, sample_sessions: list[SessionState]
    ):
        """Test count method."""
        for session in sample_sessions:
            await memory_store.save(session)

        total = await memory_store.count()
        active = await memory_store.count(SessionFilter(status=SessionStatus.ACTIVE))

        assert total == 5
        assert active == 3

    async def test_update_status(
        self, memory_store: MemorySessionStore, sample_session_state: SessionState
    ):
        """Test status update."""
        await memory_store.save(sample_session_state)

        updated = await memory_store.update_status(
            sample_session_state.session_id, SessionStatus.CLOSED
        )
        assert updated is True

        loaded = await memory_store.load(sample_session_state.session_id)
        assert loaded is not None
        assert loaded.status == SessionStatus.CLOSED

    async def test_update_status_nonexistent(self, memory_store: MemorySessionStore):
        """Test status update on nonexistent session."""
        updated = await memory_store.update_status("nonexistent", SessionStatus.ERROR)
        assert updated is False

    async def test_close(
        self, memory_store: MemorySessionStore, sample_session_state: SessionState
    ):
        """Test close clears store."""
        await memory_store.save(sample_session_state)
        assert memory_store.session_count == 1

        await memory_store.close()
        assert memory_store.session_count == 0

    async def test_clear(
        self, memory_store: MemorySessionStore, sample_session_state: SessionState
    ):
        """Test synchronous clear."""
        await memory_store.save(sample_session_state)
        assert memory_store.session_count == 1

        memory_store.clear()
        assert memory_store.session_count == 0

    async def test_deep_copy_on_save(
        self, memory_store: MemorySessionStore, sample_session_state: SessionState
    ):
        """Test that save creates deep copy."""
        await memory_store.save(sample_session_state)

        # Modify original
        sample_session_state.turn_count = 999

        # Load should have original value
        loaded = await memory_store.load(sample_session_state.session_id)
        assert loaded is not None
        assert loaded.turn_count == 3

    async def test_deep_copy_on_load(
        self, memory_store: MemorySessionStore, sample_session_state: SessionState
    ):
        """Test that load returns deep copy."""
        await memory_store.save(sample_session_state)

        loaded = await memory_store.load(sample_session_state.session_id)
        assert loaded is not None
        loaded.turn_count = 999

        # Original store should be unchanged
        loaded2 = await memory_store.load(sample_session_state.session_id)
        assert loaded2 is not None
        assert loaded2.turn_count == 3


# SQLiteSessionStore tests


class TestSQLiteSessionStore:
    """Tests for SQLiteSessionStore."""

    async def test_save_and_load(
        self, sqlite_store: SQLiteSessionStore, sample_session_state: SessionState
    ):
        """Test saving and loading session."""
        await sqlite_store.save(sample_session_state)
        loaded = await sqlite_store.load(sample_session_state.session_id)

        assert loaded is not None
        assert loaded.session_id == sample_session_state.session_id
        assert loaded.model == sample_session_state.model
        assert loaded.tags == sample_session_state.tags
        assert loaded.metadata == sample_session_state.metadata

    async def test_load_nonexistent(self, sqlite_store: SQLiteSessionStore):
        """Test loading nonexistent session returns None."""
        loaded = await sqlite_store.load("nonexistent-id")
        assert loaded is None

    async def test_update_existing(
        self, sqlite_store: SQLiteSessionStore, sample_session_state: SessionState
    ):
        """Test updating existing session."""
        await sqlite_store.save(sample_session_state)

        sample_session_state.turn_count = 10
        sample_session_state.updated_at = datetime.now()
        await sqlite_store.save(sample_session_state)

        loaded = await sqlite_store.load(sample_session_state.session_id)
        assert loaded is not None
        assert loaded.turn_count == 10

    async def test_delete(
        self, sqlite_store: SQLiteSessionStore, sample_session_state: SessionState
    ):
        """Test deleting session."""
        await sqlite_store.save(sample_session_state)
        assert await sqlite_store.exists(sample_session_state.session_id)

        deleted = await sqlite_store.delete(sample_session_state.session_id)
        assert deleted is True
        assert not await sqlite_store.exists(sample_session_state.session_id)

    async def test_delete_nonexistent(self, sqlite_store: SQLiteSessionStore):
        """Test deleting nonexistent session."""
        deleted = await sqlite_store.delete("nonexistent-id")
        assert deleted is False

    async def test_list_all(
        self, sqlite_store: SQLiteSessionStore, sample_sessions: list[SessionState]
    ):
        """Test listing all sessions."""
        for session in sample_sessions:
            await sqlite_store.save(session)

        result = await sqlite_store.list()
        assert len(result) == 5

    async def test_list_with_filters(
        self, sqlite_store: SQLiteSessionStore, sample_sessions: list[SessionState]
    ):
        """Test listing with various filters."""
        for session in sample_sessions:
            await sqlite_store.save(session)

        # Status filter
        active = await sqlite_store.list(SessionFilter(status=SessionStatus.ACTIVE))
        assert len(active) == 3

        # Model filter
        sonnet = await sqlite_store.list(SessionFilter(model="sonnet"))
        assert len(sonnet) == 3

        # Combined filters
        active_sonnet = await sqlite_store.list(
            SessionFilter(status=SessionStatus.ACTIVE, model="sonnet")
        )
        assert len(active_sonnet) == 2

    async def test_list_with_pagination(
        self, sqlite_store: SQLiteSessionStore, sample_sessions: list[SessionState]
    ):
        """Test listing with pagination."""
        for session in sample_sessions:
            await sqlite_store.save(session)

        page1 = await sqlite_store.list(SessionFilter(limit=2, offset=0))
        page2 = await sqlite_store.list(SessionFilter(limit=2, offset=2))

        assert len(page1) == 2
        assert len(page2) == 2
        assert page1[0].session_id != page2[0].session_id

    async def test_cleanup(
        self, sqlite_store: SQLiteSessionStore, sample_sessions: list[SessionState]
    ):
        """Test cleanup of old sessions."""
        for session in sample_sessions:
            await sqlite_store.save(session)

        # Clean up sessions older than 30 minutes
        deleted = await sqlite_store.cleanup(older_than=timedelta(minutes=30))
        assert deleted >= 1

    async def test_exists(
        self, sqlite_store: SQLiteSessionStore, sample_session_state: SessionState
    ):
        """Test exists check."""
        assert not await sqlite_store.exists(sample_session_state.session_id)

        await sqlite_store.save(sample_session_state)
        assert await sqlite_store.exists(sample_session_state.session_id)

    async def test_count(
        self, sqlite_store: SQLiteSessionStore, sample_sessions: list[SessionState]
    ):
        """Test count method."""
        for session in sample_sessions:
            await sqlite_store.save(session)

        total = await sqlite_store.count()
        active = await sqlite_store.count(SessionFilter(status=SessionStatus.ACTIVE))

        assert total == 5
        assert active == 3

    async def test_update_status(
        self, sqlite_store: SQLiteSessionStore, sample_session_state: SessionState
    ):
        """Test status update."""
        await sqlite_store.save(sample_session_state)

        updated = await sqlite_store.update_status(
            sample_session_state.session_id, SessionStatus.CLOSED
        )
        assert updated is True

        loaded = await sqlite_store.load(sample_session_state.session_id)
        assert loaded is not None
        assert loaded.status == SessionStatus.CLOSED

    async def test_update_status_nonexistent(self, sqlite_store: SQLiteSessionStore):
        """Test status update on nonexistent session."""
        updated = await sqlite_store.update_status("nonexistent", SessionStatus.ERROR)
        assert updated is False

    async def test_db_path_property(self, sqlite_store: SQLiteSessionStore, tmp_path: Path):
        """Test db_path property."""
        assert sqlite_store.db_path == tmp_path / "test_sessions.db"

    async def test_vacuum(
        self, sqlite_store: SQLiteSessionStore, sample_session_state: SessionState
    ):
        """Test vacuum operation."""
        await sqlite_store.save(sample_session_state)
        await sqlite_store.delete(sample_session_state.session_id)

        # Should not raise
        await sqlite_store.vacuum()

    async def test_close_and_reopen(self, tmp_path: Path, sample_session_state: SessionState):
        """Test data persists after close and reopen."""
        db_path = tmp_path / "test_persist.db"

        # Save and close
        store1 = SQLiteSessionStore(db_path=db_path)
        await store1.save(sample_session_state)
        await store1.close()

        # Reopen and verify
        store2 = SQLiteSessionStore(db_path=db_path)
        loaded = await store2.load(sample_session_state.session_id)
        await store2.close()

        assert loaded is not None
        assert loaded.session_id == sample_session_state.session_id

    async def test_sequential_writes(
        self, sqlite_store: SQLiteSessionStore, sample_sessions: list[SessionState]
    ):
        """Test sequential write operations."""
        # Save all sessions sequentially (SQLite doesn't handle concurrent writes well)
        for session in sample_sessions:
            await sqlite_store.save(session)

        # Verify all saved
        for session in sample_sessions:
            assert await sqlite_store.exists(session.session_id)

    async def test_context_manager(self, tmp_path: Path, sample_session_state: SessionState):
        """Test async context manager."""
        db_path = tmp_path / "test_context.db"

        async with SQLiteSessionStore(db_path=db_path) as store:
            await store.save(sample_session_state)
            loaded = await store.load(sample_session_state.session_id)
            assert loaded is not None


# BaseSessionStore tests


class TestBaseSessionStore:
    """Tests for BaseSessionStore default implementations."""

    async def test_exists_uses_load(
        self, memory_store: MemorySessionStore, sample_session_state: SessionState
    ):
        """Test exists uses load by default."""
        # MemorySessionStore overrides exists, but test base behavior
        await memory_store.save(sample_session_state)

        # Use the base class method directly
        result = await BaseSessionStore.exists(memory_store, sample_session_state.session_id)
        assert result is True

    async def test_count_uses_list(
        self, memory_store: MemorySessionStore, sample_sessions: list[SessionState]
    ):
        """Test count uses list by default."""
        for session in sample_sessions:
            await memory_store.save(session)

        # Use the base class method directly
        result = await BaseSessionStore.count(memory_store, None)
        assert result == 5

    async def test_update_status_loads_and_saves(
        self, memory_store: MemorySessionStore, sample_session_state: SessionState
    ):
        """Test update_status loads, updates, and saves."""
        await memory_store.save(sample_session_state)

        # Use the base class method directly
        result = await BaseSessionStore.update_status(
            memory_store, sample_session_state.session_id, SessionStatus.ERROR
        )
        assert result is True

        loaded = await memory_store.load(sample_session_state.session_id)
        assert loaded is not None
        assert loaded.status == SessionStatus.ERROR

    async def test_close_is_noop_by_default(self, memory_store: MemorySessionStore):
        """Test close is no-op by default."""
        # BaseSessionStore.close does nothing - should not raise
        await BaseSessionStore.close(memory_store)
