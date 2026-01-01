"""Tests for Session-SessionStore integration."""

from datetime import datetime
from unittest.mock import patch

import pytest

from ccflow.session import Session, load_session, resume_session
from ccflow.store import SessionState, SessionStatus
from ccflow.stores import MemorySessionStore
from ccflow.types import CLIAgentOptions


@pytest.fixture
def memory_store() -> MemorySessionStore:
    """Create memory store for testing."""
    return MemorySessionStore()


@pytest.fixture
def default_options() -> CLIAgentOptions:
    """Return default CLI agent options for tests."""
    return CLIAgentOptions(model="sonnet")


class TestSessionWithStore:
    """Tests for Session with store integration."""

    def test_session_with_store_init(self, memory_store: MemorySessionStore):
        """Test session initialization with store."""
        session = Session(store=memory_store)
        assert session._store is memory_store

    def test_session_without_store(self):
        """Test session works without store."""
        session = Session()
        assert session._store is None

    def test_to_state(self, memory_store: MemorySessionStore):
        """Test conversion to SessionState."""
        session = Session(
            session_id="test-123",
            options=CLIAgentOptions(model="opus", system_prompt="Be helpful"),
            store=memory_store,
        )
        session._turn_count = 5
        session._total_input_tokens = 100
        session._total_output_tokens = 200
        session._tags = ["test", "integration"]

        state = session.to_state()

        assert state.session_id == "test-123"
        assert state.model == "opus"
        assert state.system_prompt == "Be helpful"
        assert state.turn_count == 5
        assert state.total_input_tokens == 100
        assert state.total_output_tokens == 200
        assert state.status == SessionStatus.ACTIVE
        assert state.tags == ["test", "integration"]

    def test_to_state_closed(self, memory_store: MemorySessionStore):
        """Test to_state reflects closed status."""
        session = Session(store=memory_store)
        session._is_closed = True

        state = session.to_state()
        assert state.status == SessionStatus.CLOSED

    def test_tags_property(self, memory_store: MemorySessionStore):
        """Test tags property returns copy."""
        session = Session(store=memory_store)
        session._tags = ["tag1", "tag2"]

        tags = session.tags
        tags.append("tag3")

        # Original should be unchanged
        assert session._tags == ["tag1", "tag2"]

    def test_add_tag(self, memory_store: MemorySessionStore):
        """Test adding tags."""
        session = Session(store=memory_store)

        session.add_tag("test")
        assert "test" in session._tags

        # Adding same tag twice should not duplicate
        session.add_tag("test")
        assert session._tags.count("test") == 1

    def test_remove_tag(self, memory_store: MemorySessionStore):
        """Test removing tags."""
        session = Session(store=memory_store)
        session._tags = ["tag1", "tag2"]

        session.remove_tag("tag1")
        assert "tag1" not in session._tags
        assert "tag2" in session._tags

        # Removing non-existent tag should not error
        session.remove_tag("nonexistent")

    async def test_persist_saves_to_store(self, memory_store: MemorySessionStore):
        """Test _persist saves state to store."""
        session = Session(session_id="persist-test", store=memory_store)
        session._turn_count = 3

        await session._persist()

        # Verify saved to store
        assert await memory_store.exists("persist-test")
        loaded = await memory_store.load("persist-test")
        assert loaded is not None
        assert loaded.turn_count == 3

    async def test_persist_without_store(self):
        """Test _persist is no-op without store."""
        session = Session()  # No store
        # Should not raise
        await session._persist()

    async def test_persist_handles_error(self, memory_store: MemorySessionStore):
        """Test _persist handles store errors gracefully."""
        session = Session(store=memory_store)

        # Mock save to raise
        with patch.object(memory_store, "save", side_effect=Exception("Store error")):
            # Should not raise, just log warning
            await session._persist()

    async def test_close_persists_state(self, memory_store: MemorySessionStore):
        """Test close persists final state."""
        session = Session(session_id="close-test", store=memory_store)
        session._turn_count = 5

        await session.close()

        loaded = await memory_store.load("close-test")
        assert loaded is not None
        assert loaded.status == SessionStatus.CLOSED

    def test_update_hash(self, memory_store: MemorySessionStore):
        """Test hash update."""
        session = Session(store=memory_store)
        session._last_prompt = "Hello"
        session._last_response = "Hi there!"

        session._update_hash()

        assert len(session._messages_hash) == 16
        assert session._messages_hash != ""


class TestLoadSession:
    """Tests for load_session function."""

    async def test_load_session_success(self, memory_store: MemorySessionStore):
        """Test successfully loading session from store."""
        # Create and save a session state
        state = SessionState(
            session_id="load-test-123",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            status=SessionStatus.ACTIVE,
            model="opus",
            system_prompt="Be concise",
            turn_count=3,
            total_input_tokens=100,
            total_output_tokens=150,
            total_cost_usd=0.01,
            messages_hash="abc123",
            last_prompt="Previous question",
            last_response="Previous answer",
            tags=["important"],
        )
        await memory_store.save(state)

        # Load session
        session = await load_session("load-test-123", memory_store)

        assert session is not None
        assert session.session_id == "load-test-123"
        assert session._turn_count == 3
        assert session._total_input_tokens == 100
        assert session._total_output_tokens == 150
        assert session._messages_hash == "abc123"
        assert session._last_prompt == "Previous question"
        assert session._tags == ["important"]
        assert session._is_first_message is False

    async def test_load_session_not_found(self, memory_store: MemorySessionStore):
        """Test loading non-existent session returns None."""
        session = await load_session("nonexistent", memory_store)
        assert session is None

    async def test_load_session_with_options_override(
        self, memory_store: MemorySessionStore
    ):
        """Test load_session with options override."""
        state = SessionState(
            session_id="override-test",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            status=SessionStatus.ACTIVE,
            model="sonnet",
            system_prompt="Original prompt",
            turn_count=1,
            total_input_tokens=50,
            total_output_tokens=75,
        )
        await memory_store.save(state)

        # Load with different model
        opts = CLIAgentOptions(model="opus")
        session = await load_session("override-test", memory_store, options=opts)

        assert session is not None
        assert session._options.model == "opus"  # Override applied

    async def test_load_session_restores_prompts(self, memory_store: MemorySessionStore):
        """Test load_session restores prompts when not overridden."""
        state = SessionState(
            session_id="prompt-test",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            status=SessionStatus.ACTIVE,
            model="opus",
            system_prompt="Stored system prompt",
            append_system_prompt="Stored append prompt",
            turn_count=1,
            total_input_tokens=50,
            total_output_tokens=75,
        )
        await memory_store.save(state)

        # Load without specifying prompts - they should be restored
        session = await load_session("prompt-test", memory_store)

        assert session is not None
        assert session._options.system_prompt == "Stored system prompt"
        assert session._options.append_system_prompt == "Stored append prompt"


class TestResumeSession:
    """Tests for resume_session function."""

    async def test_resume_session_with_store(self, memory_store: MemorySessionStore):
        """Test resume_session with store."""
        session = await resume_session(
            "resume-test",
            options=CLIAgentOptions(model="sonnet"),
            store=memory_store,
        )

        assert session.session_id == "resume-test"
        assert session._store is memory_store
        assert session._is_first_message is False
        assert session._options.resume is True

    async def test_resume_session_without_store(self):
        """Test resume_session without store."""
        session = await resume_session("resume-test-2")

        assert session.session_id == "resume-test-2"
        assert session._store is None
        assert session._is_first_message is False


class TestSessionForkWithStore:
    """Tests for session forking with store."""

    async def test_fork_shares_store(self, memory_store: MemorySessionStore):
        """Test forked session shares parent's store."""
        parent = Session(store=memory_store)
        child = await parent.fork()

        assert child._store is memory_store
