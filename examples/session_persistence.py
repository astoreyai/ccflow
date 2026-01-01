#!/usr/bin/env python3
"""
Session Persistence Example

Demonstrates saving and restoring sessions using SQLite and in-memory stores.
"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import tempfile

from ccflow import Session, CLIAgentOptions, get_manager, init_manager
from ccflow.store import SessionState, SessionStatus, SessionFilter
from ccflow.stores.sqlite import SQLiteSessionStore
from ccflow.stores.memory import MemorySessionStore


async def demo_memory_store():
    """In-memory store for testing/ephemeral sessions."""
    print("\n1. In-Memory Session Store")
    print("-" * 40)

    store = MemorySessionStore()

    # Create a session state
    state = SessionState(
        session_id="mem-test-001",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        status=SessionStatus.ACTIVE,
        model="sonnet",
        turn_count=3,
        total_input_tokens=500,
        total_output_tokens=1500,
    )

    # Save
    await store.save(state)
    print(f"  Saved session: {state.session_id}")

    # Load
    loaded = await store.load(state.session_id)
    print(f"  Loaded session: {loaded.session_id if loaded else 'None'}")
    print(f"    Model: {loaded.model}")
    print(f"    Turns: {loaded.turn_count}")

    # Check existence
    exists = await store.exists(state.session_id)
    print(f"  Exists: {exists}")

    # List sessions
    sessions = await store.list()
    print(f"  Total sessions: {len(sessions)}")

    # Cleanup
    await store.close()
    print("  Store closed")


async def demo_sqlite_store():
    """SQLite store for persistent sessions."""
    print("\n2. SQLite Session Store")
    print("-" * 40)

    # Use temp file for demo
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    store = SQLiteSessionStore(db_path)
    print(f"  Database: {db_path}")

    # Create multiple sessions
    for i in range(3):
        state = SessionState(
            session_id=f"sqlite-test-{i:03d}",
            created_at=datetime.now() - timedelta(hours=i),
            updated_at=datetime.now() - timedelta(hours=i),
            status=SessionStatus.ACTIVE if i < 2 else SessionStatus.CLOSED,
            model="sonnet" if i % 2 == 0 else "haiku",
            turn_count=i + 1,
            total_input_tokens=100 * (i + 1),
            total_output_tokens=200 * (i + 1),
            tags=["demo", f"batch-{i}"],
        )
        await store.save(state)

    print(f"  Created 3 sessions")

    # List all
    all_sessions = await store.list()
    print(f"  All sessions: {len(all_sessions)}")

    # Filter by status
    active = await store.list(SessionFilter(status=SessionStatus.ACTIVE))
    print(f"  Active sessions: {len(active)}")

    # Filter by model
    sonnet_sessions = await store.list(SessionFilter(model="sonnet"))
    print(f"  Sonnet sessions: {len(sonnet_sessions)}")

    # Filter by tags
    tagged = await store.list(SessionFilter(tags=["batch-1"]))
    print(f"  Tagged 'batch-1': {len(tagged)}")

    # Count
    count = await store.count()
    print(f"  Total count: {count}")

    # Update status
    updated = await store.update_status("sqlite-test-000", SessionStatus.EXPIRED)
    print(f"  Updated status: {updated}")

    # Cleanup old sessions
    deleted = await store.cleanup(older_than=timedelta(hours=2))
    print(f"  Cleaned up: {deleted} sessions")

    # Close store
    await store.close()
    print("  Store closed")

    # Cleanup temp file
    Path(db_path).unlink()


async def demo_session_manager():
    """Session manager with automatic persistence."""
    print("\n3. Session Manager with Persistence")
    print("-" * 40)

    # Create manager with memory store
    store = MemorySessionStore()
    manager = await init_manager(store=store)

    print("  Manager initialized")

    # Create sessions with tags
    session1 = await manager.create_session(tags=["research", "demo"])
    print(f"  Created session 1: {session1.session_id[:8]}...")

    session2 = await manager.create_session(tags=["production"])
    print(f"  Created session 2: {session2.session_id[:8]}...")

    # List by tag
    research = await manager.list_sessions(tags=["research"])
    print(f"  Research sessions: {len(research)}")

    # Get session
    retrieved = await manager.get_session(session1.session_id)
    print(f"  Retrieved: {retrieved is not None}")

    # Session statistics
    stats = manager.get_stats()
    print(f"  Manager stats: {stats}")

    # Close sessions
    await session1.close()
    await session2.close()

    # Stop manager
    await manager.stop()
    print("  Manager stopped")


async def demo_session_resume():
    """Resuming sessions across runs."""
    print("\n4. Session Resume (Simulated)")
    print("-" * 40)

    # In real usage, you'd persist this across runs
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    store = SQLiteSessionStore(db_path)

    # "First run" - create session
    session_id = "resume-demo-001"
    state = SessionState(
        session_id=session_id,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        status=SessionStatus.ACTIVE,
        model="sonnet",
        system_prompt="You are a helpful assistant.",
        turn_count=5,
        total_input_tokens=1000,
        total_output_tokens=3000,
        messages_hash="abc123",
        last_prompt="What is Python?",
        last_response="Python is a programming language...",
    )
    await store.save(state)
    print(f"  First run: saved session {session_id}")

    # "Second run" - resume session
    await store.close()
    store = SQLiteSessionStore(db_path)

    loaded = await store.load(session_id)
    if loaded:
        print(f"  Second run: resumed session")
        print(f"    Turn count: {loaded.turn_count}")
        print(f"    Total tokens: {loaded.total_input_tokens + loaded.total_output_tokens}")
        print(f"    Last prompt: {loaded.last_prompt[:30]}...")

        # In real usage, you'd create a Session with this state
        # session = await load_session(session_id, store)

    await store.close()
    Path(db_path).unlink()


async def demo_session_fork():
    """Forking sessions for experimentation."""
    print("\n5. Session Forking (Concept)")
    print("-" * 40)

    store = MemorySessionStore()

    # Original session
    original = SessionState(
        session_id="original-001",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        status=SessionStatus.ACTIVE,
        model="sonnet",
        turn_count=10,
        total_input_tokens=2000,
        total_output_tokens=5000,
        messages_hash="original-hash",
    )
    await store.save(original)
    print(f"  Original session: {original.session_id}")
    print(f"    Turns: {original.turn_count}")

    # Fork creates new session with shared history
    forked = SessionState(
        session_id="forked-001",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        status=SessionStatus.ACTIVE,
        model=original.model,
        turn_count=original.turn_count,
        total_input_tokens=original.total_input_tokens,
        total_output_tokens=original.total_output_tokens,
        messages_hash=original.messages_hash,
        metadata={"forked_from": original.session_id},
    )
    await store.save(forked)
    print(f"  Forked session: {forked.session_id}")
    print(f"    Inherits turns: {forked.turn_count}")

    # Both can now diverge
    sessions = await store.list()
    print(f"  Total sessions: {len(sessions)}")

    await store.close()


async def main():
    """Run all persistence demos."""
    print("=" * 60)
    print("Session Persistence Examples")
    print("=" * 60)

    await demo_memory_store()
    await demo_sqlite_store()
    await demo_session_manager()
    await demo_session_resume()
    await demo_session_fork()

    print("\n" + "=" * 60)
    print("All persistence demos complete!")


if __name__ == "__main__":
    asyncio.run(main())
