"""
SQLite Session Store - Async SQLite-backed session persistence.

Uses aiosqlite for non-blocking database operations with WAL mode
for better concurrency.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import aiosqlite
import structlog

from ccflow.exceptions import SessionStoreError
from ccflow.store import (
    BaseSessionStore,
    SessionFilter,
    SessionMetadata,
    SessionState,
    SessionStatus,
)

logger = structlog.get_logger(__name__)

# Default database path
DEFAULT_DB_PATH = Path.home() / ".ccflow" / "sessions.db"

# Schema version for migrations
SCHEMA_VERSION = 1

# SQL statements
CREATE_SESSIONS_TABLE = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    status TEXT NOT NULL,
    model TEXT NOT NULL,
    system_prompt TEXT,
    append_system_prompt TEXT,
    turn_count INTEGER DEFAULT 0,
    total_input_tokens INTEGER DEFAULT 0,
    total_output_tokens INTEGER DEFAULT 0,
    total_cost_usd REAL DEFAULT 0.0,
    messages_hash TEXT DEFAULT '',
    last_prompt TEXT DEFAULT '',
    last_response TEXT DEFAULT '',
    tags TEXT DEFAULT '[]',
    metadata TEXT DEFAULT '{}',
    toon_savings_ratio REAL DEFAULT 0.0
)
"""

CREATE_SCHEMA_VERSION_TABLE = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
)
"""

CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON sessions(created_at)",
    "CREATE INDEX IF NOT EXISTS idx_sessions_updated_at ON sessions(updated_at)",
    "CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status)",
    "CREATE INDEX IF NOT EXISTS idx_sessions_model ON sessions(model)",
]


class SQLiteSessionStore(BaseSessionStore):
    """SQLite-backed session storage with async operations.

    Uses WAL mode for better concurrent read/write performance.
    Automatically creates database and schema on first use.

    Example:
        >>> async with SQLiteSessionStore() as store:
        ...     await store.save(session_state)
        ...     sessions = await store.list(SessionFilter(status=SessionStatus.ACTIVE))
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        timeout: float = 30.0,
        auto_vacuum: bool = True,
    ) -> None:
        """Initialize SQLite store.

        Args:
            db_path: Path to SQLite database. Uses default if None.
            timeout: Database connection timeout in seconds.
            auto_vacuum: Enable auto-vacuum for space reclamation.
        """
        self._db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self._timeout = timeout
        self._auto_vacuum = auto_vacuum
        self._conn: aiosqlite.Connection | None = None
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        """Ensure database connection and schema exist."""
        if self._initialized and self._conn is not None:
            return

        # Create directory if needed
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        # Open connection
        self._conn = await aiosqlite.connect(
            str(self._db_path),
            timeout=self._timeout,
        )

        # Enable WAL mode and other optimizations
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA synchronous=NORMAL")
        await self._conn.execute("PRAGMA temp_store=MEMORY")
        await self._conn.execute("PRAGMA mmap_size=268435456")  # 256MB

        if self._auto_vacuum:
            await self._conn.execute("PRAGMA auto_vacuum=INCREMENTAL")

        # Create schema
        await self._conn.execute(CREATE_SCHEMA_VERSION_TABLE)
        await self._conn.execute(CREATE_SESSIONS_TABLE)

        for index_sql in CREATE_INDEXES:
            await self._conn.execute(index_sql)

        # Set schema version
        await self._conn.execute(
            "INSERT OR REPLACE INTO schema_version (version) VALUES (?)",
            (SCHEMA_VERSION,),
        )

        await self._conn.commit()
        self._initialized = True

        logger.debug("sqlite_store_initialized", db_path=str(self._db_path))

    def _state_to_row(self, state: SessionState) -> dict[str, Any]:
        """Convert SessionState to database row."""
        return {
            "session_id": state.session_id,
            "created_at": state.created_at.isoformat(),
            "updated_at": state.updated_at.isoformat(),
            "status": state.status.value,
            "model": state.model,
            "system_prompt": state.system_prompt,
            "append_system_prompt": state.append_system_prompt,
            "turn_count": state.turn_count,
            "total_input_tokens": state.total_input_tokens,
            "total_output_tokens": state.total_output_tokens,
            "total_cost_usd": state.total_cost_usd,
            "messages_hash": state.messages_hash,
            "last_prompt": state.last_prompt,
            "last_response": state.last_response,
            "tags": json.dumps(state.tags),
            "metadata": json.dumps(state.metadata),
            "toon_savings_ratio": state.toon_savings_ratio,
        }

    def _row_to_state(self, row: aiosqlite.Row) -> SessionState:
        """Convert database row to SessionState."""
        return SessionState(
            session_id=row["session_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            status=SessionStatus(row["status"]),
            model=row["model"],
            system_prompt=row["system_prompt"],
            append_system_prompt=row["append_system_prompt"],
            turn_count=row["turn_count"],
            total_input_tokens=row["total_input_tokens"],
            total_output_tokens=row["total_output_tokens"],
            total_cost_usd=row["total_cost_usd"],
            messages_hash=row["messages_hash"],
            last_prompt=row["last_prompt"],
            last_response=row["last_response"],
            tags=json.loads(row["tags"]),
            metadata=json.loads(row["metadata"]),
            toon_savings_ratio=row["toon_savings_ratio"],
        )

    def _row_to_metadata(self, row: aiosqlite.Row) -> SessionMetadata:
        """Convert database row to SessionMetadata."""
        return SessionMetadata(
            session_id=row["session_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            status=SessionStatus(row["status"]),
            model=row["model"],
            turn_count=row["turn_count"],
            total_input_tokens=row["total_input_tokens"],
            total_output_tokens=row["total_output_tokens"],
            total_cost_usd=row["total_cost_usd"],
            tags=json.loads(row["tags"]),
        )

    async def save(self, state: SessionState) -> None:
        """Save or update session state."""
        try:
            await self._ensure_initialized()
            assert self._conn is not None

            row = self._state_to_row(state)
            columns = ", ".join(row.keys())
            placeholders = ", ".join("?" * len(row))
            update_clause = ", ".join(f"{k}=excluded.{k}" for k in row.keys())

            sql = f"""
                INSERT INTO sessions ({columns})
                VALUES ({placeholders})
                ON CONFLICT(session_id) DO UPDATE SET {update_clause}
            """

            await self._conn.execute(sql, tuple(row.values()))
            await self._conn.commit()

            logger.debug("session_saved", session_id=state.session_id)

        except aiosqlite.Error as e:
            logger.error("session_save_failed", session_id=state.session_id, error=str(e))
            raise SessionStoreError(f"Failed to save session: {e}") from e

    async def load(self, session_id: str) -> SessionState | None:
        """Load session state by ID."""
        try:
            await self._ensure_initialized()
            assert self._conn is not None

            self._conn.row_factory = aiosqlite.Row
            cursor = await self._conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (session_id,),
            )
            row = await cursor.fetchone()

            if row is None:
                return None

            return self._row_to_state(row)

        except aiosqlite.Error as e:
            logger.error("session_load_failed", session_id=session_id, error=str(e))
            raise SessionStoreError(f"Failed to load session: {e}") from e

    async def delete(self, session_id: str) -> bool:
        """Delete session by ID."""
        try:
            await self._ensure_initialized()
            assert self._conn is not None

            cursor = await self._conn.execute(
                "DELETE FROM sessions WHERE session_id = ?",
                (session_id,),
            )
            await self._conn.commit()

            deleted = cursor.rowcount > 0
            if deleted:
                logger.debug("session_deleted", session_id=session_id)

            return deleted

        except aiosqlite.Error as e:
            logger.error("session_delete_failed", session_id=session_id, error=str(e))
            raise SessionStoreError(f"Failed to delete session: {e}") from e

    async def list(self, filter: SessionFilter | None = None) -> list[SessionMetadata]:
        """List sessions matching filter criteria."""
        try:
            await self._ensure_initialized()
            assert self._conn is not None

            filter = filter or SessionFilter()

            # Build WHERE clause
            conditions: list[str] = []
            params: list[Any] = []

            if filter.status is not None:
                conditions.append("status = ?")
                params.append(filter.status.value)

            if filter.model is not None:
                conditions.append("model = ?")
                params.append(filter.model)

            if filter.created_after is not None:
                conditions.append("created_at >= ?")
                params.append(filter.created_after.isoformat())

            if filter.created_before is not None:
                conditions.append("created_at <= ?")
                params.append(filter.created_before.isoformat())

            if filter.updated_after is not None:
                conditions.append("updated_at >= ?")
                params.append(filter.updated_after.isoformat())

            if filter.min_turns is not None:
                conditions.append("turn_count >= ?")
                params.append(filter.min_turns)

            if filter.max_turns is not None:
                conditions.append("turn_count <= ?")
                params.append(filter.max_turns)

            if filter.tags:
                # Match any tag in the list
                tag_conditions = []
                for tag in filter.tags:
                    tag_conditions.append("tags LIKE ?")
                    params.append(f'%"{tag}"%')
                conditions.append(f"({' OR '.join(tag_conditions)})")

            where_clause = " AND ".join(conditions) if conditions else "1=1"

            # Build ORDER BY
            order_dir = "DESC" if filter.order_desc else "ASC"
            order_by = f"{filter.order_by} {order_dir}"

            sql = f"""
                SELECT session_id, created_at, updated_at, status, model,
                       turn_count, total_input_tokens, total_output_tokens,
                       total_cost_usd, tags
                FROM sessions
                WHERE {where_clause}
                ORDER BY {order_by}
                LIMIT ? OFFSET ?
            """
            params.extend([filter.limit, filter.offset])

            self._conn.row_factory = aiosqlite.Row
            cursor = await self._conn.execute(sql, params)
            rows = await cursor.fetchall()

            return [self._row_to_metadata(row) for row in rows]

        except aiosqlite.Error as e:
            logger.error("session_list_failed", error=str(e))
            raise SessionStoreError(f"Failed to list sessions: {e}") from e

    async def cleanup(self, older_than: timedelta) -> int:
        """Delete sessions older than specified duration."""
        try:
            await self._ensure_initialized()
            assert self._conn is not None

            cutoff = datetime.now() - older_than

            cursor = await self._conn.execute(
                "DELETE FROM sessions WHERE updated_at < ?",
                (cutoff.isoformat(),),
            )
            await self._conn.commit()

            deleted = cursor.rowcount
            if deleted > 0:
                logger.info("sessions_cleaned_up", count=deleted, older_than=str(older_than))

            return deleted

        except aiosqlite.Error as e:
            logger.error("session_cleanup_failed", error=str(e))
            raise SessionStoreError(f"Failed to cleanup sessions: {e}") from e

    async def exists(self, session_id: str) -> bool:
        """Check if session exists."""
        try:
            await self._ensure_initialized()
            assert self._conn is not None

            cursor = await self._conn.execute(
                "SELECT 1 FROM sessions WHERE session_id = ? LIMIT 1",
                (session_id,),
            )
            row = await cursor.fetchone()
            return row is not None

        except aiosqlite.Error as e:
            logger.error("session_exists_check_failed", session_id=session_id, error=str(e))
            raise SessionStoreError(f"Failed to check session existence: {e}") from e

    async def count(self, filter: SessionFilter | None = None) -> int:
        """Count sessions matching filter."""
        try:
            await self._ensure_initialized()
            assert self._conn is not None

            filter = filter or SessionFilter()

            # Build WHERE clause (same as list)
            conditions: list[str] = []
            params: list[Any] = []

            if filter.status is not None:
                conditions.append("status = ?")
                params.append(filter.status.value)

            if filter.model is not None:
                conditions.append("model = ?")
                params.append(filter.model)

            if filter.created_after is not None:
                conditions.append("created_at >= ?")
                params.append(filter.created_after.isoformat())

            if filter.created_before is not None:
                conditions.append("created_at <= ?")
                params.append(filter.created_before.isoformat())

            if filter.min_turns is not None:
                conditions.append("turn_count >= ?")
                params.append(filter.min_turns)

            if filter.max_turns is not None:
                conditions.append("turn_count <= ?")
                params.append(filter.max_turns)

            where_clause = " AND ".join(conditions) if conditions else "1=1"

            cursor = await self._conn.execute(
                f"SELECT COUNT(*) FROM sessions WHERE {where_clause}",
                params,
            )
            row = await cursor.fetchone()
            return row[0] if row else 0

        except aiosqlite.Error as e:
            logger.error("session_count_failed", error=str(e))
            raise SessionStoreError(f"Failed to count sessions: {e}") from e

    async def update_status(self, session_id: str, status: SessionStatus) -> bool:
        """Update session status."""
        try:
            await self._ensure_initialized()
            assert self._conn is not None

            cursor = await self._conn.execute(
                "UPDATE sessions SET status = ?, updated_at = ? WHERE session_id = ?",
                (status.value, datetime.now().isoformat(), session_id),
            )
            await self._conn.commit()

            updated = cursor.rowcount > 0
            if updated:
                logger.debug("session_status_updated", session_id=session_id, status=status.value)

            return updated

        except aiosqlite.Error as e:
            logger.error("session_status_update_failed", session_id=session_id, error=str(e))
            raise SessionStoreError(f"Failed to update session status: {e}") from e

    async def close(self) -> None:
        """Close database connection."""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None
            self._initialized = False
            logger.debug("sqlite_store_closed")

    async def vacuum(self) -> None:
        """Run VACUUM to reclaim space."""
        try:
            await self._ensure_initialized()
            assert self._conn is not None

            await self._conn.execute("VACUUM")
            logger.debug("sqlite_vacuum_complete")

        except aiosqlite.Error as e:
            logger.error("sqlite_vacuum_failed", error=str(e))
            raise SessionStoreError(f"Failed to vacuum database: {e}") from e

    @property
    def db_path(self) -> Path:
        """Get database file path."""
        return self._db_path
