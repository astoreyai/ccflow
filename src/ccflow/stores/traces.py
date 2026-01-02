"""
SQLite Trace and Project Stores - Async SQLite-backed persistence for traces and projects.

Uses aiosqlite for non-blocking database operations. Can share the same database
as SQLiteSessionStore.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import aiosqlite
import structlog

from ccflow.exceptions import SessionStoreError
from ccflow.store import SessionMetadata, SessionStatus
from ccflow.trace_store import BaseProjectStore, BaseTraceStore
from ccflow.types import ProjectData, ProjectFilter, TraceData, TraceFilter, TraceStatus

# Use uppercase List to avoid shadowing by class.list method
List = list

logger = structlog.get_logger(__name__)

# Default database path (shared with sessions)
DEFAULT_DB_PATH = Path.home() / ".ccflow" / "sessions.db"

# Schema version for traces (independent of session schema)
TRACE_SCHEMA_VERSION = 1

# SQL for projects table
CREATE_PROJECTS_TABLE = """
CREATE TABLE IF NOT EXISTS projects (
    project_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT DEFAULT '',
    parent_project_id TEXT,
    metadata TEXT DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (parent_project_id) REFERENCES projects(project_id)
)
"""

CREATE_PROJECTS_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_projects_name ON projects(name)",
    "CREATE INDEX IF NOT EXISTS idx_projects_parent ON projects(parent_project_id)",
    "CREATE INDEX IF NOT EXISTS idx_projects_created_at ON projects(created_at)",
]

# SQL for traces table
CREATE_TRACES_TABLE = """
CREATE TABLE IF NOT EXISTS traces (
    trace_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    project_id TEXT,
    parent_trace_id TEXT,
    sequence_number INTEGER NOT NULL DEFAULT 0,

    -- Content
    prompt TEXT NOT NULL,
    response TEXT DEFAULT '',
    thinking TEXT DEFAULT '',
    tool_calls TEXT DEFAULT '[]',
    message_stream TEXT,

    -- Configuration snapshot
    options_snapshot TEXT DEFAULT '{}',

    -- Metrics
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    thinking_tokens INTEGER DEFAULT 0,
    cost_usd REAL DEFAULT 0.0,
    duration_ms INTEGER DEFAULT 0,

    -- Status
    status TEXT DEFAULT 'pending',
    error_message TEXT,

    -- Timestamps
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,

    -- Extensibility
    metadata TEXT DEFAULT '{}',

    FOREIGN KEY (session_id) REFERENCES sessions(session_id),
    FOREIGN KEY (project_id) REFERENCES projects(project_id),
    FOREIGN KEY (parent_trace_id) REFERENCES traces(trace_id)
)
"""

CREATE_TRACES_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_traces_session ON traces(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_traces_project ON traces(project_id)",
    "CREATE INDEX IF NOT EXISTS idx_traces_parent ON traces(parent_trace_id)",
    "CREATE INDEX IF NOT EXISTS idx_traces_status ON traces(status)",
    "CREATE INDEX IF NOT EXISTS idx_traces_created_at ON traces(created_at)",
    "CREATE INDEX IF NOT EXISTS idx_traces_session_seq ON traces(session_id, sequence_number)",
]

# SQL to add project_id to sessions table (migration)
ADD_PROJECT_ID_TO_SESSIONS = """
ALTER TABLE sessions ADD COLUMN project_id TEXT
"""

CREATE_SESSIONS_PROJECT_INDEX = """
CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project_id)
"""

# Schema version table for traces
CREATE_TRACE_SCHEMA_VERSION_TABLE = """
CREATE TABLE IF NOT EXISTS trace_schema_version (
    version INTEGER PRIMARY KEY
)
"""


class SQLiteTraceStore(BaseTraceStore):
    """SQLite-backed trace storage with async operations.

    Stores complete prompt/response traces including thinking content,
    tool calls, and metrics.

    Example:
        >>> async with SQLiteTraceStore() as store:
        ...     await store.save(trace)
        ...     traces = await store.get_session_traces(session_id)
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        timeout: float = 30.0,
    ) -> None:
        """Initialize SQLite trace store.

        Args:
            db_path: Path to SQLite database. Uses default if None.
            timeout: Database connection timeout in seconds.
        """
        self._db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self._timeout = timeout
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

        # Enable WAL mode and optimizations
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA synchronous=NORMAL")
        await self._conn.execute("PRAGMA temp_store=MEMORY")

        # Create schema
        await self._conn.execute(CREATE_TRACE_SCHEMA_VERSION_TABLE)
        await self._conn.execute(CREATE_TRACES_TABLE)

        for index_sql in CREATE_TRACES_INDEXES:
            await self._conn.execute(index_sql)

        # Set schema version
        await self._conn.execute(
            "INSERT OR REPLACE INTO trace_schema_version (version) VALUES (?)",
            (TRACE_SCHEMA_VERSION,),
        )

        await self._conn.commit()
        self._initialized = True

        logger.debug("sqlite_trace_store_initialized", db_path=str(self._db_path))

    def _trace_to_row(self, trace: TraceData) -> dict[str, Any]:
        """Convert TraceData to database row."""
        return {
            "trace_id": trace.trace_id,
            "session_id": trace.session_id,
            "project_id": trace.project_id,
            "parent_trace_id": trace.parent_trace_id,
            "sequence_number": trace.sequence_number,
            "prompt": trace.prompt,
            "response": trace.response,
            "thinking": trace.thinking,
            "tool_calls": json.dumps(trace.tool_calls),
            "message_stream": json.dumps(trace.message_stream) if trace.message_stream else None,
            "options_snapshot": json.dumps(trace.options_snapshot),
            "input_tokens": trace.input_tokens,
            "output_tokens": trace.output_tokens,
            "thinking_tokens": trace.thinking_tokens,
            "cost_usd": trace.cost_usd,
            "duration_ms": trace.duration_ms,
            "status": trace.status.value,
            "error_message": trace.error_message,
            "created_at": trace.created_at or datetime.now().isoformat(),
            "updated_at": trace.updated_at or datetime.now().isoformat(),
            "metadata": json.dumps(trace.metadata),
        }

    def _row_to_trace(self, row: aiosqlite.Row) -> TraceData:
        """Convert database row to TraceData."""
        message_stream = None
        if row["message_stream"]:
            message_stream = json.loads(row["message_stream"])

        return TraceData(
            trace_id=row["trace_id"],
            session_id=row["session_id"],
            project_id=row["project_id"],
            parent_trace_id=row["parent_trace_id"],
            sequence_number=row["sequence_number"],
            prompt=row["prompt"],
            response=row["response"] or "",
            thinking=row["thinking"] or "",
            tool_calls=json.loads(row["tool_calls"]),
            message_stream=message_stream,
            options_snapshot=json.loads(row["options_snapshot"]),
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
            thinking_tokens=row["thinking_tokens"],
            cost_usd=row["cost_usd"],
            duration_ms=row["duration_ms"],
            status=TraceStatus(row["status"]),
            error_message=row["error_message"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            metadata=json.loads(row["metadata"]),
        )

    async def save(self, trace: TraceData) -> None:
        """Save or update a trace."""
        try:
            await self._ensure_initialized()
            assert self._conn is not None

            # Update timestamp
            trace.updated_at = datetime.now().isoformat()
            if not trace.created_at:
                trace.created_at = trace.updated_at

            row = self._trace_to_row(trace)
            columns = ", ".join(row.keys())
            placeholders = ", ".join("?" * len(row))
            update_clause = ", ".join(f"{k}=excluded.{k}" for k in row)

            sql = f"""
                INSERT INTO traces ({columns})
                VALUES ({placeholders})
                ON CONFLICT(trace_id) DO UPDATE SET {update_clause}
            """

            await self._conn.execute(sql, tuple(row.values()))
            await self._conn.commit()

            logger.debug("trace_saved", trace_id=trace.trace_id)

        except aiosqlite.Error as e:
            logger.error("trace_save_failed", trace_id=trace.trace_id, error=str(e))
            raise SessionStoreError(f"Failed to save trace: {e}") from e

    async def load(self, trace_id: str) -> TraceData | None:
        """Load trace by ID."""
        try:
            await self._ensure_initialized()
            assert self._conn is not None

            self._conn.row_factory = aiosqlite.Row
            cursor = await self._conn.execute(
                "SELECT * FROM traces WHERE trace_id = ?",
                (trace_id,),
            )
            row = await cursor.fetchone()

            if row is None:
                return None

            return self._row_to_trace(row)

        except aiosqlite.Error as e:
            logger.error("trace_load_failed", trace_id=trace_id, error=str(e))
            raise SessionStoreError(f"Failed to load trace: {e}") from e

    async def delete(self, trace_id: str) -> bool:
        """Delete trace by ID."""
        try:
            await self._ensure_initialized()
            assert self._conn is not None

            cursor = await self._conn.execute(
                "DELETE FROM traces WHERE trace_id = ?",
                (trace_id,),
            )
            await self._conn.commit()

            deleted = cursor.rowcount > 0
            if deleted:
                logger.debug("trace_deleted", trace_id=trace_id)

            return deleted

        except aiosqlite.Error as e:
            logger.error("trace_delete_failed", trace_id=trace_id, error=str(e))
            raise SessionStoreError(f"Failed to delete trace: {e}") from e

    async def list(self, filter: TraceFilter | None = None) -> List[TraceData]:
        """List traces matching filter criteria."""
        try:
            await self._ensure_initialized()
            assert self._conn is not None

            filter = filter or TraceFilter()

            # Build WHERE clause
            conditions: list[str] = []
            params: list[Any] = []

            if filter.session_id is not None:
                conditions.append("session_id = ?")
                params.append(filter.session_id)

            if filter.project_id is not None:
                conditions.append("project_id = ?")
                params.append(filter.project_id)

            if filter.status is not None:
                conditions.append("status = ?")
                params.append(filter.status.value)

            if filter.parent_trace_id is not None:
                conditions.append("parent_trace_id = ?")
                params.append(filter.parent_trace_id)

            if filter.after is not None:
                conditions.append("created_at >= ?")
                params.append(filter.after)

            if filter.before is not None:
                conditions.append("created_at <= ?")
                params.append(filter.before)

            where_clause = " AND ".join(conditions) if conditions else "1=1"

            sql = f"""
                SELECT * FROM traces
                WHERE {where_clause}
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            """
            params.extend([filter.limit, filter.offset])

            self._conn.row_factory = aiosqlite.Row
            cursor = await self._conn.execute(sql, params)
            rows = await cursor.fetchall()

            return [self._row_to_trace(row) for row in rows]

        except aiosqlite.Error as e:
            logger.error("trace_list_failed", error=str(e))
            raise SessionStoreError(f"Failed to list traces: {e}") from e

    async def get_session_traces(self, session_id: str, limit: int = 100) -> List[TraceData]:
        """Get all traces for a session ordered by sequence."""
        try:
            await self._ensure_initialized()
            assert self._conn is not None

            self._conn.row_factory = aiosqlite.Row
            cursor = await self._conn.execute(
                """
                SELECT * FROM traces
                WHERE session_id = ?
                ORDER BY sequence_number ASC
                LIMIT ?
                """,
                (session_id, limit),
            )
            rows = await cursor.fetchall()

            return [self._row_to_trace(row) for row in rows]

        except aiosqlite.Error as e:
            logger.error("trace_get_session_failed", session_id=session_id, error=str(e))
            raise SessionStoreError(f"Failed to get session traces: {e}") from e

    async def get_project_traces(self, project_id: str, limit: int = 100) -> List[TraceData]:
        """Get all traces for a project."""
        try:
            await self._ensure_initialized()
            assert self._conn is not None

            self._conn.row_factory = aiosqlite.Row
            cursor = await self._conn.execute(
                """
                SELECT * FROM traces
                WHERE project_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (project_id, limit),
            )
            rows = await cursor.fetchall()

            return [self._row_to_trace(row) for row in rows]

        except aiosqlite.Error as e:
            logger.error("trace_get_project_failed", project_id=project_id, error=str(e))
            raise SessionStoreError(f"Failed to get project traces: {e}") from e

    async def cleanup(self, older_than: timedelta) -> int:
        """Delete traces older than specified duration."""
        try:
            await self._ensure_initialized()
            assert self._conn is not None

            cutoff = datetime.now() - older_than

            cursor = await self._conn.execute(
                "DELETE FROM traces WHERE updated_at < ?",
                (cutoff.isoformat(),),
            )
            await self._conn.commit()

            deleted = cursor.rowcount
            if deleted > 0:
                logger.info("traces_cleaned_up", count=deleted, older_than=str(older_than))

            return deleted

        except aiosqlite.Error as e:
            logger.error("trace_cleanup_failed", error=str(e))
            raise SessionStoreError(f"Failed to cleanup traces: {e}") from e

    async def close(self) -> None:
        """Close database connection."""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None
            self._initialized = False
            logger.debug("sqlite_trace_store_closed")

    @property
    def db_path(self) -> Path:
        """Get database file path."""
        return self._db_path


class SQLiteProjectStore(BaseProjectStore):
    """SQLite-backed project storage with async operations.

    Stores project metadata and enables hierarchical organization
    of sessions and traces.

    Example:
        >>> async with SQLiteProjectStore() as store:
        ...     await store.save(project)
        ...     projects = await store.list(ProjectFilter(name_contains="PhD"))
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        timeout: float = 30.0,
    ) -> None:
        """Initialize SQLite project store.

        Args:
            db_path: Path to SQLite database. Uses default if None.
            timeout: Database connection timeout in seconds.
        """
        self._db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self._timeout = timeout
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

        # Enable WAL mode and optimizations
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA synchronous=NORMAL")
        await self._conn.execute("PRAGMA temp_store=MEMORY")

        # Create schema
        await self._conn.execute(CREATE_PROJECTS_TABLE)

        for index_sql in CREATE_PROJECTS_INDEXES:
            await self._conn.execute(index_sql)

        # Try to add project_id to sessions table (migration)
        await self._migrate_sessions_table()

        await self._conn.commit()
        self._initialized = True

        logger.debug("sqlite_project_store_initialized", db_path=str(self._db_path))

    async def _migrate_sessions_table(self) -> None:
        """Add project_id column to sessions table if it doesn't exist."""
        try:
            assert self._conn is not None

            # Check if column exists
            cursor = await self._conn.execute("PRAGMA table_info(sessions)")
            columns = await cursor.fetchall()
            column_names = [col[1] for col in columns]

            if "project_id" not in column_names:
                await self._conn.execute(ADD_PROJECT_ID_TO_SESSIONS)
                await self._conn.execute(CREATE_SESSIONS_PROJECT_INDEX)
                logger.info("sessions_table_migrated", added_column="project_id")

        except aiosqlite.OperationalError:
            # Sessions table doesn't exist yet, that's OK
            pass

    def _project_to_row(self, project: ProjectData) -> dict[str, Any]:
        """Convert ProjectData to database row."""
        return {
            "project_id": project.project_id,
            "name": project.name,
            "description": project.description,
            "parent_project_id": project.parent_project_id,
            "metadata": json.dumps(project.metadata),
            "created_at": project.created_at or datetime.now().isoformat(),
            "updated_at": project.updated_at or datetime.now().isoformat(),
        }

    def _row_to_project(self, row: aiosqlite.Row) -> ProjectData:
        """Convert database row to ProjectData."""
        return ProjectData(
            project_id=row["project_id"],
            name=row["name"],
            description=row["description"] or "",
            parent_project_id=row["parent_project_id"],
            metadata=json.loads(row["metadata"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    async def save(self, project: ProjectData) -> None:
        """Save or update a project."""
        try:
            await self._ensure_initialized()
            assert self._conn is not None

            # Update timestamp
            project.updated_at = datetime.now().isoformat()
            if not project.created_at:
                project.created_at = project.updated_at

            row = self._project_to_row(project)
            columns = ", ".join(row.keys())
            placeholders = ", ".join("?" * len(row))
            update_clause = ", ".join(f"{k}=excluded.{k}" for k in row)

            sql = f"""
                INSERT INTO projects ({columns})
                VALUES ({placeholders})
                ON CONFLICT(project_id) DO UPDATE SET {update_clause}
            """

            await self._conn.execute(sql, tuple(row.values()))
            await self._conn.commit()

            logger.debug("project_saved", project_id=project.project_id)

        except aiosqlite.Error as e:
            logger.error("project_save_failed", project_id=project.project_id, error=str(e))
            raise SessionStoreError(f"Failed to save project: {e}") from e

    async def load(self, project_id: str) -> ProjectData | None:
        """Load project by ID."""
        try:
            await self._ensure_initialized()
            assert self._conn is not None

            self._conn.row_factory = aiosqlite.Row
            cursor = await self._conn.execute(
                "SELECT * FROM projects WHERE project_id = ?",
                (project_id,),
            )
            row = await cursor.fetchone()

            if row is None:
                return None

            return self._row_to_project(row)

        except aiosqlite.Error as e:
            logger.error("project_load_failed", project_id=project_id, error=str(e))
            raise SessionStoreError(f"Failed to load project: {e}") from e

    async def delete(self, project_id: str) -> bool:
        """Delete project by ID."""
        try:
            await self._ensure_initialized()
            assert self._conn is not None

            cursor = await self._conn.execute(
                "DELETE FROM projects WHERE project_id = ?",
                (project_id,),
            )
            await self._conn.commit()

            deleted = cursor.rowcount > 0
            if deleted:
                logger.debug("project_deleted", project_id=project_id)

            return deleted

        except aiosqlite.Error as e:
            logger.error("project_delete_failed", project_id=project_id, error=str(e))
            raise SessionStoreError(f"Failed to delete project: {e}") from e

    async def list(self, filter: ProjectFilter | None = None) -> List[ProjectData]:
        """List projects matching filter criteria."""
        try:
            await self._ensure_initialized()
            assert self._conn is not None

            filter = filter or ProjectFilter()

            # Build WHERE clause
            conditions: list[str] = []
            params: list[Any] = []

            if filter.name_contains is not None:
                conditions.append("name LIKE ?")
                params.append(f"%{filter.name_contains}%")

            if filter.parent_project_id is not None:
                conditions.append("parent_project_id = ?")
                params.append(filter.parent_project_id)

            if filter.after is not None:
                conditions.append("created_at >= ?")
                params.append(filter.after)

            if filter.before is not None:
                conditions.append("created_at <= ?")
                params.append(filter.before)

            where_clause = " AND ".join(conditions) if conditions else "1=1"

            sql = f"""
                SELECT * FROM projects
                WHERE {where_clause}
                ORDER BY updated_at DESC
                LIMIT ? OFFSET ?
            """
            params.extend([filter.limit, filter.offset])

            self._conn.row_factory = aiosqlite.Row
            cursor = await self._conn.execute(sql, params)
            rows = await cursor.fetchall()

            return [self._row_to_project(row) for row in rows]

        except aiosqlite.Error as e:
            logger.error("project_list_failed", error=str(e))
            raise SessionStoreError(f"Failed to list projects: {e}") from e

    async def get_project_sessions(
        self, project_id: str, limit: int = 100
    ) -> List[SessionMetadata]:
        """Get all sessions for a project."""
        try:
            await self._ensure_initialized()
            assert self._conn is not None

            self._conn.row_factory = aiosqlite.Row
            cursor = await self._conn.execute(
                """
                SELECT session_id, created_at, updated_at, status, model,
                       turn_count, total_input_tokens, total_output_tokens,
                       total_cost_usd, tags
                FROM sessions
                WHERE project_id = ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (project_id, limit),
            )
            rows = await cursor.fetchall()

            return [
                SessionMetadata(
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
                for row in rows
            ]

        except aiosqlite.Error as e:
            logger.error("project_get_sessions_failed", project_id=project_id, error=str(e))
            raise SessionStoreError(f"Failed to get project sessions: {e}") from e

    async def get_subprojects(self, project_id: str) -> List[ProjectData]:
        """Get child projects of a parent."""
        return await self.list(ProjectFilter(parent_project_id=project_id))

    async def close(self) -> None:
        """Close database connection."""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None
            self._initialized = False
            logger.debug("sqlite_project_store_closed")

    @property
    def db_path(self) -> Path:
        """Get database file path."""
        return self._db_path
