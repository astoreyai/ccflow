"""
Project - Hierarchical organization for sessions and traces.

Provides project-level grouping of sessions with support for nested sub-projects,
trace replay, and aggregate analysis.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

import structlog

from ccflow.tracing import TracingSession
from ccflow.types import CLIAgentOptions, ProjectData, TraceData

if TYPE_CHECKING:
    from ccflow.store import SessionStore
    from ccflow.trace_store import ProjectStore, TraceStore

logger = structlog.get_logger(__name__)


class Project:
    """Project for organizing related sessions and traces.

    Projects provide hierarchical organization with optional nesting,
    enabling:
    - Grouping related sessions under a common project
    - Full trace capture for all project sessions
    - Replay capability for individual traces
    - Aggregate analysis across project traces

    Example:
        >>> # Initialize stores (same DB)
        >>> db_path = "ccflow.db"
        >>> project_store = SQLiteProjectStore(db_path)
        >>> trace_store = SQLiteTraceStore(db_path)
        >>> session_store = SQLiteSessionStore(db_path)

        >>> # Create project
        >>> project = Project(
        ...     name="Code Review Analysis",
        ...     description="Analyzing code review patterns with ultrathink",
        ...     store=project_store,
        ...     trace_store=trace_store,
        ...     session_store=session_store,
        ... )
        >>> await project.save()

        >>> # Create session with tracing
        >>> session = project.create_session(
        ...     options=CLIAgentOptions(model="sonnet", ultrathink=True),
        ...     detailed=True,
        ... )
        >>> async for msg in session.send_message("Analyze this function"):
        ...     print(msg.content, end="")

        >>> # Access traces
        >>> traces = await project.get_traces()
        >>> print(f"Total traces: {len(traces)}")
    """

    def __init__(
        self,
        project_id: str | None = None,
        name: str = "",
        description: str = "",
        parent_project_id: str | None = None,
        *,
        store: ProjectStore | None = None,
        trace_store: TraceStore | None = None,
        session_store: SessionStore | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize project.

        Args:
            project_id: Unique identifier (auto-generated if None)
            name: Project name
            description: Project description
            parent_project_id: Parent project for nesting
            store: Project persistence store
            trace_store: Trace persistence store
            session_store: Session persistence store
            metadata: Optional metadata dict
        """
        self.project_id = project_id or str(uuid.uuid4())
        self.name = name
        self.description = description
        self.parent_project_id = parent_project_id
        self.metadata = metadata or {}
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at

        self._store = store
        self._trace_store = trace_store
        self._session_store = session_store

        logger.debug(
            "project_created",
            project_id=self.project_id,
            name=name,
            parent_project_id=parent_project_id,
        )

    def create_session(
        self,
        options: CLIAgentOptions | None = None,
        *,
        detailed: bool = False,
    ) -> TracingSession:
        """Create a new tracing session within this project.

        Args:
            options: CLI agent options
            detailed: Capture message-level stream detail

        Returns:
            TracingSession linked to this project
        """
        return TracingSession(
            project_id=self.project_id,
            options=options,
            store=self._session_store,
            trace_store=self._trace_store,
            detailed=detailed,
        )

    def create_subproject(
        self,
        name: str,
        description: str = "",
        *,
        metadata: dict[str, Any] | None = None,
    ) -> Project:
        """Create a nested sub-project.

        Args:
            name: Sub-project name
            description: Sub-project description
            metadata: Optional metadata

        Returns:
            New Project instance as child of this project
        """
        return Project(
            name=name,
            description=description,
            parent_project_id=self.project_id,
            store=self._store,
            trace_store=self._trace_store,
            session_store=self._session_store,
            metadata=metadata,
        )

    async def get_traces(
        self,
        *,
        include_subprojects: bool = False,
        limit: int = 100,
    ) -> list[TraceData]:
        """Get all traces in this project.

        Args:
            include_subprojects: Include traces from nested sub-projects
            limit: Maximum traces to return

        Returns:
            List of traces
        """
        if not self._trace_store:
            return []

        traces = await self._trace_store.get_project_traces(self.project_id, limit)

        if include_subprojects and self._store:
            subprojects = await self._store.get_subprojects(self.project_id)
            for sub in subprojects:
                sub_traces = await self._trace_store.get_project_traces(
                    sub.project_id, limit
                )
                traces.extend(sub_traces)

        return traces

    async def replay_as_new(
        self,
        trace_id: str,
        *,
        options_override: CLIAgentOptions | None = None,
        detailed: bool = False,
    ) -> TracingSession:
        """Replay a trace as a new session.

        Creates fresh session with the original prompt. New trace is recorded.

        Args:
            trace_id: Trace to replay
            options_override: Options to override from original
            detailed: Capture message-level detail

        Returns:
            TracingSession ready for iteration

        Raises:
            ValueError: If trace not found
        """
        if not self._trace_store:
            raise ValueError("No trace store configured")

        trace = await self._trace_store.load(trace_id)
        if not trace:
            raise ValueError(f"Trace {trace_id} not found")

        # Merge original options with overrides
        opts_dict = trace.options_snapshot.copy()
        if options_override:
            override_dict = {
                k: v for k, v in self._options_to_dict(options_override).items()
                if v is not None
            }
            opts_dict.update(override_dict)

        opts = self._dict_to_options(opts_dict)

        # Create new session for replay
        session = self.create_session(options=opts, detailed=detailed)

        # Store reference to original trace in metadata
        session._current_trace = TraceData(
            trace_id=str(uuid.uuid4()),
            session_id=session.session_id,
            project_id=self.project_id,
            parent_trace_id=trace_id,  # Link to original
            prompt=trace.prompt,
            options_snapshot=opts_dict,
        )

        logger.info(
            "trace_replay_prepared",
            original_trace_id=trace_id,
            new_session_id=session.session_id,
        )

        return session

    async def replay_fork(
        self,
        trace_id: str,
        *,
        options_override: CLIAgentOptions | None = None,
    ) -> TracingSession:
        """Replay by forking from the original session state.

        Uses CLI --fork to continue from original session's state.
        Useful for branching conversations.

        Args:
            trace_id: Trace to fork from
            options_override: Options to override

        Returns:
            TracingSession forked from original

        Raises:
            ValueError: If trace not found
        """
        if not self._trace_store:
            raise ValueError("No trace store configured")

        trace = await self._trace_store.load(trace_id)
        if not trace:
            raise ValueError(f"Trace {trace_id} not found")

        # Merge options
        opts_dict = trace.options_snapshot.copy()
        if options_override:
            override_dict = {
                k: v for k, v in self._options_to_dict(options_override).items()
                if v is not None
            }
            opts_dict.update(override_dict)

        opts = self._dict_to_options(opts_dict)
        opts.session_id = trace.session_id
        opts.resume = True
        opts.fork_session = True

        session = TracingSession(
            project_id=self.project_id,
            options=opts,
            store=self._session_store,
            trace_store=self._trace_store,
        )
        session._is_first_message = False  # Mark as forked

        logger.info(
            "trace_fork_prepared",
            original_trace_id=trace_id,
            original_session_id=trace.session_id,
            forked_session_id=session.session_id,
        )

        return session

    def _options_to_dict(self, options: CLIAgentOptions) -> dict[str, Any]:
        """Convert CLIAgentOptions to dict."""
        return {
            "model": options.model,
            "fallback_model": options.fallback_model,
            "system_prompt": options.system_prompt,
            "append_system_prompt": options.append_system_prompt,
            "permission_mode": options.permission_mode.value if options.permission_mode else None,
            "allowed_tools": options.allowed_tools,
            "disallowed_tools": options.disallowed_tools,
            "max_budget_usd": options.max_budget_usd,
            "timeout": options.timeout,
            "ultrathink": options.ultrathink,
        }

    def _dict_to_options(self, opts_dict: dict[str, Any]) -> CLIAgentOptions:
        """Convert dict back to CLIAgentOptions."""
        from ccflow.types import PermissionMode

        permission_mode = None
        if opts_dict.get("permission_mode"):
            permission_mode = PermissionMode(opts_dict["permission_mode"])

        return CLIAgentOptions(
            model=opts_dict.get("model", "sonnet"),
            fallback_model=opts_dict.get("fallback_model"),
            system_prompt=opts_dict.get("system_prompt"),
            append_system_prompt=opts_dict.get("append_system_prompt"),
            permission_mode=permission_mode or PermissionMode.DEFAULT,
            allowed_tools=opts_dict.get("allowed_tools"),
            disallowed_tools=opts_dict.get("disallowed_tools"),
            max_budget_usd=opts_dict.get("max_budget_usd"),
            timeout=opts_dict.get("timeout", 300.0),
            ultrathink=opts_dict.get("ultrathink", False),
        )

    def to_data(self) -> ProjectData:
        """Convert to ProjectData for persistence."""
        return ProjectData(
            project_id=self.project_id,
            name=self.name,
            description=self.description,
            parent_project_id=self.parent_project_id,
            metadata=self.metadata,
            created_at=self.created_at,
            updated_at=datetime.now().isoformat(),
        )

    async def save(self) -> None:
        """Persist project to store."""
        if self._store:
            self.updated_at = datetime.now().isoformat()
            await self._store.save(self.to_data())
            logger.debug("project_saved", project_id=self.project_id)

    @classmethod
    async def load(
        cls,
        project_id: str,
        store: ProjectStore,
        *,
        trace_store: TraceStore | None = None,
        session_store: SessionStore | None = None,
    ) -> Project | None:
        """Load project from store.

        Args:
            project_id: Project UUID to load
            store: Project store
            trace_store: Optional trace store
            session_store: Optional session store

        Returns:
            Project instance or None if not found
        """
        data = await store.load(project_id)
        if not data:
            return None

        project = cls(
            project_id=data.project_id,
            name=data.name,
            description=data.description,
            parent_project_id=data.parent_project_id,
            store=store,
            trace_store=trace_store,
            session_store=session_store,
            metadata=data.metadata,
        )
        project.created_at = data.created_at
        project.updated_at = data.updated_at

        logger.debug("project_loaded", project_id=project_id)

        return project

    async def get_subprojects(self) -> list[Project]:
        """Get child projects.

        Returns:
            List of child Project instances
        """
        if not self._store:
            return []

        subproject_data = await self._store.get_subprojects(self.project_id)
        return [
            Project(
                project_id=data.project_id,
                name=data.name,
                description=data.description,
                parent_project_id=data.parent_project_id,
                store=self._store,
                trace_store=self._trace_store,
                session_store=self._session_store,
                metadata=data.metadata,
            )
            for data in subproject_data
        ]

    async def get_trace_summary(self) -> dict[str, Any]:
        """Get aggregate statistics for project traces.

        Returns:
            Dict with total_traces, total_tokens, total_cost, etc.
        """
        traces = await self.get_traces(include_subprojects=False)

        total_input = sum(t.input_tokens for t in traces)
        total_output = sum(t.output_tokens for t in traces)
        total_thinking = sum(t.thinking_tokens for t in traces)
        total_cost = sum(t.cost_usd for t in traces)
        total_duration = sum(t.duration_ms for t in traces)

        success_count = sum(1 for t in traces if t.status == "success")
        error_count = sum(1 for t in traces if t.status == "error")

        return {
            "total_traces": len(traces),
            "input_tokens": total_input,
            "output_tokens": total_output,
            "thinking_tokens": total_thinking,
            "total_tokens": total_input + total_output + total_thinking,
            "total_cost_usd": total_cost,
            "total_duration_ms": total_duration,
            "success_count": success_count,
            "error_count": error_count,
            "avg_duration_ms": total_duration / len(traces) if traces else 0,
        }

    def __repr__(self) -> str:
        return f"Project(id={self.project_id[:8]}..., name={self.name!r})"
