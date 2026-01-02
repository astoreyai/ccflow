"""
Agentic harness utilities for long-running agent workflows.

Provides FeatureList and ProgressTracker for maintaining state across
multiple agent sessions, following Anthropic's recommended patterns.

Based on: https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class FeatureStatus(str, Enum):
    """Status of a feature in the feature list.

    Using explicit status values for clarity in JSON output.
    """

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    PASSING = "passing"
    FAILING = "failing"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


@dataclass
class Feature:
    """A single feature in the feature list.

    Attributes:
        id: Unique feature identifier
        description: End-to-end feature description
        status: Current status (pending/passing/failing/etc.)
        priority: Priority level (1=highest)
        tags: Categorization tags
        notes: Implementation notes
        test_command: Command to verify feature
        last_tested: ISO timestamp of last test
        error_message: Error details if failing
        metadata: Additional custom data
    """

    id: str
    description: str
    status: FeatureStatus = FeatureStatus.PENDING
    priority: int = 5
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    test_command: str | None = None
    last_tested: str | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status.value,
            "priority": self.priority,
            "tags": self.tags,
            "notes": self.notes,
            "test_command": self.test_command,
            "last_tested": self.last_tested,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Feature:
        """Create Feature from dictionary."""
        return cls(
            id=data["id"],
            description=data["description"],
            status=FeatureStatus(data.get("status", "pending")),
            priority=data.get("priority", 5),
            tags=data.get("tags", []),
            notes=data.get("notes", ""),
            test_command=data.get("test_command"),
            last_tested=data.get("last_tested"),
            error_message=data.get("error_message"),
            metadata=data.get("metadata", {}),
        )


class FeatureList:
    """JSON-based feature tracking for agentic workflows.

    Maintains a list of features with passing/failing status,
    designed to be robust against model overwrites (JSON format
    is less likely to be inappropriately modified than Markdown).

    Example:
        >>> features = FeatureList("features.json")
        >>> features.add(Feature(
        ...     id="auth-login",
        ...     description="User can log in with email/password",
        ...     priority=1,
        ...     tags=["auth", "critical"],
        ... ))
        >>> features.save()
        >>> # After testing...
        >>> features.mark_passing("auth-login")
        >>> features.save()

    Attributes:
        path: Path to the JSON file
        project_name: Optional project identifier
        features: List of Feature objects
    """

    def __init__(
        self,
        path: str | Path,
        project_name: str | None = None,
        auto_save: bool = False,
    ) -> None:
        """Initialize feature list.

        Args:
            path: Path to JSON file (created if doesn't exist)
            project_name: Optional project identifier
            auto_save: Automatically save after modifications
        """
        self._path = Path(path)
        self._project_name = project_name
        self._auto_save = auto_save
        self._features: dict[str, Feature] = {}
        self._created_at: str | None = None
        self._updated_at: str | None = None

        if self._path.exists():
            self.load()
        else:
            self._created_at = datetime.now().isoformat()
            self._updated_at = self._created_at

    @property
    def path(self) -> Path:
        """Get the file path."""
        return self._path

    @property
    def project_name(self) -> str | None:
        """Get project name."""
        return self._project_name

    def add(self, feature: Feature) -> None:
        """Add a feature to the list.

        Args:
            feature: Feature to add

        Raises:
            ValueError: If feature ID already exists
        """
        if feature.id in self._features:
            raise ValueError(f"Feature '{feature.id}' already exists")

        self._features[feature.id] = feature
        self._updated_at = datetime.now().isoformat()

        logger.debug("feature_added", feature_id=feature.id)

        if self._auto_save:
            self.save()

    def update(self, feature: Feature) -> None:
        """Update an existing feature.

        Args:
            feature: Feature with updated data

        Raises:
            KeyError: If feature ID doesn't exist
        """
        if feature.id not in self._features:
            raise KeyError(f"Feature '{feature.id}' not found")

        self._features[feature.id] = feature
        self._updated_at = datetime.now().isoformat()

        logger.debug("feature_updated", feature_id=feature.id)

        if self._auto_save:
            self.save()

    def get(self, feature_id: str) -> Feature | None:
        """Get feature by ID.

        Args:
            feature_id: Feature identifier

        Returns:
            Feature or None if not found
        """
        return self._features.get(feature_id)

    def remove(self, feature_id: str) -> bool:
        """Remove a feature.

        Args:
            feature_id: Feature identifier

        Returns:
            True if removed, False if not found
        """
        if feature_id in self._features:
            del self._features[feature_id]
            self._updated_at = datetime.now().isoformat()

            logger.debug("feature_removed", feature_id=feature_id)

            if self._auto_save:
                self.save()
            return True
        return False

    def mark_status(
        self,
        feature_id: str,
        status: FeatureStatus,
        error_message: str | None = None,
    ) -> None:
        """Update feature status.

        Args:
            feature_id: Feature identifier
            status: New status
            error_message: Error details (for failing status)

        Raises:
            KeyError: If feature not found
        """
        feature = self._features.get(feature_id)
        if not feature:
            raise KeyError(f"Feature '{feature_id}' not found")

        feature.status = status
        feature.last_tested = datetime.now().isoformat()
        if error_message:
            feature.error_message = error_message
        elif status == FeatureStatus.PASSING:
            feature.error_message = None

        self._updated_at = datetime.now().isoformat()

        logger.debug(
            "feature_status_changed",
            feature_id=feature_id,
            status=status.value,
        )

        if self._auto_save:
            self.save()

    def mark_passing(self, feature_id: str) -> None:
        """Mark feature as passing."""
        self.mark_status(feature_id, FeatureStatus.PASSING)

    def mark_failing(self, feature_id: str, error: str | None = None) -> None:
        """Mark feature as failing."""
        self.mark_status(feature_id, FeatureStatus.FAILING, error)

    def mark_in_progress(self, feature_id: str) -> None:
        """Mark feature as in progress."""
        self.mark_status(feature_id, FeatureStatus.IN_PROGRESS)

    def list_all(self) -> list[Feature]:
        """Get all features sorted by priority."""
        return sorted(self._features.values(), key=lambda f: (f.priority, f.id))

    def list_by_status(self, status: FeatureStatus) -> list[Feature]:
        """Get features with specific status."""
        return [f for f in self.list_all() if f.status == status]

    def list_pending(self) -> list[Feature]:
        """Get pending features."""
        return self.list_by_status(FeatureStatus.PENDING)

    def list_failing(self) -> list[Feature]:
        """Get failing features."""
        return self.list_by_status(FeatureStatus.FAILING)

    def list_passing(self) -> list[Feature]:
        """Get passing features."""
        return self.list_by_status(FeatureStatus.PASSING)

    def next_feature(self) -> Feature | None:
        """Get next feature to work on.

        Returns highest priority pending or failing feature.
        """
        # First check in-progress
        in_progress = self.list_by_status(FeatureStatus.IN_PROGRESS)
        if in_progress:
            return in_progress[0]

        # Then failing (need to fix)
        failing = self.list_failing()
        if failing:
            return failing[0]

        # Then pending
        pending = self.list_pending()
        if pending:
            return pending[0]

        return None

    def summary(self) -> dict[str, int]:
        """Get status summary counts."""
        counts: dict[str, int] = {}
        for status in FeatureStatus:
            counts[status.value] = len(self.list_by_status(status))
        counts["total"] = len(self._features)
        return counts

    def progress_percent(self) -> float:
        """Get completion percentage (passing / total)."""
        total = len(self._features)
        if total == 0:
            return 0.0
        passing = len(self.list_passing())
        return (passing / total) * 100

    def save(self) -> None:
        """Save feature list to JSON file."""
        data = {
            "project_name": self._project_name,
            "created_at": self._created_at,
            "updated_at": datetime.now().isoformat(),
            "summary": self.summary(),
            "features": [f.to_dict() for f in self.list_all()],
        }

        # Ensure parent directory exists
        self._path.parent.mkdir(parents=True, exist_ok=True)

        # Write atomically via temp file
        temp_path = self._path.with_suffix(".tmp")
        with temp_path.open("w") as f:
            json.dump(data, f, indent=2)

        temp_path.replace(self._path)

        logger.debug("feature_list_saved", path=str(self._path))

    def load(self) -> None:
        """Load feature list from JSON file."""
        with self._path.open() as f:
            data = json.load(f)

        self._project_name = data.get("project_name")
        self._created_at = data.get("created_at")
        self._updated_at = data.get("updated_at")

        self._features = {}
        for feature_data in data.get("features", []):
            feature = Feature.from_dict(feature_data)
            self._features[feature.id] = feature

        logger.debug(
            "feature_list_loaded",
            path=str(self._path),
            count=len(self._features),
        )

    def __len__(self) -> int:
        """Get number of features."""
        return len(self._features)

    def __contains__(self, feature_id: str) -> bool:
        """Check if feature exists."""
        return feature_id in self._features


@dataclass
class ProgressEntry:
    """A single progress log entry.

    Attributes:
        timestamp: ISO timestamp
        session_id: Session that made progress
        agent_name: Agent that performed work
        action: What was done
        feature_id: Related feature (if any)
        details: Additional details
        git_commit: Git commit SHA (if any)
    """

    timestamp: str
    session_id: str | None
    agent_name: str | None
    action: str
    feature_id: str | None = None
    details: str | None = None
    git_commit: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "agent_name": self.agent_name,
            "action": self.action,
            "feature_id": self.feature_id,
            "details": self.details,
            "git_commit": self.git_commit,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProgressEntry:
        """Create from dictionary."""
        return cls(
            timestamp=data["timestamp"],
            session_id=data.get("session_id"),
            agent_name=data.get("agent_name"),
            action=data["action"],
            feature_id=data.get("feature_id"),
            details=data.get("details"),
            git_commit=data.get("git_commit"),
        )

    def format_line(self) -> str:
        """Format as single log line."""
        parts = [f"[{self.timestamp}]"]
        if self.agent_name:
            parts.append(f"({self.agent_name})")
        parts.append(self.action)
        if self.feature_id:
            parts.append(f"[{self.feature_id}]")
        if self.details:
            parts.append(f"- {self.details}")
        if self.git_commit:
            parts.append(f"(commit: {self.git_commit[:8]})")
        return " ".join(parts)


class ProgressTracker:
    """Cross-session progress tracking for agentic workflows.

    Maintains a progress file that logs agent work across sessions,
    enabling quick understanding of project status and continuity.

    Example:
        >>> tracker = ProgressTracker("progress.json")
        >>> tracker.log("Started working on authentication feature",
        ...             feature_id="auth-login",
        ...             agent_name="coding-agent")
        >>> tracker.log("Fixed login validation bug",
        ...             feature_id="auth-login",
        ...             git_commit="abc123")
        >>> tracker.save()
        >>> print(tracker.recent_summary())

    Attributes:
        path: Path to progress file
        entries: List of progress entries
    """

    def __init__(
        self,
        path: str | Path,
        max_entries: int = 1000,
        auto_save: bool = False,
    ) -> None:
        """Initialize progress tracker.

        Args:
            path: Path to progress file
            max_entries: Maximum entries to keep (oldest removed)
            auto_save: Automatically save after logging
        """
        self._path = Path(path)
        self._max_entries = max_entries
        self._auto_save = auto_save
        self._entries: list[ProgressEntry] = []
        self._session_id: str | None = None
        self._agent_name: str | None = None

        if self._path.exists():
            self.load()

    @property
    def path(self) -> Path:
        """Get file path."""
        return self._path

    def set_context(
        self,
        session_id: str | None = None,
        agent_name: str | None = None,
    ) -> None:
        """Set default context for log entries.

        Args:
            session_id: Default session ID
            agent_name: Default agent name
        """
        self._session_id = session_id
        self._agent_name = agent_name

    def log(
        self,
        action: str,
        feature_id: str | None = None,
        details: str | None = None,
        git_commit: str | None = None,
        session_id: str | None = None,
        agent_name: str | None = None,
    ) -> ProgressEntry:
        """Log a progress entry.

        Args:
            action: What was done
            feature_id: Related feature ID
            details: Additional details
            git_commit: Git commit SHA
            session_id: Override default session
            agent_name: Override default agent

        Returns:
            Created progress entry
        """
        entry = ProgressEntry(
            timestamp=datetime.now().isoformat(),
            session_id=session_id or self._session_id,
            agent_name=agent_name or self._agent_name,
            action=action,
            feature_id=feature_id,
            details=details,
            git_commit=git_commit,
        )

        self._entries.append(entry)

        # Trim if over max
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries :]

        logger.debug("progress_logged", action=action, feature_id=feature_id)

        if self._auto_save:
            self.save()

        return entry

    def log_feature_started(self, feature_id: str, description: str = "") -> ProgressEntry:
        """Log starting work on a feature."""
        return self.log(
            action="Started feature",
            feature_id=feature_id,
            details=description or None,
        )

    def log_feature_completed(self, feature_id: str, git_commit: str | None = None) -> ProgressEntry:
        """Log completing a feature."""
        return self.log(
            action="Completed feature",
            feature_id=feature_id,
            git_commit=git_commit,
        )

    def log_test_run(self, passed: bool, details: str | None = None) -> ProgressEntry:
        """Log a test run."""
        action = "Tests passed" if passed else "Tests failed"
        return self.log(action=action, details=details)

    def log_error(self, error: str, feature_id: str | None = None) -> ProgressEntry:
        """Log an error."""
        return self.log(
            action="Error encountered",
            feature_id=feature_id,
            details=error,
        )

    def log_session_start(self) -> ProgressEntry:
        """Log session start."""
        return self.log(action="Session started")

    def log_session_end(self, summary: str | None = None) -> ProgressEntry:
        """Log session end."""
        return self.log(action="Session ended", details=summary)

    def recent(self, count: int = 10) -> list[ProgressEntry]:
        """Get most recent entries.

        Args:
            count: Number of entries to return

        Returns:
            List of recent entries (newest first)
        """
        return list(reversed(self._entries[-count:]))

    def for_feature(self, feature_id: str) -> list[ProgressEntry]:
        """Get entries for a specific feature.

        Args:
            feature_id: Feature ID to filter by

        Returns:
            Entries related to the feature
        """
        return [e for e in self._entries if e.feature_id == feature_id]

    def for_session(self, session_id: str) -> list[ProgressEntry]:
        """Get entries for a specific session.

        Args:
            session_id: Session ID to filter by

        Returns:
            Entries from the session
        """
        return [e for e in self._entries if e.session_id == session_id]

    def recent_summary(self, count: int = 10) -> str:
        """Get formatted summary of recent progress.

        Args:
            count: Number of entries to include

        Returns:
            Formatted multi-line summary
        """
        entries = self.recent(count)
        if not entries:
            return "No progress logged yet."

        lines = ["Recent Progress:"]
        for entry in entries:
            lines.append(f"  {entry.format_line()}")
        return "\n".join(lines)

    def today_summary(self) -> str:
        """Get summary of today's progress."""
        today = datetime.now().date().isoformat()
        today_entries = [
            e for e in self._entries if e.timestamp.startswith(today)
        ]

        if not today_entries:
            return "No progress logged today."

        lines = [f"Today's Progress ({len(today_entries)} entries):"]
        for entry in today_entries:
            lines.append(f"  {entry.format_line()}")
        return "\n".join(lines)

    def save(self) -> None:
        """Save progress to file."""
        data = {
            "updated_at": datetime.now().isoformat(),
            "total_entries": len(self._entries),
            "entries": [e.to_dict() for e in self._entries],
        }

        # Ensure parent directory exists
        self._path.parent.mkdir(parents=True, exist_ok=True)

        # Write atomically
        temp_path = self._path.with_suffix(".tmp")
        with temp_path.open("w") as f:
            json.dump(data, f, indent=2)

        temp_path.replace(self._path)

        logger.debug("progress_saved", path=str(self._path), entries=len(self._entries))

    def load(self) -> None:
        """Load progress from file."""
        with self._path.open() as f:
            data = json.load(f)

        self._entries = [
            ProgressEntry.from_dict(e) for e in data.get("entries", [])
        ]

        logger.debug(
            "progress_loaded",
            path=str(self._path),
            entries=len(self._entries),
        )

    def clear(self) -> None:
        """Clear all entries."""
        self._entries = []
        if self._auto_save:
            self.save()

    def __len__(self) -> int:
        """Get number of entries."""
        return len(self._entries)


class AgentHarness:
    """Combines FeatureList and ProgressTracker for full harness workflow.

    Provides the complete two-agent architecture pattern with:
    - Feature tracking with JSON persistence
    - Progress logging across sessions
    - Session initialization protocol
    - Git integration helpers

    Example:
        >>> harness = AgentHarness(
        ...     project_dir="my_project",
        ...     project_name="My Project",
        ... )
        >>> harness.init_session("coding-agent", "session-123")
        >>>
        >>> # Get next feature to work on
        >>> feature = harness.features.next_feature()
        >>> if feature:
        ...     harness.start_feature(feature.id)
        ...     # ... do work ...
        ...     harness.complete_feature(feature.id)
        >>>
        >>> harness.end_session("Completed 3 features")
    """

    def __init__(
        self,
        project_dir: str | Path,
        project_name: str | None = None,
        features_file: str = "features.json",
        progress_file: str = "progress.json",
        auto_save: bool = True,
    ) -> None:
        """Initialize agent harness.

        Args:
            project_dir: Project directory
            project_name: Project identifier
            features_file: Name of features JSON file
            progress_file: Name of progress JSON file
            auto_save: Auto-save after modifications
        """
        self._project_dir = Path(project_dir)
        self._project_name = project_name
        self._auto_save = auto_save

        # Initialize feature list
        self._features = FeatureList(
            self._project_dir / features_file,
            project_name=project_name,
            auto_save=auto_save,
        )

        # Initialize progress tracker
        self._progress = ProgressTracker(
            self._project_dir / progress_file,
            auto_save=auto_save,
        )

        self._current_feature: str | None = None
        self._session_id: str | None = None
        self._agent_name: str | None = None

    @property
    def features(self) -> FeatureList:
        """Get feature list."""
        return self._features

    @property
    def progress(self) -> ProgressTracker:
        """Get progress tracker."""
        return self._progress

    @property
    def project_dir(self) -> Path:
        """Get project directory."""
        return self._project_dir

    def init_session(
        self,
        agent_name: str,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Initialize a new agent session.

        Follows the session initialization protocol:
        1. Set context
        2. Log session start
        3. Return status summary

        Args:
            agent_name: Name of the agent
            session_id: Optional session ID

        Returns:
            Status summary dict
        """
        self._agent_name = agent_name
        self._session_id = session_id
        self._progress.set_context(session_id=session_id, agent_name=agent_name)

        self._progress.log_session_start()

        summary = self._features.summary()
        next_feature = self._features.next_feature()

        status = {
            "project_name": self._project_name,
            "agent_name": agent_name,
            "session_id": session_id,
            "feature_summary": summary,
            "progress_percent": self._features.progress_percent(),
            "next_feature": next_feature.to_dict() if next_feature else None,
            "recent_progress": self._progress.recent(5),
        }

        logger.info(
            "harness_session_initialized",
            agent=agent_name,
            features=summary["total"],
            progress=f"{status['progress_percent']:.1f}%",
        )

        return status

    def end_session(self, summary: str | None = None) -> None:
        """End the current session.

        Args:
            summary: Optional session summary
        """
        self._progress.log_session_end(summary)
        self._current_feature = None

        logger.info("harness_session_ended", summary=summary)

    def start_feature(self, feature_id: str) -> Feature | None:
        """Start working on a feature.

        Args:
            feature_id: Feature to start

        Returns:
            The feature, or None if not found
        """
        feature = self._features.get(feature_id)
        if not feature:
            return None

        self._current_feature = feature_id
        self._features.mark_in_progress(feature_id)
        self._progress.log_feature_started(feature_id, feature.description)

        logger.debug("feature_started", feature_id=feature_id)

        return feature

    def complete_feature(
        self,
        feature_id: str | None = None,
        git_commit: str | None = None,
    ) -> None:
        """Mark current or specified feature as complete.

        Args:
            feature_id: Feature to complete (defaults to current)
            git_commit: Git commit SHA
        """
        fid = feature_id or self._current_feature
        if not fid:
            raise ValueError("No feature specified or in progress")

        self._features.mark_passing(fid)
        self._progress.log_feature_completed(fid, git_commit)

        if fid == self._current_feature:
            self._current_feature = None

        logger.debug("feature_completed", feature_id=fid)

    def fail_feature(
        self,
        error: str,
        feature_id: str | None = None,
    ) -> None:
        """Mark current or specified feature as failing.

        Args:
            error: Error description
            feature_id: Feature to fail (defaults to current)
        """
        fid = feature_id or self._current_feature
        if not fid:
            raise ValueError("No feature specified or in progress")

        self._features.mark_failing(fid, error)
        self._progress.log_error(error, fid)

        if fid == self._current_feature:
            self._current_feature = None

        logger.debug("feature_failed", feature_id=fid, error=error)

    def get_git_commit(self) -> str | None:
        """Get current git commit SHA.

        Returns:
            Commit SHA or None if not in git repo
        """
        try:
            import subprocess

            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self._project_dir,
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    def status_report(self) -> str:
        """Generate a full status report.

        Returns:
            Formatted status report
        """
        summary = self._features.summary()
        progress = self._features.progress_percent()

        lines = [
            f"# {self._project_name or 'Project'} Status Report",
            f"Generated: {datetime.now().isoformat()}",
            "",
            "## Feature Summary",
            f"  Total: {summary['total']}",
            f"  Passing: {summary['passing']}",
            f"  Failing: {summary['failing']}",
            f"  In Progress: {summary['in_progress']}",
            f"  Pending: {summary['pending']}",
            f"  Progress: {progress:.1f}%",
            "",
        ]

        # Failing features (need attention)
        failing = self._features.list_failing()
        if failing:
            lines.append("## Failing Features (Need Attention)")
            for f in failing:
                lines.append(f"  - [{f.id}] {f.description}")
                if f.error_message:
                    lines.append(f"    Error: {f.error_message}")
            lines.append("")

        # Next feature
        next_f = self._features.next_feature()
        if next_f:
            lines.append("## Next Feature")
            lines.append(f"  [{next_f.id}] {next_f.description}")
            lines.append("")

        # Recent progress
        lines.append("## Recent Progress")
        for entry in self._progress.recent(10):
            lines.append(f"  {entry.format_line()}")

        return "\n".join(lines)


# Global harness singleton
_global_harness: AgentHarness | None = None


def get_harness() -> AgentHarness | None:
    """Get the global harness instance.

    Returns:
        Global AgentHarness or None if not initialized
    """
    return _global_harness


def init_harness(
    project_dir: str | Path,
    project_name: str | None = None,
    **kwargs: Any,
) -> AgentHarness:
    """Initialize the global harness.

    Args:
        project_dir: Project directory
        project_name: Project name
        **kwargs: Additional AgentHarness arguments

    Returns:
        Initialized AgentHarness
    """
    global _global_harness
    _global_harness = AgentHarness(
        project_dir=project_dir,
        project_name=project_name,
        **kwargs,
    )
    return _global_harness


def reset_harness() -> None:
    """Reset the global harness (for testing)."""
    global _global_harness
    _global_harness = None
