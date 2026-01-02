"""Tests for agentic harness utilities."""

import json
import tempfile
from pathlib import Path

import pytest

from ccflow.harness import (
    AgentHarness,
    Feature,
    FeatureList,
    FeatureStatus,
    ProgressEntry,
    ProgressTracker,
    get_harness,
    init_harness,
    reset_harness,
)


class TestFeature:
    """Tests for Feature dataclass."""

    def test_create_feature(self):
        """Test feature creation with defaults."""
        feature = Feature(id="test-1", description="Test feature")
        assert feature.id == "test-1"
        assert feature.description == "Test feature"
        assert feature.status == FeatureStatus.PENDING
        assert feature.priority == 5
        assert feature.tags == []

    def test_feature_with_all_fields(self):
        """Test feature with all fields."""
        feature = Feature(
            id="auth-login",
            description="User login",
            status=FeatureStatus.PASSING,
            priority=1,
            tags=["auth", "critical"],
            notes="Important feature",
            test_command="pytest tests/test_auth.py",
            last_tested="2024-01-01T00:00:00",
            error_message=None,
            metadata={"owner": "alice"},
        )
        assert feature.priority == 1
        assert "auth" in feature.tags
        assert feature.metadata["owner"] == "alice"

    def test_feature_to_dict(self):
        """Test feature serialization."""
        feature = Feature(
            id="test-1",
            description="Test",
            status=FeatureStatus.FAILING,
            priority=2,
        )
        data = feature.to_dict()
        assert data["id"] == "test-1"
        assert data["status"] == "failing"
        assert data["priority"] == 2

    def test_feature_from_dict(self):
        """Test feature deserialization."""
        data = {
            "id": "test-1",
            "description": "Test",
            "status": "passing",
            "priority": 1,
            "tags": ["core"],
        }
        feature = Feature.from_dict(data)
        assert feature.id == "test-1"
        assert feature.status == FeatureStatus.PASSING
        assert feature.tags == ["core"]

    def test_feature_roundtrip(self):
        """Test feature serialization roundtrip."""
        original = Feature(
            id="test-1",
            description="Test feature",
            status=FeatureStatus.IN_PROGRESS,
            priority=3,
            tags=["tag1", "tag2"],
            notes="Some notes",
            metadata={"key": "value"},
        )
        data = original.to_dict()
        restored = Feature.from_dict(data)
        assert restored.id == original.id
        assert restored.status == original.status
        assert restored.tags == original.tags
        assert restored.metadata == original.metadata


class TestFeatureList:
    """Tests for FeatureList."""

    def test_create_empty_list(self):
        """Test creating empty feature list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "features.json"
            fl = FeatureList(path)
            assert len(fl) == 0
            assert fl.project_name is None

    def test_add_feature(self):
        """Test adding features."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "features.json"
            fl = FeatureList(path)

            fl.add(Feature(id="f1", description="Feature 1"))
            fl.add(Feature(id="f2", description="Feature 2"))

            assert len(fl) == 2
            assert "f1" in fl
            assert "f2" in fl

    def test_add_duplicate_raises(self):
        """Test adding duplicate feature raises."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "features.json"
            fl = FeatureList(path)

            fl.add(Feature(id="f1", description="Feature 1"))
            with pytest.raises(ValueError, match="already exists"):
                fl.add(Feature(id="f1", description="Duplicate"))

    def test_get_feature(self):
        """Test getting feature by ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "features.json"
            fl = FeatureList(path)

            fl.add(Feature(id="f1", description="Feature 1"))
            feature = fl.get("f1")

            assert feature is not None
            assert feature.description == "Feature 1"
            assert fl.get("nonexistent") is None

    def test_update_feature(self):
        """Test updating feature."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "features.json"
            fl = FeatureList(path)

            fl.add(Feature(id="f1", description="Original"))
            fl.update(Feature(id="f1", description="Updated"))

            assert fl.get("f1").description == "Updated"

    def test_update_nonexistent_raises(self):
        """Test updating nonexistent feature raises."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "features.json"
            fl = FeatureList(path)

            with pytest.raises(KeyError):
                fl.update(Feature(id="f1", description="Test"))

    def test_remove_feature(self):
        """Test removing feature."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "features.json"
            fl = FeatureList(path)

            fl.add(Feature(id="f1", description="Feature 1"))
            assert fl.remove("f1") is True
            assert "f1" not in fl
            assert fl.remove("f1") is False

    def test_mark_status(self):
        """Test marking feature status."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "features.json"
            fl = FeatureList(path)

            fl.add(Feature(id="f1", description="Feature 1"))

            fl.mark_passing("f1")
            assert fl.get("f1").status == FeatureStatus.PASSING
            assert fl.get("f1").last_tested is not None

            fl.mark_failing("f1", "Test failed")
            assert fl.get("f1").status == FeatureStatus.FAILING
            assert fl.get("f1").error_message == "Test failed"

            fl.mark_in_progress("f1")
            assert fl.get("f1").status == FeatureStatus.IN_PROGRESS

    def test_mark_nonexistent_raises(self):
        """Test marking nonexistent feature raises."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "features.json"
            fl = FeatureList(path)

            with pytest.raises(KeyError):
                fl.mark_passing("nonexistent")

    def test_list_by_status(self):
        """Test listing features by status."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "features.json"
            fl = FeatureList(path)

            fl.add(Feature(id="f1", description="F1", status=FeatureStatus.PASSING))
            fl.add(Feature(id="f2", description="F2", status=FeatureStatus.FAILING))
            fl.add(Feature(id="f3", description="F3", status=FeatureStatus.PENDING))
            fl.add(Feature(id="f4", description="F4", status=FeatureStatus.PASSING))

            assert len(fl.list_passing()) == 2
            assert len(fl.list_failing()) == 1
            assert len(fl.list_pending()) == 1

    def test_next_feature(self):
        """Test getting next feature to work on."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "features.json"
            fl = FeatureList(path)

            fl.add(Feature(id="f1", description="F1", priority=3))
            fl.add(Feature(id="f2", description="F2", priority=1))
            fl.add(Feature(id="f3", description="F3", priority=2))

            # Should return highest priority pending
            next_f = fl.next_feature()
            assert next_f.id == "f2"

            # Mark f2 as passing, should get f3 next
            fl.mark_passing("f2")
            next_f = fl.next_feature()
            assert next_f.id == "f3"

    def test_next_feature_prioritizes_failing(self):
        """Test that failing features are prioritized."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "features.json"
            fl = FeatureList(path)

            fl.add(Feature(id="f1", description="F1", priority=1, status=FeatureStatus.PENDING))
            fl.add(Feature(id="f2", description="F2", priority=3, status=FeatureStatus.FAILING))

            # Should return failing feature first (needs fix)
            next_f = fl.next_feature()
            assert next_f.id == "f2"

    def test_next_feature_prioritizes_in_progress(self):
        """Test that in-progress features are prioritized."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "features.json"
            fl = FeatureList(path)

            fl.add(Feature(id="f1", description="F1", status=FeatureStatus.FAILING))
            fl.add(Feature(id="f2", description="F2", status=FeatureStatus.IN_PROGRESS))

            # Should return in-progress first
            next_f = fl.next_feature()
            assert next_f.id == "f2"

    def test_summary(self):
        """Test summary generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "features.json"
            fl = FeatureList(path)

            fl.add(Feature(id="f1", description="F1", status=FeatureStatus.PASSING))
            fl.add(Feature(id="f2", description="F2", status=FeatureStatus.FAILING))
            fl.add(Feature(id="f3", description="F3", status=FeatureStatus.PENDING))

            summary = fl.summary()
            assert summary["total"] == 3
            assert summary["passing"] == 1
            assert summary["failing"] == 1
            assert summary["pending"] == 1

    def test_progress_percent(self):
        """Test progress percentage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "features.json"
            fl = FeatureList(path)

            fl.add(Feature(id="f1", description="F1", status=FeatureStatus.PASSING))
            fl.add(Feature(id="f2", description="F2", status=FeatureStatus.PASSING))
            fl.add(Feature(id="f3", description="F3", status=FeatureStatus.PENDING))
            fl.add(Feature(id="f4", description="F4", status=FeatureStatus.PENDING))

            assert fl.progress_percent() == 50.0

    def test_save_and_load(self):
        """Test saving and loading feature list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "features.json"

            # Create and save
            fl1 = FeatureList(path, project_name="Test Project")
            fl1.add(Feature(id="f1", description="F1", priority=1))
            fl1.add(Feature(id="f2", description="F2", status=FeatureStatus.PASSING))
            fl1.save()

            # Load in new instance
            fl2 = FeatureList(path)
            assert fl2.project_name == "Test Project"
            assert len(fl2) == 2
            assert fl2.get("f1").priority == 1
            assert fl2.get("f2").status == FeatureStatus.PASSING

    def test_auto_save(self):
        """Test auto-save functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "features.json"

            fl = FeatureList(path, auto_save=True)
            fl.add(Feature(id="f1", description="F1"))

            # Should be saved automatically
            assert path.exists()
            with open(path) as f:
                data = json.load(f)
            assert len(data["features"]) == 1


class TestProgressEntry:
    """Tests for ProgressEntry."""

    def test_create_entry(self):
        """Test entry creation."""
        entry = ProgressEntry(
            timestamp="2024-01-01T00:00:00",
            session_id="sess-1",
            agent_name="test-agent",
            action="Did something",
        )
        assert entry.action == "Did something"
        assert entry.session_id == "sess-1"

    def test_entry_to_dict(self):
        """Test entry serialization."""
        entry = ProgressEntry(
            timestamp="2024-01-01T00:00:00",
            session_id="sess-1",
            agent_name="test-agent",
            action="Test action",
            feature_id="f1",
        )
        data = entry.to_dict()
        assert data["action"] == "Test action"
        assert data["feature_id"] == "f1"

    def test_entry_from_dict(self):
        """Test entry deserialization."""
        data = {
            "timestamp": "2024-01-01T00:00:00",
            "session_id": "sess-1",
            "agent_name": "agent",
            "action": "Test",
            "git_commit": "abc123",
        }
        entry = ProgressEntry.from_dict(data)
        assert entry.git_commit == "abc123"

    def test_format_line(self):
        """Test line formatting."""
        entry = ProgressEntry(
            timestamp="2024-01-01T00:00:00",
            session_id=None,
            agent_name="coding-agent",
            action="Fixed bug",
            feature_id="auth-login",
            git_commit="abc123def456",
        )
        line = entry.format_line()
        assert "2024-01-01" in line
        assert "coding-agent" in line
        assert "Fixed bug" in line
        assert "auth-login" in line
        assert "abc123de" in line  # Truncated commit


class TestProgressTracker:
    """Tests for ProgressTracker."""

    def test_create_tracker(self):
        """Test tracker creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "progress.json"
            tracker = ProgressTracker(path)
            assert len(tracker) == 0

    def test_log_entry(self):
        """Test logging entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "progress.json"
            tracker = ProgressTracker(path)

            entry = tracker.log("Did something", feature_id="f1")
            assert entry.action == "Did something"
            assert entry.feature_id == "f1"
            assert len(tracker) == 1

    def test_set_context(self):
        """Test setting default context."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "progress.json"
            tracker = ProgressTracker(path)

            tracker.set_context(session_id="sess-1", agent_name="agent-1")
            entry = tracker.log("Test action")

            assert entry.session_id == "sess-1"
            assert entry.agent_name == "agent-1"

    def test_log_with_override(self):
        """Test logging with context override."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "progress.json"
            tracker = ProgressTracker(path)

            tracker.set_context(agent_name="default-agent")
            entry = tracker.log("Test", agent_name="other-agent")

            assert entry.agent_name == "other-agent"

    def test_convenience_methods(self):
        """Test convenience logging methods."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "progress.json"
            tracker = ProgressTracker(path)

            tracker.log_feature_started("f1", "Feature description")
            tracker.log_feature_completed("f1", "abc123")
            tracker.log_test_run(passed=True)
            tracker.log_test_run(passed=False, details="Assertion failed")
            tracker.log_error("Something broke", feature_id="f2")
            tracker.log_session_start()
            tracker.log_session_end("All done")

            assert len(tracker) == 7

    def test_recent_entries(self):
        """Test getting recent entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "progress.json"
            tracker = ProgressTracker(path)

            for i in range(10):
                tracker.log(f"Action {i}")

            recent = tracker.recent(5)
            assert len(recent) == 5
            assert recent[0].action == "Action 9"  # Most recent first
            assert recent[4].action == "Action 5"

    def test_filter_by_feature(self):
        """Test filtering by feature."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "progress.json"
            tracker = ProgressTracker(path)

            tracker.log("Action 1", feature_id="f1")
            tracker.log("Action 2", feature_id="f2")
            tracker.log("Action 3", feature_id="f1")

            f1_entries = tracker.for_feature("f1")
            assert len(f1_entries) == 2

    def test_filter_by_session(self):
        """Test filtering by session."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "progress.json"
            tracker = ProgressTracker(path)

            tracker.log("Action 1", session_id="s1")
            tracker.log("Action 2", session_id="s2")
            tracker.log("Action 3", session_id="s1")

            s1_entries = tracker.for_session("s1")
            assert len(s1_entries) == 2

    def test_max_entries(self):
        """Test max entries limit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "progress.json"
            tracker = ProgressTracker(path, max_entries=5)

            for i in range(10):
                tracker.log(f"Action {i}")

            assert len(tracker) == 5
            # Should have kept the last 5
            assert tracker.recent(1)[0].action == "Action 9"

    def test_save_and_load(self):
        """Test saving and loading progress."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "progress.json"

            # Create and save
            tracker1 = ProgressTracker(path)
            tracker1.log("Action 1", feature_id="f1")
            tracker1.log("Action 2", git_commit="abc123")
            tracker1.save()

            # Load in new instance
            tracker2 = ProgressTracker(path)
            assert len(tracker2) == 2
            entries = tracker2.recent(2)
            assert entries[0].git_commit == "abc123"
            assert entries[1].feature_id == "f1"

    def test_recent_summary(self):
        """Test summary generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "progress.json"
            tracker = ProgressTracker(path)

            tracker.log("Action 1")
            tracker.log("Action 2")

            summary = tracker.recent_summary(5)
            assert "Recent Progress:" in summary
            assert "Action 1" in summary
            assert "Action 2" in summary

    def test_clear(self):
        """Test clearing entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "progress.json"
            tracker = ProgressTracker(path)

            tracker.log("Action 1")
            tracker.log("Action 2")
            tracker.clear()

            assert len(tracker) == 0


class TestAgentHarness:
    """Tests for AgentHarness."""

    def test_create_harness(self):
        """Test harness creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            harness = AgentHarness(tmpdir, project_name="Test")
            assert harness.project_dir == Path(tmpdir)
            assert len(harness.features) == 0
            assert len(harness.progress) == 0

    def test_init_session(self):
        """Test session initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            harness = AgentHarness(tmpdir, project_name="Test")
            harness.features.add(Feature(id="f1", description="Feature 1"))

            status = harness.init_session("coding-agent", "sess-1")

            assert status["project_name"] == "Test"
            assert status["agent_name"] == "coding-agent"
            assert status["feature_summary"]["total"] == 1
            assert status["next_feature"]["id"] == "f1"
            assert len(harness.progress) >= 1  # Session start logged

    def test_feature_workflow(self):
        """Test complete feature workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            harness = AgentHarness(tmpdir)
            harness.features.add(Feature(id="f1", description="Feature 1"))
            harness.init_session("agent", "sess-1")

            # Start feature
            feature = harness.start_feature("f1")
            assert feature.id == "f1"
            assert harness.features.get("f1").status == FeatureStatus.IN_PROGRESS

            # Complete feature
            harness.complete_feature(git_commit="abc123")
            assert harness.features.get("f1").status == FeatureStatus.PASSING

    def test_fail_feature(self):
        """Test failing a feature."""
        with tempfile.TemporaryDirectory() as tmpdir:
            harness = AgentHarness(tmpdir)
            harness.features.add(Feature(id="f1", description="Feature 1"))
            harness.init_session("agent")

            harness.start_feature("f1")
            harness.fail_feature("Test failed")

            assert harness.features.get("f1").status == FeatureStatus.FAILING
            assert harness.features.get("f1").error_message == "Test failed"

    def test_end_session(self):
        """Test ending session."""
        with tempfile.TemporaryDirectory() as tmpdir:
            harness = AgentHarness(tmpdir)
            harness.init_session("agent")
            initial_count = len(harness.progress)

            harness.end_session("Completed work")

            assert len(harness.progress) == initial_count + 1

    def test_status_report(self):
        """Test status report generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            harness = AgentHarness(tmpdir, project_name="Test Project")
            harness.features.add(Feature(id="f1", description="F1", status=FeatureStatus.PASSING))
            harness.features.add(Feature(id="f2", description="F2", status=FeatureStatus.FAILING))
            harness.features.mark_failing("f2", "Bug found")

            report = harness.status_report()

            assert "Test Project" in report
            assert "Feature Summary" in report
            assert "Failing Features" in report
            assert "Bug found" in report


class TestGlobalHarness:
    """Tests for global harness singleton."""

    def test_init_and_get(self):
        """Test initializing and getting global harness."""
        reset_harness()
        assert get_harness() is None

        with tempfile.TemporaryDirectory() as tmpdir:
            harness = init_harness(tmpdir, project_name="Global Test")
            assert get_harness() is harness
            assert get_harness().project_dir == Path(tmpdir)

        reset_harness()
        assert get_harness() is None

    def test_reset(self):
        """Test resetting global harness."""
        with tempfile.TemporaryDirectory() as tmpdir:
            init_harness(tmpdir)
            assert get_harness() is not None

            reset_harness()
            assert get_harness() is None
