#!/usr/bin/env python3
"""
Agentic Harness Example (v0.2.0)

Demonstrates feature tracking and progress management for long-running agent workflows.
Based on Anthropic's "Effective Harnesses for Long-Running Agents" best practices.

Key concepts:
- JSON-based feature tracking (less likely to be overwritten than Markdown)
- Cross-session progress file for continuity
- Session initialization protocol for agent restarts
"""

import asyncio
import tempfile
from pathlib import Path

from ccflow import (
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


async def demo_feature_list():
    """Demonstrate JSON-based feature tracking."""
    print("\n1. Feature List Tracking")
    print("-" * 40)

    with tempfile.TemporaryDirectory() as tmpdir:
        feature_path = Path(tmpdir) / "features.json"

        # Create feature list
        features = FeatureList(feature_path, auto_save=True)

        # Add features with priorities
        features.add(Feature(
            id="auth-system",
            description="Implement OAuth2 authentication",
            priority=1,
            tags=["security", "backend"],
        ))
        features.add(Feature(
            id="user-profile",
            description="Add user profile page",
            priority=2,
            tags=["frontend"],
        ))
        features.add(Feature(
            id="api-docs",
            description="Generate API documentation",
            priority=3,
            tags=["docs"],
        ))

        print(f"Added {len(features)} features")
        print(f"File: {feature_path}")

        # Work through features
        next_feature = features.next_feature()
        if next_feature:
            print(f"\nNext feature: {next_feature.id} (priority {next_feature.priority})")

            # Mark as in progress
            features.mark_in_progress(next_feature.id)
            print(f"Status: {features.get(next_feature.id).status.value}")

            # Complete it
            features.mark_passing(next_feature.id)
            print(f"Completed: {next_feature.id}")

        # Simulate a failing feature
        features.mark_in_progress("user-profile")
        features.mark_failing("user-profile", error="CSS layout broken on mobile")

        # Summary
        summary = features.summary()
        print(f"\nSummary: {summary}")
        print(f"Progress: {features.progress_percent():.0f}%")

        # Next feature prioritizes failing ones
        next_up = features.next_feature()
        print(f"Next (prioritizes failing): {next_up.id if next_up else 'None'}")


async def demo_progress_tracker():
    """Demonstrate cross-session progress tracking."""
    print("\n2. Progress Tracking")
    print("-" * 40)

    with tempfile.TemporaryDirectory() as tmpdir:
        progress_path = Path(tmpdir) / "progress.json"

        # Create tracker with session context
        tracker = ProgressTracker(progress_path, auto_save=True)
        tracker.set_context(session_id="session-001", agent_name="code-reviewer")

        # Log various actions
        tracker.log("session_start", details="Beginning code review")
        tracker.log_feature_started("auth-system", "OAuth2 implementation")
        tracker.log("checkpoint", feature_id="auth-system", details="JWT tokens working")
        tracker.log_feature_completed("auth-system", git_commit="abc123")
        tracker.log("session_end", details="Review complete")

        print(f"Logged {len(tracker)} entries")

        # View recent activity
        print("\nRecent activity:")
        for entry in tracker.recent(5):
            print(f"  {entry.format_line()}")

        # Filter by feature
        auth_entries = tracker.for_feature("auth-system")
        print(f"\nAuth-system entries: {len(auth_entries)}")

        # Summary for session restart
        print("\nProgress summary (for session restart):")
        print(tracker.recent_summary(10))


async def demo_agent_harness():
    """Demonstrate full harness workflow."""
    print("\n3. Agent Harness Workflow")
    print("-" * 40)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create harness with project directory
        harness = AgentHarness(
            project_dir=tmpdir,
            project_name="demo-project",
            auto_save=True,
        )

        # Add features
        harness.features.add(Feature(id="f1", description="First feature", priority=1))
        harness.features.add(Feature(id="f2", description="Second feature", priority=2))
        harness.features.add(Feature(id="f3", description="Third feature", priority=3))

        # Initialize session (returns context for agent)
        print("Initializing session...")
        context = harness.init_session(
            agent_name="feature-builder",
            session_id="harness-demo-001",
        )
        print(f"Session: {context['session_id']}")
        print(f"Features: {context['feature_summary']}")

        # Work on features
        print("\nWorking on features...")

        # Start first feature
        feature = harness.start_feature("f1")
        print(f"Started: {feature.id}")

        # Complete it
        harness.complete_feature(git_commit="commit-001")
        print(f"Completed: f1")

        # Start second, but it fails
        harness.start_feature("f2")
        harness.fail_feature("Database migration failed")
        print("Failed: f2")

        # End session
        report = harness.end_session()
        print(f"\nSession ended")

        # Status report (what agent sees on restart)
        print("\n" + "=" * 40)
        print("STATUS REPORT (for session restart):")
        print("=" * 40)
        print(harness.status_report())


async def demo_session_restart():
    """Demonstrate session restart with context recovery."""
    print("\n4. Session Restart Simulation")
    print("-" * 40)

    with tempfile.TemporaryDirectory() as tmpdir:
        # First session
        print("--- First Session ---")
        harness1 = AgentHarness(project_dir=tmpdir, auto_save=True)
        harness1.features.add(Feature(id="task-1", description="Complete task 1"))
        harness1.features.add(Feature(id="task-2", description="Complete task 2"))
        harness1.features.add(Feature(id="task-3", description="Complete task 3"))

        harness1.init_session("worker", "session-1")
        harness1.start_feature("task-1")
        harness1.complete_feature()
        harness1.start_feature("task-2")
        # Session interrupted mid-feature!
        print("Session interrupted during task-2")
        print(f"Progress: {harness1.features.progress_percent():.0f}%")

        # Second session (restart)
        print("\n--- Second Session (restart) ---")
        harness2 = AgentHarness(project_dir=tmpdir, auto_save=True)

        # Load previous state
        harness2.features.load()
        harness2.progress.load()

        context = harness2.init_session("worker", "session-2")
        print(f"Loaded {len(harness2.features)} features from disk")
        print(f"Progress: {harness2.features.progress_percent():.0f}%")

        # Agent sees what was in progress
        in_progress = harness2.features.list_by_status(FeatureStatus.IN_PROGRESS)
        print(f"In progress: {[f.id for f in in_progress]}")

        # Resume work
        harness2.complete_feature("task-2")
        harness2.start_feature("task-3")
        harness2.complete_feature()

        print(f"Final progress: {harness2.features.progress_percent():.0f}%")


async def demo_global_harness():
    """Demonstrate global harness singleton."""
    print("\n5. Global Harness Singleton")
    print("-" * 40)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Reset any existing global harness
        reset_harness()

        # Initialize global harness
        harness = init_harness(project_dir=tmpdir)

        # Access from anywhere
        h = get_harness()
        h.features.add(Feature(id="global-task", description="A task"))
        h.init_session("global-agent")
        h.start_feature("global-task")
        h.complete_feature()

        print(f"Global harness: {h.features.progress_percent():.0f}% complete")

        # Clean up
        reset_harness()


async def main():
    """Run all harness demos."""
    print("=" * 60)
    print("Agentic Harness Examples")
    print("=" * 60)

    await demo_feature_list()
    await demo_progress_tracker()
    await demo_agent_harness()
    await demo_session_restart()
    await demo_global_harness()

    print("\n" + "=" * 60)
    print("All harness demos complete!")
    print("\nKey takeaways:")
    print("- Use JSON for feature lists (harder for model to corrupt)")
    print("- Track progress across sessions for continuity")
    print("- Generate status reports for agent restarts")
    print("- Prioritize failing/in-progress features")


if __name__ == "__main__":
    asyncio.run(main())
