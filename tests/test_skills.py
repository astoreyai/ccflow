"""
Tests for skill system.

Tests SkillDefinition, SkillLoader, and skill matching.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from ccflow.skills import (
    SkillDefinition,
    SkillLoader,
    get_skill_loader,
    parse_skill_md,
    reset_skill_loader,
)


class TestParseSkillMd:
    """Tests for SKILL.md parsing."""

    def test_parse_with_frontmatter(self):
        """Parse content with valid frontmatter."""
        content = """---
name: my-skill
description: A helpful skill
---
These are the instructions.
"""
        metadata, body = parse_skill_md(content)

        assert metadata["name"] == "my-skill"
        assert metadata["description"] == "A helpful skill"
        assert body == "These are the instructions."

    def test_parse_without_frontmatter(self):
        """Parse content without frontmatter."""
        content = "Just instructions without frontmatter."
        metadata, body = parse_skill_md(content)

        assert metadata == {}
        assert body == content

    def test_parse_empty_frontmatter(self):
        """Parse content with empty frontmatter."""
        content = """---
---
Instructions after empty frontmatter.
"""
        metadata, body = parse_skill_md(content)

        assert metadata == {}
        assert "Instructions" in body

    def test_parse_complex_frontmatter(self):
        """Parse content with complex frontmatter."""
        content = """---
name: complex-skill
description: Complex skill
tags:
  - code
  - review
metadata:
  version: "1.0"
---
Multi-line
instructions
here.
"""
        metadata, body = parse_skill_md(content)

        assert metadata["name"] == "complex-skill"
        assert metadata["tags"] == ["code", "review"]
        assert metadata["metadata"]["version"] == "1.0"
        assert "Multi-line" in body


class TestSkillDefinition:
    """Tests for SkillDefinition dataclass."""

    def test_create_minimal(self):
        """Create skill with required fields."""
        skill = SkillDefinition(
            name="test-skill",
            description="A test skill",
            instructions="Do the test.",
        )

        assert skill.name == "test-skill"
        assert skill.description == "A test skill"
        assert skill.instructions == "Do the test."
        assert skill.location == "user"
        assert skill.resources_path is None
        assert skill.metadata == {}

    def test_create_full(self):
        """Create skill with all fields."""
        resources = Path("/tmp/resources")

        skill = SkillDefinition(
            name="full-skill",
            description="Full skill",
            instructions="Full instructions",
            location="project",
            resources_path=resources,
            metadata={"version": "2.0"},
        )

        assert skill.name == "full-skill"
        assert skill.location == "project"
        assert skill.resources_path == resources
        assert skill.metadata == {"version": "2.0"}


class TestSkillLoader:
    """Tests for SkillLoader."""

    def setup_method(self):
        """Reset global loader before each test."""
        reset_skill_loader()

    def test_discover_from_filesystem(self):
        """Discover skills from filesystem."""
        with tempfile.TemporaryDirectory() as tmpdir:
            skills_dir = Path(tmpdir) / "skills"
            skills_dir.mkdir()

            # Create skill directory with SKILL.md
            skill_dir = skills_dir / "my-skill"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text("""---
name: my-skill
description: My custom skill
---
These are the skill instructions.
""")

            loader = SkillLoader()
            loader._search_paths = [skills_dir]

            discovered = loader.discover()

            assert len(discovered) == 1
            assert discovered[0].name == "my-skill"
            assert discovered[0].description == "My custom skill"
            assert "instructions" in discovered[0].instructions

    def test_discover_skill_file(self):
        """Discover single skill file (not directory)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            skills_dir = Path(tmpdir) / "skills"
            skills_dir.mkdir()

            (skills_dir / "quick-skill.md").write_text("""---
name: quick-skill
description: Quick skill file
---
Quick instructions.
""")

            loader = SkillLoader()
            loader._search_paths = [skills_dir]

            discovered = loader.discover()

            assert len(discovered) == 1
            assert discovered[0].name == "quick-skill"

    def test_discover_with_resources(self):
        """Discover skill with resources directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            skills_dir = Path(tmpdir) / "skills"
            skill_dir = skills_dir / "resource-skill"
            resources_dir = skill_dir / "resources"
            resources_dir.mkdir(parents=True)

            (skill_dir / "SKILL.md").write_text("""---
description: Skill with resources
---
Instructions here.
""")
            (resources_dir / "data.txt").write_text("Resource data")

            loader = SkillLoader()
            loader._search_paths = [skills_dir]

            discovered = loader.discover()

            assert len(discovered) == 1
            assert discovered[0].resources_path == resources_dir

    def test_get_skill(self):
        """Get skill by name."""
        loader = SkillLoader()
        loader._search_paths = []  # Disable discovery

        loader.register(SkillDefinition(
            name="registered-skill",
            description="Registered skill",
            instructions="Instructions",
        ))

        skill = loader.get("registered-skill")

        assert skill is not None
        assert skill.name == "registered-skill"

    def test_get_nonexistent(self):
        """Get nonexistent skill returns None."""
        loader = SkillLoader()
        loader._search_paths = []  # Disable discovery

        result = loader.get("nonexistent")
        assert result is None

    def test_list_skills(self):
        """List registered skills."""
        loader = SkillLoader()
        loader._search_paths = []

        loader.register(SkillDefinition(
            name="skill-a",
            description="Skill A",
            instructions="A",
        ))
        loader.register(SkillDefinition(
            name="skill-b",
            description="Skill B",
            instructions="B",
        ))

        names = loader.list()
        assert set(names) == {"skill-a", "skill-b"}

    def test_match_by_name(self):
        """Match skills by name."""
        loader = SkillLoader()
        loader._search_paths = []

        loader.register(SkillDefinition(
            name="code-review",
            description="Reviews code for quality",
            instructions="...",
        ))
        loader.register(SkillDefinition(
            name="testing",
            description="Testing utilities",
            instructions="...",
        ))

        matches = loader.match("help me review code")

        assert len(matches) > 0
        assert matches[0].name == "code-review"

    def test_match_by_description(self):
        """Match skills by description keywords."""
        loader = SkillLoader()
        loader._search_paths = []

        loader.register(SkillDefinition(
            name="analyzer",
            description="Analyzes security vulnerabilities in code",
            instructions="...",
        ))
        loader.register(SkillDefinition(
            name="formatter",
            description="Formats code nicely",
            instructions="...",
        ))

        matches = loader.match("check security vulnerabilities")

        assert len(matches) > 0
        assert matches[0].name == "analyzer"

    def test_match_returns_sorted(self):
        """Match returns skills sorted by relevance."""
        loader = SkillLoader()
        loader._search_paths = []

        loader.register(SkillDefinition(
            name="generic",
            description="Generic utility",
            instructions="...",
        ))
        loader.register(SkillDefinition(
            name="specific",
            description="Specific code review utility for quality",
            instructions="...",
        ))
        loader.register(SkillDefinition(
            name="code-review",
            description="Code review helper",
            instructions="...",
        ))

        matches = loader.match("code review")

        # "code-review" should be first (name match)
        assert matches[0].name == "code-review"

    def test_no_matches(self):
        """No matches returns empty list."""
        loader = SkillLoader()
        loader._search_paths = []

        loader.register(SkillDefinition(
            name="unrelated",
            description="Completely unrelated skill",
            instructions="...",
        ))

        matches = loader.match("quantum physics calculation")
        assert len(matches) == 0

    def test_get_resource(self):
        """Get resource from skill."""
        with tempfile.TemporaryDirectory() as tmpdir:
            skills_dir = Path(tmpdir) / "skills"
            skill_dir = skills_dir / "my-skill"
            resources_dir = skill_dir / "resources"
            resources_dir.mkdir(parents=True)

            (skill_dir / "SKILL.md").write_text("""---
description: Skill with resources
---
Instructions.
""")
            (resources_dir / "template.txt").write_text("Template content here")

            loader = SkillLoader()
            loader._search_paths = [skills_dir]
            loader.discover()

            resource = loader.get_resource("my-skill", "template.txt")

            assert resource == "Template content here"

    def test_get_resource_not_found(self):
        """Get nonexistent resource returns None."""
        loader = SkillLoader()
        loader._search_paths = []

        loader.register(SkillDefinition(
            name="no-resources",
            description="No resources",
            instructions="...",
        ))

        result = loader.get_resource("no-resources", "anything.txt")
        assert result is None

    def test_add_search_path(self):
        """Add custom search path."""
        loader = SkillLoader()
        custom_path = Path("/custom/skills")

        loader.add_search_path(custom_path)

        assert custom_path in loader._search_paths

    def test_register_skill(self):
        """Register skill programmatically."""
        loader = SkillLoader()
        loader._search_paths = []

        skill = SkillDefinition(
            name="programmatic-skill",
            description="Added programmatically",
            instructions="Instructions",
        )
        loader.register(skill)

        found = loader.get("programmatic-skill")
        assert found is not None
        assert found.name == "programmatic-skill"

    def test_clear(self):
        """Clear removes all skills."""
        loader = SkillLoader()
        loader._search_paths = []

        loader.register(SkillDefinition(
            name="skill-to-clear",
            description="Will be cleared",
            instructions="...",
        ))
        assert len(loader.list()) == 1

        loader.clear()
        assert len(loader.list()) == 0

    def test_global_singleton(self):
        """Global loader returns same instance."""
        reset_skill_loader()

        loader1 = get_skill_loader()
        loader2 = get_skill_loader()

        assert loader1 is loader2


class TestSkillLocationDetection:
    """Tests for skill location detection."""

    def test_user_location(self):
        """Skills from user directory are marked as user."""
        with tempfile.TemporaryDirectory() as tmpdir:
            skills_dir = Path(tmpdir) / "skills"
            skills_dir.mkdir()

            (skills_dir / "user-skill.md").write_text("""---
description: User skill
---
Instructions.
""")

            loader = SkillLoader()
            loader._search_paths = [skills_dir]

            discovered = loader.discover()

            assert len(discovered) == 1
            assert discovered[0].location == "user"


class TestSkillDescriptionExtraction:
    """Tests for extracting description from content."""

    def test_uses_frontmatter_description(self):
        """Uses description from frontmatter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            skills_dir = Path(tmpdir) / "skills"
            skills_dir.mkdir()

            (skills_dir / "skill.md").write_text("""---
description: Frontmatter description
---
Body content here.
""")

            loader = SkillLoader()
            loader._search_paths = [skills_dir]

            discovered = loader.discover()

            assert discovered[0].description == "Frontmatter description"

    def test_extracts_from_body(self):
        """Extracts description from body when no frontmatter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            skills_dir = Path(tmpdir) / "skills"
            skills_dir.mkdir()

            (skills_dir / "skill.md").write_text("""---
name: body-skill
---
This is extracted as description.

More content below.
""")

            loader = SkillLoader()
            loader._search_paths = [skills_dir]

            discovered = loader.discover()

            assert "extracted as description" in discovered[0].description
