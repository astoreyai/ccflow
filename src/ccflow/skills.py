"""
Skill system for ccflow.

Provides SkillLoader for discovering and loading skills from
SKILL.md files in standard locations.
"""

from __future__ import annotations

import builtins
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Default search paths for skill definitions
SKILL_SEARCH_PATHS = [
    Path.home() / ".claude" / "skills",  # User skills
    Path.cwd() / ".claude" / "skills",  # Project skills
]


@dataclass
class SkillDefinition:
    """Definition of a skill (domain knowledge).

    Skills provide specialized knowledge and instructions that can
    be loaded on-demand based on semantic matching.

    Attributes:
        name: Unique identifier (lowercase-hyphen)
        description: When to use (semantic keywords for matching)
        instructions: Full SKILL.md content
        location: Source location ("user", "project", "plugin")
        resources_path: Path to resources/ directory (optional)
        metadata: Additional frontmatter fields
    """

    name: str
    description: str
    instructions: str
    location: str = "user"
    resources_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def parse_skill_md(content: str) -> tuple[dict[str, Any], str]:
    """Parse SKILL.md with YAML frontmatter.

    Args:
        content: Markdown content with optional YAML frontmatter

    Returns:
        Tuple of (metadata_dict, instructions_body)
    """
    if not content.startswith("---"):
        return {}, content

    # Find the closing ---
    end_match = re.search(r"\n---\s*\n", content[3:])
    if not end_match:
        return {}, content

    frontmatter_end = end_match.start() + 3
    frontmatter = content[3:frontmatter_end]
    body = content[frontmatter_end + end_match.end() - end_match.start() :]

    try:
        metadata = yaml.safe_load(frontmatter) or {}
    except yaml.YAMLError:
        metadata = {}

    return metadata, body.strip()


class SkillLoader:
    """Loads and matches skills with progressive disclosure.

    Discovers skills from filesystem locations and provides
    semantic matching for finding relevant skills.

    Example:
        >>> loader = SkillLoader()
        >>> skills = loader.match("help me review code")
        >>> for skill in skills:
        ...     print(f"Matched: {skill.name}")
        >>> full_skill = loader.load_full("code-reviewer")
        >>> print(full_skill.instructions)
    """

    def __init__(self) -> None:
        """Initialize skill loader."""
        self._skills: dict[str, SkillDefinition] = {}
        self._search_paths: list[Path] = list(SKILL_SEARCH_PATHS)
        self._discovered: bool = False

    def add_search_path(self, path: Path) -> None:
        """Add a search path for skill discovery.

        Args:
            path: Directory path to search
        """
        if path not in self._search_paths:
            self._search_paths.append(path)
            self._discovered = False  # Force re-discovery

    def discover(self) -> builtins.list[SkillDefinition]:
        """Discover skills from filesystem paths.

        Searches for SKILL.md files in configured search paths and
        parses them as skill definitions.

        Returns:
            List of discovered skill definitions
        """
        discovered = []

        for path in self._search_paths:
            if not path.exists():
                continue

            # Look for skill directories with SKILL.md
            for skill_dir in path.iterdir():
                if not skill_dir.is_dir():
                    continue

                skill_file = skill_dir / "SKILL.md"
                if not skill_file.exists():
                    continue

                try:
                    skill = self._load_skill_file(skill_file, skill_dir)
                    if skill:
                        self._skills[skill.name] = skill
                        discovered.append(skill)
                except Exception:
                    # Skip files that fail to parse
                    continue

            # Also look for single .md skill files
            for md_file in path.glob("*.md"):
                if md_file.name == "README.md":
                    continue

                try:
                    skill = self._load_skill_file(md_file)
                    if skill:
                        self._skills[skill.name] = skill
                        discovered.append(skill)
                except Exception:
                    continue

        self._discovered = True
        return discovered

    def _load_skill_file(
        self,
        path: Path,
        skill_dir: Path | None = None,
    ) -> SkillDefinition | None:
        """Load skill definition from SKILL.md file.

        Args:
            path: Path to SKILL.md file
            skill_dir: Parent skill directory (for resources)

        Returns:
            SkillDefinition or None if invalid
        """
        content = path.read_text()
        metadata, body = parse_skill_md(content)

        # Extract name from metadata or directory/filename
        if skill_dir:
            name = metadata.get("name", skill_dir.name)
        else:
            name = metadata.get("name", path.stem)

        description = metadata.get("description", "")

        if not description:
            # Try to extract from first paragraph
            lines = body.strip().split("\n\n")
            if lines:
                description = lines[0][:200].replace("\n", " ")

        if not description:
            return None

        # Check for resources directory
        resources_path = None
        if skill_dir:
            resources = skill_dir / "resources"
            if resources.exists():
                resources_path = resources

        # Determine location based on path
        location = "user"
        if Path.cwd() in path.parents or path.is_relative_to(Path.cwd()):
            location = "project"

        return SkillDefinition(
            name=name,
            description=description,
            instructions=body,
            location=location,
            resources_path=resources_path,
            metadata={k: v for k, v in metadata.items() if k not in {"name", "description"}},
        )

    def get(self, name: str) -> SkillDefinition | None:
        """Get skill by name.

        Auto-discovers skills from filesystem on first access.

        Args:
            name: Skill name to look up

        Returns:
            Skill definition or None if not found
        """
        if not self._discovered:
            self.discover()

        return self._skills.get(name)

    def list(self) -> builtins.list[str]:
        """List all discovered skill names.

        Auto-discovers skills from filesystem on first access.

        Returns:
            List of skill names
        """
        if not self._discovered:
            self.discover()

        return builtins.list(self._skills.keys())

    def match(self, query: str) -> builtins.list[SkillDefinition]:
        """Semantic matching of query against skill descriptions.

        Uses keyword-based matching to find relevant skills.

        Args:
            query: User query or task description

        Returns:
            List of matching skills, sorted by relevance
        """
        if not self._discovered:
            self.discover()

        # Normalize query
        query_lower = query.lower()
        query_words = set(re.findall(r"\w+", query_lower))

        matches: list[tuple[float, SkillDefinition]] = []

        for skill in self._skills.values():
            score = self._calculate_match_score(skill, query_lower, query_words)
            if score > 0:
                matches.append((score, skill))

        # Sort by score descending
        matches.sort(key=lambda x: -x[0])

        return [skill for _, skill in matches]

    def _calculate_match_score(
        self,
        skill: SkillDefinition,
        query_lower: str,
        query_words: set[str],
    ) -> float:
        """Calculate match score between skill and query.

        Args:
            skill: Skill to score
            query_lower: Lowercase query string
            query_words: Set of query words

        Returns:
            Match score (0.0 = no match)
        """
        score = 0.0

        # Check skill name
        skill_name_lower = skill.name.lower().replace("-", " ")
        if skill_name_lower in query_lower:
            score += 3.0

        # Check description words
        desc_words = set(re.findall(r"\w+", skill.description.lower()))
        common_words = query_words & desc_words
        score += len(common_words) * 0.5

        # Exact phrase match in description
        if query_lower in skill.description.lower():
            score += 2.0

        return score

    def load_full(self, name: str) -> SkillDefinition | None:
        """Load full skill content.

        Same as get() for now, but can be extended for
        lazy loading of large skills.

        Args:
            name: Skill name

        Returns:
            Full skill definition or None
        """
        return self.get(name)

    def get_resource(self, skill_name: str, resource_name: str) -> str | None:
        """Load a skill's resource file.

        Args:
            skill_name: Name of the skill
            resource_name: Name of the resource file

        Returns:
            Resource content or None if not found
        """
        skill = self.get(skill_name)
        if skill is None or skill.resources_path is None:
            return None

        resource_path = skill.resources_path / resource_name
        if not resource_path.exists():
            return None

        return resource_path.read_text()

    def register(self, skill: SkillDefinition) -> None:
        """Register a skill definition programmatically.

        Args:
            skill: Skill to register
        """
        self._skills[skill.name] = skill

    def clear(self) -> None:
        """Clear all skills (for testing)."""
        self._skills.clear()
        self._discovered = False


# Global skill loader singleton
_global_skill_loader: SkillLoader | None = None


def get_skill_loader() -> SkillLoader:
    """Get the global skill loader singleton.

    Returns:
        The global SkillLoader instance
    """
    global _global_skill_loader
    if _global_skill_loader is None:
        _global_skill_loader = SkillLoader()
    return _global_skill_loader


def reset_skill_loader() -> None:
    """Reset the global skill loader (for testing)."""
    global _global_skill_loader
    _global_skill_loader = None
