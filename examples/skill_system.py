#!/usr/bin/env python3
"""
Skill System Example (v0.2.0)

Demonstrates domain knowledge with semantic matching (SKILL.md format).
"""

import asyncio
import tempfile
from pathlib import Path

from ccflow import SkillDefinition, get_skill_loader, reset_skill_loader


async def basic_skills():
    """Demonstrate basic skill operations."""
    print("=== Basic Skills ===")

    reset_skill_loader()
    loader = get_skill_loader()

    # Create a temporary skill for demo
    with tempfile.TemporaryDirectory() as tmpdir:
        skill_dir = Path(tmpdir) / "security-audit"
        skill_dir.mkdir()

        # Create SKILL.md
        skill_content = """---
name: security-audit
description: Security vulnerability analysis and OWASP compliance checking
keywords:
  - security
  - vulnerability
  - OWASP
  - authentication
  - authorization
---

# Security Audit Skill

When analyzing code for security issues, follow this protocol:

## 1. Injection Vulnerabilities
- Check for SQL injection
- Check for command injection
- Check for XSS vulnerabilities

## 2. Authentication
- Verify password hashing
- Check session management
- Audit token handling

## 3. Authorization
- Review access controls
- Check for privilege escalation
- Audit role-based access
"""
        (skill_dir / "SKILL.md").write_text(skill_content)

        # Register skill manually (normally auto-discovered)
        skill = SkillDefinition.from_skill_md(skill_dir / "SKILL.md")
        loader._skills[skill.name] = skill

        # List skills
        print("Registered skills:")
        for name in loader.list():
            s = loader.get(name)
            if s:
                print(f"  - {name}: {s.description[:50]}...")

        # Semantic matching
        print("\nMatching 'check for vulnerabilities':")
        matches = loader.match("check for vulnerabilities")
        for s in matches:
            print(f"  Matched: {s.name} (score based on keyword overlap)")

        # Load full content
        print("\nFull skill content preview:")
        full_skill = loader.load_full("security-audit")
        print(f"  Instructions ({len(full_skill.instructions)} chars):")
        print(f"  {full_skill.instructions[:200]}...")


async def skill_matching():
    """Demonstrate semantic skill matching."""
    print("\n=== Skill Matching ===")

    reset_skill_loader()
    loader = get_skill_loader()

    # Create multiple demo skills
    with tempfile.TemporaryDirectory() as tmpdir:
        # Security skill
        security_dir = Path(tmpdir) / "security"
        security_dir.mkdir()
        (security_dir / "SKILL.md").write_text("""---
name: security
description: Security analysis and vulnerability detection
keywords:
  - security
  - vulnerability
  - audit
---
# Security Skill
Security analysis instructions...
""")

        # Performance skill
        perf_dir = Path(tmpdir) / "performance"
        perf_dir.mkdir()
        (perf_dir / "SKILL.md").write_text("""---
name: performance
description: Performance optimization and profiling
keywords:
  - performance
  - optimization
  - profiling
  - speed
---
# Performance Skill
Performance optimization instructions...
""")

        # Testing skill
        test_dir = Path(tmpdir) / "testing"
        test_dir.mkdir()
        (test_dir / "SKILL.md").write_text("""---
name: testing
description: Test writing and coverage analysis
keywords:
  - testing
  - pytest
  - coverage
  - unit tests
---
# Testing Skill
Test writing instructions...
""")

        # Register skills
        for skill_dir in [security_dir, perf_dir, test_dir]:
            skill = SkillDefinition.from_skill_md(skill_dir / "SKILL.md")
            loader._skills[skill.name] = skill

        # Test various queries
        queries = [
            "find security vulnerabilities",
            "make the code faster",
            "write unit tests",
            "audit for problems",
        ]

        print("Query matching results:")
        for q in queries:
            matches = loader.match(q)
            match_names = [m.name for m in matches[:2]]
            print(f"  '{q}' -> {match_names}")


async def skill_resources():
    """Demonstrate skill resource files."""
    print("\n=== Skill Resources ===")

    reset_skill_loader()
    loader = get_skill_loader()

    with tempfile.TemporaryDirectory() as tmpdir:
        skill_dir = Path(tmpdir) / "api-client"
        skill_dir.mkdir()
        resources_dir = skill_dir / "resources"
        resources_dir.mkdir()

        # Create SKILL.md
        (skill_dir / "SKILL.md").write_text("""---
name: api-client
description: API client implementation patterns
keywords:
  - api
  - http
  - client
---
# API Client Skill
Use the templates in resources/ for implementation.
""")

        # Create resource files
        (resources_dir / "template.py").write_text("""
# API Client Template
import httpx

class APIClient:
    def __init__(self, base_url: str):
        self.client = httpx.AsyncClient(base_url=base_url)

    async def get(self, path: str):
        return await self.client.get(path)
""")

        # Register skill
        skill = SkillDefinition.from_skill_md(skill_dir / "SKILL.md")
        skill.resources_path = resources_dir
        loader._skills[skill.name] = skill

        # Access resources
        print("Skill with resources:")
        print(f"  Name: {skill.name}")
        print(f"  Resources path: {skill.resources_path}")
        print(f"  Available resources: {list(resources_dir.iterdir())}")


async def main():
    """Run all skill examples."""
    await basic_skills()
    await skill_matching()
    await skill_resources()
    print("\nAll skill examples complete!")


if __name__ == "__main__":
    asyncio.run(main())
