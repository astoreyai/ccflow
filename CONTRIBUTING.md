# Contributing to ccflow

Thank you for your interest in contributing to ccflow! This document provides guidelines and instructions for contributing.

## Development Setup

### Prerequisites

- Python 3.11+
- Claude Code CLI installed and authenticated
- Git

### Local Development

```bash
# Clone the repository
git clone https://github.com/astoreyai/ccflow.git
cd ccflow

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode with all extras
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Code Quality

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=ccflow --cov-report=term-missing

# Run specific test file
pytest tests/test_api.py -v

# Run tests matching pattern
pytest -k "test_query" -v
```

### Type Checking

```bash
mypy src/ccflow/
```

### Linting

```bash
# Check for issues
ruff check src/ tests/

# Auto-fix issues
ruff check --fix src/ tests/

# Format code
ruff format src/ tests/
```

## Making Changes

### Branch Naming

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): description

feat(session): add automatic session persistence
fix(parser): handle malformed NDJSON gracefully
docs(readme): update installation instructions
test(api): add coverage for batch_query
```

### Pull Request Process

1. Fork the repository
2. Create a feature branch from `main`
3. Make your changes with tests
4. Ensure all tests pass and coverage is maintained
5. Update documentation if needed
6. Submit a pull request

### PR Requirements

- All tests must pass
- Coverage must not decrease
- Type checking must pass
- Linting must pass
- Documentation updated for new features

## Code Style

### Python

- Follow PEP 8 with 100 character line limit
- Use type hints for all public functions
- Write docstrings for all public classes and functions
- Prefer `async`/`await` for I/O operations

### Documentation

- Update README.md for user-facing changes
- Update LLMS.txt for API changes
- Add docstrings with examples for new functions

## Testing Guidelines

### Test Structure

```python
class TestFeatureName:
    """Tests for feature description."""

    @pytest.mark.asyncio
    async def test_specific_behavior(self):
        """Test that specific behavior occurs."""
        # Arrange
        ...
        # Act
        ...
        # Assert
        ...
```

### Mocking

- Mock CLI execution in unit tests
- Use fixtures from `conftest.py`
- Test both success and error cases

## Architecture

### Key Modules

| Module | Purpose |
|--------|---------|
| `api.py` | High-level query functions |
| `session.py` | Multi-turn session management |
| `executor.py` | CLI subprocess handling |
| `parser.py` | NDJSON stream parsing |
| `types.py` | Type definitions |

### Adding New Features

1. Add types to `types.py`
2. Implement in appropriate module
3. Export from `__init__.py`
4. Add tests in `tests/`
5. Update documentation

## Release Process

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create release PR
4. After merge, tag release
5. GitHub Actions publishes to PyPI

## Getting Help

- Open an issue for bugs or feature requests
- Start a discussion for questions
- Check existing issues before creating new ones

## Code of Conduct

Be respectful and constructive. We welcome contributors of all backgrounds and experience levels.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
