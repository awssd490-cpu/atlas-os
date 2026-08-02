# Contributing to Atlas

Thank you for your interest in contributing to Atlas!

## Development setup

```bash
# The repository is currently private; use the Tekvora-provided access
# when it becomes public, then:
git clone https://github.com/awssd490-cpu/atlas-os.git
cd atlas-os
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Code style

- All code must pass `pytest` before submission
- Type annotations required on all public functions and classes
- Follow the existing naming conventions (see `docs/developer/style_guide.md`)
- Use `from __future__ import annotations` in every Python file
- All dataclasses must be `frozen=True`
- Use `time.perf_counter()` for measurements, never `time.time()`

## Testing

```bash
# Run all tests
pytest

# Run a specific package
pytest tests/unit/rag/pipeline/

# Run with coverage
coverage run -m pytest tests/ && coverage report
```

See `docs/developer/testing.md` for detailed testing guidelines.

## Commit messages

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): description

Body (72 char wrap).
```

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `perf`

Examples:
- `feat(rag): implement DefaultReranker`
- `fix(pipeline): skip duplicates during ingest`
- `docs(api): add persistence reference`

## Pull requests

1. Create a feature branch from `main`
2. Make your changes with tests
3. Run `pytest tests/` and ensure all pass
4. Submit a PR with a clear description

See `docs/developer/contribute.md` for the full PR checklist.

## Issue reporting

- **Bug reports**: Include Python version, OS, and a minimal reproduction
- **Feature requests**: Describe the use case and proposed API
- **Questions**: Use the Discussions tab

## Security

Report security vulnerabilities to the maintainers directly. See `SECURITY.md` for details.
