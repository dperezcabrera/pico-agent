# Contributing to Pico-Agent

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
pip install ruff
```

## Running Tests

```bash
pytest tests/ -v                # Run tests
tox                             # Full matrix (3.11-3.14)
```

## Code Style

- Python 3.11+
- Format with `ruff format src/ tests/`
- Lint with `ruff check src/ tests/`

## Pull Requests

1. Fork the repository
2. Create a feature branch
3. Ensure tests pass and code is formatted
4. Submit a PR with a clear description

## Reporting Issues

Use [GitHub Issues](https://github.com/dperezcabrera/pico-agent/issues) for bugs and feature requests.
