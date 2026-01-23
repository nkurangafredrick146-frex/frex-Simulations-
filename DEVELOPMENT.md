# Development Guide

This guide provides information for developers contributing to the frex Simulations project.

## Table of Contents
- [Setup](#setup)
- [Development Workflow](#development-workflow)
- [Testing](#testing)
- [Code Quality](#code-quality)
- [Documentation](#documentation)
- [Git Workflow](#git-workflow)

## Setup

### Prerequisites
- Python 3.10 or higher
- Git
- Virtual environment tool (venv or conda)

### Initial Setup

1. Clone the repository:
```bash
git clone https://github.com/nkurangafredrick146-code/frex-simulations.git
cd frex-simulations
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install development dependencies:
```bash
make install-dev
```

4. Install pre-commit hooks:
```bash
pip install pre-commit
pre-commit install
```

## Development Workflow

### Creating a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### Running the Application

```bash
make run
```

### Building Locally

```bash
make build
```

## Testing

### Running All Tests

```bash
make test
```

### Running Specific Tests

```bash
pytest tests/test_specific.py
```

### Running Tests with Coverage

```bash
make test-cov
```

### Running Fast Tests Only

```bash
make test-fast
```

### Running Tests by Marker

```bash
pytest -m gpu          # Run GPU tests
pytest -m integration  # Run integration tests
pytest -m "not slow"   # Run everything except slow tests
```

## Code Quality

### Code Formatting

Format code with Black and isort:
```bash
make format
```

Check formatting without making changes:
```bash
make format-check
```

### Linting

Run all linting checks:
```bash
make lint
```

Individual linting tools:
```bash
flake8 sim_env tests
pylint sim_env
```

### Type Checking

```bash
make type-check
```

### Full Quality Check

Run all checks (format, lint, type):
```bash
make check
```

## Documentation

### Building Documentation

```bash
make docs
```

### Serving Documentation Locally

```bash
make serve-docs
```

Then visit `http://localhost:8000` in your browser.

### Writing Documentation

- Documentation files are in the `docs/` directory
- Use Markdown format
- Follow the existing documentation style
- Include docstrings in all modules, classes, and functions

## Git Workflow

### Before Committing

1. Run all checks:
```bash
make check
```

2. Run tests:
```bash
make test-cov
```

3. Build documentation:
```bash
make docs
```

### Commit Message Guidelines

Follow these conventions for commit messages:

- Use present tense: "Add feature" not "Added feature"
- Use imperative mood: "Move cursor to..." not "Moves cursor to..."
- Limit the first line to 72 characters
- Reference issues and pull requests liberally after the first line

Example:
```
Add support for quantum simulation optimization

- Implement quantum circuit optimization algorithm
- Add benchmark tests for performance verification
- Update documentation with new feature usage

Fixes #123
```

### Creating a Pull Request

1. Push your feature branch:
```bash
git push origin feature/your-feature-name
```

2. Create a pull request on GitHub
3. Ensure all CI checks pass
4. Request review from team members
5. Address review feedback

## Code Style Guide

### Python Style

- Follow PEP 8
- Line length: 100 characters
- Use type hints where possible
- Document all public APIs

### Imports

- Use absolute imports
- Group imports: standard library, third-party, local
- Keep imports sorted with isort

### Naming Conventions

- Classes: `PascalCase`
- Functions/methods: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private members: `_leading_underscore`

## Performance Considerations

- Profile code before optimizing
- Use vectorized operations with NumPy
- Consider GPU acceleration for heavy computations
- Monitor memory usage for large simulations

## Troubleshooting

### Common Issues

**Issue: Import errors when running tests**
- Solution: Ensure your virtual environment is activated and dependencies are installed

**Issue: Type checking failures**
- Solution: Add type hints or update stubs for third-party libraries

**Issue: Documentation won't build**
- Solution: Install documentation dependencies with `make install-docs`

## Resources

- [PEP 8 Style Guide](https://www.python.org/dev/peps/pep-0008/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Type Hints Guide](https://docs.python.org/3/library/typing.html)
- [MkDocs Documentation](https://www.mkdocs.org/)

## Questions?

If you have questions, please:
1. Check the documentation in `docs/`
2. Review existing issues and discussions
3. Open a new issue with a clear description
