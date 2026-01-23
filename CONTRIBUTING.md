# Contributing to FrexTech Simulation

Thank you for your interest in contributing to frex Simulations! This document provides guidelines and instructions for contributing.

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please be respectful and constructive in all interactions.

## How to Contribute

### Reporting Bugs

Before creating a bug report, please check the issue list to avoid duplicates.

When filing a bug report, include:
- **Title**: Clear, descriptive summary
- **Description**: What you expected vs. what happened
- **Steps to Reproduce**: Minimal code to reproduce
- **Environment**: Python version, OS, GPU (if relevant)
- **Logs**: Any error messages or tracebacks

### Suggesting Features

Submit feature requests as GitHub issues with:
- **Title**: Clear description of the feature
- **Motivation**: Why this feature would be useful
- **Proposed Solution**: How you envision it working
- **Alternatives**: Other approaches you've considered

### Pull Requests

1. **Fork** the repository and create a feature branch:
   ```bash
   git checkout -b feature/my-feature
   ```

2. **Commit** with clear messages:
   ```bash
   git commit -m "Add feature: brief description"
   ```

3. **Push** and open a pull request:
   ```bash
   git push origin feature/my-feature
   ```

4. **Ensure**:
   - Code follows PEP 8 style guide
   - Tests pass: `pytest tests/`
   - Documentation is updated
   - Commit messages are descriptive

## Development Setup

1. Clone and install in development mode:
   ```bash
   git clone https://github.com/nkurangafredrick146-code/frex-simulations.git
   cd frex-simulations
   python -m venv venv
   source venv/bin/activate
   pip install -e ".[dev]"
   ```

2. Run tests:
   ```bash
   pytest tests/ -v
   ```

3. Format code:
   ```bash
   black sim_env/ tests/
   flake8 sim_env/ tests/
   ```

## Documentation

- Update docs in the `docs/` folder for new features
- Use docstrings for all classes and functions:
   ```python
   """
   Brief description.
   
   Args:
       param1: Description
       param2: Description
   
   Returns:
       Description of return value
   """
   ```

## Testing

- Write tests for new functionality in `tests/`
- Aim for >80% code coverage
- Use `pytest` for test execution

## License

By contributing, you agree your code will be licensed under the MIT License.

## Questions?

Feel free to open an issue for questions or contact the maintainers.

Thank you for contributing! 🎉
