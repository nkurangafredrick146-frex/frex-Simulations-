.PHONY: help install install-dev install-docs test test-cov lint format type-check clean build docs serve-docs

help:
	@echo "Available commands:"
	@echo "  make install          - Install the package"
	@echo "  make install-dev      - Install with development dependencies"
	@echo "  make install-docs     - Install with documentation dependencies"
	@echo "  make test             - Run tests"
	@echo "  make test-cov         - Run tests with coverage report"
	@echo "  make lint             - Run linting checks (flake8, pylint)"
	@echo "  make format           - Format code with black and isort"
	@echo "  make type-check       - Run type checking with mypy"
	@echo "  make check            - Run all checks (lint, format, type-check)"
	@echo "  make clean            - Clean up temporary files and caches"
	@echo "  make build            - Build distribution packages"
	@echo "  make docs             - Build documentation"
	@echo "  make serve-docs       - Serve documentation locally"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-docs:
	pip install -e ".[docs]"

install-all:
	pip install -e ".[dev,docs]"

test:
	pytest

test-cov:
	pytest --cov=sim_env --cov-report=html --cov-report=term-missing

test-fast:
	pytest -m "not slow"

test-integration:
	pytest -m integration

lint:
	flake8 sim_env tests
	pylint sim_env --disable=all --enable=E,F

format:
	black sim_env tests
	isort sim_env tests

format-check:
	black --check sim_env tests
	isort --check-only sim_env tests

type-check:
	mypy sim_env

check: format-check lint type-check

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .tox -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name .coverage -delete
	find . -type d -name htmlcov -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name dist -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name build -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

build: clean
	python -m build

docs:
	cd docs && mkdocs build

serve-docs:
	cd docs && mkdocs serve

run:
	python -m sim_env.main
