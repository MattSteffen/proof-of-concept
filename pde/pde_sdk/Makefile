# Python Development Makefile for PDE SDK
# Focused on Python-specific development tasks

.PHONY: help setup install lint format check test test-verbose
.PHONY: type-check docs clean build dist dev

# Default target
help: ## Show this help message
	@echo "Python Development Commands for PDE SDK"
	@echo "======================================"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Setup and Installation
setup: ## Install Python development dependencies
	@echo "Setting up Python development environment..."
	@if command -v uv >/dev/null 2>&1; then \
		uv sync --dev; \
		echo "✓ Python development tools installed with uv"; \
	else \
		echo "uv not found, installing with pip..."; \
		python -m pip install --upgrade pip; \
		python -m pip install ruff black pytest mypy maturin build twine; \
		echo "✓ Python development tools installed with pip"; \
	fi

install: ## Install package in development mode
	@echo "Installing PDE SDK in development mode..."
	@if command -v uv >/dev/null 2>&1; then \
		uv pip install -e ..; \
	else \
		pip install -e ..; \
	fi
	@echo "✓ PDE SDK installed in development mode"

# Future: Rust integration commands will be added here when ready
# install-rust: ## Build and install Rust extension (Phase 2)
# 	@echo "Building Rust extension..."
# 	cd ../rust/pde_sdk_rs && maturin develop
# 	@echo "✓ Rust extension built and installed"

# Code Quality
lint: ## Run ruff linter
	@echo "Linting Python code..."
	@if command -v uv >/dev/null 2>&1; then \
		uv run ruff check . && uv run ruff check ../tests/; \
	else \
		ruff check . && ruff check ../tests/; \
	fi

format: ## Format code with black and fix with ruff
	@echo "Formatting Python code..."
	@if command -v uv >/dev/null 2>&1; then \
		uv run black . && uv run ruff check --fix .; \
	else \
		black . && ruff check --fix .; \
	fi

check: format lint ## Run formatting and linting

# Testing
test: ## Run tests with pytest
	@echo "Running Python tests..."
	@if command -v uv >/dev/null 2>&1; then \
		uv run pytest ../tests/; \
	else \
		pytest ../tests/; \
	fi

test-verbose: ## Run tests with verbose output
	@echo "Running Python tests (verbose)..."
	@if command -v uv >/dev/null 2>&1; then \
		uv run pytest -v ../tests/; \
	else \
		pytest -v ../tests/; \
	fi

test-cov: ## Run tests with coverage
	@echo "Running Python tests with coverage..."
	@if command -v uv >/dev/null 2>&1; then \
		uv run pytest --cov=. --cov-report=html ../tests/; \
	else \
		pytest --cov=. --cov-report=html ../tests/; \
	fi

# Type Checking
type-check: ## Run mypy type checker
	@echo "Running type checking..."
	@if command -v uv >/dev/null 2>&1; then \
		uv run mypy .; \
	else \
		mypy .; \
	fi

# Building and Distribution
build: ## Build Python package
	@echo "Building Python package..."
	@if command -v uv >/dev/null 2>&1; then \
		uv run python -m build; \
	else \
		python -m build; \
	fi

dist: build ## Build distribution packages
	@echo "✓ Distribution packages built in dist/"

# Development Workflow
dev: install ## Setup development environment
	@echo "✓ Python development environment ready"

dev-full: install ## Setup full development environment (Python only)
	@echo "✓ Full development environment ready (Python)"

# Future: Add Rust integration to dev-full when ready
# dev-full: install install-rust ## Setup full development environment with Rust

# Examples and Demos
run-examples: ## Run example scripts
	@echo "Running Python examples..."
	find ../examples/python -name "*.py" -exec echo "Running {}" \; -exec python {} \;

# Cleaning
clean: ## Clean Python build artifacts
	@echo "Cleaning Python artifacts..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name "*.pyo" -delete 2>/dev/null || true
	rm -rf ../build/ ../dist/ ../*.egg-info/ .pytest_cache/ .coverage htmlcov/

# Utility Commands
check-all: check type-check test ## Run all code quality checks
ci: check-all build ## CI pipeline simulation

# Quick Development Cycle
cycle: format lint test ## Quick development cycle
