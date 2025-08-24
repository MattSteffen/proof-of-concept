# PDE SDK Development Makefile
# Focus on Python development first, with Rust integration planned for later
# Follows the tooling specified in README.md and Roadmap.md

.PHONY: help setup install lint format test build clean clean-all
.PHONY: check dev py-check py-dev init docs docs-serve docs-deploy

# Default target
help: ## Show this help message
	@echo "PDE SDK Development Commands (Python-Focused)"
	@echo "=============================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Setup
setup: ## Setup Python development environment
	@echo "Setting up Python development environment..."
	cd pde_sdk && $(MAKE) setup

# Installation
install: ## Install Python package in development mode
	@echo "Installing Python package..."
	cd pde_sdk && $(MAKE) install

# Linting
lint: ## Run Python linting with ruff
	@echo "Linting Python code..."
	cd pde_sdk && $(MAKE) lint

# Formatting
format: ## Format Python code with black and ruff
	@echo "Formatting Python code..."
	cd pde_sdk && $(MAKE) format

# Testing
test: ## Run Python tests with pytest
	@echo "Running Python tests..."
	cd pde_sdk && $(MAKE) test

# Building
build: ## Build Python package
	@echo "Building Python package..."
	cd pde_sdk && $(MAKE) build

# Development workflow
dev: ## Start Python development mode
	@echo "Setting up Python development environment..."
	cd pde_sdk && $(MAKE) dev

# Code checking
check: ## Run Python linting and tests
	@echo "Running Python checks..."
	cd pde_sdk && $(MAKE) check-all

# Documentation
docs: ## Build MkDocs documentation
	@echo "Building MkDocs documentation..."
	uv run mkdocs build

docs-serve: ## Serve MkDocs documentation locally
	@echo "Serving MkDocs documentation at http://localhost:8000"
	uv run mkdocs serve

docs-deploy: ## Deploy MkDocs documentation to GitHub Pages
	@echo "Deploying MkDocs documentation to GitHub Pages..."
	uv run mkdocs gh-deploy --force

# Cleaning
clean: ## Clean Python build artifacts
	@echo "Cleaning Python artifacts..."
	cd pde_sdk && $(MAKE) clean

clean-all: ## Deep clean including target directories
	@echo "Deep cleaning..."
	cd pde_sdk && $(MAKE) clean-all

# Quick commands for common workflows
py-check: ## Full Python workflow (format, lint, test)
	@echo "Running Python workflow..."
	cd pde_sdk && $(MAKE) cycle

# Convenience aliases
py-dev: dev ## Alias for Python development setup

# Setup and development
init: setup install ## Initialize Python development environment
	@echo "✓ Python development environment initialized"

# Rust-related commands (for future integration)
rust-setup: ## Setup Rust development environment (when ready)
	@echo "Setting up Rust environment..."
	cd rust/pde_sdk_rs && $(MAKE) setup

rust-dev: ## Start Rust development mode (when ready)
	@echo "Setting up Rust development environment..."
	cd rust/pde_sdk_rs && $(MAKE) dev

# Future: Full project workflow (Python + Rust)
# full-check: py-check rs-check  # Will be enabled after Rust integration
# full-setup: setup rust-setup  # Will be enabled after Rust integration
