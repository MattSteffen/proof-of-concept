# Rust Development Makefile for PDE SDK
# Focused on Rust-specific development tasks

.PHONY: help setup install check lint format test test-verbose bench
.PHONY: build build-release build-python doc doc-open clean clean-all
.PHONY: dev update audit

# Default target
help: ## Show this help message
	@echo "Rust Development Commands for PDE SDK"
	@echo "====================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Setup
setup: ## Setup Rust development environment
	@echo "Setting up Rust development environment..."
	rustup component add rustfmt clippy
	cargo install cargo-nextest --locked 2>/dev/null || echo "cargo-nextest not available, using cargo test"
	cargo install cargo-audit --locked 2>/dev/null || echo "cargo-audit not available"
	@echo "✓ Rust development tools installed"

# Code Quality
check: ## Run cargo check
	@echo "Running cargo check..."
	cargo check

lint: ## Run clippy linter
	@echo "Running clippy..."
	cargo clippy --all-targets -- -D warnings

format: ## Format code with rustfmt
	@echo "Formatting Rust code..."
	cargo fmt

format-check: ## Check if code is formatted correctly
	@echo "Checking code formatting..."
	cargo fmt --check

# Testing
test: ## Run tests
	@echo "Running tests..."
	cargo test

test-verbose: ## Run tests with verbose output
	@echo "Running tests (verbose)..."
	cargo test -- --nocapture

test-nextest: ## Run tests with nextest (if available)
	@echo "Running tests with nextest..."
	cargo nextest run 2>/dev/null || cargo test

# Benchmarking
bench: ## Run benchmarks
	@echo "Running benchmarks..."
	cargo bench

# Building
build: ## Build in debug mode
	@echo "Building in debug mode..."
	cargo build

build-release: ## Build in release mode
	@echo "Building in release mode..."
	cargo build --release

build-python: ## Build Python extension with maturin
	@echo "Building Python extension..."
	maturin develop

# Documentation
doc: ## Generate documentation
	@echo "Generating documentation..."
	cargo doc

doc-open: ## Generate and open documentation
	@echo "Generating and opening documentation..."
	cargo doc --open

# Development Workflow
dev: build ## Setup development environment
	@echo "✓ Rust development environment ready"

# Dependency Management
update: ## Update dependencies
	@echo "Updating dependencies..."
	cargo update

audit: ## Audit dependencies for security issues
	@echo "Auditing dependencies..."
	cargo audit 2>/dev/null || echo "cargo audit not installed. Run 'make setup' to install it."

# Cleaning
clean: ## Clean build artifacts
	@echo "Cleaning build artifacts..."
	cargo clean

clean-all: clean ## Deep clean including target directory
	@echo "Deep cleaning..."
	rm -rf target/

# Utility Commands
check-all: format-check lint test ## Run all code quality checks
ci: check-all build-release ## CI pipeline simulation

# Quick Development Cycle
cycle: format lint test ## Quick development cycle

# Installation
install: build-python ## Build and install Python extension
	@echo "✓ Python extension installed"

# Examples
run-examples: ## Run Rust examples (if any)
	@echo "Running Rust examples..."
	find . -name "*.rs" -path "./examples/*" -exec echo "Running {}" \; -exec rustc {} -o /tmp/example && /tmp/example \; 2>/dev/null || echo "No examples found"
