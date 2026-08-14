.PHONY: help install install-dev test test-verbose lint typecheck build clean publish

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------- setup

install: ## Install the package
	pip install -e .

install-dev: ## Install with test dependencies
	pip install -e ".[test]"

# ---------------------------------------------------------------- quality

test: ## Run all tests
	python -m pytest tests/ --tb=short -q

test-verbose: ## Run all tests with verbose output
	python -m pytest tests/ -v --tb=short

test-unit: ## Run only pure unit tests (no DB required)
	python -m pytest tests/ -v --tb=short -k "not clean_realm"

lint: ## Run ruff linter
	ruff check post_graph/ tests/

lint-fix: ## Run ruff linter with auto-fix
	ruff check --fix post_graph/ tests/

format: ## Format code with ruff
	ruff format post_graph/ tests/

format-check: ## Check formatting without changing files
	ruff format --check post_graph/ tests/

typecheck: ## Run pyright type checker
	pyright post_graph/

# ---------------------------------------------------------------- build

build: clean ## Build distribution packages
	python -m build

clean: ## Remove build artifacts
	rm -rf dist/ build/ *.egg-info post_graph/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache

publish: build ## Publish to PyPI
	twine upload dist/*

publish-test: build ## Publish to Test PyPI
	twine upload --repository testpypi dist/*

# ---------------------------------------------------------------- dev

check: lint format-check typecheck test ## Run all checks (lint + format + typecheck + test)
