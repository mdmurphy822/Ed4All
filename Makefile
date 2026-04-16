.PHONY: install test test-trainforge test-mcp lint coverage clean mcp help

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install all dependencies
	pip install -e ".[full]"

test:  ## Run all tests
	pytest

test-trainforge:  ## Run Trainforge tests
	pytest Trainforge/tests/ -v

test-mcp:  ## Run MCP tests
	pytest MCP/tests/ -v

lint:  ## Run linter
	ruff check lib/ MCP/ cli/ Trainforge/ LibV2/tools/ Courseforge/scripts/

coverage:  ## Run tests with coverage report
	pytest --cov --cov-report=html

clean:  ## Remove build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf htmlcov/ .coverage coverage.xml

mcp:  ## Start MCP server
	cd MCP && python server.py
