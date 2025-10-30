.PHONY: help install test lint format clean backtest docs

help:  ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

install:  ## Install dependencies
	pip install -r requirements.txt
	pip install -e .

install-dev:  ## Install development dependencies
	pip install -r requirements.txt
	pip install pytest pytest-cov pytest-mock pytest-xdist
	pip install black flake8 isort mypy
	pip install -e .

test:  ## Run tests
	pytest tests/ -v

test-coverage:  ## Run tests with coverage report
	pytest tests/ --cov=trading_strategy --cov=backtester --cov-report=html --cov-report=term

test-parallel:  ## Run tests in parallel
	pytest tests/ -n auto -v

format:  ## Format code with black and isort
	black trading_strategy/ backtester.py backtest.py tests/ scripts/ --line-length 100
	isort trading_strategy/ backtester.py backtest.py tests/ scripts/ --profile black --line-length 100

lint:  ## Lint code with flake8
	flake8 trading_strategy/ backtester.py backtest.py --max-line-length=100 --extend-ignore=E203,W503

type-check:  ## Run type checking with mypy
	mypy trading_strategy/ backtester.py backtest.py --ignore-missing-imports

check: format lint type-check  ## Run all code quality checks

clean:  ## Clean cache and temporary files
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/
	rm -rf .cache/

clean-data:  ## Clean temporary data files
	rm -rf data/temp/*
	find data/ -name "*.tmp" -delete

backtest-quick:  ## Run quick validation backtest
	python backtest.py quick --pair BTCUSDT --start 2023-01-01 --end 2023-03-31

backtest-single:  ## Run single period backtest (use ARGS for customization)
	python backtest.py single --pair BTCUSDT --start 2023-01-01 --end 2023-12-31

backtest-multi:  ## Run multi-period comprehensive backtest
	python backtest.py multi --pair BTCUSDT

backtest-walk-forward:  ## Run walk-forward analysis
	python backtest.py walk-forward --pair BTCUSDT --start 2021-01-01 --end 2024-10-18

backtest-optimize:  ## Run parameter optimization
	python backtest.py optimize --pair BTCUSDT --start 2023-01-01 --end 2023-12-31

analyze-signals:  ## Analyze signal generation bias
	python scripts/diagnostics/diagnose_signal_bias.py

analyze-strategy:  ## Run strategy diagnostics
	python scripts/diagnostics/strategy_diagnostic.py

analyze-risk:  ## Run risk analysis
	python scripts/analysis/risk_analysis.py

docs:  ## Generate documentation (requires sphinx)
	@echo "Documentation is in docs/ directory"
	@echo "View README.md for main documentation"

setup:  ## Initial project setup
	python -m venv venv
	@echo "Virtual environment created. Activate it with:"
	@echo "  source venv/bin/activate  (Linux/Mac)"
	@echo "  venv\\Scripts\\activate     (Windows)"
	@echo "Then run: make install-dev"

.DEFAULT_GOAL := help

