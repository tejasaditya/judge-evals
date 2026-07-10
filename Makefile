.PHONY: setup test lint format run benchmark demo

setup:
	uv sync --dev

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

run:
	uv run judge-evals --help

benchmark:
	uv run python scripts/benchmark.py

demo:
	uv run python scripts/gate_demo.py
