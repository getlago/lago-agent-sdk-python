.PHONY: sync test lint format check clean

sync:
	uv sync --all-extras

test:
	uv run pytest tests/unit -q

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests
	uv run mypy src

format:
	uv run ruff format src tests
	uv run ruff check --fix src tests

check: lint test
