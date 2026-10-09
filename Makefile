.PHONY: setup test lint format clean

setup:
	uv sync

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .

clean:
	uv run python -c "import shutil; from pathlib import Path; [shutil.rmtree(Path(name), ignore_errors=True) for name in ['.pytest_cache', '.ruff_cache', 'build', 'dist']]"
