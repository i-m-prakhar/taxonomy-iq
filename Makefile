.PHONY: setup generate train serve test lint benchmark docker
setup:
	uv sync --extra dev
generate:
	uv run taxonomyiq generate-data
train:
	uv run taxonomyiq train-evaluate
serve:
	uv run uvicorn taxonomyiq.api:app --host 127.0.0.1 --port 8000
test:
	uv run --extra dev pytest -q
lint:
	uv run --extra dev ruff check src tests
benchmark:
	uv run python scripts/benchmark_api.py
docker:
	docker compose up --build
