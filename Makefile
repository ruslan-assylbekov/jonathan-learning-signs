.PHONY: dev test lint fmt migrate revision extract train eval templates loadtest

dev:
	docker compose up -d postgres redis
	cd backend && uv run uvicorn app.main:app --reload

test:
	cd backend && uv run pytest

lint:
	cd backend && uv run ruff check . && uv run ruff format --check . && uv run mypy app tests

fmt:
	cd backend && uv run ruff format .

migrate:
	cd backend && uv run alembic upgrade head

revision:
	cd backend && uv run alembic revision --autogenerate -m "$(m)"

extract:
	uv run --project backend python ml/bukva/extract_landmarks.py

train:
	uv run --project backend python ml/train.py

eval:
	uv run --project backend python ml/evaluate.py

templates:
	uv run --project backend python ml/build_templates.py

loadtest:
	cd experiments/loadtest && uv run locust -H $(LOADTEST_HOST)
