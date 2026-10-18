.PHONY: setup dev web test lint fmt migrate revision load-templates \
        inspect extract train eval templates loadtest

# Hint templates shipped with the repository; rebuilt by `make templates`.
TEMPLATES ?= ../ml/artifacts/templates-v2.json

# --- running the app ---------------------------------------------------------

setup:            ## database up, schema migrated, shipped hint templates loaded
	docker compose up -d postgres redis
	cd backend && uv run alembic upgrade head
	cd backend && uv run python -m app.infrastructure.db.load_templates $(TEMPLATES)

dev: setup        ## the API on :8000, reloading on change
	cd backend && uv run uvicorn app.main:app --reload --port 8000

web:              ## the frontend on :5173 (run alongside `make dev`)
	cd frontend && npm install && npm run vendor && npm run dev

# --- quality -----------------------------------------------------------------

test:
	cd backend && uv run pytest

lint:
	cd backend && uv run ruff check . ../ml ../experiments \
		&& uv run ruff format --check . ../ml ../experiments \
		&& uv run mypy app tests ../ml ../experiments
	cd frontend && npx tsc --noEmit

fmt:
	cd backend && uv run ruff format . ../ml ../experiments

# --- database ----------------------------------------------------------------

migrate:
	cd backend && uv run alembic upgrade head

revision:
	cd backend && uv run alembic revision --autogenerate -m "$(m)"

load-templates:   ## apply $(TEMPLATES) to the database; no dataset needed
	cd backend && uv run python -m app.infrastructure.db.load_templates $(TEMPLATES)

# --- ML (needs the Bukva dataset in BUKVA_DIR) ---------------------------------

inspect:
	uv run --project backend --group ml python ml/bukva/inspect_dataset.py

extract:
	uv run --project backend --group ml python ml/bukva/extract_landmarks.py

train:            ## writes the next unused model-vN; never overwrites one
	uv run --project backend --group ml python ml/train.py

eval:             ## evaluates the newest model, or v=N
	uv run --project backend --group ml python ml/evaluate.py $(if $(v),--version $(v))

templates:        ## rebuild hint templates from the dataset and apply them
	uv run --project backend --group ml python ml/build_templates.py --apply

# --- experiments (CLAUDE.md 16) ------------------------------------------------

# The deployment under test, and names for the results files.
LOADTEST_HOST ?= https://bukva-learning.duckdns.org
LOADTEST_LABEL ?= azure
LOADTEST_NETWORK ?= unnamed-network

loadtest:         ## ramp simulated users against $(LOADTEST_HOST), then summarize
	uv run --project backend --group ml python experiments/loadtest/run.py 		--target $(LOADTEST_HOST) --label $(LOADTEST_LABEL) --network $(LOADTEST_NETWORK) 		$(if $(LOADTEST_STATS),--stats-via "$(LOADTEST_STATS)")
	uv run --project backend --group ml python experiments/loadtest/analyze.py
