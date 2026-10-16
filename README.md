# Sign Language Learning Assistant

Real-time feedback for learners of the Russian Sign Language (RSL) fingerspelling
alphabet. The browser runs MediaPipe Hand Landmarker and sends only 21 hand
landmarks per frame; video never leaves the machine. The backend normalizes the
landmarks, classifies the letter, and says which finger is wrong.

Course project for CSCI 436 Introduction to Cloud Computing. See
[CLAUDE.md](CLAUDE.md) for the full design and the working rules.

## Status

Working: the gesture domain (normalization, hold logic, per-finger feedback),
the `/ws/practice` protocol, `GET /letters`, and the browser client with camera,
skeleton overlay and live hints.

**No model is trained yet.** The backend therefore runs `StubClassifier`, which
labels a hand by which fingers are extended and names its classes `demo:…`. The
UI says so on screen. Nothing it reports is a result about sign language.

Next: extract landmarks from the Bukva dataset, train `model-v1`, then auth and
persistence (CLAUDE.md 19).

## Running it locally

Requires Python 3.12 (uv manages it) and Node 20+.

```bash
cp .env.example .env     # DEV_AUTH=true is already set for local use
docker compose up -d postgres redis

cd backend && uv sync --group ml
uv run alembic upgrade head
uv run uvicorn app.main:app --port 8000

cd ../frontend && npm install
npm run vendor           # fetch MediaPipe wasm + hand_landmarker.task (~26 MB)
npm run dev
```

Open <http://localhost:5173>, press **Start camera**, pick a target and hold the
gesture. Vite proxies `/letters` and `/ws` to the backend, so the page talks to
its own origin.

The camera needs a secure context, so this works on `localhost` and over HTTPS
and nowhere else. A deployed instance needs a certificate before it can be
demonstrated.

`npm run vendor` is required after a fresh clone: the MediaPipe runtime and
model are third-party binaries, self-hosted rather than loaded from a CDN, and
are not committed.

## Checks

```bash
docker compose up -d postgres redis   # integration tests need it
cd backend && uv run pytest           # 84 tests
uv run ruff check . ../ml && uv run mypy app tests ../ml
cd ../frontend && npm run build    # type-checks, then bundles
```

## Dataset

Bukva (Kvanchiani et al., 2024, [arXiv:2410.08675](https://arxiv.org/abs/2410.08675),
[github.com/ai-forever/bukva](https://github.com/ai-forever/bukva)), ~27 GB.
Unpack it outside the repository and set `BUKVA_DIR` in `.env`, then:

```bash
make inspect     # profile the annotations before processing anything
```

Licensing, including what may be redistributed, is summarized in
[ml/bukva/README.md](ml/bukva/README.md). Videos and frames are never committed.
