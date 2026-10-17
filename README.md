# Sign Language Learning Assistant

Real-time feedback for learners of the Russian Sign Language (RSL) fingerspelling
alphabet. The browser runs MediaPipe Hand Landmarker and sends only 21 hand
landmarks per frame; video never leaves the machine. The backend normalizes the
landmarks, classifies the letter, and says which finger is wrong — and whether
it should be straight, half bent or fully bent.

Course project for CSCI 436 Introduction to Cloud Computing. See
[CLAUDE.md](CLAUDE.md) for the full design and the working rules.

## Status

- **Model:** `ml/artifacts/model-v2` — logistic regression over 25 static RSL
  letters, trained on the Bukva dataset. On 20 signers it never saw: frame
  accuracy 0.702, clip accuracy (majority of frames) 0.818, 0.17 ms per frame.
  Full report in [experiments/models/eval-v2.md](experiments/models/eval-v2.md).
  Dynamic letters (Д Ё З Й Ц Щ Ъ Ь) are out of scope; see
  [docs/alphabet.md](docs/alphabet.md).
- **App:** sign-in, Learn mode with camera, skeleton overlay, hold meter and
  per-finger hints; attempts and progress are stored per account.
- **Deployment:** a Docker Compose stack for an Azure VM, tested end to end
  locally; see [deploy/README.md](deploy/README.md). Not yet running on Azure.
- **Not yet measured:** accuracy on webcam recordings (domain shift), and the
  bandwidth, latency and load experiments (CLAUDE.md 16).

## Running it locally

Requires Docker, [uv](https://docs.astral.sh/uv/) (it installs Python 3.12)
and Node 20+.

```bash
cp .env.example .env     # local defaults; DEV_AUTH=true is for local use only
make dev                 # Postgres + Redis, migrations, hint templates, API on :8000
make web                 # in a second terminal: frontend on :5173
```

Without `make`, the same steps:

```bash
docker compose up -d postgres redis
cd backend && uv sync
uv run alembic upgrade head
uv run python -m app.infrastructure.db.load_templates ../ml/artifacts/templates-v2.json
uv run uvicorn app.main:app --port 8000

cd frontend && npm install
npm run vendor           # MediaPipe wasm + hand_landmarker.task (~26 MB)
npm run dev
```

Open <http://localhost:5173>, create an account, press **Start camera**, pick a
letter and hold it. The camera needs a secure context, so it works on
`localhost` and over HTTPS only.

`npm run vendor` is required after a fresh clone: the MediaPipe runtime and
model are third-party binaries, self-hosted rather than loaded from a CDN, and
are not committed. Without the hint templates loaded, letters can still be
practised but no per-finger hints appear.

## Checks

```bash
docker compose up -d postgres redis   # the integration tests need it
make test                             # pytest: unit + integration
make lint                             # ruff, mypy (strict), tsc
cd frontend && npm run build          # type-checks, then bundles
```

## Dataset and model

Bukva (Kvanchiani et al., 2024, [arXiv:2410.08675](https://arxiv.org/abs/2410.08675),
[github.com/ai-forever/bukva](https://github.com/ai-forever/bukva)), ~27 GB.
Unpack it outside the repository and set `BUKVA_DIR` in `.env`. Then, with the
ml dependencies (`cd backend && uv sync --group ml`):

```bash
make inspect     # profile the annotations before processing anything
make extract     # videos -> ml/data/bukva_landmarks.parquet (about an hour)
make train       # writes the next unused ml/artifacts/model-vN
make eval        # held-out report for the newest model
make templates   # rebuild hint templates and apply them
```

Licensing, what was measured about the data, and how much memory extraction
needs are in [ml/bukva/README.md](ml/bukva/README.md). Videos and frames are
never committed.
