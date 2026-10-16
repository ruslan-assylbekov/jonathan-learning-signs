# CLAUDE.md — Sign Language Learning Assistant

This file gives Claude Code the context needed to work on this repository.
Read it fully at the start of every session. Keep it up to date: when a
decision changes, update the relevant section in the same commit.

---

## 1. Project context

- **What:** a web application that helps users learn the Russian Sign
  Language (RSL) fingerspelling (dactyl) alphabet. The user shows a letter to the webcam and gets
  real-time feedback: whether the sign is correct and *which finger* is wrong.
- **Why:** course project for **CSCI 436 Introduction to Cloud Computing**
  (Nazarbayev University, Fall 2026). The project is worth 30% of the grade.
  It must demonstrate cloud concepts, not only ML.
- **Who:** solo project by Ruslan (backend-focused CS student). The repo is
  also a portfolio piece, so code quality, tests, README and CI matter.
- **Core idea:** video never leaves the browser. The browser runs MediaPipe
  Hand Landmarker and sends only 21 hand landmarks per frame to the backend.
  The backend normalizes them, classifies the letter, generates per-finger
  feedback, and stores progress. This keeps the payload tiny, protects
  privacy, and makes the cloud part meaningful.

### Communication with Ruslan
- Explain things to Ruslan in **Russian**, keeping English technical terms
  as they are (e.g. "feature extraction", "WebSocket", "load testing").
- Everything inside the repo — code, comments, commit messages, docs,
  README, report drafts — is in **English**.
- Ask before: adding a new dependency, changing the DB schema, changing the
  WebSocket protocol, changing `FEATURE_VERSION`, or anything that deletes data.

---

## 2. Scope and priorities

Solo project → scope is cut aggressively. Work strictly in this order.
Do not start a lower tier until the tier above works end-to-end.

**Tier 1 — MVP (must have)**
1. Static RSL letters only (motion-based letters excluded, see §8.1)
   plus a `no_sign` class.
2. Landmark dataset built from Bukva + trained classifier with honest,
   signer-independent evaluation.
3. Backend: auth, letters, WebSocket evaluation, attempts, progress.
4. Frontend: camera, skeleton overlay, Learn mode with per-finger hints.
5. Deployed on OCI with HTTPS, via Docker Compose.
6. Experiments: bandwidth, latency, load test.

**Tier 2 — should have**
7. Practice mode (random letters, weak letters appear more often).
8. Progress page with per-letter accuracy.
9. CI/CD with GitHub Actions (tests → build → deploy).
10. Prometheus + Grafana monitoring.
11. Horizontal scaling experiment (1 → 2 → 4 API replicas behind Nginx).

**Tier 3 — nice to have / future work**
12. Spell-a-word mode.
13. Motion-based letters.
14. Client-side inference comparison (ONNX / TF.js).
15. Run the Bukva pretrained video model on our static subset as a baseline.

**Alphabet:** Russian Sign Language (RSL) dactyl, 33 letters.
Reference and primary data source: the **Bukva** dataset
(Kvanchiani et al., 2024, arXiv:2410.08675, github.com/ai-forever/bukva).
Record the final list of static letters used in the MVP in
`docs/alphabet.md`. Do not invent handshape descriptions; use only the
reference.

---

## 3. Tech stack

| Layer | Choice |
|---|---|
| Backend | Python 3.12, FastAPI, Pydantic v2, SQLAlchemy 2.x (async, asyncpg), Alembic |
| DB / cache | PostgreSQL 16, Redis 7 |
| ML | MediaPipe Tasks (Python, for dataset extraction/collection), OpenCV, NumPy, pandas, pyarrow (Parquet), scikit-learn, joblib |
| Frontend | Vite + TypeScript (vanilla, no framework), `@mediapipe/tasks-vision`, Canvas API |
| Infra | Docker, Docker Compose, Nginx, Let's Encrypt (certbot), DuckDNS |
| Cloud | Oracle Cloud Infrastructure (OCI) Ubuntu VM (existing account) |
| CI/CD | GitHub Actions |
| Monitoring | Prometheus, Grafana |
| Load testing | Locust (WebSocket users) |
| Quality | ruff (lint + format), mypy, pytest, pytest-asyncio, httpx |

Package management: `uv`. Do not add libraries outside this table without
asking. Prefer the standard library and existing dependencies.

---

## 4. Repository structure

```
sign-assistant/
├── CLAUDE.md
├── README.md
├── Makefile
├── docker-compose.yml            # local dev
├── .env.example                  # never commit .env
├── backend/
│   ├── pyproject.toml
│   ├── alembic/
│   ├── app/
│   │   ├── domain/               # pure Python, no framework imports
│   │   │   ├── gesture/
│   │   │   │   ├── features.py   # SINGLE SOURCE OF TRUTH for feature extraction
│   │   │   │   ├── feedback.py   # per-finger hint generation
│   │   │   │   └── stabilizer.py # N-consecutive-frames logic
│   │   │   ├── entities.py       # Letter, Attempt, Progress
│   │   │   └── ports.py          # repository / classifier interfaces (Protocols)
│   │   ├── application/          # use cases
│   │   │   ├── evaluate_gesture.py
│   │   │   ├── record_attempt.py
│   │   │   ├── next_letter.py
│   │   │   └── user_progress.py
│   │   ├── infrastructure/
│   │   │   ├── db/               # SQLAlchemy models + repositories
│   │   │   ├── ml/               # model loader implementing the classifier port
│   │   │   ├── cache/            # Redis
│   │   │   └── metrics.py        # Prometheus metrics
│   │   ├── presentation/
│   │   │   ├── api/              # REST routers + Pydantic schemas
│   │   │   └── ws/               # WebSocket handler
│   │   ├── config.py             # pydantic-settings
│   │   └── main.py
│   └── tests/
│       ├── unit/
│       ├── integration/
│       └── fixtures/             # recorded landmark samples (JSON)
├── frontend/
│   ├── public/mediapipe/         # self-hosted wasm + .task model
│   └── src/
│       ├── camera.ts
│       ├── landmarker.ts
│       ├── overlay.ts            # skeleton drawing, finger highlighting
│       ├── ws-client.ts
│       ├── i18n.ts               # hint codes → human text
│       └── pages/
├── ml/
│   ├── bukva/
│   │   ├── README.md             # license notes, download steps, BUKVA_DIR
│   │   ├── inspect_dataset.py    # explore annotation format FIRST
│   │   └── extract_landmarks.py  # videos → landmarks Parquet
│   ├── collect.py                # OpenCV + MediaPipe recorder (own webcam)
│   ├── visualize.py              # live overlay with predictions
│   ├── train.py
│   ├── evaluate.py               # signer-independent evaluation
│   ├── build_templates.py        # reference finger angles per letter
│   ├── labels.json               # class index ↔ Cyrillic letter, incl. no_sign
│   ├── data/                     # derived landmark files only (see §8)
│   └── artifacts/                # model-vN.joblib + model-vN.json
├── experiments/
│   ├── bandwidth/
│   ├── latency/
│   ├── loadtest/
│   ├── scaling/
│   └── models/
├── deploy/
│   ├── docker-compose.prod.yml
│   ├── nginx/
│   ├── prometheus/
│   └── grafana/
├── docs/
│   ├── alphabet.md
│   ├── architecture.md
│   └── report/                   # report drafts
└── .github/workflows/
```

---

## 5. Architecture rules (Onion Architecture)

- Dependencies point **inward only**:
  `presentation → application → domain`, `infrastructure → application/domain`.
- `domain/` imports nothing from FastAPI, SQLAlchemy, Redis, Pydantic,
  scikit-learn or joblib. Only stdlib + NumPy.
- `application/` depends on interfaces from `domain/ports.py`, never on
  concrete infrastructure classes. Wiring happens in `main.py` (DI via
  FastAPI dependencies).
- Pydantic schemas live only in `presentation/`. Domain uses dataclasses.
- No business logic in routers or the WebSocket handler: they parse input,
  call one use case, serialize output.

---

## 6. Gesture pipeline (critical — read before touching ML code)

### 6.1 Single source of truth
Feature extraction lives **only** in `backend/app/domain/gesture/features.py`.
`ml/` scripts import it (`uv pip install -e backend`). Never duplicate the
logic in `ml/` or in the frontend. The browser sends **raw** MediaPipe
landmarks; all normalization happens on the server.

### 6.2 Feature extraction steps
1. Input: 21 landmarks `(x, y, z)` from MediaPipe + `handedness` + `mirrored`
   + the frame aspect ratio (`width / height`).
2. If `mirrored` is true, undo the mirror (`x → 1 - x`) and flip handedness.
   (Browser video is usually mirrored for the user; MediaPipe handedness
   labels flip with it — this is a known source of bugs.)
3. Convert left hand to right hand by reflecting X, so one model covers both.
4. Correct the aspect ratio: `x *= aspect` and `z *= aspect`. MediaPipe
   normalizes `x` by frame width but `y` by frame height, so on a non-square
   frame the hand arrives anisotropically stretched and the flexion angles are
   skewed; `z` shares the scale of `x`. Steps 2 and 3 run **before** this one,
   while the mirror axis is still `x = 0.5`.
5. Translate so the wrist (landmark 0) is the origin.
6. Scale by the distance wrist → middle finger MCP (landmark 9).
7. Output vector: 63 normalized coordinates + 15 joint flexion angles
   (3 per finger), in degrees.

`aspect` is a required argument, never defaulted: a silently wrong aspect
produces plausible-looking but wrong features. Bukva's `annotations.tsv`
carries `width`/`height`; the browser reads `videoWidth`/`videoHeight`.

### 6.3 Versioning
- `FEATURE_VERSION` constant in `features.py`. Bump it on **any** change to
  the steps above.
- Every model artifact has a metadata file `model-vN.json` with:
  `feature_version`, `classes`, `model_type`, `metrics`, `trained_at`,
  `dataset_sha256`, `participants`.
- The backend refuses to start if the model's `feature_version` does not
  match the code. Changing features ⇒ retrain ⇒ new artifact.

### 6.4 Classification and stabilization
- Classifier: scikit-learn (kNN / Random Forest / small MLP; final choice
  recorded in `experiments/models/`).
- Target inference time: < 5 ms per frame on the VM. Measure it.
- The model includes a `no_sign` class. If the top prediction is
  `no_sign`, the server sends no hints and does not advance the hold counter.
- A letter counts as shown correctly only if predicted as the target with
  `confidence ≥ CONFIDENCE_THRESHOLD` for `HOLD_FRAMES` consecutive frames
  (defaults 0.7 and 10, configurable via env). Any miss resets the counter.
- `hold_progress = consecutive_hits / HOLD_FRAMES` is sent to the client.

### 6.5 Feedback engine
- Each letter has `template_angles` (mean joint angles from the dataset,
  built by `ml/build_templates.py`) and a tolerance in degrees.
- For each finger, compare the user's angles with the template; if the
  deviation exceeds tolerance, emit a hint code, e.g.
  `{"finger": "index", "code": "SHOULD_BE_STRAIGHT"}`.
- Send at most **2 hints** per frame, ordered by largest deviation.
- The server sends hint **codes**; human-readable text lives in
  `frontend/src/i18n.ts`.

---

## 7. WebSocket protocol (`/ws/practice`)

Any change here must be reflected in both backend and `frontend/src/ws-client.ts`
in the same commit, plus integration tests.

**Client → server**
```json
{"type": "auth", "token": "<JWT>"}
{"type": "start", "mode": "learn", "letter_id": 5, "width": 1280, "height": 720}
{"type": "frame", "seq": 123, "t_client": 1712345678.123,
 "handedness": "Right", "mirrored": true,
 "landmarks": [[0.51, 0.62, -0.03], "... 21 items total"]}
{"type": "stop"}
```

**Server → client**
```json
{"type": "eval", "seq": 123, "pred": "А", "confidence": 0.92,
 "correct": true, "hold_progress": 0.6,
 "hints": [{"finger": "index", "code": "SHOULD_BE_STRAIGHT"}],
 "server_ms": 2.1}
{"type": "success", "letter_id": 5, "time_to_success_ms": 3400}
{"type": "error", "code": "INVALID_FRAME", "detail": "..."}
```

Rules:
- Authenticate via the first `auth` message, not a query parameter
  (tokens must not appear in Nginx access logs).
- Client throttles frames to 10–15 Hz. Server drops frames above 30 Hz per
  connection and validates shape (exactly 21 × 3 finite floats).
- `seq` and `t_client` are echoed so the client can compute round-trip latency.
- Frame size travels in `start`, not in every frame: it is a property of the
  camera stream, and repeating it 15 times a second would inflate the payload
  the bandwidth experiment (§16.1) measures. The client **must** send a new
  `start` whenever the resolution changes — switching camera, or a phone
  rotating — otherwise every later frame is normalized with a stale aspect
  ratio. Server rejects `width`/`height` that are not positive integers.
- No endpoint in the application accepts images or video. Ever.

---

## 8. Data and dataset rules

### 8.1 Primary source: Bukva
- Open-source RSL dactyl video dataset: 3,757 videos, 33 letters
  (including dynamic ones), 155 deaf and hard-of-hearing signers, plus a
  "no sign" class.
- **First step, before writing any processing code:** read the license in
  the Bukva repository and write down in `ml/bukva/README.md` what it allows
  (training use, publishing derived landmarks, non-commercial limits).
  Then run `inspect_dataset.py` to learn the real annotation format
  (columns, signer IDs, sign start/end, official splits). Do not assume the
  format — read it from the data.
- Raw videos live outside the repo, in `BUKVA_DIR`. Never commit videos or frames.
- Static vs dynamic: the Bukva authors note disagreement on whether 7 or 8
  letters are dynamic. Decide the static subset from the reference, document
  it in `docs/alphabet.md`, keep dynamic letters out of the MVP.

### 8.2 Landmark extraction (`ml/bukva/extract_landmarks.py`)
- MediaPipe Tasks `HandLandmarker`, VIDEO mode, the **same `.task` file**
  as the frontend.
- Use only frames inside the annotated sign interval. If no interval is
  annotated, take the central part of the clip and document the heuristic.
- Sample up to K evenly spaced frames per video (configurable, default 8),
  so near-duplicate frames from one clip don't dominate.
- Drop frames with no detected hand or a low score; log drop counts per letter.
- Keep "no sign" frames as the `no_sign` class.
- Output: `ml/data/bukva_landmarks.parquet` with
  `video_id, signer_id, letter, frame_idx, handedness, x0, y0, z0, … x20, y20, z20`,
  plus `extraction_report.json` (videos processed, frames kept/dropped per class).
- Parallelize with multiprocessing and make the script resumable (skip
  already processed videos).

### 8.3 Own recordings
- `collect.py` (OpenCV + MediaPipe) and the hidden web data-collection mode
  record a small set from Ruslan's own webcam. Purpose: measure domain shift
  (Bukva cameras vs a laptop webcam, mirrored browser video). Not the main
  training set.
- Own participants are anonymized as `P01`, `P02`, … with verbal consent
  noted in `ml/data/README.md`. Only landmarks are stored, never frames or video.
- Webcam scripts run on Windows Python directly, not in WSL or Docker
  (no webcam access there by default).

### 8.4 Evaluation
- Splits are **by signer**: `GroupKFold` (k=5) on `signer_id` for model
  selection, plus a held-out test set of signers never seen in training.
  If Bukva ships an official split, also report results on it.
- Never use a random frame-level split as the main metric: frames from the
  same video leak between train and test.
- Report accuracy, macro F1, per-letter F1 and a confusion matrix.
- Report accuracy on own webcam data separately (domain-shift check).
- The Bukva paper's 83.6% top-1 (video model, all 33 letters incl. dynamic)
  is **not directly comparable** to our static-letter landmark classifier.
  The report must say so explicitly.

---

## 9. Database schema

```
users      (id, email UNIQUE, password_hash, created_at)
letters    (id, symbol, alphabet, reference_image_url, template_angles JSONB,
            tolerance_deg, is_static, sort_order)
attempts   (id, user_id FK, letter_id FK, mode, success, time_to_success_ms,
            avg_confidence, frames_count, created_at)
progress   (user_id FK, letter_id FK, attempts_count, success_rate,
            mastery_score, last_practiced_at, PRIMARY KEY (user_id, letter_id))
samples    (id, letter_id FK, participant, source, feature_version,
            landmarks JSONB, created_at)      -- web data-collection mode
```

- All schema changes go through Alembic migrations. Never edit an applied
  migration; create a new one.
- `progress` is updated in the same transaction as the attempt insert.

---

## 10. REST API

```
POST /auth/register          POST /auth/login        (JWT, bcrypt)
GET  /letters                GET  /letters/{id}
GET  /practice/next?mode=    → next letter, weak letters weighted higher
POST /attempts               GET  /me/progress
POST /samples                (data-collection mode, admin only)
GET  /health                 GET  /metrics            (Prometheus)
```

Errors use a consistent shape: `{"error": {"code": "...", "detail": "..."}}`.

---

## 11. Configuration

All config via environment variables (pydantic-settings), see `.env.example`:

```
DATABASE_URL, REDIS_URL, JWT_SECRET, JWT_TTL_MIN,
MODEL_PATH, CONFIDENCE_THRESHOLD=0.7, HOLD_FRAMES=10,
MAX_FPS_PER_CONN=30, CORS_ORIGINS, LOG_LEVEL,
BUKVA_DIR (ml scripts only)
```

Never commit `.env`, secrets, SSH keys or certificates.

---

## 12. Commands

Run from Git Bash or WSL on Windows (Makefile targets).

```
make dev          # docker compose up (api, postgres, redis) + vite dev server
make test         # pytest (unit + integration)
make lint         # ruff check + ruff format --check + mypy
make fmt          # ruff format
make migrate      # alembic upgrade head
make revision m="msg"   # new alembic migration
make extract      # ml/bukva/extract_landmarks.py → ml/data/bukva_landmarks.parquet
make train        # ml/train.py → ml/artifacts/model-vN.*
make eval         # ml/evaluate.py, signer-independent splits
make templates    # ml/build_templates.py → seeds letters.template_angles
make loadtest     # locust against the target in LOADTEST_HOST
```

`ml/collect.py` is run manually on Windows: `python ml/collect.py --participant P01`.

---

## 13. Testing

- `make lint` and `make test` must pass before every commit.
- **Unit:** features (normalization invariants: translation, scale,
  mirroring, left/right), stabilizer, feedback engine, use cases with fake
  repositories.
- **Integration:** REST endpoints and the WebSocket flow
  (auth → start → frames → success) against a real Postgres in Docker.
- Fixtures in `backend/tests/fixtures/` are recorded landmark sequences
  for a few letters — use them instead of random data.
- Every bug fix gets a regression test.

---

## 14. Deployment (OCI)

Production = `deploy/docker-compose.prod.yml` on the OCI Ubuntu VM:
nginx, api, postgres, redis, prometheus, grafana.

Known pitfalls — check these first when something does not work:
1. **OCI firewall is two layers.** Open 80/443 in the subnet Security List
   **and** on the VM itself: Oracle Ubuntu images ship iptables rules that
   reject inbound traffic except SSH.
2. **Camera requires HTTPS.** `getUserMedia` works only in a secure context
   (or localhost). Use a DuckDNS domain + certbot; auto-renew via cron/timer.
3. **WebSocket through Nginx** needs `proxy_http_version 1.1`,
   `Upgrade`/`Connection` headers and a larger `proxy_read_timeout`.
4. **CPU architecture.** Check `uname -m` on the VM. If it is `aarch64`
   (Ampere shape), build images for `linux/arm64` with `docker buildx`.
5. **Small VM.** Check `free -h` / `nproc`. If RAM is tight, run
   Prometheus/Grafana only during experiments (compose profile `monitoring`).
6. Self-host MediaPipe wasm and `.task` model under `frontend/public/` —
   do not rely on a CDN during the demo.

CI/CD (GitHub Actions): lint + tests → build images → push to GHCR →
SSH to VM → `docker compose pull && docker compose up -d` → health check.
Secrets live in GitHub Actions secrets only.

---

## 15. Observability

Prometheus metrics (in `infrastructure/metrics.py`):
- `ws_active_connections` (gauge)
- `ws_frames_received_total`, `ws_frames_dropped_total` (counters)
- `gesture_eval_seconds` (histogram: features + inference + feedback)
- `attempts_total{success}` (counter)
- standard HTTP request latency histogram

Structured JSON logs. Never log tokens, passwords or full landmark arrays.

---

## 16. Experiments (the cloud part of the report)

Each experiment lives in its own folder with: `README.md` (hypothesis,
setup, how to run), a run script, raw results CSV, an analysis script and
generated plots. Minimum 3 repetitions per configuration. Record environment
info (VM shape, CPU, RAM, OS, Docker version) next to the results.

1. **Bandwidth:** bytes per frame and per minute for landmarks JSON vs JPEG
   frames. JPEG sizes are measured **offline** from recorded video
   (encode at several qualities/resolutions) — not by sending images to the app.
2. **Latency:** end-to-end round trip (p50/p95/p99) using `seq`/`t_client`;
   local Docker vs OCI deployment; split into network vs `server_ms`.
3. **Load test:** Locust users streaming frames at 15 Hz over WebSocket;
   increase users until p95 latency or error rate breaks a threshold.
4. **Horizontal scaling:** 1 / 2 / 4 API replicas behind Nginx; max
   sustainable users per configuration.
5. **Model comparison:** accuracy and macro F1 with signer-independent
   splits (§8.4), inference time, and accuracy on own webcam data, for each
   candidate classifier.

### Honesty rule (strict)
**Never invent, estimate or "fill in" measurements, accuracy numbers or
plots.** The report may only contain numbers produced by scripts in
`experiments/` or `ml/evaluate.py`, from files in the repo. If data is
missing, leave a visible `TODO: run experiment X` placeholder. When the
results contradict a claim in the report, flag it — do not adjust the claim
silently.

---

## 17. Report

Draft in `docs/report/`. Planned structure:
1. Introduction and motivation
2. Related work (Bukva and other RSL/KSL dactyl datasets, landmark-based
   fingerspelling recognition, MediaPipe, sign language trainers)
3. Cloud architecture — SaaS application on IaaS (OCI VM, public cloud);
   containers vs VMs; why edge processing in the browser + cloud backend
4. Implementation (backend, gesture pipeline, frontend, deployment, CI/CD)
5. Experiments and results (§16)
6. Discussion: scalability, elasticity, privacy by design, cost, limitations
7. Conclusion and future work

Link each design decision to concepts from the course lectures and readings
(Erl, Marinescu, Poulton) where appropriate.

---

## 18. Git conventions

- Conventional commits: `feat:`, `fix:`, `test:`, `docs:`, `refactor:`,
  `chore:`, `ci:`, `exp:` (experiment runs).
- Small, focused commits; one logical change each.
- `main` is always deployable. Work in short-lived branches.
- Model artifacts (< 10 MB) are committed together with their metadata JSON.
- Bukva videos are never committed. Derived landmark files are committed
  only if the license allows redistribution; otherwise they are regenerated
  with `make extract`.

---

## 19. Milestones

Update checkboxes as work progresses.

- [ ] **M0 — Setup:** Bukva license checked, static letter subset in
      `docs/alphabet.md`, repo skeleton,
      Makefile, docker-compose, FastAPI `/health`, CI running lint + tests
- [ ] **M1 — Data:** Bukva downloaded, `inspect_dataset.py`,
      `extract_landmarks.py` + extraction report, `collect.py`,
      `visualize.py`, small own webcam set
- [ ] **M2 — Model:** `features.py` with unit tests, `train.py`,
      `evaluate.py` (signer-independent + confusion matrix), templates, model-v1
- [ ] **M3 — Backend:** DB + migrations, auth, letters, WebSocket evaluation,
      attempts/progress, integration tests
- [ ] **M4 — Frontend:** camera, landmarker, overlay with finger highlighting,
      Learn mode end-to-end, browser-landmark validation of the model
- [ ] **M5 — Deploy:** OCI + HTTPS + Nginx WS, CD pipeline, monitoring
- [ ] **M6 — Experiments:** bandwidth, latency, load test, scaling, models
- [ ] **M7 — Delivery:** README with screenshots/GIF, report, demo video, slides

---

## 20. Things Claude must not do

- Duplicate feature-extraction logic anywhere outside `features.py`.
- Add an endpoint or code path that uploads images or video to the server.
- Invent experimental results, accuracy figures or participant data.
- Change the WebSocket protocol, DB schema or `FEATURE_VERSION` without asking.
- Commit secrets, `.env`, keys, certificates or non-anonymized data.
- Commit Bukva videos or frames, or redistribute derived data the license forbids.
- Put business logic in routers/handlers or framework imports in `domain/`.
- Expand scope into Tier 2/3 while Tier 1 items are still incomplete.
