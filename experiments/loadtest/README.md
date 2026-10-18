# Experiment: load test (CLAUDE.md 16.3)

## Question

How many learners can practise at the same time on one deployment before the
feedback they get stops being timely, and what runs out first?

## Hypothesis

Per frame the API spends about a millisecond on features, inference and hints
(`server_ms`, experiments/latency), but it runs as a **single uvicorn
process**, so all of it shares one CPU core however many cores the VM has.
Capacity should therefore be set by one core: the round trip stays flat while
that core has headroom, then climbs steeply once frames queue faster than they
are answered. Horizontal scaling (16.4) is the remedy this predicts.

## Setup

- `run.py` is an asyncio generator on `websockets` (CLAUDE.md 3). Each
  simulated user has its own account, holds one WebSocket to `/ws/practice`
  and streams real held-out Bukva frames **at 15 Hz on a timer, without
  waiting for replies**, as `frontend/src/ws-client.ts` does, with coordinates
  rounded to 4 decimals. A late tick is not made up with a burst, as with a
  browser timer.
- A user practises one letter for 5 s, cycling that clip's frames, then sends
  a new `start` for the next clip. That is roughly a learner's pace, and it
  brings the database work real use has: an attempt and a progress update per
  letter.
- **Ramp:** users are added in steps (5, 10, 15, 20, 25, 30, 40, 50, ...),
  staggered over the first half of a 10 s warm-up. Each step is then measured
  for 30 s, and frames are given 5 s to be answered.
- **A step fails** when the round trip's p95 exceeds 250 ms or more than 1 % of
  frames go unanswered (or users are disconnected). A hold lasts 10 frames,
  0.67 s at 15 Hz; feedback a quarter of a second late still lands inside it.
  **Capacity** is the last step that passed before the first failure.
- **The generator checks itself.** It records how late each frame went out
  against its schedule and the client process's CPU. If the send lag's p95
  passes 40 ms the step is marked `client_limited` and the ramp ends: past that
  point it would measure the laptop, not the server. (An idle Windows client
  already shows ~15 ms here: asyncio's timers tick with the 15.6 ms system
  clock. Round trips are timed from the actual send, so this does not inflate
  them.)
- **Server CPU**, optional: `--stats-via` samples `docker stats` on the
  server during the run, and `analyze.py` reports the API container's CPU per
  step (100 % = one core).
- **Repetitions:** the whole ramp three times, signed-in accounts reused.

## How to run

```bash
# Azure, from a stable network (name it honestly), with server CPU via SSH
make loadtest LOADTEST_NETWORK=dorm-wifi \
    LOADTEST_STATS="ssh -i C:/Users/<you>/Downloads/sign-assistant_key.pem azureuser@<vm-ip>"

# or directly
uv run --project backend --group ml python experiments/loadtest/run.py \
    --target https://bukva-learning.duckdns.org --label azure --network dorm-wifi \
    --stats-via "ssh -i KEY azureuser@HOST"
uv run --project backend --group ml python experiments/loadtest/analyze.py
```

Local baseline: the production API image on its own, against the dev
Postgres and Redis (`make setup`), without nginx or TLS:

```bash
docker run -d --rm --name loadtest-api --network sign_learning_default \
    -p 127.0.0.1:8002:8000 \
    -e DATABASE_URL=postgresql+asyncpg://sign:sign@postgres:5432/sign \
    -e REDIS_URL=redis://redis:6379/0 -e JWT_SECRET=<any 32+ characters> \
    -e MODEL_PATH=/srv/ml/artifacts/model-v2.joblib \
    -v "$PWD/ml/artifacts:/srv/ml/artifacts:ro" sign-assistant-api:latest
uv run --project backend --group ml python experiments/loadtest/run.py \
    --target http://127.0.0.1:8002 --label local-docker-api --network loopback \
    --stats-via "'C:/Program Files/Git/bin/bash.exe' -c"
```

(On Windows a bare `bash` starts WSL's, which has no `docker`.) Do not test
against `uvicorn` running natively on Windows: in a smoke test its Proactor
event loop already failed at 20 users (p95 above 250 ms), where the same code
in a Linux container answers within 18 ms at p95. That says nothing about the
deployment; it was a smoke test, not a result, and it is not in `results/`.

The test accounts (`loadtest-NNNN@tests.example.com`) stay on the target.
Remove them with:
`docker compose -f deploy/docker-compose.prod.yml --env-file deploy/.env exec -T postgres psql -U sign -d sign -c "DELETE FROM users WHERE email LIKE '%@tests.example.com'"`

## Results

Generated, never hand-edited: [`results/summary.md`](results/summary.md), with
the plot `results/loadtest.png`.

- Local baseline (API container on the laptop, loopback): measured.
- TODO: run experiment 16.3 against Azure from a stable network.

## Limitations

- **Burstable VM.** The Azure VM is a B-series size: it runs at full speed
  on banked CPU credits, then drops to a baseline. A long test can drain the
  credits. Check *CPU Credits Remaining* in the portal before and after a run,
  and do not compare runs made on full and drained credits.
- The local baseline shares the laptop between generator and server, and has
  no nginx or TLS in the path. It shows the API's own ceiling, not the
  production stack's.
- One client machine: beyond a few hundred users the generator would need
  splitting across processes or machines. The `client_limited` verdict
  says when that point is reached.
- Simulated users all send at exactly 15 Hz from real but recorded frames. Real
  browsers vary their rate with the device, and they stop sending while no
  hand is visible, so real load per user is at most this.
