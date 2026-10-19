# Experiment: latency (CLAUDE.md 16.2)

## Question

How long does one frame take from the browser to a verdict and back, on the
local stack and on the Azure deployment, and how much of that is the server's
own work rather than the network?

## Hypothesis

Server-side evaluation is a few milliseconds (CLAUDE.md 6.4's budget is
5 ms), so on the internet the round trip is dominated by the network, and the
difference between local and Azure is almost entirely network distance.

## Setup

- `run.py` signs in a dedicated test account, opens `/ws/practice` through the
  full production stack (nginx, TLS, the API) and streams real held-out Bukva
  frames at the browser's 15 Hz for 60 s, three runs per configuration, each
  run on different frames. Clips are practised as the letter they show.
- **Round trip** uses the echoed `t_client` from the client's monotonic clock,
  so no clock synchronisation between machines is involved.
- **Server time** is the reply's `server_ms`: features, inference and hints
  inside the API. **Outside** = round trip − server time: network, TLS, nginx
  and the API's I/O.
- One frame is in flight at a time, as in the browser at 15 Hz against replies
  far shorter than 67 ms.
- Every run is labelled with the client's network. The round trip depends on
  it as much as on the server, so only runs on the same network compare.

## How to run

```bash
# local: the production compose stack on this machine, self-signed
SELF_SIGNED=1 bash deploy/scripts/init-letsencrypt.sh      # with DOMAIN=localhost in deploy/.env
uv run --project backend --group ml python experiments/latency/run.py \
    --target https://localhost --label local-docker --network loopback --insecure

# Azure, from a stable network (name it honestly)
uv run --project backend --group ml python experiments/latency/run.py \
    --target https://bukva-learning.duckdns.org --label azure --network dorm-wifi

uv run --project backend --group ml python experiments/latency/analyze.py
```

After an Azure run, record the server next to the results (VM size, region,
CPU, memory, OS, Docker, deployed commit):
`uv run --project backend --group ml python experiments/vm_info.py --via "ssh -i KEY azureuser@HOST" --out experiments/latency/results/<run prefix>__vm.json`

The test account (`latency@tests.example.com` by default) stays on the target.
Remove it from the deployment with:
`docker compose -f deploy/docker-compose.prod.yml --env-file deploy/.env exec -T postgres psql -U sign -d sign -c "DELETE FROM users WHERE email LIKE '%@tests.example.com'"`

## Results

Generated, never hand-edited: [`results/summary.md`](results/summary.md).

- Local Docker (Windows 11 laptop, Docker Desktop, loopback): measured.
- Azure (B2ats v2), from a student-dorm Wi-Fi network: measured 2026-10-19.

## Limitations

- The local stack runs under Docker Desktop's WSL2 VM, whose networking adds
  to "outside"; a native Linux host would be faster.
- `server_ms` excludes JSON parsing and validation of the message in the
  WebSocket handler; those land in "outside".
- One client network per deployment so far. Wi-Fi adds its own tail (p95
  well above p50 on Azure); a wired client would separate the Wi-Fi share
  from the internet path.
- The browser's own time -- MediaPipe detection before sending -- is not part
  of the round trip measured here.
