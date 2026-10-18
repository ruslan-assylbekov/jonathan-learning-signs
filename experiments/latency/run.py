"""Latency: round trip of one frame, and where the time goes.

CLAUDE.md 16.2. A client signs in, opens /ws/practice through the deployed
stack (nginx, TLS, the API) and streams real Bukva frames at the browser's
15 Hz. Each frame carries ``t_client`` from this process's monotonic clock;
the server echoes it, so the round trip is measured on one clock with no
synchronisation between machines. The reply's ``server_ms`` -- features,
inference and hints inside the API -- is subtracted to leave everything else:
network, TLS, nginx, and the API's own I/O.

The result depends on the client's network as much as on the server, so every
run records a ``--network`` label: compare runs only across the same one.

Usage (from the repository root):
    uv run --project backend --group ml python experiments/latency/run.py \\
        --target https://localhost --label local-docker --network loopback --insecure
    uv run --project backend --group ml python experiments/latency/run.py \\
        --target https://bukva-learning.duckdns.org --label azure --network dorm-wifi
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import platform
import ssl
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import websockets

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml"))
import dataset  # noqa: E402

RESULTS = Path(__file__).with_name("results")
SEND_HZ = 15
RUNS = 3
SECONDS_PER_RUN = 60
PASSWORD = "latency-experiment-password"


def main() -> None:
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, help="base URL, e.g. https://localhost")
    parser.add_argument("--label", required=True, help="deployment name, e.g. local-docker")
    parser.add_argument("--network", required=True, help="client network, e.g. dorm-wifi")
    parser.add_argument("--runs", type=int, default=RUNS)
    parser.add_argument("--seconds", type=int, default=SECONDS_PER_RUN)
    parser.add_argument("--email", default="latency@tests.example.com")
    parser.add_argument(
        "--insecure", action="store_true", help="accept a self-signed certificate (local only)"
    )
    args = parser.parse_args()

    tls = ssl.create_default_context()
    if args.insecure:
        tls.check_hostname = False
        tls.verify_mode = ssl.CERT_NONE

    token = sign_in(args.target, args.email, tls)
    letters = fetch(f"{args.target}/letters", tls)
    frames = sample_frames({letter["symbol"] for letter in letters})
    letter_ids = {letter["symbol"]: letter["id"] for letter in letters}

    RESULTS.mkdir(exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    for run in range(args.runs):
        rows = asyncio.run(stream(args, token, frames, letter_ids, tls, run))
        out = RESULTS / f"{args.label}__{args.network}__{stamp}__run{run}.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        rtt = pd.Series([row["rtt_ms"] for row in rows])
        print(
            f"run {run}: {len(rows)} frames, rtt p50 {rtt.median():.1f} ms, "
            f"p95 {rtt.quantile(0.95):.1f} ms -> {out.name}"
        )

    environment = {
        "label": args.label,
        "network": args.network,
        "target": args.target,
        "health": fetch(f"{args.target}/health", tls),
        "client_platform": platform.platform(),
        "client_python": platform.python_version(),
        "send_hz": SEND_HZ,
        "seconds_per_run": args.seconds,
        "runs": args.runs,
        "started_at": stamp,
    }
    (RESULTS / f"{args.label}__{args.network}__{stamp}__environment.json").write_text(
        json.dumps(environment, indent=2), encoding="utf-8"
    )


def sign_in(base: str, email: str, tls: ssl.SSLContext) -> str:
    """A dedicated test account: registered on first use, signed in after."""
    body = {"email": email, "password": PASSWORD}
    for path in ("/auth/register", "/auth/login"):
        try:
            return str(post(f"{base}{path}", body, tls)["token"])
        except urllib.error.HTTPError as error:
            if error.code not in (409, 401):
                raise
    raise SystemExit(f"could not sign in as {email}")


def post(url: str, body: dict[str, Any], tls: ssl.SSLContext) -> dict[str, Any]:
    request = urllib.request.Request(
        url, json.dumps(body).encode(), {"content-type": "application/json"}
    )
    with urllib.request.urlopen(request, context=tls, timeout=30) as response:
        result: dict[str, Any] = json.loads(response.read())
    return result


def fetch(url: str, tls: ssl.SSLContext) -> Any:
    with urllib.request.urlopen(url, context=tls, timeout=30) as response:
        return json.loads(response.read())


def sample_frames(offered: set[str]) -> list[dict[str, Any]]:
    """Real frames, grouped by clip so each clip is practised as one letter."""
    frame = pd.read_parquet(dataset.DEFAULT_PARQUET)
    frame = frame[frame["letter"].isin(offered) & ~frame["train"]]
    records = frame.sort_values(["video_id", "frame_idx"]).to_dict("records")
    return [{str(key): value for key, value in record.items()} for record in records]


async def stream(
    args: argparse.Namespace,
    token: str,
    frames: list[dict[str, Any]],
    letter_ids: dict[str, int],
    tls: ssl.SSLContext,
    run: int,
) -> list[dict[str, Any]]:
    url = args.target.replace("https://", "wss://").replace("http://", "ws://") + "/ws/practice"
    rows: list[dict[str, Any]] = []
    interval = 1 / SEND_HZ
    total = SEND_HZ * args.seconds
    offset = run * total  # each run streams different frames

    async with websockets.connect(url, ssl=tls if url.startswith("wss") else None) as ws:
        await ws.send(json.dumps({"type": "auth", "token": token}))
        current_video = None
        next_send = time.perf_counter()

        for seq in range(total):
            record = frames[(offset + seq) % len(frames)]
            if record["video_id"] != current_video:
                current_video = record["video_id"]
                await ws.send(
                    json.dumps(
                        {
                            "type": "start",
                            "mode": "learn",
                            "letter_id": letter_ids[str(record["letter"])],
                            "width": int(record["width"]),
                            "height": int(record["height"]),
                        }
                    )
                )

            await asyncio.sleep(max(0.0, next_send - time.perf_counter()))
            next_send += interval
            sent = time.perf_counter() * 1000
            await ws.send(
                json.dumps(
                    {
                        "type": "frame",
                        "seq": seq,
                        "t_client": sent,
                        "handedness": str(record["handedness"]),
                        "mirrored": False,
                        "landmarks": [
                            [float(record[f"{axis}{i}"]) for axis in "xyz"] for i in range(21)
                        ],
                    }
                )
            )

            # One frame in flight at a time, as the browser effectively has at
            # 15 Hz against sub-second replies; anything else in between (a
            # success message) is read past.
            while True:
                reply = json.loads(await ws.recv())
                if reply.get("type") == "eval":
                    break
            received = time.perf_counter() * 1000
            assert reply["seq"] == seq
            rows.append(
                {
                    "run": run,
                    "seq": seq,
                    "rtt_ms": received - float(reply["t_client"]),
                    "server_ms": float(reply["server_ms"]),
                    "outside_server_ms": received
                    - float(reply["t_client"])
                    - float(reply["server_ms"]),
                }
            )
    return rows


if __name__ == "__main__":
    main()
