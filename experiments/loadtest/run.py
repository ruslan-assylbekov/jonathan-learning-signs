"""Load test: how many learners can practise at once before feedback degrades.

CLAUDE.md 16.3. Simulated users each hold a WebSocket to /ws/practice and
stream real held-out Bukva frames at the browser's 15 Hz, the way
ws-client.ts does: on a timer, without waiting for replies, coordinates
rounded to 4 decimals. Users are added in steps; each step warms up, then is
measured for a fixed window. The ramp stops at the first step whose p95 round
trip or loss breaks a threshold -- the previous step is the capacity.

A single client process can itself fall behind. Every step records how late
the client sends frames relative to their schedule and how much CPU the client
used; a step where the client lagged is marked ``client_limited`` and ends the
ramp, because from there on it measures this machine rather than the server.

Each simulated user has its own account (``loadtest-NNNN@tests.example.com``),
signed in before its step starts, so attempts and progress are written for
distinct users as in real use. A user practises one letter for
``LETTER_SECONDS``, cycling that clip's frames, then moves on with a new
``start``: roughly the pace of a learner, and the database work that comes
with it (one attempt per letter).

Optionally ``--stats-via`` samples ``docker stats`` on the server while the
test runs, e.g. ``--stats-via "ssh -i KEY azureuser@HOST"`` for the VM or
``--stats-via "bash -c"`` for a local stack, so each step can report the API
container's CPU. (On Windows a bare ``bash`` resolves to WSL's; name Git Bash
in full: ``--stats-via "'C:/Program Files/Git/bin/bash.exe' -c"``.)

Usage (from the repository root):
    uv run --project backend --group ml python experiments/loadtest/run.py \\
        --target https://bukva-learning.duckdns.org --label azure --network dorm-wifi
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import math
import platform
import shlex
import ssl
import subprocess
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import websockets

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.latency import run as latency  # noqa: E402

RESULTS = Path(__file__).with_name("results")
SEND_HZ = 15
LETTER_SECONDS = 5.0
STEPS = (5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 100, 130, 160, 200)
WARMUP_SECONDS = 10.0
WINDOW_SECONDS = 30.0
REPLY_TIMEOUT_SECONDS = 5.0
"""A frame unanswered this long counts as lost; also the wait after each window."""
REPEATS = 3
MAX_P95_MS = 250.0
"""A hold lasts 10 frames (0.67 s at 15 Hz); feedback a quarter second late
still lands within it. Beyond that the overlay visibly trails the hand."""
MAX_LOSS = 0.01
"""Share of measured frames with no reply within REPLY_TIMEOUT_SECONDS."""
MAX_SEND_LAG_P95_MS = 40.0
"""Client-side lag beyond this means the generator, not the server, is the limit.
An idle Windows client already shows ~15 ms (asyncio timers tick at the system
clock's 15.6 ms), so the bound sits well above that and below the 67 ms frame
interval. Round trips are timed from the actual send, so lag does not inflate them."""
SIGN_IN_CONCURRENCY = 4
TOKEN_REFRESH_SECONDS = 45 * 60  # JWT_TTL_MIN defaults to 60
STATS_SNIPPET = (
    "while :; do t=$(date -u +%s); "
    "docker stats --no-stream --format '{{.Name}} {{.CPUPerc}} {{.MemUsage}}' "
    '| sed "s/^/$t /"; done'
)


def js_round4(value: float) -> float:
    """ws-client.ts: Math.round(value * 1e4) / 1e4 (see experiments/bandwidth)."""
    return math.floor(value * 10_000 + 0.5) / 10_000


@dataclass
class Clip:
    letter_id: int
    width: int
    height: int
    handedness: str
    landmarks: list[str]
    """Each frame's landmark array, serialized once up front: the generator
    should spend its CPU on sending, not on JSON encoding."""


@dataclass
class Window:
    """What one step measures: frames sent inside [opened, closed)."""

    opened: float
    closed: float = math.inf
    sent: list[list[float]] = field(default_factory=list)
    """[send time, round trip ms or NaN, server_ms or NaN] per frame."""
    send_lag_ms: list[float] = field(default_factory=list)
    errors: Counter[str] = field(default_factory=Counter)


class Run:
    """Shared state of one ramp: the open window and every user's tasks."""

    def __init__(self, url: str, tls: ssl.SSLContext | None, clips: list[Clip]) -> None:
        self.url, self.tls, self.clips = url, tls, clips
        self.window: Window | None = None
        self.tasks: list[asyncio.Task[None]] = []
        self.connected = 0

    def error(self, code: str) -> None:
        if self.window is not None:
            self.window.errors[code] += 1

    async def user(self, index: int, token: str) -> None:
        try:
            async with websockets.connect(self.url, ssl=self.tls, open_timeout=30) as ws:
                await ws.send(json.dumps({"type": "auth", "token": token}))
                self.connected += 1
                pending: dict[int, tuple[list[float], float]] = {}
                receiver = asyncio.create_task(self.receive(ws, pending))
                try:
                    await self.send(ws, index, pending)
                finally:
                    receiver.cancel()
                    self.connected -= 1
        except asyncio.CancelledError:
            raise
        except websockets.ConnectionClosed:
            self.error("DISCONNECTED")
        except (OSError, TimeoutError, websockets.InvalidHandshake):
            self.error("CONNECT_FAILED")

    async def send(
        self,
        ws: websockets.ClientConnection,
        index: int,
        pending: dict[int, tuple[list[float], float]],
    ) -> None:
        interval = 1 / SEND_HZ
        per_letter = round(LETTER_SECONDS * SEND_HZ)
        seq = 0
        scheduled = time.perf_counter()
        # Users start on different clips so they do not all show one letter.
        for clip_number in range(index * 7919, sys.maxsize):
            clip = self.clips[clip_number % len(self.clips)]
            await ws.send(
                json.dumps(
                    {
                        "type": "start",
                        "mode": "learn",
                        "letter_id": clip.letter_id,
                        "width": clip.width,
                        "height": clip.height,
                    }
                )
            )
            for n in range(per_letter):
                await asyncio.sleep(max(0.0, scheduled - time.perf_counter()))
                now = time.perf_counter()
                window = self.window
                if window is not None and window.opened <= now < window.closed:
                    record = [now, math.nan, math.nan]
                    window.sent.append(record)
                    window.send_lag_ms.append((now - scheduled) * 1000)
                    pending[seq] = (record, now)
                await ws.send(
                    f'{{"type":"frame","seq":{seq},"t_client":{now * 1000:.3f},'
                    f'"handedness":"{clip.handedness}","mirrored":false,'
                    f'"landmarks":{clip.landmarks[n % len(clip.landmarks)]}}}'
                )
                seq += 1
                # Like a browser timer: a late tick is not made up with a burst,
                # which the server's own per-connection limit would drop.
                scheduled = max(scheduled + interval, time.perf_counter())

    async def receive(
        self, ws: websockets.ClientConnection, pending: dict[int, tuple[list[float], float]]
    ) -> None:
        async for raw in ws:
            received = time.perf_counter()
            reply = json.loads(raw)
            kind = reply.get("type")
            if kind == "eval":
                entry = pending.pop(reply["seq"], None)
                if entry is not None:
                    record, sent = entry
                    record[1] = (received - sent) * 1000
                    record[2] = float(reply["server_ms"])
            elif kind == "error":
                self.error(str(reply.get("code")))
            # pending only ever holds measured frames; drop the stale ones so
            # the dict does not grow on a server that stopped answering.
            if len(pending) > 4 * SEND_HZ * REPLY_TIMEOUT_SECONDS:
                for seq in sorted(pending)[: len(pending) // 2]:
                    del pending[seq]


class ServerStats:
    """Background reader of `docker stats` lines: (unix time, name, cpu %, mem)."""

    def __init__(self, via: str) -> None:
        self.rows: list[dict[str, Any]] = []
        self._process = subprocess.Popen(
            [*shlex.split(via), STATS_SNIPPET],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        assert self._process.stdout is not None
        for line in self._process.stdout:
            parts = line.split(maxsplit=3)
            if len(parts) == 4 and parts[2].endswith("%"):
                self.rows.append(
                    {
                        "unix": int(parts[0]),
                        "container": parts[1],
                        "cpu_percent": float(parts[2].rstrip("%")),
                        "mem": parts[3].strip(),
                    }
                )

    def stop(self) -> None:
        self._process.terminate()


def main() -> None:
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, help="base URL, e.g. https://localhost")
    parser.add_argument("--label", required=True, help="deployment name, e.g. azure")
    parser.add_argument("--network", required=True, help="client network, e.g. dorm-wifi")
    parser.add_argument("--steps", default=",".join(map(str, STEPS)), help="users per step")
    parser.add_argument("--warmup", type=float, default=WARMUP_SECONDS)
    parser.add_argument("--window", type=float, default=WINDOW_SECONDS)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--max-p95-ms", type=float, default=MAX_P95_MS)
    parser.add_argument("--max-loss", type=float, default=MAX_LOSS)
    parser.add_argument("--stats-via", help='e.g. "ssh -i KEY user@host" or "bash -c"')
    parser.add_argument("--insecure", action="store_true", help="accept a self-signed cert")
    args = parser.parse_args()
    steps = [int(s) for s in args.steps.split(",")]

    tls = ssl.create_default_context()
    if args.insecure:
        tls.check_hostname = False
        tls.verify_mode = ssl.CERT_NONE

    letters = latency.fetch(f"{args.target}/letters", tls)
    clips = load_clips({letter["symbol"]: letter["id"] for letter in letters})
    health = latency.fetch(f"{args.target}/health", tls)
    print(f"{len(clips)} clips; target {args.target}: {health}")

    RESULTS.mkdir(exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    prefix = f"{args.label}__{args.network}__{stamp}"
    stats = ServerStats(args.stats_via) if args.stats_via else None
    tokens: dict[int, tuple[str, float]] = {}
    try:
        for repeat in range(args.repeats):
            rows = asyncio.run(ramp(args, steps, clips, tls, tokens, repeat))
            pd.DataFrame(rows).to_csv(RESULTS / f"{prefix}__rep{repeat}.csv", index=False)
    finally:
        if stats is not None:
            stats.stop()
            if not stats.rows:
                print("warning: --stats-via produced no docker stats; check the command")
            pd.DataFrame(stats.rows).to_csv(RESULTS / f"{prefix}__server_stats.csv", index=False)

    environment = {
        "label": args.label,
        "network": args.network,
        "target": args.target,
        "health": health,
        "client_platform": platform.platform(),
        "client_processor": platform.processor(),
        "client_python": platform.python_version(),
        "websockets": version("websockets"),
        "send_hz": SEND_HZ,
        "letter_seconds": LETTER_SECONDS,
        "steps": steps,
        "warmup_seconds": args.warmup,
        "window_seconds": args.window,
        "reply_timeout_seconds": REPLY_TIMEOUT_SECONDS,
        "max_p95_ms": args.max_p95_ms,
        "max_loss": args.max_loss,
        "max_send_lag_p95_ms": MAX_SEND_LAG_P95_MS,
        "repeats": args.repeats,
        "server_stats": bool(stats),
        "started_at": stamp,
    }
    (RESULTS / f"{prefix}__environment.json").write_text(
        json.dumps(environment, indent=2), encoding="utf-8"
    )


def load_clips(letter_ids: dict[str, int]) -> list[Clip]:
    """Held-out test frames, one Clip per video, in a fixed shuffled order."""
    frames = latency.sample_frames(set(letter_ids))
    by_video: dict[str, list[dict[str, Any]]] = {}
    for record in frames:
        by_video.setdefault(str(record["video_id"]), []).append(record)
    clips = []
    for records in by_video.values():
        first = records[0]
        clips.append(
            Clip(
                letter_id=letter_ids[str(first["letter"])],
                width=int(first["width"]),
                height=int(first["height"]),
                handedness=str(first["handedness"]),
                landmarks=[
                    json.dumps(
                        [[js_round4(float(r[f"{axis}{i}"])) for axis in "xyz"] for i in range(21)],
                        separators=(",", ":"),
                    )
                    for r in records
                ],
            )
        )
    order = np.random.default_rng(0).permutation(len(clips))
    return [clips[i] for i in order]


async def tokens_for(
    target: str,
    tls: ssl.SSLContext,
    tokens: dict[int, tuple[str, float]],
    indices: range,
) -> list[str]:
    """Each user's own account, registered on first use; cached across repeats."""
    gate = asyncio.Semaphore(SIGN_IN_CONCURRENCY)

    async def one(index: int) -> str:
        cached = tokens.get(index)
        if cached and time.monotonic() - cached[1] < TOKEN_REFRESH_SECONDS:
            return cached[0]
        async with gate:
            email = f"loadtest-{index:04d}@tests.example.com"
            token = await asyncio.to_thread(latency.sign_in, target, email, tls)
        tokens[index] = (token, time.monotonic())
        return token

    return list(await asyncio.gather(*(one(i) for i in indices)))


async def ramp(
    args: argparse.Namespace,
    steps: list[int],
    clips: list[Clip],
    tls: ssl.SSLContext,
    tokens: dict[int, tuple[str, float]],
    repeat: int,
) -> list[dict[str, Any]]:
    url = args.target.replace("https://", "wss://").replace("http://", "ws://") + "/ws/practice"
    run = Run(url, tls if url.startswith("wss") else None, clips)
    rows: list[dict[str, Any]] = []
    try:
        for users in steps:
            started = len(run.tasks)
            new_tokens = await tokens_for(args.target, tls, tokens, range(started, users))
            # Staggered over the first half of the warm-up: a burst of TLS
            # handshakes is a different test.
            spread = args.warmup / 2 / max(1, users - started)
            for offset, token in enumerate(new_tokens):
                run.tasks.append(asyncio.create_task(run.user(started + offset, token)))
                await asyncio.sleep(spread)
            await asyncio.sleep(args.warmup / 2)

            cpu_before, wall_before = time.process_time(), time.perf_counter()
            window = Window(opened=time.perf_counter())
            window_start_unix = time.time()
            run.window = window
            await asyncio.sleep(args.window)
            window.closed = time.perf_counter()
            client_cpu = (time.process_time() - cpu_before) / (window.closed - wall_before)
            await asyncio.sleep(REPLY_TIMEOUT_SECONDS)  # let the last replies land
            run.window = None

            row = summarize(window, args, users, run.connected, client_cpu, repeat)
            row["window_start_unix"] = window_start_unix
            row["window_end_unix"] = window_start_unix + args.window
            rows.append(row)
            print(
                f"rep {repeat} users {users:4d}: {row['achieved_hz']:7.1f}/{row['offered_hz']:.0f} "
                f"replies/s, rtt p95 {row['rtt_p95_ms']:7.1f} ms, loss {row['loss']:.2%}, "
                f"send lag p95 {row['send_lag_p95_ms']:.1f} ms, client cpu {client_cpu:.0%}"
                + (f" -> stop: {row['verdict']}" if row["verdict"] != "ok" else "")
            )
            if row["verdict"] != "ok":
                break
    finally:
        for task in run.tasks:
            task.cancel()
        await asyncio.gather(*run.tasks, return_exceptions=True)
    return rows


def summarize(
    window: Window,
    args: argparse.Namespace,
    users: int,
    connected: int,
    client_cpu: float,
    repeat: int,
) -> dict[str, Any]:
    sent = np.array(window.sent, dtype=float).reshape(-1, 3)
    rtt = sent[:, 1]
    answered = ~np.isnan(rtt) & (rtt <= REPLY_TIMEOUT_SECONDS * 1000)
    seconds = window.closed - window.opened
    errors = sum(window.errors.values())
    dropped_users = window.errors["CONNECT_FAILED"] + window.errors["DISCONNECTED"]
    loss = (len(sent) - int(answered.sum())) / max(1, len(sent))

    def pct(values: np.ndarray, q: float) -> float:
        return float(np.percentile(values, q)) if len(values) else math.nan

    rtt_ok, server = rtt[answered], sent[answered, 2]
    lag = np.array(window.send_lag_ms)
    rtt_p95 = pct(rtt_ok, 95)
    if pct(lag, 95) > MAX_SEND_LAG_P95_MS:
        verdict = "client_limited"
    elif not len(rtt_ok) or rtt_p95 > args.max_p95_ms:
        verdict = "p95_exceeded"
    elif loss > args.max_loss:
        verdict = "loss_exceeded"
    elif dropped_users:
        verdict = "users_dropped"
    else:
        verdict = "ok"
    return {
        "repeat": repeat,
        "users": users,
        "connected": connected,
        "window_s": round(seconds, 3),
        "frames_sent": len(sent),
        "replies": int(answered.sum()),
        "errors": errors,
        "error_codes": json.dumps(dict(window.errors)),
        "loss": loss,
        "offered_hz": users * SEND_HZ,
        "achieved_hz": int(answered.sum()) / seconds,
        "rtt_p50_ms": pct(rtt_ok, 50),
        "rtt_p95_ms": rtt_p95,
        "rtt_p99_ms": pct(rtt_ok, 99),
        "server_p50_ms": pct(server, 50),
        "server_p95_ms": pct(server, 95),
        "send_lag_p95_ms": pct(lag, 95),
        "client_cpu": client_cpu,
        "verdict": verdict,
    }


if __name__ == "__main__":
    main()
