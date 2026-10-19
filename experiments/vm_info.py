"""Record the server's environment next to experiment results (CLAUDE.md 16).

Runs a fixed, read-only shell snippet on the server through ``--via`` and
writes what it reports as JSON: the Azure VM size and region (from the
instance metadata service), CPU, memory, OS, Docker versions and the deployed
commit.

Usage (from the repository root):
    uv run --project backend --group ml python experiments/vm_info.py \\
        --via "ssh -i KEY azureuser@HOST" \\
        --out experiments/latency/results/azure__dorm-wifi__STAMP__vm.json
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path

IMDS = "http://169.254.169.254/metadata/instance/compute"
PROBES = {
    "vm_size": f"curl -s -m 5 -H Metadata:true '{IMDS}/vmSize?api-version=2021-02-01&format=text'",
    "region": f"curl -s -m 5 -H Metadata:true '{IMDS}/location?api-version=2021-02-01&format=text'",
    "cpu_model": "lscpu | sed -n 's/^Model name: *//p'",
    "architecture": "uname -m",
    "vcpus": "nproc",
    "memory_mib": "free -m | awk '/^Mem:/ {print $2}'",
    "swap_mib": "free -m | awk '/^Swap:/ {print $2}'",
    "os": ". /etc/os-release && echo $PRETTY_NAME",
    "kernel": "uname -r",
    "docker": "docker version --format '{{.Server.Version}}'",
    "compose": "docker compose version --short",
    "deployed_commit": "git -C ~/sign rev-parse --short HEAD",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--via", required=True, help='e.g. "ssh -i KEY user@host"')
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    # One line per probe, in order, so one connection answers all of them.
    snippet = "; ".join(f'echo "$({command})"' for command in PROBES.values())
    result = subprocess.run(
        [*shlex.split(args.via), snippet], capture_output=True, text=True, timeout=120
    )
    if result.returncode != 0:
        raise SystemExit(f"probe failed: {result.stderr.strip()}")
    lines = result.stdout.splitlines()
    if len(lines) != len(PROBES):
        raise SystemExit(f"expected {len(PROBES)} lines, got {lines!r}")

    info = dict(zip(PROBES, (line.strip() for line in lines), strict=True))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
