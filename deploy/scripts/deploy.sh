#!/usr/bin/env bash
# Update the running stack to the latest main. Run from the repository root.
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a; . deploy/.env; set +a

git pull --ff-only
docker compose -f deploy/docker-compose.prod.yml --env-file deploy/.env up -d --build

echo -n "waiting for https://${DOMAIN}/health "
for _ in $(seq 1 30); do
  if body="$(curl -fsS "https://${DOMAIN}/health" 2>/dev/null)"; then
    echo; echo "$body"
    case "$body" in
      *'"stub":true'*) echo "WARNING: running on the stub classifier -- is ml/artifacts there?"; exit 1 ;;
    esac
    exit 0
  fi
  echo -n "."; sleep 2
done
echo; echo "health check failed"; exit 1
