#!/usr/bin/env bash
# Point DUCKDNS_SUBDOMAIN at this machine's current public IP. Run by cron.
# The token has to travel in DuckDNS's URL -- that is their API -- so this
# script never prints the URL, only DuckDNS's one-word answer.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
set -a; . "$here/../.env"; set +a
: "${DUCKDNS_SUBDOMAIN:?set in deploy/.env}" "${DUCKDNS_TOKEN:?set in deploy/.env}"

answer="$(curl -fsS "https://www.duckdns.org/update?domains=${DUCKDNS_SUBDOMAIN}&token=${DUCKDNS_TOKEN}&ip=")"
echo "duckdns: ${answer}"
[ "$answer" = "OK" ]
