#!/usr/bin/env bash
# Obtain the first TLS certificate. Run once, from the repository root.
#   bash deploy/scripts/init-letsencrypt.sh           real certificate
#   STAGING=1 bash deploy/scripts/init-letsencrypt.sh  rehearse against LE staging
#   SELF_SIGNED=1 bash deploy/scripts/init-letsencrypt.sh   local test, no LE at all
#
# The chicken-and-egg: nginx will not start without a certificate, and Let's
# Encrypt needs nginx running to answer its challenge. So start nginx on a
# throwaway self-signed certificate, then swap in the real one and reload.
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a; . deploy/.env; set +a
: "${DOMAIN:?set in deploy/.env}"

compose=(docker compose -f deploy/docker-compose.prod.yml --env-file deploy/.env)
live="/etc/letsencrypt/live/${DOMAIN}"

if [ -f "deploy/certbot/conf/live/${DOMAIN}/fullchain.pem" ] && [ "${FORCE:-}" != "1" ]; then
  echo "A certificate for ${DOMAIN} already exists; FORCE=1 to replace it."
  exit 0
fi
mkdir -p deploy/certbot/conf deploy/certbot/www

echo "== Throwaway certificate so nginx can start"
"${compose[@]}" run --rm --entrypoint "sh -c 'mkdir -p ${live} && \
  openssl req -x509 -nodes -newkey rsa:2048 -days 1 \
  -keyout ${live}/privkey.pem -out ${live}/fullchain.pem -subj /CN=${DOMAIN}'" certbot

echo "== Starting the stack"
"${compose[@]}" up -d --build

if [ "${SELF_SIGNED:-}" = "1" ]; then
  echo "SELF_SIGNED=1: keeping the throwaway certificate (browsers will warn)."
  exit 0
fi

echo "== Requesting the real certificate for ${DOMAIN}"
"${compose[@]}" run --rm --entrypoint "sh -c 'rm -rf ${live} \
  /etc/letsencrypt/archive/${DOMAIN} /etc/letsencrypt/renewal/${DOMAIN}.conf'" certbot
"${compose[@]}" run --rm --entrypoint "certbot certonly --webroot -w /var/www/certbot \
  -d ${DOMAIN} --email ${LETSENCRYPT_EMAIL:?set in deploy/.env} --agree-tos --no-eff-email \
  ${STAGING:+--staging}" certbot

"${compose[@]}" exec nginx nginx -s reload
echo "Done: https://${DOMAIN}"
