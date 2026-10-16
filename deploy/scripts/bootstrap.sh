#!/usr/bin/env bash
# One-time setup of a fresh Ubuntu 24.04 VM. Run as the login user (azureuser).
#   bash deploy/scripts/bootstrap.sh
set -euo pipefail

echo "== Docker from Ubuntu's own repositories (no remote install script)"
sudo apt-get update -qq
sudo apt-get install -y -qq docker.io docker-compose-v2 git curl
sudo usermod -aG docker "$USER"

echo "== Swap: building images on a 4 GB VM is tight without it"
if ! swapon --show | grep -q /swapfile; then
  sudo fallocate -l 2G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile >/dev/null
  sudo swapon /swapfile
  echo "/swapfile none swap sw 0 0" | sudo tee -a /etc/fstab >/dev/null
fi

echo "== DuckDNS: refresh every 5 minutes so the domain follows the IP"
repo="$(cd "$(dirname "$0")/../.." && pwd)"
( crontab -l 2>/dev/null | grep -v duckdns.sh || true
  echo "*/5 * * * * $repo/deploy/scripts/duckdns.sh >/dev/null 2>&1" ) | crontab -

echo
echo "Done. Log out and back in so the docker group applies, then:"
echo "  cp deploy/.env.example deploy/.env && nano deploy/.env"
echo "  bash deploy/scripts/duckdns.sh"
echo "  bash deploy/scripts/init-letsencrypt.sh"
