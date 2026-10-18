# Deploying to Azure

One Ubuntu VM running the whole stack with Docker Compose: nginx (TLS,
static site, WebSocket proxy), the API, a one-shot migration job, Postgres,
Redis, and certbot for renewals. See CLAUDE.md 14 for the pitfalls.

Running at <https://bukva-learning.duckdns.org> since 2026-10-18 on a
`B2ats v2` VM (836 MiB usable RAM after Azure's reservation). Checked there:
the images build with the 2 GB swap, HTTP redirects to HTTPS, the certificate
is Let's Encrypt's, migrations and hint templates apply, the API runs on the
trained model, registration issues a JWT, the practice socket works over
`wss://`, the development token is refused, and `deploy.sh` updates the stack
in place.

## 1. Create the VM (portal.azure.com)

Virtual machines → Create:

| setting | value |
|---|---|
| Image | Ubuntu Server 24.04 LTS, x64 |
| Size | `B2ats v2` (2 vCPU, 1 GiB, x64) — free for 12 months on Azure for Students |
| Authentication | SSH public key, user `azureuser`; download the `.pem` |
| Inbound ports | SSH (22), HTTP (80), HTTPS (443) |
| Disk | Standard SSD, 30 GiB |

If you forgot a port: VM → Networking → *Add inbound port rule*. Azure's
Ubuntu image has no firewall of its own, so the NSG is the only layer.

### Which size

Azure for Students gives $100 of credit for 12 months, no card, plus 750
hours a month each of three small VM sizes free for those 12 months: `B1s`,
`B2pts v2` (Arm) and `B2ats v2` (x64), all with 1 GiB of RAM. 750 hours is a
whole month, so one of them can run around the clock at no cost.

1 GiB is enough. Measured on the production stack with 10 concurrent practice
sessions streaming real frames (`docker stats`, 2026-10-19): API 130 MiB,
Postgres 58 MiB, nginx 24 MiB, certbot 7 MiB, Redis 5 MiB — about 225 MiB,
leaving room for Ubuntu and Docker. Building the images on the VM is the
tight part; `bootstrap.sh` adds 2 GB of swap for it. Not yet measured: the
build's peak memory on a real 1 GiB VM.

Prefer `B2ats v2`: two cores rather than `B1s`'s one, and x64, so nothing has
to be built for Arm. Move to a larger size (VM → Size, a restart) only if the
scaling experiment's extra API replicas or Prometheus/Grafana need it; that
spends credit.

**Stop the VM from the portal when you are not using it** so it shows as
*Stopped (deallocated)*. Shutting it down from inside leaves it allocated and
still billing. For the free sizes this matters less, but a stopped VM also
cannot be abused.

## 2. Point DuckDNS at it

At duckdns.org, create a subdomain (e.g. `signlearn`) and note your token.
The VM keeps it updated itself (step 4), so the domain survives the IP
changing after a deallocate.

## 3. Get the code onto the VM

```bash
ssh -i ~/Downloads/your-key.pem azureuser@<public-ip>
git clone https://github.com/ruslan-assylbekov/jonathan-learning-signs.git sign
cd sign
```

The trained model (`ml/artifacts/`) is in the repository, so nothing else has
to be copied.

## 4. One-time setup

```bash
bash deploy/scripts/bootstrap.sh   # Docker, 2 GB swap, DuckDNS cron
exit                               # log out and back in: docker group
```

```bash
cd sign
cp deploy/.env.example deploy/.env
nano deploy/.env    # DOMAIN, email, DuckDNS; generate secrets with:
openssl rand -hex 32
bash deploy/scripts/duckdns.sh     # should print "duckdns: OK"
```

## 5. Certificate and start

```bash
STAGING=1 bash deploy/scripts/init-letsencrypt.sh   # optional rehearsal
FORCE=1 bash deploy/scripts/init-letsencrypt.sh     # the real certificate
```

Let's Encrypt rate-limits failed attempts, so rehearse with `STAGING=1` if
anything is uncertain — DNS not yet pointing at the VM is the usual cause.

Open `https://<your-domain>`, create an account, start the camera.
`https://<your-domain>/health` should report `"stub": false`.

## Updating

```bash
bash deploy/scripts/deploy.sh   # git pull, rebuild, wait for /health
```

It fails loudly if the app came up on the stub classifier, which is what
happens when the model artifact is missing.

## Useful

```bash
C="docker compose -f deploy/docker-compose.prod.yml --env-file deploy/.env"
$C ps                    # what is running
$C logs -f api           # API log
$C logs migrate          # migrations and template load
free -h                  # memory headroom
```
