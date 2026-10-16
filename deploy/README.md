# Deploying to Azure

One Ubuntu VM running the whole stack with Docker Compose: nginx (TLS,
static site, WebSocket proxy), the API, a one-shot migration job, Postgres,
Redis, and certbot for renewals. See CLAUDE.md 14 for the pitfalls.

Tested end to end on a local Docker with a self-signed certificate
(`SELF_SIGNED=1`): HTTP redirects to HTTPS, migrations and hint templates
apply, the API runs on the trained model, registration issues a JWT, the
practice socket works over `wss://` through nginx, and the development token
is refused. Not yet run on Azure itself.

## 1. Create the VM (portal.azure.com)

Virtual machines → Create:

| setting | value |
|---|---|
| Image | Ubuntu Server 24.04 LTS, x64 |
| Size | `B2s` (2 vCPU, 4 GiB) — `B1s` with 1 GiB cannot hold the stack |
| Authentication | SSH public key, user `azureuser`; download the `.pem` |
| Inbound ports | SSH (22), HTTP (80), HTTPS (443) |
| Disk | Standard SSD, 30 GiB |

If you forgot a port: VM → Networking → *Add inbound port rule*. Azure's
Ubuntu image has no firewall of its own, so the NSG is the only layer.

**Stop the VM from the portal when you are not using it** so it shows as
*Stopped (deallocated)*. Shutting it down from inside leaves it allocated and
still billing against the student credit.

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
