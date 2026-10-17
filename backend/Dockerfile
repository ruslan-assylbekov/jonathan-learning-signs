FROM python:3.12-slim

# Pinned rather than :latest, so a rebuild months from now resolves the same way.
COPY --from=ghcr.io/astral-sh/uv:0.11.27 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/srv/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

WORKDIR /srv

# Dependencies first, from the lock file, so code changes reuse this layer and
# every build installs exactly what was tested.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY app ./app
COPY alembic ./alembic
COPY alembic.ini ./
RUN uv sync --frozen --no-dev

RUN useradd --system --no-create-home app
USER app

EXPOSE 8000
# A frame message is ~1.5 KB; the 16 MB default would let one client make the
# server parse a 16 MB JSON document per message.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--proxy-headers", "--forwarded-allow-ips", "*", "--ws-max-size", "65536"]
