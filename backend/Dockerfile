# Pluvio backend — multi-stage so the runtime image stays small.
# Builder has gcc (some deps lack cp314 wheels yet, e.g. numcodecs); runtime
# gets just the installed venv.

# ──────────────────────────────────────────────────────────────── builder
FROM python:3.14-slim AS builder

# Static uv binary from the official image — used for venv + install.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/

# Build toolchain for any dep without a cp314 wheel today (numcodecs).
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc python3-dev \
    && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY src ./src

# A dedicated venv at /opt/pluvio keeps the install relocatable and lets the
# runtime stage copy a single tree.
RUN uv venv /opt/pluvio \
    && uv pip install --python /opt/pluvio/bin/python .

# ──────────────────────────────────────────────────────────────── runtime
FROM python:3.14-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/pluvio/bin:${PATH}" \
    PLUVIO_CACHE_ROOT=/data/forecasts \
    PLUVIO_API_HOST=0.0.0.0 \
    PLUVIO_API_PORT=8000

COPY --from=builder /opt/pluvio /opt/pluvio

# Non-root runtime user. Pre-create the cache dir so a named volume mounted
# there inherits writable ownership on first creation.
RUN useradd --create-home --uid 10001 pluvio \
    && mkdir -p /data/forecasts \
    && chown -R pluvio:pluvio /data
USER pluvio

WORKDIR /app
EXPOSE 8000

# Liveness: the API's /healthz. slim has no curl, so use urllib.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/healthz', timeout=4).status==200 else 1)"

# Default role is the API. The worker service overrides this with:
#   command: ["pluvio-worker", "schedule"]
CMD ["pluvio-api"]
