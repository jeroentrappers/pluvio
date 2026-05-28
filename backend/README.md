# Pluvio backend

The forecast cache + HTTP API the Pluvio app talks to. Wraps the open KMI /
KNMI feeds today; the trained CorrDiff model lands in as a one-function
swap when it's ready.

Python 3.14, managed with **uv**. Quality gate: **ruff** (lint + format) and
**mypy** must pass before commits.

## Quick start (local dev)

```bash
cd backend/
uv sync --extra dev          # creates .venv on 3.14, installs from uv.lock
cp .env.example .env

# Refresh one band once, by hand, to populate the cache.
uv run pluvio-worker tick --band nowcast

# Run the HTTP API.
uv run pluvio-api --reload

# Browse:
curl 'http://localhost:8000/healthz'
curl 'http://localhost:8000/v1/forecast?lat=50.85&lon=4.35'
curl -o /tmp/overlay.png 'http://localhost:8000/v1/overlay/nowcast/30.png'
```

## Quality gate

Run all four before committing:

```bash
uv run ruff check src tests          # lint
uv run ruff format --check src tests # format (drop --check to apply)
uv run mypy                          # type check
uv run pytest -q                     # tests
```

## Docker

One image, two roles (API + scheduled worker) sharing a cache volume.

```bash
# Local dev — rebuild from the checkout
docker compose up -d --build
curl http://localhost:8000/healthz
docker compose logs -f worker        # see the band-tick schedule
docker compose down
```

The image is multi-stage: a builder with gcc installs the wheels (some deps
lack cp314 wheels yet, so they compile from source), and a slim runtime
just copies the resulting `/opt/pluvio` venv. Final image is small and
runs as a non-root user.

## Continuous delivery (GHCR)

`.github/workflows/backend.yml` runs the quality gate on every push / PR
touching `backend/**`, and on pushes to `main` (or `v*` tags) builds and
pushes the image to **GHCR**:

```
ghcr.io/jeroentrappers/pluvio-backend:latest    # main branch
ghcr.io/jeroentrappers/pluvio-backend:sha-<7>   # immutable, per commit
ghcr.io/jeroentrappers/pluvio-backend:v1.2.3    # on v* tags
```

Auth uses the workflow's `GITHUB_TOKEN` — no manual secret. **One-time
setup:** after the first publish, open the package settings on GitHub and
either keep it private (the deploy host then needs a registry read-only
PAT) or flip it public for a friction-free `docker pull`.

## Production deploy

DNS for `pluvio.appmire.be` points at a Docker host. Then on that host:

```bash
git clone git@github.com:jeroentrappers/pluvio.git
cd pluvio/backend
docker compose pull          # pulls ghcr.io/jeroentrappers/pluvio-backend:latest
docker compose up -d
```

Redeploys are just `docker compose pull && docker compose up -d` — the
worker resumes from the persisted cache volume.

TLS in front of `api:8000`. Caddy is the simplest:

```caddyfile
pluvio.appmire.be {
    reverse_proxy localhost:8000
}
```

## What lives where

```
src/pluvio_backend/
├── config.py            ← env-driven Settings
├── schedules.py         ← Band definitions (nowcast/short/medium/long)
├── cache.py             ← Zarr layout + atomic latest-symlink swap
├── colormap.py          ← Shared with the Flutter app's PrecipitationPalette
├── tiler.py             ← Pre-renders PNG overlays
├── kmi_signing.py       ← Daily-rotating md5 for app.meteo.be
├── stubs.py             ← Stub inference using KMI getForecasts
├── inference_worker.py  ← `pluvio-worker tick --band <name>`
└── api.py               ← `pluvio-api` — FastAPI app
```

## Refresh schedule

| Band     | Lead time | Cadence | Cron             |
|----------|-----------|---------|------------------|
| nowcast  | 0–2 h     | 5 min   | `*/5 * * * *`    |
| short    | 2–12 h    | 1 h     | `0 * * * *`      |
| medium   | 12–24 h   | 3 h     | `0 */3 * * *`    |
| long     | 24–240 h  | 12 h    | `0 0,12 * * *`   |

Two run modes:

- **Cron** (production) — one process per tick, scheduled by the OS. See
  `crontab.example`. Robust, no in-process state, restarts trivially.
- **APScheduler** (development) — `pluvio-worker schedule` runs all four
  bands inside one long-lived process. Same code paths.

## API

| Endpoint | Purpose |
|---|---|
| `GET /healthz` | Liveness + cache freshness. Returns `degraded` past `PLUVIO_CACHE_STALE_AFTER_SECONDS`. |
| `GET /v1/forecast?lat=&lon=&horizon_min=` | Point forecast: per-band, per-lead-time rate + overlay URL. |
| `GET /v1/overlay/{band}/{lead_min}.png` | Pre-rendered radar-style overlay PNG. Cache-friendly. |
| `GET /v1/animation/manifest.json?band=` | Frame list for the radar animation: URLs + valid times. |

### Forecast response shape

```jsonc
{
  "issued_at": "2026-05-26T12:05:00Z",
  "location": { "lat": 50.85, "lon": 4.35 },
  "model_version": "stub-0.1",
  "horizon_min": 1440,
  "frames": [
    {
      "band": "nowcast",
      "lead_min": 0,
      "valid_time": "2026-05-26T12:05:00Z",
      "rate_mm_per_h": 0.0,
      "overlay_url": "/v1/overlay/nowcast/0.png?t=2026-05-26T12-05-00Z"
    },
    { "...": "more frames at 10-min, 1-h, 3-h cadences out to horizon_min" }
  ]
}
```

## Atomic refresh

Every refresh writes to a *new* snapshot directory. Only when the snapshot
is fully written does the worker call `os.replace` to flip the `latest`
symlink. The API only ever sees a consistent state — partial writes never
become visible.

If the worker crashes mid-write, the half-written directory is left in
place but `latest` still points at the previous good one. The next
successful tick swaps cleanly. `cache.prune()` clears stale partials at
the end of every successful tick (default: keep last 24 snapshots).

## Storage footprint

- Per snapshot: ~1 MB zarr + ~150 KB overlays + ~150 KB Parquet ≈ ~1.3 MB.
- 24 snapshots: ~30 MB.
- 7-day rolling history: ~270 MB.

Local SSD is fine for a single instance. For multi-region serving, mount
the cache root on S3 + a CDN.

## Deployment

A single small Linux box runs everything: cron for the worker, systemd
for the API, the cache on local SSD. CPU is enough — no GPU until the
trained model lands.

```bash
# /etc/systemd/system/pluvio-api.service
[Unit]
Description=Pluvio forecast API
After=network.target

[Service]
User=pluvio
WorkingDirectory=/opt/pluvio-backend
EnvironmentFile=/opt/pluvio-backend/.env
ExecStart=/opt/pluvio-backend/.venv/bin/pluvio-api
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

## Swapping in the trained model

`inference_worker.run_tick(...)` takes an ``infer`` callable defaulted to
`stub_band`. When the CorrDiff model is trained:

```python
from corrdiff_runtime import infer as model_infer
run_tick("nowcast", infer=model_infer)
```

Same input grid, same output shape — no API or cache changes.
