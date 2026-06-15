# Pluvio web (PWA)

Installable Progressive Web App for the Pluvio rain radar — a React/Vite port of
the Flutter app that runs in any browser and installs to the home screen. It
consumes the same backend (`backend/`) and renders the radar PNG overlays on a
MapLibre GL basemap served from the shared `tiles.appmire.be` vector tile server.

## Stack

- **React 18 + Vite + TypeScript**
- **MapLibre GL JS + PMTiles** — vector basemap from `tiles.appmire.be`
  (`styles/dark.json`), radar composite painted as an image source.
- **recharts** — the 2-hour intensity forecast chart.
- **i18next** — `nl` / `en` / `fr` / `de` (same copy as the Flutter app).
- **vite-plugin-pwa (Workbox)** — installable, with runtime caching of the API,
  the radar overlays, and the basemap tiles.

## Data flow

All data comes from the Pluvio API (see `../backend`):

| Endpoint | Use |
| --- | --- |
| `GET /v1/forecast?lat&lon&horizon_min=120` | per-frame rate (mm/h) + overlay URL + valid time; drives the chart, headline and timeline |
| `GET /v1/animation/manifest.json?band=nowcast` | grid `bounds` for anchoring the overlay |
| `GET /v1/overlay/{band}/{lead}.png` | the radar composite PNG per timestep |

## Configuration

Runtime config is resolved in `src/config.ts` from, in order: `window.__CONFIG__`
(injected at container start), `import.meta.env.VITE_*` (build-time / dev), then a
default.

| Key | Meaning |
| --- | --- |
| `VITE_API_BASE` | Pluvio API origin (e.g. `https://pluvio.appmire.be`) |
| `VITE_TILES_URL` | Tile server origin (`https://tiles.appmire.be`) |
| `VITE_TILES_KEY` | Coarse access key for the tile server (lives in `gpsinfo` `deploy/ansible/secrets.yml`) |

## Develop

```bash
pnpm install
cp .env.example .env   # fill in VITE_TILES_KEY
pnpm dev               # http://localhost:5173
```

## Build / container

```bash
pnpm build             # tsc --noEmit && vite build  → dist/
docker build -t pluvio-web .
```

The container is nginx serving `dist/`. `VITE_API_BASE` / `VITE_TILES_URL` /
`VITE_TILES_KEY` are injected at **startup** into `/config.js`
(`docker-entrypoint.d/40-pluvio-config.sh`), so one image works across
environments.

## Notes

- `tiles.appmire.be` serves **vector** PMTiles; the full basemap requires
  `planet.pmtiles` (built by the throttled Planetiler job in the `gpsinfo`
  deploy). Until it finishes baking, only the low-zoom relief shows — styles,
  fonts and sprites are already live.
- The Flutter app (`../lib`, `../android`, `../ios`) is kept for now; this PWA
  runs alongside it until it reaches parity in production.
