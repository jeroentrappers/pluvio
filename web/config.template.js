// Rendered at container startup (envsubst) from VITE_* env vars. In `pnpm dev`
// this file is replaced by public/config.js, whose empty strings make the app
// fall back to import.meta.env (.env). See docker-entrypoint.d/40-pluvio-config.sh.
window.__CONFIG__ = {
  apiBase: "${VITE_API_BASE}",
  tilesUrl: "${VITE_TILES_URL}",
  tilesKey: "${VITE_TILES_KEY}",
};
