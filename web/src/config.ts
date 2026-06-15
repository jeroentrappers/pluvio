// Runtime configuration resolution. Order, for each value:
//   1. window.__CONFIG__.*  — injected at container startup (/config.js,
//      rendered from VITE_* env). This is the production path.
//   2. import.meta.env.VITE_* — build-time value, handy in `pnpm dev`.
//   3. a sensible default.
const rc = (typeof window !== 'undefined' && window.__CONFIG__) || {}

const trimSlash = (s: string) => s.replace(/\/$/, '')

// Pluvio forecast API origin (e.g. https://pluvio.appmire.be).
export const API_BASE = trimSlash(
  rc.apiBase || import.meta.env.VITE_API_BASE || 'http://localhost:8000',
)

// Shared self-hosted vector tile server (e.g. https://tiles.appmire.be).
export const TILES_URL = trimSlash(
  rc.tilesUrl || import.meta.env.VITE_TILES_URL || 'https://tiles.appmire.be',
)

// Coarse access key gating tiles.appmire.be requests.
export const TILES_KEY = rc.tilesKey || import.meta.env.VITE_TILES_KEY || ''

// MapLibre style URL. The style JSON itself embeds the key in its sprite/glyph/
// pmtiles URLs; we only need to carry it on the initial style fetch.
export const STYLE_URL = TILES_KEY
  ? `${TILES_URL}/styles/dark.json?key=${TILES_KEY}`
  : `${TILES_URL}/styles/dark.json`
