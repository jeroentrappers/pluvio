// Runtime config injected by /config.js (rendered from env at container start).
interface Window {
  __CONFIG__?: {
    apiBase?: string
    tilesUrl?: string
    tilesKey?: string
  }
}
