// Wire types from the Pluvio API (see backend/src/pluvio_backend/api.py).

export interface FrameDto {
  band: string
  lead_min: number
  valid_time: string // ISO-8601 UTC
  rate_mm_per_h: number
  overlay_url: string // relative to API_BASE, e.g. /v1/overlay/nowcast/10.png?t=…
  // Provenance: which method produced this lead and how confident we are.
  // Null when the band is stub-served (no forecast-cube provenance).
  source?: string | null // "nowcast" | "blend" | "nwp"
  confidence?: number | null // 0–1, widening (decreasing) with lead
  // Tile index in the sprite sheet (ForecastDto.sprite) — the client renders
  // this frame by cropping that tile rather than fetching a per-frame PNG.
  sprite_index?: number | null
}

export interface SpriteDto {
  url: string // relative to API_BASE, e.g. /v1/sprite.png?t=…
  tile_w: number
  tile_h: number
  cols: number
  rows: number
}

export interface BandProvenance {
  source: string
  confidence: number
  producer: string // "classical" | "model" | …
}

export interface ForecastDto {
  issued_at: string // ISO-8601 UTC — the "now" reference
  location: { lat: number; lon: number }
  model_version: string
  horizon_min: number
  frames: FrameDto[]
  provenance?: Record<string, BandProvenance> | null // per-band
  // One sprite sheet with every frame tiled — animate the whole horizon from a
  // single download instead of one request per frame.
  sprite?: SpriteDto | null
  // Grid bounds for placing the overlay/sprite.
  bounds?: Bounds | null
}

export interface Bounds {
  west: number
  east: number
  south: number
  north: number
}

export interface HistoryFrameDto {
  minutes_ago: number // 0 = newest observation, negative going back
  valid_time: string // ISO-8601 UTC
  rate_mm_per_h: number
  overlay_url: string
  sprite_index?: number | null
}

export interface HistoryDto {
  observed_at: string // ISO-8601 UTC — the newest frame, the mode's "now"
  location: { lat: number; lon: number }
  span_min: number
  frames: HistoryFrameDto[]
  sprite?: SpriteDto | null
  bounds?: Bounds | null
}

export interface AnimationManifestDto {
  snapshot: string
  band: string
  bounds: Bounds | null
  frames: { lead_min: number; valid_time: string; url: string }[]
  model_version: string
}

export interface HealthDto {
  status: string
  snapshot: string | null
  issued_at: string | null
  age_seconds: number | null
  model_version: string
}
