// Wire types from the Pluvio API (see backend/src/pluvio_backend/api.py).

export interface FrameDto {
  band: string
  lead_min: number
  valid_time: string // ISO-8601 UTC
  rate_mm_per_h: number
  overlay_url: string // relative to API_BASE, e.g. /v1/overlay/nowcast/10.png?t=…
}

export interface ForecastDto {
  issued_at: string // ISO-8601 UTC — the "now" reference
  location: { lat: number; lon: number }
  model_version: string
  horizon_min: number
  frames: FrameDto[]
}

export interface Bounds {
  west: number
  east: number
  south: number
  north: number
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
