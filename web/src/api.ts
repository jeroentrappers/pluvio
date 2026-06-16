// Typed client for the Pluvio forecast API. Read-only; base URL from config.ts.
import { API_BASE } from './config'
import { levelFromMmPerHour, type PrecipLevel } from './domain/precip'
import type { Bounds, ForecastDto, HealthDto } from './types'

// Geographic extent the radar composite PNG is rendered onto. Matches the
// backend grid (cache.py) and the Flutter app's Env.radarBounds*; used as a
// fallback when the manifest doesn't carry bounds.
export const DEFAULT_BOUNDS: Bounds = { west: 1.5, east: 7.5, south: 48.9, north: 52.5 }

// One lead-time of the forecast, normalised for the UI.
export interface RadarFrame {
  leadMin: number
  validTime: Date
  rateMmPerH: number
  level: PrecipLevel
  source: string | null // "nowcast" | "blend" | "nwp" | null (stub-served)
  confidence: number | null // 0–1
  spriteIndex: number | null // tile in the sprite sheet
}

// The single sprite sheet for a prediction: one download, scrub by cropping.
export interface RadarSprite {
  url: string // absolute
  tileW: number
  tileH: number
  cols: number
  rows: number
}

export interface RadarData {
  issuedAt: Date // the "now" reference
  location: { lat: number; lon: number }
  modelVersion: string
  bounds: Bounds
  frames: RadarFrame[] // sorted by lead time (now → +horizon)
  sprite: RadarSprite | null
}

const abs = (url: string) => (url.startsWith('http') ? url : `${API_BASE}${url}`)

async function getJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { signal })
  if (!res.ok) throw new Error(`${path} → ${res.status}`)
  return res.json() as Promise<T>
}

export const getHealth = (signal?: AbortSignal) => getJson<HealthDto>('/healthz', signal)

// Load the full radar animation for a location: per-frame rates + overlay URLs
// from /v1/forecast, plus the grid bounds from the animation manifest.
export async function getRadar(
  lat: number,
  lon: number,
  horizonMin = 120,
  signal?: AbortSignal,
): Promise<RadarData> {
  // One request: the forecast carries the per-point rates, provenance, the grid
  // bounds, and the sprite-sheet descriptor. The whole animation is then served
  // by a single sprite image (fetched in RadarMap), so scrubbing hits no network.
  const forecast = await getJson<ForecastDto>(
    `/v1/forecast?lat=${lat}&lon=${lon}&horizon_min=${horizonMin}`,
    signal,
  )

  const frames: RadarFrame[] = forecast.frames
    .map((f) => ({
      leadMin: f.lead_min,
      validTime: new Date(f.valid_time),
      rateMmPerH: f.rate_mm_per_h,
      level: levelFromMmPerHour(f.rate_mm_per_h),
      source: f.source ?? null,
      confidence: f.confidence ?? null,
      spriteIndex: f.sprite_index ?? null,
    }))
    .sort((a, b) => a.leadMin - b.leadMin)

  const sprite = forecast.sprite
    ? {
        url: abs(forecast.sprite.url),
        tileW: forecast.sprite.tile_w,
        tileH: forecast.sprite.tile_h,
        cols: forecast.sprite.cols,
        rows: forecast.sprite.rows,
      }
    : null

  return {
    issuedAt: new Date(forecast.issued_at),
    location: forecast.location,
    modelVersion: forecast.model_version,
    bounds: forecast.bounds ?? DEFAULT_BOUNDS,
    frames,
    sprite,
  }
}

// Minutes from issue time until the first frame showing rain. 0 if raining now,
// null if dry across the whole horizon.
export function minutesUntilRain(data: RadarData): number | null {
  const first = data.frames.find((f) => f.level !== 'none')
  if (!first) return null
  return Math.max(0, Math.round((first.validTime.getTime() - data.issuedAt.getTime()) / 60000))
}
