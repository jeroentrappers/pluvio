// Typed client for the Pluvio forecast API. Read-only; base URL from config.ts.
import { API_BASE } from './config'
import { levelFromMmPerHour, type PrecipLevel } from './domain/precip'
import type { AnimationManifestDto, Bounds, ForecastDto, HealthDto } from './types'

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
  overlayUrl: string // absolute
}

export interface RadarData {
  issuedAt: Date // the "now" reference
  location: { lat: number; lon: number }
  modelVersion: string
  bounds: Bounds
  frames: RadarFrame[] // sorted by lead time (now → +horizon)
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
  const [forecast, manifest] = await Promise.all([
    getJson<ForecastDto>(
      `/v1/forecast?lat=${lat}&lon=${lon}&horizon_min=${horizonMin}`,
      signal,
    ),
    // Bounds only; tolerate failure and fall back to the known grid extent.
    getJson<AnimationManifestDto>('/v1/animation/manifest.json?band=nowcast', signal).catch(
      () => null,
    ),
  ])

  const frames: RadarFrame[] = forecast.frames
    .map((f) => ({
      leadMin: f.lead_min,
      validTime: new Date(f.valid_time),
      rateMmPerH: f.rate_mm_per_h,
      level: levelFromMmPerHour(f.rate_mm_per_h),
      overlayUrl: abs(f.overlay_url),
    }))
    .sort((a, b) => a.leadMin - b.leadMin)

  return {
    issuedAt: new Date(forecast.issued_at),
    location: forecast.location,
    modelVersion: forecast.model_version,
    bounds: manifest?.bounds ?? DEFAULT_BOUNDS,
    frames,
  }
}

// Minutes from issue time until the first frame showing rain. 0 if raining now,
// null if dry across the whole horizon.
export function minutesUntilRain(data: RadarData): number | null {
  const first = data.frames.find((f) => f.level !== 'none')
  if (!first) return null
  return Math.max(0, Math.round((first.validTime.getTime() - data.issuedAt.getTime()) / 60000))
}
