// Typed client for the Pluvio forecast API. Read-only; base URL from config.ts.
import { API_BASE } from './config'
import { levelFromMmPerHour, type PrecipLevel } from './domain/precip'
import type { Bounds, ForecastDto, HealthDto, HistoryDto , HistoryTilesDto} from './types'

// Geographic extent the radar composite PNG is rendered onto. Matches the
// backend grid (cache.py) and the Flutter app's Env.radarBounds*; used as a
// fallback when the manifest doesn't carry bounds.
export const DEFAULT_BOUNDS: Bounds = { west: 1.5, east: 7.5, south: 48.9, north: 52.5 }

// One lead-time of the forecast, normalised for the UI.
export interface RadarFrame {
  // Which side of t=0 this frame sits on in the seamless timeline:
  // 'obs' = measured composite, 'fc' = forecast. Absent in single-source modes.
  kind?: 'obs' | 'fc'
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

// Hi-res history tiles: the 1-km cube split into fixed square tiles, each its
// own sprite sheet. The map downloads only the tiles its viewport shows and
// falls back to the overview sprite below the zoom threshold (and always when
// tiles are absent).
export interface HistoryTiles {
  tilePx: number
  nx: number
  ny: number
  gridH: number
  gridW: number
  cols: number
  count: number
  mtime: number
  bounds: Bounds
  index: Record<string, number>
  urlFor: (tx: number, ty: number) => string
}

export interface RadarData {
  issuedAt: Date // the "now" reference
  location: { lat: number; lon: number }
  modelVersion: string
  bounds: Bounds
  frames: RadarFrame[] // sorted by lead time (now → +horizon)
  sprite: RadarSprite | null
  tiles?: HistoryTiles | null
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


// Observed rainfall (history mode): the last ~3 h of gauge-validated radar QPE
// (see /v1/history). Mapped onto the same RadarData shape the forecast uses so
// the map, timeline and chart animate it unchanged — leads are negative
// ("40 minutes ago"), the newest observation is the mode's "now".
export async function getHistory(
  lat: number,
  lon: number,
  spanMin = 180,
  signal?: AbortSignal,
): Promise<RadarData> {
  const dto = await getJson<HistoryDto>(
    `/v1/history?lat=${lat}&lon=${lon}&span_min=${spanMin}`,
    signal,
  )
  const observedAt = new Date(dto.observed_at)
  const frames: RadarFrame[] = dto.frames
    .map((f) => ({
      leadMin: f.minutes_ago,
      validTime: new Date(f.valid_time),
      rateMmPerH: f.rate_mm_per_h,
      level: levelFromMmPerHour(f.rate_mm_per_h),
      source: 'radar',
      confidence: null,
      spriteIndex: f.sprite_index ?? null,
    }))
    .sort((a, b) => a.leadMin - b.leadMin)
  const sprite = dto.sprite
    ? {
        url: abs(dto.sprite.url),
        tileW: dto.sprite.tile_w,
        tileH: dto.sprite.tile_h,
        cols: dto.sprite.cols,
        rows: dto.sprite.rows,
      }
    : null
  // Hi-res tile manifest: optional — 404 simply means "overview only". Fetched
  // alongside so the map can switch to viewport tiles past the zoom threshold.
  let tiles: RadarData['tiles'] = null
  try {
    const t = await getJson<HistoryTilesDto>('/v1/history/tiles', signal)
    tiles = {
      tilePx: t.tile_px,
      nx: t.nx,
      ny: t.ny,
      gridH: t.grid_h,
      gridW: t.grid_w,
      cols: t.cols,
      count: t.count,
      mtime: t.mtime,
      bounds: t.bounds,
      index: t.index,
      urlFor: (tx: number, ty: number) =>
        abs(`/v1/history/tile/${tx}/${ty}/sprite.png?v=${t.mtime}`),
    }
  } catch {
    tiles = null
  }
  return {
    issuedAt: observedAt,
    location: { lat, lon },
    modelVersion: 'observed-qpe',
    bounds: dto.bounds ?? DEFAULT_BOUNDS,
    frames,
    sprite,
    tiles,
  }
}


// ---- Forecast verification (/v1/verify/*): replay archived runs against the
// observed composite. ----

export interface VerifyIssue {
  issue: number // unix epoch of the run's issue time
  issuedAt: Date
}

export interface VerifyMeta {
  issue: number
  leads: number[] // minutes
  bounds: Bounds
}

export interface VerifyScores {
  biasMmH: number
  maeMmH: number
  csi01: number
  csi05: number
  csi10: number
}

export async function getVerifyIssues(signal?: AbortSignal): Promise<VerifyIssue[]> {
  const rows = await getJson<{ issue: number; issued_at: string }[]>('/v1/verify/issues', signal)
  return rows.map((r) => ({ issue: r.issue, issuedAt: new Date(r.issued_at) }))
}

export async function getVerifyIssueMeta(issue: number, signal?: AbortSignal): Promise<VerifyMeta> {
  const m = await getJson<{ issue: number; leads: number[]; bounds: number[] }>(
    `/v1/verify/issue?issue=${issue}`,
    signal,
  )
  return {
    issue: m.issue,
    leads: m.leads,
    bounds: { west: m.bounds[0], south: m.bounds[1], east: m.bounds[2], north: m.bounds[3] },
  }
}

// Frame PNGs are plain URLs — the map's overlay loader fetches and caches them.
export const verifyFrameUrl = (issue: number, lead: number, kind: string) =>
  `${API_BASE}/v1/verify/frame.png?issue=${issue}&lead=${lead}&kind=${kind}`

export async function getVerifyScores(
  issue: number,
  lead: number,
  signal?: AbortSignal,
): Promise<VerifyScores> {
  const s = await getJson<Record<string, number>>(
    `/v1/verify/scores?issue=${issue}&lead=${lead}`,
    signal,
  )
  return {
    biasMmH: s.bias_mm_h,
    maeMmH: s.mae_mm_h,
    csi01: s['csi_0.1'],
    csi05: s['csi_0.5'],
    csi10: s['csi_1.0'],
  }
}
