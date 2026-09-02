// Precipitation banding + the shared colour ramp. Mirrors the Flutter app's
// PrecipitationLevel / PrecipitationPalette so the web and native clients read
// the same. Thresholds follow the WMO 1985 classification (mm/h).

export type PrecipLevel = 'none' | 'light' | 'moderate' | 'heavy' | 'violent'

// Below this rate it's a trace — not perceptible rain, and within model noise.
// Treating it as "none" stops the headline from crying "rain expected" (and the
// chart from colouring bars) for a forecast that's effectively dry. 0.1 mm/h is
// the conventional "measurable precipitation" threshold. Tune here if needed.
export const RAIN_THRESHOLD_MM_H = 0.1

export function levelFromMmPerHour(mmPerHour: number): PrecipLevel {
  if (mmPerHour < RAIN_THRESHOLD_MM_H) return 'none'
  if (mmPerHour < 2.5) return 'light'
  if (mmPerHour < 7.5) return 'moderate'
  if (mmPerHour < 50) return 'heavy'
  return 'violent'
}

// Map colour scale — follows the Met Office rainfall key (deep blue < 0.5
// through dark red > 32 mm/h). Must match BANDS in the backend's colormap.py
// exactly: the overlay/sprite pixels are painted server-side.
export const MAP_RAMP: { min: number; color: string; label: string }[] = [
  // Drizzle band: reference maps show trace rain below 0.1 mm/h; without it
  // our forecast looked like it was missing light rain vs e.g. Buienradar.
  { min: 0.05, color: '#0c1078', label: '<0.1' },
  { min: 0.1, color: '#1219c8', label: '0.1\u20130.5' },
  { min: 0.5, color: '#3c6ee6', label: '0.5\u20131' },
  { min: 1, color: '#69c8f0', label: '1\u20132' },
  { min: 2, color: '#3cb43c', label: '2\u20134' },
  { min: 4, color: '#f0d746', label: '4\u20138' },
  { min: 8, color: '#f0a03c', label: '8\u201316' },
  { min: 16, color: '#e63c37', label: '16\u201332' },
  { min: 32, color: '#c82323', label: '>32' },
]

export function mapColorForRate(mmPerHour: number): string {
  let c = 'transparent'
  for (const b of MAP_RAMP) if (mmPerHour >= b.min) c = b.color
  return c
}

// Semantic level colours (headline, chart bars) — nearest analogue from the
// map ramp so the chart and the overlay tell one colour story.
export const PRECIP_COLOR: Record<PrecipLevel, string> = {
  none: '#2a2a2e',
  light: '#69c8f0',
  moderate: '#3cb43c',
  heavy: '#f0a03c',
  violent: '#e63c37',
}

// Levels that carry rain, in legend order.
export const RAIN_LEVELS: PrecipLevel[] = ['light', 'moderate', 'heavy', 'violent']
