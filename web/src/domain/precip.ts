// Precipitation banding + the shared colour ramp. Mirrors the Flutter app's
// PrecipitationLevel / PrecipitationPalette so the web and native clients read
// the same. Thresholds follow the WMO 1985 classification (mm/h).

export type PrecipLevel = 'none' | 'light' | 'moderate' | 'heavy' | 'violent'

export function levelFromMmPerHour(mmPerHour: number): PrecipLevel {
  if (mmPerHour <= 0) return 'none'
  if (mmPerHour < 2.5) return 'light'
  if (mmPerHour < 7.5) return 'moderate'
  if (mmPerHour < 50) return 'heavy'
  return 'violent'
}

// Colour ramp used for both the legend and the nowcast bar chart. One source
// of truth avoids drift between the two views of the same data.
export const PRECIP_COLOR: Record<PrecipLevel, string> = {
  none: '#2a2a2e',
  light: '#9ecae1',
  moderate: '#3182bd',
  heavy: '#fd8d3c',
  violent: '#e31a1c',
}

// Levels that carry rain, in legend order.
export const RAIN_LEVELS: PrecipLevel[] = ['light', 'moderate', 'heavy', 'violent']
