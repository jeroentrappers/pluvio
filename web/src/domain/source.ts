// Forecast provenance helpers: turn the backend's per-lead `source` tag and
// `confidence` (0–1) into something the UI can label honestly. The source tells
// the user *where* a lead's number came from (radar extrapolation vs a weather
// model), and the confidence widens with lead — the "never pretend the day-5
// number came from the radar" rule, surfaced.

// i18n key under `source.*` for a backend source code. Unknown/absent → null
// (badge renders nothing rather than a misleading label).
export function sourceLabelKey(source: string | null | undefined): string | null {
  switch (source) {
    case 'nowcast':
      return 'source.nowcast'
    case 'blend':
      return 'source.blend'
    case 'nwp':
      return 'source.nwp'
    case 'radar':
      return 'source.radar'
    default:
      return null
  }
}

export type ConfidenceTier = 'high' | 'medium' | 'low'

// Skill-based confidence tier, used for colour + an i18n word. Anchors match the
// product's honesty knob (research/model/classical.py confidence anchors).
export function confidenceTier(confidence: number): ConfidenceTier {
  if (confidence >= 0.7) return 'high'
  if (confidence >= 0.45) return 'medium'
  return 'low'
}

export const confidencePct = (confidence: number) => Math.round(confidence * 100)
