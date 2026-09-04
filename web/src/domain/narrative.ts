// Per-location narrative in the Buienradar style (TODO 5.2): one or two short
// sentences that say when rain starts and stops at the user's point, with the
// confidence of the band that produced it stated honestly.
//
// Pure: takes the forecast frames and returns i18n parts (key + params); the
// component joins them through `t`, so language follows the app's selection.
// Rules (mirroring how Buienradar phrases it):
//   * only the next `horizonMin` minutes of the forecast are narrated in full
//     (the radar nowcast band, 2 h);
//   * "rain" means level != 'none' (>= 0.1 mm/h, same threshold as the chart);
//   * gaps shorter than `minGapMin` inside an episode are ignored (a cell
//     passing a point shouldn't read as three showers);
//   * the first episode that starts beyond the nowcast band, if any, gets one
//     hedged sentence ("possible", named source), never a hard time.

import type { PrecipLevel } from './precip'

export interface NarrativeFrame {
  leadMin: number
  validTime: Date
  rateMmPerH: number
  level: PrecipLevel
  source: string | null
  kind?: 'obs' | 'fc'
}

export interface NarrativePart {
  key: string
  params?: Record<string, string | number>
}

export interface Episode {
  start: Date
  end: Date | null // null = still raining at the end of the horizon
  peak: PrecipLevel
  source: string | null // source of the frame where it starts
}

const LEVEL_RANK: Record<PrecipLevel, number> = { none: 0, light: 1, moderate: 2, heavy: 3, violent: 4 }

export function episodes(frames: NarrativeFrame[], horizonMin: number, minGapMin = 10): Episode[] {
  const fc = frames
    .filter((f) => f.kind !== 'obs' && f.leadMin >= 0 && f.leadMin <= horizonMin)
    .sort((a, b) => a.leadMin - b.leadMin)
  const out: Episode[] = []
  let cur: Episode | null = null
  let lastWet: NarrativeFrame | null = null
  for (const f of fc) {
    const wet = f.level !== 'none'
    if (wet) {
      if (cur && lastWet && f.leadMin - lastWet.leadMin > minGapMin) {
        cur.end = new Date(lastWet.validTime.getTime() + 60_000 * stepAfter(fc, lastWet))
        out.push(cur)
        cur = null
      }
      if (!cur) cur = { start: f.validTime, end: null, peak: f.level, source: f.source }
      if (LEVEL_RANK[f.level] > LEVEL_RANK[cur.peak]) cur.peak = f.level
      lastWet = f
    }
  }
  if (cur) {
    const last = fc[fc.length - 1]
    // ended before the horizon if the last wet frame is not the last frame
    if (lastWet && last && lastWet.leadMin < last.leadMin) {
      cur.end = new Date(lastWet.validTime.getTime() + 60_000 * stepAfter(fc, lastWet))
    }
    out.push(cur)
  }
  return out
}

function stepAfter(fc: NarrativeFrame[], f: NarrativeFrame): number {
  const i = fc.indexOf(f)
  const next = fc[i + 1]
  return next ? next.leadMin - f.leadMin : 0
}

export function formatClock(d: Date, lang: string): string {
  return new Intl.DateTimeFormat(lang, { hour: '2-digit', minute: '2-digit', hour12: false }).format(d)
}

export function narrativeParts(
  frames: NarrativeFrame[],
  issuedAt: Date,
  lang: string,
  opts: { horizonMin?: number; outlookMin?: number } = {},
): NarrativePart[] {
  const horizonMin = opts.horizonMin ?? 120
  const outlookMin = opts.outlookMin ?? 360
  const hours = Math.round(horizonMin / 60)
  const near = episodes(frames, horizonMin)
  const parts: NarrativePart[] = []
  const clock = (d: Date) => formatClock(d, lang)
  const rainingNow = near.length > 0 && near[0].start.getTime() <= issuedAt.getTime() + 5 * 60_000

  if (near.length === 0) {
    parts.push({ key: 'narrative.dryHorizon', params: { hours } })
  } else if (rainingNow) {
    const first = near[0]
    if (first.end === null) {
      parts.push({ key: 'narrative.rainingOn', params: { hours } })
    } else {
      parts.push({ key: 'narrative.rainingUntil', params: { time: clock(first.end) } })
      const next = near[1]
      if (next) {
        parts.push({
          key: 'narrative.again',
          params: { time: clock(next.start), intensity: `$t(narrative.intensity.${next.peak})` },
        })
      }
    }
  } else {
    const first = near[0]
    if (first.end === null) {
      parts.push({
        key: 'narrative.dryUntilOpenEnd',
        params: { start: clock(first.start), intensity: `$t(narrative.intensity.${first.peak})`, hours },
      })
    } else {
      parts.push({
        key: 'narrative.dryUntil',
        params: {
          start: clock(first.start),
          end: clock(first.end),
          intensity: `$t(narrative.intensity.${first.peak})`,
        },
      })
    }
  }

  // Beyond the radar band: one hedged sentence for the first rain episode in
  // the outlook window, named by its source, only if the near horizon has
  // nothing (or ends dry).
  const nearEndsWet = near.length > 0 && near[near.length - 1].end === null
  if (!nearEndsWet) {
    const later = episodes(frames, outlookMin).find(
      (e) => e.start.getTime() > issuedAt.getTime() + horizonMin * 60_000,
    )
    if (later) {
      const src = later.source === 'nwp' ? 'nwp' : later.source === 'blend' ? 'blend' : 'model'
      parts.push({
        key: 'narrative.possibleLater',
        params: { time: clock(later.start), intensity: `$t(narrative.intensity.${later.peak})`, source: `$t(narrative.source.${src})` },
      })
    }
  }
  return parts
}
