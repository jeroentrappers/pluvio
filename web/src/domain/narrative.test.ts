import { describe, expect, it } from 'vitest'
import { episodes, formatClock, narrativeParts, type NarrativeFrame } from './narrative'
import { levelFromMmPerHour } from './precip'

const T0 = new Date('2026-09-04T14:00:00Z')
function frames(rates: number[], stepMin = 10, source = 'nowcast'): NarrativeFrame[] {
  return rates.map((r, i) => ({
    leadMin: i * stepMin,
    validTime: new Date(T0.getTime() + i * stepMin * 60_000),
    rateMmPerH: r,
    level: levelFromMmPerHour(r),
    source,
  }))
}

describe('episodes', () => {
  it('finds start, end and peak of a shower and ignores short gaps', () => {
    const f = frames([0, 0, 0.5, 3, 0, 0.4, 0, 0, 0, 0, 0, 0, 0])
    const e = episodes(f, 120)
    expect(e).toHaveLength(1)
    expect(e[0].start.toISOString()).toBe('2026-09-04T14:20:00.000Z')
    expect(e[0].end?.toISOString()).toBe('2026-09-04T15:00:00.000Z')
    expect(e[0].peak).toBe('moderate')
  })
  it('splits episodes separated by more than the gap', () => {
    const f = frames([0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0])
    expect(episodes(f, 120)).toHaveLength(2)
  })
  it('leaves end null when still raining at the horizon', () => {
    const f = frames([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3])
    const e = episodes(f, 120)
    expect(e[0].end).toBeNull()
  })
})

describe('narrativeParts (Buienradar style)', () => {
  it('dry horizon', () => {
    const p = narrativeParts(frames(new Array(13).fill(0)), T0, 'nl')
    expect(p).toEqual([{ key: 'narrative.dryHorizon', params: { hours: 2 } }])
  })
  it('raining now, dry again at a time', () => {
    const p = narrativeParts(frames([2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), T0, 'nl')
    expect(p[0].key).toBe('narrative.rainingUntil')
    expect(p[0].params?.time).toBe(formatClock(new Date('2026-09-04T14:30:00Z'), 'nl'))
  })
  it('dry until a time, then rain until a time, with intensity words', () => {
    const p = narrativeParts(frames([0, 0, 0, 0, 5, 8, 0, 0, 0, 0, 0, 0, 0]), T0, 'nl')
    expect(p[0]).toEqual({
      key: 'narrative.dryUntil',
      params: {
        start: formatClock(new Date('2026-09-04T14:40:00Z'), 'nl'),
        end: formatClock(new Date('2026-09-04T15:00:00Z'), 'nl'),
        intensity: '$t(narrative.intensity.heavy)',
      },
    })
  })
  it('hedges an episode that only the weather model predicts', () => {
    const near = frames(new Array(13).fill(0))
    const later: NarrativeFrame[] = [180, 240].map((lead) => ({
      leadMin: lead,
      validTime: new Date(T0.getTime() + lead * 60_000),
      rateMmPerH: 1.5,
      level: 'light',
      source: 'nwp',
    }))
    const p = narrativeParts([...near, ...later], T0, 'nl')
    expect(p.map((x) => x.key)).toEqual(['narrative.dryHorizon', 'narrative.possibleLater'])
    expect(p[1].params?.source).toBe('$t(narrative.source.nwp)')
    expect(p[1].params?.time).toBe(formatClock(new Date('2026-09-04T17:00:00Z'), 'nl'))
  })
  it('ignores observed frames of the seamless timeline', () => {
    const obs = frames([5, 5, 5]).map((f) => ({ ...f, kind: 'obs' as const, leadMin: f.leadMin - 30 }))
    const p = narrativeParts([...obs, ...frames(new Array(13).fill(0))], T0, 'en')
    expect(p[0].key).toBe('narrative.dryHorizon')
  })
})
