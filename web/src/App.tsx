import { useCallback, useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { getHistory, getRadar, minutesUntilRain, type RadarData, type RadarFrame } from './api'
import { subscribeUpdates } from './updates'
import { useGeolocation } from './location'
import { HORIZON_MIN, frameFull, timeOfDay } from './format'
import { LANGS } from './i18n'
import RadarMap from './map/RadarMap'
import TimelineSlider from './components/TimelineSlider'
import ForecastChart from './components/ForecastChart'
import PrecipitationLegend from './components/PrecipitationLegend'
import SourceBadge from './components/SourceBadge'
import VerifyView from './components/VerifyView'

// One animation tick. Observed history serves ~109 motion-interpolated frames
// per 3 h (100 s cadence) — at 400 ms a loop would take 44 s, so observed
// frames play 150 ms ticks for fluid motion; forecast frames (5–10 min steps)
// keep the slower 400 ms so each lead stays readable. The seamless timeline
// mixes both cadences frame-by-frame.
const PLAY_TICK_MS = 400
const PLAY_TICK_HISTORY_MS = 150

// Timeline (default) crosses t=0: the last 3 h of measured composite flow
// straight into the forecast. Verify replays archived forecast runs against
// what actually fell.
type Mode = 'timeline' | 'forecast' | 'history' | 'verify'
const TABS: Exclude<Mode, 'verify'>[] = ['timeline', 'forecast', 'history']

type Load =
  | { state: 'loading' }
  | { state: 'error' }
  | {
      state: 'ok'
      data: RadarData
      // Present in timeline mode: the two sources behind the merged frames,
      // so the map can switch sprite/tiles/bounds when the scrubber crosses now.
      pair?: { hist: RadarData; fc: RadarData }
      nowIndex?: number
    }

// Merge observed history and the forecast into one continuous frame list.
// Leads are re-expressed relative to the forecast issue time so the scrubber
// runs -180 min → now → +horizon without a seam.
function mergeTimeline(hist: RadarData, fc: RadarData): { data: RadarData; nowIndex: number } {
  const now = fc.issuedAt.getTime()
  const obs: RadarFrame[] = hist.frames.map((f) => ({
    ...f,
    kind: 'obs' as const,
    leadMin: Math.round((f.validTime.getTime() - now) / 60000),
  }))
  const newestObs = obs.length > 0 ? obs[obs.length - 1].validTime.getTime() : now
  const fcs: RadarFrame[] = fc.frames
    .filter((f) => f.validTime.getTime() > newestObs)
    .map((f) => ({ ...f, kind: 'fc' as const }))
  return {
    data: { ...hist, issuedAt: fc.issuedAt, frames: [...obs, ...fcs] },
    nowIndex: Math.max(0, obs.length - 1),
  }
}

export default function App() {
  const { t, i18n } = useTranslation()
  const { center: geoCenter, status, locate } = useGeolocation()

  // The location the forecast is for. Follows geolocation until the user picks
  // a spot on the map (tap or drag), after which it stays put.
  const [location, setLocation] = useState(geoCenter)
  const [recenter, setRecenter] = useState(0) // bump to fly the map to `location`
  const userPicked = useRef(false)

  const [mode, setMode] = useState<Mode>('timeline')
  const [load, setLoad] = useState<Load>({ state: 'loading' })
  const [index, setIndex] = useState(0)
  const [isPlaying, setPlaying] = useState(false)
  const [nonce, setNonce] = useState(0) // bump to force a refetch
  const lastSnapshot = useRef<string | null>(null)

  // Live updates: when the server publishes a new prediction, refetch (deduped
  // so the connect-time "current snapshot" message doesn't double-fetch).
  useEffect(() => {
    return subscribeUpdates((snapshot) => {
      if (snapshot === lastSnapshot.current) return
      lastSnapshot.current = snapshot
      setNonce((n) => n + 1)
    })
  }, [])

  // Track geolocation until the user manually picks a location.
  useEffect(() => {
    if (userPicked.current) return
    setLocation(geoCenter)
    setRecenter((n) => n + 1)
  }, [geoCenter.lat, geoCenter.lon])

  // Pick a location by tapping/dragging on the map (don't recenter).
  const onPick = useCallback((lat: number, lon: number) => {
    userPicked.current = true
    setLocation({ lat, lon })
  }, [])

  // "Locate me": drop any manual pick and snap back to GPS (recenters).
  const onLocate = useCallback(() => {
    userPicked.current = false
    locate()
  }, [locate])

  // Fetch the radar whenever the location (or refresh nonce) changes. Timeline
  // mode needs both sources; verify mode fetches its own data.
  useEffect(() => {
    if (mode === 'verify') return
    const ctrl = new AbortController()
    setLoad({ state: 'loading' })
    const fail = (err: unknown) => {
      if (ctrl.signal.aborted) return
      console.error(err)
      setLoad({ state: 'error' })
    }
    if (mode === 'timeline') {
      Promise.all([
        getHistory(location.lat, location.lon, 180, ctrl.signal),
        getRadar(location.lat, location.lon, HORIZON_MIN, ctrl.signal),
      ])
        .then(([hist, fc]) => {
          const { data, nowIndex } = mergeTimeline(hist, fc)
          setLoad({ state: 'ok', data, pair: { hist, fc }, nowIndex })
          setIndex(nowIndex) // open at "now", the seam between measured and forecast
          setPlaying(false)
        })
        .catch(fail)
    } else {
      const fetcher =
        mode === 'history'
          ? getHistory(location.lat, location.lon, 180, ctrl.signal)
          : getRadar(location.lat, location.lon, HORIZON_MIN, ctrl.signal)
      fetcher
        .then((data) => {
          setLoad({ state: 'ok', data })
          // History opens on the newest observation; forecast on "now".
          setIndex(mode === 'history' ? Math.max(0, data.frames.length - 1) : 0)
          setPlaying(false)
        })
        .catch(fail)
    }
    return () => ctrl.abort()
  }, [location.lat, location.lon, nonce, mode])

  const ok = load.state === 'ok' ? load : null
  const frames = ok ? ok.data.frames : []
  const cur = frames[index] ?? null

  // Playback loop: one timeout per frame so the seamless timeline can honour
  // each frame's native cadence (observed fast, forecast slow).
  useEffect(() => {
    if (!isPlaying || frames.length < 2) return
    const obsFrame = mode === 'history' || (mode === 'timeline' && cur?.kind !== 'fc')
    const tick = obsFrame ? PLAY_TICK_HISTORY_MS : PLAY_TICK_MS
    const id = window.setTimeout(() => setIndex((i) => (i + 1) % frames.length), tick)
    return () => clearTimeout(id)
  }, [isPlaying, index, frames, mode, cur])

  const onIndex = useCallback((i: number) => {
    setIndex(i)
    setPlaying(false) // manual scrub pauses playback
  }, [])

  // Which source drives the map for the current frame. In timeline mode the
  // overlay switches sprite/tiles/bounds when the scrubber crosses t=0, while
  // the camera stays locked to the wide observed domain.
  const isObs = cur?.kind !== 'fc'
  const pair = ok?.pair
  const mapSprite = pair ? (isObs ? pair.hist.sprite : pair.fc.sprite) : (ok?.data.sprite ?? null)
  const mapTiles = pair ? (isObs ? (pair.hist.tiles ?? null) : null) : (ok?.data.tiles ?? null)
  const mapBounds = pair
    ? isObs
      ? pair.hist.bounds
      : pair.fc.bounds
    : ok
      ? ok.data.bounds
      : { west: 1.5, east: 7.5, south: 48.9, north: 52.5 }
  const mapDomain = pair ? pair.hist.bounds : undefined

  const headline = (() => {
    if (!ok) return ''
    if (mode === 'history') return t('history.headline')
    if (mode === 'timeline') {
      const m = minutesUntilRain(pair ? pair.fc : ok.data)
      if (m === null) return t('nowcast.dry')
      if (m === 0) return t('nowcast.raining')
      return t('nowcast.rainInMinutes', { minutes: m })
    }
    const m = minutesUntilRain(ok.data)
    if (m === null) return t('nowcast.dry')
    if (m === 0) return t('nowcast.raining')
    return t('nowcast.rainInMinutes', { minutes: m })
  })()

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <img src="/icon.svg" alt="" width={28} height={28} />
          <div>
            <strong>{t('appTitle')}</strong>
            <span className="tagline">{t('tagline')}</span>
          </div>
        </div>
        <div className="actions">
          <button
            className="icon-btn"
            onClick={onLocate}
            title={t('locate')}
            aria-label={t('locate')}
          >
            {status === 'locating' ? '⌖…' : '⌖'}
          </button>
          <button
            className="icon-btn"
            onClick={() => setNonce((n) => n + 1)}
            title={t('refresh')}
            aria-label={t('refresh')}
          >
            ↻
          </button>
          <select
            className="lang"
            value={i18n.resolvedLanguage}
            onChange={(e) => i18n.changeLanguage(e.target.value)}
            aria-label={t('language')}
          >
            {LANGS.map((l) => (
              <option key={l.code} value={l.code}>
                {l.label}
              </option>
            ))}
          </select>
        </div>
      </header>

      {status === 'denied' && <div className="note">{t('locationDenied')}</div>}

      <div className="mode-toggle" role="tablist" aria-label={t('mode.label')}>
        {TABS.map((m) => (
          <button
            key={m}
            role="tab"
            aria-selected={mode === m}
            className={mode === m ? 'mode active' : 'mode'}
            onClick={() => setMode(m)}
          >
            {t(`mode.${m}`)}
          </button>
        ))}
        <button
          role="tab"
          aria-selected={mode === 'verify'}
          className={mode === 'verify' ? 'mode active' : 'mode'}
          onClick={() => setMode('verify')}
        >
          {t('mode.verify')}
        </button>
      </div>

      {mode === 'verify' ? (
        <VerifyView />
      ) : (
        <div className="content">
          <div className="map-wrap">
            <RadarMap
              center={location}
              bounds={mapBounds}
              domain={mapDomain}
              frame={cur}
              sprite={mapSprite}
              tiles={mapTiles}
              onPick={onPick}
              recenter={recenter}
            />
            <div className="map-hint">{t('tapHint')}</div>
            {load.state === 'loading' && <div className="overlay-msg">{t('loading')}</div>}
            {load.state === 'error' && <div className="overlay-msg">{t('radarError')}</div>}
          </div>

          {ok && frames.length > 0 && (
            <TimelineSlider
              frames={frames}
              index={index}
              isPlaying={isPlaying}
              onIndex={onIndex}
              onPlayPause={() => setPlaying((p) => !p)}
              issuedAt={ok.data.issuedAt}
              nowIndex={mode === 'timeline' ? (ok.nowIndex ?? null) : null}
            />
          )}

          {ok && (
            <section className="panel">
              <h1 className="headline">{headline}</h1>
              <p className="updated">
                {mode === 'history'
                  ? t('history.observedAt', { time: timeOfDay(ok.data.issuedAt) })
                  : t('updated', { time: timeOfDay(ok.data.issuedAt) })}
              </p>
              {cur && (
                <div className="chart-readout">
                  <span className="rstamp">{frameFull(cur.validTime)}</span>
                  <span className="rrate">
                    {t('rate', { value: cur.rateMmPerH.toFixed(2) })}
                  </span>
                </div>
              )}
              {cur && <SourceBadge source={cur.source} confidence={cur.confidence} />}
              {frames.length > 0 && (
                <ForecastChart
                  frames={frames}
                  index={index}
                  issuedAt={ok.data.issuedAt}
                  onSelect={onIndex}
                  title={mode === 'history' ? t('history.chartTitle') : undefined}
                />
              )}
              <PrecipitationLegend />
            </section>
          )}
        </div>
      )}
    </div>
  )
}
