import { useCallback, useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { getHistory, getRadar, minutesUntilRain, type RadarData } from './api'
import { subscribeUpdates } from './updates'
import { useGeolocation } from './location'
import { HORIZON_MIN, frameFull, timeOfDay } from './format'
import { LANGS } from './i18n'
import RadarMap from './map/RadarMap'
import TimelineSlider from './components/TimelineSlider'
import ForecastChart from './components/ForecastChart'
import PrecipitationLegend from './components/PrecipitationLegend'
import SourceBadge from './components/SourceBadge'

// One animation tick. Forecast: ~13 nowcast frames at 400ms span ~5s. History now
// serves ~109 motion-interpolated frames per 3h (100s cadence) — at 400ms a loop
// would take 44s, so history plays 150ms ticks for fluid motion (~16s per loop).
const PLAY_TICK_MS = 400
const PLAY_TICK_HISTORY_MS = 150

type Load =
  | { state: 'loading' }
  | { state: 'error' }
  | { state: 'ok'; data: RadarData }

export default function App() {
  const { t, i18n } = useTranslation()
  const { center: geoCenter, status, locate } = useGeolocation()

  // The location the forecast is for. Follows geolocation until the user picks
  // a spot on the map (tap or drag), after which it stays put.
  const [location, setLocation] = useState(geoCenter)
  const [recenter, setRecenter] = useState(0) // bump to fly the map to `location`
  const userPicked = useRef(false)

  // Forecast (default) or observed radar history — two views over one pipeline.
  const [mode, setMode] = useState<'forecast' | 'history'>('forecast')
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

  // Fetch the radar whenever the location (or refresh nonce) changes.
  useEffect(() => {
    const ctrl = new AbortController()
    setLoad({ state: 'loading' })
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
      .catch((err) => {
        if (ctrl.signal.aborted) return
        console.error(err)
        setLoad({ state: 'error' })
      })
    return () => ctrl.abort()
  }, [location.lat, location.lon, nonce, mode])

  const frames = load.state === 'ok' ? load.data.frames : []

  // Playback loop.
  useEffect(() => {
    if (!isPlaying || frames.length < 2) return
    const tick = mode === 'history' ? PLAY_TICK_HISTORY_MS : PLAY_TICK_MS
    const id = setInterval(() => setIndex((i) => (i + 1) % frames.length), tick)
    return () => clearInterval(id)
  }, [isPlaying, frames.length, mode])

  const onIndex = useCallback((i: number) => {
    setIndex(i)
    setPlaying(false) // manual scrub pauses playback
  }, [])

  const headline = (() => {
    if (load.state !== 'ok') return ''
    if (mode === 'history') return t('history.headline')
    const m = minutesUntilRain(load.data)
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
        <button
          role="tab"
          aria-selected={mode === 'forecast'}
          className={mode === 'forecast' ? 'mode active' : 'mode'}
          onClick={() => setMode('forecast')}
        >
          {t('mode.forecast')}
        </button>
        <button
          role="tab"
          aria-selected={mode === 'history'}
          className={mode === 'history' ? 'mode active' : 'mode'}
          onClick={() => setMode('history')}
        >
          {t('mode.history')}
        </button>
      </div>

      <div className="content">
        <div className="map-wrap">
          <RadarMap
            center={location}
            bounds={load.state === 'ok' ? load.data.bounds : { west: 1.5, east: 7.5, south: 48.9, north: 52.5 }}
            frame={load.state === 'ok' ? frames[index] ?? null : null}
            sprite={load.state === 'ok' ? load.data.sprite : null}
            onPick={onPick}
            recenter={recenter}
          />
          <div className="map-hint">{t('tapHint')}</div>
          {load.state === 'loading' && <div className="overlay-msg">{t('loading')}</div>}
          {load.state === 'error' && <div className="overlay-msg">{t('radarError')}</div>}
        </div>

        {load.state === 'ok' && frames.length > 0 && (
          <TimelineSlider
            frames={frames}
            index={index}
            isPlaying={isPlaying}
            onIndex={onIndex}
            onPlayPause={() => setPlaying((p) => !p)}
            issuedAt={load.data.issuedAt}
          />
        )}

        {load.state === 'ok' && (
          <section className="panel">
            <h1 className="headline">{headline}</h1>
            <p className="updated">
              {mode === 'history'
                ? t('history.observedAt', { time: timeOfDay(load.data.issuedAt) })
                : t('updated', { time: timeOfDay(load.data.issuedAt) })}
            </p>
            {frames.length > 0 && frames[index] && (
              <div className="chart-readout">
                <span className="rstamp">{frameFull(frames[index].validTime)}</span>
                <span className="rrate">
                  {t('rate', { value: frames[index].rateMmPerH.toFixed(2) })}
                </span>
              </div>
            )}
            {frames.length > 0 && frames[index] && (
              <SourceBadge
                source={frames[index].source}
                confidence={frames[index].confidence}
              />
            )}
            {frames.length > 0 && (
              <ForecastChart
                frames={frames}
                index={index}
                issuedAt={load.data.issuedAt}
                onSelect={onIndex}
                title={mode === 'history' ? t('history.chartTitle') : undefined}
              />
            )}
            <PrecipitationLegend />
          </section>
        )}
      </div>
    </div>
  )
}
