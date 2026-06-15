import { useCallback, useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { getRadar, minutesUntilRain, type RadarData } from './api'
import { useGeolocation } from './location'
import { LANGS } from './i18n'
import RadarMap from './map/RadarMap'
import TimelineSlider from './components/TimelineSlider'
import ForecastChart from './components/ForecastChart'
import PrecipitationLegend from './components/PrecipitationLegend'

// One animation tick. ~13 nowcast frames span ~5s — fast enough to feel like
// motion, slow enough that each step reads. Matches the Flutter app.
const PLAY_TICK_MS = 400

type Load =
  | { state: 'loading' }
  | { state: 'error' }
  | { state: 'ok'; data: RadarData }

export default function App() {
  const { t, i18n } = useTranslation()
  const { center, status, locate } = useGeolocation()

  const [load, setLoad] = useState<Load>({ state: 'loading' })
  const [index, setIndex] = useState(0)
  const [isPlaying, setPlaying] = useState(false)
  const [nonce, setNonce] = useState(0) // bump to force a refetch

  // Fetch the radar whenever the location (or refresh nonce) changes.
  useEffect(() => {
    const ctrl = new AbortController()
    setLoad({ state: 'loading' })
    getRadar(center.lat, center.lon, 120, ctrl.signal)
      .then((data) => {
        setLoad({ state: 'ok', data })
        setIndex(0)
        setPlaying(false)
      })
      .catch((err) => {
        if (ctrl.signal.aborted) return
        console.error(err)
        setLoad({ state: 'error' })
      })
    return () => ctrl.abort()
  }, [center.lat, center.lon, nonce])

  const frames = load.state === 'ok' ? load.data.frames : []

  // Playback loop.
  useEffect(() => {
    if (!isPlaying || frames.length < 2) return
    const id = setInterval(() => setIndex((i) => (i + 1) % frames.length), PLAY_TICK_MS)
    return () => clearInterval(id)
  }, [isPlaying, frames.length])

  const onIndex = useCallback((i: number) => {
    setIndex(i)
    setPlaying(false) // manual scrub pauses playback
  }, [])

  const headline = (() => {
    if (load.state !== 'ok') return ''
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
            onClick={locate}
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

      <div className="content">
        <div className="map-wrap">
          <RadarMap
            center={center}
            bounds={load.state === 'ok' ? load.data.bounds : { west: 1.5, east: 7.5, south: 48.9, north: 52.5 }}
            frame={load.state === 'ok' ? frames[index] ?? null : null}
            frames={frames}
          />
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
            {frames.length > 0 && (
              <ForecastChart frames={frames} index={index} issuedAt={load.data.issuedAt} />
            )}
            <PrecipitationLegend />
          </section>
        )}
      </div>
    </div>
  )
}
