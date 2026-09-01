import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import {
  getVerifyIssueMeta,
  getVerifyIssues,
  getVerifyScores,
  verifyFrameUrl,
  type VerifyIssue,
  type VerifyMeta,
  type VerifyScores,
} from '../api'
import { frameFull, leadLabel } from '../format'
import RadarMap from '../map/RadarMap'

// Replay an archived forecast run against the observed composite: pick a past
// issue, scrub through its leads, and flip between the forecast, what actually
// fell, and their signed difference (red = over-forecast, blue = under).
const KINDS = ['forecast', 'observed', 'diff'] as const
type Kind = (typeof KINDS)[number]

const BENELUX = { west: 1.5, east: 7.5, south: 48.9, north: 52.5 }

export default function VerifyView() {
  const { t } = useTranslation()
  const [issues, setIssues] = useState<VerifyIssue[] | null>(null)
  const [issue, setIssue] = useState<number | null>(null)
  const [meta, setMeta] = useState<VerifyMeta | null>(null)
  const [li, setLi] = useState(0)
  const [kind, setKind] = useState<Kind>('diff')
  const [playing, setPlaying] = useState(false)
  const [scores, setScores] = useState<VerifyScores | 'missing' | null>(null)

  useEffect(() => {
    const ctrl = new AbortController()
    getVerifyIssues(ctrl.signal)
      .then((list) => {
        setIssues(list)
        // Default to the oldest archived run: it has the most observed frames
        // to verify against (the newest run's valid times haven't fallen yet).
        if (list.length > 0) setIssue(list[list.length - 1].issue)
      })
      .catch(() => setIssues([]))
    return () => ctrl.abort()
  }, [])

  useEffect(() => {
    if (issue == null) return
    const ctrl = new AbortController()
    setMeta(null)
    getVerifyIssueMeta(issue, ctrl.signal)
      .then((m) => {
        setMeta(m)
        setLi(0)
      })
      .catch(() => setMeta(null))
    return () => ctrl.abort()
  }, [issue])

  const lead = meta?.leads[li] ?? null

  useEffect(() => {
    if (issue == null || lead == null) return
    const ctrl = new AbortController()
    setScores(null)
    getVerifyScores(issue, lead, ctrl.signal)
      .then(setScores)
      .catch((err) => {
        if (!ctrl.signal.aborted) setScores('missing')
        void err
      })
    return () => ctrl.abort()
  }, [issue, lead])

  useEffect(() => {
    if (!playing || !meta || meta.leads.length < 2) return
    const id = window.setTimeout(() => setLi((i) => (i + 1) % meta.leads.length), 500)
    return () => clearTimeout(id)
  }, [playing, li, meta])

  if (issues !== null && issues.length === 0) {
    return <div className="note">{t('verify.empty')}</div>
  }

  const overlay =
    issue != null && lead != null && meta
      ? { url: verifyFrameUrl(issue, lead, kind), bounds: meta.bounds }
      : null

  return (
    <div className="content">
      <div className="map-wrap">
        <RadarMap
          center={{ lat: 50.8, lon: 4.5 }}
          bounds={meta?.bounds ?? BENELUX}
          frame={null}
          sprite={null}
          overlay={overlay}
        />
        {kind === 'diff' && <div className="map-hint">{t('verify.diffNote')}</div>}
      </div>

      {meta && (
        <div className="timeline">
          <button
            className="play"
            onClick={() => setPlaying((p) => !p)}
            aria-label={playing ? t('pause') : t('play')}
          >
            {playing ? '❚❚' : '►'}
          </button>
          <input
            type="range"
            min={0}
            max={Math.max(0, meta.leads.length - 1)}
            value={li}
            onChange={(e) => {
              setLi(Number(e.target.value))
              setPlaying(false)
            }}
            aria-label="lead"
          />
          <div className="timeline-label">
            <span className="clock">
              {lead != null && issue != null
                ? frameFull(new Date((issue + lead * 60) * 1000))
                : '--:--'}
            </span>
            <span className="lead">{lead === 0 ? t('now') : leadLabel(lead ?? 0)}</span>
          </div>
        </div>
      )}

      <section className="panel">
        <h1 className="headline">{t('verify.headline')}</h1>
        <div className="verify-controls">
          <label className="verify-issue">
            {t('verify.issue')}
            <select
              value={issue ?? ''}
              onChange={(e) => setIssue(Number(e.target.value))}
              disabled={!issues}
            >
              {(issues ?? []).map((it) => (
                <option key={it.issue} value={it.issue}>
                  {frameFull(it.issuedAt)}
                </option>
              ))}
            </select>
          </label>
          <div className="mode-toggle kind-toggle" role="tablist">
            {KINDS.map((k) => (
              <button
                key={k}
                role="tab"
                aria-selected={kind === k}
                className={kind === k ? 'mode active' : 'mode'}
                onClick={() => setKind(k)}
              >
                {t(`verify.kind.${k}`)}
              </button>
            ))}
          </div>
        </div>

        {scores === 'missing' && <p className="updated">{t('verify.noObserved')}</p>}
        {scores && scores !== 'missing' && (
          <div className="score-chips">
            <span className="chip">Bias {scores.biasMmH.toFixed(2)} mm/h</span>
            <span className="chip">MAE {scores.maeMmH.toFixed(2)} mm/h</span>
            <span className="chip">CSI ≥0.1: {scores.csi01.toFixed(2)}</span>
            <span className="chip">CSI ≥0.5: {scores.csi05.toFixed(2)}</span>
            <span className="chip">CSI ≥1.0: {scores.csi10.toFixed(2)}</span>
          </div>
        )}
      </section>
    </div>
  )
}
