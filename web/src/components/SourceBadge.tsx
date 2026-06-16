import { useTranslation } from 'react-i18next'
import { confidencePct, confidenceTier, sourceLabelKey } from '../domain/source'

interface Props {
  source: string | null
  confidence: number | null
}

// Provenance badge for the currently-scrubbed frame: what produced this lead
// (radar nowcast / blend / weather model) and how confident we are. The
// confidence bar + tier colour widen the uncertainty honestly as the user
// scrubs into the multi-day outlook. Renders nothing when the backend has no
// provenance for the frame (stub-served), so it never invents certainty.
export default function SourceBadge({ source, confidence }: Props) {
  const { t } = useTranslation()
  const key = sourceLabelKey(source)
  if (!key) return null

  const hasConf = typeof confidence === 'number' && Number.isFinite(confidence)
  const tier = hasConf ? confidenceTier(confidence) : null
  const pct = hasConf ? confidencePct(confidence) : null

  return (
    <div className="provenance" title={t('source.explainer')}>
      <span className="prov-source">
        <span className={`prov-dot prov-${source}`} aria-hidden="true" />
        {t(key)}
      </span>
      {hasConf && tier && (
        <span className={`prov-conf prov-conf-${tier}`}>
          <span className="prov-bar" aria-hidden="true">
            <span className="prov-bar-fill" style={{ width: `${pct}%` }} />
          </span>
          <span className="prov-conf-label">
            {t('source.confidence', { tier: t(`source.tier.${tier}`), pct })}
          </span>
        </span>
      )}
    </div>
  )
}
