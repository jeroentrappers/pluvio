import { useTranslation } from 'react-i18next'
import { PRECIP_COLOR, RAIN_LEVELS } from '../domain/precip'

// Colour-ramp key shared with the map overlay and the forecast chart.
export default function PrecipitationLegend() {
  const { t } = useTranslation()
  return (
    <div className="legend">
      {RAIN_LEVELS.map((level) => (
        <span className="legend-chip" key={level}>
          <span className="legend-swatch" style={{ background: PRECIP_COLOR[level] }} />
          {t(`legend.${level}`)}
        </span>
      ))}
    </div>
  )
}
