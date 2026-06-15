import { Bar, BarChart, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { useTranslation } from 'react-i18next'
import { PRECIP_COLOR } from '../domain/precip'
import type { RadarFrame } from '../api'

interface Props {
  frames: RadarFrame[]
  index: number // currently-scrubbed frame, highlighted
  issuedAt: Date
}

// Bar chart of expected precipitation intensity (mm/h) across the forecast
// horizon. Bars are coloured by WMO band; the scrubbed frame is highlighted.
export default function ForecastChart({ frames, index, issuedAt }: Props) {
  const { t } = useTranslation()
  const data = frames.map((f, i) => ({
    i,
    lead: Math.round((f.validTime.getTime() - issuedAt.getTime()) / 60000),
    rate: Number(f.rateMmPerH.toFixed(2)),
    level: f.level,
  }))

  return (
    <div className="chart">
      <div className="chart-title">{t('chartTitle')}</div>
      <ResponsiveContainer width="100%" height={140}>
        <BarChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: -24 }}>
          <XAxis
            dataKey="lead"
            tickFormatter={(v) => (v <= 0 ? t('now') : `+${v}`)}
            tick={{ fill: '#9aa0a6', fontSize: 11 }}
            interval="preserveStartEnd"
            axisLine={false}
            tickLine={false}
          />
          <YAxis
            tick={{ fill: '#9aa0a6', fontSize: 11 }}
            axisLine={false}
            tickLine={false}
            width={36}
          />
          <Tooltip
            cursor={{ fill: 'rgba(255,255,255,0.05)' }}
            contentStyle={{ background: '#1b1b1d', border: 'none', borderRadius: 8, fontSize: 12 }}
            labelFormatter={(v) => (Number(v) <= 0 ? t('now') : t('minutesShort', { minutes: v }))}
            formatter={(v) => [t('rate', { value: v }), '']}
          />
          <Bar dataKey="rate" radius={[3, 3, 0, 0]} isAnimationActive={false}>
            {data.map((d) => (
              <Cell
                key={d.i}
                fill={PRECIP_COLOR[d.level]}
                fillOpacity={d.i === index ? 1 : 0.55}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
