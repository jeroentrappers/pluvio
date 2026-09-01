import { useTranslation } from 'react-i18next'
import { frameTime, leadLabel } from '../format'
import type { RadarFrame } from '../api'

interface Props {
  frames: RadarFrame[]
  index: number
  isPlaying: boolean
  onIndex: (i: number) => void
  onPlayPause: () => void
  issuedAt: Date
  // Index of the newest observed frame in the seamless timeline: a tick marks
  // where measured composite hands over to forecast. Null = no marker.
  nowIndex?: number | null
}

// Video-scrubber-style transport: play/pause + a range slider over the frames.
export default function TimelineSlider({
  frames,
  index,
  isPlaying,
  onIndex,
  onPlayPause,
  issuedAt,
  nowIndex,
}: Props) {
  const { t } = useTranslation()
  const frame = frames[index]
  const leadMin = frame ? Math.round((frame.validTime.getTime() - issuedAt.getTime()) / 60000) : 0

  return (
    <div className="timeline">
      <button
        className="play"
        onClick={onPlayPause}
        aria-label={isPlaying ? t('pause') : t('play')}
      >
        {isPlaying ? '❚❚' : '►'}
      </button>
      <div className="range-wrap">
        {nowIndex != null && frames.length > 1 && (
          <div
            className="now-tick"
            style={{ left: `${(100 * nowIndex) / (frames.length - 1)}%` }}
          />
        )}
        <input
          type="range"
          min={0}
          max={Math.max(0, frames.length - 1)}
          value={index}
          onChange={(e) => onIndex(Number(e.target.value))}
          aria-label="timeline"
        />
      </div>
      <div className="timeline-label">
        <span className="clock">{frame ? frameTime(frame.validTime, leadMin) : '--:--'}</span>
        <span className="lead">{leadMin === 0 ? t('now') : leadLabel(leadMin)}</span>
      </div>
    </div>
  )
}
