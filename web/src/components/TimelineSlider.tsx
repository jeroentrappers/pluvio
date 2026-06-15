import { useTranslation } from 'react-i18next'
import type { RadarFrame } from '../api'

interface Props {
  frames: RadarFrame[]
  index: number
  isPlaying: boolean
  onIndex: (i: number) => void
  onPlayPause: () => void
  issuedAt: Date
}

const fmtClock = (d: Date) =>
  d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })

// Video-scrubber-style transport: play/pause + a range slider over the frames.
export default function TimelineSlider({
  frames,
  index,
  isPlaying,
  onIndex,
  onPlayPause,
  issuedAt,
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
      <input
        type="range"
        min={0}
        max={Math.max(0, frames.length - 1)}
        value={index}
        onChange={(e) => onIndex(Number(e.target.value))}
        aria-label="timeline"
      />
      <div className="timeline-label">
        <span className="clock">{frame ? fmtClock(frame.validTime) : '--:--'}</span>
        <span className="lead">{leadMin <= 0 ? t('now') : t('minutesShort', { minutes: leadMin })}</span>
      </div>
    </div>
  )
}
