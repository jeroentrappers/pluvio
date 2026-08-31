import { MAP_RAMP } from '../domain/precip'

// Colour-ramp key for the map overlay: the Met Office rainfall scale
// (mm/hour), eight bands from deep blue (<0.5) to dark red (>32). Labels are
// numeric so no translation is needed; the unit chip anchors the scale.
export default function PrecipitationLegend() {
  return (
    <div className="legend">
      <span className="legend-chip legend-unit">mm/h</span>
      {MAP_RAMP.map((band) => (
        <span className="legend-chip" key={band.min}>
          <span className="legend-swatch" style={{ background: band.color }} />
          {band.label}
        </span>
      ))}
    </div>
  )
}
