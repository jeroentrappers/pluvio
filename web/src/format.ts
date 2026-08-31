// Compact lead-time label, e.g. "+40m", "+6h", "+3d". The caller renders
// lead <= 0 as the localized "now". Nowcast leads (≤110 min) stay in minutes;
// the longer bands switch to hours, then days for the multi-day outlook.
export function leadLabel(minutes: number): string {
  // History mode uses negative leads: "-40m" = observed 40 minutes ago.
  if (minutes < 0) return `−${Math.abs(minutes)}m`
  if (minutes < 120) return `+${minutes}m`
  if (minutes < 2880) return `+${Math.round(minutes / 60)}h`
  return `+${Math.round(minutes / 1440)}d`
}

// Valid-time stamp for a frame. Within a day, just the clock; further out,
// prefix the weekday so a 10-day timeline stays readable.
export function frameTime(d: Date, leadMin: number): string {
  if (leadMin >= 1440) {
    return d.toLocaleString([], { weekday: 'short', hour: '2-digit', minute: '2-digit' })
  }
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

// Full day + time-of-day, for the selected-bar readout, e.g. "Mon 15 Jun 14:30".
export function frameFull(d: Date): string {
  return d.toLocaleString([], {
    weekday: 'short',
    day: 'numeric',
    month: 'short',
    hour: '2-digit',
    minute: '2-digit',
  })
}

// Time-of-day clock, e.g. "14:30". Used for the "Updated …" stamp.
export function timeOfDay(d: Date): string {
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

// The full forecast horizon we always request: 10 days, the widest band. Maps
// onto the backend bands (nowcast → short → medium → long).
export const HORIZON_MIN = 14400
