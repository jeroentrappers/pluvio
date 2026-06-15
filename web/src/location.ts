import { useCallback, useEffect, useState } from 'react'

// Centre of Belgium — the fallback when geolocation is denied/unavailable so
// the radar still has something framed.
export const DEFAULT_CENTER = { lat: 50.85, lon: 4.35 }

export type GeoStatus = 'idle' | 'locating' | 'ok' | 'denied' | 'unavailable'

export interface GeoState {
  center: { lat: number; lon: number }
  status: GeoStatus
  locate: () => void
}

// One-shot geolocation with a Belgium fallback. `locate()` re-requests (e.g.
// after the user grants permission or moves).
export function useGeolocation(): GeoState {
  const [center, setCenter] = useState(DEFAULT_CENTER)
  const [status, setStatus] = useState<GeoStatus>('idle')

  const locate = useCallback(() => {
    if (!('geolocation' in navigator)) {
      setStatus('unavailable')
      return
    }
    setStatus('locating')
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        setCenter({ lat: pos.coords.latitude, lon: pos.coords.longitude })
        setStatus('ok')
      },
      (err) => setStatus(err.code === err.PERMISSION_DENIED ? 'denied' : 'unavailable'),
      { enableHighAccuracy: false, timeout: 10000, maximumAge: 60000 },
    )
  }, [])

  // Locate once on mount.
  useEffect(() => {
    locate()
  }, [locate])

  return { center, status, locate }
}
