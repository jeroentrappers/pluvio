import { useEffect, useRef } from 'react'
import maplibregl, { type ImageSource } from 'maplibre-gl'
import { Protocol } from 'pmtiles'
import 'maplibre-gl/dist/maplibre-gl.css'
import { STYLE_URL } from '../config'
import type { Bounds } from '../types'
import type { RadarFrame } from '../api'

// Register the PMTiles protocol once so MapLibre can read the vector basemap
// straight from tiles.appmire.be (the style's source is pmtiles://…). The key
// is already embedded in the style's sub-resource URLs.
const protocol = new Protocol()
maplibregl.addProtocol('pmtiles', protocol.tile)

const RADAR_SOURCE = 'radar'
const RADAR_LAYER = 'radar-layer'
const RADAR_OPACITY = 0.8

// Image-source coordinate order: top-left, top-right, bottom-right, bottom-left.
function cornersOf(b: Bounds): [[number, number], [number, number], [number, number], [number, number]] {
  return [
    [b.west, b.north],
    [b.east, b.north],
    [b.east, b.south],
    [b.west, b.south],
  ]
}

interface Props {
  center: { lat: number; lon: number }
  bounds: Bounds
  frame: RadarFrame | null
  frames: RadarFrame[] // for prefetch
  // Called when the user picks a new location (map click or marker drag).
  onPick?: (lat: number, lon: number) => void
  // Bump to re-center the map on `center` (e.g. after "locate me"). A plain
  // `center` change only moves the marker, so picking a spot doesn't yank the
  // map out from under the user.
  recenter?: number
}

export default function RadarMap({ center, bounds, frame, frames, onPick, recenter }: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const mapRef = useRef<maplibregl.Map | null>(null)
  const markerRef = useRef<maplibregl.Marker | null>(null)
  const readyRef = useRef(false)
  // Latest callback / center, read by the once-registered map handlers.
  const onPickRef = useRef(onPick)
  onPickRef.current = onPick
  const centerRef = useRef(center)
  centerRef.current = center

  // Init the map once.
  useEffect(() => {
    if (!containerRef.current || mapRef.current) return
    const map = new maplibregl.Map({
      container: containerRef.current,
      style: STYLE_URL,
      center: [center.lon, center.lat],
      zoom: 7.5,
      minZoom: 5,
      maxZoom: 11,
      attributionControl: { compact: true },
    })
    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'top-right')
    mapRef.current = map

    // Tap/click anywhere to query the forecast for that spot.
    map.on('click', (e) => onPickRef.current?.(e.lngLat.lat, e.lngLat.lng))
    map.getCanvas().style.cursor = 'crosshair'

    map.on('load', () => {
      readyRef.current = true
      const url = frame?.overlayUrl
      map.addSource(RADAR_SOURCE, {
        type: 'image',
        // 1×1 transparent pixel until the first frame is set.
        url:
          url ||
          'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7',
        coordinates: cornersOf(bounds),
      })
      map.addLayer({
        id: RADAR_LAYER,
        type: 'raster',
        source: RADAR_SOURCE,
        paint: { 'raster-opacity': RADAR_OPACITY, 'raster-fade-duration': 0 },
      })
    })

    const marker = new maplibregl.Marker({ color: '#3182bd', draggable: true })
      .setLngLat([center.lon, center.lat])
      .addTo(map)
    // Drag the pin to query a different spot.
    marker.on('dragend', () => {
      const ll = marker.getLngLat()
      onPickRef.current?.(ll.lat, ll.lng)
    })
    markerRef.current = marker

    // The container resizes after the map is created — the responsive layout
    // reflows when data loads (the timeline + panel appear and the grid rows
    // recompute). MapLibre's built-in trackResize doesn't reliably catch that,
    // leaving the canvas stuck at its initial (smaller) size, so the basemap
    // only paints a strip. Observe the container ourselves and resize.
    const ro = new ResizeObserver(() => map.resize())
    if (containerRef.current) ro.observe(containerRef.current)

    return () => {
      ro.disconnect()
      map.remove()
      mapRef.current = null
      readyRef.current = false
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Move the pin to the queried location (no recenter — picking a spot
  // shouldn't pan the map away from where the user tapped).
  useEffect(() => {
    markerRef.current?.setLngLat([center.lon, center.lat])
  }, [center.lat, center.lon])

  // Explicit recenter (e.g. "locate me" or first geolocation fix).
  useEffect(() => {
    if (recenter === undefined) return
    mapRef.current?.easeTo({ center: [centerRef.current.lon, centerRef.current.lat], duration: 600 })
  }, [recenter])

  // Swap the overlay image (and re-anchor) when the frame or bounds change.
  useEffect(() => {
    const map = mapRef.current
    if (!map || !readyRef.current || !frame) return
    const src = map.getSource(RADAR_SOURCE) as ImageSource | undefined
    src?.updateImage({ url: frame.overlayUrl, coordinates: cornersOf(bounds) })
  }, [frame, bounds])

  // Prefetch all overlay PNGs so scrubbing/playback is instant and hits no
  // network. Two things matter for the cache to actually be reused:
  //   • crossOrigin='anonymous' — overlays come from API_BASE, a different
  //     origin, so MapLibre loads the image source as a CORS request. A bare
  //     `new Image()` is a *no-cors* request; browsers key the HTTP cache on
  //     request mode, so a no-cors prefetch can't satisfy MapLibre's CORS
  //     fetch and every frame would refetch. Matching the mode fixes that.
  //   • holding references in `prefetchRef` keeps the decoded images alive (so
  //     they aren't GC'd mid-loop) and lets us cache until a new prediction
  //     arrives, then drop the stale URLs.
  const prefetchRef = useRef<Map<string, HTMLImageElement>>(new Map())
  useEffect(() => {
    const cache = prefetchRef.current
    const wanted = new Set(frames.map((f) => f.overlayUrl))
    for (const f of frames) {
      if (cache.has(f.overlayUrl)) continue
      const img = new Image()
      img.crossOrigin = 'anonymous'
      img.src = f.overlayUrl
      cache.set(f.overlayUrl, img)
    }
    // Drop overlays from a superseded prediction so the cache doesn't grow.
    for (const url of cache.keys()) {
      if (!wanted.has(url)) cache.delete(url)
    }
  }, [frames])

  return <div ref={containerRef} className="map" />
}
