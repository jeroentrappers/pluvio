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
}

export default function RadarMap({ center, bounds, frame, frames }: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const mapRef = useRef<maplibregl.Map | null>(null)
  const markerRef = useRef<maplibregl.Marker | null>(null)
  const readyRef = useRef(false)

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

    markerRef.current = new maplibregl.Marker({ color: '#3182bd' })
      .setLngLat([center.lon, center.lat])
      .addTo(map)

    return () => {
      map.remove()
      mapRef.current = null
      readyRef.current = false
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Recentre + move the marker when the located position changes.
  useEffect(() => {
    const map = mapRef.current
    if (!map) return
    map.easeTo({ center: [center.lon, center.lat], duration: 600 })
    markerRef.current?.setLngLat([center.lon, center.lat])
  }, [center.lat, center.lon])

  // Swap the overlay image (and re-anchor) when the frame or bounds change.
  useEffect(() => {
    const map = mapRef.current
    if (!map || !readyRef.current || !frame) return
    const src = map.getSource(RADAR_SOURCE) as ImageSource | undefined
    src?.updateImage({ url: frame.overlayUrl, coordinates: cornersOf(bounds) })
  }, [frame, bounds])

  // Prefetch all overlay PNGs so scrubbing/playback doesn't flicker.
  useEffect(() => {
    frames.forEach((f) => {
      const img = new Image()
      img.src = f.overlayUrl
    })
  }, [frames])

  return <div ref={containerRef} className="map" />
}
