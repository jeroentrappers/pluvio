import { useEffect, useRef, useState } from 'react'
import maplibregl, { type CanvasSource } from 'maplibre-gl'
import { Protocol } from 'pmtiles'
import 'maplibre-gl/dist/maplibre-gl.css'
import { STYLE_URL } from '../config'
import type { Bounds } from '../types'
import type { RadarFrame, RadarSprite } from '../api'

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
  sprite: RadarSprite | null // one sheet for the whole animation
  // Called when the user picks a new location (map click or marker drag).
  onPick?: (lat: number, lon: number) => void
  // Bump to re-center the map on `center` (e.g. after "locate me"). A plain
  // `center` change only moves the marker, so picking a spot doesn't yank the
  // map out from under the user.
  recenter?: number
}

export default function RadarMap({ center, bounds, frame, sprite, onPick, recenter }: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const mapRef = useRef<maplibregl.Map | null>(null)
  const markerRef = useRef<maplibregl.Marker | null>(null)
  const readyRef = useRef(false)
  // Latest callback / center, read by the once-registered map handlers.
  const onPickRef = useRef(onPick)
  onPickRef.current = onPick
  const centerRef = useRef(center)
  centerRef.current = center
  // Sprite sheet: load the image once, then scrub by cropping tiles to a canvas.
  const spriteImgRef = useRef<HTMLImageElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const [spriteReady, setSpriteReady] = useState(0)

  // Init the map once.
  useEffect(() => {
    if (!containerRef.current || mapRef.current) return
    // The overlay is a *canvas* source: we draw the current frame's tile onto
    // this canvas and MapLibre reads its pixels straight to a GPU texture — no
    // per-frame image URL (so nothing shows up in the Network panel, and no
    // PNG re-encoding). Created once; reused for the map's whole lifetime.
    const overlayCanvas = document.createElement('canvas')
    overlayCanvas.width = 100
    overlayCanvas.height = 100
    canvasRef.current = overlayCanvas

    const map = new maplibregl.Map({
      container: containerRef.current,
      style: STYLE_URL,
      center: [center.lon, center.lat],
      zoom: 7.5,
      minZoom: 5,
      maxZoom: 11,
      // Lock panning to the radar-covered region (the overlay's bounds): the
      // forecast only exists here, so there's nothing to see outside it.
      // MapLibre also clamps zoom-out so the viewport can't exceed these bounds.
      maxBounds: [
        [bounds.west, bounds.south],
        [bounds.east, bounds.north],
      ],
      attributionControl: { compact: true },
    })
    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'top-right')
    mapRef.current = map

    // Tap/click anywhere to query the forecast for that spot.
    map.on('click', (e) => onPickRef.current?.(e.lngLat.lat, e.lngLat.lng))
    map.getCanvas().style.cursor = 'crosshair'

    map.on('load', () => {
      readyRef.current = true
      map.addSource(RADAR_SOURCE, {
        type: 'canvas',
        canvas: overlayCanvas,
        coordinates: cornersOf(bounds),
        // Static by default (no continuous repaint / battery drain). We force a
        // one-shot upload after each draw via play()→render→pause().
        animate: false,
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

  // The radar-covered region can change between modes (forecast: Belgium box;
  // history: the wide multi-country composite) — re-lock panning and reposition the
  // overlay to whatever bounds the current data declares.
  useEffect(() => {
    const map = mapRef.current
    if (!map || !readyRef.current) return
    map.setMaxBounds([
      [bounds.west, bounds.south],
      [bounds.east, bounds.north],
    ])
    const src = map.getSource(RADAR_SOURCE) as CanvasSource | undefined
    src?.setCoordinates?.(cornersOf(bounds))
  }, [bounds.west, bounds.east, bounds.south, bounds.north])

  // Explicit recenter (e.g. "locate me" or first geolocation fix).
  useEffect(() => {
    if (recenter === undefined) return
    mapRef.current?.easeTo({ center: [centerRef.current.lon, centerRef.current.lat], duration: 600 })
  }, [recenter])

  // Download the sprite sheet once per prediction (one request for the whole
  // animation). crossOrigin so the canvas we crop from it isn't tainted when
  // the API is a different origin (dev); same-origin in prod needs nothing.
  useEffect(() => {
    if (!sprite?.url) return
    let cancelled = false
    const img = new Image()
    img.crossOrigin = 'anonymous'
    img.onload = () => {
      if (cancelled) return
      spriteImgRef.current = img
      setSpriteReady((n) => n + 1)
    }
    img.src = sprite.url
    return () => {
      cancelled = true
    }
  }, [sprite?.url])

  // Render the current frame: crop its tile from the sprite onto the overlay
  // canvas, then nudge MapLibre to upload it once. No network, no data URL.
  useEffect(() => {
    const map = mapRef.current
    const img = spriteImgRef.current
    const canvas = canvasRef.current
    if (!map || !readyRef.current || !img || !canvas || !sprite || !frame || frame.spriteIndex == null)
      return
    const { tileW, tileH, cols } = sprite
    if (canvas.width !== tileW) canvas.width = tileW
    if (canvas.height !== tileH) canvas.height = tileH
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const idx = frame.spriteIndex
    const sx = (idx % cols) * tileW
    const sy = Math.floor(idx / cols) * tileH
    ctx.clearRect(0, 0, tileW, tileH)
    ctx.drawImage(img, sx, sy, tileW, tileH, 0, 0, tileW, tileH)

    // Push the freshly-drawn canvas to the GPU once: play() makes the canvas
    // source copy on the next frame; we pause() right after so the map goes
    // back to idle (no continuous repaint).
    const src = map.getSource(RADAR_SOURCE) as CanvasSource | undefined
    if (!src) return
    src.play()
    map.triggerRepaint()
    map.once('render', () => src.pause())
  }, [frame, bounds, sprite, spriteReady])

  return <div ref={containerRef} className="map" />
}
