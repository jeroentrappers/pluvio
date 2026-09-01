import { useEffect, useRef, useState } from 'react'
import maplibregl, { type CanvasSource } from 'maplibre-gl'
import { Protocol } from 'pmtiles'
import 'maplibre-gl/dist/maplibre-gl.css'
import { STYLE_URL } from '../config'
import type { Bounds } from '../types'
import type { HistoryTiles, RadarFrame, RadarSprite } from '../api'

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
  // Hi-res history tile manifest (1-km cube, viewport-tiled). Null = overview only.
  tiles?: HistoryTiles | null
  // Camera lock: when set, panning limits and the fit-on-change behaviour use
  // this box instead of `bounds`, so the seamless timeline can flip the overlay
  // between the wide observed domain and the forecast box without yanking the
  // camera at t=0.
  domain?: Bounds
  // Verify mode: draw this single image (a per-lead PNG) instead of a sprite
  // crop. `frame`/`sprite` are ignored while set.
  overlay?: { url: string; bounds: Bounds } | null
  // Auto-fit the camera when the domain box changes (default). The seamless
  // timeline passes false: the user's zoom is never touched there — only the
  // panning limits follow the domain.
  fit?: boolean
}

export default function RadarMap({ center, bounds, frame, sprite, onPick, recenter, tiles, domain, overlay, fit }: Props) {
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
  const appliedBoundsRef = useRef(bounds)
  // Hi-res tile mode: cache of tile sprite images keyed mtime_tx_ty, and a
  // counter bumped when any of them finishes loading (re-triggers the draw).
  const tileImgsRef = useRef(new Map<string, HTMLImageElement | 'loading'>())
  const [tileReady, setTileReady] = useState(0)
  // Verify-mode overlay: plain per-(issue,lead,kind) PNGs, cached by URL.
  const overlayImgsRef = useRef(new Map<string, HTMLImageElement | 'loading' | 'error'>())
  const [overlayReady, setOverlayReady] = useState(0)
  // Sprite sheets cached by URL so the seamless timeline can flip between the
  // history and forecast sheets at t=0 without re-downloading either.
  const spriteCacheRef = useRef(new Map<string, HTMLImageElement>())
  const [viewGen, setViewGen] = useState(0)   // bumped on moveend/zoomend
  // Readiness as STATE (not just the ref): effects that ran before the style
  // finished loading early-return; without a dep to re-trigger them, their
  // work (fit camera to the domain, first draw) got deferred until the next
  // prop change — which in the seamless timeline was the t=0 crossing, so the
  // zoom suddenly jumped when reaching the forecast.
  const [mapReady, setMapReady] = useState(false)

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
      minZoom: 4,
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
      setMapReady(true)
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

    // Tile mode re-evaluates what is visible after every camera move.
    map.on('moveend', () => setViewGen((n) => n + 1))
    map.on('zoomend', () => setViewGen((n) => n + 1))

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
    // Nothing displayable means a transient state (a refetch in flight, props
    // on their fallback defaults) — never let those drive the camera. Every
    // refetch used to flip bounds to the fallback box and back, re-fitting the
    // camera twice: the "auto-zoom" jumps.
    if (!sprite && !tiles && !overlay) return
    // The camera is governed by `domain` when provided (seamless timeline:
    // stay locked to the wide observed box while the overlay flips between
    // sources at t=0); otherwise by the overlay bounds as before.
    const cam = domain ?? bounds
    const prev = appliedBoundsRef.current
    const changed =
      prev.west !== cam.west || prev.east !== cam.east ||
      prev.south !== cam.south || prev.north !== cam.north
    appliedBoundsRef.current = cam
    map.setMaxBounds([
      [cam.west, cam.south],
      [cam.east, cam.north],
    ])
    const src = map.getSource(RADAR_SOURCE) as CanvasSource | undefined
    src?.setCoordinates?.(cornersOf(bounds))
    // Jump out so the whole (possibly much wider) domain is on screen: without
    // this, switching forecast -> history keeps the camera zoomed to the old
    // box and the new coverage sits off-screen with no way to reach it.
    if (changed && fit !== false) {
      map.fitBounds(
        [[cam.west, cam.south], [cam.east, cam.north]],
        { padding: 24, duration: 600 },
      )
    }
  }, [bounds.west, bounds.east, bounds.south, bounds.north,
      domain?.west, domain?.east, domain?.south, domain?.north,
      mapReady, sprite, tiles, overlay, fit])

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
    const cached = spriteCacheRef.current.get(sprite.url)
    if (cached) {
      spriteImgRef.current = cached
      setSpriteReady((n) => n + 1)
      return
    }
    let cancelled = false
    const img = new Image()
    img.crossOrigin = 'anonymous'
    img.onload = () => {
      spriteCacheRef.current.set(sprite.url, img)
      // Keep the cache small: the timeline only ever needs the current
      // history + forecast sheets (plus a refresh's replacements).
      for (const k of spriteCacheRef.current.keys()) {
        if (spriteCacheRef.current.size <= 8) break
        if (k !== sprite.url) spriteCacheRef.current.delete(k)
      }
      if (cancelled) return
      spriteImgRef.current = img
      setSpriteReady((n) => n + 1)
    }
    img.src = sprite.url
    return () => {
      cancelled = true
    }
  }, [sprite?.url])

  // Hi-res engages by VIEWPORT SIZE, not a zoom number: as soon as the view is
  // small enough that a bounded number of 256-px tiles covers it, tiles win. A
  // fixed zoom cutoff was screen-size dependent (a laptop hits it far later
  // than a phone) and read as "tiling not linked to zoom".
  const MAX_TILES = 12

  // Which hi-res tiles intersect the current viewport (null = overview mode).
  const visibleTiles = () => {
    const map = mapRef.current
    if (!map || !tiles) return null
    const b = tiles.bounds
    const v = map.getBounds()
    const degW = (b.east - b.west) / tiles.gridW
    const degH = (b.north - b.south) / tiles.gridH
    const pxW = tiles.tilePx * degW
    const pxH = tiles.tilePx * degH
    const tx0 = Math.max(0, Math.floor((v.getWest() - b.west) / pxW))
    const tx1 = Math.min(tiles.nx - 1, Math.floor((v.getEast() - b.west) / pxW))
    const ty0 = Math.max(0, Math.floor((b.north - v.getNorth()) / pxH))
    const ty1 = Math.min(tiles.ny - 1, Math.floor((b.north - v.getSouth()) / pxH))
    if (tx1 < tx0 || ty1 < ty0) return null
    const out: { tx: number; ty: number }[] = []
    for (let ty = ty0; ty <= ty1; ty++)
      for (let tx = tx0; tx <= tx1; tx++) out.push({ tx, ty })
    return out.length > 0 && out.length <= MAX_TILES ? out : null
  }

  // Render the current frame. Two paths share the one overlay canvas:
  //   overview — crop the frame from the whole-domain sprite (low zoom);
  //   tiles    — draw each visible hi-res tile's crop at native resolution and
  //              point the canvas source at the union of those tiles only.
  useEffect(() => {
    const map = mapRef.current
    const canvas = canvasRef.current
    if (!map || !readyRef.current || !canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const src = map.getSource(RADAR_SOURCE) as CanvasSource | undefined
    if (!src) return
    const push = () => {
      src.play()
      map.triggerRepaint()
      map.once('render', () => src.pause())
    }

    // Verify mode: a single plain PNG per (issue, lead, kind) instead of a
    // sprite sheet. Cached by URL, so scrubbing back over seen leads is instant.
    if (overlay) {
      const got = overlayImgsRef.current.get(overlay.url)
      if (!got) {
        const img = new Image()
        img.crossOrigin = 'anonymous'
        img.onload = () => {
          overlayImgsRef.current.set(overlay.url, img)
          setOverlayReady((n) => n + 1)
        }
        img.onerror = () => {
          overlayImgsRef.current.set(overlay.url, 'error')
          setOverlayReady((n) => n + 1)
        }
        img.src = overlay.url
        overlayImgsRef.current.set(overlay.url, 'loading')
        return
      }
      if (got === 'error') {
        // Frame unavailable (e.g. observed not archived yet): show nothing
        // rather than a stale frame, and forget the entry so a revisit retries.
        overlayImgsRef.current.delete(overlay.url)
        ctx.clearRect(0, 0, canvas.width, canvas.height)
        push()
        return
      }
      if (!(got instanceof Image)) return
      if (canvas.width !== got.width) canvas.width = got.width
      if (canvas.height !== got.height) canvas.height = got.height
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      ctx.drawImage(got, 0, 0)
      src.setCoordinates?.(cornersOf(overlay.bounds))
      push()
      return
    }
    if (!frame || frame.spriteIndex == null) return

    const vis = visibleTiles()
    if (vis && tiles) {
      let loaded = 0
      for (const { tx, ty } of vis) {
        const key = `${tiles.mtime}_${tx}_${ty}`
        const got = tileImgsRef.current.get(key)
        if (!got) {
          const img = new Image()
          img.crossOrigin = 'anonymous'
          img.onload = () => {
            tileImgsRef.current.set(key, img)
            setTileReady((n) => n + 1)
          }
          // Without this, ONE failed download left the entry stuck on 'loading'
          // forever and hi-res mode silently never engaged again.
          img.onerror = () => {
            tileImgsRef.current.delete(key)
            setTileReady((n) => n + 1)
          }
          img.src = tiles.urlFor(tx, ty)
          tileImgsRef.current.set(key, 'loading')
        } else if (got instanceof Image) {
          loaded++
        }
      }
      if (tileImgsRef.current.size > 40) {          // evict older cubes' tiles
        for (const k of tileImgsRef.current.keys())
          if (!k.startsWith(`${tiles.mtime}_`)) tileImgsRef.current.delete(k)
      }
      const txs = vis.map((t) => t.tx)
      const tys = vis.map((t) => t.ty)
      const tx0 = Math.min(...txs)
      const ty0 = Math.min(...tys)
      const tx1 = Math.max(...txs)
      const ty1 = Math.max(...tys)
      const px = tiles.tilePx
      const wPx = Math.min((tx1 + 1) * px, tiles.gridW) - tx0 * px
      const hPx = Math.min((ty1 + 1) * px, tiles.gridH) - ty0 * px
      const b = tiles.bounds
      const degW = (b.east - b.west) / tiles.gridW
      const degH = (b.north - b.south) / tiles.gridH
      const west = b.west + tx0 * px * degW
      const north = b.north - ty0 * px * degH
      const east = west + wPx * degW
      const south = north - hPx * degH
      const idx = frame.spriteIndex
      if (canvas.width !== wPx) canvas.width = wPx
      if (canvas.height !== hPx) canvas.height = hPx
      ctx.clearRect(0, 0, wPx, hPx)
      // Progressive: scaled overview underneath first, so streaming tiles
      // enhance rather than gate — no all-or-nothing wait.
      const ov = spriteImgRef.current
      if (ov && sprite && ov.src === sprite.url) {
        const sw = (wPx / tiles.gridW) * sprite.tileW
        const sh = (hPx / tiles.gridH) * sprite.tileH
        const sxo = (idx % sprite.cols) * sprite.tileW + (tx0 * px / tiles.gridW) * sprite.tileW
        const syo = Math.floor(idx / sprite.cols) * sprite.tileH + (ty0 * px / tiles.gridH) * sprite.tileH
        ctx.drawImage(ov, sxo, syo, sw, sh, 0, 0, wPx, hPx)
      }
      for (const { tx, ty } of vis) {
        const img = tileImgsRef.current.get(`${tiles.mtime}_${tx}_${ty}`)
        if (!(img instanceof Image)) continue
        const tw = Math.min(px, tiles.gridW - tx * px)
        const th = Math.min(px, tiles.gridH - ty * px)
        const sx = (idx % tiles.cols) * tw
        const sy = Math.floor(idx / tiles.cols) * th
        ctx.clearRect((tx - tx0) * px, (ty - ty0) * px, tw, th)
        ctx.drawImage(img, sx, sy, tw, th, (tx - tx0) * px, (ty - ty0) * px, tw, th)
      }
      ;(window as unknown as Record<string, unknown>).__pluvioTiles = {
        zoom: map.getZoom(), vis: vis.length, loaded,
      }
      src.setCoordinates?.([[west, north], [east, north], [east, south], [west, south]])
      push()
      return
    }

    const img = spriteImgRef.current
    if (!img || !sprite || img.src !== sprite.url) return
    const { tileW, tileH, cols } = sprite
    if (canvas.width !== tileW) canvas.width = tileW
    if (canvas.height !== tileH) canvas.height = tileH
    ctx.clearRect(0, 0, tileW, tileH)
    const idx = frame.spriteIndex
    const sx = (idx % cols) * tileW
    const sy = Math.floor(idx / cols) * tileH
    ctx.drawImage(img, sx, sy, tileW, tileH, 0, 0, tileW, tileH)
    src.setCoordinates?.(cornersOf(bounds))
    push()
  }, [frame, bounds, sprite, spriteReady, tiles, tileReady, viewGen, overlay, overlayReady, mapReady])

  return <div ref={containerRef} className="map" />
}
