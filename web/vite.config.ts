import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { VitePWA } from 'vite-plugin-pwa'

// Installable PWA. The API (VITE_API_BASE) and the shared vector tile server
// (VITE_TILES_URL = tiles.appmire.be) live on separate origins; both, plus the
// radar overlay PNGs, are runtime-cached so the map stays fast and works
// briefly offline.
export default defineConfig({
  plugins: [
    react(),
    VitePWA({
      registerType: 'autoUpdate',
      includeAssets: ['icon.svg'],
      manifest: {
        name: 'Pluvio — rain radar for Belgium',
        short_name: 'Pluvio',
        description: 'Live precipitation radar and 2-hour nowcast for Belgium.',
        theme_color: '#0c0c0c',
        background_color: '#0c0c0c',
        display: 'standalone',
        start_url: '/',
        icons: [
          { src: 'icon.svg', sizes: 'any', type: 'image/svg+xml', purpose: 'any maskable' },
        ],
      },
      workbox: {
        navigateFallback: '/index.html',
        // config.js is generated at container startup, so it must NOT be
        // precached (its build-time bytes are a dev placeholder).
        globIgnores: ['**/config.js'],
        // pmtiles ranges and large tile payloads can exceed the default 2 MiB
        // precache/runtime ceiling.
        maximumFileSizeToCacheInBytes: 8 * 1024 * 1024,
        runtimeCaching: [
          {
            urlPattern: ({ url }) => url.pathname === '/config.js',
            handler: 'StaleWhileRevalidate',
            options: { cacheName: 'runtime-config' },
          },
          {
            // Radar overlay PNGs — snapshot-stamped (?t=), so safe to cache hard
            // but kept short since a new model run lands every few minutes.
            urlPattern: ({ url }) => /\/v1\/overlay\//.test(url.pathname),
            handler: 'CacheFirst',
            options: { cacheName: 'radar-overlays', expiration: { maxEntries: 200, maxAgeSeconds: 60 * 30 } },
          },
          {
            urlPattern: ({ url }) => /\/v1\//.test(url.pathname),
            handler: 'NetworkFirst',
            options: { cacheName: 'api', networkTimeoutSeconds: 5, expiration: { maxAgeSeconds: 300, maxEntries: 50 } },
          },
          {
            // Vector basemap (tiles.appmire.be) + the keyless low-zoom relief.
            urlPattern: ({ url }) => url.host.includes('tiles.appmire.be') || url.host.includes('tiles.openfreemap.org'),
            handler: 'CacheFirst',
            options: { cacheName: 'basemap', expiration: { maxEntries: 1000, maxAgeSeconds: 60 * 60 * 24 * 7 } },
          },
        ],
      },
    }),
  ],
})
