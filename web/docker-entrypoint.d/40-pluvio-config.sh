#!/bin/sh
# Render /config.js from the VITE_* env vars at container startup.
# nginx:alpine runs every /docker-entrypoint.d/*.sh before starting nginx.
set -eu

: "${VITE_API_BASE:=http://localhost:8000}"
: "${VITE_TILES_URL:=https://tiles.appmire.be}"
: "${VITE_TILES_KEY:=}"
export VITE_API_BASE VITE_TILES_URL VITE_TILES_KEY

envsubst '${VITE_API_BASE} ${VITE_TILES_URL} ${VITE_TILES_KEY}' \
  < /etc/pluvio/config.template.js \
  > /usr/share/nginx/html/config.js

echo "pluvio: config.js apiBase=${VITE_API_BASE} tilesUrl=${VITE_TILES_URL}"
