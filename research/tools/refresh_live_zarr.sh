#!/bin/bash
# Rebuild the rolling "live" seamless zarr the learned producer infers from.
#
# c17-C needs 12 channels, and three of them are DERIVED, not collected:
# `oflow_rate` (pysteps LK advection prior per lead) and `rate_tendency`
# (growth/decay slope) come from `tools/add_nowcast_channels`, and the static
# terrain trio from `static.npz`. Rather than write a bespoke serving input path —
# untested code sitting between the model and production — this reuses the exact
# tooling that produced the training store, so serving and training assemble
# channels through the same code.
#
# Rolling window, not the full archive: the model only needs ~6 history frames
# (90 min) plus the ~10 frames pysteps uses for its motion estimate. WINDOW_HOURS
# keeps a comfortable margin while holding the rebuild to seconds of reprojection.
#
# Runs inside pluvio-producer-model:latest (torch + zarr + pysteps).
set -uo pipefail

STAGE="${LIVE_STAGE:-/stage}"
STORAGE="${LIVE_STORAGE:-/mnt/storagebox}"
OUT="${LIVE_ZARR:-$STAGE/live.zarr}"
STATIC="${LIVE_STATIC:-$STAGE/static.npz}"
WINDOW_HOURS="${LIVE_WINDOW_HOURS:-30}"
LEADS="${LIVE_LEADS:-0,10,20,30,40,50,60,70,80,90,100,110,120}"
LI_MAX_AGE_MIN="${LIVE_LI_MAX_AGE_MIN:-60}"

log() { echo "$(date -u +%FT%TZ) refresh_live_zarr: $*"; }

START=$(date -u -d "${WINDOW_HOURS} hours ago" +%Y-%m-%d)
END=$(date -u -d "tomorrow" +%Y-%m-%d)

# ── Lightning staleness gate ────────────────────────────────────────────────
# c17-C consumes `li_flash`. The LI crops encode "no flashes" and "no data"
# IDENTICALLY (nodata=0.0), so a dead feed is indistinguishable from a calm sky:
# the channel quietly reads all-zero and the model keeps forecasting with a
# permanently lightning-free view. That is not hypothetical — the EUMETSAT feed
# was down 2026-08-15..20. Refuse to build rather than serve a silently blind
# model; the caller falls back to the classical producer.
NEWEST_LI=$(find "$STORAGE/mtg_li/AF" -name '*_AF.tiff' -printf '%f\n' 2>/dev/null | sort | tail -1)
if [ -z "$NEWEST_LI" ]; then
  log "FATAL: no MTG-LI crops under $STORAGE/mtg_li/AF"
  exit 2
fi
LI_TS=${NEWEST_LI%%_*}                       # YYYYmmddHHMMSS
LI_EPOCH=$(date -u -d "${LI_TS:0:8} ${LI_TS:8:2}:${LI_TS:10:2}:${LI_TS:12:2}" +%s 2>/dev/null || echo 0)
LI_AGE_MIN=$(( ( $(date -u +%s) - LI_EPOCH ) / 60 ))
if [ "$LI_EPOCH" = "0" ] || [ "$LI_AGE_MIN" -gt "$LI_MAX_AGE_MIN" ]; then
  log "FATAL: newest MTG-LI is ${LI_AGE_MIN} min old (> ${LI_MAX_AGE_MIN}); refusing to build"
  log "       a stale li_flash reads as 'no lightning', not 'no data' — falling back is safer"
  exit 3
fi
log "MTG-LI ok: newest $LI_TS (${LI_AGE_MIN} min old)"

# ── Build the window ───────────────────────────────────────────────────────
TMP="${OUT}.new"
rm -rf "$TMP"
log "building $START -> $END (window ${WINDOW_HOURS} h) into $TMP"
python -m tools.build_seamless_zarr \
    --out "$TMP" --storage "$STORAGE" --no-aifs \
    --cadence-min 15 --leads "$LEADS" --aux-vars li_flash \
    --start "$START" --end "$END" --static "$STATIC" || { log "FATAL: build failed"; exit 4; }

# ── Derived channels (must match training: pysteps LK, not the fallback) ────
log "deriving oflow_rate + rate_tendency"
python -m tools.add_nowcast_channels --zarr "$TMP" || { log "FATAL: add_nowcast_channels failed"; exit 5; }

# ── Publish atomically ─────────────────────────────────────────────────────
# The producer may be mid-read; swap by rename so it never sees a partial store.
rm -rf "${OUT}.old"
[ -d "$OUT" ] && mv "$OUT" "${OUT}.old"
mv "$TMP" "$OUT"
rm -rf "${OUT}.old"
log "published $OUT"
