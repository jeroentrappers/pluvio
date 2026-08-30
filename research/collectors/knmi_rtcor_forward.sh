#!/bin/bash
# Forward collection of KNMI's realtime-corrected 1 km / 5-min radar precipitation
# (dataset nl_rdr_data_rtcor_5m, files RAD_NL25_RAC_RT_<YYYYMMDDHHMM>.h5).
#
# Why this exists: measured 2026-08-28 against the live KNMI API, rtcor publishes
# ~1m45s after valid time at 5-minute cadence:
#
#     valid 14:25  published 14:26:42   lag 1m42s
#     valid 14:20  published 14:21:43   lag 1m43s
#     valid 14:15  published 14:16:36   lag 1m36s
#
# versus the OPERA composite we currently train and serve on: 10m10s at 15-minute
# cadence. That is the difference between a ~13-20 min served issue age and ~4-5 min.
#
# ⚠️ This is an ARCHIVE-BUILDING collector, not a serving change. RAD_NL25 is the
# Netherlands composite: good over Flanders, poor-to-absent over the Ardennes, so it
# cannot simply replace OPERA for a Belgian product. And c17-C is trained on OPERA
# truth on the OPERA grid, so any switch means rebuilding the zarr and retraining.
# The point of starting now is that history cannot be created retroactively —
# KNMI open data keeps only a rolling window, so every day not collected is a day
# permanently missing from a future training set.
#
# fetch_knmi_archive.py requires an explicit window and skips files already on disk
# (`if target.exists()`), so re-running on a short rolling window is cheap and safe.
set -uo pipefail

RESEARCH="${KNMI_RESEARCH_DIR:-/opt/pluvio/research}"
OUT="${KNMI_RTCOR_OUT:-/mnt/storagebox/knmi}"
LOOKBACK_MIN="${KNMI_RTCOR_LOOKBACK_MIN:-45}"
MIN_FREE_GB="${KNMI_RTCOR_MIN_FREE_GB:-50}"

log() { echo "$(date -u +%FT%TZ) knmi_rtcor: $*"; }

# Stop before filling the Storage Box, same guard the other collectors use.
free_gb=$(df -BG --output=avail "$OUT" 2>/dev/null | tail -1 | tr -dc '0-9')
if [ -n "$free_gb" ] && [ "$free_gb" -lt "$MIN_FREE_GB" ]; then
  log "FATAL: only ${free_gb} GB free on $OUT (< ${MIN_FREE_GB}); refusing to collect"
  exit 3
fi

# Rolling window with generous overlap: a missed run is picked up by the next one,
# and already-present files cost one stat() each.
START=$(date -u -d "${LOOKBACK_MIN} minutes ago" +%Y-%m-%dT%H:%M)
END=$(date -u -d "5 minutes" +%Y-%m-%dT%H:%M)

log "pulling rtcor ${START} -> ${END} into ${OUT}"
cd "$RESEARCH" || { log "FATAL: no $RESEARCH"; exit 4; }
exec "$RESEARCH/.venv/bin/python" -m collectors.fetch_knmi_archive \
    --dataset rtcor --cadence-minutes 5 \
    --start "$START" --end "$END" --out "$OUT"
