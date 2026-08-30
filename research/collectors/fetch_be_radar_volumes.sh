#!/bin/bash
# Capture Belgian single-site radar polar volumes (ODIM HDF5) from the EUMETNET
# OPERA 24-h cache — the raw input for building our own Belgian composite.
#
# These live in the SAME public bucket we already poll for OPERA composites; we had
# only ever read the /COMP/ prefix. The per-country single-site tree was there all
# along:
#
#   2026/08/30/BE/bewid/PVOL/bewid@20260830T0800@0.3_0.5_..._25.0@DBZH.h5
#                                                └─ 12 elevation angles ─┘
#
# Measured 2026-08-30: 3 radars (behel Helchteren, bejab Jabbeke, bewid Wideumont),
# 5-min cadence, publication lag 4.0-5.2 min, ~0.41 MB/file, ~2300 files/day
# ≈ 1.0 GB/day. No API key and no rate limit, unlike the MeteoGate ORD API — whose
# EDR endpoints only ever return CoverageJSON point-series (one scalar per scan),
# never the volumes, so it cannot feed a composite.
#
# Why capture now: the bucket is a 24-HOUR rolling cache. Every hour not collected
# is permanently lost from the archive we will need to build and verify a composite.
# The long-term archive (openradar-archive) holds composites; single-site retention
# there is unverified, so treat this cache as the only reliable source.
#
# Products kept:
#   DBZH — corrected reflectivity, the input for Z->R rain-rate conversion
#   TH   — total (uncorrected) horizontal reflectivity, useful for QC comparison
#   VRAD — radial velocity, for dealiasing/QC and future motion estimation
set -uo pipefail

BUCKET="${BE_RADAR_BUCKET:-https://s3.waw3-1.cloudferro.com/openradar-24h}"
OUT="${BE_RADAR_OUT:-/mnt/storagebox/be_radar}"
RADARS="${BE_RADARS:-behel bejab bewid}"
LOOKBACK_H="${BE_RADAR_LOOKBACK_H:-2}"
MIN_FREE_GB="${BE_RADAR_MIN_FREE_GB:-50}"

log() { echo "$(date -u +%FT%TZ) be_radar: $*"; }

free_gb=$(df -BG --output=avail "$OUT" 2>/dev/null | tail -1 | tr -dc '0-9')
if [ -n "$free_gb" ] && [ "$free_gb" -lt "$MIN_FREE_GB" ]; then
  log "FATAL: only ${free_gb} GB free on $OUT (< ${MIN_FREE_GB}); refusing to collect"
  exit 3
fi

got=0; skip=0; fail=0
# Walk back a couple of hours so a missed run self-heals; already-present files are
# skipped by name, so overlap costs one stat() each.
for h in $(seq 0 "$LOOKBACK_H"); do
  DAY=$(date -u -d "${h} hours ago" +%Y/%m/%d)
  for R in $RADARS; do
    PREFIX="${DAY}/BE/${R}/PVOL/"
    keys=$(curl -s -m 60 "${BUCKET}/?list-type=2&prefix=${PREFIX}&max-keys=1000" 2>/dev/null \
           | grep -oE '<Key>[^<]+</Key>' | sed -e 's|<Key>||' -e 's|</Key>||')
    [ -z "$keys" ] && continue
    dest="${OUT}/${DAY}/${R}"
    mkdir -p "$dest"
    for k in $keys; do
      fn=$(basename "$k")
      if [ -s "${dest}/${fn}" ]; then skip=$((skip+1)); continue; fi
      # -f so a 4xx/5xx does not leave a 0-byte file that later looks "collected".
      if curl -sf -m 120 -o "${dest}/${fn}.part" "${BUCKET}/${k}" 2>/dev/null; then
        mv "${dest}/${fn}.part" "${dest}/${fn}"; got=$((got+1))
      else
        rm -f "${dest}/${fn}.part"; fail=$((fail+1))
      fi
    done
  done
done
log "done: ${got} fetched, ${skip} already present, ${fail} failed"
[ "$fail" -gt 0 ] && [ "$got" -eq 0 ] && exit 4
exit 0
