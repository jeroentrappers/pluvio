#!/bin/bash
# Capture Slovak (SHMU) single-site radar volumes from opendata.shmu.sk into the
# same date-partitioned layout the OPERA-bucket collector uses, so find_volume and
# the whole chain read them like any other radar.
#
# Why a dedicated collector: SHMU is NOT in the OPERA open bucket, but its open-data
# server carries four radars (skjav skkoj skkub sklaz) with FULL dual-pol moments
# (dBZ dBuZ ZDR RhoHV PhiDP KDP V W) as per-parameter ODIM HDF5 — chain-grade data
# (fuzzy declutter + Kdp attenuation apply). Verified 2026-09-01: ~5-min volumes,
# ~9-min latency.
#
# ⚠️ Their TLS chain is broken (missing intermediate); -k is required. Data is a
# public open-data portal, integrity risk accepted for radar imagery.
#
# Layout written: $OUT/YYYY/MM/DD/SK/<code>/<code>@<stamp>@<elev-or-token>@<PARAM>.h5
set -uo pipefail

BASE="https://opendata.shmu.sk/meteorology/weather/radar/volume"
OUT="${RADAR_OUT:-/mnt/storagebox/radar_volumes}"
RADARS="${SHMU_RADARS:-skjav skkoj skkub sklaz}"
PARAMS="${SHMU_PARAMS:-dBZ ZDR RhoHV PhiDP KDP V}"
LOOKBACK_FILES="${SHMU_LOOKBACK:-24}"     # newest N files per radar/param (~2 h)

log() { echo "$(date -u +%FT%TZ) shmu_vol: $*"; }

DAY=$(date -u +%Y/%m/%d)
for r in $RADARS; do
  dest="${OUT}/${DAY}/SK/${r}"
  mkdir -p "$dest"
  for prm in $PARAMS; do
    # Directory listing → newest N files. SHMU names files like
    # T_PAG...<code>_<YYYYmmddHHMMSS>.hdf; keep the original name in the token
    # slot and normalise the prefix so find_volume's globs match.
    curl -sk -m 60 "${BASE}/${r}/${prm}/" \
      | grep -oE 'href="[^"]+\.(hdf|h5)"' | sed 's/href="//;s/"//' \
      | sort | tail -n "$LOOKBACK_FILES" | while read -r fn; do
        stamp=$(echo "$fn" | grep -oE '20[0-9]{12}' | head -1)
        [ -z "$stamp" ] && continue
        short="${stamp:0:8}T${stamp:8:4}"
        p_up=$(echo "$prm" | tr 'a-z' 'A-Z' | sed 's/DBZ$/DBZH/')
        out="${dest}/${r}@${short}@vol@${p_up}.h5"
        [ -s "$out" ] && continue
        if curl -skf -m 120 -o "${out}.part" "${BASE}/${r}/${prm}/${fn}"; then
          mv "${out}.part" "$out"
        else
          rm -f "${out}.part"
        fi
      done
  done
done
total=$(find "${OUT}/${DAY}/SK" -name '*.h5' 2>/dev/null | wc -l)
log "done: ${total} SK files today"
exit 0
