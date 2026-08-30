#!/bin/bash
# Capture single-site radar polar volumes (ODIM HDF5) from the EUMETNET OPERA 24-h
# rolling cache. Default is ALL radars in all countries the feed carries.
#
# Measured 2026-08-30 (day-so-far, scaled to 24 h):
#
#   country  radars  files/day   GB/day   packaging
#   BE          3       2,300      0.9    1 param/file, all elevations
#   NL          3         570     14.7    everything in ONE ~25 MB file
#   DE         18     144,700     16.6    1 file per radar x elevation x param
#   FR         25      40,300      2.6    1 file per elevation, params combined
#   ...plus CH CZ DK EE FI HR IE IS LT MT NO PL RO SE SI
#
# ⚠️ CAPACITY. All countries ≈ 35 GB/day. With 662 GB free that is ~19 days before
# the Storage Box fills, so a fortnight fits with ~170 GB to spare and no more. The
# MIN_FREE_GB guard below stops collection rather than filling the disk — but the
# guard protects the box, it does not protect the archive: once it trips, capture
# silently stops and the 24-h cache moves on. Watch free space, or prune.
#
# ⚠️ FILE COUNT. ~190k files/day ≈ 2.7M per fortnight, on a CIFS mount. Directory
# walks get expensive fast: indexing 69k OPERA files already cost 28 s per build
# before it was date-pruned. Anything that later reads this archive must prune by
# date rather than rglob the tree.
#
# LU and UK are absent from the feed: Luxembourg has no radar of its own, and the
# UK does not share single-site data through OPERA's open exchange.
#
# Downloads run in parallel because DE alone publishes ~500 files per 5-min cycle,
# which is not achievable serially over this link.
set -uo pipefail

BUCKET="${RADAR_BUCKET:-https://s3.waw3-1.cloudferro.com/openradar-24h}"
OUT="${RADAR_OUT:-/mnt/storagebox/radar_volumes}"
LOOKBACK_H="${RADAR_LOOKBACK_H:-1}"
MIN_FREE_GB="${RADAR_MIN_FREE_GB:-80}"
JOBS="${RADAR_JOBS:-12}"
# "ALL" = discover every country/radar in the feed. Or give an explicit list of
# CC/code[:PARAM] specs, e.g. "BE/behel BE/bejab DE/deess:DBZH".
RADAR_SET="${RADAR_SET:-ALL}"

log() { echo "$(date -u +%FT%TZ) radar_vol: $*"; }

free_gb=$(df -BG --output=avail "$OUT" 2>/dev/null | tail -1 | tr -dc '0-9')
if [ -n "$free_gb" ] && [ "$free_gb" -lt "$MIN_FREE_GB" ]; then
  log "FATAL: only ${free_gb} GB free (< ${MIN_FREE_GB}) — stopping collection."
  log "       The 24-h cache will move on; this is data loss, not a pause. Prune or extend."
  exit 3
fi

list_keys() {  # $1 = prefix. Paginates; the cache exceeds the 1000-key page limit.
  local prefix="$1" token="" url body
  while :; do
    url="${BUCKET}/?list-type=2&prefix=${prefix}&max-keys=1000"
    [ -n "$token" ] && url="${url}&continuation-token=$(printf '%s' "$token" | sed -e 's|+|%2B|g' -e 's|/|%2F|g' -e 's|=|%3D|g')"
    body=$(curl -s -m 90 "$url" 2>/dev/null) || return 0
    printf '%s\n' "$body" | grep -oE '<Key>[^<]+</Key>' | sed -e 's|<Key>||' -e 's|</Key>||'
    case "$body" in
      *'<IsTruncated>true</IsTruncated>'*)
        token=$(printf '%s' "$body" | grep -oE '<NextContinuationToken>[^<]+' | sed 's|<NextContinuationToken>||' | head -1)
        [ -z "$token" ] && return 0 ;;
      *) return 0 ;;
    esac
  done
}

for h in $(seq 0 "$LOOKBACK_H"); do
  DAY=$(date -u -d "${h} hours ago" +%Y/%m/%d)

  if [ "$RADAR_SET" = "ALL" ]; then
    # Discover countries, then radars. OPERA/ holds the composites we already
    # collect elsewhere, so it is skipped here.
    specs=""
    ccs=$(curl -s -m 60 "${BUCKET}/?list-type=2&prefix=${DAY}/&delimiter=/" 2>/dev/null \
          | grep -oE '<Prefix>[^<]+' | sed -e 's|<Prefix>||' -e "s|${DAY}/||" -e 's|/$||')
    for cc in $ccs; do
      [ "$cc" = "OPERA" ] && continue
      rs=$(curl -s -m 60 "${BUCKET}/?list-type=2&prefix=${DAY}/${cc}/&delimiter=/" 2>/dev/null \
           | grep -oE '<Prefix>[^<]+' | sed -e 's|<Prefix>||' -e "s|${DAY}/${cc}/||" -e 's|/$||')
      for r in $rs; do specs="$specs ${cc}/${r}"; done
    done
  else
    specs="$RADAR_SET"
  fi

  for spec in $specs; do
    cc="${spec%%/*}"; rest="${spec#*/}"
    code="${rest%%:*}"; filt=""
    [ "$rest" != "$code" ] && filt="${rest#*:}"
    keys=$(list_keys "${DAY}/${cc}/${code}/")
    [ -n "$filt" ] && keys=$(printf '%s\n' $keys | grep -F "$filt" || true)
    [ -z "$keys" ] && continue
    dest="${OUT}/${DAY}/${cc}/${code}"
    mkdir -p "$dest"
    # Only fetch what is missing, then fan out. -f so a 4xx leaves no 0-byte file
    # that would later look collected; .part + mv so a kill mid-write cannot either.
    printf '%s\n' $keys | while read -r k; do
      [ -z "$k" ] && continue
      fn="${k##*/}"
      [ -s "${dest}/${fn}" ] || printf '%s\n' "$k"
    done | xargs -r -P "$JOBS" -I{} sh -c '
      k="{}"; fn="${k##*/}"
      if curl -sf -m 180 -o "'"$dest"'/${fn}.part" "'"$BUCKET"'/${k}"; then
        mv "'"$dest"'/${fn}.part" "'"$dest"'/${fn}"
      else
        rm -f "'"$dest"'/${fn}.part"
      fi'
  done
done

now_free=$(df -BG --output=avail "$OUT" 2>/dev/null | tail -1 | tr -dc '0-9')
total=$(find "$OUT" -name '*.h5' 2>/dev/null | wc -l)
log "done: ${total} files in archive, ${now_free} GB free"
exit 0
