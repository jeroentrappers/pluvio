#!/bin/bash
# Monthly-chunk MTG backfill: download a target date window month-by-month so
# downloads start immediately (vs a single huge newest-first request that takes
# weeks to reach the months we want). The collector skips files already on disk,
# so this composes with the forward timer and is resumable.
#
# usage: mtg_chunk_backfill.sh <image> <env-file> <data-dir> <products> <start YYYY-MM> <end YYYY-MM exclusive>
set -u
img=$1; envf=$2; data=$3; prods=$4; startm=$5; endm=$6
y=${startm%-*}; m=$((10#${startm#*-}))
ey=${endm%-*};  em=$((10#${endm#*-}))
while [ $((y*12 + m)) -lt $((ey*12 + em)) ]; do
  s=$(printf "%04d-%02d-01" "$y" "$m")
  nm=$((m+1)); ny=$y; if [ "$nm" -gt 12 ]; then nm=1; ny=$((y+1)); fi
  e=$(printf "%04d-%02d-01" "$ny" "$nm")
  echo "=== MTG chunk $s .. $e (products=$prods) ==="
  docker run --rm --env-file "$envf" -v "$data":/data "$img" \
    --mode backfill --start "$s" --end "$e" --products "$prods" --out /data --min-free-gb 50
  y=$ny; m=$nm
done
echo "ALL CHUNKS DONE ($startm .. $endm)"
