#!/bin/bash
# Adaptive OPERA poller — fire the collector when the frame actually appears,
# instead of on a blind fixed cadence.
#
# Why: OPERA composites are valid at :00/:15/:30/:45 but publish ~10 min later
# (measured 2026-08-26: 10.1 min, within 1 s across four consecutive frames). A
# fixed `OnCalendar=*:03/15` therefore misses each frame by up to 15 min — observed
# serving a 28-min-old analysis while a 14-min-old one sat in the bucket. Polling
# harder would fix it by brute force; this instead *learns* the lag and wakes just
# after the expected publication, so it stays synced if OPERA's lag drifts.
#
# Loop, per 15-min slot:
#   1. compute the next un-collected slot and its expected publish time
#      (slot + learned lag + margin)
#   2. sleep until then
#   3. cheap HEAD probe on the exact object key — no download, no docker
#   4. on hit: trigger opera-forward.service (the existing collector; this script
#      adds scheduling only, never its own fetch path) and fold the observed lag
#      into an EMA
#   5. on miss: retry every RETRY_S, up to MAX_WAIT_S past the slot, then give up
#      on that slot WITHOUT polluting the EMA (a missing frame is not a lag signal)
#
# State: LAG_FILE holds the EMA in seconds; delete it to reset to DEFAULT_LAG_S.
set -uo pipefail

BUCKET="${OPERA_BUCKET:-https://s3.waw3-1.cloudferro.com/openradar-24h}"
LAG_FILE="${OPERA_LAG_FILE:-/var/lib/pluvio/opera_lag_s}"
SERVICE="${OPERA_SERVICE:-opera-forward.service}"
PRODUCT="${OPERA_PROBE_PRODUCT:-RATE}"
SLOT_MIN=15              # OPERA analysis cadence
DEFAULT_LAG_S=610        # 10.1 min, the measured publication lag
MARGIN_S=30              # wake a little after the expected instant
RETRY_S=60               # re-probe interval on a miss
MAX_WAIT_S=1500          # give up 25 min past the slot (frame likely absent)
EMA_ALPHA_NUM=3          # ema = (7*ema + 3*observed)/10 — slow, so one odd
EMA_ALPHA_DEN=10         # frame can't yank the schedule around
LAG_MIN_S=120            # clamp: never trust an implausible learned lag
LAG_MAX_S=1800

mkdir -p "$(dirname "$LAG_FILE")"
log() { echo "$(date -u +%FT%TZ) $*"; }

read_lag() {
  local v
  v=$(cat "$LAG_FILE" 2>/dev/null || echo "$DEFAULT_LAG_S")
  # Reject anything non-numeric or out of range rather than propagating garbage.
  case "$v" in (*[!0-9]*|'') v=$DEFAULT_LAG_S ;; esac
  [ "$v" -lt "$LAG_MIN_S" ] && v=$LAG_MIN_S
  [ "$v" -gt "$LAG_MAX_S" ] && v=$LAG_MAX_S
  echo "$v"
}

write_lag() {  # $1 = newly observed lag in seconds
  local obs=$1 old new
  old=$(read_lag)
  new=$(( ( (EMA_ALPHA_DEN - EMA_ALPHA_NUM) * old + EMA_ALPHA_NUM * obs ) / EMA_ALPHA_DEN ))
  [ "$new" -lt "$LAG_MIN_S" ] && new=$LAG_MIN_S
  [ "$new" -gt "$LAG_MAX_S" ] && new=$LAG_MAX_S
  echo "$new" > "$LAG_FILE"
  log "lag: observed=${obs}s ema ${old}s -> ${new}s"
}

# Epoch of the most recent slot boundary at or before $1.
slot_floor() { echo $(( $1 / (SLOT_MIN * 60) * (SLOT_MIN * 60) )); }

probe() {  # $1 = slot epoch. 0 if the object is published.
  local key url code
  key="$(date -u -d "@$1" +%Y/%m/%d)/OPERA/COMP/OPERA@$(date -u -d "@$1" +%Y%m%dT%H%M)@0@${PRODUCT}"
  for ext in tiff h5; do
    url="$BUCKET/$key.$ext"
    code=$(curl -s -o /dev/null -m 20 -w '%{http_code}' -I "$url" 2>/dev/null || echo 000)
    [ "$code" = "200" ] && return 0
  done
  return 1
}

log "adaptive OPERA poller starting (bucket=$BUCKET service=$SERVICE lag=$(read_lag)s)"
last_done=0

while true; do
  now=$(date -u +%s)
  slot=$(slot_floor "$now")
  # If this slot is already handled, aim at the next one.
  [ "$slot" -le "$last_done" ] && slot=$(( slot + SLOT_MIN * 60 ))

  lag=$(read_lag)
  target=$(( slot + lag + MARGIN_S ))
  wait_s=$(( target - now ))
  if [ "$wait_s" -gt 0 ]; then
    log "slot $(date -u -d "@$slot" +%H:%M) — sleeping ${wait_s}s until expected publish"
    sleep "$wait_s"
  fi

  hit=0
  while :; do
    if probe "$slot"; then hit=1; break; fi
    now=$(date -u +%s)
    if [ $(( now - slot )) -ge "$MAX_WAIT_S" ]; then
      log "slot $(date -u -d "@$slot" +%H:%M) — absent ${MAX_WAIT_S}s past valid time, skipping (EMA untouched)"
      break
    fi
    sleep "$RETRY_S"
  done

  if [ "$hit" = 1 ]; then
    observed=$(( $(date -u +%s) - slot ))
    log "slot $(date -u -d "@$slot" +%H:%M) published — triggering $SERVICE (observed lag ${observed}s)"
    systemctl start --no-block "$SERVICE" || log "WARN: failed to start $SERVICE"
    write_lag "$observed"
  fi
  last_done=$slot
done
