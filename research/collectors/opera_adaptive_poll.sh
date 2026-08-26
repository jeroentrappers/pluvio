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
#   1. compute the next un-collected slot and wake EARLY of the expected publish
#      time (slot + learned lag - EARLY_S)
#   2. sleep until then
#   3. cheap HEAD probe on the exact object key — no download, no docker
#   4. on hit: trigger opera-forward.service (the existing collector; this script
#      adds scheduling only, never its own fetch path) and fold the observed lag
#      into an EMA
#   5. on miss: retry every RETRY_S, up to MAX_WAIT_S past the slot, then give up
#      on that slot WITHOUT polluting the EMA (a missing frame is not a lag signal)
#
# ⚠️ Why we wake EARLY rather than just after the estimate. The first version woke at
# slot + lag + 30 s, and if the frame was already published it recorded "observed
# lag" = its own wake time. Feeding that back gave
#     new = (7*lag + 3*(lag + margin))/10 = lag + 0.3*margin
# i.e. the estimate ratcheted up by exactly 9 s every cycle, unbounded — measured
# drifting 610 -> 884 s in three hours while OPERA's true lag stayed ~10 min, which
# pushed served issue age from ~13 min back to ~27. The estimator was measuring the
# schedule instead of the data.
#
# Waking EARLY_S before the estimate means the frame is normally still absent on the
# first probe, so the miss->hit transition is a genuine observation of publication
# time (precise to RETRY_S). If it IS already present on the first probe we were
# late, and cannot tell by how much — so we pull the estimate earlier by EARLY_S
# rather than recording a number we know is only an upper bound.
#
# State: LAG_FILE holds the EMA in seconds; delete it to reset to DEFAULT_LAG_S.
set -uo pipefail

BUCKET="${OPERA_BUCKET:-https://s3.waw3-1.cloudferro.com/openradar-24h}"
LAG_FILE="${OPERA_LAG_FILE:-/var/lib/pluvio/opera_lag_s}"
SERVICE="${OPERA_SERVICE:-opera-forward.service}"
PRODUCT="${OPERA_PROBE_PRODUCT:-RATE}"
SLOT_MIN=15              # OPERA analysis cadence
DEFAULT_LAG_S=610        # 10.1 min, the measured publication lag
EARLY_S=120              # wake BEFORE the expected instant — see the feedback note
RETRY_S=20               # re-probe interval; also the precision of the lag estimate
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
  target=$(( slot + lag - EARLY_S ))
  wait_s=$(( target - now ))
  if [ "$wait_s" -gt 0 ]; then
    log "slot $(date -u -d "@$slot" +%H:%M) — sleeping ${wait_s}s until expected publish"
    sleep "$wait_s"
  fi

  hit=0
  probes=0
  while :; do
    probes=$((probes + 1))
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
    log "slot $(date -u -d "@$slot" +%H:%M) published — triggering $SERVICE (observed ${observed}s, probes=${probes})"
    systemctl start --no-block "$SERVICE" || log "WARN: failed to start $SERVICE"
    if [ "$probes" -gt 1 ]; then
      # Genuine miss->hit transition: `observed` really is publication time.
      write_lag "$observed"
    else
      # Present on the very first probe — we woke late. `observed` is only an upper
      # bound, so recording it would ratchet the estimate upward forever (the exact
      # feedback bug noted above). Step earlier and re-measure next slot.
      old=$(read_lag); new=$(( old - EARLY_S ))
      [ "$new" -lt "$LAG_MIN_S" ] && new=$LAG_MIN_S
      echo "$new" > "$LAG_FILE"
      log "lag: first-probe hit (woke late) — stepping estimate ${old}s -> ${new}s to re-measure"
    fi
  fi
  last_done=$slot
done
