#!/bin/bash
# Change-triggered notifier for the c16 suite: polls every 5 min, but only prints
# (→ one chat notification) on a MEANINGFUL event: new epoch, stage change, new
# error, GPU >=88C, a stall (>150 min with no new epoch), or suite end.
EXP=/home/jeroentrappers/exp_c16.log
epoch_line() { grep -aE 'epoch [0-9]+:' "$EXP" 2>/dev/null | tail -1 | sed -E 's/.*train_seamless //'; }
stage_line() { grep -aE '=== .*(TRAIN|EVAL|ALL DONE)' "$EXP" 2>/dev/null | tail -1 | sed -E 's/.*=== //; s/ ===.*//'; }
err_count()  { grep -acE 'Traceback|CUDA out of memory|RuntimeError|Killed' "$EXP" 2>/dev/null; }

prev_epoch=$(epoch_line); prev_stage=$(stage_line); prev_err=$(err_count); prev_hot=0
last_progress=$(date +%s)
while true; do
  ts=$(date -u +%H:%MZ)
  ep=$(epoch_line); st=$(stage_line); er=$(err_count)
  temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader 2>/dev/null | head -1)
  now=$(date +%s)
  if [ "$er" -gt "${prev_err:-0}" ]; then echo "[$ts] ERROR in $st (errs=$er) — check exp_c16.log"; prev_err=$er; fi
  # track epochs silently (for stall detection); only NOTIFY on stage transitions
  if [ -n "$ep" ] && [ "$ep" != "$prev_epoch" ]; then prev_epoch="$ep"; last_progress=$now; fi
  if [ "$st" != "$prev_stage" ]; then
    echo "[$ts] STAGE → $st | last:${ep:-none} | gpu ${temp}C"; prev_stage="$st"; last_progress=$now
  fi
  if [ -n "$temp" ] && [ "$temp" -ge 88 ] && [ "$prev_hot" -eq 0 ]; then echo "[$ts] GPU HOT ${temp}C during $st"; prev_hot=1; fi
  [ -n "$temp" ] && [ "$temp" -lt 84 ] && prev_hot=0
  if [ $((now - last_progress)) -gt 9000 ]; then echo "[$ts] STALL? no new epoch >150min in $st (gpu ${temp}C)"; last_progress=$now; fi
  if grep -qa 'ALL DONE' "$EXP" 2>/dev/null; then echo "[$ts] SUITE COMPLETE"; break; fi
  if ! ps -eo cmd | grep -q '[r]un_c16.sh'; then echo "[$ts] suite ENDED without ALL DONE — investigate"; break; fi
  sleep 300
done
