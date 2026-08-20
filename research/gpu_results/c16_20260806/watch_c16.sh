#!/bin/bash
# Durable c16 training watchdog — runs inside tmux, independent of any Claude session.
# Appends a status line every 5 min to c16_watchdog.log; exits when the suite ends.
LOG=/home/jeroentrappers/c16_watchdog.log
EXP=/home/jeroentrappers/exp_c16.log
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] watchdog started" >> "$LOG"
while true; do
  ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  if ps -eo cmd | grep -q '[r]un_c16.sh'; then alive=UP; else alive=ENDED; fi
  stage=$(grep -aE '=== .*(TRAIN|EVAL|ALL DONE)' "$EXP" 2>/dev/null | tail -1 | sed 's/.*=== //; s/ ===.*//')
  epoch=$(grep -aE 'epoch [0-9]+:' "$EXP" 2>/dev/null | tail -1 | sed 's/.*train_seamless //')
  errs=$(grep -acE 'Traceback|CUDA out of memory|RuntimeError|Killed' "$EXP" 2>/dev/null)
  gpu=$(nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,power.draw --format=csv,noheader 2>/dev/null | tr '\n' ' ')
  echo "[$ts] suite:$alive stage:${stage:-init} last:${epoch:-none} errs:$errs gpu:${gpu:-na}" >> "$LOG"
  grep -qa 'ALL DONE' "$EXP" 2>/dev/null && { echo "[$ts] SUITE COMPLETE" >> "$LOG"; break; }
  [ "$alive" = ENDED ] && { echo "[$ts] WARNING suite ended without ALL DONE" >> "$LOG"; break; }
  sleep 300
done
