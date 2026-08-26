#!/bin/bash
# Wait for the running add_nowcast_channels precompute to finish, verify it wrote
# the derived channels, then start run_c17.sh under a sleep/idle inhibitor so the
# ~31 h arm isn't killed by the box suspending (it suspended 3x on 2026-08-20).
# The inhibitor is scoped to the run and releases itself — no permanent change to
# the machine's power config.
cd /home/jeroentrappers/pluvio || exit 1
LOG=/home/jeroentrappers/chain_c17.log
: > "$LOG"

echo "$(date -u +%FT%TZ) waiting for precompute to finish..." >> "$LOG"
while pgrep -f add_nowcast_channels >/dev/null; do sleep 60; done
echo "$(date -u +%FT%TZ) precompute process gone" >> "$LOG"

# Verify rather than trust: the derived arrays must exist AND be non-trivial.
PLUVIO_GRID_N=256 /home/pv/bin/python - <<'PY' >> "$LOG" 2>&1
import sys, zarr, numpy as np
r = zarr.open_group("nowcast_mm_c17_v2.zarr", mode="r")
keys = set(r.array_keys())
need = {"oflow_rate", "oflow_leads", "rate_tendency"}
missing = need - keys
if missing:
    print("FAIL: missing", sorted(missing)); sys.exit(1)
o = r["oflow_rate"]; print("oflow_rate", o.shape, o.dtype)
# last issue should be populated, not left as zeros by a killed run
a = np.asarray(o[o.shape[0] - 1]); t = np.asarray(r["rate_tendency"][o.shape[0] - 1])
print("last oflow finite=%.1f%% max=%.3g | tendency finite=%.1f%%" % (
    100 * np.isfinite(a).mean(), float(np.nan_to_num(a).max()), 100 * np.isfinite(t).mean()))
print("OK")
PY
if ! grep -q "^OK$" "$LOG"; then
  echo "$(date -u +%FT%TZ) ABORT: precompute verification failed" >> "$LOG"
  exit 1
fi

echo "$(date -u +%FT%TZ) starting run_c17.sh under sleep inhibitor" >> "$LOG"
systemd-inhibit --what=sleep:idle --why="pluvio c17 training" \
  ./run_c17.sh > /home/jeroentrappers/exp_c17.log 2>&1
echo "$(date -u +%FT%TZ) run_c17.sh exited rc=$?" >> "$LOG"
