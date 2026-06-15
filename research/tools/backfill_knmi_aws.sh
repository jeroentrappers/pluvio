#!/usr/bin/env bash
# One-time historical backfill of KNMI NL AWS (2024-08 → now) at 30-min cadence.
#
# Shares the KNMI 1000-req/h quota with the real-time radar pull, so it runs with
# KNMI_RATELIMIT_RESERVE set: it backs off once the quota drops to that reserve,
# leaving headroom for radar/forward pulls (which use the default reserve of 5).
# That makes the backfill slower (~750 req/h) but it never starves the backbone.
#
# Idempotent + resumable: writes to a separate parquet (no race with the 15-min
# forward writer) and resumes from the last month already on disk. Launch with:
#   setsid nohup bash tools/backfill_knmi_aws.sh > /tmp/pluvio-pull/knmi_aws_backfill.log 2>&1 < /dev/null &
set -uo pipefail
cd /home/jeroentrappers/Developer/appmire/pluvio/research
. .venv/bin/activate

export KNMI_RATELIMIT_RESERVE="${KNMI_RATELIMIT_RESERVE:-200}"
OUT="data/aws/knmi_aws_backfill.parquet"
ENDALL="2026-07-01"

# Resume from the month of the latest row already collected (re-pulls that month;
# dedup on (timestamp, station_id) makes it harmless), else from the start.
START=$(python - <<'PY'
import pandas as pd, pathlib, datetime as d
p = pathlib.Path("data/aws/knmi_aws_backfill.parquet")
if p.exists():
    mx = pd.Timestamp(pd.read_parquet(p, columns=["timestamp"]).timestamp.max())
    print(d.date(mx.year, mx.month, 1).isoformat())
else:
    print("2024-08-01")
PY
)

echo "=== KNMI AWS backfill resume from $START → $ENDALL (reserve=$KNMI_RATELIMIT_RESERVE) ==="
m="$START"
while [ "$m" \< "$ENDALL" ]; do
  nxt=$(date -u -d "$m +1 month" +%Y-%m-01)
  echo "=== month $m → $nxt ($(date -u +%H:%M:%S)) ==="
  python -m collectors.fetch_knmi_aws --start "$m" --end "$nxt" \
      --cadence-minutes 30 --out "$OUT" || echo "month $m FAILED"
  m="$nxt"
done
echo "BACKFILL COMPLETE $(date -u)"
