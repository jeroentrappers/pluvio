#!/usr/bin/env bash
# Pull the latest window of one source. Invoked by the systemd timers in
# tools/systemd/. Idempotent: collectors skip files already on disk.
#
# Usage:
#   pull_forward.sh <source>
#
# <source> is one of: knmi-radar, kmi-aws, meteosat, alaro.
#
# Window: each source has a "lookback" — how far back to re-request on
# every run. Slightly overlapping with the previous run is fine (collectors
# are idempotent) and protects against transient API failures.

set -euo pipefail

SOURCE="${1:-}"
if [ -z "$SOURCE" ]; then
    echo "usage: $0 <knmi-radar|kmi-aws|meteosat|alaro>" >&2
    exit 2
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck disable=SC1091
. .venv/bin/activate

STAGE_DIR="${PLUVIO_STAGE_DIR:-$REPO_ROOT/stage}"

# end = now, rounded down to the source's native cadence so consecutive
# runs request the same boundary times.
now_utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }
ago_utc() { date -u -d "$1 ago" +%Y-%m-%dT%H:%M:%SZ; }

case "$SOURCE" in
    knmi-radar)
        # Cron will fire every 30 min — overlap 90 min so any missed slot is
        # picked up on the next run.
        python -m collectors.fetch_knmi_archive \
            --dataset radar_forecast \
            --start "$(ago_utc '90 minutes')" \
            --end "$(now_utc)" \
            --cadence-minutes 30 \
            --out "$STAGE_DIR/knmi"
        ;;
    kmi-aws)
        # 10-min cadence native. Overlap 1 hour.
        python -m collectors.fetch_kmi_aws \
            --start "$(ago_utc '1 hour')" \
            --end "$(now_utc)" \
            --out "$STAGE_DIR/aws/kmi_aws_10min.parquet" \
            --max-features 50000
        ;;
    meteosat)
        # 15-min native, pulling at 30-min cadence. Overlap 90 min.
        python -m collectors.fetch_eumetsat_msg \
            --start "$(ago_utc '90 minutes')" \
            --end "$(now_utc)" \
            --layers "msg_fes:ir108,msg_fes:wv062,msg_fes:gii_kindex,msg_fes:gii_liftedindex,msg_fes:cth,msg_fes:rdt" \
            --cadence-minutes 30 \
            --out "$STAGE_DIR/msg"
        ;;
    alaro)
        # ALARO publishes a new ~60h forward run every 6 hours. We grab the
        # next 24h of forecast steps from whatever is current. The collector
        # is forward-only (WMS has no history), so this is how we build the
        # archive — by polling.
        python -m collectors.fetch_alaro_24h \
            --hours 24 \
            --out "$STAGE_DIR/alaro"
        ;;
    *)
        echo "unknown source: $SOURCE" >&2
        exit 2
        ;;
esac
