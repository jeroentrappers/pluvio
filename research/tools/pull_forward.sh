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
        # next 24h of forecast steps for each layer the model uses. The
        # collector is forward-only (WMS has no history), so this is how we
        # build the archive — by polling. Each layer call is idempotent and
        # skips on-disk files.
        #
        # Layer names verified against opendata.meteo.be ALARO WMS
        # GetCapabilities. The architecture doc's four channels map to:
        #   precip → Total_precipitation
        #   cloud  → Inst_flx_Tot_Cld_cover  (instantaneous-flux total)
        #   wind   → 10_m_u__wind_component + 10_m_v__wind_component
        #   RH     → 2m_Relative_humidity
        # Plus three high-value extras: CAPE (convective potential, better
        # than the originally-planned K-index), MSLP (pressure → tendency
        # via successive runs), and dewpoint (with T gives stability).
        for layer in Total_precipitation Inst_flx_Tot_Cld_cover \
                     10_m_u__wind_component 10_m_v__wind_component \
                     2m_Relative_humidity Surface_CAPE \
                     Mean_sea_level_pressure 2_m_temperature \
                     2_m_dewpoint_temperature; do
            python -m collectors.fetch_alaro_24h \
                --layer "$layer" --hours 24 \
                --out "$STAGE_DIR/alaro" \
            || echo "  alaro $layer skipped (error)"
        done
        ;;
    *)
        echo "unknown source: $SOURCE" >&2
        exit 2
        ;;
esac
