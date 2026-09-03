"""Store contract checks, factored out of qc_inputs.py / qc_watchdog.py.

Both hourly QC jobs are thin CLIs over this package now: `thresholds.py`
loads the range/limit config (defaulted to the store's OBSERVED conventions,
overridable via PLUVIO_QC_THRESHOLDS), `checks.py` holds the pure numpy
check functions, and `verdict.py` is the one Check/verdict JSON shape the
library returns internally.

The two CLIs still write their own legacy-shaped JSON files (unchanged, so
the systemd units and anything scraping /opt/pluvio/serve/*.json keep
working) — they just build that JSON from Check results instead of from
inline logic.
"""
