# Pluvio — research

A small, self-contained data-science workspace alongside the Flutter app. Lives in `research/`, isn't bundled into Pluvio itself, and has its own Python toolchain.

Two questions drive everything in here:

1. **How accurate is the radar nowcast?** Pair predicted values against the eventual observation at the same lat/lon/time, compute the standard meteorological verification metrics (MAE / RMSE / bias, POD / FAR / CSI / HSS), and stratify by lead-time and intensity. See [docs/verification.md](docs/verification.md).
2. **Can we honestly claim a precipitation forecast out to 24 h (or further)?** Yes. Three free sources cover progressively longer horizons; the trick is blending them coherently. See [docs/24h_extension.md](docs/24h_extension.md).

## Layout

```
research/
├── README.md                            ← you are here
├── pyproject.toml                       ← uv / pip deps
├── .env.example                         ← KNMI_API_KEY, output paths
├── collectors/
│   ├── collect_kmi_live.py              ← Option A: poll app.meteo.be every 10 min
│   ├── fetch_knmi_archive.py            ← Option B: retrospective from KNMI Data Platform
│   └── fetch_alaro_24h.py               ← KMI ALARO Total_precipitation (60h horizon)
├── notebooks/                           ← Jupyter — analysis
├── fixtures/                            ← redacted sample responses (tests/demos)
└── docs/
    ├── verification.md                  ← what "skill" means and how we compute it
    └── 24h_extension.md                 ← the source-blending story for 24h+
```

## Getting started

```bash
cd research/
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env                # fill in KNMI_API_KEY for Option B
```

### Option A — live KMI collection (slow but native)

Polls `https://app.meteo.be/services/appv4/?s=getForecasts` for a configurable list of locations every 10 minutes and appends each response to a Parquet file. After a few days you have enough data for a skill curve.

```bash
python collectors/collect_kmi_live.py --locations brussels,antwerp,liege --out data/kmi_live.parquet
# usually run via systemd timer / cron / Fly machine
```

### Option B — retrospective from KNMI (fast, more data, NL)

KNMI's Data Platform archives both the operational forecast (`radar_forecast` v2.0, 5-min cadence, 2h lead) **and** the observation-corrected ground truth (`nl-rdr-data-rtcor-5m`, 5-min cadence). Pair them on timestamp and lead-time. **Requires a free API key from <https://developer.dataplatform.knmi.nl/>**.

```bash
python collectors/fetch_knmi_archive.py \
    --start 2026-05-20 --end 2026-05-26 \
    --out data/knmi/
```

### 24-h extension via ALARO

KMI's ALARO numerical-weather-prediction model is exposed on `opendata.meteo.be/service/alaro/wms`. The `Total_precipitation` layer carries a `Dimension name="time"` running roughly 60 hours forward at 1-hour cadence — way past 24 h, no key needed.

```bash
python collectors/fetch_alaro_24h.py --layer Total_precipitation --hours 24 --out data/alaro/
```

## Licence & attribution

- KMI / IRM data is licensed CC BY 4.0. Cite "Royal Meteorological Institute of Belgium (KMI / IRM)" wherever the analysis or its outputs are surfaced.
- KNMI data is also CC BY 4.0. Cite "Koninklijk Nederlands Meteorologisch Instituut (KNMI)".
- ECMWF Open Data is licensed CC BY 4.0.

This subdirectory inherits Pluvio's GPL-3.0 licence for the *code*. The output data is the providers' to license; only redistribute what they allow.
