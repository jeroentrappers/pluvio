# Classical seamless-forecast producer (rec #4). Runs model/produce_forecast.py
# --producer classical on a timer: reads the latest OPERA RATE analyses (+ AIFS
# when present) off the Storage Box and writes /opt/pluvio/serve/model_forecast.npz
# with per-lead source tags + confidence, which the backend worker serves.
#
# Build context = research/ (so `model/` and `tools/` are in scope).
# Classical-path deps: pysteps (optical-flow LK; opencv backs it), rasterio
# (OPERA COG reproject), pyproj (analysis grid), scipy (semi-Lagrangian advect +
# griddata regrid), numpy. Plus the GRIB stack (xarray + cfgrib + eccodes) for
# the AIFS outlook anchor (model/aifs_cube.py). No torch — the learned
# `--producer model` path is a separate, GPU-side image.
FROM python:3.13

# ⚠️ PINNED. pysteps runs Lucas-Kanade on OpenCV, so these two decide the
# advection field and therefore every nowcast lead this image produces. Rebuilding
# unpinned on 2026-08-26 resolved pysteps 1.21.5 + opencv 5.0.0 and changed the cube
# max from 7.47 to 32.15 mm/h on the SAME analysis and the same AIFS run — a 4x
# swing from a rebuild alone. These values match what is deployed and what c17-C was
# trained against; change them deliberately, with a re-verification, never by
# rebuilding.
RUN pip install --no-cache-dir \
        "numpy==2.*" \
        "scipy==1.17.*" \
        "rasterio==1.4.*" \
        pyproj \
        "pysteps==1.21.1" \
        "opencv-python-headless==4.13.*" \
        xarray \
        cfgrib \
        eccodes

# Fail the build loudly if the ecCodes C library didn't come through (cfgrib is
# useless without it) — far better than discovering it at the first timer run.
RUN python -c "import cfgrib, xarray; from eccodes import codes_get_api_version; print('eccodes', codes_get_api_version())"

# Only what the classical path imports (model.produce_forecast → classical, geo,
# nwp_regrid, aifs_cube; tools.build_seamless_zarr._index_tiffs).
COPY model/ /app/model/
COPY tools/ /app/tools/

WORKDIR /app
ENV PYTHONPATH=/app
ENTRYPOINT ["python", "-m", "model.produce_forecast"]
