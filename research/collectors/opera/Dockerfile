# OPERA radar collector — windowed-crops the EUMETNET OPERA composite COGs onto
# our BeNeLux+context domain. rasterio's wheel bundles GDAL but still needs a
# few base system libs (e.g. libexpat) — the full python image has them, so we
# avoid hunting individual packages on slim. Build context = research/collectors.
FROM python:3.13
# h5py + pyproj are needed by crop_odim_h5: from 2026 the OPERA long-term archive
# keeps ONLY the ODIM .h5 (the pre-rendered .tiff lives just in the 24-h cache), so
# every archive backfill goes through the HDF5 path. Without these the collector
# lists keys fine and then dies on `import h5py` — which is exactly how 2026
# backfills failed silently for months while forward collection kept working.
RUN pip install --no-cache-dir rasterio==1.4.* numpy httpx h5py pyproj
COPY fetch_opera_radar.py /app/fetch_opera_radar.py
WORKDIR /app
ENTRYPOINT ["python", "fetch_opera_radar.py"]
