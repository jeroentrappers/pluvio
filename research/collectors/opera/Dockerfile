# OPERA radar collector — windowed-crops the EUMETNET OPERA composite COGs onto
# our BeNeLux+context domain. rasterio's wheel bundles GDAL but still needs a
# few base system libs (e.g. libexpat) — the full python image has them, so we
# avoid hunting individual packages on slim. Build context = research/collectors.
FROM python:3.13
RUN pip install --no-cache-dir rasterio==1.4.* numpy httpx
COPY fetch_opera_radar.py /app/fetch_opera_radar.py
WORKDIR /app
ENTRYPOINT ["python", "fetch_opera_radar.py"]
